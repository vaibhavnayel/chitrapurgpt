"""Knowledge-base vector search in Postgres + pgvector (replaced the Pinecone index).

Table `kb_article`: one row per article, keyed "<source>-<title>" (the id scheme the Pinecone index used), with the
article text, its metadata and a text-embedding-3-large embedding stored as halfvec(3072). pgvector indexes `vector`
only up to 2,000 dimensions, `halfvec` up to 4,000. Search is by inner product (`<#>`), as Pinecone's dotproduct
metric was; OpenAI embeddings are unit-length, so it ranks like cosine.

`metadata` is stored exactly as Pinecone returned it, because the agent sees it verbatim (retrievers.format_docs):
None -> "Unknown", non-scalar -> str, int -> float (Pinecone stored numbers as floats, so vector hits have always
shown e.g. 'start_page': 26.0).

Connections come from a small psycopg pool whose connections are configured for HNSW search:
`hnsw.ef_search` (CGPT_VECTOR_EF_SEARCH, default 400) and `enable_seqscan = off`. At ef_search 400 the planner
otherwise picks an exact sequential scan (its cost model ignores detoasting the halfvec column), which is ~50x
slower and competes with chat traffic. Only this module's connections are affected.

Articles belong to a knowledge base (`kb_id`, see kb.py). A search can be limited to some knowledge bases
(`filter={"kb_id": [...]}`); `hnsw.iterative_scan = strict_order` (pgvector 0.8) keeps scanning the index until
enough rows pass the filter, so a filtered search still returns its k nearest. Callers pass no filter when the user
can search every knowledge base that has articles, so that search is exactly the unfiltered one.

Long articles also have chunk embeddings (`kb_chunk`, see chunking.py). The search the agent uses
(max_marginal_relevance_search_by_vector) picks articles exactly as before, then adds up to CHUNK_EXTRA more: the
articles whose chunks are nearest the query, if not picked already. Adding rather than re-ranking means it never
drops an article the plain search finds. CGPT_CHUNK_SEARCH=0 turns it off.

DSN: VECTOR_DATABASE_URL, else DATABASE_URL (production sets only DATABASE_URL). scripts/load_vectors.py builds and
loads the table using DDL and pinecone_style_metadata() from here.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Iterable, List, Optional

import psycopg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pgvector import HalfVector
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

import chunking

logger = logging.getLogger(__name__)

DIM = 3072
TABLE = "kb_article"
INDEX = "kb_article_embedding_hnsw"
DEFAULT_EF_SEARCH = 400

DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {TABLE} (
    id        text PRIMARY KEY,          -- "<source>-<title>" (the Pinecone ids); published issues: "<source>#<n>"
    content   text NOT NULL,             -- the article text (Document.page_content)
    metadata  jsonb NOT NULL,            -- as Pinecone returned it (see module docstring)
    embedding halfvec({DIM}) NOT NULL,
    kb_id     text NOT NULL DEFAULT 'sunbeam',  -- the knowledge base (kb.py)
    seq       bigint,                    -- load order of published articles (the original import: kb_legacy_order)
    issue_id  bigint                     -- the ingest job that published it (NULL for the original import)
);
CREATE INDEX IF NOT EXISTS {INDEX} ON {TABLE} USING hnsw (embedding halfvec_ip_ops)
    WITH (m = 32, ef_construction = 200);
"""

CHUNK_TABLE = "kb_chunk"
# Chunks of long articles (chunking.py). kb.ensure_schema runs this on existing databases. Deleting or re-publishing
# an article deletes its chunks (ON DELETE CASCADE). kb_id is copied from the article for filtered searches; like
# kb_article it deliberately has no index (see kb.ARTICLE_COLUMNS).
CHUNK_DDL = f"""
CREATE TABLE IF NOT EXISTS {CHUNK_TABLE} (
    article_id text NOT NULL REFERENCES {TABLE}(id) ON DELETE CASCADE,
    n          int NOT NULL,              -- 0, 1, ... within the article
    start_char int NOT NULL,              -- the chunk is content[start_char:end_char]
    end_char   int NOT NULL,
    kb_id      text NOT NULL,
    embedding  halfvec({chunking.CHUNK_DIM}) NOT NULL,
    PRIMARY KEY (article_id, n)
);
CREATE INDEX IF NOT EXISTS {CHUNK_TABLE}_embedding_hnsw ON {CHUNK_TABLE} USING hnsw (embedding halfvec_ip_ops)
    WITH (m = 32, ef_construction = 200);
"""
# Each search adds up to CHUNK_EXTRA articles found through their chunks, looking at the CHUNK_FETCH nearest chunks.
CHUNK_EXTRA = 2
CHUNK_FETCH = 60


def vector_dsn() -> str | None:
    return os.environ.get("VECTOR_DATABASE_URL") or os.environ.get("DATABASE_URL")


def clean_metadata(metadata: dict) -> dict:
    """None -> "Unknown", scalars kept, anything else -> str (what was sent to the vector index)."""
    cleaned_metadata = {}
    for key, value in metadata.items():
        if value is None:
            cleaned_metadata[key] = "Unknown"
        elif isinstance(value, (str, int, float, bool)):
            cleaned_metadata[key] = value
        else:
            # Convert complex types to string representation
            cleaned_metadata[key] = str(value)
    return cleaned_metadata


def pinecone_style_metadata(metadata: dict) -> dict:
    """Metadata as a vector hit has always shown it: cleaned, with ints as floats."""
    return {k: float(v) if isinstance(v, int) and not isinstance(v, bool) else v
            for k, v in clean_metadata(metadata).items()}


def doc_id(metadata: dict) -> str:
    """The article id, as the Pinecone index used it."""
    return f"{metadata['source']}-{metadata['title']}"


def _maximal_marginal_relevance(
    query_embedding: List[float],
    embedding_list: List[List[float]],
    k: int = 4,
    lambda_mult: float = 0.5
) -> List[int]:
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: Query embedding
        embedding_list: List of embeddings to select from
        k: Number of embeddings to return
        lambda_mult: Diversity parameter (0=max diversity, 1=min diversity)

    Returns:
        List of indices of selected embeddings
    """
    import numpy as np

    if len(embedding_list) == 0:
        return []
    if k >= len(embedding_list):
        return list(range(len(embedding_list)))

    # Convert to numpy arrays
    query_emb = np.array(query_embedding)
    embeddings = np.array(embedding_list)

    # Calculate cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    # Calculate similarity to query
    similarity_to_query = np.array([
        cosine_similarity(query_emb, emb) for emb in embeddings
    ])

    # Initialize with most similar document
    selected = [int(np.argmax(similarity_to_query))]

    # Calculate pairwise similarities
    for _ in range(k - 1):
        if len(selected) >= len(embedding_list):
            break

        best_score = -float('inf')
        best_idx = -1

        for i in range(len(embeddings)):
            if i in selected:
                continue

            # Relevance to query
            relevance = similarity_to_query[i]

            # Max similarity to already selected documents
            max_sim_to_selected = max(
                cosine_similarity(embeddings[i], embeddings[j])
                for j in selected
            )

            # MMR score
            score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx != -1:
            selected.append(best_idx)

    return selected


class PgVectorStore(VectorStore):
    """LangChain VectorStore over the `kb_article` table. Sync API (the retriever runs in a worker thread)."""

    def __init__(self, embedding: Embeddings, dsn: str | None = None, *, ef_search: int | None = None,
                 pool_size: int = 4, chunk_search: bool | None = None):
        dsn = dsn or vector_dsn()
        if not dsn:
            raise RuntimeError("No vector database configured: set VECTOR_DATABASE_URL or DATABASE_URL")
        self._dsn = dsn
        self._embedding = embedding
        self.ef_search = int(ef_search or os.environ.get("CGPT_VECTOR_EF_SEARCH") or DEFAULT_EF_SEARCH)
        self.chunk_search = (os.environ.get("CGPT_CHUNK_SEARCH", "1") != "0") if chunk_search is None else chunk_search
        self._no_chunk_table_logged = False
        self._pool = ConnectionPool(dsn, min_size=0, max_size=pool_size, kwargs={"autocommit": True},
                                    configure=self._configure, check=ConnectionPool.check_connection,
                                    name="kb-vector-store", open=True)

    def _configure(self, conn: Connection) -> None:
        register_vector(conn)
        conn.execute(f"SET hnsw.ef_search = {self.ef_search}")
        conn.execute("SET enable_seqscan = off")
        try:
            conn.execute("SET hnsw.iterative_scan = strict_order")
        except psycopg.Error:  # pgvector < 0.8: filtered searches may return fewer than k rows
            logger.warning("hnsw.iterative_scan isn't available (pgvector < 0.8)")

    def close(self) -> None:
        self._pool.close()

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ---- schema / ingest ----

    def ensure_schema(self) -> None:
        """Create the extension, table and HNSW index if missing (idempotent). Uses its own connection: pooled
        connections register the halfvec type, which needs the extension to exist already."""
        with psycopg.connect(self._dsn, autocommit=True) as conn:
            conn.execute(DDL)
            conn.execute(CHUNK_DDL)

    def existing_ids(self, ids: list[str]) -> set[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(f"SELECT id FROM {TABLE} WHERE id = ANY(%s)", (list(ids),)).fetchall()
        return {r[0] for r in rows}

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None,
                  ids: Optional[List[str]] = None, **kwargs: Any) -> List[str]:
        """Embed and upsert. Metadata is stored Pinecone-style; ids default to "<source>-<title>"."""
        texts = list(texts)
        if not texts:
            return []
        metadatas = metadatas or [{} for _ in texts]
        if ids is None:
            ids = [doc_id(m) if "source" in m and "title" in m else str(uuid.uuid4()) for m in metadatas]
        vectors = self._embedding.embed_documents(texts)
        rows = [(i, t, Jsonb(pinecone_style_metadata(m)), HalfVector(v))
                for i, t, m, v in zip(ids, texts, metadatas, vectors)]
        with self._pool.connection() as conn:
            with conn.cursor() as cur, conn.transaction():
                cur.executemany(
                    f"""INSERT INTO {TABLE} (id, content, metadata, embedding) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata,
                                                       embedding = EXCLUDED.embedding""", rows)
        return list(ids)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return False
        with self._pool.connection() as conn:
            conn.execute(f"DELETE FROM {TABLE} WHERE id = ANY(%s)", (list(ids),))
        return True

    # ---- search ----

    @staticmethod
    def _query(cols: str, kb_ids: Optional[list[str]]) -> str:
        where = "WHERE kb_id = ANY(%(kbs)s) " if kb_ids is not None else ""
        return f"SELECT {cols} FROM {TABLE} {where}ORDER BY embedding <#> %(q)s LIMIT %(k)s"

    def _nearest(self, query_embedding: List[float], k: int, with_vectors: bool = False,
                 kb_ids: Optional[list[str]] = None, with_ids: bool = False) -> list[tuple]:
        """The k nearest articles by inner product (via the HNSW index), optionally only in some knowledge bases:
        (content, metadata, -ip[, embedding][, id])."""
        cols = ("content, metadata, embedding <#> %(q)s" + (", embedding" if with_vectors else "")
                + (", id" if with_ids else ""))
        with self._pool.connection() as conn:
            return conn.execute(self._query(cols, kb_ids),
                                {"q": HalfVector(query_embedding), "k": k, "kbs": kb_ids}).fetchall()

    def explain(self, query_embedding: List[float], k: int = 15, kb_ids: Optional[list[str]] = None) -> str:
        """The plan of a nearest-neighbour query on a vector-store connection (for checks and debugging)."""
        with self._pool.connection() as conn:
            rows = conn.execute("EXPLAIN " + self._query("content, metadata, embedding", kb_ids),
                                {"q": HalfVector(query_embedding), "k": k, "kbs": kb_ids}).fetchall()
        return "\n".join(r[0] for r in rows)

    @staticmethod
    def _kb_filter(filter: Optional[dict]) -> Optional[list[str]]:
        """The only filter supported: {"kb_id": [...]} (or a single id)."""
        if not filter:
            return None
        if set(filter) != {"kb_id"}:
            raise NotImplementedError("only a kb_id filter is supported")
        kbs = filter["kb_id"]
        return [kbs] if isinstance(kbs, str) else list(kbs)

    def similarity_search(self, query: str, k: int = 4, filter: Optional[dict] = None,
                          **kwargs: Any) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter)]

    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[dict] = None,
                                     **kwargs: Any) -> List[tuple[Document, float]]:
        """Docs with their inner-product score (Pinecone's dotproduct score)."""
        rows = self._nearest(self._embedding.embed_query(query), k, kb_ids=self._kb_filter(filter))
        return [(Document(page_content=content, metadata=metadata), -neg_ip) for content, metadata, neg_ip in rows]

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, filter: Optional[dict] = None,
                                    **kwargs: Any) -> List[Document]:
        return [Document(page_content=c, metadata=m)
                for c, m, _ in self._nearest(embedding, k, kb_ids=self._kb_filter(filter))]

    def max_marginal_relevance_search(self, query: str, k: int = 5, fetch_k: int = 15, lambda_mult: float = 0.5,
                                      filter: Optional[dict] = None, **kwargs: Any) -> List[Document]:
        """Fetch the fetch_k nearest with their vectors, then the same MMR selection the Pinecone wrapper used."""
        return self.max_marginal_relevance_search_by_vector(self._embedding.embed_query(query), k=k,
                                                            fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter)

    def max_marginal_relevance_search_by_vector(self, embedding: List[float], k: int = 5, fetch_k: int = 15,
                                                lambda_mult: float = 0.5, filter: Optional[dict] = None,
                                                **kwargs: Any) -> List[Document]:
        kb_ids = self._kb_filter(filter)
        rows = self._nearest(embedding, fetch_k, with_vectors=True, kb_ids=kb_ids, with_ids=True)
        selected = _maximal_marginal_relevance(embedding, [r[3].to_list() for r in rows], k=k,
                                               lambda_mult=lambda_mult) if rows else []
        docs = [Document(page_content=rows[i][0], metadata=rows[i][1]) for i in selected]
        if self.chunk_search and CHUNK_EXTRA:
            docs += self._found_by_chunks(embedding, {rows[i][4] for i in selected}, kb_ids)
        return docs

    def _nearest_chunks(self, query_embedding: List[float], k: int,
                        kb_ids: Optional[list[str]] = None) -> list[tuple[str, float]]:
        """The k nearest chunks: (article id, similarity). None if the chunk table doesn't exist yet."""
        where = "WHERE kb_id = ANY(%(kbs)s) " if kb_ids is not None else ""
        sql = (f"SELECT article_id, -(embedding <#> %(q)s) FROM {CHUNK_TABLE} {where}"
               "ORDER BY embedding <#> %(q)s LIMIT %(k)s")
        try:
            with self._pool.connection() as conn:
                return conn.execute(sql, {"q": HalfVector(chunking.shorten(query_embedding)), "k": k,
                                          "kbs": kb_ids}).fetchall()
        except psycopg.errors.UndefinedTable:
            if not self._no_chunk_table_logged:
                logger.warning("no %s table: searching whole articles only", CHUNK_TABLE)
                self._no_chunk_table_logged = True
            return []

    def _found_by_chunks(self, embedding: List[float], picked: set[str],
                         kb_ids: Optional[list[str]]) -> List[Document]:
        """Up to CHUNK_EXTRA articles not in `picked`, in order of their nearest chunk to the query."""
        extra: list[str] = []
        for article_id, _ in self._nearest_chunks(embedding, CHUNK_FETCH, kb_ids):
            if article_id not in picked and article_id not in extra:
                extra.append(article_id)
                if len(extra) == CHUNK_EXTRA:
                    break
        if not extra:
            return []
        with self._pool.connection() as conn:
            found = {r[2]: r for r in conn.execute(f"SELECT content, metadata, id FROM {TABLE} WHERE id = ANY(%s)",
                                                   (extra,)).fetchall()}
        return [Document(page_content=found[a][0], metadata=found[a][1]) for a in extra if a in found]

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None,
                   ids: Optional[List[str]] = None, **kwargs: Any) -> "PgVectorStore":
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
