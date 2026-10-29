"""Tests for vector_store.py (Postgres + pgvector).

Needs VECTOR_DATABASE_URL pointing at a local Postgres with pgvector holding the kb_article copy (read-only here).
Store behaviour is tested in a throwaway database created on that server for this module and dropped afterwards.

  .venv/bin/python -m pytest tests/test_vector_store.py -q
"""
import hashlib
import json
import os
import sys
import urllib.parse
import uuid
from pathlib import Path

import numpy as np
import psycopg
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from psycopg.conninfo import make_conninfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import retrievers  # noqa: E402
import vector_store  # noqa: E402
from retrievers import format_docs  # noqa: E402
from vector_store import DIM, INDEX, PgVectorStore, _maximal_marginal_relevance, pinecone_style_metadata  # noqa: E402

BASE_DSN = os.environ.get("VECTOR_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not BASE_DSN, reason="VECTOR_DATABASE_URL (local pgvector copy) not set")
if BASE_DSN and urllib.parse.urlsplit(BASE_DSN).hostname not in ("localhost", "127.0.0.1"):
    raise SystemExit("VECTOR_DATABASE_URL must be a local database for these tests")

FIXTURE = (ROOT / "tests" / "fixtures" / "pinecone_vector_hit.txt").read_text()
FIXTURE_ID = "Sunbeam-1955-April.txt-Our Guruparampara From 1708 A.D."


def unit(v) -> list[float]:
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v)).tolist()


def basis(**weights) -> list[float]:
    v = np.zeros(DIM)
    for name, w in weights.items():
        v[int(name[1:])] = w
    return unit(v)


def hashed(text: str) -> list[float]:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return unit(np.random.default_rng(seed).normal(size=DIM))


class FakeEmbeddings(Embeddings):
    def __init__(self, table: dict[str, list[float]] | None = None):
        self.table = table or {}
        self.embedded = 0

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return [self.table.get(t) or hashed(t) for t in texts]

    def embed_query(self, text):
        return self.table.get(text) or hashed(text)


# q is closest to A and to its near-duplicate A2; MMR should prefer B over A2 as the second pick.
VECTORS = {
    "q": basis(e0=1.0),
    "A": basis(e0=0.95, e1=0.31),
    "A2": basis(e0=0.94, e1=0.34, e3=0.01),
    "B": basis(e0=0.80, e2=0.60),
    "C": basis(e0=0.20, e4=0.98),
}


def meta(title, **extra):
    return {"source": "Sunbeam-2000-January.txt", "title": title, "author": None, "start_page": 3, "end_page": 4,
            "start_line": 10, "end_line": 20, "published_date": "01-01-2000", "summary": f"about {title}", **extra}


@pytest.fixture(scope="module")
def test_dsn():
    name = f"cgpt_vec_test_{uuid.uuid4().hex[:8]}"
    admin = make_conninfo(BASE_DSN, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    yield make_conninfo(BASE_DSN, dbname=name)
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def store(test_dsn):
    s = PgVectorStore(FakeEmbeddings(dict(VECTORS)), dsn=test_dsn)
    s.ensure_schema()
    s.ensure_schema()  # idempotent
    with psycopg.connect(test_dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE kb_article CASCADE")  # and its chunks
    yield s
    s.close()


@pytest.fixture(scope="module")
def local_copy():
    """The kb_article copy at VECTOR_DATABASE_URL. Read-only: only SELECT/EXPLAIN/SHOW below."""
    s = PgVectorStore(FakeEmbeddings(), dsn=BASE_DSN)
    yield s
    s.close()


def add_designed(store):
    texts = ["A", "A2", "B", "C"]
    return store.add_texts(texts, metadatas=[meta(t) for t in texts])


# ---------- store behaviour (throwaway database) ----------

def test_upsert_ids_and_idempotence(store, test_dsn):
    ids = add_designed(store)
    assert ids == [f"Sunbeam-2000-January.txt-{t}" for t in ("A", "A2", "B", "C")], "the '<source>-<title>' id scheme"
    add_designed(store)
    store.add_texts(["A, revised"], metadatas=[meta("A")])
    with psycopg.connect(test_dsn) as conn:
        assert conn.execute("SELECT count(*) FROM kb_article").fetchone()[0] == 4
        assert conn.execute("SELECT content FROM kb_article WHERE id = %s", (ids[0],)).fetchone()[0] == "A, revised"
    assert store.existing_ids(ids + ["nope"]) == set(ids)
    store.delete([ids[3]])
    assert store.existing_ids(ids) == set(ids[:3])


def test_documents_look_like_pinecone_hits(store):
    add_designed(store)
    doc, score = store.similarity_search_with_score("q", k=1)[0]
    assert isinstance(doc, Document) and doc.id is None
    assert doc.page_content == "A"
    assert doc.metadata == pinecone_style_metadata(meta("A")) == {
        "source": "Sunbeam-2000-January.txt", "title": "A", "author": "Unknown", "start_page": 3.0, "end_page": 4.0,
        "start_line": 10.0, "end_line": 20.0, "published_date": "01-01-2000", "summary": "about A"}
    assert score == pytest.approx(0.95, abs=2e-3), "inner product, like Pinecone's dotproduct score"
    header = format_docs([doc]).split("\nContent:\n")[0]
    assert "'author': 'Unknown'" in header and "'start_page': 3.0" in header and "'end_line': 20.0" in header
    assert [d.page_content for d in store.similarity_search("q", k=2)] == ["A", "A2"]


def test_non_scalar_metadata_is_stringified(store):
    store.add_texts(["x"], metadatas=[meta("X", tags=["a", "b"], flag=True)])
    doc = store.similarity_search_by_vector(store.embeddings.embed_query("x"), k=1)[0]
    assert doc.metadata["tags"] == "['a', 'b']" and doc.metadata["flag"] is True


def test_mmr_picks_match_the_moved_function(store):
    add_designed(store)
    plain = [d.page_content for d in store.similarity_search("q", k=2)]
    mmr = [d.page_content for d in store.max_marginal_relevance_search("q", k=2, fetch_k=4)]
    assert plain == ["A", "A2"] and mmr == ["A", "B"], "MMR trades the near-duplicate for a diverse hit"
    names = ["A", "A2", "B", "C"]  # nearest-first, as fetched
    picks = _maximal_marginal_relevance(VECTORS["q"], [VECTORS[n] for n in names], k=2, lambda_mult=0.5)
    assert [names[i] for i in picks] == mmr
    assert len(store.max_marginal_relevance_search("q")) == 4, "defaults k=5, fetch_k=15; only 4 rows exist"


def test_retriever_interface(store):
    add_designed(store)
    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    assert type(retriever).__name__ == "VectorStoreRetriever"
    assert [d.page_content for d in retriever.invoke("q")] == ["A", "B"]


def test_search_settings_only_on_store_connections(test_dsn, monkeypatch):
    monkeypatch.setenv("CGPT_VECTOR_EF_SEARCH", "123")
    s = PgVectorStore(FakeEmbeddings(), dsn=test_dsn)
    try:
        with s._pool.connection() as conn:
            assert conn.execute("SHOW hnsw.ef_search").fetchone()[0] == "123"
            assert conn.execute("SHOW enable_seqscan").fetchone()[0] == "off"
    finally:
        s.close()
    with psycopg.connect(test_dsn) as other:
        assert other.execute("SHOW enable_seqscan").fetchone()[0] == "on", "other connections are untouched"


def test_add_to_vector_store_skips_existing(store, monkeypatch):
    monkeypatch.setattr(retrievers, "load_vector_store", lambda: store)
    docs = [Document(page_content=t, metadata=meta(t)) for t in ("A", "B")]
    retrievers.add_to_vector_store(docs)
    assert store.embeddings.embedded == 2
    retrievers.add_to_vector_store(docs + [Document(page_content="C", metadata=meta("C"))])
    assert store.embeddings.embedded == 3, "existing ids are skipped, only the new article is embedded"
    assert store.existing_ids([retrievers.get_doc_id(d) for d in docs]) == {retrievers.get_doc_id(d) for d in docs}


def test_no_dsn_fails_loudly(monkeypatch):
    monkeypatch.delenv("VECTOR_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="VECTOR_DATABASE_URL"):
        PgVectorStore(FakeEmbeddings())


# ---------- the real index (local copy, read-only) ----------

def stored_vector(store, article_id):
    with store._pool.connection() as conn:
        return conn.execute("SELECT embedding FROM kb_article WHERE id = %s", (article_id,)).fetchone()[0].to_list()


def test_hnsw_index_is_used(local_copy):
    with local_copy._pool.connection() as conn:
        assert conn.execute("SHOW hnsw.ef_search").fetchone()[0] == str(vector_store.DEFAULT_EF_SEARCH) == "400"
    plan = local_copy.explain(stored_vector(local_copy, FIXTURE_ID))
    assert f"Index Scan using {INDEX} on kb_article" in plan, plan[:300]
    assert "Seq Scan" not in plan


def test_vector_hit_formats_like_pinecone(local_copy):
    """format_docs on a vector hit gives the same title + metadata block as the recorded Pinecone-era output."""
    doc = local_copy.similarity_search_by_vector(stored_vector(local_copy, FIXTURE_ID), k=1)[0]
    assert format_docs([doc]).split("\nContent:\n")[0] == "0. " + FIXTURE.rstrip("\n")
    kb = next(json.loads(line) for line in open(ROOT / "knowledge_base.jsonl")
              if f'"title":"Our Guruparampara From 1708 A.D."' in line.replace(": ", ":"))
    assert doc.page_content == kb["page_content"]


def test_loader_metadata_matches_pinecone_offline():
    """scripts/load_vectors.py stores pinecone_style_metadata(); formatted, it matches the recorded block."""
    kb = next(json.loads(line) for line in open(ROOT / "knowledge_base.jsonl")
              if f'"title":"Our Guruparampara From 1708 A.D."' in line.replace(": ", ":"))
    doc = Document(page_content="x", metadata=pinecone_style_metadata(kb["metadata"]))
    assert format_docs([doc]).split("\nContent:\n")[0] == "0. " + FIXTURE.rstrip("\n")


# ---------- chunks of long articles (chunking.py) ----------

def add_chunk(dsn, article_id, vector, n=0, kb_id="sunbeam"):
    import chunking
    from pgvector import HalfVector
    from pgvector.psycopg import register_vector
    with psycopg.connect(dsn, autocommit=True) as conn:
        register_vector(conn)
        conn.execute("""INSERT INTO kb_chunk (article_id, n, start_char, end_char, kb_id, embedding)
                        VALUES (%s, %s, 0, 1, %s, %s)""", (article_id, n, kb_id, HalfVector(chunking.shorten(vector))))


def test_a_matching_chunk_brings_in_its_whole_article(store, test_dsn):
    add_designed(store)
    [d_id] = store.add_texts(["D, a long article"], metadatas=[meta("D")])  # its own vector is far from q
    add_chunk(test_dsn, d_id, VECTORS["q"])                                   # but a passage inside it matches q
    picks = store.max_marginal_relevance_search("q", k=2, fetch_k=4)
    assert [d.page_content for d in picks] == ["A", "B", "D, a long article"], "added after the usual picks, whole"
    assert picks[2].metadata == pinecone_style_metadata(meta("D"))

    off = PgVectorStore(FakeEmbeddings(dict(VECTORS)), dsn=test_dsn, chunk_search=False)
    try:
        assert [d.page_content for d in off.max_marginal_relevance_search("q", k=2, fetch_k=4)] == ["A", "B"]
    finally:
        off.close()

    store.delete([d_id])
    with psycopg.connect(test_dsn) as conn:
        assert conn.execute("SELECT count(*) FROM kb_chunk").fetchone()[0] == 0, "chunks go with their article"


def test_chunk_hits_respect_the_knowledge_base_filter(store, test_dsn):
    add_designed(store)
    [d_id] = store.add_texts(["D"], metadatas=[meta("D")])
    with psycopg.connect(test_dsn, autocommit=True) as conn:
        conn.execute("UPDATE kb_article SET kb_id = 'other' WHERE id = %s", (d_id,))
    add_chunk(test_dsn, d_id, VECTORS["q"], kb_id="other")
    only_sunbeam = store.max_marginal_relevance_search("q", k=2, fetch_k=4, filter={"kb_id": ["sunbeam"]})
    assert [d.page_content for d in only_sunbeam] == ["A", "B"]
    both = store.max_marginal_relevance_search("q", k=2, fetch_k=4, filter={"kb_id": ["sunbeam", "other"]})
    assert [d.page_content for d in both] == ["A", "B", "D"]


def test_chunk_finds_are_capped_deduplicated_and_never_replace_a_pick(store, test_dsn):
    import vector_store as vs
    ids = add_designed(store)
    add_chunk(test_dsn, ids[0], VECTORS["q"])                 # A is picked anyway: not added twice
    more = store.add_texts(["D", "E", "F"], metadatas=[meta(t) for t in "DEF"])
    add_chunk(test_dsn, more[0], basis(e0=0.99, e5=0.14))      # D, E, F each have a chunk near q, D's nearest
    add_chunk(test_dsn, more[0], basis(e0=0.90, e6=0.44), n=1)  # (a second, weaker chunk of D)
    add_chunk(test_dsn, more[1], basis(e0=0.97, e7=0.24))
    add_chunk(test_dsn, more[2], basis(e0=0.95, e8=0.31))
    got = [d.page_content for d in store.max_marginal_relevance_search("q", k=2, fetch_k=4)]
    assert vs.CHUNK_EXTRA == 2 and got == ["A", "B", "D", "E"]


def test_search_works_before_the_chunk_table_exists(store, test_dsn):
    add_designed(store)
    with psycopg.connect(test_dsn, autocommit=True) as conn:
        conn.execute("DROP TABLE kb_chunk")
    try:
        assert [d.page_content for d in store.max_marginal_relevance_search("q", k=2, fetch_k=4)] == ["A", "B"]
    finally:
        store.ensure_schema()
