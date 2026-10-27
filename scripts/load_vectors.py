"""Build and load the knowledge-base vector index in Postgres + pgvector.

  .venv/bin/python scripts/load_vectors.py --dsn-env VECTOR_DATABASE_URL [--embeddings .cache/kb_embeddings.npz]

Creates (idempotently) the `vector` extension, table `kb_article` and its HNSW index, then upserts every article in
knowledge_base.jsonl. Embeddings: text-embedding-3-large (3072 dims) via OpenAIEmbeddings.embed_documents, the same
code path that built the original index; pass --embeddings to reuse a saved matrix (ids + M) instead of re-embedding.

Table layout, metric, index settings and the metadata format are defined in vector_store.py (DDL,
pinecone_style_metadata, doc_id), the module the app searches with.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vector_store import DDL, DIM, doc_id, pinecone_style_metadata  # noqa: E402


def embed(texts: list[str]) -> np.ndarray:
    from langchain_openai import OpenAIEmbeddings
    emb = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=64, max_retries=6)
    out = []
    for i in range(0, len(texts), 256):
        out.extend(emb.embed_documents(texts[i:i + 256]))
        print(f"  embedded {min(i + 256, len(texts))}/{len(texts)}", flush=True)
    return np.asarray(out, dtype=np.float32)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn-env", default="VECTOR_DATABASE_URL", help="env var holding the Postgres URL")
    ap.add_argument("--embeddings", help="saved .npz with `ids` and `M` (skip re-embedding)")
    ap.add_argument("--kb", default=str(ROOT / "knowledge_base.jsonl"))
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    dsn = os.environ.get(args.dsn_env) or os.environ.get("DATABASE_URL")
    if not dsn:
        sys.exit(f"set {args.dsn_env} (or DATABASE_URL)")

    docs = [json.loads(line) for line in open(args.kb)]
    ids = [doc_id(d["metadata"]) for d in docs]
    assert len(set(ids)) == len(ids), "duplicate article ids in the knowledge base"
    if args.embeddings:
        z = np.load(args.embeddings)
        pos = {i: n for n, i in enumerate(z["ids"].tolist())}
        missing = [i for i in ids if i not in pos]
        assert not missing, f"{len(missing)} articles have no saved embedding, e.g. {missing[:3]}"
        M = z["M"][[pos[i] for i in ids]]
    else:
        M = embed([d["page_content"] for d in docs])
    assert M.shape == (len(docs), DIM), M.shape

    ssl = None if any(h in dsn for h in ("127.0.0.1", "localhost")) else "require"
    conn = await asyncpg.connect(dsn, ssl=ssl)
    try:
        await conn.execute(DDL)
        rows = [(i, d["page_content"], json.dumps(pinecone_style_metadata(d["metadata"])),
                 "[" + ",".join(f"{x:.6g}" for x in vec) + "]") for i, d, vec in zip(ids, docs, M)]
        sql = """INSERT INTO kb_article (id, content, metadata, embedding) VALUES ($1, $2, $3::jsonb, $4::halfvec)
                 ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata,
                                                embedding = EXCLUDED.embedding"""
        for i in range(0, len(rows), 250):
            async with conn.transaction():
                await conn.executemany(sql, rows[i:i + 250])
            print(f"  upserted {min(i + 250, len(rows))}/{len(rows)}", flush=True)
        await conn.execute("ANALYZE kb_article")
        n = await conn.fetchval("SELECT count(*) FROM kb_article")
        size = await conn.fetchval("SELECT pg_size_pretty(pg_total_relation_size('kb_article'))")
        print(f"kb_article: {n} rows, {size} including index and text")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
