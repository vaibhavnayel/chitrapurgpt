"""Add chunk embeddings (chunking.py) for every article that has none yet. Idempotent and resumable: it commits
every BATCH articles and skips articles that already have chunks. New issues get their chunks when published.

    .venv/bin/python scripts/build_chunks.py --dry-run     # counts, estimated cost and size; writes nothing
    .venv/bin/python scripts/build_chunks.py

DSN: VECTOR_DATABASE_URL, else DATABASE_URL. Embeddings: text-embedding-3-large via OPENAI_API_KEY (and
OPENAI_BASE_URL), as ingest/jobs.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chunking  # noqa: E402
import vector_store  # noqa: E402

BATCH = 200            # articles per transaction
PRICE_PER_M = 0.13     # text-embedding-3-large, $ per million tokens
BYTES_PER_CHUNK = 2 * chunking.CHUNK_DIM * 2 + 400   # halfvec in the table and in the HNSW index, plus overhead


def todo(conn) -> list[tuple]:
    return conn.execute(f"""SELECT a.id, a.kb_id, coalesce(a.metadata->>'title', ''), a.content
                            FROM {vector_store.TABLE} a
                            WHERE NOT EXISTS (SELECT 1 FROM {vector_store.CHUNK_TABLE} c WHERE c.article_id = a.id)
                            ORDER BY a.id""").fetchall()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dsn = vector_store.vector_dsn()
    if not dsn:
        sys.exit("set VECTOR_DATABASE_URL or DATABASE_URL")
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(vector_store.CHUNK_DDL)
        register_vector(conn)
        articles = todo(conn)
        plan = [(a, chunking.article_chunks(a[2], a[3])) for a in articles]
        plan = [(a, c) for a, c in plan if c]
        n_chunks = sum(len(c) for _, c in plan)
        words = sum(len(t.split()) for _, c in plan for _, _, t in c)
        print(f"{len(articles)} articles without chunks; {len(plan)} are long enough: {n_chunks} chunks, "
              f"~{words / 1e6:.1f}M words, ~${words * 1.35 / 1e6 * PRICE_PER_M:.2f} to embed, "
              f"~{n_chunks * BYTES_PER_CHUNK / 1e6:.0f} MB", flush=True)
        if args.dry_run or not plan:
            return
        from ingest.jobs import _embed
        t0, done = time.time(), 0
        for i in range(0, len(plan), BATCH):
            part = plan[i:i + BATCH]
            texts = [t for _, c in part for _, _, t in c]
            vectors = iter(_embed(texts))
            rows = [(a[0], n, s, e, a[1], HalfVector(chunking.shorten(next(vectors))))
                    for a, c in part for n, (s, e, _) in enumerate(c)]
            with conn.transaction(), conn.cursor() as cur:
                cur.executemany(f"""INSERT INTO {vector_store.CHUNK_TABLE}
                                    (article_id, n, start_char, end_char, kb_id, embedding)
                                    VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""", rows)
            done += len(rows)
            print(f"  {i + len(part)}/{len(plan)} articles, {done} chunks, {time.time() - t0:.0f}s", flush=True)
        total = conn.execute(f"SELECT count(*), pg_total_relation_size('{vector_store.CHUNK_TABLE}') "
                             f"FROM {vector_store.CHUNK_TABLE}").fetchone()
        print(f"done: {total[0]} chunks, table + index {total[1] / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
