"""Knowledge bases in Postgres (kb.py, kb_index.py, vector_store.py, ingest.jobs.write_articles).

Needs VECTOR_DATABASE_URL pointing at a local Postgres with pgvector holding the kb_article copy. The copy is only
read: everything runs in a throwaway database created on that server, holding the 5,924 original articles in the
*old* table layout (as production has them before this change), and dropped afterwards.

  .venv/bin/python -m pytest tests/test_kb.py -q
"""
import asyncio
import json
import os
import random
import sys
import urllib.parse
import uuid
from pathlib import Path

import numpy as np
import psycopg
import pytest
from langchain_core.documents import Document
from psycopg.conninfo import make_conninfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kb  # noqa: E402
import kb_index  # noqa: E402
import search_engine  # noqa: E402
from ingest.jobs import write_articles  # noqa: E402
from retrievers import load_docs_from_jsonl  # noqa: E402
from vector_store import DIM, INDEX, PgVectorStore  # noqa: E402
from test_vector_store import FakeEmbeddings, unit  # noqa: E402

BASE_DSN = os.environ.get("VECTOR_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not BASE_DSN, reason="VECTOR_DATABASE_URL (local pgvector copy) not set")
if BASE_DSN and urllib.parse.urlsplit(BASE_DSN).hostname not in ("localhost", "127.0.0.1"):
    raise SystemExit("VECTOR_DATABASE_URL must be a local database for these tests")

OLD_DDL = f"""CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE kb_article (id text PRIMARY KEY, content text NOT NULL, metadata jsonb NOT NULL,
                         embedding halfvec({DIM}) NOT NULL)"""
JSONL = ROOT / "knowledge_base.jsonl"


def with_db(url: str, name: str) -> str:
    """The same server URL with another database (a URL, as DATABASE_URL is: asyncpg needs one)."""
    return urllib.parse.urlunsplit(urllib.parse.urlsplit(url)._replace(path="/" + name))


@pytest.fixture(scope="module")
def db():
    """A throwaway database with the original articles in the pre-change layout, plus the HNSW index."""
    name = f"cgpt_kb_test_{uuid.uuid4().hex[:8]}"
    admin = make_conninfo(BASE_DSN, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    dsn = with_db(BASE_DSN, name)
    try:
        with psycopg.connect(dsn, autocommit=True) as dst, psycopg.connect(BASE_DSN) as src:
            dst.execute(OLD_DDL)
            with src.cursor().copy("COPY (SELECT id, content, metadata, embedding FROM kb_article) "
                                   "TO STDOUT (FORMAT BINARY)") as out, \
                    dst.cursor().copy("COPY kb_article (id, content, metadata, embedding) FROM STDIN "
                                      "(FORMAT BINARY)") as into:
                for chunk in out:
                    into.write(chunk)
            dst.execute("SET maintenance_work_mem = '512MB'")
            dst.execute(f"CREATE INDEX {INDEX} ON kb_article USING hnsw (embedding halfvec_ip_ops) "
                        f"WITH (m = 32, ef_construction = 200)")
            dst.execute("ANALYZE kb_article")
        yield dsn
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture(scope="module")
def migrated(db):
    """The rows' physical identity before the migration, then the migrated database."""
    with psycopg.connect(db) as conn:
        before = dict(conn.execute("SELECT id, ctid::text || ':' || xmin::text FROM kb_article").fetchall())
    kb.ensure_schema(db)
    kb.ensure_schema(db)  # idempotent
    return db, before


def run(coro, dsn):
    """Run an async kb.* call against `dsn` with a fresh pool."""
    async def go():
        os.environ["VECTOR_DATABASE_URL"] = dsn
        try:
            return await coro()
        finally:
            await kb.close_pool()
    old = os.environ.get("VECTOR_DATABASE_URL")
    try:
        return asyncio.run(go())
    finally:
        os.environ["VECTOR_DATABASE_URL"] = old or ""


# ---------- migration ----------

def test_migration_adds_columns_and_order_without_touching_rows(migrated):
    db, before = migrated
    with psycopg.connect(db) as conn:
        cols = {r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns "
                                           "WHERE table_name = 'kb_article'")}
        assert {"kb_id", "seq", "issue_id"} <= cols
        after = dict(conn.execute("SELECT id, ctid::text || ':' || xmin::text FROM kb_article").fetchall())
        assert after == before, "no row rewritten (so the HNSW index is untouched)"
        assert conn.execute("SELECT count(*), count(DISTINCT kb_id), min(kb_id) FROM kb_article").fetchone() == \
            (5924, 1, "sunbeam")
        assert conn.execute("SELECT count(*) FROM kb_legacy_order").fetchone()[0] == 5924
        assert conn.execute("SELECT id, default_on FROM knowledge_base").fetchall() == [("sunbeam", True)]
        for table in ("kb_user_access", "ingest_job", "ingest_page", "ingest_article"):
            assert conn.execute(f"SELECT to_regclass('{table}')").fetchone()[0] is not None


# ---------- the original articles load exactly as the jsonl had them ----------

def test_db_articles_are_the_jsonl_articles(migrated):
    db, _ = migrated
    index = kb_index.KnowledgeIndex()
    index.load(db, build_fuzzy=False)
    from_db = index.documents()
    from_file = load_docs_from_jsonl(str(JSONL))
    assert len(from_db) == len(from_file) == 5924
    for a, b in zip(from_db, from_file):
        assert a.page_content == b.page_content
        assert list(a.metadata.items()) == list(b.metadata.items()), (a.metadata, b.metadata)
    assert index.kb_ids_present() == {"sunbeam"} and index.covers_all(["sunbeam"])


def test_search_results_are_identical(migrated):
    """The Exact/Fuzzy commands and fuzzy retrieval give the same results from the database as from the jsonl."""
    db, _ = migrated
    from_db = kb_index.KnowledgeIndex()
    from_db.load(db)
    from_file = kb_index.KnowledgeIndex()
    from_file.set_documents([(str(n), "sunbeam", d) for n, d in enumerate(load_docs_from_jsonl(str(JSONL)))])
    queries = ["rathotsava", "Kojagiri Pournima", "hari jagara", "Parijnanashram Swamiji Shirali 1973",
               "vishweshwara temple", "Gurupoornima", "sanskrit", "Karla Durga Parameshwari"]
    for q in queries:
        assert [d.page_content for d in from_db.fuzzy_retriever(["sunbeam"]).invoke(q)] == \
            [d.page_content for d in from_file.fuzzy_retriever().invoke(q)], q
    saved = kb_index.INDEX
    try:
        for q, exact in [("(temple OR mandir) AND shirali", False), ("Rathotsava", True), ("swami AND guru", False)]:
            kb_index.INDEX = from_file
            expected = search_engine.search_knowledge_base(q, exact=exact)
            kb_index.INDEX = from_db
            assert search_engine.search_knowledge_base(q, exact=exact, kb_ids=["sunbeam"]) == expected, q
    finally:
        kb_index.INDEX = saved


# ---------- filtered vector search ----------

def test_filtered_vector_search_keeps_recall(migrated):
    """Like the original exactness check (HNSW vs exact search), on filtered queries: move ~5% of the articles to a
    second knowledge base and compare the filtered top 15 with an exact (sequential) search."""
    db, _ = migrated
    rng = random.Random(7)
    with psycopg.connect(db, autocommit=True) as conn:
        conn.execute("INSERT INTO knowledge_base (id, name, source_prefix) VALUES ('small', 'Small', 'Small') "
                     "ON CONFLICT DO NOTHING")
        ids = [r[0] for r in conn.execute("SELECT id FROM kb_article ORDER BY id").fetchall()]
        small = rng.sample(ids, 300)
        conn.execute("UPDATE kb_article SET kb_id = 'small' WHERE id = ANY(%s)", (small,))
        conn.execute("ANALYZE kb_article")
    store = PgVectorStore(FakeEmbeddings(), dsn=db)
    try:
        with store._pool.connection() as c:
            queries = [r[0].to_list() for r in c.execute(
                "SELECT embedding FROM kb_article WHERE id = ANY(%s)", (rng.sample(ids, 25),)).fetchall()]
        exact_conn = psycopg.connect(db)
        from pgvector.psycopg import register_vector
        register_vector(exact_conn)
        exact_conn.execute("SET enable_indexscan = off")
        from pgvector import HalfVector
        def recall(kbs):
            """Mean share of the 15 returned whose distance is within the exact 15th distance (the knowledge base
            has duplicate articles with identical embeddings, so the 15th place can be a tie)."""
            out = []
            for n, q in enumerate(queries):
                q = unit(np.asarray(q) + np.random.default_rng(n).normal(scale=0.01, size=DIM))
                got = [r[2] for r in store._nearest(q, 15, kb_ids=kbs)]
                where = "WHERE kb_id = ANY(%s) " if kbs else ""
                exact = [r[0] for r in exact_conn.execute(
                    f"SELECT embedding <#> %s FROM kb_article {where}ORDER BY embedding <#> %s LIMIT 15",
                    (HalfVector(q), *([kbs] if kbs else []), HalfVector(q))).fetchall()]
                out.append(sum(d <= exact[-1] + 1e-6 for d in got) / 15)
            return float(np.mean(out))

        unfiltered = recall(None)
        for kbs in (["small"], ["sunbeam"]):
            assert recall(kbs) >= min(0.98, unfiltered - 0.01), (kbs, recall(kbs), unfiltered)
        for kbs in (["sunbeam"], ["small"]):
            plan = store.explain(queries[0], kb_ids=kbs)
            assert f"Index Scan using {INDEX}" in plan and "Seq Scan" not in plan and "Sort" not in plan, plan
        with store._pool.connection() as c:
            assert c.execute("SHOW hnsw.iterative_scan").fetchone()[0] == "strict_order"
        exact_conn.close()
    finally:
        store.close()
        with psycopg.connect(db, autocommit=True) as conn:
            conn.execute("UPDATE kb_article SET kb_id = 'sunbeam' WHERE kb_id = 'small'")


# ---------- publishing, and the three search paths per knowledge base ----------

def vec(seed: int) -> list[float]:
    return unit(np.random.default_rng(seed).normal(size=DIM))


def rows(source, titles, seed=0):
    return [(f"{source}#{n:03d}", f"Zyxwv {t} unique text {seed}",
             {"source": source, "published_date": "01-01-2099", "title": t, "author": None, "summary": f"About {t}",
              "start_line": None, "end_line": None, "start_page": n, "end_page": n}, vec(seed * 100 + n))
            for n, t in enumerate(titles, 1)]


def article_ids(db, where="issue_id IS NOT NULL"):
    with psycopg.connect(db) as conn:
        return sorted(r[0] for r in conn.execute(f"SELECT id FROM kb_article WHERE {where}").fetchall())


def test_publish_replaces_an_earlier_publish_in_one_transaction(migrated):
    db, _ = migrated
    src = "Sunbeam-2099-January.txt"
    assert write_articles(db, 901, "sunbeam", src, rows(src, ["A", "B", "C"])) == set()
    assert article_ids(db) == [f"{src}#001", f"{src}#002", f"{src}#003"]
    with psycopg.connect(db) as conn:
        r = conn.execute("SELECT kb_id, issue_id, seq >= 1000000, metadata FROM kb_article WHERE id = %s",
                         (f"{src}#002",)).fetchone()
    assert r[:3] == ("sunbeam", 901, True)
    assert r[3]["start_line"] == "Unknown" and r[3]["start_page"] == 2.0, "stored like the original articles"
    # Publishing the same job again replaces its rows.
    assert write_articles(db, 901, "sunbeam", src, rows(src, ["A", "B2"], seed=1)) == {901}
    assert article_ids(db) == [f"{src}#001", f"{src}#002"]
    # Another upload of the same issue replaces the first one's rows; the original articles are never touched.
    assert write_articles(db, 902, "sunbeam", src, rows(src, ["A"], seed=2)) == {901}
    assert article_ids(db) == [f"{src}#001"] and len(article_ids(db, "issue_id IS NULL")) == 5924
    # A failure inside the transaction leaves the previous publish in place.
    broken = rows(src, ["X", "Y"], seed=3)
    broken[1] = (broken[0][0], *broken[1][1:])  # duplicate id
    with pytest.raises(psycopg.errors.UniqueViolation):
        write_articles(db, 903, "sunbeam", src, broken)
    with psycopg.connect(db) as conn:
        assert conn.execute("SELECT issue_id FROM kb_article WHERE issue_id IS NOT NULL").fetchall() == [(902,)]
    with psycopg.connect(db, autocommit=True) as conn:
        conn.execute("DELETE FROM kb_article WHERE issue_id IS NOT NULL")


def test_kb_filter_in_all_three_search_paths(migrated, monkeypatch):
    db, _ = migrated
    with psycopg.connect(db, autocommit=True) as conn:
        conn.execute("INSERT INTO knowledge_base (id, name, source_prefix) VALUES ('ksa', 'Kanara Saraswat', 'KSA') "
                     "ON CONFLICT DO NOTHING")
    src = "KSA-2099-March.txt"
    write_articles(db, 950, "ksa", src, rows(src, ["Quokkafest at Talmakiwadi"], seed=9))
    index = kb_index.KnowledgeIndex()
    index.load(db, build_fuzzy=False)
    try:
        assert index.kb_ids_present() == {"sunbeam", "ksa"} and not index.covers_all(["sunbeam"])
        ksa_doc = index.documents(["ksa"])
        assert [d.metadata["source"] for d in ksa_doc] == [src] and ksa_doc[0].metadata["start_line"] is None
        assert len(index.documents(["sunbeam"])) == 5924 and len(index.documents()) == 5925
        # 1. fuzzy retrieval (the agent's tool)
        assert [d.metadata["source"] for d in index.fuzzy_retriever(["ksa"]).invoke("quokkafest")] == [src]
        assert src not in [d.metadata["source"] for d in index.fuzzy_retriever(["sunbeam"]).invoke("quokkafest")]
        # 2. the Exact/Fuzzy Search commands
        monkeypatch.setattr(kb_index, "INDEX", index)
        assert f"(Source: {src})" in search_engine.search_knowledge_base("Quokkafest", exact=True, kb_ids=["ksa"])
        assert search_engine.search_knowledge_base("Quokkafest", exact=True, kb_ids=["sunbeam"]) == ""
        # 3. vector search
        store = PgVectorStore(FakeEmbeddings(), dsn=db)
        try:
            q = vec(9 * 100 + 1)
            assert [d.metadata["source"] for d in store.similarity_search_by_vector(q, 1, filter={"kb_id": ["ksa"]})] == [src]
            assert src not in [d.metadata["source"] for d in
                               store.max_marginal_relevance_search_by_vector(q, 5, 15, filter={"kb_id": ["sunbeam"]})]
            assert [d.metadata["source"] for d in store.similarity_search_by_vector(q, 1)] == [src], "no filter: all"
        finally:
            store.close()
        # A refresh after a publish picks up new articles without re-reading the rest.
        write_articles(db, 951, "ksa", "KSA-2099-April.txt", rows("KSA-2099-April.txt", ["Second"], seed=10))
        assert index.refresh(db) == (1, 0)
        with psycopg.connect(db, autocommit=True) as conn:
            conn.execute("DELETE FROM kb_article WHERE issue_id = 951")
        assert index.refresh(db) == (0, 1)
    finally:
        with psycopg.connect(db, autocommit=True) as conn:
            conn.execute("DELETE FROM kb_article WHERE issue_id IS NOT NULL")


# ---------- who searches what ----------

def test_access_defaults_and_switches(migrated):
    db, _ = migrated

    async def scenario():
        await kb.create_kb("ksa2", "Kanara Saraswat", "the magazine of the KSA", "KSAtwo", False, "admin@x")
        with pytest.raises(ValueError, match="already exists"):
            await kb.create_kb("ksa2", "Again", "", "Other", False, "admin@x")
        with pytest.raises(ValueError, match="lowercase"):
            await kb.create_kb("Bad Id", "x", "", "X", False, "admin@x")
        ids = lambda kbs: [k["id"] for k in kbs]  # noqa: E731
        assert ids(await kb.enabled_kbs("nobody@x")) == ["sunbeam"], "no rows: the defaults"
        await kb.set_access("Reader@X", "ksa2", True, "admin@x")
        await kb.set_access("reader@x", "sunbeam", False, "admin@x")
        assert ids(await kb.enabled_kbs("reader@x")) == ["ksa2"]
        await kb.update_kb("ksa2", "Kanara Saraswat", "the magazine of the KSA", True, "admin@x")
        assert ids(await kb.enabled_kbs("nobody@x")) == ["sunbeam", "ksa2"], "a new default reaches everyone"
        assert ids(await kb.enabled_kbs("reader@x")) == ["ksa2"], "own switches win over defaults"
        await kb.set_access("reader@x", "ksa2", False, "admin@x")
        assert await kb.enabled_kbs("reader@x") == []
        with pytest.raises(ValueError, match="can't be deleted"):
            await kb.delete_kb("sunbeam", "admin@x")
        await kb.delete_kb("ksa2", "admin@x")
        assert ids(await kb.list_kbs()) == [k for k in ids(await kb.list_kbs()) if k != "ksa2"]

    run(scenario, db)
