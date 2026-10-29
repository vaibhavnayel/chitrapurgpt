"""Knowledge bases (collections of articles, e.g. "Sunbeam", later "KSA") and which ones each user searches.

Everything lives in the knowledge-base database, next to the articles: vector_store.vector_dsn(), i.e.
VECTOR_DATABASE_URL, else DATABASE_URL (in production both are the app's Postgres).

  knowledge_base   one row per collection: id (slug), name, description, source_prefix ("Sunbeam" ->
                   Sunbeam-2025-February.txt), default_on (searched by users with no setting of their own)
  kb_user_access   per-user switches set at /admin: (email, kb_id, enabled); no row means the KB's default
  kb_article       the articles (vector_store.py), now with kb_id (existing rows: sunbeam), seq (load order of
                   published articles) and issue_id (the ingest job that published it; NULL for the original import)
  kb_legacy_order  the original import's article order (knowledge_base.jsonl line numbers), so articles load in the
                   same order as before and fuzzy-search ties break the same way. Filled once, from the jsonl.

ensure_schema() makes all of that exist; it's idempotent, takes an advisory lock, adds columns only when they're
missing and never rewrites existing kb_article rows (so the HNSW index is untouched). It runs at startup.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextvars import ContextVar
from pathlib import Path

import asyncpg
import psycopg

import vector_store

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
LEGACY_JSONL = ROOT / "knowledge_base.jsonl"
DEFAULT_KB = "sunbeam"
SUNBEAM = {"id": "sunbeam", "name": "The Chitrapur Sunbeam", "description": "the magazine of Shri Chitrapur Math",
           "source_prefix": "Sunbeam", "default_on": True}
_SCHEMA_LOCK = 727275  # pg advisory lock: two instances starting together migrate once
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,30}$")

# The metadata keys of every article, in the original knowledge base's order.
METADATA_KEYS = ("source", "published_date", "title", "author", "summary", "start_line", "end_line", "start_page",
                 "end_page")
_INT_KEYS = ("start_line", "end_line", "start_page", "end_page")

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_base (
    id            text PRIMARY KEY,
    name          text NOT NULL,
    description   text NOT NULL DEFAULT '',
    source_prefix text NOT NULL UNIQUE,
    default_on    boolean NOT NULL DEFAULT false,
    created_by    text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS kb_user_access (
    email      text NOT NULL,
    kb_id      text NOT NULL REFERENCES knowledge_base(id) ON DELETE CASCADE,
    enabled    boolean NOT NULL,
    set_by     text,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (email, kb_id)
);
CREATE TABLE IF NOT EXISTS kb_legacy_order (
    id  text PRIMARY KEY,
    seq bigint NOT NULL
);
CREATE SEQUENCE IF NOT EXISTS kb_article_seq START 1000000;
"""

# Columns added to an existing kb_article (vector_store.DDL creates new tables with them). Deliberately no index on
# kb_id: with one, the planner answers a filtered nearest-neighbour query with an exact scan of the matching rows
# (sorted by distance) instead of the HNSW index, which is far slower for a large knowledge base.
ARTICLE_COLUMNS = {"kb_id": "text NOT NULL DEFAULT 'sunbeam'", "seq": "bigint", "issue_id": "bigint"}
ARTICLE_INDEXES = {"kb_article_issue_id": "CREATE INDEX kb_article_issue_id ON kb_article (issue_id)"}


def dsn() -> str | None:
    return vector_store.vector_dsn()


# ---------- schema (sync: runs at import time in steps.py, and from tests) ----------

def ensure_schema(db: str | None = None, *, legacy_jsonl: Path | None = LEGACY_JSONL) -> None:
    from ingest import jobs  # the ingest tables live here too

    db = db or dsn()
    if not db:
        raise RuntimeError("No knowledge-base database configured: set VECTOR_DATABASE_URL or DATABASE_URL")
    with psycopg.connect(db, autocommit=True) as conn:
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK,))
            conn.execute(SCHEMA)
            conn.execute(jobs.SCHEMA)
            conn.execute("""INSERT INTO knowledge_base (id, name, description, source_prefix, default_on, created_by)
                            VALUES (%(id)s, %(name)s, %(description)s, %(source_prefix)s, %(default_on)s, 'setup')
                            ON CONFLICT (id) DO NOTHING""", SUNBEAM)
        if conn.execute("SELECT to_regclass('kb_article')").fetchone()[0] is None:
            return  # no articles table here (vector_store.ensure_schema / scripts/load_vectors.py create it)
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK,))
            have = {r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'kb_article'")}
            for name, decl in ARTICLE_COLUMNS.items():
                if name not in have:
                    conn.execute(f"ALTER TABLE kb_article ADD COLUMN {name} {decl}")
                    logger.info("kb_article: added column %s", name)
            indexes = {r[0] for r in conn.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'kb_article'")}
            for name, sql in ARTICLE_INDEXES.items():
                if name not in indexes:
                    conn.execute(sql)
            conn.execute(vector_store.CHUNK_DDL)  # chunks of long articles (chunking.py); idempotent
        if legacy_jsonl is not None:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK,))
                _backfill_legacy_order(conn, legacy_jsonl)


def _backfill_legacy_order(conn, legacy_jsonl: Path) -> None:
    """Once: the original articles' order, from knowledge_base.jsonl (the file they were loaded from)."""
    if conn.execute("SELECT EXISTS (SELECT 1 FROM kb_legacy_order)").fetchone()[0]:
        return
    if not conn.execute("SELECT EXISTS (SELECT 1 FROM kb_article WHERE issue_id IS NULL)").fetchone()[0]:
        return
    if not legacy_jsonl.exists():
        logger.warning("%s is missing: the original articles will load in id order", legacy_jsonl)
        return
    ids = []
    with legacy_jsonl.open() as f:
        for line in f:
            ids.append(vector_store.doc_id(json.loads(line)["metadata"]))
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO kb_legacy_order (id, seq) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                        [(i, n) for n, i in enumerate(ids)])
    logger.info("kb_legacy_order: recorded the order of %d original articles", len(ids))


# ---------- articles (sync, for the in-memory index) ----------

def doc_metadata(metadata: dict) -> dict:
    """An article's metadata as the original knowledge base had it (what fuzzy search and the commands show),
    from the vector-store form: "Unknown" back to None, whole floats back to ints, keys in the original order.
    Exact for the original articles (checked by tests/test_kb.py against knowledge_base.jsonl)."""
    out = {}
    for k in [*METADATA_KEYS, *(k for k in metadata if k not in METADATA_KEYS)]:
        if k not in metadata:
            continue
        v = metadata[k]
        if v == "Unknown":
            v = None
        elif k in _INT_KEYS and isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    return out


_ORDER = "ORDER BY coalesce(a.seq, o.seq, 9223372036854775807), a.id"


def article_keys(db: str | None = None) -> list[tuple[str, str]]:
    """(id, kb_id) of every article, in load order."""
    with psycopg.connect(db or dsn()) as conn:
        return conn.execute(f"SELECT a.id, a.kb_id FROM kb_article a LEFT JOIN kb_legacy_order o USING (id) "
                            f"{_ORDER}").fetchall()


def load_articles(ids: list[str] | None = None, db: str | None = None) -> dict[str, tuple[str, str, dict]]:
    """{id: (kb_id, content, doc metadata)} for the given ids (all articles if None)."""
    with psycopg.connect(db or dsn()) as conn:
        if ids is None:
            rows = conn.execute("SELECT id, kb_id, content, metadata FROM kb_article").fetchall()
        else:
            rows = conn.execute("SELECT id, kb_id, content, metadata FROM kb_article WHERE id = ANY(%s)",
                                (list(ids),)).fetchall()
    return {r[0]: (r[1], r[2], doc_metadata(r[3])) for r in rows}


# ---------- which knowledge bases a turn searches ----------

# Set per turn by app.py (the user's enabled KBs, as dicts with id/name/description); read by the search tool and
# the system prompt in steps.py. None means no restriction (tests, scripts).
SEARCH_KBS: ContextVar[tuple[dict, ...] | None] = ContextVar("search_kbs", default=None)


def search_kb_ids() -> list[str] | None:
    kbs = SEARCH_KBS.get()
    return None if kbs is None else [k["id"] for k in kbs]


# ---------- async access for the app and /admin ----------

_pool = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                db = dsn()
                if not db:
                    raise RuntimeError("No knowledge-base database configured")
                await asyncio.to_thread(ensure_schema, db)
                _pool = await asyncpg.create_pool(db, min_size=0, max_size=4, init=_init_connection)
    return _pool


async def _init_connection(conn) -> None:
    for kind in ("json", "jsonb"):
        await conn.set_type_codec(kind, encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def valid_kb_id(kb_id: str) -> bool:
    return bool(_ID_RE.match(kb_id or ""))


def valid_prefix(prefix: str) -> bool:
    return bool(_PREFIX_RE.match(prefix or ""))


async def list_kbs(with_counts: bool = False) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch("SELECT id, name, description, source_prefix, default_on, created_by, created_at "
                            "FROM knowledge_base ORDER BY (id <> 'sunbeam'), name")
    kbs = [dict(r) for r in rows]
    if with_counts:
        has_articles = await pool.fetchval("SELECT to_regclass('kb_article') IS NOT NULL")
        counts = {r["kb_id"]: r["n"] for r in await pool.fetch(
            "SELECT kb_id, count(*) AS n FROM kb_article GROUP BY kb_id")} if has_articles else {}
        for kb in kbs:
            kb["articles"] = counts.get(kb["id"], 0)
    return kbs


async def get_kb(kb_id: str) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow("SELECT id, name, description, source_prefix, default_on FROM knowledge_base "
                              "WHERE id = $1", kb_id)
    return dict(row) if row else None


async def create_kb(kb_id: str, name: str, description: str, source_prefix: str, default_on: bool,
                    created_by: str) -> None:
    if not valid_kb_id(kb_id):
        raise ValueError("The id must be lowercase letters, digits or dashes (e.g. ksa).")
    if not valid_prefix(source_prefix):
        raise ValueError("The source prefix must be letters and digits only (e.g. KSA).")
    if not name.strip():
        raise ValueError("Give the collection a name.")
    pool = await get_pool()
    try:
        await pool.execute("""INSERT INTO knowledge_base (id, name, description, source_prefix, default_on, created_by)
                              VALUES ($1, $2, $3, $4, $5, $6)""",
                           kb_id, name.strip(), description.strip(), source_prefix, default_on, created_by)
    except asyncpg.UniqueViolationError:
        raise ValueError("A collection with that id or source prefix already exists.") from None
    logger.info("Admin %s created knowledge base %s", created_by, kb_id)


async def update_kb(kb_id: str, name: str, description: str, default_on: bool, updated_by: str) -> None:
    if not name.strip():
        raise ValueError("Give the collection a name.")
    pool = await get_pool()
    await pool.execute("UPDATE knowledge_base SET name = $2, description = $3, default_on = $4, updated_at = now() "
                       "WHERE id = $1", kb_id, name.strip(), description.strip(), default_on)
    logger.info("Admin %s updated knowledge base %s (default_on=%s)", updated_by, kb_id, default_on)


async def delete_kb(kb_id: str, deleted_by: str) -> None:
    """Only an empty collection can be deleted (no articles, no uploads)."""
    if kb_id == DEFAULT_KB:
        raise ValueError("The Sunbeam collection can't be deleted.")
    pool = await get_pool()
    if await pool.fetchval("SELECT EXISTS (SELECT 1 FROM kb_article WHERE kb_id = $1)", kb_id):
        raise ValueError("This collection has articles. Delete its issues first.")
    if await pool.fetchval("SELECT EXISTS (SELECT 1 FROM ingest_job WHERE kb_id = $1)", kb_id):
        raise ValueError("This collection has uploaded issues. Delete them first.")
    await pool.execute("DELETE FROM knowledge_base WHERE id = $1", kb_id)
    logger.info("Admin %s deleted knowledge base %s", deleted_by, kb_id)


async def access_rows() -> dict[str, dict[str, bool]]:
    """{email: {kb_id: enabled}} for users with their own settings."""
    pool = await get_pool()
    out: dict[str, dict[str, bool]] = {}
    for r in await pool.fetch("SELECT email, kb_id, enabled FROM kb_user_access"):
        out.setdefault(r["email"], {})[r["kb_id"]] = r["enabled"]
    return out


def resolve(kbs: list[dict], own: dict[str, bool]) -> list[dict]:
    """The KBs a user searches: their own switch where they have one, else the KB's default."""
    return [kb for kb in kbs if own.get(kb["id"], kb["default_on"])]


async def enabled_kbs(email: str | None) -> list[dict]:
    """The knowledge bases this user searches, as dicts (id, name, description). Raises if the database can't be
    reached (the caller decides what to do)."""
    pool = await get_pool()
    kbs = [dict(r) for r in await pool.fetch(
        "SELECT id, name, description, default_on FROM knowledge_base ORDER BY (id <> 'sunbeam'), name")]
    own = {r["kb_id"]: r["enabled"] for r in await pool.fetch(
        "SELECT kb_id, enabled FROM kb_user_access WHERE email = $1", (email or "").strip().lower())}
    return [{"id": kb["id"], "name": kb["name"], "description": kb["description"]} for kb in resolve(kbs, own)]


async def set_access(email: str, kb_id: str, enabled: bool, set_by: str) -> None:
    pool = await get_pool()
    await pool.execute("""INSERT INTO kb_user_access (email, kb_id, enabled, set_by) VALUES ($1, $2, $3, $4)
                          ON CONFLICT (email, kb_id) DO UPDATE SET enabled = EXCLUDED.enabled,
                                                                   set_by = EXCLUDED.set_by, updated_at = now()""",
                       email.strip().lower(), kb_id, enabled, set_by)
    logger.info("Admin %s turned %s %s for %s", set_by, kb_id, "on" if enabled else "off", email)
