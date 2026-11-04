"""Linked collections: entries that come from somewhere other than an uploaded PDF (a YouTube channel's video
lists, a website). Each entry is one article in kb_article (with source_id set), searched and cited like any other.

A source (kb_source) belongs to one collection. A sync (kb_sync) turns a fresh list of entries into that collection:
new entries are added, changed ones replaced, and entries that aren't in the new list removed. Only new and changed
entries are embedded, so re-sending the same list costs nothing.

Article ids are "src<source>:<key>:<digest>". The key is the entry's identity within its source (e.g.
"video:B0Z0WHuFjS8"), the digest a hash of its text and metadata: a changed entry gets a new id, which is what the
in-memory index (kb_index.refresh) and the live site's minute-by-minute index check go by.

For a YouTube upload the entries are worked out when the admin sends the files (a quick parse), shown as a
"checked" sync with its counts, and written when they press Apply. Applying runs in the background of the web
process; a sync that was applying when the process stopped is marked as interrupted at the next start (applying it
again is safe: it works from what's in the database then).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

KINDS = {"youtube": "YouTube video lists", "website": "Website"}
_APPLY_LOCK = 727277  # pg advisory lock: one sync writes at a time (publishing issues uses 727276)
EMBED_USD_PER_M = 0.13  # text-embedding-3-large

SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_source (
    id          bigserial PRIMARY KEY,
    kb_id       text NOT NULL REFERENCES knowledge_base(id) ON DELETE CASCADE,
    kind        text NOT NULL,
    name        text NOT NULL,
    config      jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_by  text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    synced_at   timestamptz,
    UNIQUE (kb_id, kind, name)
);
CREATE TABLE IF NOT EXISTS kb_sync (
    id          bigserial PRIMARY KEY,
    source_id   bigint NOT NULL REFERENCES kb_source(id) ON DELETE CASCADE,
    status      text NOT NULL,           -- checked, applying, done, failed, discarded, superseded
    entries     jsonb,                   -- the new list (dropped once applied)
    plan        jsonb,                   -- counts and sample titles: new / changed / removed / unchanged
    notes       jsonb,                   -- what the admin should know (skipped rows, missing descriptions...)
    result      jsonb,
    error       text,
    cost_usd    double precision NOT NULL DEFAULT 0,
    created_by  text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    started_at  timestamptz,
    finished_at timestamptz
);
"""


@dataclass
class Entry:
    key: str                    # identity within the source, e.g. "video:B0Z0WHuFjS8"
    title: str
    text: str                   # stored, read by the readers, searched
    metadata: dict              # source, published_date, title, url, kind, ...; shown to the agent
    summary: str = ""           # English summary, embedded instead of the text when the text is mostly Indic

    @property
    def digest(self) -> str:
        blob = json.dumps([self.text, self.metadata, self.summary], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def article_id(self, source_id: int) -> str:
        return f"src{source_id}:{self.key}:{self.digest}"

    def as_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "text": self.text, "metadata": self.metadata,
                "summary": self.summary}

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        return cls(key=d["key"], title=d["title"], text=d["text"], metadata=d["metadata"],
                   summary=d.get("summary") or "")


def split_id(article_id: str, source_id: int) -> tuple[str, str]:
    """(key, digest) of one of this source's article ids."""
    rest = article_id.removeprefix(f"src{source_id}:")
    key, _, digest = rest.rpartition(":")
    return key, digest


@dataclass
class Plan:
    new: list[Entry] = field(default_factory=list)
    changed: list[Entry] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)       # old article ids
    removed_titles: list[str] = field(default_factory=list)
    unchanged: int = 0
    replaced_ids: list[str] = field(default_factory=list)  # old ids of the changed entries

    def summary(self, sample: int = 8) -> dict:
        return {"new": len(self.new), "changed": len(self.changed), "removed": len(self.removed),
                "unchanged": self.unchanged,
                "new_titles": [e.title for e in self.new[:sample]],
                "changed_titles": [e.title for e in self.changed[:sample]],
                "removed_titles": self.removed_titles[:sample]}


def make_plan(source_id: int, entries: list[Entry], current: dict[str, str]) -> Plan:
    """current: {article id: title} of the source's articles now."""
    by_key: dict[str, tuple[str, str]] = {}
    for article_id in current:
        key, digest = split_id(article_id, source_id)
        by_key[key] = (article_id, digest)
    plan, seen = Plan(), set()
    for e in entries:
        if e.key in seen:
            continue
        seen.add(e.key)
        old = by_key.get(e.key)
        if old is None:
            plan.new.append(e)
        elif old[1] != e.digest:
            plan.changed.append(e)
            plan.replaced_ids.append(old[0])
        else:
            plan.unchanged += 1
    for key, (article_id, _) in by_key.items():
        if key not in seen:
            plan.removed.append(article_id)
            plan.removed_titles.append(current[article_id])
    return plan


# ---------- database ----------

async def _pool():
    import kb
    return await kb.get_pool()


async def get_or_create_source(kb_id: str, kind: str, name: str, created_by: str, config: dict | None = None) -> dict:
    pool = await _pool()
    row = await pool.fetchrow("""INSERT INTO kb_source (kb_id, kind, name, config, created_by)
                                 VALUES ($1, $2, $3, $4, $5)
                                 ON CONFLICT (kb_id, kind, name) DO UPDATE SET name = EXCLUDED.name
                                 RETURNING *""", kb_id, kind, name, config or {}, created_by)
    return dict(row)


async def get_source(source_id: int) -> dict | None:
    pool = await _pool()
    row = await pool.fetchrow("SELECT * FROM kb_source WHERE id = $1", source_id)
    return dict(row) if row else None


async def list_sources() -> list[dict]:
    pool = await _pool()
    if not await pool.fetchval("SELECT to_regclass('kb_article') IS NOT NULL"):
        return []  # no articles table yet (a fresh database: see kb.ensure_schema)
    rows = await pool.fetch("""SELECT s.*, k.name AS kb_name,
                                      (SELECT count(*) FROM kb_article a WHERE a.source_id = s.id) AS articles
                               FROM kb_source s JOIN knowledge_base k ON k.id = s.kb_id ORDER BY s.id""")
    return [dict(r) for r in rows]


async def current_articles(source_id: int) -> dict[str, str]:
    pool = await _pool()
    return {r["id"]: r["title"] or "" for r in await pool.fetch(
        "SELECT id, metadata->>'title' AS title FROM kb_article WHERE source_id = $1", source_id)}


async def create_check(source_id: int, entries: list[Entry], notes: list[str], created_by: str) -> int:
    """A "checked" sync: the new list and what applying it would change. Older checks of the source are dropped."""
    plan = make_plan(source_id, entries, await current_articles(source_id))
    pool = await _pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("UPDATE kb_sync SET status = 'superseded', entries = NULL WHERE source_id = $1 "
                           "AND status = 'checked'", source_id)
        return await conn.fetchval(
            """INSERT INTO kb_sync (source_id, status, entries, plan, notes, created_by)
               VALUES ($1, 'checked', $2, $3, $4, $5) RETURNING id""",
            source_id, [e.as_dict() for e in entries], plan.summary(), notes, created_by)


async def get_sync(sync_id: int) -> dict | None:
    pool = await _pool()
    row = await pool.fetchrow("""SELECT y.id, y.source_id, y.status, y.plan, y.notes, y.result, y.error, y.cost_usd,
                                        y.created_by, y.created_at, y.started_at, y.finished_at,
                                        y.entries IS NOT NULL AS has_entries
                                 FROM kb_sync y WHERE y.id = $1""", sync_id)
    return dict(row) if row else None


async def list_syncs(source_id: int, limit: int = 5) -> list[dict]:
    pool = await _pool()
    return [dict(r) for r in await pool.fetch(
        """SELECT id, status, plan, result, error, created_by, created_at, finished_at FROM kb_sync
           WHERE source_id = $1 AND status <> 'superseded' ORDER BY id DESC LIMIT $2""", source_id, limit)]


async def discard(sync_id: int) -> bool:
    pool = await _pool()
    status = await pool.execute("UPDATE kb_sync SET status = 'discarded', entries = NULL, finished_at = now() "
                                "WHERE id = $1 AND status = 'checked'", sync_id)
    return status.endswith(" 1")


STUCK_MINUTES = 20  # an apply takes a few minutes; one "applying" for longer was cut off by a restart


async def mark_interrupted(minutes: int = STUCK_MINUTES) -> int:
    """Put syncs left 'applying' by a process that stopped back to 'checked', so they can be applied again. Only
    those started over `minutes` ago: a PR preview shares the database and may start while the live site applies."""
    pool = await _pool()
    status = await pool.execute(
        "UPDATE kb_sync SET status = 'checked', error = 'Interrupted (the app restarted). Press Apply again.' "
        "WHERE status = 'applying' AND entries IS NOT NULL AND started_at < now() - make_interval(mins => $1)",
        minutes)
    return int(status.split()[-1])


# ---------- applying ----------

_tasks: set[asyncio.Task] = set()


def _embed(texts: list[str]) -> list[list[float]]:
    from ingest.jobs import _embed as embed
    return embed(texts)


async def start_apply(sync_id: int, by: str, *, embed=None, db: str | None = None) -> bool:
    """Mark a checked sync as applying and run it in the background. False if it isn't waiting to be applied."""
    pool = await _pool()
    status = await pool.execute("UPDATE kb_sync SET status = 'applying', error = NULL, started_at = now() "
                                "WHERE id = $1 AND status = 'checked' AND entries IS NOT NULL", sync_id)
    if not status.endswith(" 1"):
        return False
    logger.info("Admin %s applies sync %d", by, sync_id)
    task = asyncio.get_running_loop().create_task(apply(sync_id, embed=embed, db=db), name=f"kb-sync-{sync_id}")
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True


async def apply(sync_id: int, *, embed=None, db: str | None = None) -> None:
    import kb_index
    pool = await _pool()
    row = await pool.fetchrow("SELECT y.source_id, y.entries, s.kb_id FROM kb_sync y JOIN kb_source s "
                              "ON s.id = y.source_id WHERE y.id = $1", sync_id)
    t = time.monotonic()
    try:
        entries = [Entry.from_dict(d) for d in row["entries"]]
        source_id, kb_id = row["source_id"], row["kb_id"]
        plan = make_plan(source_id, entries, await current_articles(source_id))
        result = await asyncio.to_thread(write_plan, db, source_id, kb_id, plan, embed or _embed)
        if plan.new or plan.changed or plan.removed:
            await asyncio.to_thread(kb_index.INDEX.refresh, db)
        result["secs"] = round(time.monotonic() - t, 1)
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("UPDATE kb_sync SET status = 'done', entries = NULL, result = $2, cost_usd = $3, "
                               "error = NULL, finished_at = now() WHERE id = $1", sync_id, result, result["cost_usd"])
            await conn.execute("UPDATE kb_source SET synced_at = now() WHERE id = $1", source_id)
        logger.info("Sync %d applied: %s", sync_id, result)
    except Exception as e:  # noqa: BLE001  (shown on the sync page; the collection is as it was)
        logger.exception("Sync %d failed", sync_id)
        await pool.execute("UPDATE kb_sync SET status = 'checked', error = $2 WHERE id = $1", sync_id,
                           f"Couldn't apply it: {e.__class__.__name__}: {str(e)[:300]}. Nothing was changed.")


def write_plan(db: str | None, source_id: int, kb_id: str, plan: Plan, embed) -> dict:
    """Embed the new and changed entries, then in one transaction remove the old rows and add the new ones."""
    import psycopg
    from pgvector import HalfVector
    from pgvector.psycopg import register_vector
    from psycopg.types.json import Jsonb

    import chunking
    import kb
    from ingest.jobs import embedding_text
    from vector_store import pinecone_style_metadata

    todo = plan.new + plan.changed
    texts = [embedding_text(e.title, e.summary, e.text) for e in todo]
    chunks = [(ei, n, start, end, text) for ei, e in enumerate(todo)
              for n, (start, end, text) in enumerate(chunking.article_chunks(e.title, e.text))]
    all_texts = texts + [c[4] for c in chunks]
    vectors = embed(all_texts) if all_texts else []
    rows = [(e.article_id(source_id), e.text, pinecone_style_metadata(e.metadata), v)
            for e, v in zip(todo, vectors[:len(texts)])]
    chunk_rows = [(rows[ei][0], n, start, end, chunking.shorten(v))
                  for (ei, n, start, end, _), v in zip(chunks, vectors[len(texts):])]
    with psycopg.connect(db or kb.dsn()) as conn:
        register_vector(conn)
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_APPLY_LOCK,))
            gone = plan.removed + plan.replaced_ids
            if gone:
                conn.execute("DELETE FROM kb_article WHERE source_id = %s AND id = ANY(%s)", (source_id, gone))
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO kb_article (id, content, metadata, embedding, kb_id, seq, source_id)
                       VALUES (%s, %s, %s, %s, %s, nextval('kb_article_seq'), %s)
                       ON CONFLICT (id) DO NOTHING""",
                    [(i, text, Jsonb(meta), HalfVector(vec), kb_id, source_id) for i, text, meta, vec in rows])
                cur.executemany(
                    """INSERT INTO kb_chunk (article_id, n, start_char, end_char, kb_id, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                    [(a, n, s, e, kb_id, HalfVector(vec)) for a, n, s, e, vec in chunk_rows])
    chars = sum(len(t) for t in all_texts)
    return {"added": len(plan.new), "changed": len(plan.changed), "removed": len(plan.removed),
            "unchanged": plan.unchanged, "embedded": len(all_texts), "chunks": len(chunk_rows),
            "cost_usd": round(chars / 4 / 1e6 * EMBED_USD_PER_M, 4)}
