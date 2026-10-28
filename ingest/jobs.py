"""Ingest jobs in Postgres, the background worker that runs them, and publishing.

One job per uploaded PDF. The worker is one asyncio task in the web process (Render runs a single instance): it
claims a job with FOR UPDATE SKIP LOCKED, runs one at a time, and heartbeats while it works. A job left "running"
by a process that died (restart, redeploy) has a stale heartbeat and is claimed again; its page reads are in
ingest_page, so it resumes where it stopped.

  queued -> running -> review --(admin: Publish)--> publish -> publishing -> published
                    `-> failed            (admin: Re-run split / Re-run whole job -> queued; Stop -> failed)
  A newer publish of the same issue marks the older job "replaced".

Publishing embeds the articles (text-embedding-3-large, as the existing ones), then in one transaction deletes any
rows from an earlier publish of the same issue and inserts the new ones into kb_article; then the in-memory index
is refreshed. The original articles (issue_id NULL) are never touched.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import tempfile
import time
import uuid
import weakref
from pathlib import Path

import models

from . import assemble as asm
from .llm import AnthropicLLM, Spend
from .pages import PageResult, publication_phrase
from .pdf import PdfFile, page_count
from .pipeline import IssueConfig, process_issue
from .storage import Storage, preview_key

logger = logging.getLogger(__name__)

MAX_PAGES = 400
HEARTBEAT_S = 20
STALE_S = 180          # a running job whose heartbeat is older than this is claimed again
POLL_S = 5
INDEX_CHECK_S = 60     # how often the worker checks whether the article table changed under the index
_PUBLISH_LOCK = 727276

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_job (
    id            bigserial PRIMARY KEY,
    kb_id         text NOT NULL REFERENCES knowledge_base(id),
    object_key    text NOT NULL,
    filename      text NOT NULL,
    sha256        text NOT NULL,
    size_bytes    bigint NOT NULL,
    pages         int,
    status        text NOT NULL DEFAULT 'queued',
    stage         text,                  -- what a queued job redoes: 'all' (default) or 'split'
    progress      jsonb,
    error         text,
    issue         jsonb,                 -- {label, published_date, source}
    split         jsonb,
    findings      jsonb,
    flags         jsonb,
    costs         jsonb,
    secs          jsonb,
    cost_usd      double precision NOT NULL DEFAULT 0,
    created_by    text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    started_at    timestamptz,
    finished_at   timestamptz,
    heartbeat_at  timestamptz,
    worker        text,
    published_at  timestamptz,
    published_by  text,
    published_articles int
);
CREATE INDEX IF NOT EXISTS ingest_job_status ON ingest_job (status);
CREATE INDEX IF NOT EXISTS ingest_job_sha256 ON ingest_job (sha256);
CREATE TABLE IF NOT EXISTS ingest_page (
    job_id  bigint NOT NULL REFERENCES ingest_job(id) ON DELETE CASCADE,
    page    int NOT NULL,
    result  jsonb NOT NULL,
    PRIMARY KEY (job_id, page)
);
CREATE TABLE IF NOT EXISTS ingest_article (
    job_id  bigint NOT NULL REFERENCES ingest_job(id) ON DELETE CASCADE,
    n       int NOT NULL,
    data    jsonb NOT NULL,
    PRIMARY KEY (job_id, n)
);
"""

ACTIVE = ("running", "publishing")


class JobError(RuntimeError):
    """A failure the admin should see as is (e.g. "the file isn't a PDF")."""


async def _pool():
    import kb
    return await kb.get_pool()


# ---------- jobs (for /admin) ----------

async def create_job(kb_id: str, object_key: str, filename: str, sha256: str, size_bytes: int,
                     created_by: str) -> int:
    pool = await _pool()
    job_id = await pool.fetchval("""INSERT INTO ingest_job (kb_id, object_key, filename, sha256, size_bytes, created_by)
                                    VALUES ($1, $2, $3, $4, $5, $6) RETURNING id""",
                                 kb_id, object_key, filename, sha256, size_bytes, created_by)
    logger.info("Admin %s uploaded %s to %s (job %d)", created_by, filename, kb_id, job_id)
    WAKE.set()
    return job_id


async def same_file(sha256: str) -> list[dict]:
    pool = await _pool()
    return [dict(r) for r in await pool.fetch(
        "SELECT id, kb_id, filename, status, created_at, issue FROM ingest_job WHERE sha256 = $1 ORDER BY id",
        sha256)]


async def get_job(job_id: int) -> dict | None:
    pool = await _pool()
    row = await pool.fetchrow("SELECT * FROM ingest_job WHERE id = $1", job_id)
    return dict(row) if row else None


async def list_jobs(limit: int = 200) -> list[dict]:
    pool = await _pool()
    return [dict(r) for r in await pool.fetch(
        """SELECT id, kb_id, filename, pages, status, stage, progress, error, issue, findings, flags, cost_usd,
                  created_by, created_at, finished_at, published_at, published_articles,
                  (SELECT count(*) FROM ingest_article a WHERE a.job_id = j.id AND NOT coalesce((a.data->>'excluded')::boolean, false)) AS articles
           FROM ingest_job j ORDER BY id DESC LIMIT $1""", limit)]


async def job_pages(job_id: int) -> list[PageResult]:
    pool = await _pool()
    return [PageResult.from_dict(r["result"]) for r in await pool.fetch(
        "SELECT result FROM ingest_page WHERE job_id = $1 ORDER BY page", job_id)]


async def job_articles(job_id: int) -> list[asm.Article]:
    pool = await _pool()
    return [asm.Article.from_dict(r["data"]) for r in await pool.fetch(
        "SELECT data FROM ingest_article WHERE job_id = $1 ORDER BY n", job_id)]


async def _editable(job_id: int) -> dict:
    job = await get_job(job_id)
    if job is None:
        raise JobError("That upload doesn't exist any more.")
    if job["status"] not in ("review", "published", "replaced"):
        raise JobError("Articles can only be changed while the issue is ready for review or published.")
    return job


async def edit_article(job_id: int, n: int, *, title: str | None = None, excluded: bool | None = None,
                       by: str = "") -> None:
    """Rename or (un)exclude an article. A published issue needs publishing again for the change to reach search."""
    await _editable(job_id)
    pool = await _pool()
    data = await pool.fetchval("SELECT data FROM ingest_article WHERE job_id = $1 AND n = $2", job_id, n)
    if data is None:
        raise JobError("That article doesn't exist.")
    if title is not None:
        if not title.strip():
            raise JobError("A title can't be empty.")
        data["title"] = title.strip()
    if excluded is not None:
        data["excluded"] = excluded
    await pool.execute("UPDATE ingest_article SET data = $3 WHERE job_id = $1 AND n = $2", job_id, n, data)
    await pool.execute("UPDATE ingest_job SET updated_at = now() WHERE id = $1", job_id)
    logger.info("Admin %s edited article %d of job %d (title=%r, excluded=%r)", by, n, job_id, title, excluded)


async def edit_issue(job_id: int, label: str, published_date: str, source: str, by: str) -> None:
    job = await _editable(job_id)
    if not asm.parse_date(published_date):
        raise JobError("The date must look like 01-08-2024 (DD-MM-YYYY).")
    if not source.strip().endswith(".txt") or "/" in source:
        raise JobError("The source name must end in .txt (e.g. Sunbeam-2024-August.txt).")
    issue = dict(job["issue"] or {})
    issue.update(label=label.strip(), published_date=published_date.strip(), source=source.strip())
    pool = await _pool()
    await pool.execute("UPDATE ingest_job SET issue = $2, updated_at = now() WHERE id = $1", job_id, issue)
    logger.info("Admin %s set job %d's issue to %r", by, job_id, issue)


async def request_publish(job_id: int, by: str) -> None:
    job = await get_job(job_id)
    if job is None or job["status"] not in ("review", "published", "replaced"):
        raise JobError("Only an issue that is ready for review (or already published) can be published.")
    pool = await _pool()
    await pool.execute("UPDATE ingest_job SET status = 'publish', published_by = $2, error = NULL, updated_at = now() "
                       "WHERE id = $1", job_id, by)
    logger.info("Admin %s asked to publish job %d", by, job_id)
    WAKE.set()


async def request_rerun(job_id: int, stage: str, by: str) -> None:
    """stage 'split': keep the page reads, split again. 'all': read every page again too."""
    job = await get_job(job_id)
    if job is None:
        raise JobError("That upload doesn't exist any more.")
    if job["status"] in ACTIVE or job["status"] in ("queued", "publish"):
        raise JobError("This upload is already queued or being processed.")
    pool = await _pool()
    async with pool.acquire() as conn, conn.transaction():
        if stage == "all":
            await conn.execute("DELETE FROM ingest_page WHERE job_id = $1", job_id)
        await conn.execute("UPDATE ingest_job SET status = 'queued', stage = $2, error = NULL, progress = NULL, "
                           "finished_at = NULL, updated_at = now() WHERE id = $1", job_id, stage)
    logger.info("Admin %s re-ran job %d (%s)", by, job_id, stage)
    WAKE.set()


async def stop_job(job_id: int, by: str) -> None:
    pool = await _pool()
    await pool.execute("UPDATE ingest_job SET status = 'failed', error = $2, updated_at = now() "
                       "WHERE id = $1 AND status IN ('queued', 'running')", job_id, f"stopped by {by}")
    logger.info("Admin %s stopped job %d", by, job_id)


async def delete_job(job_id: int, by: str, storage: Storage | None = None) -> int:
    """Delete an upload: its articles leave the knowledge base if it was published, and the PDF and its page
    previews are deleted from R2 unless another upload is the same file. Returns the number of articles removed
    from search."""
    import kb_index
    job = await get_job(job_id)
    if job is None:
        return 0
    if job["status"] in ACTIVE:
        raise JobError("This upload is being processed. Stop it first.")
    pool = await _pool()
    async with pool.acquire() as conn, conn.transaction():
        removed = int((await conn.execute("DELETE FROM kb_article WHERE issue_id = $1", job_id)).split()[-1]) \
            if await conn.fetchval("SELECT to_regclass('kb_article') IS NOT NULL") else 0
        await conn.execute("DELETE FROM ingest_job WHERE id = $1", job_id)
        shared = await conn.fetchval("SELECT EXISTS (SELECT 1 FROM ingest_job WHERE object_key = $1)",
                                     job["object_key"])
    if removed:
        await asyncio.to_thread(kb_index.INDEX.refresh)
    if storage is not None and not shared:
        try:
            await asyncio.to_thread(storage.delete_issue, job["object_key"])
        except Exception:  # noqa: BLE001  (the job is gone either way; leftover files only cost storage)
            logger.exception("Couldn't delete %s from R2", job["object_key"])
    logger.info("Admin %s deleted job %d (%d articles removed from search)", by, job_id, removed)
    return removed


# ---------- the worker ----------

class _Wake:
    """Set when a job is created or queued in this process, so the worker doesn't wait for its next poll. One
    asyncio.Event per event loop (tests run several loops)."""

    def __init__(self):
        self._events: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Event]" = weakref.WeakKeyDictionary()

    def event(self) -> asyncio.Event:
        loop = asyncio.get_running_loop()
        if loop not in self._events:
            self._events[loop] = asyncio.Event()
        return self._events[loop]

    def set(self) -> None:
        for ev in list(self._events.values()):
            ev.set()


WAKE = _Wake()


class DbPageStore:
    """Page reads of one job, in ingest_page. Saving a page also adds its cost to the job and updates progress."""

    def __init__(self, job_id: int, total: int):
        self.job_id, self.total, self.new_cost = job_id, total, 0.0

    async def load(self) -> dict[int, dict]:
        pool = await _pool()
        return {r["page"]: r["result"] for r in await pool.fetch(
            "SELECT page, result FROM ingest_page WHERE job_id = $1", self.job_id)}

    async def save(self, result: PageResult) -> None:
        pool = await _pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("""INSERT INTO ingest_page (job_id, page, result) VALUES ($1, $2, $3)
                                  ON CONFLICT (job_id, page) DO UPDATE SET result = EXCLUDED.result""",
                               self.job_id, result.page, result.as_dict())
            done = await conn.fetchval("SELECT count(*) FROM ingest_page WHERE job_id = $1", self.job_id)
            await conn.execute("UPDATE ingest_job SET cost_usd = cost_usd + $2, progress = $3, updated_at = now() "
                               "WHERE id = $1", self.job_id, result.cost,
                               {"stage": "read", "done": done, "total": self.total})
        self.new_cost += result.cost


def _embed(texts: list[str]) -> list[list[float]]:
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=64, max_retries=6).embed_documents(texts)


class Worker:
    def __init__(self, *, storage_factory=Storage, llm=None, embed=_embed, db: str | None = None,
                 concurrency: int | None = None):
        self.storage_factory, self.llm, self.embed = storage_factory, llm, embed
        self.db = db
        self.concurrency = concurrency  # pages read at once (default: pipeline.PAGE_CONCURRENCY)
        self.name = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self._stopping = False
        self._last_index_check = 0.0
        self._index_signature = None

    # ---- loop ----

    async def run(self) -> None:
        logger.info("Ingest worker %s started", self.name)
        while not self._stopping:
            try:
                job = await self.claim()
                if job is None:
                    await self._maybe_refresh_index()
                    wake = WAKE.event()
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(wake.wait(), timeout=POLL_S)
                    wake.clear()
                    continue
                await self.process(job)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001  (never let the loop die; a database blip shouldn't stop ingest)
                logger.exception("Ingest worker loop error")
                await asyncio.sleep(POLL_S)

    def stop(self) -> None:
        self._stopping = True
        WAKE.set()

    async def claim(self) -> dict | None:
        pool = await _pool()
        row = await pool.fetchrow(
            """UPDATE ingest_job j
               SET status = CASE WHEN j.status IN ('publish', 'publishing') THEN 'publishing' ELSE 'running' END,
                   worker = $1, heartbeat_at = now(), started_at = coalesce(j.started_at, now()), updated_at = now()
               WHERE j.id = (SELECT id FROM ingest_job
                             WHERE status IN ('queued', 'publish')
                                OR (status IN ('running', 'publishing')
                                    AND heartbeat_at < now() - make_interval(secs => $2))
                             ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1)
               RETURNING j.*""", self.name, float(STALE_S))
        return dict(row) if row else None

    async def process(self, job: dict) -> None:
        work = asyncio.create_task(self._publish(job) if job["status"] == "publishing" else self._run(job))
        beat = asyncio.create_task(self._heartbeat(job["id"], work))
        try:
            await work
        except asyncio.CancelledError:
            if not work.cancelled():
                raise
            logger.info("Job %d was stopped", job["id"])
        except Exception as e:  # noqa: BLE001
            logger.exception("Job %d failed", job["id"])
            message = str(e) if isinstance(e, JobError) else f"{type(e).__name__}: {e}"
            pool = await _pool()
            await pool.execute("UPDATE ingest_job SET status = 'failed', error = $2, finished_at = now(), "
                               "updated_at = now() WHERE id = $1 AND worker = $3", job["id"], message[:2000],
                               self.name)
        finally:
            beat.cancel()

    async def _heartbeat(self, job_id: int, work: asyncio.Task) -> None:
        pool = await _pool()
        while not work.done():
            await asyncio.sleep(HEARTBEAT_S)
            status = await pool.execute("UPDATE ingest_job SET heartbeat_at = now() WHERE id = $1 AND worker = $2 "
                                        "AND status IN ('running', 'publishing')", job_id, self.name)
            if status.endswith(" 0"):  # stopped, deleted or taken over: give up this run
                work.cancel()
                return

    # ---- reading, splitting, summaries ----

    async def _run(self, job: dict) -> None:
        import kb
        job_id = job["id"]
        pool = await _pool()
        kb_row = await kb.get_kb(job["kb_id"])
        if kb_row is None:
            raise JobError(f"The collection {job['kb_id']!r} doesn't exist any more.")
        cfg = IssueConfig(publication=publication_phrase(kb_row["name"], kb_row["description"]),
                          source_prefix=kb_row["source_prefix"], reader_model=models.INGEST_READER_MODEL,
                          summary_model=models.INGEST_READER_MODEL, reread_model=models.INGEST_MODEL,
                          split_model=models.INGEST_MODEL)
        if self.concurrency:
            cfg.concurrency = self.concurrency
        storage = self.storage_factory()
        llm = self.llm or AnthropicLLM()
        spend = Spend()
        with tempfile.TemporaryDirectory(prefix="cgpt-ingest-") as tmp:
            path = str(Path(tmp) / "issue.pdf")
            await pool.execute("UPDATE ingest_job SET progress = $2 WHERE id = $1", job_id, {"stage": "download"})
            sha = await asyncio.to_thread(storage.download, job["object_key"], path)
            if sha != job["sha256"]:
                raise JobError("The uploaded file isn't the one that was announced (its checksum differs). "
                               "Please upload it again.")
            try:
                pages = await asyncio.to_thread(page_count, path)
            except Exception:  # noqa: BLE001
                raise JobError("This file couldn't be opened as a PDF.") from None
            if not 0 < pages <= MAX_PAGES:
                raise JobError(f"This PDF has {pages} pages; issues of 1 to {MAX_PAGES} pages can be processed.")
            await pool.execute("UPDATE ingest_job SET pages = $2 WHERE id = $1", job_id, pages)
            store = DbPageStore(job_id, pages)

            async def progress(kind: str, data: dict) -> None:
                if kind == "stage":
                    await pool.execute("UPDATE ingest_job SET progress = $2 WHERE id = $1", job_id, data)

            async def preview(pdf: PdfFile, page: int) -> None:
                try:
                    jpeg = await asyncio.to_thread(pdf.preview, page)
                    await asyncio.to_thread(storage.put_bytes, preview_key(job["object_key"], page), jpeg,
                                            "image/jpeg")
                except Exception:  # noqa: BLE001  (the review page renders missing previews on demand)
                    logger.exception("Couldn't store the preview of page %d of job %d", page, job_id)

            t = time.monotonic()
            result = await process_issue(path, store, llm, cfg, spend=spend, summaries=True, progress=progress,
                                         on_page=preview)
        extra = spend.usd - store.new_cost  # split + summaries (page costs were added as they were saved)
        issue = vars(result.issue) if result.issue else None
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM ingest_article WHERE job_id = $1", job_id)
            await conn.executemany("INSERT INTO ingest_article (job_id, n, data) VALUES ($1, $2, $3)",
                                   [(job_id, a.n, a.as_dict()) for a in result.articles])
            await conn.execute(
                """UPDATE ingest_job SET status = 'review', stage = NULL, progress = NULL, issue = $2, split = $3,
                          findings = $4, flags = $5, costs = $6, secs = $7, cost_usd = cost_usd + $8,
                          finished_at = now(), updated_at = now()
                   WHERE id = $1 AND worker = $9""",
                job_id, issue, result.split, result.findings, result.flags,
                {k: round(v, 5) for k, v in result.costs.items()},
                {**{k: round(v, 1) for k, v in result.secs.items()}, "wall": round(time.monotonic() - t, 1)},
                extra, self.name)
        logger.info("Job %d ready for review: %d articles, %d flags, $%.3f this run", job_id, len(result.articles),
                    len(result.flags), spend.usd)

    # ---- publishing ----

    async def _publish(self, job: dict) -> None:
        import kb_index
        job_id = job["id"]
        pool = await _pool()
        issue = job["issue"] or {}
        if not issue.get("source"):
            raise JobError("The issue has no source name. Set its label, date and source name first.")
        articles = [a for a in await job_articles(job_id) if not a.excluded and a.text.strip()]
        if not articles:
            raise JobError("No articles to publish (all are excluded).")
        info = asm.IssueInfo(label=issue.get("label", ""), published_date=issue.get("published_date", ""),
                             source=issue["source"])
        texts = [a.text for a in articles]
        vectors = await asyncio.to_thread(self.embed, texts)
        rows = [(f"{info.source}#{a.n:03d}", a.text, asm.doc_metadata(a, info), v) for a, v in zip(articles, vectors)]
        replaced = await asyncio.to_thread(write_articles, self.db, job_id, job["kb_id"], info.source, rows)
        await asyncio.to_thread(kb_index.INDEX.refresh, self.db)
        async with pool.acquire() as conn, conn.transaction():
            if replaced:
                await conn.execute("UPDATE ingest_job SET status = 'replaced', updated_at = now() "
                                   "WHERE id = ANY($1::bigint[]) AND id <> $2", list(replaced), job_id)
            await conn.execute("UPDATE ingest_job SET status = 'published', published_at = now(), "
                               "published_articles = $2, error = NULL, updated_at = now() WHERE id = $1 AND worker = $3",
                               job_id, len(rows), self.name)
        logger.info("Job %d published: %d articles in %s (replaced jobs %s)", job_id, len(rows), job["kb_id"],
                    sorted(replaced - {job_id}) or "none")

    async def _maybe_refresh_index(self) -> None:
        """Pick up articles published by another process (or loaded by hand) every minute or so."""
        import kb_index
        if time.monotonic() - self._last_index_check < INDEX_CHECK_S or not kb_index.INDEX.loaded:
            return
        self._last_index_check = time.monotonic()
        pool = await _pool()
        if not await pool.fetchval("SELECT to_regclass('kb_article') IS NOT NULL"):
            return
        signature = tuple(await pool.fetchrow("SELECT count(*), coalesce(max(seq), 0) FROM kb_article"))
        if self._index_signature is not None and signature != self._index_signature:
            await asyncio.to_thread(kb_index.INDEX.refresh, self.db)
        self._index_signature = signature


def write_articles(db: str | None, job_id: int, kb_id: str, source: str, rows: list[tuple]) -> set[int]:
    """In one transaction: delete the rows of any earlier publish of this issue (this job, or another upload of the
    same issue in the same knowledge base), insert the new ones. Returns the job ids whose rows were replaced."""
    import psycopg
    from pgvector import HalfVector
    from pgvector.psycopg import register_vector
    from psycopg.types.json import Jsonb

    import kb
    from vector_store import pinecone_style_metadata

    with psycopg.connect(db or kb.dsn()) as conn:
        register_vector(conn)
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_PUBLISH_LOCK,))
            replaced = {r[0] for r in conn.execute(
                """DELETE FROM kb_article
                   WHERE issue_id = %s OR (kb_id = %s AND issue_id IS NOT NULL AND metadata->>'source' = %s)
                   RETURNING issue_id""", (job_id, kb_id, source)).fetchall()}
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO kb_article (id, content, metadata, embedding, kb_id, seq, issue_id)
                       VALUES (%s, %s, %s, %s, %s, nextval('kb_article_seq'), %s)""",
                    [(i, text, Jsonb(pinecone_style_metadata(meta)), HalfVector(vec), kb_id, job_id)
                     for i, text, meta, vec in rows])
    return replaced


_worker: Worker | None = None


def start_worker() -> Worker | None:
    """Start the worker on the running event loop (app startup). INGEST_WORKER=0 turns it off."""
    global _worker
    if os.environ.get("INGEST_WORKER", "1") == "0" or _worker is not None:
        return _worker
    _worker = Worker()
    asyncio.get_running_loop().create_task(_worker.run(), name="ingest-worker")
    return _worker
