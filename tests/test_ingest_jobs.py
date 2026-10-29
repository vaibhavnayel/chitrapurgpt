"""Ingest jobs end to end with fakes: a presigned upload to a local S3 server (moto, standing in for R2), the worker
reading, splitting and summarising an issue, resuming after a crash, publishing, and deleting.

Needs VECTOR_DATABASE_URL (a local Postgres with pgvector): a throwaway database is created on that server and
dropped afterwards. No API calls: the LLM and the embeddings are fakes.

  .venv/bin/python -m pytest tests/test_ingest_jobs.py -q
"""
import asyncio
import hashlib
import json
import os
import socket
import sys
import urllib.parse
import uuid
from pathlib import Path

import httpx
import numpy as np
import psycopg
import pytest
from psycopg.conninfo import make_conninfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kb  # noqa: E402
import kb_index  # noqa: E402
from ingest import jobs, storage  # noqa: E402
from ingest.llm import Usage  # noqa: E402
from vector_store import DIM, PgVectorStore  # noqa: E402
from test_ingest_pipeline import make_pdf, read_json  # noqa: E402
from test_kb import with_db  # noqa: E402
from test_vector_store import FakeEmbeddings  # noqa: E402

BASE_DSN = os.environ.get("VECTOR_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not BASE_DSN, reason="VECTOR_DATABASE_URL (local pgvector server) not set")
if BASE_DSN and urllib.parse.urlsplit(BASE_DSN).hostname not in ("localhost", "127.0.0.1"):
    raise SystemExit("VECTOR_DATABASE_URL must be a local database for these tests")
PAGES = 6


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """A throwaway knowledge-base database, a local S3 server, and the env vars pointing at them."""
    from moto.server import ThreadedMotoServer
    name = f"cgpt_jobs_test_{uuid.uuid4().hex[:8]}"
    admin = make_conninfo(BASE_DSN, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    db = with_db(BASE_DSN, name)
    port = free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()
    saved = {k: os.environ.get(k) for k in ("VECTOR_DATABASE_URL", "R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID",
                                            "R2_SECRET_ACCESS_KEY", "R2_ACCOUNT_ID")}
    os.environ.update({"VECTOR_DATABASE_URL": db, "R2_ENDPOINT": f"http://127.0.0.1:{port}", "R2_BUCKET": "issues",
                       "R2_ACCESS_KEY_ID": "testing", "R2_SECRET_ACCESS_KEY": "testing"})
    os.environ.pop("R2_ACCOUNT_ID", None)
    saved_index = kb_index.INDEX
    try:
        store = PgVectorStore(FakeEmbeddings(), dsn=db)
        store.ensure_schema()
        store.close()
        kb.ensure_schema(db, legacy_jsonl=None)
        import boto3  # the bucket exists in R2 already; moto wants a plain region to create one
        boto3.client("s3", endpoint_url=f"http://127.0.0.1:{port}", region_name="us-east-1",
                     aws_access_key_id="testing", aws_secret_access_key="testing").create_bucket(Bucket="issues")
        kb_index.INDEX = kb_index.KnowledgeIndex()
        kb_index.INDEX.load(db)
        pdf = tmp_path_factory.mktemp("pdf") / "issue.pdf"
        make_pdf(pdf, [f"Page {p} of a test issue." for p in range(1, PAGES + 1)])
        yield {"db": db, "pdf": pdf.read_bytes()}
    finally:
        kb_index.INDEX = saved_index
        server.stop()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


SPLIT = json.dumps({"issue": {"label": "August 2099", "published_date": "01-08-2099"},
                    "articles": [{"title": "Cover", "kind": "cover", "ranges": ["p1.b0-p1.b1"]},
                                 {"title": "Gurupournima at Shirali", "kind": "article", "ranges": ["p2.b0-p4.b1"]},
                                 {"title": "Seva Rates", "kind": "notice", "ranges": ["p5.b0-p6.b1"]}],
                    "excluded": [], "notes": ""})


class WorkerLLM:
    """Page reads (two content blocks per page), one split, summaries. `gate` (an asyncio.Event) can hold page
    reads after `release` of them have been answered."""

    def __init__(self, release: int | None = None):
        self.calls = {"page": 0, "split": 0, "summary": 0}
        self.release, self.gate = release, None

    async def complete(self, *, model, messages, system=None, max_tokens=16000, thinking=None):
        content = messages[0]["content"]
        if system:  # the split
            self.calls["split"] += 1
            return SPLIT, Usage(model=model, input_tokens=5000, output_tokens=800, cost=0.02, stop_reason="end_turn")
        if isinstance(content, list) and content[0]["type"] == "image":
            if self.release is not None and self.calls["page"] >= self.release:
                await self.gate.wait()
            self.calls["page"] += 1
            reply = read_json(("headline", "GURUPOURNIMA AT SHIRALI"),
                              ("body", "Devotees gathered at the Math for Gurupournima with bhajans and pravachana."),
                              ("page_furniture", "The Chitrapur Sunbeam August 2099"))
            return reply, Usage(model=model, input_tokens=1500, output_tokens=300, cost=0.001, stop_reason="end_turn")
        self.calls["summary"] += 1
        return "A short summary.", Usage(model=model, input_tokens=500, output_tokens=20, cost=0.0001,
                                         stop_reason="end_turn")


def fake_embed(texts):
    return [(np.ones(DIM) / np.sqrt(DIM)).tolist() for _ in texts]


async def upload(pdf: bytes, filename: str) -> tuple[str, str]:
    sha = hashlib.sha256(pdf).hexdigest()
    key = storage.object_key("sunbeam", filename, sha)
    await asyncio.to_thread(storage.Storage().client.put_object, Bucket="issues", Key=key, Body=pdf)
    return key, sha


async def page_rows(job_id: int) -> int:
    pool = await kb.get_pool()
    return await pool.fetchval("SELECT count(*) FROM ingest_page WHERE job_id = $1", job_id)


def run(coro_fn):
    async def go():
        try:
            return await coro_fn()
        finally:
            await kb.close_pool()
    return asyncio.run(go())


# ---------- storage ----------

def test_presigned_upload_round_trip(env):
    store = storage.Storage()
    sha = hashlib.sha256(env["pdf"]).hexdigest()
    key = storage.object_key("sunbeam", "Sunbeam 2099-08 August.pdf", sha)
    assert key == f"sunbeam/2099-08/{sha}.pdf"
    url = store.presign_put(key)
    assert "X-Amz-Signature=" in url and urllib.parse.urlsplit(url).path.endswith(f"{sha}.pdf")
    # What the browser does: PUT the file to the presigned URL (no app server in between).
    r = httpx.put(url, content=env["pdf"], headers={"Content-Type": "application/pdf"})
    assert r.status_code == 200, r.text
    assert store.head(key)["ContentLength"] == len(env["pdf"]) and store.head("nope.pdf") is None
    assert store.read_start(key).startswith(b"%PDF-")
    out = Path(storage.__file__).parent / f"_tmp_{uuid.uuid4().hex}.pdf"
    try:
        assert store.download(key, str(out)) == sha
    finally:
        out.unlink(missing_ok=True)
    assert storage.issue_folder("Sunbeam August 2024 Issue.pdf") == "2024-08"
    assert storage.issue_folder("Sunbeam 2023-05 & 06 May & June.pdf") == "2023-05"
    assert len(storage.issue_folder("scan.pdf")) == 10, "no month in the name: the upload date"


# ---------- the worker ----------

def test_worker_processes_a_job_and_publishes_it(env):
    llm = WorkerLLM()

    async def scenario():
        key, sha = await upload(env["pdf"], "Sunbeam August 2099.pdf")
        job_id = await jobs.create_job("sunbeam", key, "Sunbeam August 2099.pdf", sha, len(env["pdf"]), "admin@x")
        assert [j["id"] for j in await jobs.same_file(sha)] == [job_id]
        worker = jobs.Worker(llm=llm, embed=fake_embed, concurrency=2)
        job = await worker.claim()
        assert job["id"] == job_id and job["status"] == "running" and await worker.claim() is None
        await worker.process(job)
        job = await jobs.get_job(job_id)
        assert job["status"] == "review", job["error"]
        assert job["pages"] == PAGES and await page_rows(job_id) == PAGES
        assert job["issue"]["source"] == "Sunbeam-2099-August.txt" and job["findings"]["unassigned"] == 0
        assert job["cost_usd"] == pytest.approx(PAGES * 0.001 + 0.02 + 3 * 0.0001)
        articles = await jobs.job_articles(job_id)
        assert [a.title for a in articles] == ["Cover", "Gurupournima at Shirali", "Seva Rates"]
        assert all(a.summary == "A short summary." for a in articles)
        assert llm.calls == {"page": PAGES, "split": 1, "summary": 3}
        listed = storage.Storage().client.list_objects_v2(Bucket="issues", Prefix=key.removesuffix(".pdf") + "/")
        assert len(listed.get("KeyCount") and listed["Contents"]) == PAGES, "a preview per page"

        # Review edits, then publish.
        await jobs.edit_article(job_id, 1, excluded=True, by="admin@x")
        await jobs.edit_article(job_id, 3, title="Seva Rates from 1-8-2099", by="admin@x")
        await jobs.request_publish(job_id, "admin@x")
        job = await worker.claim()
        assert job["status"] == "publishing"
        await worker.process(job)
        job = await jobs.get_job(job_id)
        assert job["status"] == "published" and job["published_articles"] == 2, job["error"]
        docs = kb_index.INDEX.documents(["sunbeam"])
        assert [d.metadata["title"] for d in docs] == ["Gurupournima at Shirali", "Seva Rates from 1-8-2099"]
        assert docs[0].metadata["source"] == "Sunbeam-2099-August.txt" and docs[0].metadata["summary"]
        assert docs[0].metadata["start_line"] is None and docs[0].metadata["start_page"] == 2
        with psycopg.connect(env["db"]) as conn:
            assert conn.execute("SELECT id, kb_id, issue_id FROM kb_article ORDER BY seq").fetchall() == [
                ("Sunbeam-2099-August.txt#002", "sunbeam", job_id), ("Sunbeam-2099-August.txt#003", "sunbeam", job_id)]

        # Deleting a published upload takes its articles out of search and its files out of storage.
        assert await jobs.delete_job(job_id, "admin@x", storage.Storage()) == 2
        assert kb_index.INDEX.documents() == [] and await jobs.get_job(job_id) is None
        assert storage.Storage().head(key) is None

    run(scenario)


def test_worker_resumes_after_a_crash(env):
    first = WorkerLLM(release=2)

    async def scenario():
        first.gate = asyncio.Event()
        key, sha = await upload(env["pdf"], "Sunbeam September 2099.pdf")
        job_id = await jobs.create_job("sunbeam", key, "Sunbeam September 2099.pdf", sha, len(env["pdf"]), "admin@x")
        worker = jobs.Worker(llm=first, embed=fake_embed, concurrency=1)
        task = asyncio.create_task(worker.process(await worker.claim()))
        for _ in range(200):
            if await page_rows(job_id) == 2:
                break
            await asyncio.sleep(0.05)
        task.cancel()  # the process dies: the job stays "running" with 2 pages read
        await asyncio.gather(task, return_exceptions=True)
        job = await jobs.get_job(job_id)
        assert job["status"] == "running" and await page_rows(job_id) == 2

        second = WorkerLLM()
        other = jobs.Worker(llm=second, embed=fake_embed, concurrency=1)
        assert await other.claim() is None, "its heartbeat is still fresh"
        pool = await kb.get_pool()
        await pool.execute("UPDATE ingest_job SET heartbeat_at = now() - interval '1 hour' WHERE id = $1", job_id)
        job = await other.claim()
        assert job["id"] == job_id and job["worker"] == other.name
        await other.process(job)
        job = await jobs.get_job(job_id)
        assert job["status"] == "review", job["error"]
        assert second.calls["page"] == PAGES - 2, "pages read before the crash are not read again"
        assert await page_rows(job_id) == PAGES
        assert job["cost_usd"] == pytest.approx(PAGES * 0.001 + 0.02 + 3 * 0.0001), "every page paid once"

        # Re-run split keeps the page reads; re-run whole job reads every page again.
        await jobs.request_rerun(job_id, "split", "admin@x")
        await other.process(await other.claim())
        assert second.calls["page"] == PAGES - 2 and second.calls["split"] == 2
        await jobs.request_rerun(job_id, "all", "admin@x")
        await other.process(await other.claim())
        assert second.calls["page"] == 2 * PAGES - 2
        with pytest.raises(jobs.JobError):
            await jobs.request_rerun(job_id, "all", "admin@x") or await jobs.request_rerun(job_id, "all", "admin@x")
        await jobs.delete_job(job_id, "admin@x", storage.Storage())

    run(scenario)


def test_a_bad_upload_fails_with_a_clear_message(env):
    async def scenario():
        key, sha = await upload(b"%PDF-1.4 not really a pdf", "broken.pdf")
        job_id = await jobs.create_job("sunbeam", key, "broken.pdf", sha, 25, "admin@x")
        worker = jobs.Worker(llm=WorkerLLM(), embed=fake_embed)
        await worker.process(await worker.claim())
        job = await jobs.get_job(job_id)
        assert job["status"] == "failed" and job["error"] == "This file couldn't be opened as a PDF."
        key2, sha2 = await upload(env["pdf"], "other.pdf")
        job2 = await jobs.create_job("sunbeam", key2, "other.pdf", "0" * 64, len(env["pdf"]), "admin@x")
        await worker.process(await worker.claim())
        assert "checksum" in (await jobs.get_job(job2))["error"]
        for j in (job_id, job2):
            await jobs.delete_job(j, "admin@x", storage.Storage())

    run(scenario)


def test_publishing_writes_chunks_and_a_republish_replaces_them(env):
    import chunking
    text = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(29)) + "." for p in range(30))  # 900 words
    vec = fake_embed(["x"])[0]
    meta = {"source": "Sunbeam-2098-May.txt", "title": "A long talk", "summary": "s", "start_page": 3, "end_page": 9}
    chunks = chunking.article_chunks("A long talk", text)
    assert len(chunks) > 1
    jobs.write_articles(env["db"], 9001, "sunbeam", "Sunbeam-2098-May.txt",
                        [("Sunbeam-2098-May.txt#001", text, meta, vec)],
                        [("Sunbeam-2098-May.txt#001", n, s, e, chunking.shorten(vec)) for n, (s, e, _) in enumerate(chunks)])
    with psycopg.connect(env["db"]) as conn:
        got = conn.execute("SELECT n, start_char, end_char, kb_id FROM kb_chunk WHERE article_id = %s ORDER BY n",
                           ("Sunbeam-2098-May.txt#001",)).fetchall()
    assert got == [(n, s, e, "sunbeam") for n, (s, e, _) in enumerate(chunks)]

    # Another upload of the same issue replaces the articles, and their chunks with them.
    replaced = jobs.write_articles(env["db"], 9002, "sunbeam", "Sunbeam-2098-May.txt",
                                   [("Sunbeam-2098-May.txt#001", "Now short.", meta, vec)], [])
    with psycopg.connect(env["db"], autocommit=True) as conn:
        assert replaced == {9001}
        assert conn.execute("SELECT count(*) FROM kb_chunk").fetchone()[0] == 0
        conn.execute("DELETE FROM kb_article WHERE issue_id = 9002")
