"""Ingest work in a helper process (ingest/isolated.py): an issue is read there and comes back for review exactly as
when it's read in the web process; a failure there is shown the same way; stopping a job stops its helper; a
website is read and written there, while this process's search index is refreshed. The fakes go into the helper
through INGEST_CHILD_SETUP (tests/helper_fakes.py).

Needs VECTOR_DATABASE_URL (a local Postgres with pgvector), like test_ingest_jobs.py, except the first test.
  .venv/bin/python -m pytest tests/test_isolated.py -q
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kb  # noqa: E402
import kb_index  # noqa: E402
from ingest import isolated, jobs, sources, storage  # noqa: E402
from test_ingest_jobs import BASE_DSN, PAGES, env, fake_embed, page_rows, run, upload  # noqa: E402,F401

needs_db = pytest.mark.skipif(not BASE_DSN, reason="VECTOR_DATABASE_URL (local pgvector server) not set")


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.setenv("INGEST_ISOLATE", "1")
    monkeypatch.setenv("INGEST_CHILD_SETUP", "helper_fakes:install")
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "tests"))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_helper_that_dies_is_reported(helper, monkeypatch):
    monkeypatch.setenv("INGEST_CHILD_SETUP", "helper_fakes:crash")
    with pytest.raises(isolated.ChildError, match="stopped unexpectedly, exit code 5"):
        asyncio.run(isolated.run("job", 1, "worker", 0))


def test_the_worker_checks_helpers_work_and_falls_back_if_not(helper, monkeypatch):
    monkeypatch.setattr(isolated, "_fallen_back", False)
    assert asyncio.run(isolated.check()) and isolated.enabled()
    monkeypatch.setenv("INGEST_CHILD_SETUP", "helper_fakes:crash")
    assert not asyncio.run(isolated.check())
    assert not isolated.enabled(), "from now on the work is done in the web process"
    assert not jobs.Worker().isolate


@needs_db
def test_an_issue_is_read_in_a_helper_process(env, helper):
    async def scenario():
        key, sha = await upload(env["pdf"], "Sunbeam September 2099.pdf")
        job_id = await jobs.create_job("sunbeam", key, "Sunbeam September 2099.pdf", sha, len(env["pdf"]), "admin@x")
        worker = jobs.Worker(embed=fake_embed)
        assert worker.isolate, "no fakes passed in: reading goes to a helper"
        await worker.process(await worker.claim())
        job = await jobs.get_job(job_id)
        assert job["status"] == "review", job["error"]
        assert job["worker"] == worker.name and job["pages"] == PAGES and await page_rows(job_id) == PAGES
        assert job["issue"]["source"] == "Sunbeam-2099-August.txt", "named by the (fake) split"
        assert [a.title for a in await jobs.job_articles(job_id)] == ["Cover", "Gurupournima at Shirali", "Seva Rates"]
        assert job["cost_usd"] == pytest.approx(PAGES * 0.001 + 0.02 + 3 * 0.0001)
        listed = storage.Storage().client.list_objects_v2(Bucket="issues", Prefix=key.removesuffix(".pdf") + "/")
        assert listed["KeyCount"] == PAGES, "a preview per page"

        # Publishing stays in this process: it refreshes this process's search index.
        await jobs.request_publish(job_id, "admin@x")
        await worker.process(await worker.claim())
        assert (await jobs.get_job(job_id))["status"] == "published"
        assert "Gurupournima at Shirali" in [d.metadata["title"] for d in kb_index.INDEX.documents(["sunbeam"])]
        await jobs.delete_job(job_id, "admin@x", storage.Storage())

    run(scenario)


@needs_db
def test_a_failure_in_the_helper_is_shown_as_before(env, helper):
    async def scenario():
        key, sha = await upload(b"%PDF-1.4 not really", "Sunbeam October 2099.pdf")
        job_id = await jobs.create_job("sunbeam", key, "Sunbeam October 2099.pdf", sha, 19, "admin@x")
        worker = jobs.Worker(embed=fake_embed)
        await worker.process(await worker.claim())
        job = await jobs.get_job(job_id)
        assert job["status"] == "failed" and job["error"] == "This file couldn't be opened as a PDF."
        await jobs.delete_job(job_id, "admin@x", storage.Storage())

    run(scenario)


@needs_db
def test_stopping_a_job_stops_its_helper(env, helper, monkeypatch, tmp_path):
    started = tmp_path / "helper.pid"
    monkeypatch.setenv("INGEST_CHILD_SETUP", "helper_fakes:install_slow")
    monkeypatch.setenv("HELPER_STARTED", str(started))
    monkeypatch.setattr(jobs, "HEARTBEAT_S", 0.2)

    async def scenario():
        key, sha = await upload(env["pdf"], "Sunbeam November 2099.pdf")
        job_id = await jobs.create_job("sunbeam", key, "Sunbeam November 2099.pdf", sha, len(env["pdf"]), "admin@x")
        worker = jobs.Worker(embed=fake_embed)
        work = asyncio.create_task(worker.process(await worker.claim()))
        for _ in range(300):
            if started.exists():
                break
            await asyncio.sleep(0.1)
        pid = int(started.read_text())
        assert alive(pid)
        await jobs.stop_job(job_id, "admin@x")
        await asyncio.wait_for(work, 30)
        assert not alive(pid), "the helper was stopped with the job"
        job = await jobs.get_job(job_id)
        assert job["status"] == "failed" and job["error"] == "stopped by admin@x"
        await jobs.delete_job(job_id, "admin@x", storage.Storage())

    run(scenario)


@needs_db
def test_a_website_is_read_and_written_in_helper_processes(env, helper):
    async def wait_for(sync_id, *statuses):
        for _ in range(600):
            sync = await sources.get_sync(sync_id)
            if sync["status"] in statuses:
                return sync
            await asyncio.sleep(0.05)
        raise AssertionError(f"sync {sync_id} is {sync['status']}: {sync.get('error')}")

    async def scenario():
        pool = await kb.get_pool()
        await pool.execute("INSERT INTO knowledge_base (id, name, source_prefix) VALUES ('scm', 'SCM Website', 'SCM') "
                           "ON CONFLICT DO NOTHING")
        source = await sources.get_or_create_source("scm", "website", "example.org", "admin@x",
                                                    {"start": "https://example.org/"})
        sync_id = await sources.start_refresh(source["id"], "admin@x")
        sync = await wait_for(sync_id, "checked", "failed")
        assert sync["status"] == "checked" and sync["plan"]["new"] == 2, sync.get("error")
        assert await sources.start_apply(sync_id, "admin@x")
        sync = await wait_for(sync_id, "done")
        assert sync["result"]["added"] == 2 and sync["result"]["embedded"] == 2 and "peak_mb" not in sync["result"]
        assert sorted(d.metadata["title"] for d in kb_index.INDEX.documents(["scm"])) == \
            ["Gurupournima", "Shrimat Anandashram Swamiji"], "this process's index was refreshed"

    run(scenario)
