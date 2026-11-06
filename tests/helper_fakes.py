"""Fakes for ingest work in a helper process (ingest/isolated.py): the helper calls one of these first when
INGEST_CHILD_SETUP=helper_fakes:<function> (tests/ on PYTHONPATH). Used by tests/test_isolated.py."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install() -> None:
    """Fake page reads, split and summaries (the same as test_ingest_jobs.WorkerLLM), fake embeddings, and a fake
    website crawl."""
    from ingest import jobs, llm, sources
    from test_ingest_jobs import WorkerLLM
    from test_vector_store import hashed

    jobs.AnthropicLLM = WorkerLLM
    llm.AnthropicLLM = WorkerLLM
    sources._embed = lambda texts: [hashed(t) for t in texts]

    async def crawl(config, progress=None):
        if progress:
            await progress({"pages_read": 2, "pages_found": 2, "minutes": 0})
        e = sources.Entry
        return ([e(key="page:/a", title="Shrimat Anandashram Swamiji", text="A page about Shrimat Anandashram.",
                   metadata={"title": "Shrimat Anandashram Swamiji", "url": "https://example.org/a"}),
                 e(key="page:/b", title="Gurupournima", text="Gurupournima at Shirali, with bhajans.",
                   metadata={"title": "Gurupournima", "url": "https://example.org/b"})],
                ["2 pages read"])

    sources._crawler = lambda kind: crawl


def install_slow() -> None:
    """install(), with page reads that never finish (for stopping a job)."""
    install()
    from ingest import jobs
    from test_ingest_jobs import WorkerLLM

    class Slow(WorkerLLM):
        async def complete(self, **kw):
            import asyncio
            Path(os.environ["HELPER_STARTED"]).write_text(str(os.getpid()))
            await asyncio.sleep(3600)

    jobs.AnthropicLLM = Slow


def crash() -> None:
    """The helper dies at once (as if killed)."""
    time.sleep(0.1)
    os._exit(5)
