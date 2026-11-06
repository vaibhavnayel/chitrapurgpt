"""Heavy background work in a short-lived helper process, so the memory it needs is given back when it ends.

The web process holds every collection's search index (about 0.7 GB). Reading an uploaded issue, re-reading a
website, or embedding a website's pages needs a few hundred MB more while it runs. Python keeps memory it has used,
so in the web process that memory stayed taken until the next restart, and a run of uploads kept adding to it. So
the web process starts `python -m ingest.isolated <task> ...` and waits; the helper does the work, writes to the
database itself, and exits. Only what has to happen in the web process (refreshing its search index) stays there.

  job <job id> <worker name> <pages read at once, 0: default>   read, split, summarise an issue (jobs.Worker._run)
  check <sync id> <source id> <auto 0/1>                        re-read a website, store what would change
  write <sync id>                                               a sync's summaries and embeddings, then its rows
  ping                                                          start up and stop (checked when the worker starts)

The last argument is a file the helper writes its result to: a JSON object, or {"error": ..., "kind": ...} if the
task failed. The helper exits by itself if the web process goes away.

INGEST_ISOLATE=0 runs all of this in the web process instead (as before), and so does a web process where the ping
fails (fall_back()). INGEST_CHILD_SETUP=module:function is called in the helper before the task (the tests use it to
put in fakes).
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import logging
import os
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
STOP_GRACE_S = 10


_fallen_back = False


def enabled() -> bool:
    return os.environ.get("INGEST_ISOLATE", "1") != "0" and not _fallen_back


def fall_back() -> None:
    """Helper processes don't work here: do the work in the web process from now on."""
    global _fallen_back
    _fallen_back = True


async def check() -> bool:
    """Start a helper once (the worker does this when it starts). If that fails, fall back to the web process."""
    if not enabled():
        return False
    try:
        result = await run("ping")
    except Exception:  # noqa: BLE001
        logger.exception("Helper processes don't work here: reading issues and websites in the web process instead")
        fall_back()
        return False
    logger.info("Helper processes work (a helper takes %s MB to start)", result.get("peak_mb", "?"))
    return True


class ChildError(RuntimeError):
    """The helper's task failed (`kind`: the exception's class name there; `job_error`: a jobs.JobError, whose
    message is meant for the admin as is), or the helper itself stopped."""

    def __init__(self, message: str, kind: str = "HelperStopped", job_error: bool = False):
        super().__init__(message)
        self.message, self.kind, self.job_error = message, kind, job_error


# ---------- in the web process ----------

async def run(task: str, *args) -> dict:
    """Run a task in a helper process and return its result. Raises ChildError if it failed. Cancelling this
    stops the helper (SIGTERM, then SIGKILL after STOP_GRACE_S)."""
    t = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cgpt-helper-") as tmp:
        out = Path(tmp) / "result.json"
        proc = await asyncio.create_subprocess_exec(sys.executable, "-m", "ingest.isolated", task,
                                                    *(str(a) for a in args), str(out), cwd=str(ROOT))
        logger.info("Helper process %d started: %s %s", proc.pid, task, " ".join(str(a) for a in args))
        try:
            code = await proc.wait()
        finally:
            if proc.returncode is None:
                await _stop(proc)
        try:
            result = json.loads(out.read_text())
        except (OSError, ValueError):
            result = None
    logger.info("Helper process %d (%s) ended with code %s after %.0fs; it used up to %s MB", proc.pid, task, code,
                time.monotonic() - t, (result or {}).get("peak_mb", "?"))
    if result is None or code != 0 or "error" in result:
        if result and "error" in result:
            raise ChildError(result["error"], result.get("kind") or "Error", bool(result.get("job_error")))
        why = (" (it was killed, probably for lack of memory)" if code == -9 else "")
        raise ChildError(f"the helper process stopped unexpectedly, exit code {code}{why}")
    return result


async def _stop(proc) -> None:
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), STOP_GRACE_S)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
    logger.info("Helper process %d stopped", proc.pid)


# ---------- in the helper ----------

def _watch_parent() -> None:
    """Exit if the web process that started us is gone (e.g. it crashed): nobody would use the result."""
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(5)
            if os.getppid() != parent:
                logger.warning("The web process went away; stopping")
                os._exit(3)

    threading.Thread(target=watch, name="watch-parent", daemon=True).start()


async def _job(job_id: int, worker_name: str, concurrency: int) -> dict:
    from ingest import jobs
    job = await jobs.get_job(job_id)
    if job is None:
        raise jobs.JobError("This upload no longer exists.")
    worker = jobs.Worker(concurrency=concurrency or None, isolate=False)
    worker.name = worker_name   # the web process's worker owns the job (and keeps its heartbeat going)
    await worker._run(job)
    return {}


async def _task(task: str, args: list[str]) -> dict:
    import kb
    from ingest import sources
    try:
        if task == "job":
            return await _job(int(args[0]), args[1], int(args[2]))
        if task == "check":
            return await sources.check_site(int(args[0]), int(args[1]), auto=args[2] == "1")
        if task == "write":
            return await sources.write_sync(int(args[0]))
        if task == "ping":
            from ingest import jobs, website  # noqa: F401  (everything the tasks import loads)
            return {}
        raise ValueError(f"unknown task {task!r}")
    finally:
        await kb.close_pool()


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format=f"%(asctime)s - %(levelname)s - %(name)s [helper {os.getpid()}] - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    task, args, out = argv[0], argv[1:-1], Path(argv[-1])
    _watch_parent()
    try:
        setup = os.environ.get("INGEST_CHILD_SETUP")
        if setup:
            module, _, function = setup.partition(":")
            getattr(importlib.import_module(module), function)()
        result, code = asyncio.run(_task(task, args)), 0
    except Exception as e:  # noqa: BLE001  (reported to the web process, which records it as before)
        from ingest.jobs import JobError
        if not isinstance(e, JobError):
            logger.exception("Helper task %s %s failed", task, " ".join(args))
        result, code = {"error": str(e), "kind": type(e).__name__, "job_error": isinstance(e, JobError)}, 1
    result["peak_mb"] = peak_mb()
    out.write_text(json.dumps(result))
    return code


def peak_mb() -> int:
    """This process's own peak memory. (Not ru_maxrss: across fork and exec it also counts the web process's.)"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
