"""End-to-end test on a real local Chainlit server (real models, ~$0.5 of API calls).

Starts `chainlit run app.py` against a throwaway local Postgres (E2E_DATABASE_URL, wiped), signs in with a JWT
minted here from a freshly generated CHAINLIT_AUTH_SECRET (no auth bypass in the app), and drives it with a
socket.io client like the browser does. Checks:
  1. a new chat offers the Detailed / Quick picker and streams its answer (Detailed by default), and the saved
     message equals what was displayed
  2. steps are persisted with the right names and nesting
  3. a PDF and an image upload are used in the answer
  4. Exact Search and Fuzzy Search are unchanged
  5. editing a message truncates the agent's history
  6. a Quick-mode message runs the quick agent, streams and saves normally, and keeps its mode when edited
  7. resuming a thread works (the picker is offered again) and the agent remembers the conversation
  8. /admin's chat viewer renders a new-stack thread
  9. switching a collection off for one person at /admin stops their searches (and on again brings them back)
 10. adding a magazine issue: a new collection, upload (browser -> R2, a local S3 server here) -> job -> review ->
     publish; the issue is then searchable (commands and the agent) for a person with the collection switched on,
     and not for one without. Needs a small PDF that isn't in the repo (public): cut 4 pages of a real issue with
       .venv/bin/python -c "from ingest.pdf import extract_pages; extract_pages('<issue>.pdf',
                            '.cache/ingest-fixture/sunbeam-2024-08-sample.pdf', [6, 11, 31, 40])"
     (pages of the August 2024 issue; check 10 is skipped if the file is missing).

The articles and collections live in a throwaway copy of the local kb_article index (VECTOR_DATABASE_URL), made on
the same server for the run and dropped afterwards.

  .venv/bin/python tests/e2e/run_e2e.py              # add --no-llm for plumbing only, --no-vector for fuzzy-only retrieval
"""
import asyncio
import glob
import hashlib
import html
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import httpx
import socketio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DB = os.environ.get("E2E_DATABASE_URL", "")
if urllib.parse.urlsplit(DB).hostname not in ("localhost", "127.0.0.1"):
    sys.exit("Set E2E_DATABASE_URL to a throwaway *local* Postgres database (it gets wiped).")
# The articles and collections live in a throwaway copy of the local kb_article index, made on the same server for
# this run and dropped afterwards (the server writes to it: schema setup, publishing). The local copy is only read.
VECTOR_DB = os.environ.get("VECTOR_DATABASE_URL", "")
if urllib.parse.urlsplit(VECTOR_DB).hostname not in ("localhost", "127.0.0.1"):
    sys.exit("Set VECTOR_DATABASE_URL to a *local* database holding kb_article (see README).")

SECRET = secrets.token_hex(32)
os.environ["CHAINLIT_AUTH_SECRET"] = SECRET  # for minting the JWT in this process
EMAIL = "e2e.admin@example.com"
READER_A, READER_B = "e2e.reader.a@example.com", "e2e.reader.b@example.com"
# 4 pages of a real issue, cut by hand into .cache/ (git-ignored: the repo is public); the ingest check skips without it.
FIXTURE = ROOT / ".cache" / "ingest-fixture" / "sunbeam-2024-08-sample.pdf"
E2E_KB = {"id": "e2etest", "name": "E2E Test Magazine", "description": "a collection made by the E2E test",
          "source_prefix": "E2ETest"}
OLD_LAYOUT = """CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE kb_article (id text PRIMARY KEY, content text NOT NULL, metadata jsonb NOT NULL,
                         embedding halfvec(3072) NOT NULL)"""
RESULTS = {}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = free_port()
BASE = f"http://127.0.0.1:{PORT}"


def check(name: str, ok: bool, detail: str = ""):
    RESULTS[name] = ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


async def reset_db():
    conn = await asyncpg.connect(DB)
    await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    for path in sorted(glob.glob(str(ROOT / "prisma/migrations/*/migration.sql"))):
        await conn.execute(open(path).read())
    await conn.executemany("""INSERT INTO "AllowedUser"(email, "addedBy") VALUES ($1, 'e2e')""",
                           [(EMAIL,), (READER_A,), (READER_B,)])
    await conn.close()


def clone_vector_db() -> tuple[str, str]:
    """A throwaway database holding the local copy's articles in the pre-change layout (as production has them), with
    the HNSW index. Returns (URL, name)."""
    import psycopg
    from psycopg.conninfo import make_conninfo
    name = f"cgpt_e2e_vec_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(make_conninfo(VECTOR_DB, dbname="postgres"), autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    url = urllib.parse.urlunsplit(urllib.parse.urlsplit(VECTOR_DB)._replace(path="/" + name))
    with psycopg.connect(url, autocommit=True) as dst, psycopg.connect(VECTOR_DB) as src:
        dst.execute(OLD_LAYOUT)
        with src.cursor().copy("COPY (SELECT id, content, metadata, embedding FROM kb_article) TO STDOUT "
                               "(FORMAT BINARY)") as out, \
                dst.cursor().copy("COPY kb_article (id, content, metadata, embedding) FROM STDIN (FORMAT BINARY)") \
                as into:
            for chunk in out:
                into.write(chunk)
        dst.execute("SET maintenance_work_mem = '512MB'")
        dst.execute("CREATE INDEX kb_article_embedding_hnsw ON kb_article USING hnsw (embedding halfvec_ip_ops) "
                    "WITH (m = 32, ef_construction = 200)")
        dst.execute("ANALYZE kb_article")
    return url, name


def drop_vector_db(name: str) -> None:
    import psycopg
    from psycopg.conninfo import make_conninfo
    with psycopg.connect(make_conninfo(VECTOR_DB, dbname="postgres"), autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def start_r2():
    """A local S3 server standing in for Cloudflare R2, with the bucket. Returns (server, endpoint)."""
    import boto3
    from moto.server import ThreadedMotoServer
    port = free_port()
    s3 = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    s3.start()
    endpoint = f"http://127.0.0.1:{port}"
    boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1", aws_access_key_id="e2e",
                 aws_secret_access_key="e2e").create_bucket(Bucket="cgpt-e2e")
    return s3, endpoint


def start_server(log_path: Path, vector_db: str, r2_endpoint: str) -> subprocess.Popen:
    env = {**os.environ, "DATABASE_URL": DB, "VECTOR_DATABASE_URL": vector_db, "CHAINLIT_AUTH_SECRET": SECRET,
           "ADMIN_EMAILS": EMAIL, "OAUTH_GOOGLE_CLIENT_ID": "e2e-client", "OAUTH_GOOGLE_CLIENT_SECRET": "e2e-secret",
           "CHAINLIT_URL": BASE, "PYTHONPATH": str(ROOT / "tests" / "e2e"), "CGPT_E2E_SHIMS": "1",
           "R2_ENDPOINT": r2_endpoint, "R2_BUCKET": "cgpt-e2e", "R2_ACCESS_KEY_ID": "e2e", "R2_SECRET_ACCESS_KEY": "e2e"}
    for k in ("IS_PULL_REQUEST", "RENDER_EXTERNAL_HOSTNAME", "RENDER_EXTERNAL_URL", "BUCKET_NAME", "R2_ACCOUNT_ID"):
        env.pop(k, None)
    if "--no-vector" in sys.argv:  # fuzzy retrieval only, when the embeddings API is unreachable
        env["CGPT_E2E_NO_VECTOR"] = "1"
    log = open(log_path, "w")
    return subprocess.Popen([str(ROOT / ".venv/bin/chainlit"), "run", "app.py", "-h", "--port", str(PORT),
                             "--host", "127.0.0.1"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def jwt(email: str = EMAIL) -> str:
    from chainlit.auth.jwt import create_jwt
    from chainlit.user import User
    return create_jwt(User(identifier=email, metadata={"provider": "google"}))


_expected_index = False


def expected_command(query: str, exact: bool, kb_ids=None) -> str:
    """What an Exact/Fuzzy Search over the original articles returns, computed here from knowledge_base.jsonl (the
    server loads the same articles from its database: tests/test_kb.py shows they're identical)."""
    global _expected_index
    import kb_index
    import search_engine
    from retrievers import load_docs_from_jsonl
    if not _expected_index:
        kb_index.INDEX.set_documents([(str(n), "sunbeam", d) for n, d in
                                      enumerate(load_docs_from_jsonl(str(ROOT / "knowledge_base.jsonl")))])
        _expected_index = True
    return search_engine.search_knowledge_base(query, exact=exact, kb_ids=kb_ids)


def csrf_of(page: str) -> str:
    return re.search(r'name="csrf" value="([0-9a-f]+)"', page).group(1)


def row_of(page: str, email: str) -> str:
    m = re.search(rf'<tr[^>]*data-email="{re.escape(email)}".*?</tr>', page, re.S)
    return m.group(0) if m else ""


NO_COLLECTIONS = "no collection of articles is switched on for you"


class Chat:
    """One browser tab: a socket.io session that records every event the server sends."""

    def __init__(self, token: str, thread_id: str | None = None):
        self.token, self.thread_id = token, thread_id
        self.session_id = str(uuid.uuid4())
        self.events: list[tuple[float, str, object]] = []
        self.sio = socketio.AsyncClient(reconnection=False)
        self.sio.on("*", self._record)

    async def _record(self, event, data=None):
        self.events.append((time.perf_counter(), event, data))
        if event == "first_interaction" and isinstance(data, dict):
            self.thread_id = data.get("thread_id") or self.thread_id

    async def connect(self):
        await self.sio.connect(BASE, socketio_path="/ws/socket.io", transports=["websocket"],
                               headers={"Cookie": f"access_token={self.token}"},
                               auth={"clientType": "webapp", "sessionId": self.session_id,
                                     "threadId": self.thread_id, "userEnv": "{}", "chatProfile": None})
        await self.sio.emit("connection_successful")
        await self.wait_for("task_end", since=0, timeout=60)

    async def wait_for(self, event: str, since: int, timeout: float = 600):
        end = time.time() + timeout
        while time.time() < end:
            if any(e == event for _, e, _ in self.events[since:]):
                return
            await asyncio.sleep(0.2)
        raise TimeoutError(f"no {event} within {timeout}s")

    async def upload(self, path: Path, mime: str) -> str:
        async with httpx.AsyncClient(base_url=BASE, cookies={"access_token": self.token}, timeout=60) as client:
            r = await client.post("/project/file", params={"session_id": self.session_id},
                                  files={"file": (path.name, path.read_bytes(), mime)})
            r.raise_for_status()
            return r.json()["id"]

    async def send(self, text: str, command: str | None = None, files=(), modes: dict | None = None) -> tuple[str, list]:
        refs = [{"id": await self.upload(p, m)} for p, m in files]
        mark = len(self.events)
        mid = str(uuid.uuid4())
        t0 = time.perf_counter()
        await self.sio.emit("client_message", {"message": {
            "id": mid, "threadId": self.thread_id or "", "type": "user_message", "name": EMAIL, "output": text,
            "createdAt": datetime.now(timezone.utc).isoformat(), "command": command, "modes": modes},
            "fileReferences": refs})
        await self.wait_for("task_end", since=mark)
        return mid, [(t - t0, e, d) for t, e, d in self.events[mark:]]

    async def edit(self, mid: str, text: str) -> list:
        mark = len(self.events)
        t0 = time.perf_counter()
        await self.sio.emit("edit_message", {"message": {"id": mid, "threadId": self.thread_id, "type": "user_message",
                                                         "name": EMAIL, "output": text}})
        await self.wait_for("task_end", since=mark)
        return [(t - t0, e, d) for t, e, d in self.events[mark:]]

    async def close(self):
        await self.sio.disconnect()


def answers(events) -> list[dict]:
    """Assistant messages of a turn as the browser would show them: streamed text, then the final content."""
    msgs: dict[str, dict] = {}
    for t, event, data in events:
        if not isinstance(data, dict):
            continue
        if event == "stream_start" and data.get("type") == "assistant_message":
            msgs[data["id"]] = {"streamed": data.get("output") or "", "first_token_s": t, "final": None, "data": data}
        elif event == "stream_token" and data.get("id") in msgs:
            m = msgs[data["id"]]
            m["streamed"] = data["token"] if data.get("isSequence") else m["streamed"] + data["token"]
        elif event in ("new_message", "update_message") and data.get("type") == "assistant_message":
            m = msgs.setdefault(data["id"], {"streamed": None, "first_token_s": None, "final": None, "data": data})
            m["final"], m["data"] = data.get("output"), data
        elif event == "delete_message" and data.get("id") in msgs:
            msgs.pop(data["id"])
    return list(msgs.values())


QUICK = {"answer": "quick"}
ANSWER_PICKER = {"answer": (["detailed", "quick"], "detailed")}


def offered_modes(events) -> dict:
    """The composer pickers the server last offered: {mode id: ([option ids], default option id)}."""
    for _, event, data in reversed(events):
        if event == "set_modes":
            return {m["id"]: ([o["id"] for o in m["options"]], next((o["id"] for o in m["options"] if o["default"]), None))
                    for m in data}
    return {}


def answer_modes(rows) -> list:
    """The answer mode saved with each assistant message."""
    return [json.loads(r["metadata"] or "{}").get("cgpt_answer_mode") for r in rows if r["type"] == "assistant_message"]


async def db_steps(thread_id: str) -> list[dict]:
    await asyncio.sleep(3)  # Chainlit persists steps in background tasks
    conn = await asyncpg.connect(DB)
    rows = await conn.fetch("""SELECT id, name, type::text AS type, "parentId", output, metadata, "isError"
                               FROM "Step" WHERE "threadId" = $1 ORDER BY "startTime", "createdAt" """, thread_id)
    await conn.close()
    return [dict(r) for r in rows]


def make_image(path: Path):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=56)
    draw.text((30, 40), "Rathotsava 2027", fill="black", font=font)
    draw.text((30, 140), "Date: 21 April", fill="darkred", font=font)
    img.save(path)


async def command(token: str, query: str, exact: bool = True) -> str:
    chat = Chat(token)
    await chat.connect()
    _, ev = await chat.send(query, command="Exact Search" if exact else "Fuzzy Search")
    await chat.close()
    return (answers(ev)[-1]["final"] or "") if answers(ev) else ""


async def collections_toggle_check(admin_token: str):
    """A collection switched off at /admin stops that person's searches; switched on again, they're back."""
    async with httpx.AsyncClient(base_url=BASE, cookies={"access_token": admin_token}) as admin:
        csrf = csrf_of((await admin.get("/admin")).text)
        r = await admin.post("/admin/kb/access", data={"csrf": csrf, "email": READER_A, "kb_id": "sunbeam",
                                                       "enabled": "0"})
        page = (await admin.get("/admin")).text
        check("/admin switches Sunbeam off for one person", r.status_code == 303 and "Sunbeam ✗" in row_of(page, READER_A)
              and "Sunbeam ✓" in row_of(page, READER_B))
        off = await command(jwt(READER_A), "Rathotsava")
        await admin.post("/admin/kb/access", data={"csrf": csrf, "email": READER_A, "kb_id": "sunbeam", "enabled": "1"})
        back = await command(jwt(READER_A), "Rathotsava")
    check("switched off: a clear message instead of results", NO_COLLECTIONS in off, off[:100])
    check("switched on again: the same results as before", back == expected_command("Rathotsava", True))


async def wait_for_job(client, job_id: int, done: tuple, timeout: float = 600) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = (await client.get(f"/admin/kb/jobs/{job_id}.json")).json()
        if job["status"] in done or job["status"] == "failed":
            return job
        await asyncio.sleep(2)
    raise TimeoutError(f"job {job_id} not {done} within {timeout}s")


async def ingest_check(admin_token: str):
    """An admin adds an issue to a new collection: upload (browser -> R2) -> job -> review -> publish; it is then
    searchable for a person with that collection switched on, and not for one without."""
    pdf = FIXTURE.read_bytes()
    sha = hashlib.sha256(pdf).hexdigest()
    async with httpx.AsyncClient(base_url=BASE, cookies={"access_token": admin_token}, timeout=60) as admin:
        csrf = csrf_of((await admin.get("/admin/kb")).text)
        r = await admin.post("/admin/kb/create", data={"csrf": csrf, **E2E_KB})
        check("a collection is created at /admin/kb", r.status_code == 303 and E2E_KB["name"] in
              (await admin.get("/admin/kb")).text)
        upload = {"kb_id": E2E_KB["id"], "filename": "Sunbeam August 2024 sample.pdf", "size": len(pdf), "sha256": sha}
        slot = await admin.post("/admin/kb/upload-url", json=upload, headers={"X-CSRF-Token": csrf})
        bad = await admin.post("/admin/kb/upload-url", json={**upload, "size": 300 * 1024 * 1024},
                               headers={"X-CSRF-Token": csrf})
        check("upload slot: a presigned R2 URL; too big is refused", slot.status_code == 200 and bad.status_code == 400
              and "X-Amz-Signature" in slot.json()["url"], bad.text[:80])
        async with httpx.AsyncClient(timeout=120) as browser:  # the browser uploads straight to R2
            put = await browser.put(slot.json()["url"], content=pdf, headers={"Content-Type": "application/pdf"})
        done = await admin.post("/admin/kb/uploaded", json={**upload, "key": slot.json()["key"]},
                                headers={"X-CSRF-Token": csrf})
        check("upload to R2 and the job is queued", put.status_code == 200 and done.status_code == 200, done.text[:120])
        job_id = done.json()["job_id"]
        t0 = time.time()
        job = await wait_for_job(admin, job_id, ("review",))
        check("the job reads, splits and summarises the issue", job["status"] == "review" and job["articles"],
              f"{time.time() - t0:.0f}s, {job['pages']} pages, {len(job['articles'])} articles, "
              f"{len(job['flags'] or [])} flags, ${job['cost_usd']:.3f}, cited as {(job['issue'] or {}).get('source')}")
        review = (await admin.get(f"/admin/kb/jobs/{job_id}")).text
        first = job["articles"][0]
        article = await admin.get(f"/admin/kb/jobs/{job_id}/articles/{first['n']}")
        image = await admin.get(f"/admin/kb/jobs/{job_id}/pages/{first['start_page']}")
        async with httpx.AsyncClient() as browser:
            jpeg = await browser.get(image.headers.get("location", "http://invalid"))
        check("review page: counts, articles, text beside the page image",
              "contents entries matched" in review and html.escape(first["title"]) in review
              and article.status_code == 200 and image.status_code == 302 and jpeg.content[:2] == b"\xff\xd8")
        await admin.post(f"/admin/kb/jobs/{job_id}/publish", data={"csrf": csrf})
        job = await wait_for_job(admin, job_id, ("published",), timeout=180)
        source = (job["issue"] or {}).get("source", "")
        check("publish puts the articles into the knowledge base", job["status"] == "published"
              and job["published_articles"] == sum(1 for a in job["articles"] if not a["excluded"]), job.get("error"))
        for kb_id, on in ((E2E_KB["id"], "1"), ("sunbeam", "0")):  # reader A: only the new collection
            await admin.post("/admin/kb/access", data={"csrf": csrf, "email": READER_A, "kb_id": kb_id, "enabled": on})
        page = (await admin.get("/admin")).text
        check("/admin shows the new collection per person", "E2ETest ✓" in row_of(page, READER_A)
              and "E2ETest ✗" in row_of(page, READER_B))
    a_hits, b_hits = await command(jwt(READER_A), "Ekanta"), await command(jwt(READER_B), "Ekanta")
    check("searchable with the collection switched on", f"(Source: {source})" in a_hits and "Sunbeam-" not in a_hits,
          a_hits[:120].replace("\n", " "))
    check("not searchable without it", source not in b_hits and b_hits == expected_command("Ekanta", True, ["sunbeam"]))
    reader = Chat(jwt(READER_A))
    await reader.connect()
    _, ev = await reader.send("When does H.H. Swamiji observe Ekanta (Mauna) in December 2024?", modes=QUICK)
    reply = (answers(ev)[-1]["final"] or "") if answers(ev) else ""
    await reader.close()
    check("the agent finds and cites the new articles", source in reply, reply[-200:].replace("\n", " "))


async def main():
    await reset_db()
    work = Path(tempfile.mkdtemp(prefix="cgpt-e2e-"))
    log_path = work / "server.log"
    vector_db, vector_name = clone_vector_db()
    r2, r2_endpoint = start_r2()
    server = start_server(log_path, vector_db, r2_endpoint)
    try:
        async with httpx.AsyncClient(base_url=BASE) as client:
            for _ in range(240):
                try:
                    if (await client.get("/")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if server.poll() is not None:
                    raise RuntimeError("server exited:\n" + log_path.read_text()[-3000:])
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError("server didn't start:\n" + log_path.read_text()[-3000:])
        token = jwt()
        if "--no-llm" in sys.argv:  # server, auth, socket and database plumbing only (no model or embedding calls)
            chat = Chat(token)
            await chat.connect()
            check("answer modes offered on chat start", offered_modes(chat.events) == ANSWER_PICKER,
                  str(offered_modes(chat.events)))
            for cmd, query in (("Exact Search", "Rathotsava"), ("Fuzzy Search", "rathotsava AND shirali")):
                _, ev = await chat.send(query, command=cmd, modes=QUICK)  # commands ignore the answer mode
                (m,) = answers(ev)
                expected = expected_command(query, exact=cmd == "Exact Search")
                check(f"{cmd} unchanged", m["final"] == expected and m["data"].get("tags") == ["command_output"])
            rows = await db_steps(chat.thread_id)
            check("command output marked in saved metadata",
                  [r["metadata"] for r in rows if r["type"] == "assistant_message"] == ['{"cgpt_kind": "command_output"}'] * 2,
                  str([r["metadata"] for r in rows if r["type"] == "assistant_message"]))
            async with httpx.AsyncClient(base_url=BASE, cookies={"access_token": token}) as client:
                r = await client.get(f"/admin/chat/{chat.thread_id}")
                check("/admin renders the thread", r.status_code == 200 and "Rathotsava" in r.text, f"HTTP {r.status_code}")
            await chat.close()
            await collections_toggle_check(token)
            return

        # 1-2. a new chat: streaming, saved == displayed, steps
        chat = Chat(token)
        await chat.connect()
        check("answer modes offered on chat start", offered_modes(chat.events) == ANSWER_PICKER,
              str(offered_modes(chat.events)))
        _, ev = await chat.send("When was the Rathotsava revived at Shirali? Answer briefly.")
        (a,) = answers(ev)
        check("answer streamed", bool(a["streamed"]) and a["first_token_s"] is not None,
              f"first token after {a['first_token_s']:.1f}s, turn {ev[-1][0]:.1f}s" if a["first_token_s"] else "")
        check("displayed == final message", a["streamed"] == a["final"])
        thread_id = chat.thread_id
        rows = await db_steps(thread_id)
        saved = [r for r in rows if r["type"] == "assistant_message"]
        check("saved message == displayed", len(saved) == 1 and saved[0]["output"] == a["final"])
        check("no mode sent -> Detailed", answer_modes(rows) == ["detailed"], str(answer_modes(rows)))
        engines = [r for r in rows if r["name"] == "knowledge base search engine"]
        nested = all([c["name"] for c in rows if c["parentId"] == e["id"]] == ["search query generator", "document analysis"]
                     for e in engines)
        check("steps persisted with names and nesting", bool(engines) and nested,
              f"{len(engines)} search(es); " + ", ".join(sorted({r['name'] for r in rows if r['type'] == 'undefined'})))

        # 3. uploads in the same chat
        img = work / "poster.png"
        make_image(img)
        _, ev = await chat.send("Translate the attached note into English. Also, what date is written in the image?",
                                files=[(ROOT / "eval/fixtures/note.pdf", "application/pdf"), (img, "image/png")])
        texts = [m["final"] or "" for m in answers(ev)]
        reply = texts[-1] if texts else ""
        check("PDF upload used", "navaratri" in reply.lower() and "vijayadashami" in reply.lower(), reply[:160].replace("\n", " "))
        check("image upload used", "21 april" in reply.lower() or "april 21" in reply.lower(), reply[-200:].replace("\n", " "))

        # 4. commands
        for cmd, query in (("Exact Search", "Rathotsava"), ("Fuzzy Search", "rathotsava AND shirali")):
            _, ev = await chat.send(query, command=cmd)
            (m,) = answers(ev)
            expected = expected_command(query, exact=cmd == "Exact Search")
            check(f"{cmd} unchanged", m["final"] == expected and m["data"].get("tags") == ["command_output"])

        # 5. edit
        edit_chat = Chat(token)
        await edit_chat.connect()
        first, _ = await edit_chat.send("My name is Ravi. In one sentence, what is Hari Jagara?")
        await edit_chat.edit(first, "In one sentence, what is Hari Jagara?")
        _, ev = await edit_chat.send("What is my name? If I haven't told you my name, reply exactly: unknown")
        reply = (answers(ev)[-1]["final"] or "")
        check("edit truncates history", "ravi" not in reply.lower(), reply[:120].replace("\n", " "))
        await edit_chat.close()

        # 6. Quick answer mode: streams and saves like Detailed; an edited message keeps its mode
        quick_chat = Chat(token)
        await quick_chat.connect()
        first, ev = await quick_chat.send("What is Kojagiri Pournima and how is it celebrated?", modes=QUICK)
        (q,) = answers(ev)
        rows = await db_steps(quick_chat.thread_id)
        saved = [r for r in rows if r["type"] == "assistant_message"]
        check("Quick answer streamed, displayed == saved",
              bool(q["streamed"]) and q["streamed"] == q["final"] and len(saved) == 1 and saved[0]["output"] == q["final"],
              f"turn {ev[-1][0]:.1f}s, {len(q['final'] or '')} chars")
        check("Quick mode ran the quick agent", answer_modes(rows) == ["quick"], str(answer_modes(rows)))
        ev = await quick_chat.edit(first, "What is Hari Jagara?")
        rows = await db_steps(quick_chat.thread_id)
        check("edit keeps the Quick mode", answer_modes(rows) == ["quick"] and any(m["final"] for m in answers(ev)),
              str(answer_modes(rows)))
        await quick_chat.close()

        # 7. resume chat 1 in a new connection
        await chat.close()
        resumed = Chat(token, thread_id=thread_id)
        await resumed.connect()
        check("thread resumed", any(e == "resume_thread" for _, e, _ in resumed.events))
        check("answer modes offered on resume", offered_modes(resumed.events) == ANSWER_PICKER,
              str(offered_modes(resumed.events)))
        _, ev = await resumed.send("In a few words: what festival did I ask about first in this chat?")
        reply = (answers(ev)[-1]["final"] or "")
        check("agent remembers after resume", "rathotsava" in reply.lower(), reply[:120].replace("\n", " "))
        await resumed.close()

        # 8. admin chat viewer
        async with httpx.AsyncClient(base_url=BASE, cookies={"access_token": token}) as client:
            r = await client.get(f"/admin/chat/{thread_id}")
            snippet = html.escape(a["final"].strip()[:60])
            check("/admin renders the new-stack thread", r.status_code == 200 and snippet in r.text, f"HTTP {r.status_code}")

        # 9. collections: switched off for one person at /admin
        await collections_toggle_check(token)

        # 10. adding a magazine issue, and who can search it
        if FIXTURE.exists():
            await ingest_check(token)
        else:
            print(f"SKIP  adding an issue ({FIXTURE.relative_to(ROOT)} isn't there; see this file's docstring)")
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        r2.stop()
        drop_vector_db(vector_name)
        print(f"server log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
    # Here, not in main(): --no-llm returns early, and its result must count too.
    failed = [k for k, ok in RESULTS.items() if not ok]
    print(json.dumps({"passed": len(RESULTS) - len(failed), "failed": failed}))
    sys.exit(1 if failed else 0)
