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

  .venv/bin/python tests/e2e/run_e2e.py              # add --no-llm for plumbing only, --no-vector for fuzzy-only retrieval
"""
import asyncio
import glob
import html
import json
import os
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
# Vector search goes to a local copy of the kb_article index (the server prefers VECTOR_DATABASE_URL over the chat
# database it gets as DATABASE_URL). Read-only here.
VECTOR_DB = os.environ.get("VECTOR_DATABASE_URL", "")
if "--no-vector" not in sys.argv and urllib.parse.urlsplit(VECTOR_DB).hostname not in ("localhost", "127.0.0.1"):
    sys.exit("Set VECTOR_DATABASE_URL to a *local* database holding kb_article (see README), or pass --no-vector.")

SECRET = secrets.token_hex(32)
os.environ["CHAINLIT_AUTH_SECRET"] = SECRET  # for minting the JWT in this process
EMAIL = "e2e.admin@example.com"
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
    await conn.execute("""INSERT INTO "AllowedUser"(email, "addedBy") VALUES ($1, 'e2e')""", EMAIL)
    await conn.close()


def start_server(log_path: Path) -> subprocess.Popen:
    env = {**os.environ, "DATABASE_URL": DB, "CHAINLIT_AUTH_SECRET": SECRET, "ADMIN_EMAILS": EMAIL,
           "OAUTH_GOOGLE_CLIENT_ID": "e2e-client", "OAUTH_GOOGLE_CLIENT_SECRET": "e2e-secret",
           "CHAINLIT_URL": BASE, "PYTHONPATH": str(ROOT / "tests" / "e2e"), "CGPT_E2E_SHIMS": "1"}
    for k in ("IS_PULL_REQUEST", "RENDER_EXTERNAL_HOSTNAME", "RENDER_EXTERNAL_URL", "BUCKET_NAME"):
        env.pop(k, None)
    if "--no-vector" in sys.argv:  # fuzzy retrieval only, when the embeddings API is unreachable
        env["CGPT_E2E_NO_VECTOR"] = "1"
    log = open(log_path, "w")
    return subprocess.Popen([str(ROOT / ".venv/bin/chainlit"), "run", "app.py", "-h", "--port", str(PORT),
                             "--host", "127.0.0.1"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def jwt() -> str:
    from chainlit.auth.jwt import create_jwt
    from chainlit.user import User
    return create_jwt(User(identifier=EMAIL, metadata={"provider": "google"}))


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


async def main():
    await reset_db()
    work = Path(tempfile.mkdtemp(prefix="cgpt-e2e-"))
    log_path = work / "server.log"
    server = start_server(log_path)
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
            import search_engine
            for command, query in (("Exact Search", "Rathotsava"), ("Fuzzy Search", "rathotsava AND shirali")):
                _, ev = await chat.send(query, command=command, modes=QUICK)  # commands ignore the answer mode
                (m,) = answers(ev)
                expected = search_engine.search_knowledge_base(query, exact=command == "Exact Search")
                check(f"{command} unchanged", m["final"] == expected and m["data"].get("tags") == ["command_output"])
            rows = await db_steps(chat.thread_id)
            check("command output marked in saved metadata",
                  [r["metadata"] for r in rows if r["type"] == "assistant_message"] == ['{"cgpt_kind": "command_output"}'] * 2,
                  str([r["metadata"] for r in rows if r["type"] == "assistant_message"]))
            async with httpx.AsyncClient(base_url=BASE, cookies={"access_token": token}) as client:
                r = await client.get(f"/admin/chat/{chat.thread_id}")
                check("/admin renders the thread", r.status_code == 200 and "Rathotsava" in r.text, f"HTTP {r.status_code}")
            await chat.close()
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
        import search_engine
        for command, query in (("Exact Search", "Rathotsava"), ("Fuzzy Search", "rathotsava AND shirali")):
            _, ev = await chat.send(query, command=command)
            (m,) = answers(ev)
            expected = search_engine.search_knowledge_base(query, exact=command == "Exact Search")
            check(f"{command} unchanged", m["final"] == expected and m["data"].get("tags") == ["command_output"])

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
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        print(f"server log: {log_path}")
    failed = [k for k, ok in RESULTS.items() if not ok]
    print(json.dumps({"passed": len(RESULTS) - len(failed), "failed": failed}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
