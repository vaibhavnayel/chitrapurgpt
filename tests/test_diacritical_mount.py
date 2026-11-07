"""Diacritical served at /diacritical inside ChitrapurGPT (diacritical_mount.py, diacritical/).

Diacritical's database is a throwaway Postgres made for the run in a temp dir (Homebrew postgresql@16 or @15
binaries: initdb -A trust, a free port); the tests skip without them.
  * In-process: Chainlit's real FastAPI app behind the mount (httpx ASGITransport): the redirect, Diacritical's own
    login and session cookie, its pages, API and static files under the prefix, every link on its pages staying
    under /diacritical, the 503 placeholder (not configured, ChitrapurGPT's own DATABASE_URL, a failed load) while
    Chainlit's routes keep answering, and Chainlit not waiting while Diacritical loads.
  * Real server: `chainlit run app.py` (as on Render), configured and unconfigured. ChitrapurGPT's own database is a
    throwaway one made for the run on the local pgvector server (VECTOR_DATABASE_URL, as for the rest of the suite)
    and dropped afterwards; skipped if that isn't set to a local server.
Run: pytest tests/test_diacritical_mount.py   (or python tests/test_diacritical_mount.py)
"""
import asyncio
import functools
import html.parser
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import diacritical_mount  # noqa: E402

USER, PASSWORD, SECRET = "dad", "test-password", "test-secret-key"
TABLES = ("diacritic_mappings", "feedback")


# ---------- a throwaway Postgres ----------

def pg_bin() -> Path | None:
    for version in ("16", "15"):
        b = Path(f"/opt/homebrew/opt/postgresql@{version}/bin")
        if (b / "initdb").exists() and (b / "pg_ctl").exists():
            return b
    return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def pg():
    """Base URL of a throwaway Postgres server with the databases `diacritical`, `diacritical_server`,
    `diacritical_unset` and `chitrapurgpt` (standing in for ChitrapurGPT's own database)."""
    b = pg_bin()
    if b is None:
        pytest.skip("Homebrew postgresql@16/@15 binaries not found")
    tmp = tempfile.mkdtemp(prefix="diacritical-pg-")
    data, port = os.path.join(tmp, "data"), free_port()
    env = {**os.environ, "LC_ALL": "en_US.UTF-8"}  # else postmaster dies: "became multithreaded during startup"
    subprocess.run([b / "initdb", "-A", "trust", "-U", "postgres", "-D", data], env=env, check=True,
                   capture_output=True)
    subprocess.run([b / "pg_ctl", "-D", data, "-l", os.path.join(tmp, "pg.log"), "-w", "-o",
                    f"-p {port} -k {tmp} -c listen_addresses=127.0.0.1", "start"], env=env, check=True,
                   capture_output=True)
    base = f"postgresql://postgres@127.0.0.1:{port}"
    try:
        for name in ("diacritical", "diacritical_server", "diacritical_unset", "chitrapurgpt"):
            subprocess.run([b / "createdb", "-h", "127.0.0.1", "-p", str(port), "-U", "postgres", name], env=env,
                           check=True, capture_output=True)
        yield base
    finally:
        subprocess.run([b / "pg_ctl", "-D", data, "-m", "immediate", "stop"], env=env, capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


def tables(url: str) -> set[str]:
    import psycopg
    with psycopg.connect(url) as conn:
        return {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}


# ---------- links on the rendered pages ----------

class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src", "action", "formaction") and value is not None:
                self.urls.append(value)


# url: '...' / url: `...` / url: "..." and fetch('...') in inline and static JavaScript
JS_URL = re.compile(r"""(?:\burl\s*:\s*|\bfetch\(\s*)(`[^`]*`|'[^']*'|"[^"]*")""")


def links(page: str) -> list[str]:
    parser = Links()
    parser.feed(page)
    return parser.urls + [m.group(1)[1:-1] for m in JS_URL.finditer(page)]


def outside_prefix(urls) -> list[str]:
    """The URLs that would leave /diacritical (external sites, fragments and the script-root prefix are fine)."""
    bad = []
    for u in urls:
        if re.match(r"^(https?:)?//", u) or u.startswith(("#", "data:", "mailto:", "javascript:")):
            continue  # another site (CDNs: Bootstrap, jQuery, DataTables, fonts), or not a request
        if u.startswith("/diacritical/") or u.startswith("${window.DIACRITICAL_ROOT"):
            continue
        bad.append(u)
    return bad


# ---------- in-process: Chainlit's real app behind the mount ----------

@pytest.fixture(scope="module")
def chainlit_app():
    from chainlit.server import app
    return app


def client(app, **kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver", **kwargs)


@pytest.fixture(scope="module")
def mounted(pg, chainlit_app):
    """The mount over Chainlit's app, with Diacritical configured. DATABASE_URL (ChitrapurGPT's) is a different
    database, which Diacritical must leave alone."""
    environ = {"DATABASE_URL": f"{pg}/chitrapurgpt", "DIACRITICAL_DATABASE_URL": f"postgres://{pg[13:]}/diacritical",
               "DIACRITICAL_SECRET_KEY": SECRET, "DIACRITICAL_USERNAME": USER, "DIACRITICAL_PASSWORD": PASSWORD}
    return diacritical_mount.DiacriticalMount(chainlit_app, environ=environ)


def run(coro):
    return asyncio.run(coro)


def test_bare_prefix_redirects(mounted):
    async def go():
        async with client(mounted) as c:
            r = await c.get("/diacritical")
            assert r.status_code == 307 and r.headers["location"] == "/diacritical/"
            r = await c.get("/diacritical?a=1")
            assert r.headers["location"] == "/diacritical/?a=1"
            r = await c.get("/diacritical/")  # Flask's index -> its translator page
            assert r.status_code == 302 and r.headers["location"] == "/diacritical/translator"
            r = await c.get("/diacriticalx")  # not under the prefix: Chainlit's (its catch-all UI route)
            assert r.status_code == 200 and "<title>" in r.text and "Diacritical" not in r.text
    run(go())


def test_unauthenticated_pages_go_to_prefixed_login(mounted):
    async def go():
        async with client(mounted) as c:
            for path in ("/diacritical/mappings", "/diacritical/api/mappings", "/diacritical/api/feedback"):
                r = await c.get(path)
                assert r.status_code == 302 and r.headers["location"] == "/diacritical/login", path
            r = await c.get("/diacritical/login")
            assert r.status_code == 200 and 'action="/diacritical/login"' in r.text
    run(go())


def test_login_session_cookie_and_api(mounted, pg):
    async def go():
        async with client(mounted) as c:
            r = await c.post("/diacritical/login", data={"username": USER, "password": "wrong"})
            assert r.status_code == 200 and "Invalid credentials" in r.text
            assert (await c.get("/diacritical/mappings")).status_code == 302

            r = await c.post("/diacritical/login", data={"username": USER, "password": PASSWORD})
            assert r.status_code == 302 and r.headers["location"] == "/diacritical/translator"
            cookie = r.headers["set-cookie"]
            assert cookie.startswith("diacritical_session=") and "Path=/diacritical" in cookie, cookie
            assert "HttpOnly" in cookie

            r = await c.get("/diacritical/mappings")
            assert r.status_code == 200 and "mappings" in r.text.lower()
            r = await c.post("/diacritical/api/mappings", json={"plain_text": "shirali", "diacritic_text": "śirāli"})
            assert r.status_code == 201, r.text
            r = await c.get("/diacritical/api/mappings")
            assert r.status_code == 200 and [m["plain_text"] for m in r.json()] == ["shirali"]
            r = await c.get("/diacritical/api/mappings/download")
            assert r.status_code == 200 and r.text == "shirali,śirāli\n"

            r = await c.get("/diacritical/logout")
            assert r.status_code == 302 and r.headers["location"] == "/diacritical/translator"
            assert (await c.get("/diacritical/mappings")).status_code == 302
    run(go())
    assert set(TABLES) <= tables(f"{pg}/diacritical")
    assert not set(TABLES) & tables(f"{pg}/chitrapurgpt")  # never in ChitrapurGPT's database


def test_translator_json_api_and_static(mounted):
    async def go():
        async with client(mounted) as c:
            r = await c.get("/diacritical/translator")
            assert r.status_code == 200 and "Diacritical Translator" in r.text
            assert 'window.DIACRITICAL_ROOT = "/diacritical";' in r.text
            await c.post("/diacritical/login", data={"username": USER, "password": PASSWORD})
            r = await c.post("/diacritical/api/mappings", json={"plain_text": "shirali", "diacritic_text": "śirāli"})
            assert r.status_code in (201, 400)  # 400: already added by the login test
            r = await c.post("/diacritical/translate", json={"text": "Going to Shirali"})
            assert r.status_code == 200 and r.json() == {"result": "Going to Śirāli"}, r.text
            r = await c.post("/diacritical/api/feedback", json={"message": "works under the prefix"})
            assert r.status_code == 201
            r = await c.get("/diacritical/static/js/translator.js")
            assert r.status_code == 200 and "javascript" in r.headers["content-type"]
            assert "${window.DIACRITICAL_ROOT || ''}/translate" in r.text
            r = await c.get("/diacritical/static/css/base.css")
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/css")
    run(go())


def test_every_link_stays_under_prefix(mounted):
    async def go():
        async with client(mounted) as c:
            pages = {"login": (await c.get("/diacritical/login")).text}
            await c.post("/diacritical/login", data={"username": USER, "password": PASSWORD})
            pages["translator"] = (await c.get("/diacritical/translator")).text
            pages["mappings"] = (await c.get("/diacritical/mappings")).text
            found = {}
            for name, page in pages.items():
                urls = links(page)
                assert urls, name
                assert outside_prefix(urls) == [], (name, outside_prefix(urls))
                found[name] = urls
                # every link and asset on the page is served (not 404) under the prefix
                for u in urls:
                    if u.startswith("/diacritical/") and not u.endswith("/logout"):
                        r = await c.get(u)
                        assert r.status_code in (200, 302), (name, u, r.status_code)
            for js in ("translator.js", "mappings.js"):
                text = (await c.get(f"/diacritical/static/js/{js}")).text
                urls = links(text)
                assert urls and outside_prefix(urls) == [], (js, urls)
            # the download links and the API calls were the hard-coded ones
            assert "/diacritical/api/mappings/download" in found["mappings"]
            assert "/diacritical/api/feedback/download" in found["mappings"]
            assert "/diacritical/api/feedback" in found["translator"]
    run(go())


def test_chainlit_routes_unaffected(mounted, chainlit_app):
    async def go():
        async with client(mounted) as via_mount, client(chainlit_app) as direct:
            for path in ("/", "/auth/config", "/project/settings?language=en-US"):
                a, b = await via_mount.get(path), await direct.get(path)
                assert (a.status_code, a.headers.get("content-type")) == (b.status_code, b.headers.get("content-type"))
                assert a.content == b.content, path
            assert (await via_mount.get("/auth/config")).json().keys() >= {"requireLogin", "oauthProviders"}
    run(go())


def test_not_configured_answers_503_and_chainlit_still_works(pg, chainlit_app, caplog):
    for environ, reason in (
            ({"DATABASE_URL": f"{pg}/chitrapurgpt"},  # no DIACRITICAL_DATABASE_URL: never falls back
             "DIACRITICAL_DATABASE_URL is not set"),
            ({"DATABASE_URL": f"{pg}/chitrapurgpt",  # pointed at ChitrapurGPT's own database
              "DIACRITICAL_DATABASE_URL": f"postgres://{pg[13:]}/chitrapurgpt"},
             "DIACRITICAL_DATABASE_URL is ChitrapurGPT's own DATABASE_URL")):
        mount = diacritical_mount.DiacriticalMount(chainlit_app, environ=environ)
        caplog.clear()

        async def go():
            async with client(mount) as c:
                for path in ("/diacritical/", "/diacritical/translator", "/diacritical/static/css/base.css"):
                    r = await c.get(path)
                    assert r.status_code == 503 and r.text == "Diacritical is not configured", (path, r.text)
                assert (await c.get("/diacritical")).status_code == 307
                r = await c.get("/auth/config")
                assert r.status_code == 200 and "requireLogin" in r.json()
                assert (await c.get("/")).status_code == 200
        with caplog.at_level("ERROR", logger="diacritical_mount"):
            run(go())
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1 and reason in errors[0], errors  # logged once, with the reason
    assert not set(TABLES) & tables(f"{pg}/chitrapurgpt")


def test_failed_load_is_a_503_placeholder(chainlit_app):
    calls = []

    def broken(environ):
        calls.append(1)
        raise ImportError("simulated broken import")

    mount = diacritical_mount.DiacriticalMount(chainlit_app, environ={}, loader=broken)

    async def go():
        async with client(mount) as c:
            for _ in range(2):
                r = await c.get("/diacritical/translator")
                assert r.status_code == 503 and r.text == "Diacritical is temporarily unavailable"
            assert (await c.get("/auth/config")).status_code == 200
    run(go())
    assert calls == [1]  # tried once, not on every request


async def _ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": scope["root_path"].encode()})


def test_chainlit_does_not_wait_while_diacritical_loads(chainlit_app):
    calls = []

    def slow(environ):  # e.g. its database still resuming: init_db retries for up to a minute
        calls.append(1)
        time.sleep(1.5)
        return _ok_app

    mount = diacritical_mount.DiacriticalMount(chainlit_app, environ={}, loader=slow)

    async def go():
        async with client(mount) as c:
            t = time.monotonic()
            first = asyncio.create_task(c.get("/diacritical/translator"))
            second = asyncio.create_task(c.get("/diacritical/login"))
            await asyncio.sleep(0.1)
            r = await c.get("/auth/config")
            assert r.status_code == 200 and time.monotonic() - t < 1.0 and not first.done()
            a, b = await first, await second
            assert a.status_code == b.status_code == 200 and a.text == "/diacritical"
            assert time.monotonic() - t >= 1.4
    run(go())
    assert calls == [1]  # concurrent first requests share one load


def test_root_path_and_websocket(chainlit_app):
    mount = diacritical_mount.DiacriticalMount(chainlit_app, environ={}, loader=lambda environ: _ok_app)
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    async def go():
        await mount({"type": "http", "method": "GET", "root_path": "/x", "path": "/x/diacritical/translator",
                     "query_string": b"", "headers": []}, receive, send)
        await mount({"type": "websocket", "root_path": "", "path": "/diacritical/ws", "headers": []}, receive, send)
    run(go())
    assert sent[1]["body"] == b"/x/diacritical"
    assert sent[2] == {"type": "websocket.close", "code": 1000}


def test_install_loads_nothing():
    """Installing the mount (at ChitrapurGPT's startup) imports nothing of Diacritical's, Flask's or a2wsgi's."""
    code = ("import sys, diacritical_mount; from fastapi import FastAPI; app = FastAPI(); "
            "assert diacritical_mount.install(app, environ={}) is True; "
            "assert app.user_middleware[0].cls is diacritical_mount.DiacriticalMount; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('flask', 'flask_sqlalchemy', 'flask_bootstrap', 'sqlalchemy', 'a2wsgi', 'docx', 'psycopg2') "
            "or m.startswith('diacritical.')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


# ---------- the real server: chainlit run app.py ----------

VECTOR_DB = os.environ.get("VECTOR_DATABASE_URL", "")
OLD_LAYOUT = """CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE kb_article (id text PRIMARY KEY, content text NOT NULL, metadata jsonb NOT NULL,
                         embedding halfvec(3072) NOT NULL)"""


@pytest.fixture(scope="module")
def chitrapurgpt_db():
    """A throwaway ChitrapurGPT database (no articles) on the local pgvector server; dropped afterwards."""
    if urllib.parse.urlsplit(VECTOR_DB).hostname not in ("localhost", "127.0.0.1"):
        pytest.skip("VECTOR_DATABASE_URL isn't a local pgvector server")
    import psycopg
    from psycopg.conninfo import make_conninfo
    name = f"cgpt_diacritical_test_{uuid.uuid4().hex[:8]}"
    admin = make_conninfo(VECTOR_DB, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    url = urllib.parse.urlunsplit(urllib.parse.urlsplit(VECTOR_DB)._replace(path="/" + name))
    try:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(OLD_LAYOUT)
        yield url
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@functools.cache
def chainlit_bin() -> str:
    candidate = Path(sys.executable).parent / "chainlit"
    return str(candidate if candidate.exists() else ROOT / ".venv" / "bin" / "chainlit")


class Server:
    def __init__(self, db: str, diacritical: dict):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        env = {k: v for k, v in os.environ.items() if not k.startswith("DIACRITICAL_")}
        for k in ("OPENAI_API_KEY", "IS_PULL_REQUEST", "RENDER_EXTERNAL_HOSTNAME", "RENDER_EXTERNAL_URL",
                  "VECTOR_DATABASE_URL"):
            env.pop(k, None)
        env.update({"DATABASE_URL": db, "CHAINLIT_AUTH_SECRET": uuid.uuid4().hex * 2, "INGEST_WORKER": "0",
                    "OAUTH_GOOGLE_CLIENT_ID": "test-client", "OAUTH_GOOGLE_CLIENT_SECRET": "test-secret",
                    "CHAINLIT_URL": self.base, "LANGCHAIN_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
                    **diacritical})
        self.proc = subprocess.Popen([chainlit_bin(), "run", "app.py", "-h", "--port", str(self.port),
                                      "--host", "127.0.0.1"], cwd=ROOT, env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT)

    def wait(self):
        for _ in range(240):
            try:
                if httpx.get(self.base + "/auth/config", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(0.5)
        raise RuntimeError("server didn't start:\n" + self.output()[-3000:])

    def output(self) -> str:
        self.log.flush()
        return Path(self.log.name).read_text()

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        os.unlink(self.log.name)


def test_real_server(pg, chitrapurgpt_db):
    server = Server(chitrapurgpt_db, {"DIACRITICAL_DATABASE_URL": f"{pg}/diacritical_server",
                                      "DIACRITICAL_SECRET_KEY": SECRET, "DIACRITICAL_USERNAME": USER,
                                      "DIACRITICAL_PASSWORD": PASSWORD}).wait()
    try:
        assert "Diacritical loaded" not in server.output()  # nothing loaded until the first request
        with httpx.Client(base_url=server.base) as c:
            r = c.get("/")
            assert r.status_code == 200 and "<title>" in r.text and "Diacritical" not in r.text
            r = c.get("/auth/config")
            assert r.status_code == 200 and r.json()["requireLogin"] is True  # Chainlit's sign-in is on...
            assert c.get("/project/settings?language=en-US").status_code == 401  # ...and gates its own routes
            r = c.get("/diacritical")
            assert r.status_code == 307 and r.headers["location"] == "/diacritical/"
            r = c.get("/diacritical/translator")  # ...but not Diacritical's (no Chainlit token here)
            assert r.status_code == 200 and "Diacritical Translator" in r.text
            r = c.get("/diacritical/mappings")
            assert r.status_code == 302 and r.headers["location"] == "/diacritical/login"
            r = c.post("/diacritical/login", data={"username": USER, "password": PASSWORD})
            assert r.status_code == 302 and "Path=/diacritical" in r.headers["set-cookie"]
            assert c.get("/diacritical/api/mappings").json() == []
            assert c.get("/diacritical/static/js/mappings.js").status_code == 200
        assert "Diacritical loaded" in server.output()
        assert set(TABLES) <= tables(f"{pg}/diacritical_server")
        assert not set(TABLES) & tables(chitrapurgpt_db)
    finally:
        server.stop()


def test_real_server_without_diacritical_config(pg, chitrapurgpt_db):
    server = Server(chitrapurgpt_db, {}).wait()
    try:
        with httpx.Client(base_url=server.base) as c:
            r = c.get("/diacritical/translator")
            assert r.status_code == 503 and r.text == "Diacritical is not configured"
            assert c.get("/auth/config").status_code == 200 and c.get("/").status_code == 200
        out = server.output()
        assert "DIACRITICAL_DATABASE_URL is not set" in out and "ERROR - diacritical_mount" in out
        assert not set(TABLES) & tables(chitrapurgpt_db)
    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
