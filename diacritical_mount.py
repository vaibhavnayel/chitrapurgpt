"""Diacritical (the Flask app in diacritical/) served at /diacritical, in the same process as Chainlit.

An ASGI middleware in front of Chainlit's FastAPI app (installed from app.py, like preview_oauth.py): a request for
/diacritical/... goes to the Flask app, wrapped as ASGI by a2wsgi with root_path=/diacritical, so Flask sees
SCRIPT_NAME=/diacritical and url_for builds prefixed URLs. Everything else goes to Chainlit untouched. Requests are
answered here, before Chainlit's routing, so Chainlit's sign-in never gates /diacritical; Diacritical has its own
login (DIACRITICAL_USERNAME / DIACRITICAL_PASSWORD) and session cookie (`diacritical_session`, path /diacritical).
/diacritical without the slash redirects to /diacritical/.

Configuration, only from these env vars (never DATABASE_URL, which is ChitrapurGPT's own database):
  DIACRITICAL_DATABASE_URL  its Postgres database (required)
  DIACRITICAL_SECRET_KEY    signs its session cookie
  DIACRITICAL_USERNAME, DIACRITICAL_PASSWORD  its login

Diacritical never holds up or breaks ChitrapurGPT: nothing of it is imported at startup. The first /diacritical
request loads it in a worker thread (import, then the table check, which retries while its database is unreachable),
and requests that arrive meanwhile wait for that. If DIACRITICAL_DATABASE_URL is missing (or is ChitrapurGPT's
DATABASE_URL), /diacritical/... answers 503 "Diacritical is not configured"; if loading fails, 503 "Diacritical is
temporarily unavailable". Either way the reason is logged once and the chat keeps working. The outcome holds until
the process restarts (on Render an env change redeploys).
"""
import asyncio
import logging
import os
import urllib.parse

logger = logging.getLogger("diacritical_mount")

# The package itself (its __init__ loads nothing), imported now: Chainlit takes the app's folder off sys.path again
# once app.py has loaded, so `diacritical.app` can only be imported later, on the first request, through it.
try:
    import diacritical  # noqa: F401
except Exception:  # noqa: BLE001  (never stops ChitrapurGPT from starting; /diacritical then answers 503)
    logger.exception("Couldn't import the diacritical package")

PREFIX = "/diacritical"
WORKERS = 4  # threads running Flask requests (and so at most this many database connections in use at once)

NOT_CONFIGURED = "Diacritical is not configured"
UNAVAILABLE = "Diacritical is temporarily unavailable"


class NotConfigured(Exception):
    pass


def _same_database(a: str, b: str) -> bool:
    """True if two Postgres URLs name the same database (user, database name, and host, internal or external)."""
    def key(url):
        u = urllib.parse.urlsplit(url.strip())
        return u.username, u.path.strip("/"), (u.hostname or "").split(".")[0]
    try:
        return key(a) == key(b)
    except ValueError:
        return a.strip() == b.strip()


def settings(environ=os.environ) -> dict:
    """The Flask app's settings, from the DIACRITICAL_* env vars only."""
    url = (environ.get("DIACRITICAL_DATABASE_URL") or "").strip()
    if not url:
        raise NotConfigured("DIACRITICAL_DATABASE_URL is not set")
    if environ.get("DATABASE_URL") and _same_database(url, environ["DATABASE_URL"]):
        raise NotConfigured("DIACRITICAL_DATABASE_URL is ChitrapurGPT's own DATABASE_URL; refusing to use it")
    missing = [k for k in ("DIACRITICAL_SECRET_KEY", "DIACRITICAL_USERNAME", "DIACRITICAL_PASSWORD")
               if not environ.get(k)]
    if missing:
        logger.error("Diacritical: %s not set; signing in to Diacritical won't work", ", ".join(missing))
    return {"database_url": url, "secret_key": environ.get("DIACRITICAL_SECRET_KEY"),
            "username": environ.get("DIACRITICAL_USERNAME"), "password": environ.get("DIACRITICAL_PASSWORD")}


def load_app(environ=os.environ, **configure_kwargs):
    """Import and configure the Flask app; returns it as an ASGI app. Blocking (runs in a worker thread)."""
    config = settings(environ)
    from a2wsgi import WSGIMiddleware
    from diacritical.app import app, configure
    configure(**config, cookie_path=PREFIX, **configure_kwargs)
    logger.info("Diacritical loaded; serving it at %s/", PREFIX)
    return WSGIMiddleware(app, workers=WORKERS)


async def _plain(send, status: int, body: bytes = b"", headers=()):
    await send({"type": "http.response.start", "status": status, "headers": [
        (b"content-type", b"text/plain; charset=utf-8"), (b"cache-control", b"no-store"), *headers]})
    await send({"type": "http.response.body", "body": body})


class DiacriticalMount:
    def __init__(self, app, environ=os.environ, loader=load_app):
        self.app = app
        self.environ = environ
        self.loader = loader
        self._loading = None  # future of (asgi_app, None) or (None, 503 message)
        self._result = None

    def _load(self):
        try:
            return self.loader(self.environ), None
        except NotConfigured as exc:
            logger.error("%s (%s); %s/ answers 503, the rest of the app is unaffected", NOT_CONFIGURED, exc, PREFIX)
            return None, NOT_CONFIGURED
        except Exception:  # noqa: BLE001  (an import or start-up error in Diacritical must not reach ChitrapurGPT)
            logger.exception("Diacritical failed to load; %s/ answers 503, the rest of the app is unaffected", PREFIX)
            return None, UNAVAILABLE

    async def handler(self):
        """The Flask app as ASGI, loading it on first use (once; concurrent first requests share the load)."""
        if self._result is None:
            loop = asyncio.get_running_loop()
            if self._loading is None or self._loading.get_loop() is not loop:
                self._loading = loop.run_in_executor(None, self._load)
            # shield: a client giving up mid-load mustn't cancel the load other requests wait for
            self._result = await asyncio.shield(self._loading)
        return self._result

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        root = scope.get("root_path", "")
        path = scope.get("path", "")
        local = path[len(root):] if root and path.startswith(root) else path
        if local != PREFIX and not local.startswith(PREFIX + "/"):
            return await self.app(scope, receive, send)

        if scope["type"] == "websocket":  # Diacritical has none
            await send({"type": "websocket.close", "code": 1000})
            return
        if local == PREFIX:
            query = scope.get("query_string", b"")
            location = (root + PREFIX + "/").encode() + (b"?" + query if query else b"")
            return await _plain(send, 307, headers=[(b"location", location)])

        flask_app, error = await self.handler()
        if flask_app is None:
            return await _plain(send, 503, (error or UNAVAILABLE).encode())
        await flask_app(dict(scope, root_path=root + PREFIX), receive, send)


def install(app, environ=os.environ):
    """Serve Diacritical at /diacritical in front of Chainlit's app. Loads nothing yet (see the module docstring)."""
    app.add_middleware(DiacriticalMount, environ=environ)
    if not (environ.get("DIACRITICAL_DATABASE_URL") or "").strip():
        logger.warning("DIACRITICAL_DATABASE_URL is not set: %s/ will answer 503", PREFIX)
    return True
