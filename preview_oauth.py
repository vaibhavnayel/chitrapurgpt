"""Google sign-in for Render pull-request previews.

Google only accepts sign-in redirects to domains registered on the OAuth client (10 at most while
the app is unverified), and every Render preview gets its own domain
(chitrapur-gpt-pr-<n>.onrender.com). So previews don't get their own redirect. Instead they reuse
the live site's registered callback:

  1. preview  /auth/oauth/google : Chainlit redirects to Google. CHAINLIT_URL is the live site
     (Render copies it to previews), so redirect_uri is the live callback. We tag the OAuth
     `state` with "!pr<n>". Chainlit's state alphabet never contains "!".
  2. Google -> live /auth/oauth/google/callback?code=..&state=<state>!pr<n>
  3. live site : the tag means a preview started this sign-in, so send the browser on, unchanged,
     to https://chitrapur-gpt-pr-<n>.onrender.com/auth/oauth/google/callback.
  4. preview callback : strip the tag. Chainlit checks the state cookie (set on the preview's own
     domain in step 1), exchanges the code with the same live redirect_uri Google issued it for,
     and signs the user in on the preview.

Sign-ins on the live site carry no tag and pass through untouched. The relay only ever targets
this app's own preview hostnames, and a bare code is useless without the client secret.
"""
import logging
import os
import re
import urllib.parse

from starlette.responses import RedirectResponse

LOGIN_PATH = "/auth/oauth/google"
CALLBACK_PATH = "/auth/oauth/google/callback"
PREVIEW_HOST = "chitrapur-gpt-pr-{n}.onrender.com"
_HOST_RE = re.compile(r"^chitrapur-gpt-pr-(\d{1,5})\.onrender\.com$")
_TAGGED_STATE_RE = re.compile(r"^(?P<state>[^!]+)!pr(?P<pr>\d{1,5})$")

logger = logging.getLogger(__name__)


def current_preview_number(environ=os.environ):
    """PR number when running as a Render PR preview, else None."""
    if environ.get("IS_PULL_REQUEST") != "true":
        return None
    host = environ.get("RENDER_EXTERNAL_HOSTNAME") or urllib.parse.urlsplit(environ.get("RENDER_EXTERNAL_URL", "")).hostname or ""
    m = _HOST_RE.match(host)
    return m.group(1) if m else None


def _tag_location(location: str, pr: str) -> str:
    parts = urllib.parse.urlsplit(location)
    query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    if "state" in query:
        query["state"] = [query["state"][0] + "!pr" + pr]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query, doseq=True)))


class PreviewOAuthRelay:
    """ASGI middleware implementing the relay described in the module docstring."""

    def __init__(self, app, preview_number=None):
        self.app = app
        self.preview = preview_number

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "GET":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")

        if path == CALLBACK_PATH:
            raw_qs = scope.get("query_string", b"").decode()
            query = urllib.parse.parse_qs(raw_qs, keep_blank_values=True)
            m = _TAGGED_STATE_RE.match((query.get("state") or [""])[0])
            if m:
                if self.preview is None:
                    # Live site: hand this sign-in back to the preview that started it.
                    target = f"https://{PREVIEW_HOST.format(n=m['pr'])}{CALLBACK_PATH}?{raw_qs}"
                    logger.info("Relaying Google sign-in to preview PR #%s", m["pr"])
                    return await RedirectResponse(target, status_code=302)(scope, receive, send)
                if m["pr"] == self.preview:
                    query["state"] = [m["state"]]
                    scope = dict(scope, query_string=urllib.parse.urlencode(query, doseq=True).encode())

        elif path == LOGIN_PATH and self.preview is not None:
            pr = self.preview

            async def send_tagged(message):
                if message["type"] == "http.response.start":
                    message = dict(message, headers=[
                        (k, _tag_location(v.decode("latin-1"), pr).encode("latin-1") if k.lower() == b"location" else v)
                        for k, v in message.get("headers", [])
                    ])
                await send(message)

            return await self.app(scope, receive, send_tagged)

        return await self.app(scope, receive, send)


def install(app):
    """Add the relay to Chainlit's FastAPI app (call at import time, before the server starts)."""
    pr = current_preview_number()
    app.add_middleware(PreviewOAuthRelay, preview_number=pr)
    logger.info("Google sign-in relay installed (%s)", f"preview PR #{pr}" if pr else "live site")
