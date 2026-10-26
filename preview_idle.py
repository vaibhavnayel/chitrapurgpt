"""Idle tracking for Render PR previews, so a watcher can switch a preview off when nobody uses it.

Only installed when the app runs as a PR preview (IS_PULL_REQUEST=true). It counts as activity:
any normal HTTP request (pages, admin pages, API calls) and every socket.io event a browser sends
(a chat message, opening a chat...). Socket.io's own keep-alive pings don't count, so a forgotten
open tab still goes idle.

GET /__preview/idle returns {"idle_seconds": N, "up_seconds": M} and doesn't count as activity.
"""
import json
import os
import time

IDLE_PATH = "/__preview/idle"
SOCKET_PREFIX = "/ws/socket.io"


class IdleTracker:
    def __init__(self, app, clock=time.monotonic):
        self.app = app
        self.clock = clock
        self.started = self.last_activity = clock()

    def touch(self):
        self.last_activity = self.clock()

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        path = scope.get("path", "")
        if kind == "http":
            if path == IDLE_PATH:
                now = self.clock()
                body = json.dumps({"idle_seconds": int(now - self.last_activity),
                                   "up_seconds": int(now - self.started)}).encode()
                await send({"type": "http.response.start", "status": 200, "headers": [
                    (b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
                await send({"type": "http.response.body", "body": body})
                return
            if not path.startswith(SOCKET_PREFIX):  # long-polling transport carries pings too
                self.touch()
        elif kind == "websocket":
            self.touch()

            async def receive_and_track():
                message = await receive()
                if message.get("type") == "websocket.receive" and (message.get("text") or "").startswith("42"):
                    self.touch()  # "42..." is a socket.io event; "2"/"3" are ping/pong
                return message

            return await self.app(scope, receive_and_track, send)
        return await self.app(scope, receive, send)


def install(app, environ=os.environ):
    """Track activity on PR previews only. Returns True if installed."""
    if environ.get("IS_PULL_REQUEST") != "true":
        return False
    app.add_middleware(IdleTracker)
    return True
