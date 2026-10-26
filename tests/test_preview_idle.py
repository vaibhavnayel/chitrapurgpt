"""Tests for the idle tracker used on Render PR previews (preview_idle.py).
Run: python tests/test_preview_idle.py
"""
import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import preview_idle  # noqa: E402


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


async def inner_app(scope, receive, send):
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    else:  # websocket: read every frame the client sends
        while (await receive())["type"] != "websocket.disconnect":
            pass


async def http(app, path):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    await app({"type": "http", "method": "GET", "path": path}, receive, send)
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


async def websocket(app, clock, frames, seconds_between=0):
    """Open a socket.io websocket and send `frames`, the clock moving before each one."""
    queue = [{"type": "websocket.receive", "text": f} for f in frames] + [{"type": "websocket.disconnect"}]

    async def receive():
        message = queue.pop(0)
        if message["type"] == "websocket.receive":
            clock.t += seconds_between
        return message

    async def send(message):
        pass

    await app({"type": "websocket", "path": "/ws/socket.io/"}, receive, send)


async def status(app):
    return json.loads(await http(app, preview_idle.IDLE_PATH))


async def main():
    clock = Clock()
    app = preview_idle.IdleTracker(inner_app, clock=clock)

    clock.t += 100
    assert (await status(app))["idle_seconds"] == 100
    assert (await status(app))["idle_seconds"] == 100, "checking idle must not count as activity"

    await http(app, "/admin")
    assert (await status(app))["idle_seconds"] == 0, "a page request is activity"

    clock.t += 50
    await http(app, "/ws/socket.io/?EIO=4&transport=polling")
    assert (await status(app))["idle_seconds"] == 50, "socket.io long-polling (keep-alive) is not activity"

    # A forgotten open tab: connecting counts, the pings/pongs after it don't.
    await websocket(app, clock, ["2", "3", "2", "3"], seconds_between=60)
    assert (await status(app))["idle_seconds"] == 240, await status(app)

    # A chat message (socket.io event "42...") is activity.
    await websocket(app, clock, ["2", '42["client_message",{"message":{}}]'], seconds_between=60)
    assert (await status(app))["idle_seconds"] == 0

    assert (await status(app))["up_seconds"] == 100 + 50 + 240 + 120

    # Only installed on previews.
    class FakeApp:
        def __init__(self):
            self.added = []

        def add_middleware(self, cls, **kw):
            self.added.append(cls)

    fake = FakeApp()
    assert preview_idle.install(fake, environ={}) is False
    assert preview_idle.install(fake, environ={"IS_PULL_REQUEST": "false"}) is False
    assert fake.added == []
    assert preview_idle.install(fake, environ={"IS_PULL_REQUEST": "true"}) is True
    assert fake.added == [preview_idle.IdleTracker]
    print("preview_idle: all checks passed")


if __name__ == "__main__":
    asyncio.run(main())
