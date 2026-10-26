"""End-to-end test of the preview Google sign-in relay against real Chainlit routes.

Runs two Chainlit apps in-process (live site and preview PR #7), fakes Google's token/userinfo
endpoints, and walks a browser through: preview login -> Google -> live callback -> preview
callback -> signed in on the preview. Also checks normal live sign-in is unaffected.
Run: python tests/test_preview_oauth.py   (needs chainlit==2.9.6 and httpx installed)
"""
import importlib
import os
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LIVE = "https://chitrapurgpt.onrender.com"
PREVIEW = "https://chitrapur-gpt-pr-7.onrender.com"
CALLBACK = "/auth/oauth/google/callback"


def fresh_app(env):
    """Import a fresh Chainlit server app configured like Render would (live or preview)."""
    for k in ("IS_PULL_REQUEST", "RENDER_EXTERNAL_HOSTNAME", "RENDER_EXTERNAL_URL"):
        os.environ.pop(k, None)
    os.environ.update({
        "CHAINLIT_URL": LIVE,  # copied from the live service to previews by Render
        "OAUTH_GOOGLE_CLIENT_ID": "test-client",
        "OAUTH_GOOGLE_CLIENT_SECRET": "test-secret",
        "CHAINLIT_AUTH_SECRET": "x" * 64,
        **env,
    })
    for mod in [m for m in list(sys.modules) if m == "preview_oauth" or m.startswith("chainlit")]:
        del sys.modules[mod]
    import chainlit as cl
    from chainlit.server import app
    from chainlit.oauth_providers import GoogleOAuthProvider

    seen = {}

    async def fake_get_token(self, code, url):
        seen["redirect_uri"] = url
        assert code == "good-code"
        return "tok"

    async def fake_user_info(self, token):
        return ({"email": "someone@example.com"}, cl.User(identifier="someone@example.com", metadata={"provider": "google"}))

    GoogleOAuthProvider.get_token = fake_get_token
    GoogleOAuthProvider.get_user_info = fake_user_info

    @cl.oauth_callback
    def oauth_callback(provider_id, token, raw_user_data, default_user):
        return default_user

    import preview_oauth
    importlib.reload(preview_oauth)
    preview_oauth.install(app)
    return app, seen


def run():
    from starlette.testclient import TestClient

    # ---- live site: plain sign-in is untouched
    live_app, live_seen = fresh_app({})
    live = TestClient(live_app, base_url=LIVE)
    r = live.get("/auth/oauth/google", follow_redirects=False)
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(r.headers["location"]).query)
    assert "!pr" not in q["state"][0], "live login must not be tagged"
    assert q["redirect_uri"][0] == LIVE + CALLBACK
    r = live.get(CALLBACK, params={"code": "good-code", "state": q["state"][0]}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith("/login/callback"), (r.status_code, r.headers)
    print("ok  live sign-in unchanged:", r.status_code, r.headers["location"])

    # ---- preview PR #7: login is tagged and goes to the LIVE callback
    prev_app, prev_seen = fresh_app({"IS_PULL_REQUEST": "true", "RENDER_EXTERNAL_HOSTNAME": "chitrapur-gpt-pr-7.onrender.com"})
    preview = TestClient(prev_app, base_url=PREVIEW)
    r = preview.get("/auth/oauth/google", follow_redirects=False)
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(r.headers["location"]).query)
    state = q["state"][0]
    assert state.endswith("!pr7"), state
    assert q["redirect_uri"][0] == LIVE + CALLBACK
    print("ok  preview login tagged:", state[-6:], "redirect_uri:", q["redirect_uri"][0])

    # Google sends the browser to the live callback; the live site (relay) bounces it to the preview
    live_app2, _ = fresh_app({})
    live2 = TestClient(live_app2, base_url=LIVE)
    r = live2.get(CALLBACK, params={"code": "good-code", "state": state}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith(PREVIEW + CALLBACK + "?"), r.headers
    print("ok  live site relays to:", r.headers["location"].split("?")[0])

    # Preview callback (fresh preview app instance shares nothing but the browser's cookie)
    prev_app2, prev_seen2 = fresh_app({"IS_PULL_REQUEST": "true", "RENDER_EXTERNAL_HOSTNAME": "chitrapur-gpt-pr-7.onrender.com"})
    preview2 = TestClient(prev_app2, base_url=PREVIEW, cookies={"oauth_state": state[:-4]})
    relay_q = urllib.parse.urlsplit(r.headers["location"]).query
    r = preview2.get(CALLBACK + "?" + relay_q, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith("/login/callback"), (r.status_code, r.text[:200])
    assert prev_seen2["redirect_uri"] == LIVE + CALLBACK, prev_seen2
    assert any(c.startswith("access_token") for c in r.headers.get_list("set-cookie")), r.headers.get_list("set-cookie")
    print("ok  preview signs the user in; code exchanged with redirect_uri", prev_seen2["redirect_uri"])

    # Tampered state (wrong PR number) is not accepted on the preview
    r = preview2.get(CALLBACK, params={"code": "good-code", "state": state[:-4] + "!pr8"}, follow_redirects=False)
    assert r.status_code == 401, r.status_code
    print("ok  state tagged for another preview is rejected:", r.status_code)
    print("ALL PASSED")


if __name__ == "__main__":
    run()
