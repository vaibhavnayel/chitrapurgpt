"""End-to-end tests for the sign-in list (access.py), the Google sign-in check and the /admin pages.

Needs a throwaway local Postgres database; the test wipes it. Run:
  TEST_DATABASE_URL=postgresql://postgres@localhost:5432/cgpt_test python tests/test_admin.py
(needs chainlit==2.9.6, asyncpg and httpx installed)
"""
import asyncio
import glob
import os
import re
import sys
import tempfile
import types
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB = os.environ.get("TEST_DATABASE_URL", "")
if urllib.parse.urlsplit(DB).hostname not in ("localhost", "127.0.0.1"):
    sys.exit("Set TEST_DATABASE_URL to a throwaway *local* Postgres database (it gets wiped).")

os.environ.update({
    "DATABASE_URL": DB,
    "CHAINLIT_AUTH_SECRET": "x" * 64,
    "OAUTH_GOOGLE_CLIENT_ID": "test-client",
    "OAUTH_GOOGLE_CLIENT_SECRET": "test-secret",
    "CHAINLIT_URL": "https://chitrapurgpt.onrender.com",
    "ADMIN_EMAILS": "Admin@gmail.com, second.admin@gmail.com",
})
for k in ("IS_PULL_REQUEST", "RENDER_EXTERNAL_HOSTNAME", "RENDER_EXTERNAL_URL"):
    os.environ.pop(k, None)

# steps.py connects to Pinecone when imported; the chat logic isn't under test here.
_steps = types.ModuleType("steps")
_steps.respond_to_user_message = None
sys.modules["steps"] = _steps

import asyncpg  # noqa: E402
import httpx  # noqa: E402


async def reset_db():
    conn = await asyncpg.connect(DB)
    await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    for path in sorted(glob.glob(os.path.join(ROOT, "prisma/migrations/2025*/migration.sql"))):
        await conn.execute(open(path).read())  # Chainlit's tables, as on the live database
    await conn.execute("""
        INSERT INTO "User"(id, identifier, metadata) VALUES
          ('u1', 'alice@gmail.com', '{}'), ('u2', 'dave@gmail.com', '{}');
        INSERT INTO "Thread"(id, name, metadata, "userId", "createdAt", "updatedAt") VALUES
          ('t1', 'First swamiji', '{}', 'u1', '2026-10-20 10:00', '2026-10-20 10:05'),
          ('t2', '<b>bold</b> & co', '{}', 'u1', '2026-10-21 04:00', '2026-10-21 04:30'),
          ('t3', 'From dave', '{}', 'u2', '2026-09-01 08:00', '2026-09-01 08:02');
        INSERT INTO "Step"(id, name, type, "threadId", output, metadata, "startTime", "endTime") VALUES
          ('s1', 'alice', 'user_message', 't1', 'Who was the first swamiji?', '{}', '2026-10-20 10:00', '2026-10-20 10:00'),
          ('s2', 'run', 'run', 't1', 'INTERNAL STEP', '{}', '2026-10-20 10:00:01', '2026-10-20 10:00:02'),
          ('s3', 'Assistant', 'assistant_message', 't1', 'The first Swamiji was X [1]', '{}', '2026-10-20 10:01', '2026-10-20 10:01'),
          ('s4', 'alice', 'user_message', 't2', '<script>alert(1)</script>', '{}', '2026-10-21 04:00', '2026-10-21 04:00');
    """)
    await conn.close()


def jwt_for(email):
    from chainlit.auth.jwt import create_jwt
    from chainlit.user import User
    return create_jwt(User(identifier=email, metadata={"provider": "google"}))


async def main():
    await reset_db()
    workdir = tempfile.mkdtemp()
    os.chdir(workdir)  # Chainlit writes its default config files into the working directory

    # --- without the secret file the list is never created empty (that would lock everyone out) ---
    import access
    access.WHITELIST_PATHS = ("email_whitelist.txt",)
    try:
        await access.list_allowed()
        raise AssertionError("expected an error without email_whitelist.txt")
    except RuntimeError:
        pass
    conn = await asyncpg.connect(DB)
    assert await conn.fetchval("""SELECT to_regclass('public."AllowedUser"')""") is None
    await conn.close()

    with open("email_whitelist.txt", "w") as f:
        f.write("alice@gmail.com, Bob@Gmail.com,carol@gmail.com\n")

    import app  # registers the Google sign-in callback and the /admin pages
    import chainlit as cl
    from chainlit.config import config
    from chainlit.server import app as server

    # --- the sign-in list is created from the secret file, once ---
    assert [r["email"] for r in await access.list_allowed()] == ["alice@gmail.com", "bob@gmail.com", "carol@gmail.com"]
    with open("email_whitelist.txt", "w") as f:
        f.write("someone.else@gmail.com\n")
    await access.close_pool()
    assert len(await access.list_allowed()) == 3, "the file must only seed a brand-new table"
    conn = await asyncpg.connect(DB)  # the Prisma migration is harmless after the app created the table
    await conn.execute(open(glob.glob(os.path.join(ROOT, "prisma/migrations/*_add_allowed_user/migration.sql"))[0]).read())
    await conn.close()

    # --- Google sign-in check ---
    async def sign_in(email):
        return await config.code.oauth_callback("google", "token", {}, cl.User(identifier=email), None)

    assert (await sign_in("alice@gmail.com")).identifier == "alice@gmail.com"
    assert await sign_in("BOB@gmail.com"), "emails are compared case-insensitively"
    assert await sign_in("stranger@gmail.com") is None
    assert await sign_in("dave@gmail.com") is None
    assert await sign_in("admin@gmail.com"), "admins can always sign in"

    # --- people already signed in are re-checked when they use the app ---
    said = []
    real_cl = app.cl
    app.cl = types.SimpleNamespace(
        user_session=types.SimpleNamespace(get=lambda key: current_user),
        Message=lambda content: types.SimpleNamespace(send=lambda: _record(said, content)),
    )
    current_user = cl.User(identifier="alice@gmail.com")
    assert await app.still_allowed() is True and said == []
    current_user = cl.User(identifier="dave@gmail.com")
    assert await app.still_allowed() is False and said == [app.NO_ACCESS_MESSAGE]
    app.cl = real_cl

    # --- /admin pages ---
    transport = httpx.ASGITransport(app=server)
    async with httpx.AsyncClient(transport=transport, base_url="https://chitrapurgpt.onrender.com") as anon:
        r = await anon.get("/admin")
        assert r.status_code == 401 and "Please sign in first" in r.text
        r = await anon.post("/admin/allow", data={"email": "x@gmail.com"})
        assert r.status_code == 401
        r = await anon.get("/admin/")
        assert r.status_code == 307 and r.headers["location"] == "/admin"
        assert (await anon.get("/admin/me")).json() == {"admin": False}

    async with httpx.AsyncClient(transport=transport, base_url="https://chitrapurgpt.onrender.com",
                                 cookies={"access_token": jwt_for("alice@gmail.com")}) as alice:
        for path in ("/admin", "/admin/user?email=dave%40gmail.com", "/admin/chat/t3"):
            r = await alice.get(path)
            assert r.status_code == 403 and "Admins only" in r.text, path
        r = await alice.post("/admin/allow", data={"email": "x@gmail.com"})
        assert r.status_code == 403
        assert (await alice.get("/admin/me")).json() == {"admin": False}

    from chainlit.auth.jwt import get_jwt_secret  # noqa: F401  (forged cookie: wrong secret)
    os.environ["CHAINLIT_AUTH_SECRET"] = "y" * 64
    forged = jwt_for("admin@gmail.com")
    os.environ["CHAINLIT_AUTH_SECRET"] = "x" * 64
    async with httpx.AsyncClient(transport=transport, base_url="https://chitrapurgpt.onrender.com",
                                 cookies={"access_token": forged}) as mallory:
        assert (await mallory.get("/admin")).status_code == 401

    async with httpx.AsyncClient(transport=transport, base_url="https://chitrapurgpt.onrender.com",
                                 cookies={"access_token": jwt_for("admin@gmail.com")}) as admin:
        assert (await admin.get("/admin/me")).json() == {"admin": True}  # shows the Admin button
        r = await admin.get("/admin")
        assert r.status_code == 200
        assert "You are looking at LIVE data" in r.text
        assert "5 people can sign in, and 1 more has chats but no longer has access" in r.text, r.text
        assert re.search(r'data-email="dave@gmail.com".*No access', r.text)
        assert 'href="/admin/user?email=alice%40gmail.com">2 chats</a>' in r.text
        csrf = re.search(r'name="csrf" value="([0-9a-f]+)"', r.text).group(1)

        async def post(path, **data):
            r = await admin.post(path, data=data)
            assert r.status_code == 303, r.status_code
            return urllib.parse.parse_qs(urllib.parse.urlsplit(r.headers["location"]).query)["msg"][0]

        assert (await post("/admin/allow", csrf=csrf, email="New.Person@Gmail.com")).startswith("Added new.person@gmail.com")
        assert await sign_in("new.person@gmail.com")
        assert "already" in await post("/admin/allow", csrf=csrf, email="new.person@gmail.com")
        assert "doesn't look like" in await post("/admin/allow", csrf=csrf, email="not an email")
        assert "doesn't look like" in await post("/admin/allow", csrf=csrf, email="a'+alert(1)+'@x.com")
        assert "expired" in await post("/admin/allow", csrf="0" * 64, email="evil@gmail.com")
        assert "expired" in await post("/admin/remove", csrf="", email="alice@gmail.com")
        assert await sign_in("evil@gmail.com") is None

        assert (await post("/admin/remove", csrf=csrf, email="Bob@gmail.com")).startswith("Removed bob@gmail.com")
        assert await sign_in("bob@gmail.com") is None
        assert "wasn't on the list" in await post("/admin/remove", csrf=csrf, email="bob@gmail.com")
        assert "is an admin" in await post("/admin/remove", csrf=csrf, email="second.admin@gmail.com")
        assert (await post("/admin/allow", csrf=csrf, email="dave@gmail.com")).startswith("Added dave@gmail.com")
        assert await sign_in("dave@gmail.com")

        conn = await asyncpg.connect(DB)
        assert await conn.fetchval('SELECT COUNT(*) FROM "Thread"') == 3, "chats are never deleted"
        assert await conn.fetchval("""SELECT "addedBy" FROM "AllowedUser" WHERE email = 'dave@gmail.com'""") == "admin@gmail.com"
        await conn.close()

        r = await admin.get("/admin", params={"msg": "<script>x</script>", "kind": "bad"})
        assert "&lt;script&gt;x&lt;/script&gt;" in r.text and "<script>x" not in r.text

        r = await admin.get("/admin/user", params={"email": "Alice@gmail.com"})
        assert r.status_code == 200 and "Chats by alice@gmail.com" in r.text
        assert r.text.index("&lt;b&gt;bold&lt;/b&gt; &amp; co") < r.text.index("First swamiji"), "newest first"
        assert "21 Oct 2026, 9:30 am" in r.text, "times shown in IST"

        r = await admin.get("/admin/chat/t1")
        assert r.status_code == 200
        assert "Who was the first swamiji?" in r.text and "The first Swamiji was X [1]" in r.text
        assert "INTERNAL STEP" not in r.text, "only user and assistant messages are shown"
        assert r.text.index("Who was the first swamiji?") < r.text.index("The first Swamiji was X")
        r = await admin.get("/admin/chat/t2")
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in r.text and "<script>alert" not in r.text
        assert (await admin.get("/admin/chat/nope")).status_code == 404

    # Removed people can be added again, and the preview shows its own warning.
    os.environ["IS_PULL_REQUEST"] = "true"
    async with httpx.AsyncClient(transport=transport, base_url="https://chitrapurgpt.onrender.com",
                                 cookies={"access_token": jwt_for("second.admin@gmail.com")}) as admin2:
        r = await admin2.get("/admin")
        assert "This preview site uses the real ChitrapurGPT database" in r.text
        assert re.search(r'data-email="bob@gmail.com".*No access', r.text) is None, "bob has no chats, so he's gone"
    os.environ.pop("IS_PULL_REQUEST")

    # The Chainlit UI still serves everything else.
    async with httpx.AsyncClient(transport=transport, base_url="https://chitrapurgpt.onrender.com") as anon:
        r = await anon.get("/")
        assert r.status_code == 200 and "<html" in r.text.lower() and "ChitrapurGPT admin" not in r.text

    await access.close_pool()
    print("admin: all checks passed")


async def _record(said, content):
    said.append(content)


if __name__ == "__main__":
    asyncio.run(main())
