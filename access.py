"""Who may sign in to ChitrapurGPT, and who is an admin.

- The sign-in list lives in the app's Postgres database (table "AllowedUser"), so admins can add
  and remove people from the /admin page. The first time the table is created it is filled from
  the Render secret file email_whitelist.txt, which is left untouched as a backup.
- Admins are the emails in the ADMIN_EMAILS env var (comma-separated). Admins can always sign in.
- Removing someone only takes away their sign-in. Their past chats stay in the database.
"""
import asyncio
import logging
import os
import re

import asyncpg

logger = logging.getLogger(__name__)

WHITELIST_FILE = "email_whitelist.txt"
# Render puts secret files in the app's root directory and in /etc/secrets.
WHITELIST_PATHS = (WHITELIST_FILE, "/etc/secrets/" + WHITELIST_FILE)
_SEED_LOCK_ID = 727274  # pg advisory lock so two instances starting together seed only once
_EMAIL_RE = re.compile(r"^[a-z0-9._%+\-]+@[a-z0-9\-]+(\.[a-z0-9\-]+)*\.[a-z]{2,}$")

_pool = None
_pool_lock = asyncio.Lock()


def normalize(email: str | None) -> str:
    return (email or "").strip().lower()


def parse_emails(text: str | None) -> list[str]:
    """Emails separated by commas, spaces or newlines, normalized, without duplicates."""
    seen = []
    for part in re.split(r"[,\s]+", text or ""):
        email = normalize(part)
        if email and email not in seen:
            seen.append(email)
    return seen


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(normalize(email)))


def admin_emails() -> list[str]:
    return parse_emails(os.environ.get("ADMIN_EMAILS", ""))


def is_admin(email: str | None) -> bool:
    return bool(email) and normalize(email) in admin_emails()


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=0, max_size=3)
                try:
                    await _ensure_table(pool)
                except BaseException:
                    await pool.close()
                    raise
                _pool = pool
    return _pool


async def close_pool():
    """Used by tests (each test may run in its own event loop)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def _ensure_table(pool):
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1)", _SEED_LOCK_ID)
        if await conn.fetchval("""SELECT to_regclass('public."AllowedUser"') IS NOT NULL"""):
            return
        emails = _read_whitelist_file()
        if not emails:
            # Never start an empty list: that would lock everyone out. Try again on the next request.
            raise RuntimeError(f"{WHITELIST_FILE} is missing or empty, so the sign-in list can't be created yet")
        await conn.execute("""
            CREATE TABLE "AllowedUser" (
                "email"     TEXT PRIMARY KEY,
                "addedBy"   TEXT,
                "createdAt" TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")
        await conn.executemany(
            """INSERT INTO "AllowedUser" ("email", "addedBy") VALUES ($1, $2) ON CONFLICT DO NOTHING""",
            [(e, WHITELIST_FILE) for e in emails],
        )
        logger.info("Created AllowedUser table with %d emails from %s", len(emails), WHITELIST_FILE)


def _read_whitelist_file() -> list[str]:
    for path in WHITELIST_PATHS:
        try:
            with open(path) as f:
                return parse_emails(f.read())
        except FileNotFoundError:
            continue
    return []


async def is_allowed(email: str | None) -> bool:
    """True if this email may use the app. Raises if the database can't be reached."""
    email = normalize(email)
    if not email:
        return False
    if is_admin(email):
        return True
    pool = await get_pool()
    return bool(await pool.fetchval("""SELECT 1 FROM "AllowedUser" WHERE "email" = $1""", email))


async def list_allowed() -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch("""SELECT "email", "addedBy", "createdAt" FROM "AllowedUser" ORDER BY "email" """)
    return [dict(r) for r in rows]


async def add(email: str, added_by: str) -> bool:
    """Allow an email to sign in. Returns False if it was already on the list."""
    email = normalize(email)
    if not is_valid_email(email):
        raise ValueError(f"not a valid email address: {email!r}")
    pool = await get_pool()
    status = await pool.execute(
        """INSERT INTO "AllowedUser" ("email", "addedBy") VALUES ($1, $2) ON CONFLICT DO NOTHING""",
        email, normalize(added_by),
    )
    added = status.endswith(" 1")
    logger.info("Admin %s %s %s", added_by, "allowed" if added else "re-allowed (no change)", email)
    return added


async def remove(email: str, removed_by: str) -> bool:
    """Stop an email from signing in (their chats are kept). Returns False if it wasn't on the list."""
    email = normalize(email)
    pool = await get_pool()
    status = await pool.execute("""DELETE FROM "AllowedUser" WHERE "email" = $1""", email)
    removed = status.endswith(" 1")
    logger.info("Admin %s %s %s", removed_by, "removed" if removed else "tried to remove (not listed)", email)
    return removed
