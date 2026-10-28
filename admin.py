"""Admin dashboard at /admin: manage who can sign in and read users' chats.

Only emails in the ADMIN_EMAILS env var can open it. They sign in with Google on the normal
ChitrapurGPT page first; the dashboard reads the same sign-in cookie. Pages are plain
server-rendered HTML (no extra dependencies). Chats are read-only here.
"""
import hashlib
import hmac
import html
import logging
import os
import urllib.parse
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from chainlit.auth import get_token_from_cookies
from chainlit.auth.jwt import decode_jwt

import access
import kb

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
MESSAGE_TYPES = ("user_message", "assistant_message")
router = APIRouter()


# ---------- helpers ----------

def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _fmt_time(value) -> str:
    """Database times are UTC (Prisma's timestamp columns have no zone). Show them in IST."""
    if not value:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    t = value.astimezone(IST)
    return t.strftime("%-d %b %Y, %-I:%M %p").replace("AM", "am").replace("PM", "pm")


def _signed_in_email(request: Request) -> str | None:
    token = get_token_from_cookies(request.cookies)
    if not token:
        return None
    try:
        return access.normalize(decode_jwt(token).identifier)
    except Exception:
        return None


def _csrf_token(email: str) -> str:
    secret = os.environ.get("CHAINLIT_AUTH_SECRET", "").encode()
    return hmac.new(secret, f"admin-form:{email}".encode(), hashlib.sha256).hexdigest()


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _no_access_note(n: int) -> str:
    if not n:
        return ""
    return f", and {n} more {'has' if n == 1 else 'have'} chats but no longer {'has' if n == 1 else 'have'} access"


def _is_preview() -> bool:
    return os.environ.get("IS_PULL_REQUEST") == "true"


CSS = """
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;background:#f5f6f8;color:#1d2433;font-size:17px;line-height:1.5}
header{background:#1f3a5f;color:#fff;padding:12px 20px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
header a{color:#fff}
header .title{font-size:20px;font-weight:600}
main{max-width:1000px;margin:0 auto;padding:18px}
h1{font-size:24px;margin:8px 0 14px}
h2{font-size:20px;margin:26px 0 10px}
.banner{border-radius:8px;padding:12px 16px;margin-bottom:16px}
.warn{background:#fff4e0;border:2px solid #e8a33d}
.ok{background:#e7f5ea;border:1px solid #63a974}
.bad{background:#fdecea;border:1px solid #d9534f}
.card{background:#fff;border:1px solid #dfe3e8;border-radius:8px;padding:14px 16px}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #dfe3e8;border-radius:8px}
th,td{padding:9px 12px;border-bottom:1px solid #e6e9ed;text-align:left;vertical-align:middle}
th{background:#eef1f5;font-weight:600;white-space:nowrap}
tr.off td{color:#7a828c}
.tag{display:inline-block;font-size:14px;padding:1px 8px;border-radius:10px;white-space:nowrap}
.tag.yes{background:#e3f2e6;color:#1e6b32}
.tag.no{background:#f1f1f1;color:#666}
.tag.admin{background:#e6eefb;color:#1f4fa0}
button,.button{font-size:16px;padding:6px 14px;border-radius:6px;border:1px solid #8a94a0;background:#fff;color:#1d2433;cursor:pointer;text-decoration:none;display:inline-block}
button.primary{background:#1f6feb;border-color:#1f6feb;color:#fff}
button.danger{border-color:#c62828;color:#c62828}
input[type=email],input[type=search]{font-size:16px;padding:7px 10px;border:1px solid #8a94a0;border-radius:6px;width:320px;max-width:100%}
form.inline{display:inline;margin:0}
.muted{color:#6b7480;font-size:15px}
.msg{background:#fff;border:1px solid #dfe3e8;border-radius:8px;padding:10px 14px;margin:10px 0}
.msg.user_message{border-left:5px solid #1f6feb}
.msg.assistant_message{border-left:5px solid #2e9e50}
.msg .who{font-weight:600;font-size:15px;margin-bottom:4px}
.msg .text{white-space:pre-wrap;overflow-wrap:anywhere}
header nav a{margin-right:14px}
main.wide{max-width:1500px}
.tag.on{background:#e3f2e6;color:#1e6b32;border:1px solid #9ccfa6}
.tag.off{background:#f6f6f6;color:#888;border:1px solid #ddd}
.tag.run{background:#fff4e0;color:#8a5a00}
.tag.bad{background:#fdecea;color:#a12622}
form.kb-toggle{display:inline;margin:0 4px 0 0}
form.kb-toggle button{font-size:14px;padding:1px 8px;border-radius:10px}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0 16px}
.stat{background:#fff;border:1px solid #dfe3e8;border-radius:8px;padding:10px 14px;min-width:120px}
.stat .n{font-size:22px;font-weight:600}
.stat .l{color:#6b7480;font-size:14px}
details{margin:10px 0}
summary{cursor:pointer;font-weight:600}
.two-col{display:flex;gap:18px;align-items:flex-start}
.two-col>div{flex:1;min-width:0}
.article-text{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;border:1px solid #dfe3e8;border-radius:8px;padding:14px;font-size:16px}
.page-img{width:100%;border:1px solid #dfe3e8;border-radius:4px;margin-bottom:10px;background:#fff}
input[type=text],select,textarea{font-size:16px;padding:6px 9px;border:1px solid #8a94a0;border-radius:6px;max-width:100%}
@media (max-width:900px){.two-col{flex-direction:column}}
@media (max-width:640px){
  main{padding:12px}
  table,tbody,tr,td{display:block;width:100%}
  tr.head{display:none}
  table tr{padding:8px 0;border-bottom:1px solid #e6e9ed}
  td{border:none;padding:3px 12px}
  td[data-label]::before{content:attr(data-label) ": ";color:#6b7480}
  td:empty{display:none}
}
"""


def _page(title: str, body: str, *, email: str | None = None, flash: str = "", flash_kind: str = "ok",
          status: int = 200, banner: bool = True, wide: bool = False) -> HTMLResponse:
    if _is_preview():
        warning = ("<strong>⚠️ You are looking at LIVE data.</strong> This preview site uses the real "
                   "ChitrapurGPT database: the people and chats below are real, and any change you make "
                   "here is saved for real.")
    else:
        warning = ("<strong>⚠️ You are looking at LIVE data.</strong> These are the real ChitrapurGPT users "
                   "and chats. Changes take effect straight away.")
    who = f'<span class="muted" style="color:#d6e2f2">Signed in as {_e(email)}</span>' if email else ""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">
<title>{_e(title)} · ChitrapurGPT admin</title><style>{CSS}</style></head><body>
<header><span class="title"><a href="/admin" style="text-decoration:none">ChitrapurGPT admin</a></span>
{'<nav><a href="/admin">People</a><a href="/admin/kb">Collections &amp; uploads</a></nav>' if email else ''}
<span>{who} &nbsp; <a href="/">Open ChitrapurGPT</a></span></header>
<main{' class="wide"' if wide else ''}>
{f'<div class="banner warn">{warning}</div>' if banner else ''}
{f'<div class="banner {flash_kind}">{_e(flash)}</div>' if flash else ''}
{body}
</main></body></html>"""
    return HTMLResponse(doc, status_code=status, headers={
        "Cache-Control": "no-store", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
    })


def _require_admin(request: Request):
    """Returns (email, None) for an admin, else (None, response to send instead)."""
    email = _signed_in_email(request)
    if not email:
        body = """<h1>Please sign in first</h1><div class="card">
<p>Sign in to ChitrapurGPT with your Google account, then come back to this page.</p>
<p><a class="button" href="/login">Sign in</a></p></div>"""
        return None, _page("Sign in", body, status=401, banner=False)
    if not access.is_admin(email):
        logger.info("Non-admin %s tried to open the admin page", email)
        body = f"""<h1>Admins only</h1><div class="card"><p>This page is only for ChitrapurGPT admins.
You are signed in as <strong>{_e(email)}</strong>.</p><p><a class="button" href="/">Back to ChitrapurGPT</a></p></div>"""
        return None, _page("Admins only", body, email=email, status=403, banner=False)
    return email, None


async def _form(request: Request) -> dict[str, str]:
    body = (await request.body()).decode("utf-8", "replace")
    return {k: v[0] for k, v in urllib.parse.parse_qs(body, keep_blank_values=True).items()}


def _back_to_dashboard(message: str, kind: str = "ok") -> RedirectResponse:
    return RedirectResponse("/admin?" + urllib.parse.urlencode({"msg": message, "kind": kind}), status_code=303)


# ---------- pages ----------

@router.get("/admin/me")
async def me(request: Request):
    """Used by public/admin-button.js to show the Admin button in the chat page to admins only."""
    return JSONResponse({"admin": access.is_admin(_signed_in_email(request))},
                        headers={"Cache-Control": "no-store"})


@router.get("/admin/")
async def dashboard_slash():
    return RedirectResponse("/admin", status_code=307)


@router.get("/admin")
async def dashboard(request: Request):
    email, denied = _require_admin(request)
    if denied:
        return denied

    pool = await access.get_pool()
    allowed = {r["email"]: r for r in await access.list_allowed()}
    usage = {}
    for r in await pool.fetch("""
            SELECT u."identifier" AS email, COUNT(t."id") AS chats, MAX(t."updatedAt") AS last_chat
            FROM "User" u LEFT JOIN "Thread" t ON t."userId" = u."id"
            GROUP BY u."identifier" """):
        e = access.normalize(r["email"])
        prev = usage.get(e, {"chats": 0, "last_chat": None})
        last = max([d for d in (prev["last_chat"], r["last_chat"]) if d], default=None)
        usage[e] = {"chats": prev["chats"] + r["chats"], "last_chat": last}
    admins = set(access.admin_emails())

    people = sorted(set(allowed) | set(usage) | admins)
    can_sign_in = [p for p in people if p in allowed or p in admins]
    no_access = [p for p in people if p not in can_sign_in]
    csrf = _csrf_token(email)
    try:
        kbs, own_settings = await kb.list_kbs(), await kb.access_rows()
    except Exception:  # noqa: BLE001  (the people list still works without the knowledge-base database)
        logger.exception("Couldn't read the knowledge bases")
        kbs, own_settings = None, {}

    def collections(p):
        if kbs is None:
            return '<span class="muted">unavailable</span>'
        own = own_settings.get(p, {})
        on = {k["id"] for k in kb.resolve(kbs, own)}
        out = []
        for k in kbs:
            state = "on" if k["id"] in on else "off"
            why = "set for this person" if k["id"] in own else "the collection's default"
            out.append(f"""<form class="kb-toggle" method="post" action="/admin/kb/access">
<input type="hidden" name="csrf" value="{csrf}"><input type="hidden" name="email" value="{_e(p)}">
<input type="hidden" name="kb_id" value="{_e(k['id'])}"><input type="hidden" name="enabled" value="{0 if state == 'on' else 1}">
<button class="tag {state}" title="{_e(k['name'])}: {state} ({why}). Click to turn it {'off' if state == 'on' else 'on'}.">{_e(k['source_prefix'])} {'✓' if state == 'on' else '✗'}</button></form>""")
        return "".join(out)

    def row(p):
        u = usage.get(p, {"chats": 0, "last_chat": None})
        if p in admins:
            status, action = '<span class="tag admin">Admin · can sign in</span>', ""
        elif p in allowed:
            status = '<span class="tag yes">Can sign in</span>'
            action = f"""<form class="inline" method="post" action="/admin/remove" data-email="{_e(p)}"
onsubmit="return confirm('Remove ' + this.dataset.email + '? They will no longer be able to sign in. Their chats are kept.')">
<input type="hidden" name="csrf" value="{csrf}"><input type="hidden" name="email" value="{_e(p)}">
<button class="danger">Remove</button></form>"""
        else:
            status = '<span class="tag no">No access</span>'
            action = f"""<form class="inline" method="post" action="/admin/allow">
<input type="hidden" name="csrf" value="{csrf}"><input type="hidden" name="email" value="{_e(p)}">
<button>Give access</button></form>"""
        chats = (f'<a href="/admin/user?{urllib.parse.urlencode({"email": p})}">{u["chats"]} chat'
                 f'{"" if u["chats"] == 1 else "s"}</a>') if u["chats"] else '<span class="muted">none</span>'
        cls = "" if p in can_sign_in else ' class="off"'
        return (f'<tr{cls} data-email="{_e(p)}"><td><strong>{_e(p)}</strong></td><td>{status}</td>'
                f'<td data-label="Collections">{collections(p)}</td>'
                f'<td data-label="Chats">{chats}</td>'
                f'<td data-label="Last chat">{_fmt_time(u["last_chat"]) or "<span class=muted>–</span>"}</td>'
                f'<td>{action}</td></tr>')

    rows = "".join(row(p) for p in can_sign_in + no_access)
    body = f"""
<h1>People</h1>
<p>{_count(len(can_sign_in), "person", "people")} can sign in{_no_access_note(len(no_access))}.
Click the number of chats to read them.</p>

<div class="card"><form method="post" action="/admin/allow" style="margin:0">
<label for="new-email"><strong>Add someone:</strong> enter the Gmail address they sign in with</label><br>
<input type="hidden" name="csrf" value="{csrf}">
<input id="new-email" type="email" name="email" required placeholder="name@gmail.com" style="margin:8px 8px 0 0">
<button class="primary">Add</button></form></div>

<h2>Everyone</h2>
<p><input type="search" id="filter" placeholder="Search by email" oninput="
var q=this.value.toLowerCase();document.querySelectorAll('tr[data-email]').forEach(function(r){{
r.style.display=r.dataset.email.indexOf(q)<0?'none':''}})"></p>
<div class="scroll"><table>
<tr class="head"><th>Email</th><th>Access</th><th>Collections</th><th>Chats</th><th>Last chat</th><th></th></tr>
{rows or '<tr><td colspan="6" class="muted">Nobody yet.</td></tr>'}
</table></div>
<p class="muted">Removing someone only stops them signing in. Their chats stay here.
Admins are set by the site owner and can always sign in.
Collections: which magazines each person's questions are answered from (✓ on, ✗ off). Click one to switch it.
Someone you haven't switched gets each collection's default (<a href="/admin/kb">Collections &amp; uploads</a>).</p>"""
    return _page("People", body, email=email, flash=request.query_params.get("msg", ""),
                 flash_kind="bad" if request.query_params.get("kind") == "bad" else "ok")


@router.get("/admin/user")
async def user_chats(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    target = access.normalize(request.query_params.get("email"))
    pool = await access.get_pool()
    threads = await pool.fetch("""
        SELECT t."id", t."name", t."createdAt", t."updatedAt",
               (SELECT COUNT(*) FROM "Step" s WHERE s."threadId" = t."id"
                  AND s."type"::text IN ('user_message', 'assistant_message')) AS messages
        FROM "Thread" t JOIN "User" u ON u."id" = t."userId"
        WHERE lower(u."identifier") = $1
        ORDER BY t."updatedAt" DESC
        LIMIT 500""", target)
    rows = "".join(
        f'<tr><td><a href="/admin/chat/{urllib.parse.quote(t["id"])}">{_e(t["name"] or "(untitled chat)")}</a></td>'
        f'<td data-label="Messages">{t["messages"]}</td><td data-label="Started">{_fmt_time(t["createdAt"])}</td>'
        f'<td data-label="Last message">{_fmt_time(t["updatedAt"])}</td></tr>'
        for t in threads)
    body = f"""<p><a href="/admin">← All people</a></p>
<h1>Chats by {_e(target)}</h1>
<p>{len(threads)} chat{"" if len(threads) == 1 else "s"}, newest first.</p>
<div class="scroll"><table><tr class="head"><th>Chat</th><th>Messages</th><th>Started</th><th>Last message</th></tr>
{rows or '<tr><td colspan="4" class="muted">No chats.</td></tr>'}</table></div>"""
    return _page(f"Chats by {target}", body, email=admin)


@router.get("/admin/chat/{thread_id}")
async def chat(request: Request, thread_id: str):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    pool = await access.get_pool()
    thread = await pool.fetchrow("""
        SELECT t."id", t."name", t."createdAt", u."identifier" AS email
        FROM "Thread" t LEFT JOIN "User" u ON u."id" = t."userId" WHERE t."id" = $1""", thread_id)
    if not thread:
        return _page("Chat not found", '<p><a href="/admin">← All people</a></p><h1>Chat not found</h1>',
                     email=admin, status=404)
    steps = await pool.fetch("""
        SELECT "type"::text AS type, "output", "startTime" FROM "Step"
        WHERE "threadId" = $1 AND "type"::text IN ('user_message', 'assistant_message')
        ORDER BY "startTime", "createdAt" """, thread_id)
    owner = thread["email"] or "unknown user"
    msgs = "".join(
        f'<div class="msg {s["type"]}"><div class="who">'
        f'{_e(owner) if s["type"] == "user_message" else "ChitrapurGPT"}'
        f' <span class="muted">· {_fmt_time(s["startTime"])}</span></div>'
        f'<div class="text">{_e((s["output"] or "").strip())}</div></div>'
        for s in steps)
    back = f'/admin/user?{urllib.parse.urlencode({"email": access.normalize(owner)})}'
    body = f"""<p><a href="{_e(back)}">← All chats by {_e(owner)}</a></p>
<h1>{_e(thread["name"] or "(untitled chat)")}</h1>
<p class="muted">Started {_fmt_time(thread["createdAt"])} by {_e(owner)}. Read-only.</p>
{msgs or '<p class="muted">This chat has no messages.</p>'}"""
    return _page(thread["name"] or "Chat", body, email=admin)


@router.post("/admin/allow")
async def allow(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    form = await _form(request)
    if not hmac.compare_digest(form.get("csrf", ""), _csrf_token(admin)):
        return _back_to_dashboard("That form had expired. Please try again.", "bad")
    email = access.normalize(form.get("email"))
    if not access.is_valid_email(email):
        return _back_to_dashboard(f"“{email}” doesn't look like an email address. Nothing was changed.", "bad")
    if await access.add(email, added_by=admin):
        return _back_to_dashboard(f"Added {email}. They can sign in now.")
    return _back_to_dashboard(f"{email} could already sign in. Nothing was changed.")


@router.post("/admin/remove")
async def remove(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    form = await _form(request)
    if not hmac.compare_digest(form.get("csrf", ""), _csrf_token(admin)):
        return _back_to_dashboard("That form had expired. Please try again.", "bad")
    email = access.normalize(form.get("email"))
    if access.is_admin(email):
        return _back_to_dashboard(f"{email} is an admin, so they can't be removed here.", "bad")
    if await access.remove(email, removed_by=admin):
        return _back_to_dashboard(f"Removed {email}. They can no longer sign in. Their chats are kept.")
    return _back_to_dashboard(f"{email} wasn't on the list. Nothing was changed.")


def install(app):
    """Add the /admin pages (and admin_kb.py's collections and uploads) to Chainlit's FastAPI app, ahead of
    Chainlit's catch-all UI route."""
    import admin_kb
    before = list(app.router.routes)
    app.include_router(router)
    app.include_router(admin_kb.router)
    added = [r for r in app.router.routes if r not in before]
    app.router.routes[:] = added + before
