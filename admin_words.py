"""/admin/words: the Diacritical word list dashboard (for ChitrapurGPT admins: Shantish and Vaibhav), as an Excel round
trip.

One page (/admin/words): upload a document (or an offline word list) -> download its report (Excel); fill in the
decisions in Excel -> upload them -> a confirm page with one sentence and an Apply button; the rolling pending list
(download, fill in, upload the same way); History with undo; the whole list as Excel. Single words are changed on the
old Mappings page.

It works on the list chosen by DIACRITICAL_WORDS_LIST: "trial" (default, a copy; the translator never sees it) or
"live" (the real list). The logic is in diacritical/harvest.py (reading files, sorting words), diacritical/sheets.py
(the workbooks, reading them back, what Apply does) and diacritical/wordlist.py (the database). Same sign-in, admin
check and CSRF tokens as admin.py; plain HTML.
"""
from __future__ import annotations

import asyncio
import functools
import hmac
import logging
import re
import urllib.parse
from datetime import datetime

import psycopg
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response

from admin import IST, _csrf_token, _e, _fmt_time, _form, _page, _require_admin
from diacritical import harvest, sheets, wordlist

logger = logging.getLogger(__name__)
router = APIRouter()

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
HOME = "/admin/words"

CSS = """
body{font-size:18px}
main{max-width:1060px}
h1{font-size:28px}
h2{font-size:23px;margin:26px 0 8px}
.step h2{margin:0 0 8px}
.words-banner{border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:18px}
.words-banner.trial{background:#eaf3ff;border:2px solid #5b8fd6}
.words-banner.live{background:#fff4e0;border:2px solid #e8a33d}
.banner{font-size:19px}
.facts{font-size:19px;margin:0 0 16px}
.step{background:#fff;border:1px solid #dfe3e8;border-radius:10px;padding:16px 20px;margin:0 0 18px}
.step p{margin:4px 0 10px}
.dim{color:#5d6670;font-size:16px}
button,.button,input,select,textarea{font-family:inherit}
button,.button{font-size:18px;padding:8px 16px}
table.up .button{white-space:nowrap}
.big{font-size:20px !important;padding:10px 22px !important;font-weight:600}
input[type=file]{font-size:18px;margin:6px 0 10px;max-width:100%}
input[type=file]::file-selector-button{font-size:18px;padding:7px 14px;margin-right:12px;cursor:pointer}
table.up td{vertical-align:middle;font-size:18px}
table.up tr.new td{background:#f1f9f2}
td.st{min-width:190px}
.tag.new{background:#1e6b32;color:#fff}
.sentence{font-size:25px;font-weight:600;background:#fff;border:2px solid #1f3a5f;border-radius:10px;padding:16px 20px;margin:12px 0 14px}
.actions{display:flex;gap:14px;align-items:center;margin:0 0 22px}
.actions .primary{font-size:22px;padding:12px 40px}
.actions .cancel{font-size:19px;padding:10px 22px}
table.list td{font-size:18px;vertical-align:top}
td.w{font-weight:600;white-space:nowrap}
ul.cols{columns:3 250px;padding-left:20px;margin:8px 0}
ul.cols li{break-inside:avoid;margin:2px 0}
details.more{background:#fff;border:1px solid #dfe3e8;border-radius:8px;padding:10px 16px;margin:10px 0}
details.more>summary{font-size:19px}
.links{font-size:19px;margin:22px 0}
.links a{margin-right:22px}
"""


# ---------- helpers ----------

def _list() -> str:
    return wordlist.list_name()


def _list_label() -> str:
    return "the trial copy of the list" if _list() == "trial" else "the real list"


async def _db(fn):
    """Run fn(conn, list name) on the word-list database in a worker thread, in one transaction."""
    def run():
        with wordlist.connect() as conn:
            return fn(conn, wordlist.list_name())
    return await asyncio.to_thread(run)


def _banner() -> str:
    if _list() == "live":
        return ('<div class="words-banner live"><strong>This is the REAL word list:</strong> applied changes are used '
                'by the translator straight away.</div>')
    return ('<div class="words-banner trial"><strong>TRIAL:</strong> you are working on a copy of the word list. '
            'Nothing you do here changes the real translator.</div>')


def _flash_html(request: Request) -> str:
    msg = request.query_params.get("msg", "")
    if not msg:
        return ""
    kind = "bad" if request.query_params.get("kind") == "bad" else "ok"
    return f'<div class="banner {kind}">{_e(msg)}</div>'


def _words_page(request: Request, title: str, body: str, *, email: str, status: int = 200):
    top = f"<style>{CSS}</style>{_banner()}{_flash_html(request)}"
    return _page(title, top + body, email=email, banner=False, status=status)


def _back(path: str, message: str, kind: str = "ok", **extra) -> RedirectResponse:
    path, _, anchor = path.partition("#")
    joiner = "&" if "?" in path else "?"
    return RedirectResponse(f"{path}{joiner}" + urllib.parse.urlencode({"msg": message, "kind": kind, **extra})
                            + (f"#{anchor}" if anchor else ""), status_code=303)


def _safe(handler):
    """A friendly page instead of an error when the word list's database isn't set up or can't be reached."""
    @functools.wraps(handler)
    async def wrapped(request: Request, *args, **kwargs):
        try:
            return await handler(request, *args, **kwargs)
        except wordlist.NotConfigured as exc:
            logger.error("Word list dashboard: %s", exc)
            message = "The word list isn't set up here (DIACRITICAL_DATABASE_URL is missing)."
        except psycopg.OperationalError:
            logger.exception("Word list dashboard: database unreachable")
            message = "The word list's database can't be reached just now. Please try again in a minute."
        email, denied = _require_admin(request)
        if denied:
            return denied
        return _page("Diacritical words", f'<h1>Diacritical words</h1><div class="banner bad">{_e(message)}</div>',
                     email=email, banner=False, status=503)
    return wrapped


async def _checked(request: Request, back: str = HOME):
    """(admin email, form fields, None) or (None, None, response) for a POST from these pages."""
    admin, denied = _require_admin(request)
    if denied:
        return None, None, denied
    form = await _form(request)
    if not hmac.compare_digest(form.get("csrf", ""), _csrf_token(admin)):
        return None, None, _back(back, "That form had expired. Please try again.", "bad")
    return admin, form, None


def _hidden(csrf: str) -> str:
    return f'<input type="hidden" name="csrf" value="{csrf}">'


def _xlsx(data: bytes, filename: str) -> Response:
    return Response(data, media_type=XLSX, headers={"Content-Disposition": f'attachment; filename="{filename}"',
                                                    "Cache-Control": "no-store"})


def _today() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def _day(value) -> str:
    return _fmt_time(value).split(",")[0] if value else ""


def _shown(code: str | None) -> str:
    return _e(harvest.describe(code)) if code else '<span class="dim">–</span>'


def _status(r: dict) -> tuple[str, str]:
    """(status text, link to the confirm page or '') of an upload."""
    if r["status"] == "undone":
        return f"Applied on {_day(r['accepted_at'])}, then undone on {_day(r['undone_at'])}", ""
    if r["status"] == "applied":
        return f"Applied on {_day(r['accepted_at'])}", ""
    if r["status"] == "open" and r.get("decided"):
        return "Your decisions are uploaded, not applied yet", f"{HOME}/confirm/{r['id']}"
    if r["status"] == "open":
        return "Waiting for your decisions", ""
    return r["status"].capitalize(), ""


def _counts_text(kind: str, counts: dict) -> str:
    counts = counts or {}
    if kind == "document":
        parts = [(counts.get("clash", 0), "different from our list"), (counts.get("new", 0), "new"),
                 (counts.get("unmarked", 0), "without marks"), (counts.get("already", 0), "already in our list")]
    else:
        parts = [(counts.get("lnew", 0), "new"), (counts.get("lchanged", 0), "changed"),
                 (counts.get("lmissing", 0), "missing"), (counts.get("problems", 0), "problems")]
    return " · ".join(f"{n:,} {label}" for n, label in parts if n)


# ---------- the main page ----------

UPLOAD_JS = """(function(){
  document.querySelectorAll('form.wait').forEach(function(f){
    f.addEventListener('submit', function(){
      f.querySelector('button').disabled = true;
      var w = f.querySelector('.reading'); if (w) w.style.display = '';
    });
  });
})();"""


@router.get("/admin/words")
@_safe
async def home(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    counts, recent = await _db(lambda c, n: (wordlist.counts(c, n), wordlist.uploads(c, n)))
    csrf = _csrf_token(admin)
    new_id = request.query_params.get("new", "")

    def row(r):
        text, link = _status(r)
        files = ", ".join(f["name"] for f in r["files"])
        status = f'<a href="{link}">{_e(text)}: check and apply</a>' if link else _e(text)
        fresh = str(r["id"]) == new_id
        tag = '<span class="tag new">New</span> ' if fresh else ""
        return (f'<tr{" class=new" if fresh else ""}><td data-label="Date" style="white-space:nowrap">'
                f'{_fmt_time(r["created_at"])}</td>'
                f'<td data-label="Files">{tag}<strong>{_e(files)}</strong>'
                f'<br><span class="dim">{"a word list · " if r["kind"] == "list" else ""}'
                f'{_e(_counts_text(r["kind"], r["counts"]))}</span></td>'
                f'<td data-label="Status" class="st">{status}</td>'
                f'<td><a class="button{" big" if fresh else ""}" href="{HOME}/report/{r["id"]}.xlsx">'
                f'Download the report</a></td></tr>')
    shown, older = recent[:8], recent[8:]
    rows = "".join(row(r) for r in shown)
    more = ""
    if older:
        more = (f'<details><summary>Older uploads ({len(older)})</summary><div class="scroll"><table class="up">'
                + "".join(row(r) for r in older) + "</table></div></details>")
    reset = ""
    if _list() == "trial":
        reset = f"""<details style="margin-top:40px"><summary class="dim">Start the trial again</summary><div class="card">
<p>This makes the trial copy the same as the real list again. Changes made in the trial are lost (History keeps a
record of them). The pending list and your uploads are kept.</p>
<form method="post" action="{HOME}/reset-trial" onsubmit="return confirm('Start the trial again from the real list? The changes made in the trial copy will be lost.')">
{_hidden(csrf)}<button class="danger">Start the trial again from the real list</button></form></div></details>"""
    pending = counts["pending"]
    body = f"""<h1>Diacritical words</h1>
<p class="facts">The list has <strong>{counts['entries']:,}</strong> entries
({counts['words']:,} different words with marks) · <strong>{pending:,}</strong> {'word' if pending == 1 else 'words'}
on the pending list.</p>

<section class="step" id="upload">
<h2>Upload a document</h2>
<p>Word (.docx), PDF or text (.txt) files. Choose several at once to get one report for all of them. You get an Excel
report to download, with a column for your decisions.</p>
<form class="wait" method="post" action="{HOME}/upload" enctype="multipart/form-data">{_hidden(csrf)}
<input type="file" name="files" multiple required accept=".docx,.pdf,.txt,.xlsx"><br>
<button class="primary big">Make the report</button>
<span class="reading dim" style="display:none">Reading… a big PDF takes up to a minute. Please keep this page open.</span>
</form>
<p class="dim">A word list of your own (Excel with a plain word and a result column, or a .txt of “plain,result” lines)
can be uploaded here too: it is compared with ours entry by entry.</p>
</section>

<section class="step" id="uploads">
<h2>Your uploads</h2>
<div class="scroll"><table class="up"><tr class="head"><th>Date</th><th>Files</th><th>Status</th><th></th></tr>
{rows or '<tr><td colspan="4" class="dim">Nothing uploaded yet.</td></tr>'}</table></div>
{more}
</section>

<section class="step" id="decisions">
<h2>Upload your decisions</h2>
<p>The report, or the pending list, with your decisions filled in. You will see what will change, with an
<strong>Apply</strong> button, before anything in the list changes.</p>
<form class="wait" method="post" action="{HOME}/decisions" enctype="multipart/form-data">{_hidden(csrf)}
<input type="file" name="file" required accept=".xlsx"><br>
<button class="primary big">Read my decisions</button>
<span class="reading dim" style="display:none">Reading…</span>
</form>
</section>

<section class="step" id="pending">
<h2>Pending list ({pending:,} {'word' if pending == 1 else 'words'})</h2>
<p>Words waiting for an answer from the Diacriticals team. In the Excel file, choose <strong>Add</strong> (with the
agreed spelling in “Result”) or <strong>Remove from pending</strong>; rows left blank stay on the list. Then upload it
above, under “Upload your decisions”.</p>
<p><a class="button big" href="{HOME}/pending.xlsx">Download the pending list (Excel)</a></p>
</section>

<p class="links"><a href="{HOME}/history">History (undo a change)</a>
<a href="{HOME}/list.xlsx">Download the whole word list (Excel)</a></p>
<p class="dim">To change a single word, use the Mappings page of Diacritical.</p>
{reset}
<script>{UPLOAD_JS}</script>"""
    return _words_page(request, "Diacritical words", body, email=admin)


@router.get("/admin/words/list.xlsx")
@_safe
async def download_list(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    rows = await _db(wordlist.entries)
    data = await asyncio.to_thread(sheets.list_xlsx, rows)
    return _xlsx(data, f"diacritical-word-list-{_list()}-{_today()}.xlsx")


@router.post("/admin/words/reset-trial")
@_safe
async def reset_trial(request: Request):
    admin, form, denied = await _checked(request)
    if denied:
        return denied
    if _list() != "trial":
        return _back(HOME, "This only works on the trial copy.", "bad")
    await _db(lambda c, n: wordlist.reset_trial(c, admin))
    return _back(HOME, "The trial copy is the same as the real list again.")


# ---------- uploading a document -> its report ----------

def _file_name(f) -> str:
    return (getattr(f, "filename", None) or "").replace("\\", "/").rsplit("/", 1)[-1][:200]


@router.post("/admin/words/upload")
@_safe
async def upload(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    try:
        form = await request.form(max_files=50)
    except Exception:  # noqa: BLE001  (a broken or oversized upload)
        return _back(HOME, "The upload didn't arrive in one piece. Please try again.", "bad")
    if not hmac.compare_digest(str(form.get("csrf", "")), _csrf_token(admin)):
        return _back(HOME, "That form had expired. Please try again.", "bad")
    chosen = [f for f in form.getlist("files") if _file_name(f)]
    if len(chosen) == 1 and _file_name(chosen[0]).lower().endswith(".xlsx"):
        data = await chosen[0].read()
        if sheets.is_dashboard_file(data):          # a filled report put in the wrong box: read it as decisions
            return await _take_decisions(admin, _file_name(chosen[0]), data)
        await chosen[0].seek(0)
    documents, lists, problems = [], [], []
    for f in chosen:
        name = _file_name(f)
        if (getattr(f, "size", None) or 0) > harvest.MAX_BYTES:
            problems.append(f"{name}: this file is over {harvest.MAX_BYTES // 1024 // 1024} MB.")
            continue
        data = await f.read()
        if name.lower().endswith(".xlsx") and sheets.is_dashboard_file(data):
            problems.append(f"{name}: this is a report or pending list from this page; upload it on its own under "
                            "“Upload your decisions”.")
            continue
        try:
            read = await asyncio.to_thread(harvest.read_upload, name, data)
        except harvest.ReadProblem as exc:
            problems.append(f"{name}: {exc}")
            continue
        except Exception:  # noqa: BLE001
            logger.exception("Couldn't read %s", name)
            problems.append(f"{name}: this file couldn't be read.")
            continue
        (documents if read["kind"] == "document" else lists).append((name, len(data), read))
    if not documents and not lists:
        return _back(HOME + "#upload", " ".join(problems) or "Please choose a file.", "bad")

    def make(conn, n):
        current = wordlist.mappings(conn, n)
        made = []
        if documents:
            review = harvest.review_document([(name, r["pages"]) for name, _, r in documents], current)
            files = [{"name": name, "size": size, "kind": "document", "pages": len(r["pages"])}
                     for name, size, r in documents]
            made.append(wordlist.create_review(conn, n, "document", ", ".join(f["name"] for f in files), files, admin,
                                               review))
        for name, size, r in lists:
            review = harvest.review_list(r["rows"], r["problems"], current)
            made.append(wordlist.create_review(conn, n, "list", name, [{"name": name, "size": size, "kind": "list",
                                                                        "rows": len(r["rows"])}], admin, review))
        return made
    made = await _db(make)
    note = (" Not read: " + " ".join(problems)) if problems else ""
    ready = "Your report is ready" if len(made) == 1 else f"Your {len(made)} reports are ready"
    return _back(HOME + "#uploads", f"{ready}: download it below.{note}", "bad" if problems else "ok",
                 new=str(made[0]))


def _report_info(rv: dict) -> dict:
    text, _ = _status(rv | {"decided": bool(rv["decisions"])})
    if rv["status"] in ("applied", "undone"):
        text += f" by {rv['accepted_by']}"
    return {"review": rv["id"], "list": rv["list_name"], "list_label": _list_label(),
            "files": [f["name"] for f in rv["files"]], "made": _fmt_time(rv["created_at"]), "by": rv["created_by"],
            "status": text, "made_iso": rv["created_at"].isoformat(timespec="seconds")}


@router.get("/admin/words/report/{review_id}.xlsx")
@_safe
async def report(request: Request, review_id: int):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    rv = await _db(lambda c, n: wordlist.get_review(c, n, review_id))
    if rv is None or rv["kind"] not in ("document", "list"):
        return Response("Not found", status_code=404)
    cells = (rv["decisions"] or {}).get("cells")
    data = await asyncio.to_thread(sheets.report_xlsx, rv["items"], _report_info(rv), cells)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", rv["files"][0]["name"].rsplit(".", 1)[0])[:60] if rv["files"] else "upload"
    return _xlsx(data, f"word-check-{review_id}-{stem}.xlsx")


# ---------- uploading decisions -> the confirm page -> Apply ----------

async def _take_decisions(admin: str, name: str, data: bytes):
    try:
        filled = await asyncio.to_thread(sheets.read_filled, name, data)
    except harvest.ReadProblem as exc:
        return _back(HOME + "#decisions", str(exc), "bad")

    def store(conn, n):
        if filled["kind"] == "report":
            rid = wordlist.store_report_answers(conn, n, filled, admin, name)
            return rid, None
        rid = wordlist.store_pending_answers(conn, n, filled, admin, name)
        p = wordlist.plan_for(conn, n, wordlist.get_review(conn, n, rid))
        if not any(p[k] for k in ("add", "update", "delete", "remove", "notes", "unclear", "skipped", "resolve")):
            wordlist.cancel(conn, n, rid)
            return None, "Nothing to apply: no row of the pending list says Add or Remove from pending."
        return rid, None
    try:
        rid, nothing = await _db(store)
    except wordlist.WordListError as exc:
        return _back(HOME + "#decisions", str(exc), "bad")
    if rid is None:
        return _back(HOME + "#pending", nothing)
    return RedirectResponse(f"{HOME}/confirm/{rid}", status_code=303)


@router.post("/admin/words/decisions")
@_safe
async def decisions(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    try:
        form = await request.form(max_files=2)
    except Exception:  # noqa: BLE001
        return _back(HOME + "#decisions", "The upload didn't arrive in one piece. Please try again.", "bad")
    if not hmac.compare_digest(str(form.get("csrf", "")), _csrf_token(admin)):
        return _back(HOME + "#decisions", "That form had expired. Please try again.", "bad")
    f = form.get("file")
    if not _file_name(f):
        return _back(HOME + "#decisions", "Please choose the file with your decisions.", "bad")
    if (getattr(f, "size", None) or 0) > harvest.MAX_BYTES:
        return _back(HOME + "#decisions", "This file is too big to be a report.", "bad")
    return await _take_decisions(admin, _file_name(f), await f.read())


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f'<td data-label="{headers[i]}"{" class=w" if i == 0 else ""}>{v}</td>'
                                    for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="list"><tr class="head">{head}</tr>{body}</table></div>'


def _where(x: dict) -> str:
    return _e(f"{x['sheet']}, row {x['row']}") if x.get("sheet") else ""


def _confirm_body(rv: dict, p: dict, csrf: str) -> str:
    answers = rv["decisions"] or {}
    pending_upload = rv["kind"] == "pending"
    if pending_upload:
        source = f"the pending list <strong>{_e(answers.get('file', ''))}</strong>"
    else:
        source = (f"<strong>{_e(answers.get('file', ''))}</strong>, the report for "
                  f"{_e(', '.join(f['name'] for f in rv['files']))} (uploaded {_fmt_time(rv['created_at'])})")
    parts = []
    unclear = p["unclear"]
    if unclear:
        what = "stay on the pending list" if pending_upload else "go to the pending list"
        parts.append(f'<div class="banner warn">{len(unclear)} {"row" if len(unclear) == 1 else "rows"} couldn’t be '
                     f'read: {"it" if len(unclear) == 1 else "they"} {what}. See the list below.</div>')
    if p["skipped"]:
        one = len(p["skipped"]) == 1
        parts.append(f'<div class="banner warn">{len(p["skipped"])} {"word was" if one else "words were"} changed in '
                     f'the list after this file was made, so {"it is" if one else "they are"} skipped. See the list '
                     'below.</div>')
    buttons = f"""<form method="post" action="{HOME}/confirm/{rv['id']}" class="actions">{_hidden(csrf)}
<input type="hidden" name="fingerprint" value="{p['fingerprint']}">
<button class="primary" name="action" value="apply">Apply</button>
<button class="cancel" name="action" value="cancel">Cancel</button></form>"""
    out = [f"""<p><a href="{HOME}">← Back</a></p><h1>Check before applying</h1>
<p>Your decisions from {source}.</p>
<div class="sentence">{_e(sheets.sentence(p))}</div>{''.join(parts)}{buttons}"""]
    if p["add"]:
        out.append(f"<h2>Words added ({len(p['add'])})</h2>" + _table(
            ["Plain word", "In the list it will be"], [[_e(c["plain"]), _shown(c["result"])] for c in p["add"]]))
    if p["update"]:
        out.append(f"<h2>Words changed ({len(p['update'])})</h2>" + _table(
            ["Plain word", "Now", "After"], [[_e(c["plain"]), _shown(c["old"]), _shown(c["result"])]
                                             for c in p["update"]]))
    if p["delete"]:
        out.append(f"<h2>Words deleted ({len(p['delete'])})</h2>" + _table(
            ["Plain word", "Now"], [[_e(c["plain"]), _shown(c["old"])] for c in p["delete"]]))
    if unclear:
        out.append(f"<h2>Couldn’t read ({len(unclear)})</h2><p class=\"dim\">Treated as Pending: "
                   + ("they stay on the pending list." if pending_upload else "they go to the pending list.")
                   + " Upload the file again after correcting them if you like.</p>" + _table(
                       ["Word", "Where", "What it says", "Why"],
                       [[_e(x.get("word") or x.get("plain")), _where(x), _e(x.get("typed") or ""), _e(x["why"])]
                        for x in unclear]))
    if p["skipped"]:
        out.append(f"<h2>Skipped: changed in the list since the file was made ({len(p['skipped'])})</h2>" + _table(
            ["Plain word", "Then", "Now", "Your decision wanted"],
            [[_e(x["plain"]), _shown(x["then"]), _shown(x["now"]),
              _e(x.get("why")) if x.get("why") else _shown(x["wanted"])] for x in p["skipped"]]))

    def listed(title, items, note=""):
        if not items:
            return ""
        return (f'<details class="more"><summary>{_e(title)} ({len(items)})</summary>{note}<ul class="cols">'
                + "".join(f"<li>{x}</li>" for x in items) + "</ul></details>")
    out.append(listed("Put on the pending list", [f"{_e(x['plain'])} → {_shown(x['suggested'])}"
                                                  for x in p["pending_new"]]))
    already = [x for x in p["pending"] if x not in p["pending_new"]]
    out.append(listed("Already on the pending list (left as they are)", [_e(x["plain"]) for x in already]))
    if p.get("remove"):
        out.append(f'<details class="more"><summary>Removed from the pending list ({len(p["remove"])})</summary>'
                   f'<p class="dim">Taken off the list without changing it.</p></details>')
    if p["pending_off"]:
        out.append(f'<p class="dim">{p["pending_off"]} {"word comes" if p["pending_off"] == 1 else "words come"} off '
                   'the pending list because they are added, changed, confirmed or removed here.</p>')
    if pending_upload and p.get("stay"):
        out.append(f'<p class="dim">{p["stay"]} {"row stays" if p["stay"] == 1 else "rows stay"} on the pending list.'
                   + (f' Notes changed on {len(p["notes"])}.' if p.get("notes") else "") + "</p>")
    elif p.get("notes"):
        out.append(f'<p class="dim">Notes changed on {len(p["notes"])} pending words.</p>')
    if p["words"]:
        out.append(f'<details class="more"><summary>Read from your own words ({len(p["words"])})</summary>' + _table(
            ["Word", "Where", "You wrote", "Read as"],
            [[_e(x["word"]), _where(x), _e(x["typed"]), _e(x["read"])] for x in p["words"]]) + "</details>")
    if p.get("missing"):
        out.append(listed("Not in your file (rows deleted?): they go to the pending list",
                          [_e(x) for x in p["missing"]]))
    if len(p["add"]) + len(p["update"]) + len(p["delete"]) + len(unclear) + len(p["skipped"]) > 12:
        out.append(buttons)
    return "".join(out)


@router.get("/admin/words/confirm/{review_id}")
@_safe
async def confirm_page(request: Request, review_id: int):
    admin, denied = _require_admin(request)
    if denied:
        return denied

    def load(conn, n):
        rv = wordlist.get_review(conn, n, review_id)
        if rv is None or rv["status"] != "open" or not rv["decisions"]:
            return rv, None
        return rv, wordlist.plan_for(conn, n, rv)
    rv, p = await _db(load)
    if rv is None:
        return _back(HOME, "That upload wasn't found.", "bad")
    if p is None:
        if rv["status"] in ("applied", "undone"):
            return _back(HOME, wordlist.applied_note(rv).split(". To")[0] + ".")
        return _back(HOME + "#decisions", "There are no decisions waiting here: upload your filled report first.",
                     "bad")
    body = _confirm_body(rv, p, _csrf_token(admin))
    return _words_page(request, "Check before applying", body, email=admin)


@router.post("/admin/words/confirm/{review_id}")
@_safe
async def confirm_post(request: Request, review_id: int):
    back = f"{HOME}/confirm/{review_id}"
    admin, form, denied = await _checked(request, back)
    if denied:
        return denied
    if form.get("action") == "cancel":
        await _db(lambda c, n: wordlist.cancel(c, n, review_id))
        return _back(HOME, "Cancelled: nothing was changed.")
    try:
        done = await _db(lambda c, n: wordlist.apply(c, n, review_id, admin, form.get("fingerprint", "")))
    except wordlist.Changed as exc:
        return _back(back, str(exc), "bad")
    except wordlist.WordListError as exc:
        return _back(HOME, str(exc), "bad")
    said = sheets.sentence(done["plan"], done=True)
    undo = " You can undo it on the History page." if any(done["plan"][k] for k in ("add", "update", "delete")) else ""
    return _back(HOME, said + undo)


# ---------- pending ----------

@router.get("/admin/words/pending.xlsx")
@_safe
async def pending_download(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    items, current = await _db(lambda c, n: (wordlist.pending(c, n), wordlist.mappings(c, n)))
    now = datetime.now(IST)
    info = {"list": _list(), "when": _fmt_time, "made_iso": now.isoformat(timespec="seconds"), "by": admin}
    data = await asyncio.to_thread(sheets.pending_xlsx, items, current, info)
    return _xlsx(data, f"diacritical-pending-{_list()}-{_today()}.xlsx")


# ---------- history ----------

KINDS = {"edit": "Edit", "undo": "Undo", "reset": "Trial restarted"}   # reports and pending lists say so themselves


@router.get("/admin/words/history")
@_safe
async def history(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied

    def load(conn, n):
        rows = wordlist.batches(conn, n)
        return rows, {b["id"]: wordlist.changes(conn, b["id"]) for b in rows[:60]}
    rows, changes = await _db(load)
    csrf = _csrf_token(admin)
    last_reset = max((b["id"] for b in rows if b["kind"] == "reset"), default=0)
    ops = {"add": "Added", "update": "Changed", "delete": "Deleted"}

    def what(b):
        return ", ".join(f"{b[k]} {label}" for k, label in (("adds", "added"), ("updates", "changed"),
                                                             ("deletes", "deleted")) if b[k])

    def tag(b):
        return f'<span class="tag admin">{_e(KINDS[b["kind"]])}</span> ' if b["kind"] in KINDS else ""

    def row(b):
        n = b["adds"] + b["updates"] + b["deletes"]
        undo = ""
        if b["undone_at"]:
            undo = (f'<span class="tag off">Undone</span><br><span class="dim" style="white-space:nowrap">'
                    f'{_fmt_time(b["undone_at"])}</span>')
        elif n and b["kind"] not in ("undo", "reset") and b["id"] > last_reset:
            undo = (f'<form class="inline" method="post" action="{HOME}/history/{b["id"]}/undo" '
                    f'onsubmit="return confirm(\'Undo this change? The words go back to how they were before it, and '
                    f'the pending words it answered go back on the pending list.\')">'
                    f'{_hidden(csrf)}<button>Undo</button></form>')
        detail = ""
        if n and b["id"] in changes:
            detail = (f'<details><summary class="dim">{_e(what(b))}: see the words</summary>' + _table(
                ["Plain word", "What", "Before", "After"],
                [[_e(c["new_plain"] or c["old_plain"]), ops[c["op"]], _shown(c["old_result"]),
                  _shown(c["new_result"])] for c in changes[b["id"]]]) + "</details>")
        report = (f' · <a href="{HOME}/report/{b["review_id"]}.xlsx">the report</a>'
                  if b["review_id"] and b["kind"] == "review" else "")
        return (f'<tr{" class=off" if b["undone_at"] else ""}><td data-label="When" style="white-space:nowrap">'
                f'{_fmt_time(b["created_at"])}<br><span class="dim">{_e(b["created_by"])}</span></td>'
                f'<td>{tag(b)}{_e(b["summary"])}'
                f'{report}{detail}</td><td>{undo}</td></tr>')
    body = f"""<p><a href="{HOME}">← Back</a></p><h1>History</h1>
<p>Every change made here, newest first. Undo puts the words back as they were before that change (it can't if they
were changed again since), and the pending words it answered go back on the pending list. Words it put on the
pending list stay there.</p>
<div class="scroll"><table class="list"><tr class="head"><th>When</th><th>What</th><th></th></tr>
{''.join(row(b) for b in rows) or '<tr><td colspan="3" class="dim">No changes yet.</td></tr>'}</table></div>"""
    return _words_page(request, "History", body, email=admin)


@router.post("/admin/words/history/{batch_id}/undo")
@_safe
async def undo(request: Request, batch_id: int):
    admin, form, denied = await _checked(request, f"{HOME}/history")
    if denied:
        return denied
    try:
        new = await _db(lambda c, n: wordlist.undo(c, n, batch_id, admin))
    except wordlist.WordListError as exc:
        return _back(f"{HOME}/history", str(exc), "bad")
    return _back(f"{HOME}/history", f"Undone (recorded as change {new}).")
