"""/admin/words: the Diacritical word list dashboard (for ChitrapurGPT admins: Shantish and Vaibhav).

Pages: the word list (search, add, edit, download), Add words (upload a document or an offline list, which makes a
review), the review (one choice per row, then Accept), Pending (words waiting for the Diacriticals team), History
(every change, with undo) and Try it (the translator on the list being worked on).

It works on the list chosen by DIACRITICAL_WORDS_LIST: "trial" (default, a copy; the translator never sees it) or
"live" (the real list). The logic is in diacritical/harvest.py (reading files, sorting words, what Accept does) and
diacritical/wordlist.py (the database). Same sign-in, admin check and CSRF tokens as admin.py; plain HTML, with a
little JavaScript only for conveniences (every page works without it).
"""
from __future__ import annotations

import asyncio
import functools
import hmac
import json
import logging
import re
import urllib.parse
from datetime import datetime

import psycopg
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response

from admin import IST, _csrf_token, _e, _fmt_time, _form, _page, _require_admin
from diacritical import harvest, wordlist
from diacritical.cleaning import clean_plain
from diacritical.diacritics import translate_text

logger = logging.getLogger(__name__)
router = APIRouter()

PER_PAGE = 50
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

CSS = """
body{font-family:"Noto Sans","Segoe UI",Arial,system-ui,sans-serif;font-size:18px}
.words-banner{border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:18px}
.words-banner.trial{background:#eaf3ff;border:2px solid #5b8fd6}
.words-banner.live{background:#fff4e0;border:2px solid #e8a33d}
nav.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 18px}
nav.tabs a{padding:8px 16px;border-radius:20px;background:#fff;border:1px solid #c9d1db;text-decoration:none;color:#1d2433;font-size:17px}
nav.tabs a.on{background:#1f3a5f;border-color:#1f3a5f;color:#fff}
.w{font-size:21px;font-weight:600}
.dim{color:#5d6670;font-size:16px}
.line{background:#f7f8fa;border-left:4px solid #c9d1db;padding:6px 10px;margin:6px 0;font-size:17px;overflow-wrap:anywhere}
mark{background:#ffe58a;padding:0 2px;border-radius:3px}
.pick{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}
.pick .lbl{color:#5d6670;font-size:16px;margin-right:4px;min-width:70px}
.pick label{border:1px solid #9aa4b0;border-radius:7px;padding:5px 12px;cursor:pointer;background:#fff;white-space:nowrap;font-size:17px}
.pick label:has(input:checked){background:#dce9fb;border-color:#1f6feb;font-weight:600}
.pick input{margin:0 7px 0 0;transform:scale(1.25);vertical-align:-1px}
.pick input:disabled+span{color:#555}
.item{background:#fff;border:1px solid #dfe3e8;border-left:5px solid #dfe3e8;border-radius:8px;padding:12px 16px;margin:10px 0}
.item.todo,tr.todo td:first-child{border-left-color:#e8a33d}
tr.todo td:first-child{border-left:5px solid #e8a33d}
.item .head{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline}
.note{color:#8a5a00;font-size:16px}
.changed{color:#a12622;font-weight:600}
details.sec{background:#fff;border:1px solid #dfe3e8;border-radius:8px;padding:10px 16px;margin:14px 0}
details.sec>summary{font-size:21px;padding:4px 0}
details.sec[open]>summary{margin-bottom:8px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 16px}
.chip{background:#fff;border:1px solid #c9d1db;border-radius:20px;padding:6px 14px;text-decoration:none;color:#1d2433;font-size:17px}
.chip b{font-size:19px}
.actions{background:#fff;border:2px solid #1f3a5f;border-radius:8px;padding:12px 16px;margin:14px 0}
.actions p{margin:4px 0 10px}
.actions button{margin:4px 8px 4px 0}
button,.button{font-size:17px;padding:7px 16px}
button:disabled{opacity:.45;cursor:not-allowed}
.big{font-size:19px}
table.words td,table.words th{font-size:18px}
table.list td{vertical-align:top}
.extras{margin-top:6px;font-size:16px}
td.choose{min-width:360px}
td.acts{min-width:200px}
td.acts .button,td.acts button{white-space:nowrap}
td.choose .pick{margin:2px 0 6px}
.extras label{display:block;margin:3px 0;cursor:pointer}
ul.cols{columns:3 260px;padding-left:18px;margin:6px 0}
ul.cols li{break-inside:avoid;margin:2px 0}
.form-row{margin:14px 0}
.form-row label.t{display:block;font-weight:600;margin-bottom:4px}
.form-row input[type=text],.form-row textarea{font-size:20px;padding:8px 10px;width:420px;max-width:100%}
.types label{display:block;margin:6px 0;cursor:pointer}
.types input{transform:scale(1.25);margin-right:8px}
.preview{background:#f3f8f1;border:1px solid #9ccfa6;border-radius:8px;padding:10px 16px;margin:12px 0}
.preview table{width:auto;border:none;background:none;margin:6px 0}
.preview td{border:none;padding:4px 14px 4px 0;font-size:19px}
input,textarea,button,select{font-family:inherit}
input[type=file]{font-size:18px}
input[type=file]::file-selector-button{font-size:18px;padding:6px 14px;margin-right:10px;cursor:pointer}
.pager{margin:12px 0}
.pager a,.pager span{margin-right:12px}
textarea.try{width:100%;font-size:19px;padding:10px}
.out{white-space:pre-wrap;background:#fff;border:1px solid #9ccfa6;border-radius:8px;padding:12px 16px;font-size:20px;overflow-wrap:anywhere}
@media (max-width:640px){.pick label{white-space:normal}.form-row input[type=text]{width:100%}}
"""

TABS = (("list", "/admin/words", "Word list"), ("add", "/admin/words/add", "Add words"),
        ("pending", "/admin/words/pending", "Pending"), ("history", "/admin/words/history", "History"),
        ("try", "/admin/words/try", "Try it"))


# ---------- helpers ----------

def _list() -> str:
    return wordlist.list_name()


async def _db(fn):
    """Run fn(conn, list name) on the word-list database in a worker thread, in one transaction."""
    def run():
        with wordlist.connect() as conn:
            return fn(conn, wordlist.list_name())
    return await asyncio.to_thread(run)


def _banner() -> str:
    if _list() == "live":
        return ('<div class="words-banner live"><strong>This is the REAL word list:</strong> accepted changes are used '
                'by the translator straight away.</div>')
    return ('<div class="words-banner trial"><strong>TRIAL:</strong> you are working on a copy of the word list. '
            'Nothing you do here changes the real translator.</div>')


def _flash_html(request: Request) -> str:
    msg = request.query_params.get("msg", "")
    if not msg:
        return ""
    kind = "bad" if request.query_params.get("kind") == "bad" else "ok"
    return f'<div class="banner {kind}">{_e(msg)}</div>'


def _words_page(request: Request, title: str, body: str, *, email: str, tab: str = "", pending: int | None = None,
                status: int = 200, wide: bool = False, flash: str = ""):
    tabs = "".join(f'<a href="{url}"{" class=on" if key == tab else ""}>{label}'
                   f'{f" ({pending})" if key == "pending" and pending is not None else ""}</a>'
                   for key, url, label in TABS)
    top = (f"<style>{CSS}</style>{_banner()}<nav class=\"tabs\">{tabs}</nav>"
           f"{flash or _flash_html(request)}")
    return _page(title, top + body, email=email, banner=False, status=status, wide=wide)


def _back(path: str, message: str, kind: str = "ok") -> RedirectResponse:
    path, _, anchor = path.partition("#")
    joiner = "&" if "?" in path else "?"
    return RedirectResponse(f"{path}{joiner}" + urllib.parse.urlencode({"msg": message, "kind": kind})
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


async def _checked(request: Request, back: str = "/admin/words"):
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


def _times(n: int) -> str:
    return f"{n} time{'s' if n != 1 else ''}"


def _highlight(line: str, word: str) -> str:
    """The example line, escaped, with the word marked."""
    if not word:
        return _e(line)
    parts = re.split(f"({re.escape(word)})", line)
    return "".join(f"<mark>{_e(p)}</mark>" if i % 2 else _e(p) for i, p in enumerate(parts))


def _pages(where: str) -> str:
    if not where:
        return ""
    return (" · pages " if "," in where else " · page ") + _e(where)


def _shown(code: str | None) -> str:
    return _e(harvest.describe(code)) if code else '<span class="dim">–</span>'


# ---------- the word list ----------

@router.get("/admin/words")
@_safe
async def word_list(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    q = (request.query_params.get("q") or "").strip()
    page = request.query_params.get("page", "1")
    page = max(1, int(page)) if page.isdigit() else 1
    rows, counts = await _db(lambda c, n: (wordlist.entries(c, n), wordlist.counts(c, n)))
    if q:
        needle = clean_plain(q)
        rows = [r for r in rows if needle in r["plain_text"]
                or needle in clean_plain(re.sub(r"[{}]", " ", r["diacritic_text"]))]
        rows.sort(key=lambda r: (not r["plain_text"].startswith(needle), r["plain_text"]))
    pages = max(1, -(-len(rows) // PER_PAGE))
    page = min(page, pages)
    shown = rows[(page - 1) * PER_PAGE:page * PER_PAGE]

    def url(p):
        return "/admin/words?" + urllib.parse.urlencode({**({"q": q} if q else {}), "page": p})
    body_rows = "".join(
        f'<tr><td data-label="Plain word" class="big">{_e(r["plain_text"])}</td>'
        f'<td data-label="Result" class="big" title="stored as {_e(r["diacritic_text"])}">'
        f'{_e(harvest.describe(r["diacritic_text"]))}</td>'
        f'<td><a class="button" href="/admin/words/edit?id={r["id"]}">Edit</a></td></tr>' for r in shown)
    pager = ""
    if pages > 1:
        pager = (f'<p class="pager">Page {page} of {pages} · '
                 + (f'<a href="{_e(url(page - 1))}">‹ Previous</a>' if page > 1 else "")
                 + (f'<a href="{_e(url(page + 1))}">Next ›</a>' if page < pages else "") + "</p>")
    found = (f'<p>{len(rows):,} {"entry matches" if len(rows) == 1 else "entries match"} “{_e(q)}”. '
             f'<a href="/admin/words">Show all</a></p>') if q else ""
    reset = ""
    if _list() == "trial":
        reset = f"""<details style="margin-top:30px"><summary>Start the trial again</summary><div class="card">
<p>This makes the trial copy the same as the real list again. Changes made in the trial are lost (History keeps a
record of them). The pending list and your reviews are kept.</p>
<form method="post" action="/admin/words/reset-trial" onsubmit="return confirm('Start the trial again from the real list? The changes made in the trial copy will be lost.')">
{_hidden(_csrf_token(admin))}<button class="danger">Start the trial again from the real list</button></form></div></details>"""
    body = f"""<h1>Diacritical words</h1>
<div class="cards">
<div class="stat"><div class="n">{counts['entries']:,}</div><div class="l">entries in the list</div></div>
<div class="stat"><div class="n">{counts['words']:,}</div><div class="l">different diacritical words</div></div>
<div class="stat"><div class="n"><a href="/admin/words/pending">{counts['pending']:,}</a></div><div class="l">words pending</div></div>
</div>
<form method="get" action="/admin/words" style="margin:0 0 10px">
<input type="search" name="q" value="{_e(q)}" placeholder="Find a word (with or without marks)" style="font-size:18px;width:380px">
<button>Find</button></form>
<p><a class="button" href="/admin/words/new">Add a word</a>
<a class="button" href="/admin/words/list.xlsx">Download the whole list (Excel)</a></p>
{found}
<div class="scroll"><table class="words"><tr class="head"><th>Plain word</th><th>Result</th><th></th></tr>
{body_rows or '<tr><td colspan="3" class="dim">No words found.</td></tr>'}</table></div>
{pager}
<p class="dim">The result shows what the translator gives: “(always capital)” means it always starts with a capital
letter; “as a name” is what it gives when the word is typed with a capital; “choice” shows every spelling.</p>
{reset}"""
    return _words_page(request, "Diacritical words", body, email=admin, tab="list", pending=counts["pending"])


@router.get("/admin/words/list.xlsx")
@_safe
async def download_list(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    rows = await _db(wordlist.entries)
    data = await asyncio.to_thread(harvest.list_xlsx, rows)
    return _xlsx(data, f"diacritical-word-list-{_list()}-{_today()}.xlsx")


@router.post("/admin/words/reset-trial")
@_safe
async def reset_trial(request: Request):
    admin, form, denied = await _checked(request)
    if denied:
        return denied
    if _list() != "trial":
        return _back("/admin/words", "This only works on the trial copy.", "bad")
    await _db(lambda c, n: wordlist.reset_trial(c, admin))
    return _back("/admin/words", "The trial copy is the same as the real list again.")


# ---------- add or change one word ----------

async def _entry_form(request: Request, admin: str, *, entry: dict | None, plain: str, boxes: dict,
                      pending: dict | None, problems: list[str], show_preview: bool, expected: tuple[str, str] | None,
                      status: int = 200, other: dict | None = None):
    csrf = _csrf_token(admin)
    code = harvest.boxes_to_code(boxes) if not problems else ""
    kind = boxes.get("type") or "normal"
    preview = ""
    if show_preview and code and plain:
        small, capital = harvest.preview(clean_plain(plain), code)
        typed = clean_plain(plain)
        preview = f"""<div class="preview"><strong>What the translator will give</strong><table>
<tr><td class="dim">typed</td><td>{_e(typed)}</td><td>→</td><td><strong>{_e(small)}</strong></td></tr>
<tr><td class="dim">typed</td><td>{_e(harvest.typed_with_capital(typed))}</td><td>→</td><td><strong>{_e(capital)}</strong></td></tr>
</table><span class="dim">stored as {_e(code)}</span></div>"""
    types = "".join(
        f'<label><input type="radio" name="type" value="{k}"{" checked" if kind == k else ""}>'
        f'<strong>{_e(label)}</strong> <span class="dim">{_e(hint)}</span></label>'
        for k, label, hint in (
            ("normal", "Normal", "follows how it's typed: krishna → kṛs͟hṇa, Krishna → Kṛs͟hṇa"),
            ("capital", "Always capital", "for places and names: always starts with a capital letter"),
            ("name", "Name", "different when typed with a capital, e.g. god → gŏḍ but God → God"),
            ("choices", "Choices", "two or more spellings, all shown")))
    from_pending = ""
    if pending:
        from_pending = f"""<div class="card" style="margin-bottom:12px"><strong>From the pending list:</strong>
{_e(pending['plain_text'])} → {_shown(pending['suggested'])}<br><span class="dim">{_e(pending['reason'])}
{(' · ' + _e(pending['source'])) if pending['source'] else ''}</span>
{f'<div class="line">{_highlight(pending["context"], "")}</div>' if pending['context'] else ''}
<span class="dim">Saving takes it off the pending list.</span></div>"""
    errors = "".join(f"<li>{_e(p)}</li>" for p in problems)
    if other:  # the plain word belongs to another entry
        errors += (f'<li>“{_e(other["plain_text"])}” is already in the list as '
                   f'{_e(harvest.describe(other["diacritic_text"]))}. '
                   f'<a href="/admin/words/edit?id={other["id"]}">Open that entry</a></li>')
    title = "Change a word" if entry else "Add a word"
    advanced = boxes.get("advanced") or ""
    delete = ""
    if entry:
        delete = f"""<form method="post" action="/admin/words/delete" style="margin-top:24px"
data-plain="{_e(entry['plain_text'])}" onsubmit="return confirm('Delete ' + this.dataset.plain + ' from the list?')">{_hidden(csrf)}
<input type="hidden" name="id" value="{entry['id']}">
<input type="hidden" name="expected_plain" value="{_e(entry['plain_text'])}">
<input type="hidden" name="expected_result" value="{_e(entry['diacritic_text'])}">
<button class="danger">Delete this word</button></form>"""
    body = f"""<p><a href="/admin/words">← Word list</a></p><h1>{title}</h1>
{from_pending}
{f'<div class="banner bad"><ul style="margin:0">{errors}</ul></div>' if errors else ''}
<form method="post" action="/admin/words/save" class="entry" id="entry">{_hidden(csrf)}
<input type="hidden" name="id" value="{entry['id'] if entry else ''}">
<input type="hidden" name="pending" value="{pending['id'] if pending else ''}">
<input type="hidden" name="expected_plain" value="{_e(expected[0]) if expected else ''}">
<input type="hidden" name="expected_result" value="{_e(expected[1]) if expected else ''}">
<div class="form-row"><label class="t" for="plain">Plain word</label>
<span class="dim">as people type it, without marks (two words or a hyphen are fine, e.g. laddu-s)</span><br>
<input type="text" id="plain" name="plain" value="{_e(plain)}" required autocomplete="off"></div>
{'<div class="banner warn">This entry is stored in a special way the boxes below can’t show (for example always '
  'capital with choices). Change it in the <strong>Stored code</strong> box further down, or empty that box to use '
  'the boxes instead.</div>' if advanced else ''}
<div id="simple">
<div class="form-row types"><span class="t" style="font-weight:600">Type</span>{types}</div>
<div class="form-row" data-for="normal capital name"><label class="t" for="result">Result</label>
<span class="dim">the word with its marks</span><br>
<input type="text" id="result" name="result" value="{_e(boxes.get('result', ''))}" autocomplete="off"></div>
<div class="form-row" data-for="name"><label class="t" for="name">As a name</label>
<span class="dim">what it gives when typed with a capital, e.g. God</span><br>
<input type="text" id="name" name="name" value="{_e(boxes.get('name', ''))}" autocomplete="off"></div>
<div class="form-row" data-for="choices"><label class="t" for="choices">Choices</label>
<span class="dim">one spelling per line; a label in brackets is fine, e.g. nāmāvalī (hindi)</span><br>
<textarea id="choices" name="choices" rows="4">{_e(chr(10).join(boxes.get('choices') or []))}</textarea></div>
</div>
<details{' open' if advanced else ''} class="form-row"><summary>Advanced: stored code</summary>
<p class="dim">{'This entry is stored in a way the boxes above can’t show, so it is shown here. ' if advanced else ''}When
this box is filled in, it is saved exactly as written and the boxes above are ignored. Empty it to use the boxes.</p>
<input type="text" id="advanced" name="advanced" value="{_e(advanced)}" style="width:420px;max-width:100%;font-size:19px"></details>
{preview}
<p><button name="action" value="preview">Preview</button>
<button class="primary" name="action" value="save">Save</button></p>
</form>
{delete}
<script>(function(){{
  var f = document.getElementById('entry');
  function show(){{
    var t = (f.querySelector('input[name=type]:checked') || {{}}).value || 'normal';
    f.querySelectorAll('[data-for]').forEach(function(el){{
      el.style.display = el.getAttribute('data-for').split(' ').indexOf(t) < 0 ? 'none' : '';
    }});
  }}
  var adv = document.getElementById('advanced'), simple = document.getElementById('simple');
  function dim(){{ simple.style.opacity = adv.value.trim() ? '0.4' : ''; }}
  f.addEventListener('change', show); adv.addEventListener('input', dim); show(); dim();
}})();</script>"""
    return _words_page(request, title, body, email=admin, tab="list", status=status,
                       pending=await _db(wordlist.pending_count))


async def _pending_for(pid: str | None):
    if not pid or not pid.isdigit():
        return None
    return await _db(lambda c, n: wordlist.pending_item(c, n, int(pid)))


@router.get("/admin/words/new")
@_safe
async def new_word(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    pending = await _pending_for(request.query_params.get("pending"))
    plain, boxes = request.query_params.get("plain", ""), harvest.code_to_boxes("")
    if pending:
        plain, boxes = pending["plain_text"], harvest.code_to_boxes(pending["suggested"] or "")
        existing = await _db(lambda c, n: wordlist.entry_by_plain(c, n, pending["plain_text"]))
        if existing:
            return RedirectResponse(f"/admin/words/edit?id={existing['id']}&pending={pending['id']}", status_code=303)
    return await _entry_form(request, admin, entry=None, plain=plain, boxes=boxes, pending=pending, problems=[],
                       show_preview=bool(pending and pending["suggested"]), expected=None)


@router.get("/admin/words/edit")
@_safe
async def edit_word(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    eid = request.query_params.get("id", "")
    row = await _db(lambda c, n: wordlist.entry(c, n, int(eid))) if eid.isdigit() else None
    if row is None:
        return _words_page(request, "Not found", '<p><a href="/admin/words">← Word list</a></p>'
                           '<h1>That word isn’t in the list</h1>', email=admin, tab="list", status=404)
    pending = await _pending_for(request.query_params.get("pending"))
    boxes = harvest.code_to_boxes(row["diacritic_text"])
    if pending and pending["suggested"]:
        boxes = harvest.code_to_boxes(pending["suggested"])
    return await _entry_form(request, admin, entry=row, plain=row["plain_text"], boxes=boxes, pending=pending, problems=[],
                       show_preview=True, expected=(row["plain_text"], row["diacritic_text"]))


@router.post("/admin/words/save")
@_safe
async def save_word(request: Request):
    admin, form, denied = await _checked(request)
    if denied:
        return denied
    eid = form.get("id", "")
    row = await _db(lambda c, n: wordlist.entry(c, n, int(eid))) if eid.isdigit() else None
    if eid and row is None:
        return _back("/admin/words", "That word was deleted meanwhile. Nothing was saved.", "bad")
    pending = await _pending_for(form.get("pending"))
    boxes = {"type": form.get("type") or "normal", "result": form.get("result", ""), "name": form.get("name", ""),
             "choices": form.get("choices", "").splitlines(), "advanced": form.get("advanced", "")}
    plain = clean_plain(form.get("plain", ""))
    expected = (form["expected_plain"], form["expected_result"]) if form.get("expected_plain") else None
    problems = harvest.box_problems(boxes)
    if not plain:
        problems.insert(0, "Please type the plain word.")
    elif not re.fullmatch(r"[a-z0-9' \-]+", plain):
        problems.insert(0, "The plain word can only have the letters a to z, spaces and hyphens.")
    if plain and not problems:
        other = await _db(lambda c, n: wordlist.entry_by_plain(c, n, plain))
        if other and (row is None or other["id"] != row["id"]):
            return await _entry_form(request, admin, entry=row, plain=form.get("plain", ""), boxes=boxes, pending=pending,
                               problems=[], show_preview=False, expected=expected, status=409, other=other)
    if problems or form.get("action") == "preview":
        return await _entry_form(request, admin, entry=row, plain=form.get("plain", ""), boxes=boxes, pending=pending,
                           problems=problems, show_preview=not problems, expected=expected,
                           status=400 if problems else 200)
    code = harvest.boxes_to_code(boxes)
    try:
        await _db(lambda c, n: wordlist.save_entry(c, n, admin, plain=plain, result=code,
                                                   entry_id=row["id"] if row else None, expected=expected,
                                                   pending_id=pending["id"] if pending else None))
    except wordlist.Conflict as exc:
        now = exc.words[0][1]
        return _back(f"/admin/words/edit?id={row['id']}", "Someone changed this word while you were editing it "
                     f"(it is now {harvest.describe(now)}). Nothing was saved; please check it again.", "bad")
    except wordlist.WordListError as exc:
        return await _entry_form(request, admin, entry=row, plain=form.get("plain", ""), boxes=boxes, pending=pending,
                           problems=[str(exc)], show_preview=False, expected=expected, status=400)
    back = "/admin/words/pending" if pending else "/admin/words?" + urllib.parse.urlencode({"q": plain})
    return _back(back, f"Saved: {plain} → {harvest.describe(code)}." + (" It left the pending list." if pending else ""))


@router.post("/admin/words/delete")
@_safe
async def delete_word(request: Request):
    admin, form, denied = await _checked(request)
    if denied:
        return denied
    eid = form.get("id", "")
    if not eid.isdigit():
        return _back("/admin/words", "Nothing was deleted.", "bad")
    expected = (form.get("expected_plain", ""), form.get("expected_result", ""))
    try:
        await _db(lambda c, n: wordlist.delete_entry(c, n, admin, int(eid), expected))
    except wordlist.Conflict:
        return _back(f"/admin/words/edit?id={eid}", "Someone changed this word meanwhile. Nothing was deleted; "
                     "please check it again.", "bad")
    except wordlist.WordListError as exc:
        return _back("/admin/words", str(exc), "bad")
    return _back("/admin/words", f"Deleted {expected[0]}. You can undo it on the History page.")


# ---------- add words: uploads and the list of reviews ----------

UPLOAD_JS = """(function(){
  var f = document.getElementById('upload'); if (!f) return;
  f.addEventListener('submit', function(){
    var b = f.querySelector('button'); b.disabled = true;
    document.getElementById('reading').style.display = '';
  });
})();"""


def _counts_text(kind: str, counts: dict) -> str:
    counts = counts or {}
    if kind == "document":
        parts = [(counts.get("clash", 0), "different"), (counts.get("new", 0), "new"),
                 (counts.get("unmarked", 0), "without marks"), (counts.get("already", 0), "already in the list")]
    else:
        parts = [(counts.get("lnew", 0), "new"), (counts.get("lchanged", 0), "changed"),
                 (counts.get("lmissing", 0), "missing"), (counts.get("problems", 0), "problems")]
    return " · ".join(f"{n} {label}" for n, label in parts)


@router.get("/admin/words/add")
@_safe
async def add_words(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    open_, done, discarded, pending = await _db(lambda c, n: (
        wordlist.reviews(c, n, "open"), wordlist.reviews(c, n, "accepted", 20), wordlist.reviews(c, n, "discarded", 10),
        wordlist.pending_count(c, n)))

    def row(r, finished=False):
        files = ", ".join(f["name"] for f in r["files"])
        when = _fmt_time(r["accepted_at"]) if finished else _fmt_time(r["created_at"])
        return (f'<tr><td data-label="Review">#{r["id"]}</td><td><a href="/admin/words/review/{r["id"]}">{_e(files)}</a>'
                f'<br><span class="dim">{"a document" if r["kind"] == "document" else "a word list"} · '
                f'{_e(_counts_text(r["kind"], r["counts"]))}</span></td>'
                f'<td data-label="{"Finished" if finished else "Made"}">{when}<br><span class="dim">'
                f'{_e(r["accepted_by"] if finished else r["created_by"])}</span></td></tr>')
    open_rows = "".join(row(r) for r in open_)
    done_rows = "".join(row(r, True) for r in done)
    discarded_rows = "".join(row(r, True) for r in discarded)
    body = f"""<h1>Add words</h1>
<div class="card"><form id="upload" method="post" action="/admin/words/upload" enctype="multipart/form-data" style="margin:0">
{_hidden(_csrf_token(admin))}
<label class="big"><strong>Choose files</strong><br>
<input type="file" name="files" multiple required accept=".docx,.pdf,.txt,.xlsx" style="font-size:17px;margin:8px 0"></label><br>
<button class="primary">Read the files</button>
<span id="reading" style="display:none" class="dim">Reading… a big PDF takes up to a minute. Please keep this page open.</span>
</form></div>
<ul class="dim" style="margin-top:10px">
<li><strong>A document</strong> (Word .docx, PDF or .txt): its words with marks are compared with the list.</li>
<li><strong>A word list</strong> (Excel .xlsx with a plain word column and a result column, or a .txt of
“plain,result” lines), for example your own list or the pending list with the team's answers: compared word by word.</li>
<li>Nothing in the list changes until you press <strong>Accept</strong> on the review. Up to
{harvest.MAX_BYTES // 1024 // 1024} MB per file.</li></ul>
<h2>Unfinished reviews</h2>
<div class="scroll"><table class="list"><tr class="head"><th>#</th><th>Files</th><th>Made</th></tr>
{open_rows or '<tr><td colspan="3" class="dim">None.</td></tr>'}</table></div>
<details><summary>Finished reviews ({len(done)} accepted{f", {len(discarded)} discarded" if discarded else ""})</summary>
<div class="scroll"><table class="list"><tr class="head"><th>#</th><th>Files</th><th>Finished</th></tr>
{done_rows + discarded_rows or '<tr><td colspan="3" class="dim">None yet.</td></tr>'}</table></div></details>
<script>{UPLOAD_JS}</script>"""
    return _words_page(request, "Add words", body, email=admin, tab="add", pending=pending)


@router.post("/admin/words/upload")
@_safe
async def upload(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    try:
        form = await request.form(max_files=50)
    except Exception:  # noqa: BLE001  (a broken or oversized upload)
        return _back("/admin/words/add", "The upload didn't arrive in one piece. Please try again.", "bad")
    if not hmac.compare_digest(str(form.get("csrf", "")), _csrf_token(admin)):
        return _back("/admin/words/add", "That form had expired. Please try again.", "bad")
    documents, lists, problems = [], [], []
    for f in form.getlist("files"):
        name = getattr(f, "filename", None)
        if not name:
            continue
        name = name.replace("\\", "/").rsplit("/", 1)[-1][:200]
        if (getattr(f, "size", None) or 0) > harvest.MAX_BYTES:
            problems.append(f"{name}: this file is over {harvest.MAX_BYTES // 1024 // 1024} MB.")
            continue
        data = await f.read()
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
        return _back("/admin/words/add", " ".join(problems) or "Please choose a file.", "bad")

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
    if len(made) == 1:
        return _back(f"/admin/words/review/{made[0]}", "Here is what was found. Rows with an orange edge need a "
                     "choice." + note, "bad" if problems else "ok")
    return _back("/admin/words/add", f"Made {len(made)} reviews: see Unfinished reviews below." + note,
                 "bad" if problems else "ok")


# ---------- a review ----------

SEEN = {"small": "in small letters",
        "name": "with a capital mid-sentence: a name or title?",
        "start": "with a capital only at a sentence start"}

SECTIONS_DOC = (("clash", "Different from our list"), ("new", "New words"),
                ("unmarked", "Written without marks in the document"), ("style", "Style only (fixed automatically)"),
                ("doubtful", "Doubtful (look-alike letters or PDF quirks, not compared)"),
                ("already", "Already in our list"))
SECTIONS_LIST = (("lnew", "New"), ("lchanged", "Changed"), ("lmissing", "Missing from your file"),
                 ("problems", "Problems (rows skipped)"), ("empty", "Left empty"), ("unchanged", "Unchanged"))


def _radios(item: dict, decision: dict, disabled: bool) -> str:
    chosen = (decision or {}).get("choice")
    off = " disabled" if disabled else ""
    return "".join(f'<label><input type="radio" name="c:{item["id"]}" value="{_e(v)}"'
                   f'{" checked" if v == chosen else ""}{off}><span>{_e(label)}</span></label>'
                   for v, label in harvest.choice_options(item))


def _cap_radios(item: dict, decision: dict, disabled: bool) -> str:
    chosen = (decision or {}).get("cap")
    off = " disabled" if disabled else ""
    return "".join(f'<label><input type="radio" name="cap:{item["id"]}" value="{v}"'
                   f'{" checked" if v == chosen else ""}{off}><span>{_e(label)}</span></label>'
                   for v, label in harvest.CAPITAL_OPTIONS[item["kind"]])


def _todo(item: dict, decision: dict) -> bool:
    return item.get("kind") in harvest.NEEDS_CHOICE and \
        (decision or {}).get("choice") not in {v for v, _ in harvest.choice_options(item)}


def _changed_note(item: dict) -> str:
    if not item.get("changed_meanwhile"):
        return ""
    return (f'<div class="changed">Changed in the list after this review was made: it is now '
            f'{_e(item["changed_meanwhile"])}. Please choose again.</div>')


def _clash_card(item: dict, d: dict, disabled: bool) -> str:
    others = [f for f in item["forms"][1:]]
    also = ""
    if others:
        also = ('<div class="dim">The document also has: ' + "; ".join(
            f'<strong>{_e(f["form"])}</strong> ({_e(f["status"])}, {_times(f["times"])})' for f in others) + "</div>")
    stored = (f'stored as {_e(harvest.describe(item["entry"]))}' if item.get("entry")
              else "not one entry: our list gives its parts")
    return f"""<div class="item{' todo' if _todo(item, d) else ''}" id="{item['id']}">
<div class="head"><span class="w">{_e(item['form'])}</span>
<span class="dim">plain word {_e(item['plain'])} · {_times(item['times'])}{_pages(item['where'])}</span></div>
{also}
<div class="line">{_highlight(item['line'], item['form'])}</div>
<div>Our list gives <strong class="big">{_e(item['gives'])}</strong> <span class="dim">({stored})</span></div>
{f'<div class="note">{_e(item["note"])}</div>' if item.get('note') else ''}{_changed_note(item)}
<div class="pick"><span class="lbl">Choose:</span>{_radios(item, d, disabled)}</div>
<div class="pick"><span class="lbl">Capital:</span>{_cap_radios(item, d, disabled)}</div></div>"""


def _new_row(item: dict, d: dict, disabled: bool) -> str:
    chosen = set((d or {}).get("extras", []))
    extras = "".join(
        f'<label><input type="checkbox" name="x:{item["id"]}:{i}" value="1"{" checked" if i in chosen else ""}'
        f'{" disabled" if disabled else ""}> also add {_e(x["plain"])} → {_e(x["result"])}</label>'
        for i, x in enumerate(item.get("extras", [])))
    spellings = (f'<br><span class="dim">spelt {" and ".join(_e(s) for s in item["spellings"])}</span>'
                 if len(item.get("spellings", [])) > 1 else "")
    greek = ('<br><span class="note">also written with a Greek look-alike letter (Ᾱ) in the document</span>'
             if item.get("greek") else "")
    now = (f'<br><span class="changed">Added to the list meanwhile, as {_e(harvest.describe(item["now_in_list"]))}'
           f'</span>' if item.get("now_in_list") else "")
    return f"""<tr id="{item['id']}"{' class="todo"' if _todo(item, d) else ''}>
<td data-label="Word"><span class="w">{_e(item['form'])}</span><br><span class="dim">{_e(item['plain'])} ·
{_times(item['times'])}{_pages(item['where'])}<br>seen {_e(SEEN[item['seen']])}</span>{spellings}{greek}{now}</td>
<td data-label="Example"><div class="line" style="margin:0">{_highlight(item['line'], item['form'])}</div></td>
<td class="choose"><div class="pick">{_radios(item, d, disabled)}</div>
<div class="pick"><span class="lbl">Capital:</span>{_cap_radios(item, d, disabled)}</div>
{f'<div class="extras">{extras}</div>' if extras else ''}</td></tr>"""


def _unmarked_row(item: dict, d: dict, disabled: bool) -> str:
    return f"""<tr id="{item['id']}"><td data-label="Word"><span class="w">{_e(item['form'])}</span><br>
<span class="dim">{_times(item['times'])}{_pages(item['where'])}</span></td>
<td data-label="Our list gives" class="big">{_e(item['gives'])}</td>
<td data-label="Example"><div class="line" style="margin:0">{_highlight(item['line'], item['mark'])}</div></td>
<td><div class="pick">{_radios(item, d, disabled)}</div></td></tr>"""


def _section(key: str, title: str, items: list, inner: str, *, needs: bool, extra: str = "", todo: int = 0) -> str:
    if not items:
        return ""
    left = f' <span class="note">· {todo} to decide</span>' if todo else ""
    return (f'<details class="sec" id="{key}"{" open" if needs else ""}><summary>{_e(title)} ({len(items)}){left}'
            f'</summary>{extra}{inner}</details>')


def _review_sections(rv: dict, disabled: bool) -> str:
    items, dec = rv["items"], rv["decisions"]
    s = items["sections"]
    out = []
    open_ = not disabled   # a finished review shows its sections closed

    def todo(key):
        return sum(1 for i in s.get(key, []) if _todo(i, dec.get(i["id"])))
    if items["kind"] == "document":
        out.append(_section("clash", "Different from our list", s["clash"],
                            "".join(_clash_card(i, dec.get(i["id"]), disabled) for i in s["clash"]),
                            needs=open_, todo=todo("clash"),
                            extra='<p class="dim">“The document is right” puts the document\'s spelling in the list '
                                  '(the capital setting stays as it is unless you change it below).</p>'))
        bulk = "" if disabled else (
            '<p><button name="action" value="new_pending">Send all undecided new words to Pending</button>'
            '<button name="action" value="new_add">Add all undecided new words</button></p>')
        out.append(_section("new", "New words", s["new"],
                            '<div class="scroll"><table class="list new"><tr class="head"><th style="width:27%">Word</th>'
                            '<th>Example line</th><th style="width:36%">Choose</th></tr>'
                            + "".join(_new_row(i, dec.get(i["id"]), disabled) for i in s["new"]) + "</table></div>",
                            needs=open_, todo=todo("new"), extra=bulk))
        out.append(_section("unmarked", "Written without marks in the document", s["unmarked"],
                            '<div class="scroll"><table class="list"><tr class="head"><th>In the document</th>'
                            '<th>Our list gives</th><th>Example line</th><th>Choose</th></tr>'
                            + "".join(_unmarked_row(i, dec.get(i["id"]), disabled) for i in s["unmarked"])
                            + "</table></div>", needs=False,
                            extra='<p>Nothing changes in the list for these; they are listed in the report so the '
                                  'teams can be told.</p>'))
        out.append(_section("style", "Style only (fixed automatically)", s["style"],
                            '<div class="scroll"><table class="list"><tr class="head"><th>In the document</th>'
                            '<th>Our list gives</th><th>Difference</th><th>Example line</th></tr>' + "".join(
                                f'<tr><td class="big">{_e(i["form"])}</td><td class="big">{_e(i["gives"])}</td>'
                                f'<td>{_e(i["why"])}</td><td><div class="line" style="margin:0">'
                                f'{_highlight(i["line"], i["form"])}</div></td></tr>' for i in s["style"])
                            + "</table></div>", needs=False,
                            extra='<p class="dim">The same word, typed differently in the document (an underline '
                                  'instead of the tie, a Greek look-alike letter). Nothing to do.</p>'))
        out.append(_section("doubtful", "Doubtful (look-alike letters or PDF quirks, not compared)", s["doubtful"],
                            '<div class="scroll"><table class="list"><tr class="head"><th>Word</th><th>What it is</th>'
                            '<th>Line</th></tr>' + "".join(
                                f'<tr><td class="big">{_e(i["form"])}<br><span class="dim">'
                                f'{(" page " + _e(i["where"])) if i["where"] else ""}</span></td><td>{_e(i["why"])}</td>'
                                f'<td><div class="line" style="margin:0">{_highlight(i["line"], i["form"])}</div></td>'
                                f'</tr>' for i in s["doubtful"]) + "</table></div>", needs=False))
        out.append(_section("already", "Already in our list", s["already"],
                            '<ul class="cols">' + "".join(
                                f'<li>{_e(i["form"])} <span class="dim">({i["times"]})</span></li>'
                                for i in s["already"]) + "</ul>", needs=False))
    else:
        out.append(_section("lnew", "New", s["lnew"],
                            '<div class="scroll"><table class="list"><tr class="head"><th>Plain word</th>'
                            '<th>Result in your file</th><th>Choose</th></tr>' + "".join(
                                f'<tr id="{i["id"]}"><td class="big">{_e(i["plain"])}<br><span class="dim">row '
                                f'{i["row"]}</span></td><td class="big">{_e(harvest.describe(i["result"]))}'
                                f'{_changed_note(i)}</td><td><div class="pick">{_radios(i, dec.get(i["id"]), disabled)}'
                                f'</div></td></tr>' for i in s["lnew"]) + "</table></div>", needs=open_))
        out.append(_section("lchanged", "Changed", s["lchanged"],
                            '<div class="scroll"><table class="list"><tr class="head"><th>Plain word</th>'
                            '<th>Our list</th><th>Your file</th><th>Choose</th></tr>' + "".join(
                                f'<tr id="{i["id"]}"{" class=todo" if _todo(i, dec.get(i["id"])) else ""}>'
                                f'<td class="big">{_e(i["plain"])}<br><span class="dim">row {i["row"]}</span></td>'
                                f'<td class="big">{_shown(i["entry"])}{_changed_note(i)}</td>'
                                f'<td class="big">{_e(harvest.describe(i["result"]))}</td>'
                                f'<td><div class="pick">{_radios(i, dec.get(i["id"]), disabled)}</div></td></tr>'
                                for i in s["lchanged"]) + "</table></div>", needs=open_, todo=todo("lchanged")))
        out.append(_section("lmissing", "Missing from your file", s["lmissing"],
                            '<div class="scroll"><table class="list"><tr class="head"><th>Plain word</th>'
                            '<th>Our list</th><th>Choose</th></tr>' + "".join(
                                f'<tr id="{i["id"]}"><td class="big">{_e(i["plain"])}</td><td class="big">'
                                f'{_shown(i["entry"])}</td><td><div class="pick">'
                                f'{_radios(i, dec.get(i["id"]), disabled)}</div></td></tr>' for i in s["lmissing"])
                            + "</table></div>", needs=False,
                            extra="<p>Words missing from your file are kept unless you choose Delete.</p>"))
        out.append(_section("problems", "Problems (rows skipped)", s["problems"],
                            '<div class="scroll"><table class="list"><tr class="head"><th>Row</th><th>Plain word</th>'
                            '<th>Result</th><th>Why</th></tr>' + "".join(
                                f'<tr><td>{i["row"]}</td><td>{_e(i["plain"])}</td><td>{_e(i["result"])}</td>'
                                f'<td>{_e(i["why"])}</td></tr>' for i in s["problems"]) + "</table></div>", needs=False))
        out.append(_section("empty", "Left empty", s.get("empty", []),
                            '<p class="dim">These rows have a plain word but no result (for example words the team '
                            'hasn\'t answered yet). They are left out; pending words stay pending.</p>'
                            '<ul class="cols">' + "".join(f'<li>{_e(i["plain"])} <span class="dim">(row {i["row"]})'
                                                         f'</span></li>' for i in s.get("empty", [])) + "</ul>",
                            needs=False))
        out.append(_section("unchanged", "Unchanged", s["unchanged"],
                            '<ul class="cols">' + "".join(f'<li>{_e(i["plain"])}</li>' for i in s["unchanged"])
                            + "</ul>", needs=False))
    return "".join(out)


REVIEW_JS = r"""(function(){
  var form = document.getElementById('review'); if (!form) return;
  var need = JSON.parse(form.getAttribute('data-need'));
  function left(){
    return need.filter(function(id){ return !form.querySelector('input[name="c:' + id + '"]:checked'); });
  }
  function update(){
    var l = left();
    document.querySelectorAll('.left-count').forEach(function(el){
      el.textContent = l.length ? (l.length + (l.length === 1 ? ' row still needs' : ' rows still need') + ' a choice.') : 'Every row has a choice.';
    });
    document.querySelectorAll('button[value=accept]').forEach(function(b){ b.disabled = l.length > 0; });
    document.querySelectorAll('button[value=rest_pending]').forEach(function(b){ b.style.display = l.length ? '' : 'none'; });
    need.forEach(function(id){
      var el = document.getElementById(id);
      if (el) el.classList.toggle('todo', l.indexOf(id) >= 0);
    });
  }
  form.addEventListener('change', function(){
    update();
    document.querySelectorAll('.plan-line').forEach(function(el){ el.classList.add('dim'); });
    document.querySelectorAll('.unsaved').forEach(function(el){ el.style.display = ''; });
  });
  update();
})();"""


def _actions(rv: dict, p: dict, where: str) -> str:
    undecided = len(p["undecided"])
    left = (f"{undecided} {'row still needs' if undecided == 1 else 'rows still need'} a choice."
            if undecided else "Every row has a choice.")
    return f"""<div class="actions" id="actions-{where}">
<p class="big plan-line"><strong>{_e(harvest.plan_sentence(p))}</strong></p>
<p class="unsaved dim" style="display:none">(You have changed some choices: press Save my choices to update this line.)</p>
<p><span class="left-count">{left}</span>
<button name="action" value="rest_pending"{'' if undecided else ' style="display:none"'}>Send the rest to Pending</button></p>
<button name="action" value="save">Save my choices</button>
<button class="primary" name="action" value="accept"{' disabled' if undecided else ''}
onclick="return confirm('Accept your choices? The list is changed now (you can undo it on the History page).')">Accept</button>
</div>"""


@router.get("/admin/words/review/{review_id}")
@_safe
async def review_page(request: Request, review_id: int):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    rv, pending = await _db(lambda c, n: (wordlist.get_review(c, n, review_id), wordlist.pending_count(c, n)))
    if rv is None:
        return _words_page(request, "Review not found", '<p><a href="/admin/words/add">← Add words</a></p>'
                           '<h1>Review not found</h1>', email=admin, tab="add", status=404)
    csrf = _csrf_token(admin)
    items, dec = rv["items"], rv["decisions"]
    p = harvest.plan(items, dec)
    is_open = rv["status"] == "open"
    sections = SECTIONS_DOC if items["kind"] == "document" else SECTIONS_LIST
    chips = "".join(f'<a class="chip" href="#{k}"><b>{items["counts"].get(k, 0):,}</b> {_e(label.split(" (")[0])}</a>'
                    for k, label in sections if items["counts"].get(k))
    files = rv["files"]
    detail = "; ".join(f'{f["name"]}' + (f' ({f["pages"]} pages)' if f.get("pages") and f["name"].lower().endswith(".pdf")
                                          else f' ({f["rows"]} rows)' if f.get("rows") is not None else "")
                       for f in files)
    if rv["status"] == "accepted":
        state = (f'<div class="banner ok">Accepted {_fmt_time(rv["accepted_at"])} by {_e(rv["accepted_by"])}. '
                 f'<a href="/admin/words/history/{rv["batch_id"]}">See what changed</a> (it can be undone on the '
                 f'<a href="/admin/words/history">History</a> page).</div>')
    elif rv["status"] == "discarded":
        state = (f'<div class="banner warn">Discarded {_fmt_time(rv["accepted_at"])} by {_e(rv["accepted_by"])}: '
                 f'nothing was changed.</div>')
    else:
        state = ""
    if items["kind"] == "list" and not items.get("whole_list"):
        state += ('<p class="dim">Your file has fewer than half of the list’s words, so it is read as additions and '
                  'changes: the words not in it are simply kept.</p>')
    need = [i["id"] for sec in items["sections"].values() for i in sec if i.get("kind") in harvest.NEEDS_CHOICE]
    body_sections = _review_sections(rv, not is_open)
    tools = f"""<div><a class="button" href="/admin/words/review/{review_id}/report.xlsx">Download the report (Excel)</a>
{f'''<form class="inline" method="post" action="/admin/words/review/{review_id}/discard"
onsubmit="return confirm('Discard this review? Nothing in the list changes.')">{_hidden(csrf)}
<button class="danger">Discard this review</button></form>''' if is_open else ''}</div>"""
    if is_open:
        main = f"""<form method="post" action="/admin/words/review/{review_id}" id="review" data-need='{_e(json.dumps(need))}'>
{_hidden(csrf)}
{_actions(rv, p, "top")}
{body_sections}
{_actions(rv, p, "bottom")}
</form><script>{REVIEW_JS}</script>"""
    else:
        main = body_sections
    body = f"""<p><a href="/admin/words/add">← Add words</a></p>
<h1>Review #{review_id}: {_e(rv['title'])}</h1>
<p class="dim">{_e(detail)} · read as {'a document' if items['kind'] == 'document' else 'a word list'} ·
made {_fmt_time(rv['created_at'])} by {_e(rv['created_by'])}</p>
{state}
<div class="chips">{chips}</div>
{tools}
{main}"""
    return _words_page(request, f"Review #{review_id}", body, email=admin, tab="add", pending=pending, wide=True)


def _decisions_from_form(items: dict, form: dict, old: dict) -> dict:
    out = {}
    for section in items["sections"].values():
        for item in section:
            kind = item.get("kind")
            if not kind or "id" not in item or not harvest.choice_options(item):
                continue
            d = {}
            valid = {v for v, _ in harvest.choice_options(item)}
            if form.get(f"c:{item['id']}") in valid:
                d["choice"] = form[f"c:{item['id']}"]
            caps = {v for v, _ in harvest.CAPITAL_OPTIONS.get(kind, [])}
            if form.get(f"cap:{item['id']}") in caps:
                d["cap"] = form[f"cap:{item['id']}"]
            elif (old.get(item["id"]) or {}).get("cap"):
                d["cap"] = old[item["id"]]["cap"]
            extras = [i for i in range(len(item.get("extras", []))) if form.get(f"x:{item['id']}:{i}") == "1"]
            if extras:
                d["extras"] = extras
            out[item["id"]] = d
    return out


@router.post("/admin/words/review/{review_id}")
@_safe
async def review_post(request: Request, review_id: int):
    back = f"/admin/words/review/{review_id}"
    admin, form, denied = await _checked(request, back)
    if denied:
        return denied
    rv = await _db(lambda c, n: wordlist.get_review(c, n, review_id))
    if rv is None or rv["status"] != "open":
        return _back(back, "This review is already finished.", "bad")
    items = rv["items"]
    dec = _decisions_from_form(items, form, rv["decisions"])
    action = form.get("action", "save")
    anchor = ""
    for section in items["sections"].values():
        for item in section:
            if item.get("kind") not in harvest.NEEDS_CHOICE or (dec.get(item["id"]) or {}).get("choice"):
                continue
            if action == "rest_pending" or (action == "new_pending" and item["kind"] == "new"):
                dec[item["id"]]["choice"] = "pending"
            elif action == "new_add" and item["kind"] == "new" and not item.get("now_in_list"):
                dec[item["id"]]["choice"] = "add:0"
    if action in ("new_pending", "new_add"):
        anchor = "#new"
    await _db(lambda c, n: wordlist.save_decisions(c, n, review_id, dec))
    if action != "accept":
        left = len(harvest.undecided(items, dec))
        return _back(back + anchor, "Saved your choices." + (f" {left} rows still need a choice." if left else
                                                              " Every row has a choice: you can Accept now."))
    try:
        done = await _db(lambda c, n: wordlist.accept_review(c, n, review_id, admin))
    except wordlist.Conflict as exc:
        await _db(lambda c, n: wordlist.refresh_review(c, n, review_id, exc.words))
        listed = "; ".join(f"{w} (now {harvest.describe(v) if v else 'deleted'})" for w, v in exc.words[:10])
        return _back(back, f"Nothing was changed: these words were changed in the list after this review was made: "
                     f"{listed}. Their rows are marked in red; please choose again, then Accept.", "bad")
    except wordlist.WordListError as exc:
        return _back(back, f"Nothing was changed: {exc}", "bad")
    said = harvest.plan_sentence(done["plan"], done=True)
    extra = f" {done['resolved']} pending {'word was' if done['resolved'] == 1 else 'words were'} resolved." \
        if done["resolved"] else ""
    return _back(back, said + extra)


@router.post("/admin/words/review/{review_id}/discard")
@_safe
async def discard(request: Request, review_id: int):
    admin, form, denied = await _checked(request, f"/admin/words/review/{review_id}")
    if denied:
        return denied
    try:
        await _db(lambda c, n: wordlist.discard_review(c, n, review_id, admin))
    except wordlist.WordListError as exc:
        return _back(f"/admin/words/review/{review_id}", str(exc), "bad")
    return _back("/admin/words/add", f"Discarded review #{review_id}. Nothing in the list was changed.")


@router.get("/admin/words/review/{review_id}/report.xlsx")
@_safe
async def report(request: Request, review_id: int):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    rv = await _db(lambda c, n: wordlist.get_review(c, n, review_id))
    if rv is None:
        return Response("Review not found", status_code=404)
    status = {"open": "not accepted yet", "accepted": f"accepted {_fmt_time(rv['accepted_at'])} by {rv['accepted_by']}",
              "discarded": f"discarded {_fmt_time(rv['accepted_at'])} (nothing changed)"}[rv["status"]]
    info = {"title": f"Review #{review_id}: {rv['title']}",
            "list": "the trial copy" if _list() == "trial" else "the real list",
            "files": [f["name"] for f in rv["files"]], "made": _fmt_time(rv["created_at"]), "by": rv["created_by"],
            "status": status}
    data = await asyncio.to_thread(harvest.report_xlsx, rv["items"], rv["decisions"], info)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", rv["files"][0]["name"].rsplit(".", 1)[0])[:60] if rv["files"] else "review"
    return _xlsx(data, f"word-review-{review_id}-{stem}.xlsx")


# ---------- pending ----------

@router.get("/admin/words/pending")
@_safe
async def pending_page(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    items, done, in_list = await _db(lambda c, n: (wordlist.pending(c, n), wordlist.pending(c, n, resolved=True,
                                                                                           limit=100),
                                                   set(wordlist.mappings(c, n))))
    csrf = _csrf_token(admin)
    rows = "".join(f"""<tr id="p{i['id']}"><td data-label="Plain word" class="big"><strong>{_e(i['plain_text'])}</strong></td>
<td data-label="Suggested" class="big">{_shown(i['suggested'])}</td>
<td data-label="Why">{_e(i['reason'])}<br><span class="dim">{_e(i['source'])}{(' · page ' + _e(i['page'])) if i['page'] and not i['page'].startswith('row') else (' · ' + _e(i['page'])) if i['page'] else ''}</span>
{f'<div class="line">{_e(i["context"])}</div>' if i['context'] else ''}</td>
<td class="acts"><a class="button" href="/admin/words/new?pending={i['id']}">{'Change in the list' if i['plain_text'] in in_list else 'Add to the list'}</a>
<form method="post" action="/admin/words/pending/{i['id']}/remove" style="margin:8px 0 0">{_hidden(csrf)}
<button>Remove from pending</button></form></td></tr>""" for i in items)
    done_rows = "".join(
        f'<tr><td class="big">{_e(i["plain_text"])}</td><td>{_shown(i["suggested"])}</td>'
        f'<td>{_e(i["resolution"])}</td><td>{_fmt_time(i["resolved_at"])}<br><span class="dim">{_e(i["resolved_by"])}'
        f'</span></td></tr>' for i in done)
    body = f"""<h1>Pending ({len(items)})</h1>
<p>Words waiting for an answer from the Diacriticals team.</p>
<p><a class="button" href="/admin/words/pending.xlsx">Download the pending list (Excel)</a></p>
<p class="dim">The download has an empty “Team's answer” column. When the team has filled it in, upload the sheet on
<a href="/admin/words/add">Add words</a>: it is read as a word list, and when you accept that review the answered
words leave this list. Rows left empty stay pending.</p>
<div class="scroll"><table class="list"><tr class="head"><th style="width:16%">Plain word</th>
<th style="width:22%">Suggested</th><th>Why</th><th style="width:210px"></th></tr>
{rows or '<tr><td colspan="4" class="dim">Nothing is pending.</td></tr>'}</table></div>
<details><summary>Recently resolved ({len(done)})</summary><div class="scroll"><table class="list">
<tr class="head"><th>Plain word</th><th>Suggested</th><th>How</th><th>When</th></tr>
{done_rows or '<tr><td colspan="4" class="dim">None yet.</td></tr>'}</table></div></details>"""
    return _words_page(request, "Pending", body, email=admin, tab="pending", pending=len(items), wide=True)


@router.get("/admin/words/pending.xlsx")
@_safe
async def pending_download(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    items = await _db(wordlist.pending)
    data = await asyncio.to_thread(harvest.pending_xlsx, items, _fmt_time)
    return _xlsx(data, f"diacritical-pending-{_today()}.xlsx")


@router.post("/admin/words/pending/{item_id}/remove")
@_safe
async def pending_remove(request: Request, item_id: int):
    admin, form, denied = await _checked(request, "/admin/words/pending")
    if denied:
        return denied
    try:
        plain = await _db(lambda c, n: wordlist.remove_pending(c, n, item_id, admin))
    except wordlist.WordListError as exc:
        return _back("/admin/words/pending", str(exc), "bad")
    return _back("/admin/words/pending", f"Removed {plain} from the pending list.")


# ---------- history ----------

KINDS = {"review": "Review", "edit": "Edit", "pending": "From pending", "undo": "Undo", "reset": "Trial restarted"}


@router.get("/admin/words/history")
@_safe
async def history(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    rows, pending = await _db(lambda c, n: (wordlist.batches(c, n), wordlist.pending_count(c, n)))
    csrf = _csrf_token(admin)
    last_reset = max((b["id"] for b in rows if b["kind"] == "reset"), default=0)

    def what(b):
        parts = [f"{b[k]} {label}" for k, label in (("adds", "added"), ("updates", "changed"), ("deletes", "deleted"))
                 if b[k]]
        return ", ".join(parts)

    def row(b):
        n = b["adds"] + b["updates"] + b["deletes"]
        undo = ""
        if b["undone_at"]:
            undo = (f'<span class="tag off">Undone</span><br><span class="dim" style="white-space:nowrap">'
                    f'{_fmt_time(b["undone_at"])}</span>')
        elif n and b["kind"] not in ("undo", "reset") and b["id"] > last_reset:
            undo = (f'<form class="inline" method="post" action="/admin/words/history/{b["id"]}/undo" '
                    f'onsubmit="return confirm(\'Undo this change? The words go back to how they were before it. '
                    f'(The pending list stays as it is.)\')">'
                    f'{_hidden(csrf)}<button>Undo</button></form>')
        return (f'<tr{" class=off" if b["undone_at"] else ""}><td data-label="When" style="white-space:nowrap">'
                f'{_fmt_time(b["created_at"])}<br>'
                f'<span class="dim">{_e(b["created_by"])}</span></td>'
                f'<td><span class="tag admin">{_e(KINDS.get(b["kind"], b["kind"]))}</span> {_e(b["summary"])}<br>'
                + (f'<a href="/admin/words/history/{b["id"]}">{_e(what(b))}</a>' if n else "")
                + (f' · <a href="/admin/words/review/{b["review_id"]}">the review</a>' if b["review_id"] else "")
                + f'</td><td>{undo}</td></tr>')
    body = f"""<h1>History</h1>
<p>Every change to the list made here, newest first. Undo puts the words back as they were before that change
(it can't if they were changed again since). The pending list isn't changed by Undo.</p>
<div class="scroll"><table class="list"><tr class="head"><th>When</th><th>What</th><th></th></tr>
{''.join(row(b) for b in rows) or '<tr><td colspan="3" class="dim">No changes yet.</td></tr>'}</table></div>"""
    return _words_page(request, "History", body, email=admin, tab="history", pending=pending)


@router.get("/admin/words/history/{batch_id}")
@_safe
async def history_detail(request: Request, batch_id: int):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    b, rows, pending = await _db(lambda c, n: (wordlist.get_batch(c, n, batch_id), wordlist.changes(c, batch_id),
                                               wordlist.pending_count(c, n)))
    if b is None:
        return _words_page(request, "Not found", '<p><a href="/admin/words/history">← History</a></p>'
                           '<h1>Change not found</h1>', email=admin, tab="history", status=404)
    ops = {"add": "Added", "update": "Changed", "delete": "Deleted"}
    table = "".join(f'<tr><td>{ops[c["op"]]}</td><td class="big">{_e(c["new_plain"] or c["old_plain"])}'
                    f'{" (was " + _e(c["old_plain"]) + ")" if c["op"] == "update" and c["old_plain"] != c["new_plain"] else ""}</td>'
                    f'<td class="big">{_shown(c["old_result"])}</td><td class="big">{_shown(c["new_result"])}</td></tr>'
                    for c in rows)
    body = f"""<p><a href="/admin/words/history">← History</a></p>
<h1>Change {batch_id}: {_e(KINDS.get(b['kind'], b['kind']))}</h1>
<p>{_e(b['summary'])}<br><span class="dim">{_fmt_time(b['created_at'])} by {_e(b['created_by'])}
{(' · undone ' + _fmt_time(b['undone_at']) + ' by ' + _e(b['undone_by'])) if b['undone_at'] else ''}</span></p>
<div class="scroll"><table class="list"><tr class="head"><th></th><th>Plain word</th><th>Before</th><th>After</th></tr>
{table or '<tr><td colspan="4" class="dim">No words were changed.</td></tr>'}</table></div>"""
    return _words_page(request, f"Change {batch_id}", body, email=admin, tab="history", pending=pending)


@router.post("/admin/words/history/{batch_id}/undo")
@_safe
async def undo(request: Request, batch_id: int):
    admin, form, denied = await _checked(request, "/admin/words/history")
    if denied:
        return denied
    try:
        new = await _db(lambda c, n: wordlist.undo(c, n, batch_id, admin))
    except wordlist.WordListError as exc:
        return _back("/admin/words/history", str(exc), "bad")
    return _back("/admin/words/history", f"Undone (recorded as change {new}).")


# ---------- try it ----------

async def _try_page(request: Request, admin: str, text: str, result: str | None):
    pending = await _db(wordlist.pending_count)
    body = f"""<h1>Try it</h1>
<p>Type or paste some text: the translator adds the marks using {'the trial copy of the list' if _list() == 'trial'
                                                                   else 'the real list'}.</p>
<form method="post" action="/admin/words/try">{_hidden(_csrf_token(admin))}
<textarea class="try" name="text" rows="6" placeholder="e.g. Shri Krishna came to Shirali">{_e(text)}</textarea>
<p><button class="primary">Add the marks</button></p></form>
{f'<h2>Result</h2><div class="out">{_e(result)}</div>' if result is not None else ''}"""
    return _words_page(request, "Try it", body, email=admin, tab="try", pending=pending)


@router.get("/admin/words/try")
@_safe
async def try_get(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    return await _try_page(request, admin, "", None)


@router.post("/admin/words/try")
@_safe
async def try_post(request: Request):
    admin, form, denied = await _checked(request, "/admin/words/try")
    if denied:
        return denied
    text = form.get("text", "")[:20000]
    current = await _db(wordlist.mappings)
    result = await asyncio.to_thread(translate_text, text, current)
    return await _try_page(request, admin, text, result)
