"""/admin/kb: collections (knowledge bases), per-person collection switches, and adding magazine issues.

Adding an issue: the admin picks a collection and a PDF; the browser checks it, asks the app for a presigned R2 URL,
uploads straight to R2, then tells the app, which checks the object and queues an ingest job (ingest/jobs.py). The
job page shows progress, then the review: counts first (pages, articles by kind, contents entries matched, flags),
the article list, and each article's text beside its page images. Publish puts the articles into search.
Same sign-in, admin check and CSRF tokens as admin.py.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import tempfile
import time
import urllib.parse
from collections import Counter, OrderedDict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

import kb
from admin import _csrf_token, _e, _fmt_time, _form, _page, _require_admin, access_button, person_anchor
from ingest import jobs, storage
from ingest.pdf import PdfFile, is_pdf_header

logger = logging.getLogger(__name__)
router = APIRouter()

STATUS = {"queued": ("Waiting to start", "run"), "running": ("Processing", "run"), "review": ("Ready for review", "on"),
          "publish": ("Waiting to publish", "run"), "publishing": ("Publishing", "run"),
          "published": ("Published", "yes"), "failed": ("Failed", "bad"),
          "replaced": ("Replaced by a newer upload", "off")}
STAGES = {"download": "fetching the PDF", "read": "reading pages", "split": "splitting into articles",
          "summaries": "writing summaries"}


def _storage() -> storage.Storage | None:
    return storage.Storage() if storage.configured() else None


def _back(path: str, message: str, kind: str = "ok") -> RedirectResponse:
    path, _, anchor = path.partition("#")
    joiner = "&" if "?" in path else "?"
    return RedirectResponse(f"{path}{joiner}" + urllib.parse.urlencode({"msg": message, "kind": kind})
                            + (f"#{anchor}" if anchor else ""), status_code=303)


def _flash(request: Request) -> dict:
    return {"flash": request.query_params.get("msg", ""),
            "flash_kind": "bad" if request.query_params.get("kind") == "bad" else "ok"}


async def _checked_form(request: Request):
    """(admin email, form fields, None) or (None, None, response) for a POST from an admin page."""
    admin, denied = _require_admin(request)
    if denied:
        return None, None, denied
    form = await _form(request)
    if not hmac.compare_digest(form.get("csrf", ""), _csrf_token(admin)):
        return None, None, _back(form.get("back") or "/admin/kb", "That form had expired. Please try again.", "bad")
    return admin, form, None


def status_label(job: dict) -> str:
    text, cls = STATUS.get(job["status"], (job["status"], "off"))
    p = job.get("progress") or {}
    if job["status"] == "running" and p.get("stage"):
        text = f"Processing: {STAGES.get(p['stage'], p['stage'])}"
        if p.get("stage") == "read" and p.get("total"):
            text += f" ({p.get('done', 0)}/{p['total']})"
    return f'<span class="tag {cls}">{_e(text)}</span>'


def _issue_name(job: dict) -> str:
    issue = job.get("issue") or {}
    return issue.get("label") or job["filename"]


def _contents(job: dict) -> str:
    f = job.get("findings") or {}
    if not f.get("contents_total"):
        return "–"
    return f"{f.get('contents_matched', 0)}/{f['contents_total']}"


# ---------- per-person switches (from the People page) ----------

@router.post("/admin/kb/access")
async def set_access(request: Request):
    admin, form, denied = await _checked_form(request)
    if denied:
        return denied
    email, kb_id = (form.get("email") or "").strip().lower(), form.get("kb_id", "")
    target = await kb.get_kb(kb_id)
    in_place = request.headers.get("x-requested-with") == "fetch"   # the switch on /admin (admin.ACCESS_JS)
    if not email or target is None:
        if in_place:
            return JSONResponse({"error": "unknown person or collection"}, status_code=400)
        return _back("/admin", "Nothing was changed: unknown person or collection.", "bad")
    enabled = form.get("enabled") == "1"
    await kb.set_access(email, kb_id, enabled, admin)
    message = f"{target['name']} is now {'on' if enabled else 'off'} for {email}."
    if in_place:
        return JSONResponse({"on": enabled, "message": message, "button": access_button(target, enabled, True)},
                            headers={"Cache-Control": "no-store"})
    return _back("/admin#" + person_anchor(email), message)


# ---------- collections ----------

@router.get("/admin/kb")
async def kb_page(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    csrf = _csrf_token(admin)
    kbs = await kb.list_kbs(with_counts=True)
    job_list = await jobs.list_jobs(limit=100000)
    r2 = storage.configured()

    def kb_row(k):
        return f"""<tr><td><strong>{_e(k['name'])}</strong><br><span class="muted">id {_e(k['id'])} ·
citations look like {_e(k['source_prefix'])}-2025-February.txt</span></td>
<td>{_e(k['description']) or '<span class="muted">–</span>'}</td>
<td>{'<span class="tag yes">On</span>' if k['default_on'] else '<span class="tag no">Off</span>'}</td>
<td>{k['articles']}</td>
<td><details><summary>Edit</summary>
<form method="post" action="/admin/kb/{_e(k['id'])}/update" style="margin-top:8px">
<input type="hidden" name="csrf" value="{csrf}">
<label>Name<br><input type="text" name="name" value="{_e(k['name'])}" required></label><br>
<label>Description<br><input type="text" name="description" value="{_e(k['description'])}" style="width:320px"></label><br>
<label><input type="checkbox" name="default_on" value="1" {'checked' if k['default_on'] else ''}>
On for everyone who hasn't been switched individually</label><br>
<button class="primary" style="margin-top:6px">Save</button></form>
{'' if k['id'] == kb.DEFAULT_KB or k['articles'] else f'''<form method="post" action="/admin/kb/{_e(k['id'])}/delete"
onsubmit="return confirm('Delete this empty collection?')" style="margin-top:8px"><input type="hidden" name="csrf" value="{csrf}">
<button class="danger">Delete collection</button></form>'''}
</details></td></tr>"""

    names = {k["id"]: k["name"] for k in kbs}

    def job_row(j):
        flags = len(j.get("flags") or [])
        return (f'<tr><td>#{j["id"]}</td><td><a href="/admin/kb/jobs/{j["id"]}">{_e(_issue_name(j))}</a><br>'
                f'<span class="muted">{_e(j["filename"])}</span></td>'
                f'<td data-label="Date added">{_fmt_time(j["created_at"])}<br><span class="muted">{_e(j["created_by"])}</span></td>'
                f'<td>{_e(names.get(j["kb_id"], j["kb_id"]))}</td>'
                f'<td>{status_label(j)}{"<br><span class=muted>" + _e(j["error"][:160]) + "</span>" if j.get("error") else ""}</td>'
                f'<td data-label="Pages">{j["pages"] or "–"}</td><td data-label="Articles">{j["articles"] or "–"}</td>'
                f'<td data-label="Contents">{_contents(j)}</td><td data-label="Flags">{flags if j.get("flags") is not None else "–"}</td>'
                f'<td data-label="Cost">${j["cost_usd"]:.2f}</td></tr>')

    active = any(j["status"] in ("queued", "running", "publish", "publishing") for j in job_list)
    view = issue_view(request.query_params, kbs)
    shown, nav = issue_page(job_list, view, kbs)
    target = next(k for k in kbs if k["id"] == view["kb"])
    import admin_sources
    linked = (await admin_sources.sources_by_kb()).get(target["id"], [])
    kind = linked[0]["kind"] if linked else "issues"
    options = "".join(f'<option value="{_e(k["id"])}"{" selected" if k["id"] == target["id"] else ""}>'
                      f'{_e(k["name"])}</option>' for k in kbs)
    upload = f"""<div class="card"><form id="upload" style="margin:0">
<input type="hidden" id="up-kb" value="{_e(target['id'])}">
<label style="display:block"><strong>Magazine issues (PDF)</strong>
<span class="muted">one issue per file; you can choose several files at once</span><br>
<input id="up-file" type="file" accept="application/pdf,.pdf" multiple required></label>
<button class="primary" style="margin-top:10px">Upload and process</button>
<ul id="up-list" style="margin:8px 0 0;padding-left:20px"></ul>
<p id="up-status" class="muted" style="margin:8px 0 0"></p></form></div>
<p class="muted">Each PDF goes straight to storage (up to {storage.MAX_UPLOAD_BYTES // 1024 // 1024} MB each), one
after the other, then ChitrapurGPT reads every page, splits the issue into articles and writes a summary of each,
which takes a few minutes per issue. Nothing is searchable until you review each issue and press Publish.</p>
<script>{UPLOAD_JS.replace("__CSRF__", csrf).replace("__MAX__", str(storage.MAX_UPLOAD_BYTES))}</script>""" if r2 else \
        """<div class="banner warn">Uploads need Cloudflare R2: set R2_ACCOUNT_ID (or R2_ENDPOINT), R2_BUCKET,
R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY (see docs/INGEST.md).</div>"""

    if kind == "youtube":
        panel = await admin_sources.youtube_panel(target, linked[0], csrf)
    elif kind == "website":
        panel = await admin_sources.website_panel(target, linked[0], csrf)
    else:
        empty = not target["articles"] and not any(j["kb_id"] == target["id"] for j in job_list)
        panel = f"""<h2>Add magazine issues</h2>
{upload}
<h2 id="issues">{'Uploaded issues: all collections' if view['all'] else 'Uploaded issues'}</h2>
{nav}
<div class="scroll"><table><tr class="head">{issue_headers(view)}</tr>
{''.join(job_row(j) for j in shown) or '<tr><td colspan="10" class="muted">No issues uploaded yet.</td></tr>'}</table></div>
{nav if len(shown) > 10 else ''}
{admin_sources.start_panel(target, csrf) if empty and target['id'] != kb.DEFAULT_KB else ''}"""
    what = {"youtube": "videos and playlists from the YouTube channel's lists",
            "website": f"pages from the website {linked[0]['name'] if linked else ''}",
            "issues": "magazine issues"}[kind]
    body = f"""
<h1>Collections &amp; uploads</h1>
<div class="card"><label><strong>Collection</strong> <select id="kb-pick">{options}</select></label>
<span class="muted" style="margin-left:8px">{_e(what)} · {target['articles']:,} searchable entries ·
{'on' if target['default_on'] else 'off'} for everyone not switched individually</span>
{f'<br><span class="muted">{_e(target["description"])}</span>' if target['description'] else ''}</div>
<script>{FILTER_JS}</script>
{panel}
<h2>All collections</h2>
<p>Each collection is a set of articles people can search. Switch collections on or off per person on the
<a href="/admin">People</a> page; people you haven't switched get the collection's default.</p>
<div class="scroll"><table><tr class="head"><th>Collection</th><th>Description</th><th>Default</th><th>Articles</th><th></th></tr>
{''.join(kb_row(k) for k in kbs)}</table></div>
<details><summary>Add a collection</summary><div class="card" style="margin-top:8px">
<form method="post" action="/admin/kb/create" style="margin:0"><input type="hidden" name="csrf" value="{csrf}">
<label>Id (short, lowercase, e.g. ksa)<br><input type="text" name="id" required pattern="[a-z0-9][a-z0-9-]*"></label><br>
<label>Name (e.g. Kanara Saraswat)<br><input type="text" name="name" required></label><br>
<label>Description (e.g. the magazine of the Kanara Saraswat Association)<br>
<input type="text" name="description" style="width:420px"></label><br>
<label>Source prefix for citations (letters and digits, e.g. KSA → KSA-2025-February.txt)<br>
<input type="text" name="source_prefix" required pattern="[A-Za-z][A-Za-z0-9]*"></label><br>
<label><input type="checkbox" name="default_on" value="1"> On for everyone who hasn't been switched individually</label><br>
<button class="primary" style="margin-top:8px">Add collection</button></form></div></details>
{LIST_REFRESH_JS if active else ''}"""
    return _page("Collections & uploads", body, email=admin, **_flash(request))


# ---------- the uploaded issues list: one collection at a time (the one picked for uploading), sortable ----------

ISSUES_PER_PAGE = 25
_FAR = "9999-99-99"


def _issue_date(j: dict) -> str:
    """The issue's own date as yyyy-mm-dd (from its dd-mm-yyyy published date), for sorting by issue."""
    d = str((j.get("issue") or {}).get("published_date") or "")
    parts = d.split("-")
    if len(parts) == 3 and len(parts[2]) == 4:
        return f"{parts[2]}-{parts[1]}-{parts[0]}"
    return _FAR


# column -> (header, sort key, newest/biggest first by default)
ISSUE_COLUMNS = {
    "id": ("#", lambda j: j["id"], True),
    "issue": ("Issue", lambda j: (_issue_date(j), _issue_name(j).lower()), False),
    "added": ("Date added", lambda j: j["created_at"], True),
    "collection": ("Collection", lambda j: j["kb_id"], False),
    "status": ("Status", lambda j: j["status"], False),
    "pages": ("Pages", lambda j: j["pages"] or 0, True),
    "articles": ("Articles", lambda j: j["articles"] or 0, True),
    "contents": ("Contents matched", None, None),
    "flags": ("Flags", lambda j: len(j.get("flags") or []), True),
    "cost": ("Cost", lambda j: j["cost_usd"] or 0, True),
}


def issue_view(query, kbs) -> dict:
    """What the page shows, from the address: ?kb=<collection> (the page's collection), issues=all (list every
    collection's uploads), sort=<column>, dir=asc|desc, page=N."""
    ids = {k["id"] for k in kbs}
    show = query.get("kb") or kb.DEFAULT_KB
    if show not in ids:
        show = kb.DEFAULT_KB
    sort = query.get("sort")
    if sort not in ISSUE_COLUMNS or ISSUE_COLUMNS[sort][1] is None:
        sort = "added"
    direction = query.get("dir")
    desc = ISSUE_COLUMNS[sort][2] if direction not in ("asc", "desc") else direction == "desc"
    page = query.get("page") or "1"
    return {"kb": show, "all": query.get("issues") == "all", "sort": sort, "desc": desc,
            "page": max(1, int(page)) if page.isdigit() else 1}


def issue_url(view: dict, **changes) -> str:
    v = {**view, **changes}
    params = {"kb": v["kb"], **({"issues": "all"} if v.get("all") else {}), "sort": v["sort"],
              "dir": "desc" if v["desc"] else "asc"}
    if v.get("page", 1) > 1:
        params["page"] = v["page"]
    return "/admin/kb?" + urllib.parse.urlencode(params) + "#issues"


def issue_headers(view: dict) -> str:
    cells = []
    for col, (label, key, first_desc) in ISSUE_COLUMNS.items():
        if key is None:
            cells.append(f"<th>{_e(label)}</th>")
            continue
        here = view["sort"] == col
        arrow = (" ▼" if view["desc"] else " ▲") if here else ""
        url = issue_url(view, sort=col, desc=(not view["desc"]) if here else first_desc, page=1)
        cells.append(f'<th><a href="{_e(url)}" title="Sort by {_e(label.lower())}" '
                     f'style="color:inherit;text-decoration:underline dotted">{_e(label)}{arrow}</a></th>')
    return "".join(cells)


def issue_page(job_list: list[dict], view: dict, kbs) -> tuple[list[dict], str]:
    """(the uploads to list, the line above the table: what's shown, pages, See all)."""
    names = {k["id"]: k["name"] for k in kbs}
    rows = job_list if view["all"] else [j for j in job_list if j["kb_id"] == view["kb"]]
    _, key, _ = ISSUE_COLUMNS[view["sort"]]
    rows = sorted(rows, key=lambda j: (key(j), j["id"]), reverse=view["desc"])
    total = len(rows)
    if view["all"]:
        back = issue_url(view, all=False, page=1)
        return rows, (f'<p class="muted">Showing all {total:,} uploads, from every collection. '
                      f'<a href="{_e(back)}">Show only {_e(names.get(view["kb"], view["kb"]))}</a><br>'
                      f'Click a column heading to sort by it.</p>')
    pages = max(1, -(-total // ISSUES_PER_PAGE))
    page = min(view["page"], pages)
    shown = rows[(page - 1) * ISSUES_PER_PAGE:page * ISSUES_PER_PAGE]
    links = []
    if page > 1:
        links.append(f'<a href="{_e(issue_url(view, page=page - 1))}">‹ Previous</a>')
    if page < pages:
        links.append(f'<a href="{_e(issue_url(view, page=page + 1))}">Next ›</a>')
    where = f"page {page} of {pages} · " if pages > 1 else ""
    see_all = issue_url(view, all=True, page=1)
    return shown, (f'<p class="muted">{total:,} upload{"s" if total != 1 else ""} in '
                   f'{_e(names.get(view["kb"], view["kb"]))} · {where}{" · ".join(links) + " · " if links else ""}'
                   f'<a href="{_e(see_all)}">See all collections\' uploads ({len(job_list):,})</a><br>'
                   f'Click a column heading to sort by it.</p>')


# The collection picker at the top: the whole page switches to the chosen collection. Files already chosen for an
# upload would be lost, so it asks first.
FILTER_JS = r"""(function(){
  var sel = document.getElementById('kb-pick'); if (!sel) return;
  var was = sel.value;
  sel.addEventListener('change', function(){
    var chosen = ['up-file', 'yt-file'].some(function(id){ var f = document.getElementById(id); return f && f.files && f.files.length; });
    if (chosen && !confirm('Switch collection? The files you chose will need choosing again.')) { sel.value = was; return; }
    location.href = '/admin/kb?kb=' + encodeURIComponent(sel.value);
  });
})();"""


# While a job is processing, the list refreshes itself so its status moves. It must never reload in the middle of
# an upload (that aborts the PUT to R2) or while someone is typing in a form, so it only reloads when idle.
LIST_REFRESH_JS = """<script>(function(){
  function busy(){
    if (window.cgptUploading) return true;
    var f = document.getElementById('up-file'); if (f && f.files && f.files.length) return true;
    var y = document.getElementById('yt-file'); if (y && y.files && y.files.length) return true;
    var a = document.activeElement; return !!(a && /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName));
  }
  setInterval(function(){ if (!busy()) location.reload(); }, 10000);
})();</script>"""

UPLOAD_JS = r"""
(function(){
var CSRF = "__CSRF__", MAX = __MAX__;
var form = document.getElementById('upload'), status = document.getElementById('up-status');
function say(t){ status.textContent = t; }
async function post(url, data){
  var r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json','X-CSRF-Token':CSRF},
                            body:JSON.stringify(data), credentials:'same-origin'});
  var j = await r.json().catch(function(){ return {error:'The server sent an unexpected reply ('+r.status+').'}; });
  if (!r.ok) throw new Error(j.error || ('HTTP '+r.status));
  return j;
}
function put(url, file, say){
  return new Promise(function(resolve, reject){
    var x = new XMLHttpRequest(); x.open('PUT', url); x.setRequestHeader('Content-Type', 'application/pdf');
    x.upload.onprogress = function(e){ if (e.lengthComputable) say('Uploading… ' + Math.round(100*e.loaded/e.total) + '%'); };
    x.onload = function(){ x.status < 300 ? resolve() : reject(new Error('The upload failed (HTTP '+x.status+').')); };
    x.onerror = function(){ reject(new Error('The upload failed. Check your connection and try again.')); };
    x.send(file);
  });
}
// One file: check it, upload it, queue it. Returns the job, or null if the admin skipped a duplicate. Throws on
// errors. `say` reports progress for this file.
async function uploadOne(file, kbId, say){
  if (file.size > MAX) throw new Error('This file is over ' + Math.round(MAX/1048576) + ' MB.');
  var head = new Uint8Array(await file.slice(0, 1024).arrayBuffer());
  if (String.fromCharCode.apply(null, head).indexOf('%PDF-') < 0) throw new Error('This file is not a PDF.');
  say('Checking the file…');
  var digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  var sha = Array.from(new Uint8Array(digest)).map(function(b){ return b.toString(16).padStart(2, '0'); }).join('');
  var slot = await post('/admin/kb/upload-url', {kb_id: kbId, filename: file.name, size: file.size, sha256: sha});
  if (slot.warning && !confirm(file.name + ': ' + slot.warning)) return null;
  await put(slot.url, file, say);
  say('Uploaded. Queuing it…');
  return await post('/admin/kb/uploaded', {kb_id: kbId, key: slot.key, filename: file.name, size: file.size, sha256: sha});
}
form.addEventListener('submit', async function(e){
  e.preventDefault();
  var input = document.getElementById('up-file'), files = Array.from(input.files), kbId = document.getElementById('up-kb').value;
  if (!files.length) return;
  var button = form.querySelector('button'); button.disabled = true; input.disabled = true; window.cgptUploading = true;
  var list = document.getElementById('up-list'); list.innerHTML = '';
  var lines = files.map(function(f){
    var li = document.createElement('li'); li.textContent = f.name + ': waiting'; list.appendChild(li); return li;
  });
  var jobs = [], skipped = 0, failed = 0;
  for (var i = 0; i < files.length; i++) {
    var li = lines[i], name = files[i].name;
    var sayFile = (function(li, name){ return function(t){ li.textContent = name + ': ' + t; }; })(li, name);
    if (files.length > 1) say('File ' + (i + 1) + ' of ' + files.length + '. Please keep this page open.');
    try {
      var job = await uploadOne(files[i], kbId, sayFile);
      if (job) { jobs.push(job); li.innerHTML = ''; li.appendChild(document.createTextNode(name + ': queued as ')); var a = document.createElement('a'); a.href = job.redirect; a.textContent = '#' + job.job_id; li.appendChild(a); }
      else { skipped++; sayFile('skipped (already uploaded before). Nothing was uploaded.'); }
    } catch (err) { failed++; sayFile('not uploaded. ' + err.message); li.style.color = '#c62828'; }
  }
  window.cgptUploading = false;
  if (files.length === 1 && jobs.length === 1) { location.href = jobs[0].redirect; return; }
  var parts = [jobs.length + ' queued'];
  if (skipped) parts.push(skipped + ' skipped');
  if (failed) parts.push(failed + ' not uploaded');
  if (files.length === 1) { say(failed ? 'The file was not uploaded.' : 'Cancelled. Nothing was uploaded.'); }
  else if (!jobs.length) say('Done: nothing was queued (' + parts.slice(1).join(', ') + ').');
  else say('Done: ' + parts.join(', ') + '. They are processed one after the other.' +
           (failed ? ' Reload the page to see them in the list below.' : ''));
  button.disabled = false; input.disabled = false; input.value = '';
  if (jobs.length && !failed) setTimeout(function(){ location.reload(); }, 2500);
});
})();
"""


@router.post("/admin/kb/create")
async def create_kb(request: Request):
    admin, form, denied = await _checked_form(request)
    if denied:
        return denied
    try:
        await kb.create_kb(form.get("id", "").strip(), form.get("name", ""), form.get("description", ""),
                           form.get("source_prefix", "").strip(), form.get("default_on") == "1", admin)
    except ValueError as e:
        return _back("/admin/kb", f"{e} Nothing was added.", "bad")
    return _back("/admin/kb?" + urllib.parse.urlencode({"kb": form.get("id", "").strip()}),
                 f"Added the collection {form.get('name', '').strip()}. Fill it below.")


@router.post("/admin/kb/{kb_id}/update")
async def update_kb(request: Request, kb_id: str):
    admin, form, denied = await _checked_form(request)
    if denied:
        return denied
    if await kb.get_kb(kb_id) is None:
        return _back("/admin/kb", "That collection doesn't exist.", "bad")
    try:
        await kb.update_kb(kb_id, form.get("name", ""), form.get("description", ""), form.get("default_on") == "1",
                           admin)
    except ValueError as e:
        return _back("/admin/kb?" + urllib.parse.urlencode({"kb": kb_id}), f"{e} Nothing was changed.", "bad")
    return _back("/admin/kb?" + urllib.parse.urlencode({"kb": kb_id}), "Saved.")


@router.post("/admin/kb/{kb_id}/delete")
async def delete_kb(request: Request, kb_id: str):
    admin, form, denied = await _checked_form(request)
    if denied:
        return denied
    try:
        await kb.delete_kb(kb_id, admin)
    except ValueError as e:
        return _back("/admin/kb", str(e), "bad")
    return _back("/admin/kb", "Deleted the collection.")


# ---------- uploads (JSON, called by UPLOAD_JS) ----------

async def _json_admin(request: Request):
    admin, denied = _require_admin(request)
    if denied:
        return None, None, JSONResponse({"error": "Admins only. Please sign in again."}, status_code=403)
    if not hmac.compare_digest(request.headers.get("x-csrf-token", ""), _csrf_token(admin)):
        return None, None, JSONResponse({"error": "This page has expired. Reload it and try again."}, status_code=403)
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        return None, None, JSONResponse({"error": "Bad request."}, status_code=400)
    return admin, data, None


def _check_upload(data: dict) -> str | None:
    sha = str(data.get("sha256", ""))
    size = data.get("size")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        return "The file's checksum is missing."
    if not isinstance(size, int) or size <= 0:
        return "The file is empty."
    if size > storage.MAX_UPLOAD_BYTES:
        return f"The file is over {storage.MAX_UPLOAD_BYTES // 1024 // 1024} MB."
    if not str(data.get("filename", "")).lower().endswith(".pdf"):
        return "Only PDF files can be added."
    return None


@router.post("/admin/kb/upload-url")
async def upload_url(request: Request):
    admin, data, denied = await _json_admin(request)
    if denied:
        return denied
    store = _storage()
    if store is None:
        return JSONResponse({"error": "Uploads aren't set up (R2 isn't configured)."}, status_code=503)
    target = await kb.get_kb(str(data.get("kb_id", "")))
    problem = "Unknown collection." if target is None else _check_upload(data)
    if problem:
        return JSONResponse({"error": problem}, status_code=400)
    key = storage.object_key(target["id"], data["filename"], data["sha256"])
    earlier = await jobs.same_file(data["sha256"])
    warning = None
    if earlier:
        listed = "; ".join(f"#{j['id']} {_issue_name(j)} in {j['kb_id']} ({STATUS.get(j['status'], (j['status'],))[0]})"
                           for j in earlier[:5])
        warning = f"This exact file was uploaded before: {listed}. Upload it again anyway?"
    url = await asyncio.to_thread(store.presign_put, key)
    return JSONResponse({"url": url, "key": key, "warning": warning, "duplicates": [j["id"] for j in earlier]})


@router.post("/admin/kb/uploaded")
async def uploaded(request: Request):
    admin, data, denied = await _json_admin(request)
    if denied:
        return denied
    store = _storage()
    if store is None:
        return JSONResponse({"error": "Uploads aren't set up (R2 isn't configured)."}, status_code=503)
    target = await kb.get_kb(str(data.get("kb_id", "")))
    problem = "Unknown collection." if target is None else _check_upload(data)
    key = str(data.get("key", ""))
    if not problem and key != storage.object_key(target["id"], data["filename"], data["sha256"]):
        problem = "That upload doesn't match the one that was prepared."
    if problem:
        return JSONResponse({"error": problem}, status_code=400)
    head = await asyncio.to_thread(store.head, key)
    if head is None:
        return JSONResponse({"error": "The file didn't arrive in storage. Please try again."}, status_code=400)
    if head.get("ContentLength") != data["size"] or head["ContentLength"] > storage.MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "The stored file's size doesn't match. Please upload it again."}, status_code=400)
    if not is_pdf_header(await asyncio.to_thread(store.read_start, key)):
        await asyncio.to_thread(store.delete_issue, key)
        return JSONResponse({"error": "That file isn't a PDF."}, status_code=400)
    job_id = await jobs.create_job(target["id"], key, os.path.basename(data["filename"])[:200], data["sha256"],
                                   data["size"], admin)
    return JSONResponse({"job_id": job_id, "redirect": f"/admin/kb/jobs/{job_id}"})


# ---------- a job: progress and review ----------

def _articles_table(job_id: int, articles, csrf: str) -> str:
    rows = []
    for a in articles:
        pages = f"{a.start_page}" if a.start_page == a.end_page else f"{a.start_page}–{a.end_page}"
        words = len(a.text.split())
        state = f'<span class="tag off">Excluded</span> <span class="muted">{_e(a.note)}</span>' if a.excluded else ""
        toggle = "include" if a.excluded else "exclude"
        rows.append(f"""<tr{' class="off"' if a.excluded else ''}><td>{a.n}</td>
<td><a href="/admin/kb/jobs/{job_id}/articles/{a.n}">{_e(a.title)}</a>{'<br><span class="muted">by ' + _e(a.author) + '</span>' if a.author else ''} {state}</td>
<td data-label="Kind">{_e(a.kind.replace('_', ' '))}</td><td data-label="Pages">{pages}</td>
<td data-label="Length">{words:,} words</td>
<td><form class="inline" method="post" action="/admin/kb/jobs/{job_id}/articles/{a.n}/{toggle}">
<input type="hidden" name="csrf" value="{csrf}"><button>{toggle.capitalize()}</button></form></td></tr>""")
    return "".join(rows)


@router.get("/admin/kb/jobs/{job_id}.json")
async def job_json(request: Request, job_id: int):
    admin, denied = _require_admin(request)
    if denied:
        return JSONResponse({"error": "admins only"}, status_code=403)
    job = await jobs.get_job(job_id)
    if job is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    articles = await jobs.job_articles(job_id)
    return JSONResponse({k: job[k] for k in ("id", "kb_id", "status", "progress", "error", "issue", "findings",
                                              "flags", "pages", "published_articles")}
                        | {"cost_usd": round(job["cost_usd"], 4),
                           "articles": [{"n": a.n, "title": a.title, "kind": a.kind, "excluded": a.excluded,
                                         "start_page": a.start_page, "end_page": a.end_page, "chars": len(a.text)}
                                        for a in articles]},
                        headers={"Cache-Control": "no-store"})


@router.get("/admin/kb/jobs/{job_id}")
async def job_page(request: Request, job_id: int):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    job = await jobs.get_job(job_id)
    if job is None:
        return _page("Upload not found", '<p><a href="/admin/kb">← Collections &amp; uploads</a></p>'
                     '<h1>Upload not found</h1>', email=admin, status=404)
    csrf = _csrf_token(admin)
    target = await kb.get_kb(job["kb_id"]) or {"name": job["kb_id"]}
    articles = await jobs.job_articles(job_id)
    pages = await jobs.job_pages(job_id) if job["status"] not in ("queued",) else []
    issue = job.get("issue") or {}
    f = job.get("findings") or {}
    shown = [a for a in articles if not a.excluded]
    kinds = Counter(a.kind for a in shown)
    secs = job.get("secs") or {}
    active = job["status"] in ("queued", "running", "publish", "publishing")
    reread = sum(1 for p in pages if p.reread)

    def stat(n, label):
        return f'<div class="stat"><div class="n">{n}</div><div class="l">{label}</div></div>'

    cards = "".join([
        stat(job["pages"] or "–", "pages"),
        stat(len(shown), "articles: " + (", ".join(f"{v} {k.replace('_', ' ')}" for k, v in kinds.most_common())
                                       or "none yet")),
        stat(_contents(job), "contents entries matched"),
        stat(f"{f.get('unassigned', '–')} / {f.get('doubled', '–')}", "blocks in no article / in two"),
        stat(reread, "pages re-read by Sonnet"),
        stat(len(job.get("flags") or []), "flags"),
        stat(f"${job['cost_usd']:.2f}", "cost so far"),
        stat(f"{secs.get('wall', 0) / 60:.1f} min" if secs.get("wall") else "–", "processing time"),
    ])
    flags = "".join(f"<li>{_e(x)}</li>" for x in job.get("flags") or [])
    warnings = []
    if issue.get("source"):
        pool = await kb.get_pool()
        legacy = await pool.fetchval("SELECT count(*) FROM kb_article WHERE kb_id = $1 AND issue_id IS NULL "
                                     "AND source_id IS NULL AND metadata->>'source' = $2", job["kb_id"], issue["source"])
        other = await pool.fetch("SELECT DISTINCT issue_id FROM kb_article WHERE kb_id = $1 AND issue_id <> $2 "
                                 "AND metadata->>'source' = $3", job["kb_id"], job_id, issue["source"])
        if legacy:
            warnings.append(f"The original knowledge base already has {legacy} article(s) from "
                            f"{issue['source']}. Publishing adds these alongside them; the originals stay.")
        if other:
            warnings.append("This issue is already published from upload "
                            + ", ".join(f"#{r['issue_id']}" for r in other) + ". Publishing replaces those articles.")
    actions = []
    if job["status"] in ("review", "published", "replaced"):
        label = "Publish again" if job["status"] in ("published", "replaced") else "Publish"
        actions.append(f"""<form class="inline" method="post" action="/admin/kb/jobs/{job_id}/publish"
onsubmit="return confirm('Put {len(shown)} articles into search for everyone with this collection?')">
<input type="hidden" name="csrf" value="{csrf}"><button class="primary">{label}</button></form>""")
    if job["status"] in ("review", "failed", "published", "replaced"):
        actions.append(f"""<form class="inline" method="post" action="/admin/kb/jobs/{job_id}/rerun-split"
onsubmit="return confirm('Split the issue into articles again? Renamed or excluded articles are reset.')">
<input type="hidden" name="csrf" value="{csrf}"><button>Re-run split</button></form>""")
        actions.append(f"""<form class="inline" method="post" action="/admin/kb/jobs/{job_id}/rerun"
onsubmit="return confirm('Read every page again and split again? This costs about as much as the first run.')">
<input type="hidden" name="csrf" value="{csrf}"><button>Re-run whole job</button></form>""")
    if job["status"] in ("queued", "running"):
        actions.append(f"""<form class="inline" method="post" action="/admin/kb/jobs/{job_id}/stop">
<input type="hidden" name="csrf" value="{csrf}"><button class="danger">Stop</button></form>""")
    if job["status"] not in jobs.ACTIVE:
        published = job["status"] == "published"
        actions.append(f"""<form class="inline" method="post" action="/admin/kb/jobs/{job_id}/delete"
onsubmit="return confirm('{'Remove its articles from search and delete' if published else 'Delete'} this upload?')">
<input type="hidden" name="csrf" value="{csrf}"><button class="danger">Delete</button></form>""")
    issue_form = f"""<details><summary>Issue name, date and citation file name</summary><div class="card">
<form method="post" action="/admin/kb/jobs/{job_id}/issue" style="margin:0"><input type="hidden" name="csrf" value="{csrf}">
<label>Issue<br><input type="text" name="label" value="{_e(issue.get('label', ''))}"></label><br>
<label>Published (DD-MM-YYYY)<br><input type="text" name="published_date" value="{_e(issue.get('published_date', ''))}"></label><br>
<label>Cited as<br><input type="text" name="source" value="{_e(issue.get('source', ''))}" style="width:360px"></label><br>
<button style="margin-top:6px">Save</button></form></div></details>""" if job["status"] in ("review", "published", "replaced") else ""
    notes = (job.get("split") or {}).get("notes")
    page_rows = "".join(
        f'<tr><td><a href="/admin/kb/jobs/{job_id}/pages/{p.page}" target="_blank">{p.page}</a></td>'
        f'<td>{_e(p.model or "–")}</td><td>{_e(p.layer_kind)}</td>'
        f'<td>{"–" if p.recall is None else f"{p.recall:.1%}"}</td><td>{_e("; ".join(p.flags))}</td></tr>' for p in pages)
    body = f"""<p><a href="/admin/kb?{urllib.parse.urlencode({'kb': job['kb_id']})}#issues">← {_e(target['name'])}</a></p>
<h1>{_e(issue.get('label') or job['filename'])}</h1>
<p>{status_label(job)} · {_e(target['name'])} · <span class="muted">upload #{job_id}, {_e(job['filename'])},
{_fmt_time(job['created_at'])} by {_e(job['created_by'])}{', cited as ' + _e(issue['source']) if issue.get('source') else ''}
{', published ' + _fmt_time(job['published_at']) if job.get('published_at') else ''}</span></p>
{f'<div class="banner bad">{_e(job["error"])}</div>' if job.get("error") else ''}
<div class="cards">{cards}</div>
{''.join(f'<div class="banner warn">{_e(w)}</div>' for w in warnings)}
<p>{' '.join(actions)}</p>
{f'<h2>Flags</h2><ul>{flags}</ul>' if flags else ''}
{issue_form}
{f'<details><summary>How the issue was split (notes)</summary><p class="card">{_e(notes)}</p></details>' if notes else ''}
<details{' open' if articles else ''}><summary>Articles ({len(shown)}{f", {len(articles) - len(shown)} excluded" if len(articles) != len(shown) else ''})</summary>
<div class="scroll"><table><tr class="head"><th>#</th><th>Title</th><th>Kind</th><th>Pages</th><th>Length</th><th></th></tr>
{_articles_table(job_id, articles, csrf) or '<tr><td colspan="6" class="muted">None yet.</td></tr>'}</table></div></details>
<details><summary>Pages ({len(pages)})</summary><div class="scroll"><table>
<tr class="head"><th>Page</th><th>Read by</th><th>Text layer</th><th>Words matched</th><th>Flags</th></tr>{page_rows}</table></div></details>
{'<script>setTimeout(function(){location.reload()}, 5000)</script>' if active else ''}"""
    return _page(issue.get("label") or job["filename"], body, email=admin, **_flash(request))


@router.get("/admin/kb/jobs/{job_id}/articles/{n}")
async def article_page(request: Request, job_id: int, n: int):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    job = await jobs.get_job(job_id)
    articles = await jobs.job_articles(job_id) if job else []
    a = next((x for x in articles if x.n == n), None)
    if a is None:
        return _page("Article not found", f'<p><a href="/admin/kb/jobs/{job_id}">← The issue</a></p>'
                     '<h1>Article not found</h1>', email=admin, status=404)
    csrf = _csrf_token(admin)
    prev_n = next((x.n for x in reversed(articles) if x.n < n), None)
    next_n = next((x.n for x in articles if x.n > n), None)
    nav = " · ".join(filter(None, [
        f'<a href="/admin/kb/jobs/{job_id}/articles/{prev_n}">← previous</a>' if prev_n else "",
        f'<a href="/admin/kb/jobs/{job_id}">all articles</a>',
        f'<a href="/admin/kb/jobs/{job_id}/articles/{next_n}">next →</a>' if next_n else ""]))
    images = "".join(f'<p class="muted" style="margin:0">PDF page {p}</p>'
                     f'<img class="page-img" loading="lazy" src="/admin/kb/jobs/{job_id}/pages/{p}" alt="page {p}">'
                     for p in range(a.pdf_start, a.pdf_end + 1))
    toggle = "include" if a.excluded else "exclude"
    body = f"""<p>{nav}</p>
<h1>{_e(a.title)}</h1>
<p class="muted">{_e(a.kind.replace('_', ' '))} · pages {a.start_page}–{a.end_page} · {len(a.text.split()):,} words
{' · by ' + _e(a.author) if a.author else ''}{' · contents entry: ' + _e(a.toc_title) if a.toc_title else ''}
{' · <strong>excluded</strong> ' + _e(a.note) if a.excluded else ''}</p>
{f'<p class="card"><strong>Summary:</strong> {_e(a.summary)}</p>' if a.summary else ''}
<p><form class="inline" method="post" action="/admin/kb/jobs/{job_id}/articles/{n}/rename">
<input type="hidden" name="csrf" value="{csrf}"><input type="text" name="title" value="{_e(a.title)}" style="width:420px">
<button>Rename</button></form>
<form class="inline" method="post" action="/admin/kb/jobs/{job_id}/articles/{n}/{toggle}">
<input type="hidden" name="csrf" value="{csrf}"><button>{toggle.capitalize()}</button></form></p>
<div class="two-col"><div><div class="article-text">{_e(a.text)}</div></div><div>{images}</div></div>"""
    return _page(a.title, body, email=admin, wide=True)


# Page images: previews are stored in R2 while the job reads the pages; a missing one is rendered on demand from
# the PDF (downloaded once per job and kept in a small cache) and stored for next time.
_pdf_cache: "OrderedDict[int, str]" = OrderedDict()
_pdf_cache_lock = asyncio.Lock()
PDF_CACHE_SIZE = 2


async def _local_pdf(job: dict, store: storage.Storage) -> str:
    async with _pdf_cache_lock:
        path = _pdf_cache.get(job["id"])
        if path and os.path.exists(path):
            _pdf_cache.move_to_end(job["id"])
            return path
        path = os.path.join(tempfile.gettempdir(), f"cgpt-review-{job['id']}-{int(time.time())}.pdf")
        await asyncio.to_thread(store.download, job["object_key"], path)
        _pdf_cache[job["id"]] = path
        while len(_pdf_cache) > PDF_CACHE_SIZE:
            _, old = _pdf_cache.popitem(last=False)
            try:
                os.remove(old)
            except OSError:
                pass
        return path


def _render_preview(path: str, page: int) -> bytes:
    with PdfFile(path) as pdf:
        return pdf.preview(page)


@router.get("/admin/kb/jobs/{job_id}/pages/{page}")
async def page_image(request: Request, job_id: int, page: int):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    job = await jobs.get_job(job_id)
    store = _storage()
    if job is None or store is None or not 1 <= page <= (job["pages"] or 0):
        return Response(status_code=404)
    key = storage.preview_key(job["object_key"], page)
    if await asyncio.to_thread(store.head, key) is None:
        path = await _local_pdf(job, store)
        jpeg = await asyncio.to_thread(_render_preview, path, page)
        await asyncio.to_thread(store.put_bytes, key, jpeg, "image/jpeg")
    url = await asyncio.to_thread(store.presign_get, key)
    return RedirectResponse(url, status_code=302, headers={"Cache-Control": "private, max-age=600"})


# ---------- job actions ----------

async def _job_action(request: Request, job_id: int, action, done: str, *, back: str | None = None):
    admin, form, denied = await _checked_form(request)
    if denied:
        return denied
    try:
        await action(admin, form)
    except jobs.JobError as e:
        return _back(back or f"/admin/kb/jobs/{job_id}", str(e), "bad")
    return _back(back or f"/admin/kb/jobs/{job_id}", done)


@router.post("/admin/kb/jobs/{job_id}/publish")
async def publish(request: Request, job_id: int):
    return await _job_action(request, job_id, lambda admin, form: jobs.request_publish(job_id, admin),
                             "Publishing… the articles are searchable once the status says Published.")


@router.post("/admin/kb/jobs/{job_id}/rerun-split")
async def rerun_split(request: Request, job_id: int):
    return await _job_action(request, job_id, lambda admin, form: jobs.request_rerun(job_id, "split", admin),
                             "Splitting the issue again (the page reads are kept).")


@router.post("/admin/kb/jobs/{job_id}/rerun")
async def rerun_all(request: Request, job_id: int):
    return await _job_action(request, job_id, lambda admin, form: jobs.request_rerun(job_id, "all", admin),
                             "Processing the whole issue again.")


@router.post("/admin/kb/jobs/{job_id}/stop")
async def stop(request: Request, job_id: int):
    return await _job_action(request, job_id, lambda admin, form: jobs.stop_job(job_id, admin), "Stopped.")


@router.post("/admin/kb/jobs/{job_id}/delete")
async def delete(request: Request, job_id: int):
    admin, form, denied = await _checked_form(request)
    if denied:
        return denied
    job = await jobs.get_job(job_id)
    try:
        removed = await jobs.delete_job(job_id, admin, _storage())
    except jobs.JobError as e:
        return _back(f"/admin/kb/jobs/{job_id}", str(e), "bad")
    return _back("/admin/kb?" + urllib.parse.urlencode({"kb": job["kb_id"]}) if job else "/admin/kb",
                 "Deleted the upload" + (f" and removed its {removed} articles from search." if removed
                                                       else "."))


@router.post("/admin/kb/jobs/{job_id}/issue")
async def edit_issue(request: Request, job_id: int):
    return await _job_action(request, job_id, lambda admin, form: jobs.edit_issue(
        job_id, form.get("label", ""), form.get("published_date", ""), form.get("source", ""), admin),
        "Saved. (If the issue is already published, publish it again for search to show the change.)")


@router.post("/admin/kb/jobs/{job_id}/articles/{n}/rename")
async def rename_article(request: Request, job_id: int, n: int):
    return await _job_action(request, job_id, lambda admin, form: jobs.edit_article(
        job_id, n, title=form.get("title", ""), by=admin), "Renamed.", back=f"/admin/kb/jobs/{job_id}/articles/{n}")


@router.post("/admin/kb/jobs/{job_id}/articles/{n}/exclude")
async def exclude_article(request: Request, job_id: int, n: int):
    return await _job_action(request, job_id, lambda admin, form: jobs.edit_article(
        job_id, n, excluded=True, by=admin), "Excluded: it won't be published.")


@router.post("/admin/kb/jobs/{job_id}/articles/{n}/include")
async def include_article(request: Request, job_id: int, n: int):
    return await _job_action(request, job_id, lambda admin, form: jobs.edit_article(
        job_id, n, excluded=False, by=admin), "Included again.")
