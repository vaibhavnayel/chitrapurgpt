"""/admin/kb, "Videos and websites": collections filled from somewhere other than magazine PDFs (ingest/sources.py).

YouTube: the admin chooses the channel's two exported CSV files; the browser sends their text, the app works out
the entries and shows what would change (new, changed, removed), and Apply writes them in the background.
Same sign-in, admin check and CSRF tokens as admin.py.
"""
from __future__ import annotations

import logging
import urllib.parse

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

import kb
from admin import _csrf_token, _e, _fmt_time, _page, _require_admin
from admin_kb import _checked_form, _json_admin
from ingest import sources, youtube

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_CSV_CHARS = 30_000_000
SYNC_STATUS = {"crawling": ("Reading the site…", "run"), "checked": ("Waiting for you to press Apply", "on"),
               "applying": ("Updating the collection…", "run"),
               "done": ("Done", "yes"), "failed": ("Failed", "bad"), "discarded": ("Discarded", "off")}


def _sync_tag(status: str) -> str:
    text, cls = SYNC_STATUS.get(status, (status, "off"))
    return f'<span class="tag {cls}">{_e(text)}</span>'


async def section(kbs: list[dict], csrf: str) -> str:
    """The "Videos and websites" part of /admin/kb."""
    srcs = await sources.list_sources()
    rows = []
    for s in srcs:
        last = (await sources.list_syncs(s["id"], 1) or [None])[0]
        refresh = (f"""<form class="inline" method="post" action="/admin/kb/sources/{s['id']}/refresh">
<input type="hidden" name="csrf" value="{csrf}"><button>Read the site again now</button></form>"""
                   if s["kind"] == "website" and not (last and last["status"] in ("crawling", "applying")) else "")
        rows.append(f"""<tr><td><strong>{_e(sources.KINDS.get(s['kind'], s['kind']))}</strong><br>
<span class="muted">{_e(s['name'])}</span></td><td>{_e(s['kb_name'])}</td><td data-label="Entries">{s['articles']:,}</td>
<td data-label="Last updated">{_fmt_time(s['synced_at']) if s['synced_at'] else '–'}</td>
<td>{f'<a href="/admin/kb/syncs/{last["id"]}">{_sync_tag(last["status"])}</a>' if last else '–'} {refresh}</td></tr>""")
    yt_kb = next((k["id"] for k in kbs if "youtube" in (k["name"] + k["id"]).lower()), None)
    options = "".join(f'<option value="{_e(k["id"])}"{" selected" if k["id"] == yt_kb else ""}>{_e(k["name"])}</option>'
                      for k in kbs if k["id"] != kb.DEFAULT_KB)
    web_kb = next((k["id"] for k in kbs if "website" in (k["name"] + k["id"]).lower()), None)
    web_options = "".join(f'<option value="{_e(k["id"])}"{" selected" if k["id"] == web_kb else ""}>{_e(k["name"])}</option>'
                          for k in kbs if k["id"] != kb.DEFAULT_KB)
    return f"""
<h2>Videos and websites</h2>
<div class="card"><form id="yt" style="margin:0">
<strong>YouTube video lists</strong>
<p class="muted" style="margin:4px 0 8px">Choose both CSV files from the channel export (the video list and the
playlist list). You'll see what would change before anything is updated. Send new lists whenever you like: only new
and changed videos are added again, and videos missing from the new lists are taken out.</p>
<label>Collection <select id="yt-kb" required>{'' if yt_kb else '<option value="">Choose…</option>'}{options}</select></label><br>
<input id="yt-file" type="file" accept=".csv,text/csv" multiple required style="margin-top:8px"><br>
<button class="primary" style="margin-top:10px">Check the lists</button>
<p id="yt-status" class="muted" style="margin:8px 0 0"></p></form></div>
<script>{YOUTUBE_JS.replace("__CSRF__", csrf)}</script>
<details><summary>Add a website</summary><div class="card" style="margin-top:8px">
<form method="post" action="/admin/kb/websites/add" style="margin:0"><input type="hidden" name="csrf" value="{csrf}">
<p class="muted" style="margin:0 0 8px">ChitrapurGPT reads every page of the site (not the files it links to: PDFs,
pictures and audio are listed by name with their links), which takes about an hour, then shows what it found before
anything is added. After that it reads the site again once a week and adds what's new or changed by itself.</p>
<label>Collection <select name="kb_id" required>{'' if web_kb else '<option value="">Choose…</option>'}{web_options}</select></label><br>
<label>Address<br><input type="url" name="start" value="https://www.chitrapurmath.net/" required style="width:360px"></label><br>
<button class="primary" style="margin-top:8px">Read the site</button></form></div></details>
{f'''<div class="scroll"><table><tr class="head"><th>Source</th><th>Collection</th><th>Entries</th><th>Last updated</th>
<th>Latest update</th></tr>{"".join(rows)}</table></div>''' if rows else ''}"""


YOUTUBE_JS = r"""
(function(){
var CSRF = "__CSRF__";
var form = document.getElementById('yt'), status = document.getElementById('yt-status');
form.addEventListener('submit', async function(ev){
  ev.preventDefault();
  var files = Array.from(document.getElementById('yt-file').files), kbId = document.getElementById('yt-kb').value;
  if (!kbId) { status.textContent = 'Please choose the collection.'; return; }
  if (files.length !== 2) { status.textContent = 'Please choose both CSV files (the video list and the playlist list).'; return; }
  window.cgptUploading = true; form.querySelector('button').disabled = true;
  try {
    status.textContent = 'Reading the files…';
    var sent = [];
    for (var f of files) sent.push({name: f.name, text: await f.text()});
    status.textContent = 'Checking the lists…';
    var r = await fetch('/admin/kb/youtube/check', {method:'POST', credentials:'same-origin',
      headers:{'Content-Type':'application/json','X-CSRF-Token':CSRF}, body: JSON.stringify({kb_id: kbId, files: sent})});
    var j = await r.json().catch(function(){ return {error:'The server sent an unexpected reply ('+r.status+').'}; });
    if (!r.ok) throw new Error(j.error || ('HTTP '+r.status));
    location.href = j.redirect;
  } catch (e) {
    status.textContent = e.message; window.cgptUploading = false; form.querySelector('button').disabled = false;
  }
});
})();
"""


@router.post("/admin/kb/youtube/check")
async def youtube_check(request: Request):
    admin, data, denied = await _json_admin(request)
    if denied:
        return denied
    target = await kb.get_kb(str(data.get("kb_id", "")))
    if target is None or target["id"] == kb.DEFAULT_KB:
        return JSONResponse({"error": "Please choose the collection for the videos."}, status_code=400)
    files = data.get("files")
    if (not isinstance(files, list) or len(files) != 2
            or not all(isinstance(f, dict) and isinstance(f.get("text"), str) for f in files)):
        return JSONResponse({"error": "Please choose both CSV files."}, status_code=400)
    if sum(len(f["text"]) for f in files) > MAX_CSV_CHARS:
        return JSONResponse({"error": "These files are too big to be the YouTube lists."}, status_code=400)
    try:
        videos, playlists = youtube.sort_files([(str(f.get("name", "file"))[:200], f["text"]) for f in files])
        entries, notes = youtube.build_entries(videos, playlists)
    except youtube.CsvProblem as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not entries:
        return JSONResponse({"error": "These lists have no public or unlisted videos."}, status_code=400)
    source = await sources.get_or_create_source(target["id"], "youtube", youtube.SOURCE_NAME, admin)
    sync_id = await sources.create_check(source["id"], entries, notes, admin)
    logger.info("Admin %s checked YouTube lists for %s: sync %d", admin, target["id"], sync_id)
    return JSONResponse({"sync_id": sync_id, "redirect": f"/admin/kb/syncs/{sync_id}"})


def _titles(label: str, titles: list[str], total: int) -> str:
    if not total:
        return ""
    more = f'<li class="muted">and {total - len(titles):,} more</li>' if total > len(titles) else ""
    return (f"<details><summary>{_e(label)} ({total:,})</summary><ul>"
            + "".join(f"<li>{_e(t)}</li>" for t in titles) + more + "</ul></details>")


@router.get("/admin/kb/syncs/{sync_id}")
async def sync_page(request: Request, sync_id: int):
    admin, denied = _require_admin(request)
    if denied:
        return denied
    sync = await sources.get_sync(sync_id)
    if sync and sync["status"] == "applying" and await sources.mark_interrupted():
        sync = await sources.get_sync(sync_id)
    source = await sources.get_source(sync["source_id"]) if sync else None
    if sync is None or source is None:
        return _page("Update not found", '<p><a href="/admin/kb">← Collections &amp; uploads</a></p>'
                     "<h1>Update not found</h1>", email=admin, status=404)
    target = await kb.get_kb(source["kb_id"]) or {"name": source["kb_id"]}
    csrf = _csrf_token(admin)
    plan, result = sync["plan"] or {}, sync["result"] or {}
    status = sync["status"]

    def stat(n, label):
        return f'<div class="stat"><div class="n">{n}</div><div class="l">{label}</div></div>'

    if status == "crawling":
        p = sync["result"] or {}
        cards = "".join([stat(f"{p.get('pages_read', 0):,}", "pages read so far"),
                         stat(f"{p.get('pages_found', 0):,}", "pages found"),
                         stat(f"{p.get('minutes', 0):.0f} min", "so far")])
    elif status == "done":
        cards = "".join([stat(f"{result.get('added', 0):,}", "added"), stat(f"{result.get('changed', 0):,}", "updated"),
                         stat(f"{result.get('removed', 0):,}", "taken out"),
                         stat(f"{result.get('unchanged', 0):,}", "unchanged"),
                         stat(f"${result.get('cost_usd', 0):.2f}", "cost"),
                         stat(f"{result.get('secs', 0) / 60:.1f} min", "time")])
    else:
        cards = "".join([stat(f"{plan.get('new', 0):,}", "new"), stat(f"{plan.get('changed', 0):,}", "changed"),
                         stat(f"{plan.get('removed', 0):,}", "to take out"),
                         stat(f"{plan.get('unchanged', 0):,}", "unchanged")])
    notes = "".join(f"<li>{_e(n)}</li>" for n in sync["notes"] or [])
    lists = (_titles("New", plan.get("new_titles", []), plan.get("new", 0))
             + _titles("Changed", plan.get("changed_titles", []), plan.get("changed", 0))
             + _titles("To take out", plan.get("removed_titles", []), plan.get("removed", 0)))
    actions = ""
    if status == "checked" and sync["has_entries"]:
        nothing = not (plan.get("new") or plan.get("changed") or plan.get("removed"))
        actions = f"""<div class="card">
<form class="inline" method="post" action="/admin/kb/syncs/{sync_id}/apply"><input type="hidden" name="csrf" value="{csrf}">
<button class="primary">{'Apply (nothing changes)' if nothing else 'Apply: update ' + _e(target['name'])}</button></form>
<form class="inline" method="post" action="/admin/kb/syncs/{sync_id}/discard"><input type="hidden" name="csrf" value="{csrf}">
<button>Discard</button></form>
<p class="muted" style="margin:8px 0 0">Applying takes a minute or two. People who have {_e(target['name'])} switched on
can search the new entries as soon as it's done.</p></div>"""
    refresh = ('<script>setTimeout(function(){ location.reload(); }, 15000);</script>'
               if status in ("applying", "crawling") else "")
    body = f"""<p><a href="/admin/kb">← Collections &amp; uploads</a></p>
<h1>{_e(sources.KINDS.get(source['kind'], source['kind']))} → {_e(target['name'])}</h1>
<p>{_sync_tag(status)} <span class="muted">{'sent' if source['kind'] == 'youtube' else 'started'} by
{_e(sync['created_by'] or '–')}, {_fmt_time(sync['created_at'])}</span></p>
{'<p class="muted">Reading the site page by page, a few at a time so it isn&#39;t slowed down. This takes about an hour; you can close this page.</p>' if status == 'crawling' else ''}
{f'<div class="banner warn">{_e(sync["error"])}</div>' if sync.get("error") else ''}
<div class="cards">{cards}</div>
{f'<h2>Notes</h2><ul>{notes}</ul>' if notes else ''}
{actions}
{f'<h2>What changes</h2>{lists}' if lists and status != 'done' else ''}
{refresh}"""
    return _page(f"{target['name']}: update", body, email=admin, **_flash(request))


def _flash(request: Request) -> dict:
    return {"flash": request.query_params.get("msg", ""),
            "flash_kind": "bad" if request.query_params.get("kind") == "bad" else "ok"}


def _back(path: str, message: str, kind: str = "ok") -> RedirectResponse:
    return RedirectResponse(f"{path}?" + urllib.parse.urlencode({"msg": message, "kind": kind}), status_code=303)


@router.post("/admin/kb/syncs/{sync_id}/apply")
async def sync_apply(request: Request, sync_id: int):
    admin, form, denied = await _checked_form(request)
    if denied:
        return denied
    page = f"/admin/kb/syncs/{sync_id}"
    if await sources.start_apply(sync_id, admin):
        return _back(page, "Updating the collection. This page refreshes by itself.")
    return _back(page, "This update can't be applied (it was already applied, discarded or replaced).", "bad")


@router.post("/admin/kb/websites/add")
async def website_add(request: Request):
    admin, form, denied = await _checked_form(request)
    if denied:
        return denied
    target = await kb.get_kb(form.get("kb_id", ""))
    start = (form.get("start") or "").strip()
    parsed = urllib.parse.urlsplit(start)
    if target is None or target["id"] == kb.DEFAULT_KB:
        return _back("/admin/kb", "Please choose the collection for the website.", "bad")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return _back("/admin/kb", "Please give the website's full address, starting with https://", "bad")
    start = urllib.parse.urlunsplit(("https", parsed.netloc, parsed.path or "/", "", ""))
    source = await sources.get_or_create_source(target["id"], "website", parsed.netloc.removeprefix("www."), admin,
                                                {"start": start})
    sync_id = await sources.start_refresh(source["id"], admin)
    if sync_id is None:
        return _back("/admin/kb", "That website is already being read. It's in the list below.", "bad")
    return _back(f"/admin/kb/syncs/{sync_id}", "Reading the site. This takes about an hour; you can close this page "
                                               "and come back to it from Collections & uploads.")


@router.post("/admin/kb/sources/{source_id}/refresh")
async def source_refresh(request: Request, source_id: int):
    admin, form, denied = await _checked_form(request)
    if denied:
        return denied
    source = await sources.get_source(source_id)
    if source is None or source["kind"] != "website":
        return _back("/admin/kb", "Nothing was changed: unknown website.", "bad")
    sync_id = await sources.start_refresh(source_id, admin)
    if sync_id is None:
        return _back("/admin/kb", "That website is already being read or updated.", "bad")
    return _back(f"/admin/kb/syncs/{sync_id}", "Reading the site again. You can close this page and come back later.")


@router.post("/admin/kb/syncs/{sync_id}/discard")
async def sync_discard(request: Request, sync_id: int):
    admin, form, denied = await _checked_form(request)
    if denied:
        return denied
    if await sources.discard(sync_id):
        return _back("/admin/kb", "Discarded. Nothing was changed.")
    return _back(f"/admin/kb/syncs/{sync_id}", "This update can't be discarded now.", "bad")
