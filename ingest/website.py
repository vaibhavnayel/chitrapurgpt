"""Website collections (ingest/sources.py): read a site page by page and turn it into entries.

The crawl starts at the site's home page and follows links on the same site, politely (a few pages at a time, with
a pause), skipping files (PDFs, pictures, audio), pages that need a login and the paths listed in the source's
config. Each page with something to say becomes an entry: its title, address and text. Links on a page to PDFs,
audio and YouTube are kept with their names, so answers can point to them; the files themselves aren't read.

chitrapurmath.net specifics (all in DEFAULT_CONFIG or small helpers below): a page's own content sits between the
"breadcrumbs" bar and the footer; a photo album (gallery/default/view-gallery?id=N) becomes one entry with the
album's name and month (from the links to it) and every caption, with how many photos share it; audio pages list
their recordings (a title above each <audio> player).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from html.parser import HTMLParser

from ingest.sources import Entry

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "start": "https://www.chitrapurmath.net/",
    # Paths left out: files, the shop and bookings (need a login), the video gallery (the Youtube Channel
    # collection has those), live stream and radio (nothing to read), the gallery's album lists (each album is
    # its own entry), the glossary (filled in by JavaScript: the pages are empty).
    "skip": ["/assets/", "/vendorAssets/", "/myAssets/", "/documents/", "/booking", "/site/login", "/user/",
             "/shop", "/cart", "/gallery/default/videos", "/gallery/default/clips", "/site/live-stream",
             "/site/radio", "/site/sanskrit-index", "/site/signup", "/site/request-password-reset"],
    "lists": ["/gallery/default/index", "/gallery"],   # read for their links only (album names), not entries
    "content_start": 'class="breadcrumbs',
    "content_end": 'class="footer-v1',
    "max_pages": 8000,
    "concurrency": 3,
    "pause_s": 0.3,
}
SITE_NAME = "chitrapurmath.net"
FILE_EXT = re.compile(r"\.(jpe?g|png|gif|webp|svg|bmp|tiff?|css|js|ico|pdf|mp3|mp4|m4a|wav|ogg|aac|zip|rar|docx?|"
                      r"xlsx?|pptx?|ttf|woff2?|eot|xml|txt|csv)$", re.I)
MEDIA = re.compile(r"\.(pdf|mp3|m4a|wav|ogg|aac|mp4|docx?|xlsx?|pptx?)$", re.I)
MIN_CHARS = 150          # a page with less text than this (after its title) is left out
MAX_TOKENS = 7000        # longer pages become parts (the embeddings model reads up to 8,191 tokens)
ALBUM_PATH = "/gallery/default/view-gallery"
USER_AGENT = "Mozilla/5.0 (compatible; ChitrapurGPT/1.0; +https://chitrapurgpt.onrender.com)"


# ---------- reading one page ----------

BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "blockquote",
         "section", "article", "header", "dd", "dt", "hr", "pre", "figure", "figcaption"}
SKIP_TAGS = {"script", "style", "noscript", "select", "option", "button", "svg", "iframe", "template", "textarea"}
SKIP_CLASSES = {"pagination", "breadcrumb", "nav", "nav-tabs", "nav-pills", "sr-only", "hide", "hidden",
                "navbar-toggle"}
VOID = {"br", "img", "input", "hr", "meta", "link", "source", "area", "base", "col", "embed", "param", "track", "wbr"}


@dataclass
class Page:
    url: str
    title: str = ""
    text: str = ""
    media: list[tuple[str, str]] = field(default_factory=list)   # (name, url) of PDFs, audio, YouTube
    captions: list[str] = field(default_factory=list)            # photo captions (albums)
    links: list[str] = field(default_factory=list)               # absolute hrefs, for the crawl
    album_names: dict[str, tuple[str, str]] = field(default_factory=dict)  # album id -> (name, "Oct-2026")


class _Reader(HTMLParser):
    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.out: list[str] = []
        self.skip_depth = 0
        self.stack: list[tuple[str, bool]] = []    # (tag, skipping this subtree)
        self.h1 = ""
        self.headings: list[str] = []
        self.media: list[tuple[str, str]] = []
        self.captions: list[str] = []
        self._in = {"h1": None, "heading": None, "a": None, "caption": None}
        self._pending: tuple[str, list[str]] | None = None
        self._last_heading_text = ""
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = set((a.get("class") or "").split())
        skip = tag in SKIP_TAGS or bool(classes & SKIP_CLASSES) or a.get("aria-hidden") == "true"
        href = a.get("href")
        if tag == "a" and href:
            url = urllib.parse.urljoin(self.base, href.strip())
            self.links.append(url)
        if tag in ("audio", "source") and a.get("src") and MEDIA.search(urllib.parse.urlsplit(a["src"]).path):
            self.media.append((self._last_heading_text or "Recording",
                               urllib.parse.urljoin(self.base, a["src"].strip())))
        if tag == "img" and not self.skip_depth:
            pass
        if tag in VOID:
            if tag in BLOCK and not self.skip_depth:
                self.out.append("\n")
            return
        self.stack.append((tag, skip))
        if skip:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in BLOCK:
            self.out.append("\n")
        if tag == "h1":
            self._in["h1"] = []
        if tag in ("h2", "h3", "h4", "h5"):
            self._in["heading"] = []
        if "caption" in classes:
            self._in["caption"] = []
        if tag == "a" and href:
            self._in["a"] = (urllib.parse.urljoin(self.base, href.strip()), [])

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        # pop to the matching tag (HTML in the wild isn't always well nested)
        while self.stack:
            t, skip = self.stack.pop()
            if skip:
                self.skip_depth -= 1
            if t == tag:
                break
        if self.skip_depth:
            return
        if tag == "h1" and self._in["h1"] is not None:
            self.h1 = self.h1 or _clean("".join(self._in["h1"]))
            self._in["h1"] = None
        if tag in ("h2", "h3", "h4", "h5") and self._in["heading"] is not None:
            text = _clean("".join(self._in["heading"]))
            if text:
                self.headings.append(text)
                self._last_heading_text = text
            self._in["heading"] = None
        if tag == "a" and self._in["a"] is not None:
            url, parts = self._in["a"]
            name = _clean("".join(parts))
            path = urllib.parse.urlsplit(url)
            if MEDIA.search(path.path) or "youtube.com" in path.netloc or "youtu.be" in path.netloc:
                if name:
                    self.media.append((name, url))
                else:  # an icon link: the name is the text that follows (a PDF icon above its title)
                    self._pending = (url, [])
            self._in["a"] = None
            self.out.append(" ")  # links side by side in a menu stay separate words
        if tag in BLOCK:
            self.out.append("\n")
            if self._pending is not None and _clean("".join(self._pending[1])):
                self.media.append((_clean("".join(self._pending[1])), self._pending[0]))
                self._pending = None
        if tag == "div" and self._in["caption"] is not None:
            text = _clean(" ".join(self._in["caption"]))
            if text:
                self.captions.append(text)
            self._in["caption"] = None

    def handle_data(self, data):
        if self.skip_depth:
            return
        self.out.append(data)
        if self._pending is not None:
            self._pending[1].append(data)
        for k in ("h1", "heading", "caption"):
            if self._in[k] is not None:
                self._in[k].append(data)
        if self._in["a"] is not None:
            self._in["a"][1].append(data)


def _encode(url: str) -> str:
    """A link that survives being written in an answer (file names here have spaces and brackets)."""
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.netloc, urllib.parse.quote(urllib.parse.unquote(p.path), safe="/@:+,;=~!$&*-._"),
                                    p.query, ""))


def _clean(text: str) -> str:
    return " ".join(unescape(text).replace("\xa0", " ").split())


NOISE = re.compile(r"does not support the (audio|video|audop)|^Your (browser|bbser)", re.I)


def _tidy(text: str) -> str:
    lines = [" ".join(line.replace("\xa0", " ").split()) for line in text.split("\n")]
    lines = ["" if NOISE.search(line) else line for line in lines]
    out, blank = [], False
    for line in lines:
        if not line:
            blank = bool(out)
            continue
        if blank:
            out.append("")
        out.append(line)
        blank = False
    return "\n".join(out).strip()


def content_region(html: str, config: dict) -> str:
    start, end = config.get("content_start"), config.get("content_end")
    i = html.find(start) if start else -1
    if i >= 0:
        i = html.rfind("<", 0, i)
        html = html[i:]
    j = html.find(end) if end else -1
    if j > 0:
        html = html[:html.rfind("<", 0, j)]
    return html


def read_page(url: str, html: str, config: dict) -> Page:
    page = Page(url=url)
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    links = _Reader(url)
    links.feed(html)  # every link on the page (menus included) for the crawl
    page.links = links.links
    reader = _Reader(url)
    reader.feed(content_region(html, config))
    page.text = _tidy("".join(reader.out))
    page.media = list(dict.fromkeys((name, _encode(u)) for name, u in reader.media + script_files(url, html)))
    page.captions = reader.captions
    page.title = page_title(reader.h1, reader.headings, _clean(title.group(1)) if title else "", url)
    for link in page.links:  # album links carry the album's name and month
        p = urllib.parse.urlsplit(link)
        if p.path == ALBUM_PATH:
            q = urllib.parse.parse_qs(p.query)
            if q.get("id") and q.get("name"):
                page.album_names[q["id"][0]] = (_clean(q["name"][0]), (q.get("my") or [""])[0])
    return page


def script_files(base: str, html: str) -> list[tuple[str, str]]:
    """Files listed in a page's scripts as {"name": ..., "filePath": ...} (the Publications page builds its list of
    magazines that way)."""
    import json
    out = []
    for m in re.finditer(r'\{[^{}]*"filePath"\s*:\s*"[^"]+"[^{}]*\}', html):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        name, path = _clean(str(d.get("name") or "")), str(d.get("filePath") or "")
        if name and MEDIA.search(path):
            writer = _clean(str(d.get("writer") or ""))
            out.append((f"{name} (by {writer})" if writer else name, urllib.parse.urljoin(base, "/" + path.lstrip("/"))))
    return out



def page_title(h1: str, headings: list[str], html_title: str, url: str) -> str:
    """The section heading, made specific by the page's own first heading where it has one (a news story:
    "A Magical Evening ... - Prarthanavarga - Reports")."""
    h1 = re.sub(r"\(Go back to .*?\)", "", h1).strip()
    first = headings[0] if headings else ""
    if "story" in url and first and first.lower() != h1.lower():
        return f"{first.rstrip(': ')} - {h1}" if h1 else first
    return h1 or first or html_title or url


# ---------- entries ----------

def canonical(url: str, start: str) -> str | None:
    """The address a page is known by: https, the start's host, sorted query, no fragment; albums by id only.
    None for links to leave alone (other sites, files, mail, javascript)."""
    p = urllib.parse.urlsplit(url)
    host = urllib.parse.urlsplit(start).netloc
    if p.scheme not in ("http", "https") or p.netloc.removeprefix("www.") != host.removeprefix("www."):
        return None
    if FILE_EXT.search(p.path):
        return None
    query = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
    if p.path == ALBUM_PATH:
        query = [(k, v) for k, v in query if k == "id"]
    query = sorted((k, v) for k, v in query if not k.startswith("_"))
    path = re.sub(r"/{2,}", "/", p.path) or "/"
    return urllib.parse.urlunsplit(("https", host, path, urllib.parse.urlencode(query), ""))


def skipped(url: str, config: dict) -> bool:
    path = urllib.parse.urlsplit(url).path
    return any(path.startswith(s) or (s.endswith("/") and s in path) for s in config.get("skip", []))


def is_list(url: str, config: dict) -> bool:
    path = urllib.parse.urlsplit(url).path.rstrip("/") or "/"
    return any(path == s.rstrip("/") for s in config.get("lists", []))


def _month(my: str) -> tuple[str | None, str]:
    """("2026-10-01", "October 2026") from "Oct-2026"."""
    try:
        d = datetime.strptime(my, "%b-%Y")
    except ValueError:
        return None, ""
    return d.date().isoformat(), f"{d:%B %Y}"


def album_entry(page: Page, album_id: str, name: str, my: str) -> Entry | None:
    counts = Counter(page.captions)
    if not counts and not name:
        return None
    iso, month = _month(my)
    title = f"Photo album: {name}" if name else f"Photo album {album_id}"
    lines = [title, f"A photo album on {SITE_NAME}" + (f", {month}" if month else "")
             + f", {len(page.captions)} photo{'s' if len(page.captions) != 1 else ''}.", f"Link: {page.url}", "",
             "Photo captions:"]
    lines += [f"- {c}" + (f" ({n} photos)" if n > 1 else "") for c, n in counts.items()]
    return Entry(key=f"album:{album_id}", title=title, text="\n".join(lines),
                 metadata={"source": SITE_NAME, "published_date": iso, "title": title, "url": page.url,
                           "kind": "photo album", "photos": len(page.captions)})


def page_entries(page: Page) -> list[Entry]:
    body = page.text
    if body.startswith(page.title):
        body = body[len(page.title):].strip()
    if len(body) < MIN_CHARS and not page.media:
        return []
    media = ""
    if page.media:
        media = "\n\nLinks on this page:\n" + "\n".join(f"- {name}: {url}" for name, url in page.media)
    path = urllib.parse.urlsplit(page.url)
    key = "page:" + path.path + (f"?{path.query}" if path.query else "")
    kind = "audio recordings" if any(MEDIA.search(u) and re.search(r"\.(mp3|m4a|wav|ogg|aac)$", u, re.I)
                                     for _, u in page.media) else "web page"
    parts = split_text(body + media, MAX_TOKENS)
    entries = []
    for n, part in enumerate(parts, 1):
        title = page.title if len(parts) == 1 else f"{page.title} (part {n} of {len(parts)})"
        text = f"{title}\nA {kind if kind != 'audio recordings' else 'page of audio recordings'} on {SITE_NAME}: " \
               f"{page.url}\n\n{part}"
        entries.append(Entry(key=key if len(parts) == 1 else f"{key}#part{n}", title=title, text=text,
                             metadata={"source": SITE_NAME, "published_date": None, "title": title,
                                       "url": page.url, "kind": kind}))
    return entries


def split_text(text: str, max_tokens: int) -> list[str]:
    """Parts of at most max_tokens, split between paragraphs (or lines, or words, if a paragraph is too long)."""
    from ingest.assemble import count_tokens
    if count_tokens(text) <= max_tokens:
        return [text]
    parts, current = [], []
    for para in re.split(r"(\n\n+)", text):
        candidate = "".join(current + [para])
        if current and count_tokens(candidate) > max_tokens:
            parts.append("".join(current).strip())
            current = [para]
        else:
            current.append(para)
    if current and "".join(current).strip():
        parts.append("".join(current).strip())
    out = []
    for p in parts:  # a single paragraph over the limit: cut it by words
        while count_tokens(p) > max_tokens:
            words = p.split(" ")
            lo, hi = 1, len(words)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if count_tokens(" ".join(words[:mid])) <= max_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            out.append(" ".join(words[:lo]))
            p = " ".join(words[lo:])
        if p.strip():
            out.append(p)
    return out


# ---------- the crawl ----------

async def crawl(config: dict, *, progress=None, fetch=None) -> tuple[list[Entry], list[str]]:
    """Read the site. Returns its entries and notes for the admin. `fetch(url) -> (status, final url, content type,
    text)` replaces the network in tests."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    start = canonical(config["start"], config["start"])
    max_pages = int(config["max_pages"])
    seen, pages, failed = {start}, {}, Counter()
    album_names: dict[str, tuple[str, str]] = {}
    texts_seen: dict[str, str] = {}
    queue: asyncio.Queue[str] = asyncio.Queue()
    queue.put_nowait(start)
    t0, done = time.monotonic(), 0
    client = None
    if fetch is None:
        import httpx
        client = httpx.AsyncClient(follow_redirects=True, timeout=30, headers={"User-Agent": USER_AGENT})

        async def fetch(url):
            for attempt in range(3):
                try:
                    r = await client.get(url)
                    return r.status_code, str(r.url), r.headers.get("content-type", ""), r.text
                except Exception:  # noqa: BLE001
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 * (attempt + 1))

    async def worker():
        nonlocal done
        while True:
            url = await queue.get()
            try:
                try:
                    status, final, ctype, html = await fetch(url)
                except Exception as e:  # noqa: BLE001
                    failed[e.__class__.__name__] += 1
                    continue
                if status != 200 or "html" not in ctype:
                    failed[f"HTTP {status}" if status != 200 else "not a page"] += 1
                    continue
                final_c = canonical(final, start)
                if final_c is None or skipped(final_c, config):
                    continue  # redirected away (e.g. to the login page)
                page = read_page(final_c, html, config)
                album_names.update(page.album_names)
                if final_c not in pages and (final_c == url or final_c not in seen):
                    pages[final_c] = page
                for link in page.links:
                    c = canonical(link, start)
                    if c and c not in seen and not skipped(c, config) and len(seen) < max_pages:
                        seen.add(c)
                        queue.put_nowait(c)
                await asyncio.sleep(float(config["pause_s"]))
            finally:
                done += 1
                queue.task_done()
                if progress and done % 50 == 0:
                    await progress({"pages_read": done, "pages_found": len(seen),
                                    "minutes": round((time.monotonic() - t0) / 60, 1)})

    workers = [asyncio.create_task(worker()) for _ in range(int(config["concurrency"]))]
    try:
        await queue.join()
    finally:
        for w in workers:
            w.cancel()
        if client is not None:
            await client.aclose()

    entries, keys = [], set()
    albums = 0
    for url, page in sorted(pages.items()):
        p = urllib.parse.urlsplit(url)
        if is_list(url, config):
            continue
        if p.path == ALBUM_PATH:
            album_id = (urllib.parse.parse_qs(p.query).get("id") or [""])[0]
            name, my = album_names.get(album_id, ("", ""))
            e = album_entry(page, album_id, name, my)
            new = [e] if e else []
            albums += bool(e)
        else:
            digest = hashlib.sha256(page.text.encode()).hexdigest()
            if digest in texts_seen:  # the same page under two addresses
                continue
            texts_seen[digest] = url
            new = page_entries(page)
        for e in new:
            if e.key not in keys:
                keys.add(e.key)
                entries.append(e)
    notes = [f"Read {len(pages):,} pages in {(time.monotonic() - t0) / 60:.0f} minutes: {len(entries):,} entries "
             f"({albums:,} photo albums, {len(entries) - albums:,} pages)."]
    if failed:
        notes.append("Pages that couldn't be read: " + ", ".join(f"{n:,} ({k})" for k, n in failed.most_common()) + ".")
    if len(seen) >= max_pages:
        notes.append(f"Stopped at the limit of {max_pages:,} pages; some pages may be missing.")
    return entries, notes
