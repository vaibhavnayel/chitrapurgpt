"""Website collections (ingest/website.py): reading pages, photo albums and audio lists, the crawl with a fake
network, and the weekly re-read through ingest/sources.py (applied by itself, or held when too much would go).

The database test needs VECTOR_DATABASE_URL (a local Postgres with pgvector), like tests/test_sources.py.
  .venv/bin/python -m pytest tests/test_website.py -q
"""
import asyncio
import sys
from pathlib import Path

import psycopg
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest import sources, website  # noqa: E402
from test_sources import CountingEmbed, db, needs_db, run, wait_done  # noqa: E402,F401

SITE = "https://www.example-math.net"


def shell(body: str, title: str = "Site") -> str:
    """A page the way chitrapurmath.net lays them out: menus, the breadcrumbs bar, the content, the footer."""
    return f"""<html><head><title>{title}</title><link href="/assets/css/footers/footer-v1.css"></head><body>
<div class="header"><a href="/">Home</a> <a href="/site/about">About</a> <a href="/news?id=7">News</a>
<a href="/gallery">Photos</a> <a href="/booking">Book a seva</a> <a href="/site/audio">Audio</a></div>
<div class="breadcrumbs"><div class="container">{body}</div>
<div class="footer-v1"><a href="/site/terms">Terms</a> Copyright Shri Chitrapur Math</div></body></html>"""


ABOUT = shell("""<h1 class="pull-left">About Ma<u>th</u>s</h1><ul class="breadcrumb"><li><a href="/">Home</a></li></ul>
<div class="row"><p>Shri Chitrapur Math, Shirali, is the spiritual centre of the Chitrapur Saraswat community.</p>
<p>In 1953 property was acquired in Bengaluru in the name of Lord Bhavanishankar. The Math was used as a summer
retreat and transit accommodation for Parama Pujya Shrimad Anandashram Swamiji.</p>
<script>var x = "not text";</script><select><option>2016</option></select></div>""")
STORY = shell("""<h1 class="pull-left">Prarthanavarga - Reports</h1><div class="post-content">
<h3>A Magical Evening of Learning and Laughter: </h3><p>The Parijna Activity Workshop at Santacruz on 5th July 2025,
with games and crafts for children between 5 and 12. Teachers from Santacruz and Vile Parle planned it for weeks.</p>
</div>""")
GALLERY = shell("""<h1 class="pull-left">Photos</h1><ul class="pagination"><li><a href="/gallery/default/index?page=2">2</a></li></ul>
<div class="thumbnails"><a href='/gallery/default/view-gallery?id=650&name=Mandala Puja at Mangaluru (8, 9 June 2026)&my=Jun-2026'
title="Mandala Puja"><img src="/documents/gallery/x/thumb/a.jpg"></a><div class="caption text-center"><h3>
<a href='/gallery/default/view-gallery?id=650&name=Mandala Puja at Mangaluru (8, 9 June 2026)&my=Jun-2026'>Mandala Puja</a></h3></div></div>""")
GALLERY2 = shell("""<h1>Photos</h1><a href='/gallery/default/view-gallery?id=651&name=Rathotsava&my=Feb-2026'>Rathotsava</a>""")
ALBUM = shell("""<h1 class="pull-left"><a href="/gallery"><small>(Go back to Photos)</small></a></h1>
<form class="form-horizontal hide"><select><option>2016</option></select></form>""" + "".join(
    f"""<div class="thumbnails"><a href="/documents/gallery/x/{n}.jpg" title="{cap}"><img src="/documents/gallery/x/thumb/{n}.jpg"
alt="{cap}"></a><div class="caption text-center"><h3>{cap}</h3></div></div>"""
    for n, cap in enumerate(["Kalasha Sthapana (8 June 2026)"] * 2 + ["Homa and Abhisheka (9 June 2026)"] * 3)) + """
<div class="hide"><div class="caption"><h3>template</h3></div></div>""")
ALBUM2 = shell("""<h1><small>(Go back to Photos)</small></h1><div class="caption"><h3>Rathotsava at Shirali, Shri Ram Bhat with the chariot</h3></div>""")
AUDIO = shell("""<h1 class="pull-left">Audio</h1><ul class="nav nav-pills"><li><a href="/site/audio?id=8">Ashirvachana</a></li></ul>
<h4 class="col-md-12">Ashirvachana</h4><div class="listing-item"><h4 class="title"><a>Pravachana by HH Shrimad Anandashram
Swamiji - Golden Jubilee 1965</a></h4><audio preload="none" controls src="/documents/audio/1965 Pravachan.mp3"/>
<i>Your bbser does not support the audop element.</i></audio></div>""")
DOCS = shell("""<h1>Documents</h1><div class="col-md-3"><a href="/documents/docs/Shri Lalita Sahasranam.pdf" target="_blank">
<i class="fa fa-file-pdf-o"></i></a><p class="text-center">SHRI LALITA SAHASRANAM</p></div>
<script>var magazineObj = [{"id":38,"name":"Parijna Patrika English - Vasanta Rutu 2023","writer":"Team Parijna Patrika",
"filePath":"documents\\/magazine\\/38_Parijna.pdf"}];</script>""")

PAGES = {
    f"{SITE}/": shell('<h1>Home</h1><a href="/site/about">About</a><a href="/news/default/story?id=300">Story</a>'
                      '<a href="/gallery/default/docs">Docs</a><a href="/documents/x.pdf">x</a>'
                      '<a href="https://www.youtube.com/@shrichitrapurmathvideos">YouTube</a>'),
    f"{SITE}/site/about": ABOUT,
    f"{SITE}/news?id=7": shell('<h1>News</h1><a href="/news/default/story?id=300&amp;_csrf=abc">A Magical Evening</a>'),
    f"{SITE}/news/default/story?id=300": STORY,
    f"{SITE}/gallery": GALLERY,
    f"{SITE}/gallery/default/index?page=2": GALLERY2,
    f"{SITE}/gallery/default/view-gallery?id=650": ALBUM,
    f"{SITE}/gallery/default/view-gallery?id=651": ALBUM2,
    f"{SITE}/site/audio": AUDIO,
    f"{SITE}/gallery/default/docs": DOCS,
    f"{SITE}/site/terms": shell("<h1>Terms</h1><p>Short.</p>"),
}
CONFIG = {"start": f"{SITE}/", "pause_s": 0, "skip": website.DEFAULT_CONFIG["skip"],
          "lists": website.DEFAULT_CONFIG["lists"]}


def fake_fetch(pages: dict, fetched: list):
    async def fetch(url):
        fetched.append(url)
        if url.endswith("/site/audio?id=8"):
            return 200, f"{SITE}/site/login", "text/html", "<html>log in</html>"  # (redirects are followed)
        if url in pages:
            return 200, url, "text/html; charset=UTF-8", pages[url]
        return 404, url, "text/html", "not found"
    return fetch


def crawl(pages=PAGES):
    fetched = []
    entries, notes = asyncio.run(website.crawl(CONFIG, fetch=fake_fetch(pages, fetched)))
    return {e.key: e for e in entries}, notes, fetched


# ---------- reading pages ----------

def test_a_page_keeps_its_content_without_menus_scripts_or_footer():
    page = website.read_page(f"{SITE}/site/about", ABOUT, website.DEFAULT_CONFIG)
    assert page.title == "About Maths", "inline tags inside a word don't split it"
    assert "Bhavanishankar" in page.text and "not text" not in page.text and "2016" not in page.text
    assert "Copyright" not in page.text and "Book a seva" not in page.text
    assert f"{SITE}/booking" in page.links, "menu links are still followed (and then filtered)"


def test_a_news_story_is_titled_by_its_own_heading():
    page = website.read_page(f"{SITE}/news/default/story?id=300", STORY, website.DEFAULT_CONFIG)
    assert page.title == "A Magical Evening of Learning and Laughter - Prarthanavarga - Reports"


def test_links_to_files_are_kept_with_their_names():
    page = website.read_page(f"{SITE}/gallery/default/docs", DOCS, website.DEFAULT_CONFIG)
    assert page.media == [("SHRI LALITA SAHASRANAM", f"{SITE}/documents/docs/Shri%20Lalita%20Sahasranam.pdf"),
                          ("Parijna Patrika English - Vasanta Rutu 2023 (by Team Parijna Patrika)",
                           f"{SITE}/documents/magazine/38_Parijna.pdf")]
    audio = website.read_page(f"{SITE}/site/audio", AUDIO, website.DEFAULT_CONFIG)
    assert audio.media == [("Pravachana by HH Shrimad Anandashram Swamiji - Golden Jubilee 1965",
                            f"{SITE}/documents/audio/1965%20Pravachan.mp3")]
    assert "bbser" not in audio.text


def test_titles_and_album_dates():
    t = website.page_title
    assert t("Gauravam", ["Shirali, Pt. Vishnudass Mangesh", "Profile"], "", f"{SITE}/gauravam?id=12") == \
        "Shirali, Pt. Vishnudass Mangesh - Gauravam", "one of a set (?id=)"
    assert t("About Maths", ["Shri Chitrapur Math, Bengaluru"], "", f"{SITE}/site/about-maths") == "About Maths"
    assert t("Publications", ["The Chitrapur Sunbeam"], "", f"{SITE}/publications/sunbeam", shared=True) == \
        "The Chitrapur Sunbeam - Publications", "a heading several pages share"
    page = website.Page(url=f"{SITE}/gallery/default/view-gallery?id=393",
                        captions=["HH Swamiji's visit to Shri Venkatramana Temple, Mangaluru (24 Sep 2024)"])
    e = website.album_entry(page, "393", "HH Swamiji", "")
    assert e.metadata["published_date"] == "2024-09-01" and "September 2024" in e.text


def test_canonical_addresses():
    c = lambda u: website.canonical(u, f"{SITE}/")  # noqa: E731
    assert c("http://example-math.net/news?id=7#top") == f"{SITE}/news?id=7"
    assert c(f"{SITE}/gallery/default/view-gallery?name=X&id=650&my=Jun-2026") == \
        f"{SITE}/gallery/default/view-gallery?id=650"
    assert c(f"{SITE}/documents/a.pdf") is None and c("https://www.youtube.com/x") is None
    assert c("mailto:a@b.c") is None and c(f"{SITE}/x?b=2&a=1") == f"{SITE}/x?a=1&b=2"


def test_long_pages_are_split_into_parts():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 300 for i in range(40))
    parts = website.split_text(text, 2000)
    from ingest.assemble import count_tokens
    assert len(parts) > 1 and all(count_tokens(p) <= 2000 for p in parts)
    assert "".join(parts).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")


# ---------- the crawl ----------

def test_crawl_makes_entries_for_pages_albums_and_lists():
    entries, notes, fetched = crawl()
    assert sorted(entries) == ["album:650", "album:651", "page:/", "page:/gallery/default/docs",
                               "page:/news/default/story?id=300", "page:/site/about", "page:/site/audio"]
    assert "YouTube: https://www.youtube.com/@shrichitrapurmathvideos" in entries["page:/"].text, \
        "a short page is kept when it links to files or videos"
    album = entries["album:650"]
    assert album.title == "Photo album: Mandala Puja at Mangaluru (8, 9 June 2026)"
    assert album.metadata["published_date"] == "2026-06-01" and album.metadata["photos"] == 5
    assert album.metadata["url"] == f"{SITE}/gallery/default/view-gallery?id=650"
    assert "- Kalasha Sthapana (8 June 2026) (2 photos)\n- Homa and Abhisheka (9 June 2026) (3 photos)" in album.text
    assert "template" not in album.text
    assert "Shri Ram Bhat" in entries["album:651"].text, "an album found on the second list page"
    audio = entries["page:/site/audio"]
    assert audio.metadata["kind"] == "audio recordings" and "1965%20Pravachan.mp3" in audio.text
    assert "SHRI LALITA SAHASRANAM: " in entries["page:/gallery/default/docs"].text
    # never fetched: files, other sites, skipped paths; the login redirect isn't an entry
    assert not any(u.endswith((".pdf", ".jpg")) or "/booking" in u or "youtube" in u for u in fetched)
    assert f"{SITE}/site/audio?id=8" in fetched and "page:/site/audio?id=8" not in entries
    assert "page:/site/terms" not in entries, "too short"
    assert "page:/news/default/story?id=300&_csrf=abc" not in entries
    assert notes[0].startswith("Read ") and "2 photo albums, 5 pages" in notes[0]
    assert len(notes) == 1, notes  # nothing failed: the login redirect is just left out


# ---------- the weekly re-read ----------

def test_extra_pages_are_read_even_though_no_menu_links_to_them():
    hidden = shell("<h1>Jai Guru Parijnan</h1><p>" + "Resources for the celebration: bhajans and texts. " * 5 + "</p>"
                   '<a href="/site/jgp-texts">Texts</a>')
    pages = {**PAGES, f"{SITE}/site/jgp-resource": hidden,
             f"{SITE}/site/jgp-texts": shell("<h1>JGP texts</h1><p>" + "The texts of the bhajans. " * 10 + "</p>")}
    fetched = []
    entries, _ = asyncio.run(website.crawl({**CONFIG, "extra": ["https://example-math.net/site/jgp-resource"]},
                                           fetch=fake_fetch(pages, fetched)))
    keys = {e.key for e in entries}
    assert {"page:/site/jgp-resource", "page:/site/jgp-texts"} <= keys, "the extra page and the pages it links to"
    assert "page:/site/jgp-resource" not in crawl(pages)[0], "not found without it"
    good, bad = website.check_extra(["https://www.example-math.net/site/jgp-resource", "", "example-math.net/site/x",
                                     "https://other.org/page", f"{SITE}/booking/form"], f"{SITE}/")
    assert good == [f"{SITE}/site/jgp-resource", f"{SITE}/site/x"]
    assert bad == ["https://other.org/page", f"{SITE}/booking/form"]


# ---------- the weekly re-read ----------

@needs_db
def test_refresh_is_applied_by_itself_unless_too_much_would_go(db, monkeypatch):
    pages = dict(PAGES)

    async def fake_crawl(config, progress=None):
        await progress({"pages_read": 1, "pages_found": 2, "minutes": 0})
        return await website.crawl(CONFIG, fetch=fake_fetch(pages, []))

    async def scenario():
        pool = await sources._pool()
        await pool.execute("INSERT INTO knowledge_base (id, name, source_prefix) VALUES ('scm', 'SCM Website', 'SCM') "
                           "ON CONFLICT DO NOTHING")
        source = await sources.get_or_create_source("scm", "website", "example-math.net", "admin@x",
                                                    {"start": f"{SITE}/"})
        # the first read: checked, waiting for an admin
        first = await sources.start_refresh(source["id"], "admin@x", crawl=fake_crawl)
        assert first and await sources.start_refresh(source["id"], "admin@x", crawl=fake_crawl) is None, "one at a time"
        sync = await wait_for(first, "checked")
        assert sync["plan"]["new"] == 7
        assert await sources.start_apply(first, "admin@x", embed=CountingEmbed(), db=db)
        assert (await wait_done(first))["status"] == "done"

        # weekly, nothing changed: done at once, nothing embedded
        embed = CountingEmbed()
        second = await sources.start_refresh(source["id"], "weekly refresh", auto=True, crawl=fake_crawl,
                                             embed=embed, db=db)
        sync = await wait_for(second, "done")
        assert sync["result"]["unchanged"] == 7 and embed.texts == []

        # weekly, one story edited: applied by itself
        pages[f"{SITE}/news/default/story?id=300"] = STORY.replace("for weeks", "for many weeks")
        third = await sources.start_refresh(source["id"], "weekly refresh", auto=True, crawl=fake_crawl,
                                            embed=embed, db=db)
        sync = await wait_for(third, "done")
        assert sync["result"]["changed"] == 1 and len(embed.texts) == 1

        # weekly, the site half down: held for an admin
        current = await sources.current_articles(source["id"])
        await pool.executemany("INSERT INTO kb_article (id, content, metadata, embedding, kb_id, source_id) "
                               "SELECT $1, 'x', '{\"title\": \"old\"}', embedding, kb_id, source_id FROM kb_article "
                               "WHERE id = $2", [(f"src{source['id']}:page:/old{i}:abc", next(iter(current)))
                                                 for i in range(30)])
        fourth = await sources.start_refresh(source["id"], "weekly refresh", auto=True, crawl=fake_crawl,
                                             embed=embed, db=db)
        sync = await wait_for(fourth, "checked")
        assert sync["plan"]["removed"] == 30 and "Not applied by itself" in sync["notes"][-1]

        # due: none (read this week); a source last read 8 days ago is
        assert await sources.refresh_due_sources() == []
        await pool.execute("UPDATE kb_sync SET created_at = now() - interval '8 days' WHERE source_id = $1",
                           source["id"])
        monkeypatch.setattr(sources, "_crawler", lambda kind: fake_crawl)
        monkeypatch.setattr(sources, "_embed", CountingEmbed())
        started = await sources.refresh_due_sources()
        assert len(started) == 1
        sync = await wait_for(started[0], "checked")  # still held: the 30 old pages would go
        assert sync["created_by"] == "weekly refresh" and sync["plan"]["removed"] == 30

    run(scenario)


async def wait_for(sync_id, *statuses):
    for _ in range(400):
        sync = await sources.get_sync(sync_id)
        if sync["status"] in statuses:
            return sync
        await asyncio.sleep(0.05)
    raise AssertionError(f"sync {sync_id} is {sync['status']}, not {statuses}: {sync.get('error')}")
