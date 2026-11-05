"""Linked collections (ingest/sources.py) and the YouTube lists (ingest/youtube.py): parsing the two CSV exports,
working out what an update changes, and applying it to the articles table, the chunks and the in-memory index.

The parsing tests need nothing. The database tests need VECTOR_DATABASE_URL (a local Postgres with pgvector): a
throwaway database is created on that server and dropped afterwards. Embeddings are fakes.
  .venv/bin/python -m pytest tests/test_sources.py -q
"""
import asyncio
import csv
import io
import os
import sys
import urllib.parse
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest import sources, youtube  # noqa: E402
from ingest.sources import Entry, make_plan  # noqa: E402

BASE_DSN = os.environ.get("VECTOR_DATABASE_URL", "")
needs_db = pytest.mark.skipif(not BASE_DSN, reason="VECTOR_DATABASE_URL (local pgvector server) not set")

VIDEO_HEAD = ["Title", "Privacy Status", "Content Format", "Published Date", "URL", "View Count", "Duration (s)",
              "Playlists", "Description"]
PLAYLIST_HEAD = ["Playlist Name", "Playlist Privacy", "Playlist URL", "Position in Playlist", "Video Title",
                 "Video Privacy", "Content Format", "Published Date", "Video URL", "View Count", "Duration (s)"]


def to_csv(head, rows) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(head)
    w.writerows(rows)
    return "\ufeff" + out.getvalue()  # the exports start with a byte-order mark


def video(vid, title, privacy="Public", desc="", fmt="Standard Video", date="2023-05-04T07:51:32Z", secs="3530"):
    return [title, privacy, fmt, date, f"https://www.youtube.com/watch?v={vid}", "10", secs, "", desc]


def in_list(pid, name, pos, vid, title, privacy="Public", list_privacy="Public", date="2023-05-04T07:51:32Z"):
    return [name, list_privacy, f"https://www.youtube.com/playlist?list={pid}", str(pos), title, privacy,
            "Standard Video", date, f"https://www.youtube.com/watch?v={vid}", "5", "60"]


def sample_files(desc="Swadhyaya on the Bhagavadgita, chapter 12.") -> list[tuple[str, str]]:
    videos = to_csv(VIDEO_HEAD, [
        video("aaa", "Gita Swadhyaya Day 1", desc=desc),
        video("bbb", "Rathotsava 2023", privacy="Unlisted", desc="Shirali, February 2023", fmt="Live Stream", secs="0"),
        video("ccc", "Private rehearsal", privacy="Private"),
        video("ddd", "Old upload", privacy="Deleted / Removed"),
    ])
    playlists = to_csv(PLAYLIST_HEAD, [
        in_list("PL1", "Gita Swadhyaya", 2, "eee", "Gita Swadhyaya Day 2 by Dr. Sudha Tinaikar"),
        in_list("PL1", "Gita Swadhyaya", 1, "aaa", "Gita Swadhyaya Day 1"),
        in_list("PL1", "Gita Swadhyaya", 3, "ccc", "Private rehearsal", privacy="Private"),
        in_list("PL2", "for deletion", 1, "aaa", "Gita Swadhyaya Day 1", list_privacy="Private"),
        in_list("PL3", "Utsava, 2023", 1, "bbb", "Rathotsava 2023", privacy="Unlisted", list_privacy="Unlisted"),
    ])
    return [("youtube_playlists_contents 2026.csv", playlists), ("youtube_master_catalog 2026.csv", videos)]


# ---------- parsing ----------

def test_files_are_told_apart_by_their_columns():
    videos, playlists = youtube.sort_files(sample_files())
    assert len(videos) == 4 and len(playlists) == 5
    with pytest.raises(youtube.CsvProblem, match="video list"):
        youtube.sort_files(sample_files()[:1])
    with pytest.raises(youtube.CsvProblem, match="doesn't look like"):
        youtube.sort_files([("notes.csv", "Name,Age\nA,1\n"), sample_files()[1]])


def test_entries_cover_public_and_unlisted_videos_and_playlists():
    entries, notes = youtube.build_entries(*youtube.sort_files(sample_files()))
    by_key = {e.key: e for e in entries}
    assert sorted(by_key) == ["playlist:PL1", "playlist:PL3", "video:aaa", "video:bbb", "video:eee"]

    gita = by_key["video:aaa"]
    assert gita.metadata == {"source": "YouTube", "published_date": "2023-05-04", "title": "Gita Swadhyaya Day 1",
                             "url": "https://www.youtube.com/watch?v=aaa", "kind": "YouTube video",
                             "privacy": "public", "length": "59 min", "playlists": "Gita Swadhyaya"}
    assert "published 4 May 2023, 59 min long" in gita.text and "for deletion" not in gita.text
    assert gita.text.endswith("Description:\nSwadhyaya on the Bhagavadgita, chapter 12.")

    rath = by_key["video:bbb"]
    assert rath.metadata["privacy"] == "unlisted" and rath.metadata["kind"] == "YouTube live stream"
    assert "unlisted" in rath.text and "long" not in rath.text, "a live stream's 0 s length isn't shown"
    assert rath.metadata["playlists"] == "Utsava, 2023", "a comma in a playlist name is fine"

    only = by_key["video:eee"]  # only in the playlist file: no description
    assert "Description" not in only.text and only.metadata["title"].endswith("Sudha Tinaikar")

    pl = by_key["playlist:PL1"]
    assert pl.title == "Playlist: Gita Swadhyaya" and pl.metadata["videos"] == 2
    assert pl.text.index("1. Gita Swadhyaya Day 1") < pl.text.index("2. Gita Swadhyaya Day 2"), "playlist order"
    assert "Private rehearsal" not in pl.text
    assert "[unlisted]" in by_key["playlist:PL3"].text

    assert notes[0] == "3 videos (1 of them unlisted) and 2 playlists will be searchable."
    assert "Left out: 1 deleted, 1 private videos." in notes
    assert "Private playlists left out: for deletion." in notes
    assert any(n.startswith("1 videos appear only in the playlist file") for n in notes)


def test_plan_finds_new_changed_removed_and_unchanged():
    one, two, three = (Entry(f"video:{k}", k, f"text {k}", {"source": "YouTube", "title": k}) for k in "abc")
    current = {one.article_id(7): "a", two.article_id(7): "b", "src7:video:z:0123456789ab": "z"}
    two_changed = Entry("video:b", "b", "new text b", two.metadata)
    plan = make_plan(7, [one, two_changed, three, three], current)
    assert [e.key for e in plan.new] == ["video:c"] and [e.key for e in plan.changed] == ["video:b"]
    assert plan.removed == ["src7:video:z:0123456789ab"] and plan.removed_titles == ["z"]
    assert plan.unchanged == 1 and plan.replaced_ids == [two.article_id(7)]
    assert one.article_id(7) != Entry("video:a", "a", "text a", {"source": "YouTube", "title": "a2"}).article_id(7)


# ---------- the database ----------

@pytest.fixture(scope="module")
def db():
    if urllib.parse.urlsplit(BASE_DSN).hostname not in ("localhost", "127.0.0.1", None, ""):
        pytest.skip("VECTOR_DATABASE_URL must be a local database for these tests")
    import kb
    import kb_index
    from test_kb import with_db
    from test_vector_store import FakeEmbeddings
    from vector_store import PgVectorStore

    name = f"cgpt_sources_test_{uuid.uuid4().hex[:8]}"
    admin = make_conninfo(BASE_DSN, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    url = with_db(BASE_DSN, name)
    saved_env, saved_index = os.environ.get("VECTOR_DATABASE_URL"), kb_index.INDEX
    os.environ["VECTOR_DATABASE_URL"] = url
    try:
        store = PgVectorStore(FakeEmbeddings(), dsn=url)
        store.ensure_schema()
        store.close()
        kb.ensure_schema(url, legacy_jsonl=None)
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("INSERT INTO knowledge_base (id, name, source_prefix) VALUES ('youtube', 'Youtube Channel', 'YT')")
        kb_index.INDEX = kb_index.KnowledgeIndex()
        kb_index.INDEX.load(url)
        yield url
    finally:
        kb_index.INDEX = saved_index
        if saved_env is None:
            os.environ.pop("VECTOR_DATABASE_URL", None)
        else:
            os.environ["VECTOR_DATABASE_URL"] = saved_env
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def run(coro_fn):
    import kb

    async def go():
        try:
            return await coro_fn()
        finally:
            await kb.close_pool()
    return asyncio.run(go())


class CountingEmbed:
    def __init__(self):
        from test_vector_store import hashed
        self.hashed, self.texts = hashed, []

    def __call__(self, texts):
        self.texts += texts
        return [self.hashed(t) for t in texts]


async def wait_done(sync_id: int) -> dict:
    for _ in range(200):
        sync = await sources.get_sync(sync_id)
        if sync["status"] != "applying":
            return sync
        await asyncio.sleep(0.05)
    raise AssertionError("the update didn't finish")


@needs_db
def test_check_apply_and_update(db):
    import kb_index
    embed = CountingEmbed()

    async def scenario():
        entries, notes = youtube.build_entries(*youtube.sort_files(sample_files()))
        source = await sources.get_or_create_source("youtube", "youtube", youtube.SOURCE_NAME, "admin@x")
        assert (await sources.get_or_create_source("youtube", "youtube", youtube.SOURCE_NAME, "admin@x"))["id"] == source["id"]
        first = await sources.create_check(source["id"], entries, notes, "admin@x")
        sync = await sources.get_sync(first)
        assert sync["status"] == "checked" and sync["plan"]["new"] == 5 and sync["has_entries"]
        assert await sources.start_apply(first, "admin@x", embed=embed, db=db)
        assert not await sources.start_apply(first, "admin@x", embed=embed, db=db), "only once"
        sync = await wait_done(first)
        assert sync["status"] == "done" and not sync["has_entries"], sync.get("error")
        assert sync["result"]["added"] == 5 and sync["result"]["embedded"] == 5
        assert (await sources.list_sources())[0]["articles"] == 5

        # The in-memory index has them, and searches limited to the collection find them.
        docs = kb_index.INDEX.documents(["youtube"])
        assert sorted(d.metadata["title"] for d in docs)[0] == "Gita Swadhyaya Day 1"
        assert kb_index.INDEX.documents(["sunbeam"]) == []
        assert docs[0].metadata["url"].startswith("https://www.youtube.com/")

        # A new export: one description edited, one video gone (made private). Only the edited one is embedded.
        embed.texts.clear()
        files = sample_files(desc="Swadhyaya on the Bhagavadgita, chapter 12 (revised).")
        videos = to_csv(VIDEO_HEAD, [video("aaa", "Gita Swadhyaya Day 1", desc="Swadhyaya on the Bhagavadgita, chapter 12 (revised)."),
                                     video("bbb", "Rathotsava 2023", privacy="Private")])
        files = [files[0], ("videos.csv", videos)]
        entries, notes = youtube.build_entries(*youtube.sort_files(files))
        second = await sources.create_check(source["id"], entries, notes, "admin@x")
        plan = (await sources.get_sync(second))["plan"]
        assert (plan["new"], plan["changed"], plan["removed"]) == (0, 1, 2), plan
        assert set(plan["removed_titles"]) == {"Rathotsava 2023", "Playlist: Utsava, 2023"}
        assert await sources.start_apply(second, "admin@x", embed=embed, db=db)
        sync = await wait_done(second)
        assert sync["result"]["changed"] == 1 and sync["result"]["removed"] == 2
        assert len(embed.texts) == 1 and "(revised)" in embed.texts[0]
        titles = sorted(d.metadata["title"] for d in kb_index.INDEX.documents(["youtube"]))
        assert titles == ["Gita Swadhyaya Day 1", "Gita Swadhyaya Day 2 by Dr. Sudha Tinaikar", "Playlist: Gita Swadhyaya"]
        assert "(revised)" in next(d.page_content for d in kb_index.INDEX.documents(["youtube"])
                                   if d.metadata["title"] == "Gita Swadhyaya Day 1")

        # Re-sending the same lists changes nothing and embeds nothing.
        embed.texts.clear()
        third = await sources.create_check(source["id"], entries, notes, "admin@x")
        assert (await sources.get_sync(third))["plan"]["unchanged"] == 3
        assert await sources.discard(third) and not await sources.discard(third)
        assert (await sources.get_sync(third))["status"] == "discarded" and embed.texts == []

        # The files sent are kept with the update (the page shows which lists are in use).
        videos_rows, playlist_rows, sent = youtube.read_files(sample_files())
        assert [f["what"] for f in sent] == ["video list", "playlist list"] and sent[0]["rows"] == 4
        named = await sources.create_check(source["id"], entries, notes, "admin@x", files=sent)
        assert (await sources.get_sync(named))["files"] == sent
        assert (await sources.list_syncs(source["id"], 1))[0]["files"] == sent
        await sources.discard(named)

        # A check replaced by a newer one can't be applied.
        a = await sources.create_check(source["id"], entries, notes, "admin@x")
        b = await sources.create_check(source["id"], entries, notes, "admin@x")
        assert (await sources.get_sync(a))["status"] == "superseded"
        assert not await sources.start_apply(a, "admin@x", embed=embed, db=db)
        await sources.discard(b)

    run(scenario)
    import chunking
    with psycopg.connect(db) as conn:
        n_chunks = conn.execute("SELECT count(*) FROM kb_chunk c JOIN kb_article a ON a.id = c.article_id "
                                "WHERE a.source_id IS NOT NULL").fetchone()[0]
        expected = sum(len(chunking.article_chunks(t, c)) for t, c in conn.execute(
            "SELECT metadata->>'title', content FROM kb_article WHERE source_id IS NOT NULL"))
        orphans = conn.execute("SELECT count(*) FROM kb_chunk c LEFT JOIN kb_article a ON a.id = c.article_id "
                               "WHERE a.id IS NULL").fetchone()[0]
    assert n_chunks == expected and orphans == 0


@needs_db
def test_a_long_entry_gets_chunks(db):
    import chunking
    long_text = " ".join(f"word{i}" for i in range(3000))
    entry = Entry("video:long", "Long talk", long_text, {"source": "YouTube", "title": "Long talk"})

    async def scenario():
        source = await sources.get_or_create_source("youtube", "youtube", "Long", "admin@x")
        sync_id = await sources.create_check(source["id"], [entry], [], "admin@x")
        assert await sources.start_apply(sync_id, "admin@x", embed=CountingEmbed(), db=db)
        return (await wait_done(sync_id))["result"]

    result = run(scenario)
    assert result["chunks"] == len(chunking.article_chunks("Long talk", long_text)) > 1
    with psycopg.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM kb_chunk WHERE article_id = %s",
                            (entry.article_id(_source_id(conn, "Long")),)).fetchone()[0] == result["chunks"]


def _source_id(conn, name: str) -> int:
    return conn.execute("SELECT id FROM kb_source WHERE name = %s", (name,)).fetchone()[0]


@needs_db
def test_a_failed_apply_changes_nothing_and_can_be_retried(db):
    def broken(texts):
        raise RuntimeError("embeddings are down")

    async def scenario():
        source = await sources.get_or_create_source("youtube", "youtube", "Other channel", "admin@x")
        entries = [Entry("video:q1", "Q1", "text q1", {"source": "YouTube", "title": "Q1"})]
        sync_id = await sources.create_check(source["id"], entries, [], "admin@x")
        assert await sources.start_apply(sync_id, "admin@x", embed=broken, db=db)
        sync = await wait_done(sync_id)
        assert sync["status"] == "checked" and "embeddings are down" in sync["error"] and sync["has_entries"]
        assert await sources.current_articles(source["id"]) == {}
        assert await sources.start_apply(sync_id, "admin@x", embed=CountingEmbed(), db=db)
        assert (await wait_done(sync_id))["status"] == "done"
        assert list((await sources.current_articles(source["id"])).values()) == ["Q1"]

        # Left "applying" by a process that stopped: put back so it can be applied again.
        again = await sources.create_check(source["id"], entries, [], "admin@x")
        pool = await sources._pool()
        await pool.execute("UPDATE kb_sync SET status = 'applying', started_at = now() WHERE id = $1", again)
        assert await sources.mark_interrupted() == 0, "a recent one may still be running (e.g. on the live site)"
        await pool.execute("UPDATE kb_sync SET started_at = now() - interval '1 hour' WHERE id = $1", again)
        assert await sources.mark_interrupted() == 1
        assert (await sources.get_sync(again))["status"] == "checked"

    run(scenario)
