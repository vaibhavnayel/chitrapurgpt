"""The Diacritical words dashboard's database layer (diacritical/wordlist.py): the trial copy, Accept (changes, log,
pending, auto-resolve), the snapshot check, undo and its refusal, starting the trial again, live mode.

Needs a local Postgres: each test makes its own throwaway database on it and drops it afterwards.
  WORDS_TEST_DATABASE_URL=postgresql://postgres@localhost:5433/postgres .venv/bin/python -m pytest tests/test_words_db.py -q
"""
import os
import sys
import urllib.parse
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = os.environ.get("WORDS_TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not BASE, reason="WORDS_TEST_DATABASE_URL (a local Postgres) not set")

import psycopg  # noqa: E402

from diacritical import harvest, wordlist  # noqa: E402

LIVE = [("krishna", "kṛs͟hṇa"), ("krshna", "{{}}{{kṛs͟hṇa}}"), ("atma", "ātmā"), ("god", "{{gŏḍ}}{{god}}"),
        ("chitrapur", "{{}}{{chitrāpur}}"), ("puja", "pūjā"), ("shri", "shrī")]
TEXT = ("Shrī Kṛshṇa came to Chitrapur. The Ātma is one.\n"
        "We offered pūjā with the dīpa to Gaṇesha and Akartā.\n")


def with_db(url: str, name: str) -> str:
    u = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(u._replace(path="/" + name))


@pytest.fixture
def db(monkeypatch):
    if urllib.parse.urlsplit(BASE).hostname not in ("localhost", "127.0.0.1"):
        pytest.skip("WORDS_TEST_DATABASE_URL must be a local database")
    name = "words_test_" + uuid.uuid4().hex[:10]
    with psycopg.connect(BASE, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    url = with_db(BASE, name)
    with psycopg.connect(url) as c:
        c.execute("""CREATE TABLE diacritic_mappings (id serial PRIMARY KEY, plain_text varchar(255) UNIQUE NOT NULL,
                     diacritic_text varchar(255) NOT NULL, created_at timestamp, updated_at timestamp)""")
        c.execute("CREATE TABLE diacritic_mapping_changes (id serial PRIMARY KEY, note text)")
        for i, (p, r) in enumerate(LIVE, 1):
            c.execute("INSERT INTO diacritic_mappings (id, plain_text, diacritic_text, created_at, updated_at) "
                      "VALUES (%s, %s, %s, now(), now())", (i * 10, p, r))   # ids with gaps, like the real list
        c.execute("SELECT setval('diacritic_mappings_id_seq', 70)")
    monkeypatch.setenv("DIACRITICAL_DATABASE_URL", url)
    monkeypatch.setenv("DIACRITICAL_WORDS_LIST", "trial")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    wordlist._ready.clear()
    try:
        yield url
    finally:
        wordlist._ready.clear()
        with psycopg.connect(BASE, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def run(fn):
    with wordlist.connect() as conn:
        return fn(conn, wordlist.list_name())


def live_rows(url):
    with psycopg.connect(url) as c:
        return dict(c.execute("SELECT plain_text, diacritic_text FROM diacritic_mappings").fetchall())


def make_review(conn, name, text=TEXT):
    items = harvest.review_document([("talk.docx", [(None, text)])], wordlist.mappings(conn, name))
    return wordlist.create_review(conn, name, "document", "talk.docx", [{"name": "talk.docx"}], "a@gmail.com", items)


def decide(rid, choices):
    """choices: {plain word: decision}; every other row that needs a choice is skipped / our list."""
    def go(conn, name):
        rv = wordlist.get_review(conn, name, rid)
        d = rv["decisions"]
        for section in rv["items"]["sections"].values():
            for item in section:
                if item.get("kind") not in harvest.NEEDS_CHOICE:
                    if item["plain"] in choices and item.get("kind") == "unmarked":
                        d[item["id"]] = choices[item["plain"]]
                    continue
                default = {"choice": "skip" if item["kind"] == "new" else "list", "cap": d[item["id"]].get("cap")}
                d[item["id"]] = choices.get(item["plain"], default)
        wordlist.save_decisions(conn, name, rid, d)
    run(go)


def test_trial_copy_is_made_once_with_its_own_ids(db):
    rows = run(wordlist.entries)
    assert [(r["id"], r["plain_text"]) for r in rows][:2] == [(30, "atma"), (50, "chitrapur")]
    assert len(rows) == len(LIVE)
    run(lambda c, n: wordlist.save_entry(c, n, "a@gmail.com", plain="dipa", result="dīpa"))
    new = run(lambda c, n: wordlist.entry_by_plain(c, n, "dipa"))
    assert new["id"] == 71                                                    # the sequence was set past the copy
    assert "dipa" not in live_rows(db)                                         # the real list is untouched
    wordlist._ready.clear()
    assert len(run(wordlist.entries)) == len(LIVE) + 1                         # not copied again
    with psycopg.connect(db) as c:
        assert c.execute("SELECT count(*) FROM diacritic_mapping_changes").fetchone()[0] == 0


def test_accept_applies_logs_and_fills_pending(db):
    def setup(conn, name):
        conn.execute("INSERT INTO dia_pending (list_name, plain_text, suggested, reason) "
                     "VALUES ('trial', 'akarta', 'akartā', 'asked before')")
        return make_review(conn, name)
    rid = run(setup)
    decide(rid, {"krshna": {"choice": "doc:0", "cap": "keep"},          # the document is right
                 "atma": {"choice": "list", "cap": "capital"},          # our list, always capital
                 "akarta": {"choice": "add:0", "cap": "capital"},
                 "dipa": {"choice": "add:0", "cap": "normal"},
                 "ganesha": {"choice": "pending", "cap": "capital"},
                 "chitrapur": {"choice": "pending"}})
    done = run(lambda c, n: wordlist.accept_review(c, n, rid, "a@gmail.com"))
    assert harvest.plan_sentence(done["plan"]) == "Accept will add 2 words, change 2, and put 2 on the pending list."
    words = {r["plain_text"]: r["diacritic_text"] for r in run(wordlist.entries)}
    assert words["krshna"] == "{{}}{{kṛshṇa}}" and words["atma"] == "{{}}{{ātmā}}"
    assert words["akarta"] == "{{}}{{akartā}}" and words["dipa"] == "dīpa"
    assert live_rows(db)["krshna"] == "{{}}{{kṛs͟hṇa}}"                       # trial only
    batch = run(lambda c, n: wordlist.get_batch(c, n, done["batch"]))
    assert batch["kind"] == "review" and batch["review_id"] == rid and "2 added, 2 changed" in batch["summary"]
    log = run(lambda c, n: wordlist.changes(c, done["batch"]))
    assert sorted((x["op"], x["new_plain"] or x["old_plain"]) for x in log) == [
        ("add", "akarta"), ("add", "dipa"), ("update", "atma"), ("update", "krshna")]
    assert [x["old_result"] for x in log if x["new_plain"] == "krshna"] == ["{{}}{{kṛs͟hṇa}}"]
    pending = {p["plain_text"]: p for p in run(wordlist.pending)}
    assert set(pending) == {"ganesha", "chitrapur"}
    assert pending["ganesha"]["suggested"] == "{{}}{{gaṇesha}}" and pending["ganesha"]["source"].startswith("talk.docx")
    assert "Akartā" in pending["ganesha"]["context"] or pending["ganesha"]["context"]
    resolved = run(lambda c, n: wordlist.pending(c, n, resolved=True))
    assert [(r["plain_text"], r["resolution"]) for r in resolved] == [("akarta", f"added or changed in review {rid}")]
    rv = run(lambda c, n: wordlist.get_review(c, n, rid))
    assert rv["status"] == "accepted" and rv["batch_id"] == done["batch"]
    with pytest.raises(wordlist.WordListError, match="already finished"):
        run(lambda c, n: wordlist.accept_review(c, n, rid, "a@gmail.com"))


def test_accept_refuses_when_words_changed_meanwhile(db):
    rid = run(make_review)
    decide(rid, {"krshna": {"choice": "doc:0", "cap": "keep"}, "dipa": {"choice": "add:0", "cap": "normal"}})
    run(lambda c, n: wordlist.save_entry(c, n, "b@gmail.com", plain="krshna", result="kṛṣṇa",
                                         entry_id=wordlist.entry_by_plain(c, n, "krshna")["id"]))
    run(lambda c, n: wordlist.save_entry(c, n, "b@gmail.com", plain="dipa", result="dīpā"))
    with pytest.raises(wordlist.Conflict) as caught:
        run(lambda c, n: wordlist.accept_review(c, n, rid, "a@gmail.com"))
    assert sorted(caught.value.words) == [("dipa", "dīpā"), ("krshna", "kṛṣṇa")]
    words = {r["plain_text"]: r["diacritic_text"] for r in run(wordlist.entries)}
    assert words["krshna"] == "kṛṣṇa" and words["dipa"] == "dīpā"            # nothing was overwritten
    assert run(lambda c, n: wordlist.get_review(c, n, rid))["status"] == "open"
    run(lambda c, n: wordlist.refresh_review(c, n, rid, caught.value.words))
    rv = run(lambda c, n: wordlist.get_review(c, n, rid))
    items = {i["plain"]: i for sec in rv["items"]["sections"].values() for i in sec if "kind" in i}
    assert items["krshna"]["entry"] == "kṛṣṇa" and items["krshna"]["changed_meanwhile"] == "kṛṣṇa"
    assert "choice" not in rv["decisions"][items["krshna"]["id"]]
    assert [v for v, _ in harvest.choice_options(items["dipa"])] == ["pending", "skip"]
    assert set(harvest.undecided(rv["items"], rv["decisions"])) == {items["krshna"]["id"], items["dipa"]["id"]}


def test_undo_and_its_refusal(db):
    rid = run(make_review)
    decide(rid, {"krshna": {"choice": "doc:0", "cap": "keep"}, "dipa": {"choice": "add:0", "cap": "normal"},
                 "akarta": {"choice": "add:0", "cap": "normal"}})
    first = run(lambda c, n: wordlist.accept_review(c, n, rid, "a@gmail.com"))["batch"]
    before = {r["plain_text"]: r["diacritic_text"] for r in run(wordlist.entries)}
    undo = run(lambda c, n: wordlist.undo(c, n, first, "b@gmail.com"))
    after = {r["plain_text"]: (r["id"], r["diacritic_text"]) for r in run(wordlist.entries)}
    assert {k: v for k, (_, v) in after.items()} == dict(LIVE) and after["krshna"][0] == 20
    assert set(before) - set(after) == {"dipa", "akarta"}
    b = run(lambda c, n: wordlist.get_batch(c, n, first))
    assert b["undone_by"] == "b@gmail.com" and run(lambda c, n: wordlist.get_batch(c, n, undo))["kind"] == "undo"
    with pytest.raises(wordlist.WordListError, match="already undone"):
        run(lambda c, n: wordlist.undo(c, n, first, "a@gmail.com"))
    with pytest.raises(wordlist.WordListError, match="can't be undone"):
        run(lambda c, n: wordlist.undo(c, n, undo, "a@gmail.com"))

    # a deleted entry comes back with its own id; a word changed again later blocks the undo
    gid = run(lambda c, n: wordlist.entry_by_plain(c, n, "god"))["id"]
    deleted = run(lambda c, n: wordlist.delete_entry(c, n, "a@gmail.com", gid, ("god", "{{gŏḍ}}{{god}}")))
    edit = run(lambda c, n: wordlist.save_entry(c, n, "a@gmail.com", plain="puja", result="pūja",
                                                entry_id=wordlist.entry_by_plain(c, n, "puja")["id"]))
    run(lambda c, n: wordlist.save_entry(c, n, "a@gmail.com", plain="puja", result="pūjā-pūjā",
                                         entry_id=wordlist.entry_by_plain(c, n, "puja")["id"],
                                         expected=("puja", "pūja")))
    with pytest.raises(wordlist.WordListError, match="changed again after this.*puja"):
        run(lambda c, n: wordlist.undo(c, n, edit, "a@gmail.com"))
    run(lambda c, n: wordlist.undo(c, n, deleted, "a@gmail.com"))
    assert run(lambda c, n: wordlist.entry(c, n, gid))["plain_text"] == "god"
    with pytest.raises(wordlist.Conflict):                                   # the form showed an older value
        run(lambda c, n: wordlist.save_entry(c, n, "a@gmail.com", plain="puja", result="x",
                                             entry_id=wordlist.entry_by_plain(c, n, "puja")["id"],
                                             expected=("puja", "pūja")))
    with pytest.raises(wordlist.WordListError, match="already in the list"):
        run(lambda c, n: wordlist.save_entry(c, n, "a@gmail.com", plain="god", result="x"))


def test_saving_from_pending_resolves_it(db):
    def setup(conn, name):
        return conn.execute("INSERT INTO dia_pending (list_name, plain_text, suggested, reason) "
                            "VALUES ('trial', 'ganesha', 'gaṇesha', 'New word') RETURNING id").fetchone()["id"]
    pid = run(setup)
    batch = run(lambda c, n: wordlist.save_entry(c, n, "a@gmail.com", plain="ganesha", result="{{}}{{gaṇesha}}",
                                                 pending_id=pid))
    assert run(lambda c, n: wordlist.get_batch(c, n, batch))["kind"] == "pending"
    assert run(wordlist.pending) == []
    pid2 = run(setup)
    assert run(lambda c, n: wordlist.remove_pending(c, n, pid2, "a@gmail.com")) == "ganesha"
    assert run(lambda c, n: wordlist.pending(c, n, resolved=True))[0]["resolution"] == "removed"


def test_list_review_resolves_answered_pending_items(db):
    def setup(conn, name):
        for p in ("dipa", "puja", "ganesha"):
            conn.execute("INSERT INTO dia_pending (list_name, plain_text, suggested, reason) VALUES "
                         "('trial', %s, '', 'asked')", (p,))
        rows = [{"row": 2, "plain": "dipa", "result": "dīpa"}, {"row": 3, "plain": "puja", "result": "pūjā"}]
        items = harvest.review_list(rows, [], wordlist.mappings(conn, name))
        return wordlist.create_review(conn, name, "list", "answers.xlsx", [{"name": "answers.xlsx"}], "a@gmail.com",
                                      items)
    rid = run(setup)
    run(lambda c, n: wordlist.accept_review(c, n, rid, "a@gmail.com"))
    assert [p["plain_text"] for p in run(wordlist.pending)] == ["ganesha"]
    assert run(lambda c, n: wordlist.entry_by_plain(c, n, "dipa"))["diacritic_text"] == "dīpa"


def test_start_the_trial_again(db):
    run(lambda c, n: wordlist.save_entry(c, n, "a@gmail.com", plain="dipa", result="dīpa"))
    added = run(wordlist.batches)[0]["id"]
    reset = run(lambda c, n: wordlist.reset_trial(c, "a@gmail.com"))
    assert {r["plain_text"]: r["diacritic_text"] for r in run(wordlist.entries)} == dict(LIVE)
    assert run(lambda c, n: wordlist.get_batch(c, n, reset))["kind"] == "reset"
    with pytest.raises(wordlist.WordListError, match="started again"):
        run(lambda c, n: wordlist.undo(c, n, added, "a@gmail.com"))
    run(lambda c, n: wordlist.save_entry(c, n, "a@gmail.com", plain="dipa", result="dīpa"))  # ids still free


def test_live_mode_changes_the_real_list_and_never_mixes(db, monkeypatch):
    run(lambda c, n: wordlist.save_entry(c, n, "a@gmail.com", plain="dipa", result="dīpa"))     # trial
    monkeypatch.setenv("DIACRITICAL_WORDS_LIST", "live")
    assert run(lambda c, n: n) == "live"
    assert run(wordlist.batches) == [] and "dipa" not in run(wordlist.mappings)
    run(lambda c, n: wordlist.save_entry(c, n, "a@gmail.com", plain="seva", result="sevā"))
    assert live_rows(db)["seva"] == "sevā"
    monkeypatch.setenv("DIACRITICAL_DATABASE_URL", "")
    with pytest.raises(wordlist.NotConfigured):
        run(wordlist.entries)
    monkeypatch.setenv("DIACRITICAL_DATABASE_URL", db)
    monkeypatch.setenv("DATABASE_URL", db)
    with pytest.raises(wordlist.NotConfigured, match="refusing"):
        run(wordlist.entries)
