"""The Diacritical words dashboard's database layer (diacritical/wordlist.py): the trial copy, decisions read from a
filled report and applied in one transaction (changes, log, pending with notes, auto-resolve), rows skipped because the
list changed meanwhile, the confirm page's fingerprint, already-applied and trial/live refusals, the pending list round
trip, undo and its refusal, starting the trial again, live mode.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = os.environ.get("WORDS_TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not BASE, reason="WORDS_TEST_DATABASE_URL (a local Postgres) not set")

import psycopg  # noqa: E402

from diacritical import harvest, sheets, wordlist  # noqa: E402
from test_words_harvest import fill  # noqa: E402

LIVE = [("krishna", "kṛs͟hṇa"), ("krshna", "{{}}{{kṛs͟hṇa}}"), ("atma", "ātmā"), ("god", "{{gŏḍ}}{{god}}"),
        ("chitrapur", "{{}}{{chitrāpur}}"), ("puja", "pūjā"), ("shri", "shrī")]
TEXT = ("Shrī Kṛshṇa came to Chitrapur. The Ātma is one.\n"
        "We offered pūjā with the dīpa to Gaṇesha and Akartā.\n")
BY = "a@gmail.com"
DIFF, NEW, UNMARKED = "Different from our list", "New words", "Written without marks"


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


def words():
    return {r["plain_text"]: r["diacritic_text"] for r in run(wordlist.entries)}


def set_word(url, table, plain, result):
    """A change made elsewhere (e.g. on the old Mappings page), straight in the table."""
    with psycopg.connect(url) as c:
        if result is None:
            c.execute(f"DELETE FROM {table} WHERE plain_text = %s", (plain,))
        elif c.execute(f"UPDATE {table} SET diacritic_text = %s WHERE plain_text = %s", (result, plain)).rowcount == 0:
            c.execute(f"INSERT INTO {table} (plain_text, diacritic_text) VALUES (%s, %s)", (plain, result))


def make_review(text=TEXT):
    def go(conn, name):
        items = harvest.review_document([("talk.docx", [(None, text)])], wordlist.mappings(conn, name))
        return wordlist.create_review(conn, name, "document", "talk.docx", [{"name": "talk.docx"}], BY, items)
    return run(go)


def report(rid) -> bytes:
    def go(conn, name):
        rv = wordlist.get_review(conn, name, rid)
        return sheets.report_xlsx(rv["items"], {"review": rid, "list": rv["list_name"], "files": ["talk.docx"],
                                                "made_iso": rv["created_at"].isoformat(timespec="seconds")})
    return run(go)


def upload(rid, answers, data=None) -> dict:
    """Fill the report like a person, upload it; returns what the confirm page would show."""
    filled = sheets.read_filled("filled.xlsx", fill(data or report(rid), answers))
    run(lambda c, n: wordlist.store_report_answers(c, n, filled, BY, "filled.xlsx"))
    return run(lambda c, n: wordlist.plan_for(c, n, wordlist.get_review(c, n, rid)))


def apply(rid, p, by=BY):
    return run(lambda c, n: wordlist.apply(c, n, rid, by, p["fingerprint"]))


def test_trial_copy_is_made_once_with_its_own_ids(db):
    rows = run(wordlist.entries)
    assert [(r["id"], r["plain_text"]) for r in rows][:2] == [(30, "atma"), (50, "chitrapur")]
    assert len(rows) == len(LIVE)
    rid = make_review()
    apply(rid, upload(rid, {(NEW, "dipa"): {"Your decision": "Add"}}))
    new = [r for r in run(wordlist.entries) if r["plain_text"] == "dipa"][0]
    assert new["id"] == 71                                                    # the sequence was set past the copy
    assert "dipa" not in live_rows(db)                                         # the real list is untouched
    wordlist._ready.clear()
    assert len(run(wordlist.entries)) == len(LIVE) + 1                         # not copied again
    with psycopg.connect(db) as c:
        assert c.execute("SELECT count(*) FROM diacritic_mapping_changes").fetchone()[0] == 0


def test_apply_from_a_filled_report(db):
    run(lambda c, n: c.execute("INSERT INTO dia_pending (list_name, plain_text, suggested, reason) "
                               "VALUES ('trial', 'akarta', 'akartā', 'asked before')"))
    rid = make_review()
    p = upload(rid, {(DIFF, "krshna"): {"Your decision": "Document is right"},
                     (DIFF, "atma"): {"Your decision": "Our list is right", "Capital": "Always capital"},
                     (NEW, "akarta"): {"Your decision": "Add", "Capital": "Always capital"},
                     (NEW, "dipa"): {"Your decision": "Add"},
                     (NEW, "ganesha"): {"Notes for self": "ask the team"},                 # blank: pending, with the note
                     (UNMARKED, "chitrapur"): {"Your decision": "Pending"}})
    assert sheets.sentence(p) == ("This will add 2 words, change 2, put 2 on the pending list, and take 1 off the "
                                  "pending list.")                             # akarta: it is added
    done = apply(rid, p)
    assert sheets.sentence(done["plan"], done=True).startswith("Applied: added 2 words, changed 2, put 2")
    w = words()
    assert w["krshna"] == "{{}}{{kṛshṇa}}" and w["atma"] == "{{}}{{ātmā}}"
    assert w["akarta"] == "{{}}{{akartā}}" and w["dipa"] == "dīpa"
    assert live_rows(db)["krshna"] == "{{}}{{kṛs͟hṇa}}"                       # trial only
    batch = run(lambda c, n: wordlist.get_batch(c, n, done["batch"]))
    assert batch["kind"] == "review" and batch["review_id"] == rid
    assert batch["summary"] == f"Report {rid} (talk.docx): 2 added, 2 changed, 2 put on the pending list"
    log = run(lambda c, n: wordlist.changes(c, done["batch"]))
    assert sorted((x["op"], x["new_plain"] or x["old_plain"]) for x in log) == [
        ("add", "akarta"), ("add", "dipa"), ("update", "atma"), ("update", "krshna")]
    assert [x["old_result"] for x in log if x["new_plain"] == "krshna"] == ["{{}}{{kṛs͟hṇa}}"]
    pending = {p["plain_text"]: p for p in run(wordlist.pending)}
    assert set(pending) == {"ganesha", "chitrapur"}
    assert pending["ganesha"]["suggested"] == "{{}}{{gaṇesha}}" and pending["ganesha"]["notes"] == "ask the team"
    assert pending["ganesha"]["source"] == f"talk.docx (upload {rid})" and pending["ganesha"]["context"]
    resolved = run(lambda c, n: wordlist.pending(c, n, resolved=True))
    assert [(r["plain_text"], r["resolution"]) for r in resolved] == [
        ("akarta", f"added or changed from report {rid} (change {done['batch']})")]
    rv = run(lambda c, n: wordlist.get_review(c, n, rid))
    assert rv["status"] == "applied" and rv["batch_id"] == done["batch"]
    with pytest.raises(wordlist.WordListError, match="already applied on"):
        upload(rid, {})
    with pytest.raises(wordlist.WordListError, match="nothing waiting"):
        apply(rid, p)


def test_rows_changed_meanwhile_are_skipped_and_the_fingerprint_guards_apply(db):
    rid = make_review()
    answers = {(DIFF, "krshna"): {"Your decision": "Document is right"}, (NEW, "dipa"): {"Your decision": "Add"},
               (NEW, "akarta"): {"Your decision": "Add"}}
    set_word(db, "diacritic_mappings_trial", "krshna", "kṛṣṇa")                # changed since the report was made
    set_word(db, "diacritic_mappings_trial", "dipa", "dīpā")
    p = upload(rid, answers)
    assert sorted((x["plain"], x["then"], x["now"]) for x in p["skipped"]) == [
        ("dipa", None, "dīpā"), ("krshna", "{{}}{{kṛs͟hṇa}}", "kṛṣṇa")]
    assert [c["plain"] for c in p["add"]] == ["akarta"] and p["update"] == []
    set_word(db, "diacritic_mappings_trial", "akarta", "akartā")               # and once more after the page
    with pytest.raises(wordlist.Changed):
        apply(rid, p)
    assert words()["krshna"] == "kṛṣṇa" and run(lambda c, n: wordlist.get_review(c, n, rid))["status"] == "open"
    p = run(lambda c, n: wordlist.plan_for(c, n, wordlist.get_review(c, n, rid)))
    apply(rid, p)                                                              # nothing left to change
    assert words()["akarta"] == "akartā" and words()["krshna"] == "kṛṣṇa"


def test_cancel_and_upload_again(db):
    rid = make_review()
    upload(rid, {(NEW, "dipa"): {"Your decision": "Add"}})
    run(lambda c, n: wordlist.cancel(c, n, rid))
    rv = run(lambda c, n: wordlist.get_review(c, n, rid))
    assert rv["status"] == "open" and rv["decisions"] == {}
    p = upload(rid, {(NEW, "dipa"): {"Your decision": "Skip"}})                 # a new upload replaces them
    assert p["add"] == []


def test_undo_and_its_refusal(db):
    rid = make_review()
    first = apply(rid, upload(rid, {(DIFF, "krshna"): {"Your decision": "Document is right"},
                                    (NEW, "dipa"): {"Your decision": "Add"},
                                    (NEW, "akarta"): {"Your decision": "Add"}}))["batch"]
    assert [x["plain_text"] for x in run(wordlist.pending)] == ["atma", "ganesha"]       # the blank rows
    undo = run(lambda c, n: wordlist.undo(c, n, first, "b@gmail.com"))
    after = {r["plain_text"]: (r["id"], r["diacritic_text"]) for r in run(wordlist.entries)}
    assert {k: v for k, (_, v) in after.items()} == dict(LIVE) and after["krshna"][0] == 20
    assert [x["plain_text"] for x in run(wordlist.pending)] == ["atma", "ganesha"]       # put on pending: they stay
    b = run(lambda c, n: wordlist.get_batch(c, n, first))
    assert b["undone_by"] == "b@gmail.com" and run(lambda c, n: wordlist.get_batch(c, n, undo))["kind"] == "undo"
    assert run(lambda c, n: wordlist.get_review(c, n, rid))["status"] == "undone"
    with pytest.raises(wordlist.WordListError, match=r"already applied on .* \(and undone on"):
        upload(rid, {})
    with pytest.raises(wordlist.WordListError, match="already undone"):
        run(lambda c, n: wordlist.undo(c, n, first, BY))
    with pytest.raises(wordlist.WordListError, match="can't be undone"):
        run(lambda c, n: wordlist.undo(c, n, undo, BY))

    # a word changed again by a later batch blocks the undo of the earlier one
    r2 = make_review()
    second = apply(r2, upload(r2, {(DIFF, "atma"): {"Your decision": "Our list is right",
                                                    "Capital": "Always capital"}}))
    r3 = make_review("The Ātma is one.\n")
    apply(r3, upload(r3, {(DIFF, "atma"): {"Your decision": "Document is right"}}))
    assert words()["atma"] == "{{}}{{ātma}}"
    with pytest.raises(wordlist.WordListError, match="changed again after this.*atma"):
        run(lambda c, n: wordlist.undo(c, n, second["batch"], BY))


def pending_file(answers) -> dict:
    def go(conn, name):
        return sheets.pending_xlsx(wordlist.pending(conn, name), wordlist.mappings(conn, name),
                                   {"list": name, "when": str})
    return sheets.read_filled("pending.xlsx", fill(run(go), answers))


def test_the_pending_list_round_trip(db):
    def setup(conn, name):
        for plain, sug in (("ganesha", "{{}}{{gaṇesha}}"), ("dipa", "dīpa"), ("seva", "sevā"), ("krshna", "kṛṣṇa")):
            conn.execute("INSERT INTO dia_pending (list_name, plain_text, suggested, reason) VALUES "
                         "('trial', %s, %s, 'asked')", (plain, sug))
    run(setup)
    filled = pending_file({("Pending", "ganesha"): {"Your decision": "Add"},
                           ("Pending", "dipa"): {"Your decision": "Add", "Result": "Dīpā"},
                           ("Pending", "seva"): {"Your decision": "Remove from pending"},
                           ("Pending", "krshna"): {"Notes for self": "the team is checking"}})
    pid = run(lambda c, n: wordlist.store_pending_answers(c, n, filled, BY, "pending.xlsx"))
    p = run(lambda c, n: wordlist.plan_for(c, n, wordlist.get_review(c, n, pid)))
    assert sheets.sentence(p) == "This will add 2 words and take 3 off the pending list."
    done = apply(pid, p)
    assert words()["ganesha"] == "{{}}{{gaṇesha}}" and words()["dipa"] == "dīpā"
    left = run(wordlist.pending)
    assert [(x["plain_text"], x["notes"]) for x in left] == [("krshna", "the team is checking")]
    how = {x["plain_text"]: x["resolution"] for x in run(lambda c, n: wordlist.pending(c, n, resolved=True))}
    assert how["seva"] == "removed" and how["dipa"].startswith("added or changed from the pending list")
    assert run(lambda c, n: wordlist.get_batch(c, n, done["batch"]))["kind"] == "pending"
    run(lambda c, n: wordlist.undo(c, n, done["batch"], BY))       # the words go, and their questions come back
    assert "dipa" not in words() and "ganesha" not in words()
    assert sorted(x["plain_text"] for x in run(wordlist.pending)) == ["dipa", "ganesha", "krshna", "seva"]


def test_trial_and_real_files_never_mix(db, monkeypatch):
    rid = make_review()
    data = report(rid)
    monkeypatch.setenv("DIACRITICAL_WORDS_LIST", "live")
    with pytest.raises(wordlist.WordListError, match="made on the trial copy"):
        upload(rid, {}, data=data)
    filled = sheets.read_filled("p.xlsx", sheets.pending_xlsx([], {}, {"list": "trial"}))
    with pytest.raises(wordlist.WordListError, match="made on the trial copy"):
        run(lambda c, n: wordlist.store_pending_answers(c, n, filled, BY, "p.xlsx"))


def test_list_review_resolves_answered_pending_items(db):
    def setup(conn, name):
        for p in ("dipa", "puja", "ganesha"):
            conn.execute("INSERT INTO dia_pending (list_name, plain_text, suggested, reason) VALUES "
                         "('trial', %s, '', 'asked')", (p,))
        rows = [{"row": 2, "plain": "dipa", "result": "dīpa"}, {"row": 3, "plain": "puja", "result": "pūjā"}]
        items = harvest.review_list(rows, [], wordlist.mappings(conn, name))
        return wordlist.create_review(conn, name, "list", "answers.xlsx", [{"name": "answers.xlsx"}], BY, items)
    rid = run(setup)
    apply(rid, upload(rid, {}))                                                # "Add" is filled in for a list
    assert [p["plain_text"] for p in run(wordlist.pending)] == ["ganesha"]
    assert words()["dipa"] == "dīpa"


def test_start_the_trial_again(db):
    rid = make_review()
    added = apply(rid, upload(rid, {(NEW, "dipa"): {"Your decision": "Add"}}))["batch"]
    reset = run(lambda c, n: wordlist.reset_trial(c, BY))
    assert words() == dict(LIVE)
    assert run(lambda c, n: wordlist.get_batch(c, n, reset))["kind"] == "reset"
    with pytest.raises(wordlist.WordListError, match="started again"):
        run(lambda c, n: wordlist.undo(c, n, added, BY))
    r2 = make_review()
    apply(r2, upload(r2, {(NEW, "dipa"): {"Your decision": "Add"}}))          # ids still free
    assert words()["dipa"] == "dīpa"


def test_live_mode_changes_the_real_list_and_never_mixes(db, monkeypatch):
    rid = make_review()
    apply(rid, upload(rid, {(NEW, "dipa"): {"Your decision": "Add"}}))          # trial
    monkeypatch.setenv("DIACRITICAL_WORDS_LIST", "live")
    assert run(lambda c, n: n) == "live"
    assert run(wordlist.batches) == [] and "dipa" not in run(wordlist.mappings)
    assert run(lambda c, n: wordlist.uploads(c, n)) == []
    r2 = make_review("The sevā was done.\n")
    apply(r2, upload(r2, {(NEW, "seva"): {"Your decision": "Add"}}))
    assert live_rows(db)["seva"] == "sevā"
    monkeypatch.setenv("DIACRITICAL_DATABASE_URL", "")
    with pytest.raises(wordlist.NotConfigured):
        run(wordlist.entries)
    monkeypatch.setenv("DIACRITICAL_DATABASE_URL", db)
    monkeypatch.setenv("DATABASE_URL", db)
    with pytest.raises(wordlist.NotConfigured, match="refusing"):
        run(wordlist.entries)



def test_also_typed_as_and_own_rows_apply_carry_over_and_undo(db):
    from test_words_harvest import add_rows, rows_of, workbook
    with psycopg.connect(db) as c:                 # the live database had dia_pending before the 'also' column
        c.execute("""CREATE TABLE dia_pending (id serial PRIMARY KEY, list_name text NOT NULL, plain_text text NOT NULL,
                     suggested text NOT NULL DEFAULT '', reason text NOT NULL DEFAULT '',
                     source text NOT NULL DEFAULT '', page text NOT NULL DEFAULT '', context text NOT NULL DEFAULT '', added_by text,
                     added_at timestamp NOT NULL DEFAULT now(), resolved_by text, resolved_at timestamp,
                     resolution text, review_id int)""")
        c.execute("INSERT INTO dia_pending (list_name, plain_text, suggested, reason) "
                  "VALUES ('trial', 'ganesha', '{{}}{{gaṇesha}}', 'asked before')")
    run(lambda c, n: c.execute("UPDATE dia_pending SET also = 'ganesa' WHERE plain_text = 'ganesha'"))  # first use
    rid = make_review()
    data = add_rows(fill(report(rid), {
        (NEW, "dipa"): {"Your decision": "Add", "Also typed as": "deepa"},
        (DIFF, "krshna"): {"Your decision": "Our list is right", "Also typed as": "krushna, krishna"},
        (NEW, "ganesha"): {"Also typed as": "ganesh"},                      # pending (already waiting): they go along
        (NEW, "akarta"): {"Your decision": "Skip", "Also typed as": "akartaa"}}),
        NEW, [{"Word (as in the document)": "āvaraṇa-s", "Your decision": "Add"}])
    p = upload(rid, {}, data=data)
    assert sheets.sentence(p) == "This will add 4 words (2 of them other ways of typing a word) and put 1 on the " \
                                 "pending list."
    assert {x["typed"] for x in p["also_left"]} == {"krishna", "akartaa"}
    done = apply(rid, p)
    w = words()
    assert w["deepa"] == w["dipa"] == "dīpa" and w["krushna"] == w["krshna"] == "{{}}{{kṛs͟hṇa}}"
    assert w["avarana-s"] == "āvaraṇa-s" and w["krishna"] == "kṛs͟hṇa" and "akartaa" not in w
    batch = run(lambda c, n: wordlist.get_batch(c, n, done["batch"]))
    assert batch["summary"] == (f"Report {rid} (talk.docx): 4 added (2 of them other ways of typing), "
                                "1 put on the pending list")
    pending = {x["plain_text"]: x for x in run(wordlist.pending)}
    assert set(pending) == {"atma", "ganesha"} and pending["ganesha"]["also"] == "ganesa, ganesh"   # merged

    # the pending list shows them; Add brings them with the same code
    raw = run(lambda c, n: sheets.pending_xlsx(wordlist.pending(c, n), wordlist.mappings(c, n),
                                               {"list": n, "when": str}))
    assert {r["Plain word"]: r["Also typed as"] for r in rows_of(workbook(raw)["Pending"])}["ganesha"] == \
        "ganesa, ganesh"
    filled = sheets.read_filled("pending.xlsx", fill(raw, {("Pending", "ganesha"): {"Your decision": "Add"}}))
    pid = run(lambda c, n: wordlist.store_pending_answers(c, n, filled, BY, "pending.xlsx"))
    p2 = run(lambda c, n: wordlist.plan_for(c, n, wordlist.get_review(c, n, pid)))
    assert sheets.sentence(p2) == "This will add 3 words (2 of them other ways of typing a word) and take 1 off " \
                                  "the pending list."
    second = apply(pid, p2)
    w = words()
    assert w["ganesha"] == w["ganesa"] == w["ganesh"] == "{{}}{{gaṇesha}}"

    # undo takes the other ways of typing away with their word
    run(lambda c, n: wordlist.undo(c, n, second["batch"], BY))
    assert not {"ganesha", "ganesa", "ganesh"} & set(words())
    pending = {x["plain_text"]: x for x in run(wordlist.pending)}
    assert pending["ganesha"]["also"] == "ganesa, ganesh"                   # back on the list, with them
    run(lambda c, n: wordlist.undo(c, n, done["batch"], BY))
    assert words() == dict(LIVE)



def stored_review1() -> tuple[int, dict]:
    """Shantish's review 1 (trimmed fixture) stored as it was made before one row per word, with his uploaded
    decisions waiting, on a trial list whose entries are what the review saw."""
    from psycopg.types.json import Jsonb
    from test_words_harvest import review1
    fx = review1()
    run(wordlist.entries)                                                      # the trial copy exists
    for plain, code in fx["entries"].items():
        set_word(wordlist.database_url(), "diacritic_mappings_trial", plain, code)

    def go(conn, name):
        rid = wordlist.create_review(conn, name, "document", "20261127 New Diacritical words.docx",
                                     [{"name": "20261127 New Diacritical words.docx"}], BY, fx["items"])
        conn.execute("UPDATE dia_review SET decisions = %s WHERE id = %s", (Jsonb(fx["decisions"]), rid))
        return rid
    return run(go), fx


def report_of(rid) -> bytes:
    """The report as the page makes it: with what was uploaded for it filled in."""
    def go(conn, name):
        rv = wordlist.get_review(conn, name, rid)
        dec = rv["decisions"] or {}
        return sheets.report_xlsx(rv["items"], {"review": rid, "list": rv["list_name"], "files": ["x.docx"],
                                                "made_iso": rv["created_at"].isoformat(timespec="seconds")},
                                  dec.get("cells"), dec.get("extra_cells"))
    return run(go)


def test_a_review_made_before_one_row_per_word(db):
    from diacritical.diacritics import translate_text
    from test_words_harvest import old_layout_file, rows_of, workbook
    rid, fx = stored_review1()
    # its uploaded decisions are read the new way: Correct spelling + Always capital, no decision = Document is right
    p = run(lambda c, n: wordlist.plan_for(c, n, wordlist.get_review(c, n, rid)))
    assert {c["plain"]: c["result"] for c in p["update"]} == {"bankikodla": "{{}}{{baṅkīkoḍlā}}",
                                                              "santacruz": "{{}}{{sāntācrūz}}"}
    assert run(lambda c, n: wordlist.get_review(c, n, rid))["items"]["sections"]["already"][1]["form"] == "āratī"
    # downloaded again: one row per word, with what he uploaded filled in
    data = report_of(rid)
    rows = {x["Plain word"]: x for x in rows_of(workbook(data)["Different from our list"])}
    assert rows["bankikodla"]["Result"] == "Baṅkīkoḍlā" and rows["bankikodla"]["Capital"] == "Always capital"
    assert rows["trtiya"]["Word (as in the document)"] == "Tṛtīya".join(["Tṛtīyā, ", " (same as our list)"])
    # he fills in the choices he wrote in Notes, decisions left empty, and uploads it
    p = upload(rid, {(DIFF, "deva"): {"Result": "Dĕvā, Dĕva (koṅkaṇī), Deva (sanskrit)",
                                      "Capital": "Always capital"},
                     (DIFF, "trtiya"): {"Result": "Tṛtīya (masculine), Tṛtīyā (feminine)"}}, data=data)
    assert {c["plain"]: c["result"] for c in p["update"]} == {
        "bankikodla": "{{}}{{baṅkīkoḍlā}}", "santacruz": "{{}}{{sāntācrūz}}",
        "deva": "{{}}{{dĕvā, dĕva (koṅkaṇī), deva (sanskrit)}}", "trtiya": "{{tṛtīya (masculine), tṛtīyā (feminine)}}"}
    assert {x["plain"] for x in p["pending_new"]} == {"arati", "arati-s"}
    apply(rid, p)
    w = words()
    assert w["deva"] == "{{}}{{dĕvā, dĕva (koṅkaṇī), deva (sanskrit)}}" and w["thali"] == "thālī"
    assert translate_text("deva Deva trtiya Trtiya", w) == (
        "{{Dĕvā,Dĕva (koṅkaṇī),Deva (sanskrit)}} {{Dĕvā,Dĕva (koṅkaṇī),Deva (sanskrit)}} "
        "{{tṛtīya (masculine),tṛtīyā (feminine)}} {{Tṛtīya (masculine),Tṛtīyā (feminine)}}")

    # a filled file in the old layout (a row per spelling) is still read
    r2, fx = stored_review1()
    rv = run(lambda c, n: wordlist.get_review(c, n, r2))
    info = {"review": r2, "list": "trial", "made_iso": rv["created_at"].isoformat(timespec="seconds")}
    old = old_layout_file(fx, {"c4:1": {"Your decision": "Document is right"},
                               "c6:0": {"Your decision": "Our list is right"}}, info)
    p = upload(r2, {}, data=old)
    assert {c["plain"]: c["result"] for c in p["update"]} == {"deva": "dĕvā"}
