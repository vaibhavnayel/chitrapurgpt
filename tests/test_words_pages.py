"""The Diacritical words dashboard's pages (admin_words.py) through FastAPI's TestClient, signed in with a Chainlit
access_token cookie: admins only (downloads too), CSRF on every POST; upload a Word file -> 'Your report is ready' ->
download the report -> fill it with openpyxl -> upload it -> the confirm page's sentence -> Apply -> History -> undo;
the pending list download/upload round trip; refusing a non-dashboard .xlsx, an already-applied report and a trial
report on the real list; Cancel.

Needs a local Postgres (a throwaway database is made and dropped):
  WORDS_TEST_DATABASE_URL=postgresql://postgres@localhost:5433/postgres .venv/bin/python -m pytest tests/test_words_pages.py -q
"""
import re
import sys
import urllib.parse
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_words_db import BASE, db  # noqa: E402,F401  (the throwaway database fixture)
from test_words_harvest import fill, make_docx, make_xlsx, rows_of, workbook  # noqa: E402

pytestmark = pytest.mark.skipif(not BASE, reason="WORDS_TEST_DATABASE_URL (a local Postgres) not set")

ADMIN = "admin@gmail.com"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOC = ["Shrī Kṛshṇa came to the Mat̲h̲ at Chitrapur.",
       "The Ātma is one; Gaṇesha and Akartā were there with the dīpa."]


@pytest.fixture
def client(db, monkeypatch):  # noqa: F811  (db is the imported fixture)
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "x" * 64)
    monkeypatch.setenv("ADMIN_EMAILS", ADMIN)
    monkeypatch.delenv("IS_PULL_REQUEST", raising=False)
    from fastapi import FastAPI
    from starlette.testclient import TestClient
    import admin_words
    app = FastAPI()
    app.include_router(admin_words.router)
    c = TestClient(app, base_url="http://testserver", follow_redirects=False)
    c.cookies.set("access_token", jwt_for(ADMIN))
    return c


def jwt_for(email):
    from chainlit.auth.jwt import create_jwt
    from chainlit.user import User
    return create_jwt(User(identifier=email, metadata={"provider": "google"}))


def csrf():
    from admin import _csrf_token
    return _csrf_token(ADMIN)


def flash(response) -> str:
    query = urllib.parse.urlsplit(response.headers["location"]).query
    return urllib.parse.parse_qs(query).get("msg", [""])[0]


def upload_doc(client, paragraphs=DOC, name="talk.docx"):
    r = client.post("/admin/words/upload", data={"csrf": csrf()},
                    files=[("files", (name, make_docx(paragraphs), "application/octet-stream"))])
    assert r.status_code == 303 and flash(r).startswith("Your report is ready"), flash(r)
    return int(urllib.parse.parse_qs(urllib.parse.urlsplit(r.headers["location"]).query)["new"][0])


def send_decisions(client, data: bytes, name="filled.xlsx", token=None):
    return client.post("/admin/words/decisions", data={"csrf": token or csrf()}, files=[("file", (name, data, XLSX))])


def fingerprint(page: str) -> str:
    return re.search(r'name="fingerprint" value="([^"]+)"', page).group(1)


def test_admins_only(client):
    for path in ("/admin/words", "/admin/words/list.xlsx", "/admin/words/pending.xlsx", "/admin/words/history",
                 "/admin/words/report/1.xlsx", "/admin/words/confirm/1"):
        client.cookies.set("access_token", jwt_for("alice@gmail.com"))
        r = client.get(path)
        assert r.status_code == 403 and "Admins only" in r.text, path
        client.cookies.clear()
        assert client.get(path).status_code == 401, path
    client.cookies.set("access_token", jwt_for("alice@gmail.com"))
    assert send_decisions(client, b"x").status_code == 403
    assert client.post("/admin/words/upload", data={"csrf": csrf()},
                       files=[("files", ("a.txt", b"x", "text/plain"))]).status_code == 403
    assert client.post("/admin/words/confirm/1", data={"csrf": csrf(), "action": "apply"}).status_code == 403


def test_upload_report_decisions_apply_and_undo(client):
    docx = make_docx(DOC)
    r = client.post("/admin/words/upload", data={"csrf": "wrong"},
                    files=[("files", ("talk.docx", docx, "application/octet-stream"))])
    assert r.status_code == 303 and "expired" in flash(r)

    r = client.post("/admin/words/upload", data={"csrf": csrf()},
                    files=[("files", ("talk.docx", docx, "application/octet-stream")),
                           ("files", ("scan.doc", b"x", "application/msword"))])
    assert r.status_code == 303 and flash(r).startswith("Your report is ready") and ".docx" in flash(r)
    rid = int(urllib.parse.parse_qs(urllib.parse.urlsplit(r.headers["location"]).query)["new"][0])
    home = client.get(r.headers["location"]).text
    assert "TRIAL:" in home and "Your report is ready" in home and "Waiting for your decisions" in home
    assert f'href="/admin/words/report/{rid}.xlsx"' in home and '<span class="tag new">New</span>' in home

    report = client.get(f"/admin/words/report/{rid}.xlsx")
    assert report.status_code == 200 and report.headers["content-type"].startswith(XLSX)
    wb = workbook(report.content)
    assert wb.sheetnames[:4] == ["Start here", "Different from our list", "New words", "Written without marks"]
    assert {row["Plain word"] for row in rows_of(wb["Different from our list"])} == {"krshna", "atma"}
    filled = fill(report.content, {
        ("Different from our list", "krshna"): {"Your decision": "Document is right"},
        ("Different from our list", "atma"): {"Your decision": "Our list is right", "Capital": "Always capital"},
        ("New words", "akarta"): {"Your decision": "Add"},
        ("New words", "dipa"): {"Your decision": "perhaps"}})                  # not a choice: pending, listed

    r = send_decisions(client, filled, token="nope")
    assert r.status_code == 303 and "expired" in flash(r)
    r = send_decisions(client, filled)
    assert r.status_code == 303 and r.headers["location"] == f"/admin/words/confirm/{rid}"
    page = client.get(r.headers["location"]).text
    assert "This will add 1 word, change 2, and put 3 on the pending list." in page
    assert "Couldn’t read (1)" in page and "perhaps" in page and "New words, row" in page
    assert "Kṛs͟hṇa (always capital)" in page and "Kṛshṇa (always capital)" in page   # before -> after
    assert "Upload your decisions" not in page and 'value="apply">Apply</button>' in page

    r = client.post(f"/admin/words/confirm/{rid}", data={"csrf": "x", "action": "apply",
                                                        "fingerprint": fingerprint(page)})
    assert "expired" in flash(r)
    r = client.post(f"/admin/words/confirm/{rid}", data={"csrf": csrf(), "action": "apply", "fingerprint": "old"})
    assert r.headers["location"].startswith(f"/admin/words/confirm/{rid}") and "changed after this page" in flash(r)
    r = client.post(f"/admin/words/confirm/{rid}", data={"csrf": csrf(), "action": "apply",
                                                        "fingerprint": fingerprint(page)})
    assert r.status_code == 303 and flash(r).startswith("Applied: added 1 word, changed 2, and put 3 on the pending"), \
        flash(r)
    home = client.get("/admin/words").text
    assert "Applied on" in home and "3</strong> words\non the pending list" in home

    r = send_decisions(client, filled)                                          # the same report again
    assert r.status_code == 303 and "These decisions were already applied on" in flash(r)
    again = workbook(client.get(f"/admin/words/report/{rid}.xlsx").content)   # the record keeps the decisions
    assert "Applied on" in str(again["Start here"]["A3"].value)
    assert {row["Plain word"]: row["Your decision"] for row in rows_of(again["New words"])}["akarta"] == "Add"

    history = client.get("/admin/words/history").text
    batch = re.search(r'action="/admin/words/history/(\d+)/undo"', history).group(1)
    assert "1 added, 2 changed" in history and "Kṛshṇa (always capital)" in history
    r = client.post(f"/admin/words/history/{batch}/undo", data={"csrf": "wrong"})
    assert "expired" in flash(r)
    r = client.post(f"/admin/words/history/{batch}/undo", data={"csrf": csrf()})
    assert r.status_code == 303 and flash(r).startswith("Undone")
    assert "Undone" in client.get("/admin/words/history").text
    assert "then undone on" in client.get("/admin/words").text
    whole = workbook(client.get("/admin/words/list.xlsx").content)
    assert {row[0]: row[1] for row in whole.active.iter_rows(min_row=2, values_only=True)}["krshna"] == \
        "{{}}{{kṛs͟hṇa}}"


def test_pending_list_download_and_upload(client):
    rid = upload_doc(client)
    report = client.get(f"/admin/words/report/{rid}.xlsx").content
    r = send_decisions(client, fill(report, {}))                               # all blank: everything to pending
    page = client.get(r.headers["location"]).text
    assert "This will put 6 on the pending list." in page                       # 2 different + 4 new (math: not
    #                                                                           in this small list)
    client.post(f"/admin/words/confirm/{rid}", data={"csrf": csrf(), "action": "apply",
                                                     "fingerprint": fingerprint(page)})
    assert "Pending list (6 words)" in client.get("/admin/words").text

    pending = client.get("/admin/words/pending.xlsx")
    assert pending.status_code == 200
    rows = {row["Plain word"]: row for row in rows_of(workbook(pending.content)["Pending"])}
    assert set(rows) == {"krshna", "atma", "akarta", "dipa", "ganesha", "math"}
    assert rows["krshna"]["Result"] == "kṛshṇa" and rows["krshna"]["Capital"] == "Always capital"
    data = fill(pending.content, {("Pending", "dipa"): {"Your decision": "Add", "Result": "dīpā"},
                                  ("Pending", "krshna"): {"Your decision": "Add"},
                                  ("Pending", "akarta"): {"Your decision": "Remove from pending"},
                                  ("Pending", "ganesha"): {"Notes for self": "ask the team"}})
    r = send_decisions(client, data, "pending.xlsx")
    page = client.get(r.headers["location"]).text
    assert "This will add 1 word, change 1, and take 3 off the pending list." in page
    r = client.post(r.headers["location"], data={"csrf": csrf(), "action": "apply", "fingerprint": fingerprint(page)})
    assert flash(r).startswith("Applied: added 1 word, changed 1, and took 3 off the pending list."), flash(r)
    assert "Pending list (3 words)" in client.get("/admin/words").text
    again = rows_of(workbook(client.get("/admin/words/pending.xlsx").content)["Pending"])
    assert {row["Plain word"]: row["Notes for self"] for row in again}["ganesha"] == "ask the team"
    history = client.get("/admin/words/history").text
    assert "Pending list answers (pending.xlsx): 1 added, 1 changed, 1 removed from the pending list" in history

    r = send_decisions(client, data, "pending.xlsx")                            # the same answers again
    page = client.get(r.headers["location"]).text if "confirm" in r.headers["location"] else flash(r)
    assert "already taken off the pending list" in page


def test_refusals_and_cancel(client, monkeypatch):
    r = send_decisions(client, make_xlsx([["plain_text", "diacritic_text"], ["puja", "pūjā"]]), "mine.xlsx")
    assert r.status_code == 303 and "isn't a report or pending list from this page" in flash(r)
    r = send_decisions(client, b"%PDF-1.4", "issue.pdf")
    assert "Please upload the filled report" in flash(r)

    rid = upload_doc(client)
    report = client.get(f"/admin/words/report/{rid}.xlsx").content
    filled = fill(report, {("New words", "dipa"): {"Your decision": "Add"}})
    r = client.post("/admin/words/upload", data={"csrf": csrf()},             # put in the wrong box: still read
                    files=[("files", ("filled.xlsx", filled, XLSX))])
    assert r.headers["location"] == f"/admin/words/confirm/{rid}"
    assert "Your decisions are uploaded, not applied yet" in client.get("/admin/words").text
    r = client.post(f"/admin/words/confirm/{rid}", data={"csrf": csrf(), "action": "cancel"})
    assert flash(r) == "Cancelled: nothing was changed."
    assert "Waiting for your decisions" in client.get("/admin/words").text
    assert "no decisions waiting" in flash(client.get(f"/admin/words/confirm/{rid}"))

    monkeypatch.setenv("DIACRITICAL_WORDS_LIST", "live")
    r = send_decisions(client, filled)
    assert "made on the trial copy" in flash(r)
    assert "REAL word list" in client.get("/admin/words").text
    r = client.post("/admin/words/reset-trial", data={"csrf": csrf()})
    assert "only works on the trial copy" in flash(r)
    monkeypatch.setenv("DIACRITICAL_WORDS_LIST", "trial")
    r = client.post("/admin/words/reset-trial", data={"csrf": csrf()})
    assert r.status_code == 303 and "same as the real list" in flash(r)



def test_confirm_page_shows_other_ways_of_typing_and_own_rows(client):
    from test_words_harvest import add_rows
    rid = upload_doc(client)
    report = client.get(f"/admin/words/report/{rid}.xlsx").content
    data = add_rows(fill(report, {("New words", "dipa"): {"Your decision": "Add", "Also typed as": "deepa, krishna"},
                                  ("New words", "akarta"): {"Your decision": "Skip", "Also typed as": "akartaa"}}),
                    "New words", [{"Word (as in the document)": "āvaraṇa-s", "Your decision": "Add"},
                                  {"Word (as in the document)": "Pūjā", "Your decision": "Add"}])
    r = send_decisions(client, data)
    page = client.get(r.headers["location"]).text
    assert "This will add 3 words (1 of them another way of typing a word) and put 4 on the pending list." in page
    assert "(also typed as, for dipa)" in page and "your own row: New words, row" in page
    assert "Other ways of typing, not added (2)" in page
    assert "krishna is already in the list as kṛs͟hṇa, so it was left as it is; change it on the Mappings page if " \
           "needed." in page
    assert "not used, because the word was skipped." in page
    assert "Words you typed in, not added (1)" in page and "puja is already in the list exactly like this" in page
    r = client.post(r.headers["location"], data={"csrf": csrf(), "action": "apply", "fingerprint": fingerprint(page)})
    assert flash(r).startswith("Applied: added 3 words (1 of them another way of typing a word)"), flash(r)
    whole = workbook(client.get("/admin/words/list.xlsx").content)
    listed = {row[0]: row[1] for row in whole.active.iter_rows(min_row=2, values_only=True)}
    assert listed["deepa"] == listed["dipa"] == "dīpa" and listed["avarana-s"] == "āvaraṇa-s"
    again = workbook(client.get(f"/admin/words/report/{rid}.xlsx").content)       # the record keeps them
    rows = rows_of(again["New words"])
    assert {r["Plain word"]: r["Also typed as"] for r in rows}["dipa"] == "deepa, krishna"
    assert rows[-2]["Word (as in the document)"] == "āvaraṇa-s"
    history = client.get("/admin/words/history").text
    assert "3 added (1 of them another way of typing)" in history



def test_one_row_per_word_and_several_spellings_through_the_pages(client, db):  # noqa: F811
    from test_words_db import set_word
    for plain, code in (("trtiya", "tṛtīya"), ("deva", "deva"), ("devi", "dĕvī")):
        set_word(db, "diacritic_mappings", plain, code)                     # before the trial copy is made
    rid = upload_doc(client, ["Tṛtīyā (feminine) and Tṛtīya (masculine).", "Dĕvā, Dĕva (Koṅkaṇī); Deva (Sanskrit)",
                              "Shrī Kṛshṇa came."])
    report = client.get(f"/admin/words/report/{rid}.xlsx").content
    wb = workbook(report)
    rows = {r["Plain word"]: r for r in rows_of(wb["Different from our list"])}
    assert rows["trtiya"]["Word (as in the document)"] == "Tṛtīyā, Tṛtīya (same as our list)"
    assert rows["deva"]["Word (as in the document)"] == "Dĕva, Dĕvā" and rows["deva"]["Times"] == 2
    assert "Tṛtīya" not in {r["Word (as in the document)"] for r in rows_of(wb["Already in our list"])}
    diff = "Different from our list"
    data = fill(report, {(diff, "deva"): {"Correct spelling": "Dĕvā, Dĕva (koṅkaṇī), Deva (sanskrit)",
                                          "Capital": "Always capital"},
                         (diff, "trtiya"): {"Correct spelling": "Tṛtīya (masculine), Tṛtīyā (feminine)"}})
    r = send_decisions(client, data)
    page = client.get(r.headers["location"]).text
    assert "This will change 2 and put 2 on the pending list." in page        # krshna and konkani: blank
    assert "choice: Dĕvā or Dĕva (koṅkaṇī) or Deva (sanskrit) (always capital)" in page
    assert "choice: tṛtīya (masculine) or tṛtīyā (feminine)" in page
    assert "Correct spelling: Tṛtīya (masculine), Tṛtīyā (feminine) (no decision)" in page   # how it was read
    r = client.post(r.headers["location"], data={"csrf": csrf(), "action": "apply", "fingerprint": fingerprint(page)})
    assert flash(r).startswith("Applied: changed 2"), flash(r)
    listed = {row[0]: row[1] for row in workbook(client.get("/admin/words/list.xlsx").content).active.iter_rows(
        min_row=2, values_only=True)}
    assert listed["deva"] == "{{}}{{dĕvā, dĕva (koṅkaṇī), deva (sanskrit)}}"
    assert listed["trtiya"] == "{{tṛtīya (masculine), tṛtīyā (feminine)}}"


def test_a_review_made_before_one_row_per_word_through_the_pages(client):
    from test_words_db import stored_review1
    rid, _ = stored_review1()
    home = client.get("/admin/words").text
    assert "Your decisions are uploaded, not applied yet" in home
    assert "7 different from our list · 5 new · 2 without marks · 3 already in our list" in home   # regrouped
    wb = workbook(client.get(f"/admin/words/report/{rid}.xlsx").content)
    rows = {r["Plain word"]: r for r in rows_of(wb["Different from our list"])}
    assert (rows["bankikodla"]["Word (as in the document)"], rows["bankikodla"]["Correct spelling"],
            rows["bankikodla"]["Capital"]) == ("Baṅkīkoḍla, Baṅkīkoḍlā", "Baṅkīkoḍlā", "Always capital")
    assert rows["arati"]["Word (as in the document)"] == "ārati, āratī (same as our list)"
    page = client.get(f"/admin/words/confirm/{rid}").text
    assert "Baṅkīkoḍlā (always capital)" in page and "Baṅkikoḍlā (always capital)" in page    # before -> after
    assert "Correct spelling: Baṅkīkoḍlā (no decision)" in page
