"""The Diacritical words dashboard's pages (admin_words.py) through FastAPI's TestClient, signed in with a Chainlit
access_token cookie: admins only (downloads too), CSRF on every POST, upload a Word file -> review -> choices ->
Accept -> the word list shows it -> undo on History; the report, word list and pending downloads are real .xlsx;
the edit form; Try it.

Needs a local Postgres (a throwaway database is made and dropped):
  WORDS_TEST_DATABASE_URL=postgresql://postgres@localhost:5433/postgres .venv/bin/python -m pytest tests/test_words_pages.py -q
"""
import io
import re
import sys
import urllib.parse
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_words_db import BASE, db  # noqa: E402,F401  (the throwaway database fixture)
from test_words_harvest import make_docx  # noqa: E402

pytestmark = pytest.mark.skipif(not BASE, reason="WORDS_TEST_DATABASE_URL (a local Postgres) not set")

ADMIN = "admin@gmail.com"


@pytest.fixture
def client(db, monkeypatch):
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


def test_admins_only(client):
    for path in ("/admin/words", "/admin/words/list.xlsx", "/admin/words/pending.xlsx", "/admin/words/history",
                 "/admin/words/add", "/admin/words/try", "/admin/words/review/1/report.xlsx"):
        client.cookies.set("access_token", jwt_for("alice@gmail.com"))
        r = client.get(path)
        assert r.status_code == 403 and "Admins only" in r.text, path
        client.cookies.clear()
        assert client.get(path).status_code == 401, path
    client.cookies.set("access_token", jwt_for("alice@gmail.com"))
    r = client.post("/admin/words/try", data={"csrf": csrf(), "text": "krishna"})
    assert r.status_code == 403


def test_upload_review_accept_and_undo(client):
    docx = make_docx(["Shrī Kṛshṇa came to the Mat̲h̲ at Chitrapur.",
                      "The Ātma is one; Gaṇesha and Akartā were there with the dīpa."])
    r = client.post("/admin/words/upload", data={"csrf": "wrong"},
                    files=[("files", ("talk.docx", docx, "application/octet-stream"))])
    assert r.status_code == 303 and "expired" in flash(r)

    r = client.post("/admin/words/upload", data={"csrf": csrf()},
                    files=[("files", ("talk.docx", docx, "application/octet-stream")),
                           ("files", ("scan.doc", b"x", "application/msword"))])
    assert r.status_code == 303 and r.headers["location"].startswith("/admin/words/review/")
    assert ".docx" in flash(r)                                    # the .doc wasn't read, and it says why
    review = r.headers["location"].split("?")[0]
    page = client.get(review)
    assert page.status_code == 200
    assert "TRIAL:" in page.text and "Different from our list" in page.text and "New words" in page.text
    assert "<mark>Kṛshṇa</mark>" in page.text
    ids = dict(re.findall(r'<div class="item[^"]*" id="(c\d+)">\s*<div class="head"><span class="w">([^<]+)<', page.text))
    clash = {w: i for i, w in ids.items()}
    new = dict((w, i) for i, w in re.findall(r'<tr id="(n\d+)"[^>]*>\s*<td data-label="Word"><span class="w">([^<]+)<',
                                            page.text))
    assert set(clash) == {"Kṛshṇa", "Ātma"} and {"Gaṇesha", "Akartā", "dīpa"} <= set(new)
    assert "button class=\"primary\" name=\"action\" value=\"accept\" disabled" in page.text.replace("\n", " ")

    # a wrong CSRF token changes nothing
    r = client.post(review, data={"csrf": "nope", "action": "accept", f"c:{clash['Kṛshṇa']}": "doc:0"})
    assert r.status_code == 303 and "expired" in flash(r)

    choices = {"csrf": csrf(), f"c:{clash['Kṛshṇa']}": "doc:0", f"cap:{clash['Kṛshṇa']}": "keep",
               f"c:{clash['Ātma']}": "list", f"cap:{clash['Ātma']}": "capital",
               f"c:{new['Akartā']}": "add:0", f"cap:{new['Akartā']}": "capital"}
    r = client.post(review, data=choices | {"action": "save"})
    assert r.status_code == 303 and "rows still need a choice" in flash(r)
    r = client.post(review, data=choices | {"action": "new_pending"})        # the rest of the new words to Pending
    assert r.headers["location"].endswith("#new")
    page = client.get(review).text
    assert "Accept will add 1 word, change 2, and put" in page
    assert f'name="c:{new["Gaṇesha"]}" value="pending" checked' in page
    r = client.post(review, data={k: v for k, v in choices.items()} | {
        "action": "accept", **{f"c:{i}": "pending" for w, i in new.items() if w != "Akartā"}})
    assert r.status_code == 303 and flash(r).startswith("Accepted: added 1 word, changed 2"), flash(r)
    page = client.get(review).text
    assert "Accepted" in page and "disabled" in page

    words = client.get("/admin/words?q=krshna").text
    assert "Kṛshṇa (always capital)" in words
    assert "Ātmā (always capital)" in client.get("/admin/words?q=ātmā").text     # found with marks too
    pending = client.get("/admin/words/pending").text
    assert "gaṇesha" in pending.lower() and "Pending (" in pending

    report = client.get(review + "/report.xlsx")
    assert report.status_code == 200 and report.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml")
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(report.content))
    assert wb.sheetnames[:3] == ["Summary", "1 Different from our list", "2 New words"]
    decided = {row[0]: row[8] for row in wb["1 Different from our list"].iter_rows(min_row=3, values_only=True)}
    assert decided == {"Kṛshṇa": "The document is right", "Ātma": "Our list is right"}

    history = client.get("/admin/words/history").text
    batch = re.search(r'action="/admin/words/history/(\d+)/undo"', history).group(1)
    assert "1 added, 2 changed" in history
    r = client.post(f"/admin/words/history/{batch}/undo", data={"csrf": "wrong"})
    assert "expired" in flash(r)
    r = client.post(f"/admin/words/history/{batch}/undo", data={"csrf": csrf()})
    assert r.status_code == 303 and flash(r).startswith("Undone")
    assert "Kṛs͟hṇa (always capital)" in client.get("/admin/words?q=krshna").text
    assert "Undone" in client.get("/admin/words/history").text
    detail = client.get(f"/admin/words/history/{batch}").text
    assert "Kṛs͟hṇa (always capital)" in detail and "Kṛshṇa (always capital)" in detail


def test_downloads_edit_form_and_try_it(client):
    from openpyxl import load_workbook
    r = client.get("/admin/words/list.xlsx")
    wb = load_workbook(io.BytesIO(r.content))
    assert [c.value for c in wb.active[1]] == ["plain_text", "diacritic_text"] and wb.active.max_row == 8
    r = client.get("/admin/words/pending.xlsx")
    assert [c.value for c in load_workbook(io.BytesIO(r.content)).active[1]][-1] == "Team's answer"

    form = client.get("/admin/words/edit?id=40").text                       # god -> {{gŏḍ}}{{god}}: a Name entry
    assert 'name="type" value="name" checked' in form and 'value="gŏḍ"' in form and 'value="god"' in form
    assert "<strong>gŏḍ</strong>" in form and "<strong>God</strong>" in form             # the preview, typed small and Capital
    base = {"csrf": csrf(), "id": "40", "expected_plain": "god", "expected_result": "{{gŏḍ}}{{god}}",
            "plain": "god", "type": "name", "result": "gŏḍ", "name": "God"}
    r = client.post("/admin/words/save", data=base | {"plain": "krishna", "action": "save"})
    assert r.status_code == 409 and "already in the list" in r.text and "/admin/words/edit?id=10" in r.text
    r = client.post("/admin/words/save", data=base | {"type": "capital", "result": "Gŏḍ", "action": "preview"})
    assert r.status_code == 200 and "{{}}{{gŏḍ}}" in r.text
    r = client.post("/admin/words/save", data=base | {"type": "capital", "result": "Gŏḍ", "action": "save"})
    assert r.status_code == 303 and "Gŏḍ (always capital)" in flash(r)
    r = client.post("/admin/words/save", data=base | {"action": "save"})    # the form is out of date now
    assert r.status_code == 303 and "changed this word while you were editing" in flash(r)
    form = client.get("/admin/words/new").text
    r = client.post("/admin/words/save", data={"csrf": csrf(), "plain": "Kārtik", "type": "choices",
                                               "choices": "kārtik\nkārtika", "action": "save"})
    assert r.status_code == 303 and "kartik → choice: kārtik or kārtika" in flash(r)
    r = client.post("/admin/words/save", data={"csrf": csrf(), "plain": "x", "type": "choices", "choices": "one",
                                               "action": "save"})
    assert r.status_code == 400 and "at least two" in r.text

    r = client.post("/admin/words/try", data={"csrf": csrf(), "text": "Shri Krishna and god"})
    assert "Shrī Kṛs͟hṇa and Gŏḍ" in r.text            # god is always capital now (saved above)
    r = client.post("/admin/words/try", data={"csrf": "x", "text": "Shri"})
    assert r.status_code == 303 and "expired" in flash(r)
    r = client.post("/admin/words/reset-trial", data={"csrf": csrf()})
    assert r.status_code == 303 and "same as the real list" in flash(r)
    assert "{{gŏḍ}}{{god}}" in client.get("/admin/words?q=god").text      # stored code, in the hover title
