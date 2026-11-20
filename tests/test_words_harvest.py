"""The Diacritical words dashboard's pure logic (diacritical/harvest.py): reading uploads (.docx, .txt, PDF, .xlsx
lists), sorting a document's words against a small fake list, the review of an offline list, what Accept does, the
edit page's boxes <-> the stored code, and the report. No database.
  .venv/bin/python -m pytest tests/test_words_harvest.py -q
"""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diacritical import harvest  # noqa: E402

LIST = {
    "krishna": "kṛs͟hṇa",
    "krshna": "{{}}{{kṛs͟hṇa}}",
    "math": "mat͟h",
    "shri": "shrī",
    "chitrapur": "{{}}{{chitrāpur}}",
    "atma": "ātmā",
    "god": "{{gŏḍ}}{{god}}",
    "anandashram": "{{}}{{ānandāshram}}",
    "laddu": "laḍḍū",
    "puja": "pūjā",
    "swamiji": "{{}}{{swāmījī}}",
    "mangaluru": "{{}}{{maṅgal̃ūru}}",
    "kshama yachana": "kshamā-yāchanā",
    "namavali": "{{nāmāvalī (hindi), nāmāvaliḥ (sanskrit)}}",
    "kanara": "{{}}{{kănarā}}",
}

TEXT = """Shrī Kṛshṇa came to the Mat̲h̲ at Chitrāpur.
The Ātma is one, said Ᾱnandāshram Swāmījī.
We offered laḍḍū-s and pūjā at the Akartā-Abhoktā TempleShrī Gaṇesha.
Swāmījī’s blessings came to Chitrapur and to god, and God.
HebĨekar and San\u0342jīvanī were there with Bhaṭmām.
Then the Kshamā Yāchanā was said, and the dīpa was lit.
Dīpa again, then the Nāmāvalī (hindi) and the kănarā.
"""


def make_docx(paragraphs, table=None) -> bytes:
    import docx
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    if table:
        t = d.add_table(rows=len(table), cols=len(table[0]))
        for i, row in enumerate(table):
            for j, cell in enumerate(row):
                t.cell(i, j).text = cell
    d.add_paragraph("After the table: Mat͟h")
    out = io.BytesIO()
    d.save(out)
    return out.getvalue()


def review(text=TEXT, mappings=LIST, name="talk.docx"):
    return harvest.review_document([(name, [(None, text)])], mappings)


def by_form(items):
    return {i["form"]: i for i in items}


# ---------- reading files ----------

def test_docx_paragraphs_and_table_cells_in_order():
    data = make_docx(["First Kṛshṇa paragraph.", "Second one."], [["Cell Mat̲h̲", "Pūjā cell"], ["x", "Swāmījī"]])
    read = harvest.read_upload("talk.docx", data)
    assert read["kind"] == "document"
    text = read["pages"][0][1]
    order = ["First Kṛshṇa", "Second one", "Cell Mat̲h̲", "Pūjā cell", "Swāmījī", "After the table"]
    positions = [text.index(t) for t in order]
    assert positions == sorted(positions)
    assert read["pages"][0][0] is None


def test_txt_encodings_and_list_detection():
    text = "Shrī Kṛshṇa came to the Mat͟h.\nA second line, with a comma in it and many more words in it.\n"
    for data in (text.encode("utf-8"), b"\xef\xbb\xbf" + text.encode("utf-8"), text.encode("utf-16")):
        read = harvest.read_upload("a.txt", data)
        assert read["kind"] == "document" and "Kṛshṇa" in read["pages"][0][1]
    assert harvest.decode_text("café".encode("cp1252")) == "café"
    listed = harvest.read_upload("mappings.txt", "plain_text,diacritic_text\nkrishna,kṛs͟hṇa\npuja,pūjā\n".encode())
    assert listed["kind"] == "list"
    assert [(r["plain"], r["result"]) for r in listed["rows"]] == [("krishna", "kṛs͟hṇa"), ("puja", "pūjā")]


def test_unreadable_files_get_plain_messages():
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    out = io.BytesIO()
    w.write(out)
    with pytest.raises(harvest.ReadProblem, match="no text inside"):
        harvest.read_upload("scan.pdf", out.getvalue())
    with pytest.raises(harvest.ReadProblem, match="couldn't be opened as a PDF"):
        harvest.read_upload("broken.pdf", b"not a pdf at all")
    with pytest.raises(harvest.ReadProblem, match="Word document"):
        harvest.read_upload("broken.docx", b"PK nothing")
    with pytest.raises(harvest.ReadProblem, match=".docx"):
        harvest.read_upload("old.doc", b"x")
    with pytest.raises(harvest.ReadProblem, match="over 40 MB"):
        harvest.read_upload("huge.txt", b"x" * (harvest.MAX_BYTES + 1))


def minimal_pdf(text: str) -> bytes:
    """A one-page PDF with a line of ASCII text (Helvetica)."""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_pdf_pages_keep_their_numbers():
    read = harvest.read_upload("issue.pdf", minimal_pdf("Shri Krishna came"))
    assert read["kind"] == "document"
    assert read["pages"][0][0] == 1 and "Shri Krishna came" in read["pages"][0][1]


def test_pdf_space_before_a_mark_is_removed():
    assert harvest.fix_pdf_text("Maṅgal \u0303ūru") == "Maṅgal̃ūru"
    assert harvest.fix_pdf_text("s \u0332h\u0332") == "s\u0332h\u0332"
    assert harvest.fix_pdf_text("plain text") == "plain text"


def test_pdf_words_split_by_spaces_are_joined():
    known = {"swamiji", "vittal", "maya"}
    join = harvest.join_split_words
    assert join("by H.H. Sw āmījī today", known) == "by H.H. Swāmījī today"      # the joined word is in the list
    assert join("at Viṭṭ al, then", known) == "at Viṭṭal, then"                  # the ṭṭ ligature
    assert join("in dedicated sādhan ā. One", known) == "in dedicated sādhanā. One"  # a lone marked letter
    assert join("Nirguṇa Nir ākāra (without form)", known) == "Nirguṇa Nirākāra (without form)"
    assert join("the nature of Saṁ vit", known) == "the nature of Saṁvit"
    for line in ("pūjā and sevā", "Shrī Kṛshṇa", "the Ātmā is", "and ākāra too", "Umā Maheshwara"):
        assert join(line, known) == line


# ---------- sorting a document's words ----------

def test_document_review_sorts_the_words():
    r = review()
    s = r["sections"]
    clash, new, style, already = by_form(s["clash"]), by_form(s["new"]), by_form(s["style"]), by_form(s["already"])

    assert "Kṛshṇa" in clash and clash["Kṛshṇa"]["gives"] == "Kṛs͟hṇa"         # in the list, spelt differently
    assert clash["Kṛshṇa"]["entry"] == "{{}}{{kṛs͟hṇa}}" and clash["Kṛshṇa"]["options"] == ["kṛshṇa"]
    assert clash["Ātma"]["gives"] == "Ātmā"
    assert style["Mat̲h̲"]["why"] == "underline instead of the tie"              # style only
    assert style["Ᾱnandāshram"]["why"] == "Greek look-alike letter"
    assert {"Shrī", "Chitrāpur", "pūjā", "laḍḍū-s"} <= set(already)            # the translator gives exactly these
    assert {i["plain"] for i in s["new"]} == {"akarta", "abhokta", "ganesha", "dipa"}
    assert by_form(s["new"])["Dīpa"]["spellings"] == ["Dīpa", "dīpa"]
    assert "Swāmījī" in already and already["Swāmījī"]["times"] == 2         # "Swāmījī’s" counts, without "’s"
    assert "TempleShrī" not in new and "Akartā-Abhoktā" not in new           # camel split, hyphen split
    assert not any(i["form"] in ("s", "Temple") for sec in s.values() for i in sec)

    unmarked = {i["plain"]: i for i in s["unmarked"]}
    assert unmarked["chitrapur"]["gives"] == "Chitrāpur"                      # written plainly; the list adds marks
    assert unmarked["god"]["forms"] == ["god"] and unmarked["god"]["gives"] == "gŏḍ"   # "God" stays English
    odd = {i["form"]: i["why"] for i in s["doubtful"]}
    assert "HebĨekar" in odd and odd["HebĨekar"].startswith("Ĩ")
    assert odd["Bhaṭmām"] == "letters our list never uses (perhaps garbled in the PDF)"   # ṭ: not in this small list
    assert any("Greek accent" in why for why in odd.values())

    phrase = [c for c in s["clash"] if c.get("phrase")]
    assert [p["form"] for p in phrase] == ["Kshamā Yāchanā"] and phrase[0]["gives"] == "Kshamā-Yāchanā"
    assert r["counts"] == {k: len(v) for k, v in s.items()}
    ids = [i["id"] for sec in s.values() for i in sec]
    assert len(ids) == len(set(ids))


def test_how_a_new_word_was_seen():
    text = ("The Gaṇesha idol came, and Gaṇesha left.\nDīpa was lit. The dīpa shone.\n"
            "Abhoktā is one. Abhoktā is two.\n")
    new = by_form(review(text)["sections"]["new"])
    assert new["Gaṇesha"]["seen"] == "name"         # a capital, also mid-sentence
    assert new["Dīpa"]["seen"] == "small"           # also seen in small letters (the most common spelling leads)
    assert new["Abhoktā"]["seen"] == "start"        # a capital only at the start of a sentence
    assert harvest.default_decisions(review(text))[new["Gaṇesha"]["id"]] == {"cap": "capital"}


def test_spellings_of_one_word_are_one_row():
    text = "Kṛshṇa and Kṛs͟hṇa and Krshṇa met. Then Vināyaka and Vinayakā came.\n"
    r = review(text)
    assert len(r["sections"]["clash"]) == 1
    clash = r["sections"]["clash"][0]
    forms = {f["form"]: f["status"] for f in clash["forms"]}
    assert forms == {"Kṛshṇa": "differs", "Krshṇa": "differs", "Kṛs͟hṇa": "same as our list"}
    assert sorted(clash["options"]) == ["krshṇa", "kṛshṇa"]
    assert [label for _, label in harvest.choice_options(clash)][1:3] == [
        f"The document is right: {clash['options'][0]}", f"The document is right: {clash['options'][1]}"]
    new = r["sections"]["new"]
    assert len(new) == 1 and sorted(new[0]["options"]) == ["vinayakā", "vināyaka"]
    assert [v for v, _ in harvest.choice_options(new[0])][:2] == ["add:0", "add:1"]


def test_suggestions_for_plurals_and_hyphenated_words():
    assert harvest.suggestions("modaka-s", "modaka-s", LIST) == [{"plain": "modakas", "result": "modaka-s"}]
    assert harvest.suggestions("bhokta-s", "bhoktā-s", LIST) == [
        {"plain": "bhoktas", "result": "bhoktā-s"}, {"plain": "bhokta", "result": "bhoktā"}]
    assert harvest.suggestions("laddu-s", "laḍḍū-s", LIST) == [{"plain": "laddus", "result": "laḍḍū-s"}]  # laddu exists
    assert harvest.suggestions("sandhya-vandana", "sandhyā-vandana", LIST) == [
        {"plain": "sandhya vandana", "result": "sandhyā-vandana"}]


# ---------- offline lists ----------

def make_xlsx(rows) -> bytes:
    from openpyxl import Workbook
    wb = Workbook()
    for r in rows:
        wb.active.append(r)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def test_xlsx_list_columns_found_by_header():
    pending_sheet = make_xlsx([
        ["Plain word", "Suggested", "Shown as", "Why", "Source", "Page", "Example line", "Added", "Team's answer"],
        ["abhokta", "abhoktā", "abhoktā", "New word", "x.pdf", "39", "…", "", "Abhoktā"],
        ["gaurish", "gaurīsh", "", "", "", "", "", "", None],                       # not answered yet
    ])
    rows, problems = harvest.read_list_xlsx(pending_sheet)
    assert rows == [{"row": 2, "plain": "abhokta", "result": "abhoktā"}]
    assert problems[0]["why"] == "no result (left empty)"

    with_codes = make_xlsx([["notes"], ["plain_text", "diacritic_text"], ["Krishna ", "Kṛs̲h̲ṇa"],
                            ["god", "{{gŏḍ}}{{God}}"]])
    rows, _ = harvest.read_list_xlsx(with_codes)
    assert [(r["plain"], r["result"]) for r in rows] == [("krishna", "kṛs͟hṇa"), ("god", "{{gŏḍ}}{{god}}")]

    no_header = make_xlsx([["puja", "pūjā"], ["mangaιuru", "x"], ["broken", "{{a}}{{b}}{{c}}"], ["", ""],
                           ["puja", "pūja"], ["puja", "pūjā"], ["seva", ""]])
    rows, problems = harvest.read_list_xlsx(no_header)
    assert [r["plain"] for r in rows] == ["puja"]
    whys = [p["why"] for p in problems]
    assert whys[0] == "the plain word has letters our list can't use"
    assert whys[1].startswith("the result: its {{ }} brackets")
    assert whys[2].startswith("the plain word is given again in row 1")
    assert whys[3] == "no result (left empty)" and len(problems) == 4


def test_list_review():
    rows = [{"row": 2, "plain": "krishna", "result": "kṛs͟hṇa"}, {"row": 3, "plain": "puja", "result": "pūja"},
            {"row": 4, "plain": "abhokta", "result": "abhoktā"}]
    r = harvest.review_list(rows, [{"row": 5, "plain": "x", "result": "", "why": "no result"}], LIST)
    s = r["sections"]
    assert [i["plain"] for i in s["lnew"]] == ["abhokta"]
    assert [(i["plain"], i["entry"], i["result"]) for i in s["lchanged"]] == [("puja", "pūjā", "pūja")]
    assert [i["plain"] for i in s["unchanged"]] == ["krishna"]
    assert s["lmissing"] == [] and not r["whole_list"]          # a short file: additions and changes only
    whole = [{"row": n, "plain": k, "result": v} for n, (k, v) in enumerate(LIST.items()) if k != "god"]
    r2 = harvest.review_list(whole, [], LIST)
    assert r2["whole_list"] and [i["plain"] for i in r2["sections"]["lmissing"]] == ["god"]
    d = harvest.default_decisions(r2)
    assert d[r2["sections"]["lmissing"][0]["id"]] == {"choice": "keep"}


# ---------- choices and Accept ----------

def test_plan_from_choices():
    r = review()
    s = r["sections"]
    clash, new, unmarked = by_form(s["clash"]), by_form(s["new"]), {i["plain"]: i for i in s["unmarked"]}
    d = harvest.default_decisions(r)
    assert set(harvest.undecided(r, d)) == {i["id"] for i in s["clash"] + s["new"]}
    d[clash["Kṛshṇa"]["id"]] = {"choice": "doc:0", "cap": "keep"}             # always capital stays
    d[clash["Ātma"]["id"]] = {"choice": "list", "cap": "capital"}             # our list, but always capital
    d[clash["Kshamā Yāchanā"]["id"]] = {"choice": "pending", "cap": "keep"}
    assert clash["kănarā"]["note"] == "only the capital letter differs" and clash["kănarā"]["gives"] == "Kănarā"
    d[clash["kănarā"]["id"]] = {"choice": "list", "cap": "keep"}             # nothing changes
    d[new["Akartā"]["id"]] = {"choice": "add:0", "cap": "capital"}
    d[new["Dīpa"]["id"]] = {"choice": "add:0", "cap": "normal"}
    d[new["Abhoktā"]["id"]] = {"choice": "pending", "cap": "normal"}
    d[unmarked["chitrapur"]["id"]] = {"choice": "pending"}
    for i in s["new"]:
        d.setdefault(i["id"], {}).setdefault("choice", "skip")
    p = harvest.plan(r, d, "talk.docx")
    assert p["undecided"] == [] and p["clashing"] == []
    assert {(c["plain"], c["old"], c["result"]) for c in p["update"]} == {
        ("krshna", "{{}}{{kṛs͟hṇa}}", "{{}}{{kṛshṇa}}"), ("atma", "ātmā", "{{}}{{ātmā}}")}
    assert {(c["plain"], c["result"]) for c in p["add"]} == {("akarta", "{{}}{{akartā}}"), ("dipa", "dīpa")}
    assert sorted(x["plain"] for x in p["pending"]) == ["abhokta", "chitrapur", "kshama yachana"]
    assert {x["plain"]: x["suggested"] for x in p["pending"]}["kshama yachana"] == "kshamā yāchanā"
    assert all(x["source"] == "talk.docx" and x["context"] for x in p["pending"])
    assert harvest.plan_sentence(p) == "Accept will add 2 words, change 2, and put 3 on the pending list."
    assert harvest.plan_sentence(p, done=True).startswith("Accepted: added 2 words")


def test_extras_and_name_entries():
    r = review("We offered thālī-s to the Gŏd.\n")
    new = by_form(r["sections"]["new"])["thālī-s"]
    assert new["extras"] == [{"plain": "thalis", "result": "thālī-s"}, {"plain": "thali", "result": "thālī"}]
    d = {new["id"]: {"choice": "add:0", "cap": "normal", "extras": [0]}}
    clash = r["sections"]["clash"][0]
    d[clash["id"]] = {"choice": "doc:0", "cap": "keep"}                       # a Name entry typed with a capital
    p = harvest.plan(r, d)
    assert {(c["plain"], c["result"]) for c in p["add"]} == {("thali-s", "thālī-s"), ("thalis", "thālī-s")}
    assert p["update"] == [{"plain": "god", "old": "{{gŏḍ}}{{god}}", "result": "{{gŏḍ}}{{gŏd}}", "item": clash["id"]}]
    assert harvest.replace_result("{{gŏḍ}}{{god}}", "gŏd", capitalised=False) == "{{gŏd}}{{god}}"
    assert harvest.with_type("{{}}{{kārtik, kārtika}}", "normal") == "{{kārtik, kārtika}}"
    assert harvest.with_type("ātmā", "capital") == "{{}}{{ātmā}}"
    assert harvest.with_type("{{gŏḍ}}{{god}}", "capital") == "{{}}{{gŏḍ}}"


# ---------- the edit page's boxes <-> stored code ----------

@pytest.mark.parametrize("code,boxes", [
    ("kṛs͟hṇa", {"type": "normal", "result": "kṛs͟hṇa"}),
    ("{{}}{{kṛs͟hṇa}}", {"type": "capital", "result": "kṛs͟hṇa"}),
    ("{{abhis͟heka}}{{abhis͟hek}}", {"type": "name", "result": "abhis͟heka", "name": "abhis͟hek"}),
    ("{{nāmāvalī (hindi), nāmāvaliḥ (sanskrit)}}",
     {"type": "choices", "choices": ["nāmāvalī (hindi)", "nāmāvaliḥ (sanskrit)"]}),
])
def test_boxes_and_code_both_ways(code, boxes):
    got = harvest.code_to_boxes(code)
    assert got["advanced"] == "" and {k: got[k] for k in boxes} == boxes
    assert harvest.boxes_to_code(got) == code
    assert harvest.box_problems(got) == []


@pytest.mark.parametrize("code", ["{{}}{{kārtik, kārtika}}", "{{saṅgīta}}{{saṅgeetā, saṅgītā}}", "{{pūjā}}",
                                  "{{a}}{{b}}{{c}}", "a, b"])
def test_codes_the_boxes_cant_show_go_to_advanced(code):
    boxes = harvest.code_to_boxes(code)
    assert boxes["advanced"] == code
    assert harvest.boxes_to_code(boxes) == harvest.clean_result(code)


def test_boxes_are_cleaned_and_checked():
    assert harvest.boxes_to_code({"type": "normal", "result": "  Kṛs̲h̲ṇa "}) == "kṛs͟hṇa"
    assert harvest.boxes_to_code({"type": "name", "result": "Gŏḍ", "name": "God"}) == "{{gŏḍ}}{{god}}"
    assert harvest.boxes_to_code({"type": "choices", "choices": ["Kārtik", "", "kārtika "]}) == "{{kārtik, kārtika}}"
    assert harvest.boxes_to_code({"type": "normal", "result": "x", "advanced": "{{}}{{Kārtik, kārtika}}"}) == \
        "{{}}{{kārtik, kārtika}}"
    assert harvest.box_problems({"type": "normal", "result": ""}) == ["Please type the result."]
    assert "Choices needs at least two" in harvest.box_problems({"type": "choices", "choices": ["one"]})[0]
    assert "without {{ }} or commas" in harvest.box_problems({"type": "normal", "result": "a, b"})[0]
    assert harvest.box_problems({"type": "name", "result": "gŏḍ", "name": ""}) == ["Please type how it's spelt as a name."]
    assert "can't be used" in harvest.box_problems({"advanced": "{{a}}{{b}}{{c}}"})[0]


def test_describe_and_preview():
    assert harvest.describe("{{}}{{kṛs͟hṇa}}") == "Kṛs͟hṇa (always capital)"
    assert harvest.describe("{{gŏḍ}}{{god}}") == "gŏḍ / as a name: God"
    assert harvest.describe("{{nāmāvalī (hindi), nāmāvaliḥ (sanskrit)}}") == \
        "choice: nāmāvalī (hindi) or nāmāvaliḥ (sanskrit)"
    assert harvest.describe("{{}}{{kārtik, kārtika}}") == "choice: Kārtik or Kārtika (always capital)"
    assert harvest.describe("{{}}{{parijñānāshrama trayodashī}}") == "Parijñānāshrama Trayodashī (always capital)"
    assert harvest.preview("god", "{{gŏḍ}}{{god}}") == ("gŏḍ", "God")
    assert harvest.preview("krshna", "{{}}{{kṛs͟hṇa}}") == ("Kṛs͟hṇa", "Kṛs͟hṇa")
    assert harvest.small_result("{{}}{{Kṛs͟hṇa}}") == "kṛs͟hṇa"


# ---------- the report ----------

def test_report_has_every_section_with_the_decisions():
    from openpyxl import load_workbook
    r = review()
    d = harvest.default_decisions(r)
    data = harvest.report_xlsx(r, d, {"title": "Review #1", "list": "the trial copy", "files": ["talk.docx"],
                                      "made": "20 Nov 2026", "by": "admin@gmail.com", "status": "not accepted yet"})
    wb = load_workbook(io.BytesIO(data))
    assert wb.sheetnames == ["Summary", "1 Different from our list", "2 New words", "3 Written without marks",
                             "4 Style only", "5 Doubtful", "6 Already in our list"]
    unmarked = [row for row in wb["3 Written without marks"].iter_rows(min_row=3, values_only=True)]
    assert unmarked and all(row[-1] == "Nothing to change" for row in unmarked)
    assert [row[-1] for row in wb["4 Style only"].iter_rows(min_row=3, values_only=True)] == ["Nothing to change"] * 2
    assert any("Not accepted yet" in str(c) for row in wb["Summary"].iter_rows(values_only=True) for c in row)
    lst = harvest.list_xlsx([{"plain_text": "b", "diacritic_text": "bā"}, {"plain_text": "a", "diacritic_text": "ā"}])
    rows, problems = harvest.read_list_xlsx(lst)                              # the download reads back as a list
    assert [(x["plain"], x["result"]) for x in rows] == [("a", "ā"), ("b", "bā")] and problems == []
