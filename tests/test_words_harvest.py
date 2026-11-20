"""The Diacritical words dashboard's pure logic: reading uploads (.docx, .txt, PDF, .xlsx lists) and sorting a
document's words against a small fake list (diacritical/harvest.py); the report and pending list workbooks, reading
them back filled in (drop-down values, free text, sorted and deleted rows, correct spellings, capitals) and what Apply
does (diacritical/sheets.py). No database.
  .venv/bin/python -m pytest tests/test_words_harvest.py -q
"""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diacritical import harvest, sheets  # noqa: E402

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


def test_spellings_of_one_word_are_one_item_with_a_row_each():
    text = "Kṛshṇa and Kṛs͟hṇa and Krshṇa met. Then Vināyaka and Vinayakā came.\n"
    r = review(text)
    assert len(r["sections"]["clash"]) == 1
    clash = r["sections"]["clash"][0]
    forms = {f["form"]: f["status"] for f in clash["forms"]}
    assert forms == {"Kṛshṇa": "differs", "Krshṇa": "differs", "Kṛs͟hṇa": "same as our list"}
    assert sorted(clash["options"]) == ["krshṇa", "kṛshṇa"]
    assert [x["result"] for x in clash["rows"]] == clash["options"] and all(x["times"] == 1 for x in clash["rows"])
    assert sheets.refs(clash) == [f"{clash['id']}:0", f"{clash['id']}:1"]
    new = r["sections"]["new"]
    assert len(new) == 1 and sorted(new[0]["options"]) == ["vinayakā", "vināyaka"]
    assert {x["form"] for x in new[0]["rows"]} == {"Vināyaka", "Vinayakā"}


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


# ---------- stored codes ----------

def test_codes_types_and_descriptions():
    assert harvest.with_type("{{}}{{kārtik, kārtika}}", "normal") == "{{kārtik, kārtika}}"
    assert harvest.with_type("ātmā", "capital") == "{{}}{{ātmā}}"
    assert harvest.with_type("{{gŏḍ}}{{god}}", "capital") == "{{}}{{gŏḍ}}"
    assert harvest.with_type("ātmā", "keep") == "ātmā"
    assert harvest.replace_result("{{gŏḍ}}{{god}}", "gŏd", capitalised=False) == "{{gŏd}}{{god}}"
    assert harvest.replace_result("{{gŏḍ}}{{god}}", "gŏd", capitalised=True) == "{{gŏḍ}}{{gŏd}}"
    assert harvest.replace_result("{{}}{{kanarā}}", "kănarā", capitalised=True) == "{{}}{{kănarā}}"
    assert harvest.describe("{{}}{{kṛs͟hṇa}}") == "Kṛs͟hṇa (always capital)"
    assert harvest.describe("{{gŏḍ}}{{god}}") == "gŏḍ / as a name: God"
    assert harvest.describe("{{nāmāvalī (hindi), nāmāvaliḥ (sanskrit)}}") == \
        "choice: nāmāvalī (hindi) or nāmāvaliḥ (sanskrit)"
    assert harvest.describe("{{}}{{kārtik, kārtika}}") == "choice: Kārtik or Kārtika (always capital)"
    assert harvest.small_result("{{}}{{Kṛs͟hṇa}}") == "kṛs͟hṇa"


# ---------- the report ----------

INFO = {"review": 7, "list": "trial", "list_label": "the trial copy", "files": ["talk.docx"], "made": "20 Nov 2026",
        "by": "a@gmail.com", "status": "Waiting for your decisions", "made_iso": "2026-11-20T10:00:00"}


def workbook(data):
    from openpyxl import load_workbook
    return load_workbook(io.BytesIO(data))


def rows_of(ws):
    heads = [c.value for c in ws[2]]
    return [dict(zip(heads, r)) for r in ws.iter_rows(min_row=3, values_only=True)]


def fill(data, answers, *, delete=(), reverse=()) -> bytes:
    """Fill a workbook as a person would. answers: {(sheet, plain word or Ref): {column header: value}}. Rows of the
    sheets in `reverse` are put in the opposite order (as if sorted); rows whose plain word is in `delete` are
    deleted."""
    wb = workbook(data)
    for ws in wb.worksheets:
        if ws.title in ("Start here", sheets.DASHBOARD):
            continue
        heads = [c.value for c in ws[2]]
        values = [list(r) for r in ws.iter_rows(min_row=3, values_only=True)]
        if ws.title in reverse:
            values.reverse()
        for row in values:
            plain = row[heads.index("Plain word")] if "Plain word" in heads else None
            ref = row[heads.index("Ref")] if "Ref" in heads else None
            for key in ((ws.title, plain), (ws.title, ref)):
                for header, value in answers.get(key, {}).items():
                    row[heads.index(header)] = value
        for n, row in enumerate(values, 3):
            for i, v in enumerate(row, 1):
                ws.cell(n, i).value = v
        for n in range(len(values) + 2, 2, -1):
            if "Plain word" in heads and ws.cell(n, heads.index("Plain word") + 1).value in delete:
                ws.delete_rows(n)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def test_report_layout_drop_downs_and_hidden_parts():
    r = review()
    wb = workbook(sheets.report_xlsx(r, INFO))
    assert wb.sheetnames == ["Start here", "Different from our list", "New words", "Written without marks",
                             "Style only", "Doubtful", "Already in our list", "_dashboard"]
    assert wb["_dashboard"].sheet_state == "hidden" and all(
        ws.sheet_state == "visible" for ws in wb.worksheets[:-1])
    meta = {row[0]: row[1] for row in wb["_dashboard"].iter_rows(min_row=2, values_only=True) if row[0]}
    assert meta["kind"] == "report" and meta["review"] == 7 and meta["list"] == "trial"
    start = "\n".join(str(c) for row in wb["Start here"].iter_rows(values_only=True) for c in row if c)
    assert "Fill in the yellow column “Your decision” on “Different from our list” and “New words”" in start
    assert "Anything left blank goes to the pending list" in start and "Upload your decisions" in start

    clash = wb["Different from our list"]
    heads = [c.value for c in clash[2]]
    assert heads == ["Word (as in the document)", "Plain word", "Times", "Pages", "Example line", "Our list gives",
                     "Your decision", "Capital", "Correct spelling", "Notes", "Ref"]
    assert clash.freeze_panes == "A3" and clash.auto_filter.ref.startswith("A2:K")
    assert clash.column_dimensions["K"].hidden                              # Ref
    last = clash.max_row
    dvs = {str(dv.sqref): dv for dv in clash.data_validations.dataValidation}
    assert dvs[f"G3:G{last}"].formula1 == '"Our list is right,Document is right,Pending"'
    assert dvs[f"H3:H{last}"].formula1 == '"Normal,Always capital"'
    assert all(dv.errorStyle == "warning" and dv.allow_blank and dv.type == "list" for dv in dvs.values())
    assert clash["G3"].fill.fgColor.rgb.endswith("FFF2A8") and clash["F3"].fill.fgColor.rgb.endswith("000000")
    gives = {row["Plain word"]: row["Our list gives"] for row in rows_of(clash)}
    assert gives["krshna"] == "Kṛs͟hṇa (always capital)" and gives["atma"] == "Ātmā"

    new = wb["New words"]
    dvs = {str(dv.sqref): dv.formula1 for dv in new.data_validations.dataValidation}
    assert dvs[f"F3:F{new.max_row}"] == '"Add,Pending,Skip"'
    caps = {row["Plain word"]: row["Capital"] for row in rows_of(new)}
    assert caps["ganesha"] == "Always capital" and caps["dipa"] is None   # only ever seen with a capital mid-sentence
    unmarked = wb["Written without marks"]
    assert {str(dv.sqref): dv.formula1 for dv in unmarked.data_validations.dataValidation} == {
        f"G3:G{unmarked.max_row}": '"Nothing to change,Pending"'}
    assert all(row["Your decision"] == "Nothing to change" for row in rows_of(unmarked))
    for title in ("Style only", "Doubtful", "Already in our list"):
        ws = wb[title]
        assert not ws.data_validations.dataValidation and ws["A2"].fill.fgColor.rgb.endswith("E4E4E4")
        assert {row["Decision"] for row in rows_of(ws)} == {"Nothing to change"}

    small = sheets.report_xlsx(review("Shrī Kṛshṇa came.\n"), INFO)          # empty sheets are left out
    assert workbook(small).sheetnames == ["Start here", "Different from our list", "Already in our list", "_dashboard"]


def test_reading_back_a_filled_report():
    r = review()
    data = fill(sheets.report_xlsx(r, INFO), {
        ("Different from our list", "krshna"): {"Your decision": "Document is right"},
        ("Different from our list", "atma"): {"Your decision": "Our list is correct. Always capitalised"},
        ("Different from our list", "kanara"): {"Your decision": "maybe?"},
        ("Different from our list", "kshama yachana"): {"Your decision": "  PENDING ", "Notes": "ask Gopal"},
        ("New words", "akarta"): {"Your decision": "Add", "Capital": "Always capital", "Correct spelling": "Akartā "},
        ("New words", "dipa"): {"Your decision": "add", "Capital": None, "Correct spelling": "dīpā"},
        ("New words", "ganesha"): {"Your decision": "Skip"},
        ("Written without marks", "chitrapur"): {"Your decision": "Pending", "Notes": "registered name"},
    }, reverse=("Different from our list", "New words"), delete=("abhokta",))
    filled = sheets.read_filled("talk-filled.xlsx", data)
    assert filled["kind"] == "report" and filled["review"] == 7 and filled["list"] == "trial"
    answers = sheets.answers_from_report(r, filled)
    assert [(x["plain"], x["typed"]) for x in answers["unclear"]] == [("kanara", "maybe?")]
    assert answers["missing"] == ["abhokta"]                                  # its row was deleted: pending
    assert [(x["word"], x["read"]) for x in answers["words"]] == [("Ātma", "Our list is right · Always capital")]
    p = sheets.plan(r, answers, dict(LIST), "talk.docx")
    assert {(c["plain"], c["old"], c["result"]) for c in p["update"]} == {
        ("krshna", "{{}}{{kṛs͟hṇa}}", "{{}}{{kṛshṇa}}"), ("atma", "ātmā", "{{}}{{ātmā}}")}
    assert {(c["plain"], c["result"]) for c in p["add"]} == {("akarta", "{{}}{{akartā}}"), ("dipa", "dīpā")}
    pending = {x["plain"]: x for x in p["pending"]}
    assert set(pending) == {"kanara", "kshama yachana", "abhokta", "chitrapur"}
    assert pending["kshama yachana"]["notes"] == "ask Gopal" and pending["chitrapur"]["notes"] == "registered name"
    assert pending["kanara"]["suggested"] == "{{}}{{kănarā}}"               # in the entry's own form
    assert pending["chitrapur"]["suggested"] == "{{}}{{chitrāpur}}" and pending["abhokta"]["source"] == "talk.docx"
    assert p["resolve"] == ["atma"] and p["skipped"] == []

    changed = dict(LIST, krshna="kṛṣṇa", dipa="dīpa")                         # changed since the report was made
    p2 = sheets.plan(r, answers, changed)
    assert sorted((x["plain"], x["now"]) for x in p2["skipped"]) == [("dipa", "dīpa"), ("krshna", "kṛṣṇa")]
    assert {c["plain"] for c in p2["add"] + p2["update"]} == {"akarta", "atma"}
    p2.update(pending_new=p2["pending"], pending_off=1)
    assert sheets.sentence(p2) == "This will add 1 word, change 1, put 4 on the pending list, and take 1 off the " \
                                  "pending list."
    assert sheets.sentence(dict(p2, add=[], update=[], pending_new=[], pending_off=0)) == \
        "This changes nothing in the list."

    # a report re-downloaded shows what was uploaded
    again = workbook(sheets.report_xlsx(r, INFO, answers["cells"]))
    assert {row["Plain word"]: row["Your decision"] for row in rows_of(again["Different from our list"])}["krshna"] \
        == "Document is right"


def test_a_word_spelt_two_ways_and_unclear_rows():
    r = review("Kṛshṇa and Krshṇa met at the dīpa.\n")
    clash = r["sections"]["clash"][0]
    blank = sheets.report_xlsx(r, INFO)
    both = fill(blank, {("Different from our list", ref): {"Your decision": "Document is right"}
                        for ref in sheets.refs(clash)})
    a = sheets.answers_from_report(r, sheets.read_filled("x.xlsx", both))
    assert a["rows"][clash["id"]]["choice"] == "pending" and "more than one spelling" in a["unclear"][0]["why"]
    one = fill(blank, {("Different from our list", sheets.refs(clash)[1]): {"Your decision": "magazine is right"}})
    p = sheets.plan(r, sheets.answers_from_report(r, sheets.read_filled("x.xlsx", one)), LIST)
    assert [(c["plain"], c["result"]) for c in p["update"]] == [("krshna", "{{}}{{" + clash["options"][1] + "}}")]
    odd = fill(blank, {("Different from our list", sheets.refs(clash)[0]): {"Your decision": "Our list is right",
                                                                          "Correct spelling": "kṛṣṇa"},
                       ("New words", "dipa"): {"Your decision": "Add", "Capital": "sometimes"}})
    a = sheets.answers_from_report(r, sheets.read_filled("x.xlsx", odd))
    assert sorted(x["plain"] for x in a["unclear"]) == ["dipa", "krshna"]
    assert {d["choice"] for d in a["rows"].values()} == {"pending"}


@pytest.mark.parametrize("kind,typed,key", [
    ("clash", "Our list is right", "list"), ("clash", "our  list is RIGHT.", "list"),
    ("clash", "Magazine is correct. Always capitalised", "doc"), ("clash", "Keep on pending list", "pending"),
    ("clash", "Our list is right. But when the input is capitalised the output should be too", "list"),
    ("clash", "our list is right, or the magazine is right?", None), ("clash", "yes", None), ("clash", "", None),
    ("new", "Keep on Pending list", "pending"), ("new", "yes", "add"), ("new", "ignore", "skip"),
    ("unmarked", "ignore", "none"), ("unmarked", "Our list is correct", "none"),
    ("pending", "Remove from pending", "remove"), ("pending", "add", "add"), ("pending", "pending", "stay"),
])
def test_reading_what_people_type(kind, typed, key):
    assert sheets.read_choice(kind, typed)[0] == key


def test_capitals_in_words():
    assert sheets.capital_in_words("This word should always be capitalised, it's a place name") == "capital"
    assert sheets.capital_in_words("Need not be capitalised always (unless at the beginning)") == "normal"
    assert sheets.capital_in_words("when the input is capitalised the output should also be") is None
    assert sheets.read_capital("always capitalised") == "capital" and sheets.read_capital(" ") is None
    assert sheets.read_capital("Normal") == "normal" and sheets.read_capital("sometimes") == "?"


def test_files_that_are_not_ours_are_refused():
    with pytest.raises(sheets.NotOurs, match="isn't a report or pending list"):
        sheets.read_filled("mine.xlsx", make_xlsx([["plain_text", "diacritic_text"], ["puja", "pūjā"]]))
    with pytest.raises(harvest.ReadProblem, match="save the file as an Excel workbook"):
        sheets.read_filled("old.xls", b"x")
    with pytest.raises(harvest.ReadProblem, match="couldn't be opened"):
        sheets.read_filled("broken.xlsx", b"PK broken")
    assert not sheets.is_dashboard_file(make_xlsx([["a"]]))
    assert sheets.is_dashboard_file(sheets.report_xlsx(review(), INFO))


def test_offline_list_report_round_trip():
    rows = [{"row": 2, "plain": "abhokta", "result": "abhoktā"}, {"row": 3, "plain": "puja", "result": "pūja"},
            {"row": 4, "plain": "dipa", "result": "dīpa"}]
    r = harvest.review_list(rows, [], LIST)
    wb = workbook(sheets.report_xlsx(r, INFO))
    assert wb.sheetnames == ["Start here", "New in your list", "Changed", "_dashboard"]
    assert {row["Plain word"]: row["Your decision"] for row in rows_of(wb["New in your list"])} == {
        "abhokta": "Add", "dipa": "Add"}                                       # Add is filled in for a list's words
    data = fill(sheets.report_xlsx(r, INFO), {("New in your list", "dipa"): {"Your decision": "Pending"},
                                              ("Changed", "puja"): {"Your decision": "Use your list"}})
    p = sheets.plan(r, sheets.answers_from_report(r, sheets.read_filled("x.xlsx", data)), LIST)
    assert [(c["plain"], c["result"]) for c in p["add"]] == [("abhokta", "abhoktā")]
    assert [(c["plain"], c["result"]) for c in p["update"]] == [("puja", "pūja")]
    assert [x["plain"] for x in p["pending"]] == ["dipa"]


# ---------- the pending list ----------

def test_pending_list_round_trip():
    from datetime import datetime
    when = datetime(2026, 11, 20, 5, 0)
    items = [
        {"id": 11, "plain_text": "abhokta", "suggested": "{{}}{{abhoktā}}", "reason": "New word", "source": "x.pdf",
         "page": "39", "context": "Akartā-Abhoktā", "notes": "", "added_at": when},
        {"id": 12, "plain_text": "kanara", "suggested": "{{}}{{kănarā}}", "reason": "Different", "source": "x.pdf",
         "page": "21", "context": "Kănarā Sāraswat", "notes": "asked Gopal", "added_at": when},
        {"id": 13, "plain_text": "dipa", "suggested": "dīpa", "reason": "New word", "source": "x.pdf", "page": "2",
         "context": "", "notes": "", "added_at": when},
        {"id": 14, "plain_text": "ganesha", "suggested": "gaṇesha", "reason": "New word", "source": "", "page": "",
         "context": "", "notes": "", "added_at": when},
        {"id": 15, "plain_text": "seva", "suggested": "sevā", "reason": "New word", "source": "", "page": "",
         "context": "", "notes": "", "added_at": when},
        {"id": 16, "plain_text": "atma", "suggested": "ātma", "reason": "Different", "source": "", "page": "",
         "context": "", "notes": "", "added_at": when},
    ]
    current = dict(LIST, kanara="{{}}{{kanarā}}")
    data = sheets.pending_xlsx(items, current, {"list": "trial", "made_iso": "2026-11-20T10:00:00",
                                                "when": lambda t: t.strftime("%d %b %Y")})
    wb = workbook(data)
    assert wb.sheetnames == ["Pending", "_dashboard"] and wb["_dashboard"].sheet_state == "hidden"
    ws = wb["Pending"]
    rows = {row["Plain word"]: row for row in rows_of(ws)}
    assert rows["kanara"]["Result"] == "kănarā" and rows["kanara"]["Capital"] == "Always capital"
    assert rows["kanara"]["Our list now gives"] == "Kanarā (always capital)"
    assert rows["kanara"]["Notes"] == "asked Gopal"
    assert rows["dipa"]["Result"] == "dīpa" and rows["dipa"]["Capital"] is None
    assert {str(dv.sqref).split(":")[0][0]: dv.formula1 for dv in ws.data_validations.dataValidation} == {
        "I": '"Add,Remove from pending"', "K": '"Normal,Always capital"'}
    assert ws.column_dimensions["M"].hidden

    filled = sheets.read_filled("pending.xlsx", fill(data, {
        ("Pending", "abhokta"): {"Your decision": "Add", "Result": "Abhoktā", "Capital": None},   # blank: Normal
        ("Pending", "kanara"): {"Your decision": "Add", "Capital": None},      # blank: keeps always capital
        ("Pending", "dipa"): {"Your decision": "Remove from pending"},
        ("Pending", "ganesha"): {"Notes": "team asked"},                       # stays, with a note
        ("Pending", "seva"): {"Your decision": "perhaps"},                     # can't be read: stays
        ("Pending", "atma"): {"Your decision": "Add"},                         # its entry changed since: skipped
    }, reverse=("Pending",)))
    assert filled["kind"] == "pending" and filled["snapshot"]["12"]["entry"] == "{{}}{{kanarā}}"
    answers = sheets.answers_from_pending(filled)
    open_items = {i["id"]: i for i in items}
    p = sheets.plan_pending(answers, open_items, dict(current, atma="{{}}{{ātmā}}"))
    assert [(c["plain"], c["result"]) for c in p["add"]] == [("abhokta", "abhoktā")]
    assert [(c["plain"], c["old"], c["result"]) for c in p["update"]] == [
        ("kanara", "{{}}{{kanarā}}", "{{}}{{kănarā}}")]
    assert p["remove"] == [13] and p["notes"] == {14: "team asked"} and sorted(p["done"]) == [11, 12]
    assert [x["plain"] for x in p["unclear"]] == ["seva"] and [x["plain"] for x in p["skipped"]] == ["atma"]
    del open_items[11]                                                         # answered meanwhile
    p = sheets.plan_pending(answers, open_items, current)
    assert [x["why"] for x in p["skipped"] if x["plain"] == "abhokta"] == ["already taken off the pending list"]
    assert sheets.compose("Kṛs̲h̲ṇa", None, None) == "kṛs͟hṇa"                  # house style
    assert sheets.compose("gŏd", None, "{{gŏḍ}}{{god}}") == "{{gŏd}}{{god}}"
    assert sheets.compose("gŏd", "normal", "{{gŏḍ}}{{god}}") == "gŏd"


def test_whole_list_reads_back_as_a_list():
    lst = sheets.list_xlsx([{"plain_text": "b", "diacritic_text": "bā"}, {"plain_text": "a", "diacritic_text": "ā"}])
    rows, problems = harvest.read_list_xlsx(lst)
    assert [(x["plain"], x["result"]) for x in rows] == [("a", "ā"), ("b", "bā")] and problems == []
