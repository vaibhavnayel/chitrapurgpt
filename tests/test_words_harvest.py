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
    assert sheets.refs(clash) == [clash["id"]]                               # one row per word
    assert harvest.old_refs(clash) == {f"{clash['id']}:0": 0, f"{clash['id']}:1": 1}   # what reports had before
    assert [x["form"] for x in clash["same"]] == ["Kṛs͟hṇa"] and clash["times"] == 3
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
    assert harvest.describe("{{gŏḍ}}{{god}}") == "gŏḍ / with a capital: God"
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
                     "Your decision", "Capital", "Correct spelling", "Also typed as", "Notes for self", "Ref"]
    assert clash.freeze_panes == "A3" and clash.auto_filter.ref.startswith("A2:L")
    assert clash.column_dimensions["L"].hidden                              # Ref
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
        ("Different from our list", "kshama yachana"): {"Your decision": "  PENDING ", "Notes for self": "ask Gopal"},
        ("New words", "akarta"): {"Your decision": "Add", "Capital": "Always capital", "Correct spelling": "Akartā "},
        ("New words", "dipa"): {"Your decision": "add", "Capital": None, "Correct spelling": "dīpā"},
        ("New words", "ganesha"): {"Your decision": "Skip"},
        ("Written without marks", "chitrapur"): {"Your decision": "Pending", "Notes for self": "registered name"},
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
    row = {x["Plain word"]: x for x in rows_of(workbook(blank)[DIFF])}["krshna"]
    assert row["Word (as in the document)"] == "Krshṇa, Kṛshṇa" and row["Ref"] == clash["id"] and row["Times"] == 2
    # 'Document is right' alone can't say which of the document's spellings: pending, listed
    which = fill(blank, {(DIFF, "krshna"): {"Your decision": "Document is right"}})
    a = sheets.answers_from_report(r, sheets.read_filled("x.xlsx", which))
    assert a["rows"][clash["id"]]["choice"] == "pending" and a["rows"][clash["id"]].get("unclear")
    assert [x["why"] for x in a["unclear"]] == ["the document spells it more than one way: type the spelling(s) "
                                                "you want in “Correct spelling”"]
    assert [x["plain"] for x in sheets.plan(r, a, LIST)["pending"]] == ["krshna", "dipa"]
    one = fill(blank, {(DIFF, "krshna"): {"Your decision": "magazine is right", "Correct spelling": "Krshṇa"}})
    p = sheets.plan(r, sheets.answers_from_report(r, sheets.read_filled("x.xlsx", one)), LIST)
    assert [(c["plain"], c["result"]) for c in p["update"]] == [("krshna", "{{}}{{krshṇa}}")]
    odd = fill(blank, {(DIFF, "krshna"): {"Your decision": "Our list is right", "Correct spelling": "kṛṣṇa"},
                       (NEW, "dipa"): {"Your decision": "Add", "Capital": "sometimes"}})
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
    assert rows["kanara"]["Notes for self"] == "asked Gopal"
    assert rows["dipa"]["Result"] == "dīpa" and rows["dipa"]["Capital"] is None
    assert {str(dv.sqref).split(":")[0][0]: dv.formula1 for dv in ws.data_validations.dataValidation} == {
        "I": '"Add,Remove from pending"', "L": '"Normal,Always capital"'}
    assert ws.column_dimensions["N"].hidden

    filled = sheets.read_filled("pending.xlsx", fill(data, {
        ("Pending", "abhokta"): {"Your decision": "Add", "Result": "Abhoktā", "Capital": None},   # blank: Normal
        ("Pending", "kanara"): {"Your decision": "Add", "Capital": None},      # blank: keeps always capital
        ("Pending", "dipa"): {"Your decision": "Remove from pending"},
        ("Pending", "ganesha"): {"Notes for self": "team asked"},                       # stays, with a note
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



# ---------- 'Also typed as' and words typed in by hand ----------

VLIST = dict(LIST, krsna="{{}}{{kṛs͟hṇa}}")
VTEXT = "Shrī Kṛshṇa came. The Nāmavali was sung.\nGaṇesha and Akartā and Abhoktā with the dīpa.\n"
DIFF, NEW = "Different from our list", "New words"


def add_rows(data: bytes, sheet: str, rows: list[dict], at: int | None = None) -> bytes:
    """Rows typed in by hand ({column header: value}, no Ref): at the bottom, or inserted before row `at`."""
    wb = workbook(data)
    ws = wb[sheet]
    heads = [c.value for c in ws[2]]
    for n, values in enumerate(rows):
        row = (at + n) if at else ws.max_row + 1
        if at:
            ws.insert_rows(row)
        for header, value in values.items():
            ws.cell(row, heads.index(header) + 1).value = value
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def test_also_typed_as_column_and_start_here():
    wb = workbook(sheets.report_xlsx(review(), INFO))
    start = "\n".join(str(c) for row in wb["Start here"].iter_rows(values_only=True) for c in row if c)
    assert "“Also typed as” (optional): other ways people type the word, separated by commas, e.g. krishna, " \
           "krushna. Each gets the same result as the word." in start
    assert "To add a word that isn't in the report, type it on a new row at the bottom of “New words” (the word " \
           "with its marks) and choose Add." in start
    for title, after in ((DIFF, "Correct spelling"), (NEW, "Correct spelling")):
        ws = wb[title]
        heads = [c.value for c in ws[2]]
        i = heads.index("Also typed as")
        assert heads[i - 1] == after and heads[i + 1] == "Notes for self", title
        letter = ws.cell(2, i + 1).column_letter
        assert ws.column_dimensions[letter].width == 24
        assert ws[f"{letter}2"].fill.fgColor.rgb.endswith("FFF2A8") and ws[f"{letter}3"].fill.fgColor.rgb.endswith(
            "FFF2A8")                                                            # yellow, to fill in
        assert not any(str(dv.sqref).startswith(letter) for dv in ws.data_validations.dataValidation)  # no drop-down
    assert "Also typed as" not in [c.value for c in wb["Written without marks"][2]]

    lst = harvest.review_list([{"row": 2, "plain": "abhokta", "result": "abhoktā"},
                               {"row": 3, "plain": "puja", "result": "pūja"}], [], LIST)
    wb = workbook(sheets.report_xlsx(lst, INFO))
    for title in ("New in your list", "Changed"):
        heads = [c.value for c in wb[title][2]]
        assert heads[heads.index("Your decision") + 1] == "Also typed as", title
    start = "\n".join(str(c) for row in wb["Start here"].iter_rows(values_only=True) for c in row if c)
    assert "“Also typed as” (optional)" in start and "New words" not in start


def test_also_typed_as_in_a_filled_report():
    r = review(VTEXT, VLIST)
    data = fill(sheets.report_xlsx(r, INFO), {
        (DIFF, "krshna"): {"Your decision": "Our list is right",                      # always capital
                           "Also typed as": "krsna, Krishna; kṛshṇa,  KRUSHNA ,krushna"},
        (DIFF, "namavali"): {"Your decision": "Our list is right", "Also typed as": "naamavali"},   # choices
        (NEW, "ganesha"): {"Your decision": "Add", "Capital": "Always capital",
                           "Also typed as": "ganesh, Gaṇeśa\nganapati; gaṇesha (old)"},
        (NEW, "dipa"): {"Your decision": "Add", "Also typed as": "deepa; ganapati; akarta"},
        (NEW, "abhokta"): {"Your decision": "Skip", "Also typed as": "abhoktaa"},
        (NEW, "akarta"): {"Also typed as": "akartaa, akarta"},                       # blank: pending, they go along
    })
    filled = sheets.read_filled("x.xlsx", data)
    a = sheets.answers_from_report(r, filled)
    ids = {i["plain"]: i["id"] for i in sheets._items(r)}
    assert a["rows"][ids["krshna"]]["also"] == ["krsna", "krishna", "krushna"]  # cleaned; own word and repeats out
    assert a["rows"][ids["ganesha"]]["also_bad"] == ["gaṇesha (old)"]
    p = sheets.plan(r, a, VLIST, "talk.docx")
    adds = {c["plain"]: (c["result"], c.get("also_for")) for c in p["add"]}
    assert adds == {
        "ganesha": ("{{}}{{gaṇesha}}", None), "dipa": ("dīpa", None),
        "krushna": ("{{}}{{kṛs͟hṇa}}", "krshna"),                                   # the entry's code: always capital
        "naamavali": ("{{nāmāvalī (hindi), nāmāvaliḥ (sanskrit)}}", "namavali"),    # choices
        "ganesh": ("{{}}{{gaṇesha}}", "ganesha"), "ganesa": ("{{}}{{gaṇesha}}", "ganesha"),
        "deepa": ("dīpa", "dipa")}
    assert "krsna" not in adds                         # already in the list with the same code: nothing, not listed
    left = {(x["plain"], x["typed"]): x["why"] for x in p["also_left"]}
    assert ("krshna", "krsna") not in left
    assert left[("krshna", "krishna")] == ("krishna is already in the list as kṛs͟hṇa, so it was left as it is; "
                                           "change it on the Mappings page if needed.")
    assert left[("dipa, ganesha", "ganapati")].startswith("ganapati was given for dipa and ganesha, which get "
                                                          "different results")
    assert left[("dipa", "akarta")].startswith("akarta has its own row in this file")
    assert left[("abhokta", "abhoktaa")] == "not used, because the word was skipped."
    assert left[("ganesha", "gaṇesha (old)")].startswith("“gaṇesha (old)” can't be a typed word")
    assert len(left) == 5
    assert [(x["plain"], x["also"]) for x in p["pending"]] == [("akarta", "akartaa")]
    p["pending_new"] = p["pending"]
    assert sheets.sentence(p) == "This will add 7 words (5 of them other ways of typing a word) and put 1 on the " \
                                 "pending list."
    assert sheets.sentence(dict(p, add=p["add"][-1:]), done=True).startswith(
        "Applied: added 1 word (another way of typing a word)")

    # the word changed in the list since the report: its other ways aren't used either
    p2 = sheets.plan(r, a, dict(VLIST, krshna="kṛṣṇa"))
    assert "krushna" not in {c["plain"] for c in p2["add"]}
    assert "not used, because krshna was changed in the list after this file was made." in \
        [x["why"] for x in p2["also_left"]]
    # a re-downloaded report shows them again
    again = workbook(sheets.report_xlsx(r, INFO, a["cells"], a["extra_cells"]))
    assert {row["Plain word"]: row["Also typed as"] for row in rows_of(again[NEW])}["dipa"] == "deepa; ganapati; akarta"


def test_words_typed_in_on_new_words():
    r = review(VTEXT, VLIST)
    blank = sheets.report_xlsx(r, INFO)
    data = add_rows(fill(blank, {(NEW, "dipa"): {"Your decision": "Add"}}), NEW, [
        {"Word (as in the document)": "āvaraṇa-s", "Your decision": "Add", "Also typed as": "avaranas"},
        {"Word (as in the document)": "Āvaraṇa", "Your decision": "add", "Capital": "Always capital"},
        {"Word (as in the document)": "Kṛṣṇa", "Plain word": "krishna", "Your decision": "Add"},   # in the list
        {"Word (as in the document)": "Gaṇesha", "Your decision": "Add"},          # has its own row in the report
        {"Word (as in the document)": "sevā", "Your decision": "Add"},
        {"Word (as in the document)": "sēva", "Your decision": "Add"},             # the same plain word, differently
        {"Word (as in the document)": "pūjā", "Plain word": "puja", "Your decision": "Add"},   # exactly as in the list
        {"Word (as in the document)": "dhūpa", "Notes for self": "ask the team", "Also typed as": "dhoopa"},   # blank: pending
        {"Plain word": "foo", "Your decision": "Add"},                              # no word with marks
        {"Notes for self": "just a remark"},                                                 # no word at all: ignored
        {"Word (as in the document)": "tīrtha", "Your decision": "Skip"},
        {"Word (as in the document)": "mālā", "Your decision": "perhaps"},
    ])
    filled = sheets.read_filled("x.xlsx", data)
    assert len(filled["extra"]) == 11
    a = sheets.answers_from_report(r, filled)
    p = sheets.plan(r, a, VLIST, "talk.docx")
    adds = {c["plain"]: (c["result"], c.get("own_row"), c.get("also_for")) for c in p["add"]}
    rows = {x["word"]: x["row"] for x in filled["extra"]}
    assert adds == {"dipa": ("dīpa", None, None),
                    "avarana-s": ("āvaraṇa-s", f"New words, row {rows['āvaraṇa-s']}", None),
                    "avarana": ("{{}}{{āvaraṇa}}", f"New words, row {rows['Āvaraṇa']}", None),
                    "avaranas": ("āvaraṇa-s", None, "avarana-s")}
    left = {x["word"]: x["why"] for x in p["own_left"]}
    assert left["Kṛṣṇa"].startswith("krishna is already in the list as kṛs͟hṇa, so it was left as it is")
    assert left["Gaṇesha"].startswith("ganesha has its own row in this report")
    assert left["sevā"] == left["sēva"] and "more than one of your own rows" in left["sevā"]
    assert left["pūjā"] == "puja is already in the list exactly like this, so nothing changes."
    assert left["foo"] == "not added: there is no word with its marks in the “Word” column."
    assert left["mālā"] == "not added: “perhaps” isn't Add, Pending or Skip."
    assert "tīrtha" not in left and len(left) == 7
    pend = {x["plain"]: x for x in p["pending"]}
    assert set(pend) == {"abhokta", "akarta", "ganesha", "krshna", "namavali", "dhupa"}
    assert pend["dhupa"]["suggested"] == "dhūpa" and pend["dhupa"]["notes"] == "ask the team"
    assert pend["dhupa"]["also"] == "dhoopa" and "“New words” sheet" in pend["dhupa"]["reason"]
    # shown again at the bottom of a re-downloaded report
    again = rows_of(workbook(sheets.report_xlsx(r, INFO, a["cells"], a["extra_cells"]))[NEW])
    assert [row["Word (as in the document)"] for row in again[-11:]][:2] == ["āvaraṇa-s", "Āvaraṇa"]


def test_an_old_report_with_a_column_and_a_row_added_by_hand():
    r = review(VTEXT, VLIST)
    wb = workbook(sheets.report_xlsx(r, INFO))
    for title in (DIFF, NEW):                     # a report made before the column existed
        ws = wb[title]
        ws.delete_cols([c.value for c in ws[2]].index("Also typed as") + 1)
    out = io.BytesIO()
    wb.save(out)
    old = out.getvalue()
    assert sheets.read_filled("x.xlsx", old)["cells"]
    assert all(c["also"] is None for c in sheets.read_filled("x.xlsx", old)["cells"].values())
    wb = workbook(old)
    ws = wb[NEW]
    ws.insert_cols(3)                             # he inserts a column himself, with a heading of his own
    ws.cell(2, 3).value = "  ALSO typed  as (optional) "
    ws2 = wb[DIFF]
    ws2.insert_cols(12)
    ws2.cell(2, 12).value = "Variants"
    out = io.BytesIO()
    wb.save(out)
    data = fill(out.getvalue(), {(NEW, "dipa"): {"Your decision": "Add", "  ALSO typed  as (optional) ": "deepa"},
                                 (DIFF, "krshna"): {"Your decision": "Our list is right", "Variants": "krushna"}})
    data = add_rows(data, NEW, [{"Word (as in the document)": "āvaraṇa-s", "Your decision": "Add"}], at=4)
    p = sheets.plan(r, sheets.answers_from_report(r, sheets.read_filled("x.xlsx", data)), VLIST)
    assert {c["plain"]: c["result"] for c in p["add"]} == {"dipa": "dīpa", "deepa": "dīpa",
                                                         "krushna": "{{}}{{kṛs͟hṇa}}", "avarana-s": "āvaraṇa-s"}
    for header in ("Also add", "other spellings", "Also_typed_as", "Variants:"):
        assert sheets._head(header) == "also", header


def test_a_copied_row_with_another_word_is_a_word_of_its_own():
    r = review(VTEXT, VLIST)
    wb = workbook(sheets.report_xlsx(r, INFO))
    ws = wb[NEW]
    heads = [c.value for c in ws[2]]
    src = next(n for n in range(3, ws.max_row + 1) if ws.cell(n, heads.index("Plain word") + 1).value == "dipa")
    new = ws.max_row + 1
    for i in range(1, len(heads) + 1):            # copied with its Ref, then the word changed
        ws.cell(new, i).value = ws.cell(src, i).value
    ws.cell(new, 1).value = "dīpa-s"
    ws.cell(new, heads.index("Your decision") + 1).value = "Add"
    ws.cell(src, heads.index("Your decision") + 1).value = "Skip"           # the original row: skipped
    out = io.BytesIO()
    wb.save(out)
    p = sheets.plan(r, sheets.answers_from_report(r, sheets.read_filled("x.xlsx", out.getvalue())), VLIST)
    assert [(c["plain"], c["result"]) for c in p["add"]] == [("dipa-s", "dīpa-s")]


def test_also_typed_as_on_the_pending_list():
    from datetime import datetime
    when = datetime(2026, 11, 20, 5, 0)

    def item(i, plain, sug, also=""):
        return {"id": i, "plain_text": plain, "suggested": sug, "reason": "New word", "source": "", "page": "",
                "context": "", "notes": "", "also": also, "added_at": when}
    items = [item(21, "ganesha", "{{}}{{gaṇesha}}", "ganesh, ganesa"), item(22, "dipa", "dīpa", "deepa"),
             item(23, "seva", "sevā", "sewa"), item(24, "akarta", "akartā"), item(25, "mala", "mālā", "maala")]
    data = sheets.pending_xlsx(items, LIST, {"list": "trial", "made_iso": "2026-11-20T10:00:00", "when": str})
    ws = workbook(data)["Pending"]
    heads = [c.value for c in ws[2]]
    assert heads[heads.index("Result") + 1] == "Also typed as"
    assert {row["Plain word"]: row["Also typed as"] for row in rows_of(ws)}["ganesha"] == "ganesh, ganesa"
    filled = sheets.read_filled("p.xlsx", fill(data, {
        ("Pending", "ganesha"): {"Your decision": "Add"},                             # as pre-filled
        ("Pending", "dipa"): {"Your decision": "Add", "Also typed as": "deepa, deepaa; mala"},  # edited
        ("Pending", "seva"): {"Your decision": "Remove from pending"},
        ("Pending", "akarta"): {"Also typed as": "akartaa"},                          # stays, keeps the new ones
    }))
    open_items = {i["id"]: i for i in items}
    p = sheets.plan_pending(sheets.answers_from_pending(filled), open_items, LIST)
    assert {c["plain"]: (c["result"], c.get("also_for")) for c in p["add"]} == {
        "ganesha": ("{{}}{{gaṇesha}}", None), "ganesh": ("{{}}{{gaṇesha}}", "ganesha"),
        "ganesa": ("{{}}{{gaṇesha}}", "ganesha"), "dipa": ("dīpa", None), "deepa": ("dīpa", "dipa"),
        "deepaa": ("dīpa", "dipa")}
    assert p["also_set"] == {22: "deepa, deepaa, mala", 24: "akartaa"}
    whys = {x["typed"]: x["why"] for x in p["also_left"]}
    assert whys["mala"].startswith("mala has its own row in this file")
    assert whys["sewa"] == "not used, because the word was taken off the pending list."
    assert sheets.sentence(dict(p, pending_off=3)) == ("This will add 6 words (4 of them other ways of typing a "
                                                       "word) and take 3 off the pending list.")

    # a pending list downloaded before the column existed: Add brings what the item carries
    wb = workbook(data)
    wb["Pending"].delete_cols(heads.index("Also typed as") + 1)
    out = io.BytesIO()
    wb.save(out)
    old = sheets.read_filled("p.xlsx", fill(out.getvalue(), {("Pending", "ganesha"): {"Your decision": "Add"}}))
    p = sheets.plan_pending(sheets.answers_from_pending(old), open_items, LIST)
    assert sorted(c["plain"] for c in p["add"]) == ["ganesa", "ganesh", "ganesha"] and p["also_set"] == {}
    # cleared in the file: nothing extra
    p = sheets.plan_pending(sheets.answers_from_pending(sheets.read_filled("p.xlsx", fill(data, {
        ("Pending", "ganesha"): {"Your decision": "Add", "Also typed as": None}}))), open_items, LIST)
    assert [c["plain"] for c in p["add"]] == ["ganesha"]


def test_split_also():
    assert sheets.split_also(" Krishna,krushna ;\nKṛṣṇa, , krishna.", "krsna") == (["krishna", "krushna"], [])
    assert sheets.split_also("kṛs͟hṇa", "krshna") == ([], [])
    assert sheets.split_also("x (y), shri-vidya, o'brien", "") == (["shri-vidya", "o'brien"], ["x (y)"])
    assert sheets.also_text("Krushna;krishna", "") == "krushna, krishna"



# ---------- one row per plain word, several spellings as choices, a correct spelling being enough ----------

CLIST = dict(LIST, trtiya="tṛtīya", deva="deva", arati="āratī", devi="dĕvī", bhatta="bhaṭṭa")   # ĕ, ṭ: letters in use
CTEXT = ("Tṛtīyā (feminine) and Tṛtīya (masculine).\nDĕvā, Dĕva (Koṅkaṇī); Deva (Sanskrit)\n"
         "The Ātma, the Ātmā and the Ātmā again.\nThe Mat̲h̲, the Maṭh and the Mat͟h.\n"
         "Then the Kshamā Yāchanā and the kshamā yāchanā.\nGaṇesha came with the dīpa and the Dīpa for the pūjā.\n")


def test_one_row_per_word_with_every_spelling():
    r = review(CTEXT, CLIST)
    s = r["sections"]
    plains = [i["plain"] for i in s["clash"]] + [i["plain"] for i in s["new"]]
    assert len(plains) == len(set(plains))                                    # never on both, never twice
    assert not {i["plain"] for i in s["already"] + s["style"]} & set(plains)   # nor also on 'Already' / 'Style only'
    assert {"trtiya", "deva", "atma", "math", "kshama yachana"} <= {i["plain"] for i in s["clash"]}
    wb = workbook(sheets.report_xlsx(r, INFO))
    rows = {x["Plain word"]: x for x in rows_of(wb[DIFF])}
    assert len(rows) == len(s["clash"])
    assert rows["trtiya"]["Word (as in the document)"] == "Tṛtīyā, Tṛtīya (same as our list)"
    assert rows["atma"]["Word (as in the document)"] == "Ātma, Ātmā (same as our list)" and rows["atma"]["Times"] == 3
    assert rows["math"]["Word (as in the document)"] == ("Maṭh, Mat͟h (same as our list), Mat̲h̲ (same as our list, "
                                                         "typed differently: underline instead of the tie)")
    assert rows["trtiya"]["Example line"] == "Tṛtīyā (feminine) and Tṛtīya (masculine)."   # the same line once
    assert rows["atma"]["Example line"] == "The Ātma, the Ātmā and the Ātmā again."
    assert rows["kshama yachana"]["Times"] == 2                               # a phrase typed two ways: one row
    assert rows["deva"]["Ref"] == next(i["id"] for i in s["clash"] if i["plain"] == "deva")
    assert {x["Plain word"] for x in rows_of(wb["Already in our list"])} & {"trtiya", "atma", "math"} == set()
    new = {x["Plain word"]: x for x in rows_of(wb[NEW])}
    assert new["dipa"]["Word (as in the document)"] == "Dīpa" and new["dipa"]["Times"] == 2   # dīpa too: one spelling
    assert wb[DIFF]["E3"].alignment.wrap_text
    assert r["grouped"] and harvest.regroup(r) is r                           # made in the new layout already


def test_pages_of_all_spellings_and_merged_items():
    r = harvest.review_document([("issue.txt", [(2, "The Ātma is one."), (7, "The Ātmā is one."),
                                                (9, "The Ātma again.")])], LIST)
    atma = next(i for i in r["sections"]["clash"] if i["plain"] == "atma")
    assert atma["where"] == "2, 7, 9" and atma["times"] == 3
    assert harvest.merge_where(["3, 12", "12, 40", ""]) == "3, 12, 40"
    assert harvest.merge_where(["a.pdf p. 3; b.docx", "a.pdf p. 1, 3", "c 2.pdf p. 4"]) == \
        "a.pdf p. 1, 3; b.docx; c 2.pdf p. 4"
    # a review stored before: two items with one plain word are merged, the Refs of their rows remembered
    old = {"kind": "document", "sections": {
        "clash": [{"id": "c1", "kind": "clash", "plain": "atma", "entry": "ātmā", "options": ["ātma"],
                   "rows": [{"form": "Ātma", "result": "ātma", "times": 1, "where": "2", "line": "Ātma"}],
                   "forms": [{"form": "Ātma", "times": 1, "status": "differs", "gives": "Ātmā"}], "form": "Ātma",
                   "gives": "Ātmā", "times": 1, "where": "2", "line": "Ātma", "note": ""}],
        "new": [{"id": "n1", "kind": "new", "plain": "dipa", "options": ["dīpa"], "form": "dīpa", "times": 1,
                 "rows": [{"form": "dīpa", "result": "dīpa", "times": 1, "where": "3", "line": "the dīpa"}],
                 "where": "3", "line": "the dīpa", "spellings": ["dīpa"]},
                {"id": "n2", "kind": "new", "plain": "dipa", "options": ["dīpā"], "form": "dīpā", "times": 2,
                 "rows": [{"form": "dīpā", "result": "dīpā", "times": 2, "where": "4, 5", "line": "a dīpā"}],
                 "where": "4, 5", "line": "a dīpā", "spellings": ["dīpā"]},
                {"id": "n3", "kind": "new", "plain": "atma", "options": ["atmā"], "form": "atmā", "times": 1,
                 "rows": [{"form": "atmā", "result": "atmā", "times": 1, "where": "6", "line": "atmā"}],
                 "where": "6", "line": "atmā", "spellings": ["atmā"]}],
        "already": [{"id": "a1", "plain": "atma", "form": "Ātmā", "times": 4, "where": "8", "line": "Ātmā",
                     "gives": "Ātmā"}, {"id": "a2", "plain": "puja", "form": "pūjā", "times": 1, "where": "",
                                        "line": "pūjā", "gives": "pūjā"}]}}
    g = harvest.regroup(old)
    assert old["sections"]["already"][0]["id"] == "a1" and len(old["sections"]["new"]) == 3   # stored copy unchanged
    assert g["counts"] == {"clash": 1, "new": 1, "already": 1, "style": 0}
    atma, dipa = g["sections"]["clash"][0], g["sections"]["new"][0]
    assert atma["options"] == ["ātma", "atmā"] and atma["old_refs"] == {"c1:0": 0, "n3:0": 1}
    assert atma["times"] == 6 and atma["where"] == "2, 6, 8" and [x["form"] for x in atma["same"]] == ["Ātmā"]
    assert dipa["id"] == "n1" and dipa["options"] == ["dīpa", "dīpā"] and dipa["old_refs"] == {"n1:0": 0, "n2:0": 1}
    assert dipa["times"] == 3 and dipa["where"] == "3, 4, 5" and dipa["spellings"] == ["dīpa", "dīpā"]
    assert harvest.regroup(g) is g


def test_several_spellings_become_choices():
    assert sheets.spellings("Tṛtīya (masculine), Tṛtīyā (feminine)") == (
        ["tṛtīya (masculine)", "tṛtīyā (feminine)"], None, None)
    assert sheets.spellings("Dĕvā, Dĕva (Koṅkaṇī); Deva (Sanskrit).")[0] == ["dĕvā", "dĕva (koṅkaṇī)",
                                                                             "deva (sanskrit)"]
    assert sheets.spellings("āratī (hindi, marathi), ārati (sanskrit)")[0] == [
        "āratī (hindi, marathi)", "ārati (sanskrit)"]                        # a comma in brackets stays
    assert sheets.spellings("Kṛs̲h̲ṇa")[0] == ["kṛs͟hṇa"] and sheets.spellings("  ") == ([], None, None)
    assert sheets.spellings("{{Dĕva}}{{Dĕvā}}") == ([], "{{dĕva}}{{dĕvā}}", None)     # a code, as it is
    assert "more after the brackets" in sheets.spellings("Tṛtīyā (feminine) and Tṛtīya (masculine)")[2]
    assert sheets.spellings("{{dĕva}")[1] == "{{dĕva}}"                       # the house style repairs brackets
    assert "brackets don't make sense" in sheets.spellings("{{a}}{{b}}{{c}}")[2]
    assert harvest.several(["a", "b"], "capital") == "{{}}{{a, b}}"
    assert harvest.several(["a", "b"], None, "{{}}{{x}}") == "{{}}{{a, b}}"   # blank keeps an always-capital entry
    assert harvest.several(["a", "b"], None, "x") == harvest.several(["a", "b"], "normal", "{{}}{{x}}") == "{{a, b}}"

    r = review(CTEXT, CLIST)
    data = fill(sheets.report_xlsx(r, INFO), {
        (DIFF, "trtiya"): {"Correct spelling": "Tṛtīya (masculine), Tṛtīyā (feminine)"},          # no decision
        (DIFF, "deva"): {"Correct spelling": "Dĕvā, Dĕva (koṅkaṇī), Deva (sanskrit)", "Capital": "Always capital"},
        (DIFF, "atma"): {"Your decision": "Document is right", "Correct spelling": "{{ātma}}{{Ātmā}}"},
        (DIFF, "math"): {"Your decision": "Pending", "Correct spelling": "mat͟h, maṭh"},
        (DIFF, "kshama yachana"): {"Correct spelling": "Kshamā Yāchanā (one), kshamā-yāchanā"},
        (NEW, "ganesha"): {"Your decision": "Add", "Capital": None, "Correct spelling": "gaṇesha; gaṇeśa (old)"},
        (NEW, "dipa"): {"Correct spelling": "Dīpa"},                                                 # no decision
    })
    data = add_rows(data, NEW, [{"Word (as in the document)": "āvaraṇa, āvaraṇā (pl., old)",
                                 "Capital": "Always capital", "Correct spelling": "āvaraṇa, āvaraṇā (pl., old)"}])
    a = sheets.answers_from_report(r, sheets.read_filled("x.xlsx", data))
    assert a["unclear"] == []
    p = sheets.plan(r, a, CLIST, "talk.docx")
    upd = {c["plain"]: c["result"] for c in p["update"]}
    assert upd == {"trtiya": "{{tṛtīya (masculine), tṛtīyā (feminine)}}",
                   "deva": "{{}}{{dĕvā, dĕva (koṅkaṇī), deva (sanskrit)}}",
                   "atma": "{{ātma}}{{ātmā}}",
                   "kshama yachana": "{{kshamā yāchanā (one), kshamā-yāchanā}}"}
    adds = {c["plain"]: c["result"] for c in p["add"]}
    assert adds == {"ganesha": "{{gaṇesha, gaṇeśa (old)}}", "dipa": "dīpa",
                    "avarana": "{{}}{{āvaraṇa, āvaraṇā (pl., old)}}"}          # own row: Correct spelling is enough
    assert {x["plain"]: x["suggested"] for x in p["pending"]} == {"math": "{{mat͟h, maṭh}}",
                                                                  "konkani": "{{}}{{koṅkaṇī}}"}   # blank: pending
    read = {x["word"]: x["read"] for x in a["words"]}
    assert read["Tṛtīyā, Tṛtīya (same as our list)"] == "Document is right"
    assert read["Dĕva, Dĕvā"] == "Document is right · Always capital" and read["Dīpa"] == "Add"
    assert harvest.describe(upd["deva"]) == "choice: Dĕvā or Dĕva (koṅkaṇī) or Deva (sanskrit) (always capital)"
    from diacritical.diacritics import translate_text
    after = dict(CLIST, **upd, **adds)
    assert translate_text("deva Deva trtiya Trtiya", after) == (
        "{{Dĕvā,Dĕva (koṅkaṇī),Deva (sanskrit)}} {{Dĕvā,Dĕva (koṅkaṇī),Deva (sanskrit)}} "
        "{{tṛtīya (masculine),tṛtīyā (feminine)}} {{Tṛtīya (masculine),Tṛtīyā (feminine)}}")
    # a correct spelling with Our list is right / Skip still can't be read
    odd = fill(sheets.report_xlsx(r, INFO), {(NEW, "ganesha"): {"Your decision": "Skip", "Correct spelling": "x, y"}})
    assert [x["plain"] for x in sheets.answers_from_report(r, sheets.read_filled("x.xlsx", odd))["unclear"]] == \
        ["ganesha"]


def test_start_here_and_sheet_notes_explain_it():
    wb = workbook(sheets.report_xlsx(review(CTEXT, CLIST), INFO))
    start = "\n".join(str(c) for row in wb["Start here"].iter_rows(values_only=True) for c in row if c)
    assert sheets.CORRECT_HELP in start and "Each word has one row, with every spelling the document uses" in start
    assert "Tṛtīya (masculine), Tṛtīyā (feminine)" in sheets.CORRECT_HELP
    assert "it counts as “Document is right”" in wb[DIFF]["A1"].value and "it counts as “Add”" in wb[NEW]["A1"].value
    assert wb[DIFF].row_dimensions[1].height > 36                            # the longer note fits
    heads = [c.value for c in wb[DIFF][2]]
    assert wb[DIFF].cell(3, heads.index("Correct spelling") + 1).fill.fgColor.rgb.endswith("FFF2A8")


def test_pending_list_result_with_several_spellings():
    from datetime import datetime
    when = datetime(2026, 11, 20, 5, 0)

    def item(i, plain, sug):
        return {"id": i, "plain_text": plain, "suggested": sug, "reason": "", "source": "", "page": "",
                "context": "", "notes": "", "also": "", "added_at": when}
    items = [item(31, "trtiya", "tṛtīyā"), item(32, "deva", "{{}}{{dĕva}}"), item(33, "arati", "{{ārati, āratī}}"),
             item(34, "namavali", "{{}}{{nāmāvalī (hindi, marathi), nāmāvaliḥ}}")]
    current = dict(CLIST, deva="{{}}{{deva}}")
    data = sheets.pending_xlsx(items, current, {"list": "trial", "made_iso": "2026-11-20T10:00:00", "when": str})
    ws = workbook(data)["Pending"]
    shown = {x["Plain word"]: (x["Result"], x["Capital"]) for x in rows_of(ws)}
    assert shown["arati"] == ("ārati, āratī", None)                          # choices written as they are typed
    assert shown["namavali"] == ("nāmāvalī (hindi, marathi), nāmāvaliḥ", "Always capital")
    assert "several spellings, separated by commas, become choices" in ws["A1"].value
    filled = sheets.read_filled("p.xlsx", fill(data, {
        ("Pending", "trtiya"): {"Your decision": "Add", "Result": "Tṛtīya (masculine), Tṛtīyā (feminine)"},
        ("Pending", "deva"): {"Your decision": "Add", "Result": "Dĕvā, Dĕva (koṅkaṇī)", "Capital": None},
        ("Pending", "arati"): {"Your decision": "Add"},
        ("Pending", "namavali"): {"Your decision": "Add"}}))
    p = sheets.plan_pending(sheets.answers_from_pending(filled), {i["id"]: i for i in items}, current)
    got = {c["plain"]: c["result"] for c in p["add"] + p["update"]}
    assert got == {"trtiya": "{{tṛtīya (masculine), tṛtīyā (feminine)}}",
                   "deva": "{{}}{{dĕvā, dĕva (koṅkaṇī)}}",                   # blank Capital keeps always capital
                   "arati": "{{ārati, āratī}}",
                   "namavali": "{{}}{{nāmāvalī (hindi, marathi), nāmāvaliḥ}}"}
    assert sheets.compose("a, b (x)", "capital", None) == "{{}}{{a, b (x)}}"
    assert sheets.compose("{{}}{{a, b}}", None, None) == "{{}}{{a, b}}"


# ---------- a review made before one row per word: Shantish's review 1 (trimmed) ----------

def review1():
    import json
    return json.loads((Path(__file__).parent / "fixtures" / "words_review1.json").read_text(encoding="utf-8"))


def test_review_made_before_downloads_in_the_new_layout_with_its_decisions():
    fx = review1()
    items, dec = fx["items"], fx["decisions"]
    assert any(":" in ref for ref in dec["cells"])                           # stored with a row per spelling
    info = dict(INFO, review=1)
    wb = workbook(sheets.report_xlsx(items, info, dec["cells"], dec["extra_cells"]))
    rows = {x["Plain word"]: x for x in rows_of(wb[DIFF])}
    assert len(rows) == len(items["sections"]["clash"]) == 7
    b = rows["bankikodla"]
    assert (b["Word (as in the document)"], b["Correct spelling"], b["Capital"], b["Your decision"], b["Ref"]) == (
        "Baṅkīkoḍla, Baṅkīkoḍlā", "Baṅkīkoḍlā", "Always capital", None, "c3")
    assert rows["trtiya"]["Word (as in the document)"] == "Tṛtīyā, Tṛtīya (same as our list)"
    assert rows["arati"]["Word (as in the document)"] == "ārati, āratī (same as our list)"
    assert rows["trtiya"]["Notes for self"].startswith("It's spelt according to Gender.")
    assert rows["deva"]["Notes for self"] == "Result should Show: Dĕvā, Dĕva (Koṅkaṇī); Deva (Sanskrit)"   # once, not twice
    assert rows["santacruz"]["Your decision"] == "Document is right" and rows["yama"]["Your decision"] == \
        "Our list is right"
    already = {x["Word (as in the document)"] for x in rows_of(wb["Already in our list"])}
    assert "Tṛtīya" not in already and "āratī" not in already and "Ahaṅkāra" in already
    assert rows_of(wb[NEW])[-1]["Word (as in the document)"] == "thālī"     # his own row, at the bottom
    assert {x["Plain word"]: x["Also typed as"] for x in rows_of(wb[NEW])}["ishwara"] == "Iswara, Ishwar"

    # uploaded again as it is: Correct spelling with no decision now reads as Document is right
    again = sheets.read_filled("again.xlsx", sheets.report_xlsx(items, info, dec["cells"], dec["extra_cells"]))
    assert "c3" in again["cells"] and not any(":" in ref for ref in again["cells"])
    a = sheets.answers_from_report(items, again)
    assert a["rows"]["c3"]["choice"] == "doc" and a["unclear"] == []
    p = sheets.plan(items, a, fx["entries"], "review 1")
    assert {c["plain"]: c["result"] for c in p["update"]} == {"bankikodla": "{{}}{{baṅkīkoḍlā}}",
                                                              "santacruz": "{{}}{{sāntācrūz}}"}
    assert {x["plain"] for x in p["pending"]} == {"arati", "arati-s", "deva", "trtiya"}
    assert {c["plain"] for c in p["add"]} >= {"abhokta", "thali", "iswara", "ishwar"}
    # the cells as stored (a row per spelling) read the same way
    old = sheets.plan(items, sheets.answers_from_report(items, {"cells": dec["cells"], "extra": dec["extra_cells"]}),
                      fx["entries"], "review 1")
    def same(plan):                    # his own row sits higher up in the trimmed report: its row number differs
        return {k: sorted(str({f: v for f, v in x.items() if f not in ("item", "own_row")}) for x in plan[k])
                for k in ("add", "update", "pending")} | {"resolve": sorted(plan["resolve"])}
    assert same(old) == same(p)


def old_layout_file(fx, answers: dict, info: dict | None = None) -> bytes:
    """A filled report as it was before one row per word: a row per spelling, Refs 'c3:0', 'c3:1'."""
    items = fx["items"]
    wb = workbook(sheets.report_xlsx(items, info or dict(INFO, review=1)))
    ws = wb[DIFF]
    heads = [c.value for c in ws[2]]
    grouped = {i["id"]: i for i in harvest.regroup(items)["sections"]["clash"]}
    values = [list(r) for r in ws.iter_rows(min_row=3, values_only=True)]
    out = []
    for row in values:
        item = grouped[row[heads.index("Ref")]]
        for k, spelt in enumerate(item["rows"]):
            new = list(row)
            new[heads.index("Word (as in the document)")] = spelt["form"]
            new[heads.index("Ref")] = f"{item['id']}:{k}"
            for header, value in answers.get(new[heads.index("Ref")], {}).items():
                new[heads.index(header)] = value
            out.append(new)
    ws.delete_rows(3, ws.max_row)
    for n, row in enumerate(out, 3):
        for i, v in enumerate(row, 1):
            ws.cell(n, i).value = v
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_an_old_layout_filled_file_is_still_read():
    fx = review1()
    items = fx["items"]
    data = old_layout_file(fx, {"c3:1": {"Your decision": "Document is right", "Capital": "Always capital"},
                                "c4:0": {"Your decision": "Document is right"},
                                "c4:1": {"Your decision": "Document is right"},
                                "c6:0": {"Your decision": "Document is right"}})
    filled = sheets.read_filled("old.xlsx", data)
    assert {"c3:0", "c3:1", "c4:1"} <= set(filled["cells"])
    a = sheets.answers_from_report(items, filled)
    p = sheets.plan(items, a, fx["entries"])
    assert {c["plain"]: c["result"] for c in p["update"]} == {"bankikodla": "{{}}{{baṅkīkoḍlā}}",   # its row 2
                                                              "trtiya": "tṛtīyā"}
    assert [(x["plain"], x["why"]) for x in a["unclear"]] == [
        ("deva", "“Document is right” was chosen on more than one spelling")]
    # the same spelling on both rows is no conflict (he filled Correct spelling on both)
    both = old_layout_file(fx, {r: {"Correct spelling": "Baṅkīkoḍlā", "Capital": "Always capital"}
                                for r in ("c3:0", "c3:1")})
    a = sheets.answers_from_report(items, sheets.read_filled("old.xlsx", both))
    assert a["rows"]["c3"]["choice"] == "doc" and a["unclear"] == []
    # shown again in the new layout: the rows merged; a different value goes into Notes
    cells = {"c4:0": {"decision": "Our list is right", "capital": "", "spelling": "", "notes": "ask",
                      "sheet": DIFF, "row": 3},
             "c4:1": {"decision": "Pending", "capital": "Normal", "spelling": "", "notes": "ask", "sheet": DIFF,
                      "row": 4},
             "c3:1": {"decision": "Document is right", "capital": "", "spelling": "", "notes": "", "sheet": DIFF,
                      "row": 5}}
    rows = {x["Plain word"]: x for x in rows_of(workbook(sheets.report_xlsx(items, INFO, cells))[DIFF])}
    assert (rows["deva"]["Your decision"], rows["deva"]["Capital"]) == ("Our list is right", "Normal")
    assert rows["deva"]["Notes for self"] == "ask · the Dĕvā row said Your decision: Pending"
    assert (rows["bankikodla"]["Your decision"], rows["bankikodla"]["Correct spelling"]) == (
        "Document is right", "Baṅkīkoḍlā")                                  # which spelling it was on


def test_notes_column_is_notes_for_self_and_old_notes_still_read():
    from diacritical import sheets as s
    assert s.NOTES == "Notes for self"
    assert s.HEADERS["notes for self"] == "notes" and s.HEADERS["notes"] == "notes"   # files made before the rename
