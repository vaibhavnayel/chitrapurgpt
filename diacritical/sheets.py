"""The words dashboard's Excel round trip (admin_words.py): the report of an upload, the pending list and the whole list
as .xlsx; reading a filled report or pending list back; and what Apply does with it. Pure: no database, no web.

The report: 'Start here', then the sheets that need a decision ('Different from our list', 'New words', 'Written without
marks'; for an offline list 'New in your list', 'Changed', 'Missing from your list'), then read-only sheets ('Style
only', 'Doubtful', 'Already in our list'). Empty sheets are left out. The decision columns have drop-downs (warning
style: other text is allowed; it is read where the meaning is obvious, otherwise as Pending, and listed on the confirm
page). A hidden column 'Ref' ties each row to its item, so sorting or deleting rows is harmless; a hidden sheet
'_dashboard' says what the file is (a report of review N, or the pending list with what each row showed) and for which
list (trial or live).
"""
from __future__ import annotations

import hashlib
import io
import json
import re

from . import harvest
from .cleaning import clean_result

DASHBOARD = "_dashboard"
FONT = 12

# The choices of each kind of row: (key, what the drop-down shows)
CHOICES = {
    "clash": [("list", "Our list is right"), ("doc", "Document is right"), ("pending", "Pending")],
    "new": [("add", "Add"), ("pending", "Pending"), ("skip", "Skip")],
    "unmarked": [("none", "Nothing to change"), ("pending", "Pending")],
    "lnew": [("add", "Add"), ("pending", "Pending"), ("skip", "Skip")],
    "lchanged": [("keep", "Keep our list"), ("use", "Use your list"), ("pending", "Pending")],
    "lmissing": [("keep", "Keep"), ("delete", "Delete")],
    "pending": [("add", "Add"), ("remove", "Remove from pending")],
}
CAPITALS = [("normal", "Normal"), ("capital", "Always capital")]
CAP_LABEL = dict(CAPITALS)
# blank (or a deleted row) means: pending for words that need a decision; the harmless choice for the others
BLANK = {"clash": "pending", "new": "pending", "lnew": "pending", "lchanged": "pending", "unmarked": "none",
         "lmissing": "keep", "pending": "stay"}

_PENDING = ["pending", "keep on pending list", "keep on the pending list", "keep on pending", "keep pending",
            "on pending list", "on the pending list", "to pending", "pending list", "ask the team", "stays pending",
            "stay pending", "put on pending", "send to pending"]
# What people write besides the drop-down's words, read where the meaning is obvious (lowercase, single spaces).
WORDS = {
    "clash": {"list": ["our list is right", "our list is correct", "list is right", "list is correct", "our list",
                       "keep our list", "keep", "ignore", "no change", "nothing to change"],
              "doc": ["document is right", "document is correct", "the document is right", "the document is correct",
                      "magazine is right", "magazine is correct", "the magazine is right", "the magazine is correct",
                      "use the document", "use the magazine", "doc is right"],
              "pending": _PENDING},
    "new": {"add": ["add", "yes", "add it", "add to the list", "add to list"], "pending": _PENDING,
            "skip": ["skip", "no", "ignore", "don't add", "do not add", "not needed"]},
    "unmarked": {"none": ["nothing to change", "nothing", "no change", "ignore", "ok", "our list is right",
                          "our list is correct", "keep"], "pending": _PENDING},
    "lchanged": {"keep": ["keep our list", "keep", "our list is right", "our list is correct", "our list", "ignore"],
                 "use": ["use your list", "use my list", "use the file", "use your file", "use my file", "use the list",
                         "my list is right", "your list is right", "the file is right"],
                 "pending": _PENDING},
    "lmissing": {"keep": ["keep", "keep it"], "delete": ["delete", "delete it", "remove"]},
    "pending": {"add": ["add", "yes", "add it", "add to the list", "add to list", "agreed"],
                "remove": ["remove from pending", "remove", "remove from the pending list", "not needed", "ignore",
                           "skip", "delete", "drop"],
                "stay": _PENDING + ["keep", "wait", "stays", "stay"]},
}
WORDS["lnew"] = WORDS["new"]
CAP_WORDS = {"normal": ["normal", "not always capital", "small", "small letters", "follows typing"],
             "capital": ["always capital", "always capitalised", "always capitalized", "always capitals", "capital",
                         "capitalised", "capitalized", "always"],
             "keep": ["keep", "keep as it is", "as it is", "no change"]}

YELLOW, BLUE, GREY = "FFF2A8", "DDE7F3", "E4E4E4"


class NotOurs(harvest.ReadProblem):
    """An .xlsx that isn't a report or pending list of this dashboard."""


# ---------- reading what people typed ----------

def _norm(text) -> str:
    t = str(text or "").replace("\u2019", "'").replace("\u2018", "'").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", t).strip().strip(".!;:,").strip().lower()


def label(kind: str, key: str) -> str:
    return dict(CHOICES[kind] + CAPITALS).get(key, "")


def read_choice(kind: str, text) -> tuple[str | None, str, str]:
    """(key, the rest of what was typed, how) for a decision cell. key None = blank; how: 'exact' (a drop-down
    choice), 'words' (read from someone's own words), 'unclear' (key None too)."""
    raw = str(text or "").strip()
    t = _norm(raw)
    if not t:
        return None, "", "exact"
    table = WORDS[kind]
    labels = {lab.lower(): key for key, lab in CHOICES[kind]}
    if t in labels:
        return labels[t], "", "exact"
    for key, phrases in table.items():
        if t in phrases:
            return key, "", "words"
    # "Our list is correct. Always capitalised": a choice, then a stop and a remark that names no other choice
    best = None
    for key, phrases in list(table.items()) + [(k, [lab]) for lab, k in labels.items()]:
        for ph in phrases:
            if t.startswith(ph) and len(t) > len(ph) and t[len(ph)] in ".,;:!-\u2013\u2014(" and \
                    (best is None or len(ph) > len(best[1])):
                best = (key, ph)
    if best:
        rest = t[len(best[1]):].lstrip(" .,;:!-\u2013\u2014")
        others = [ph for k, phs in table.items() if k != best[0] for ph in phs
                  if len(ph) > 3 and re.search(rf"\b{re.escape(ph)}\b", rest)]
        if not others:
            cut = re.match(r"\s*" + r"\s*".join(re.escape(w) for w in best[1].split()) + r"[\s.,;:!\-\u2013\u2014]*",
                           raw, re.I)
            return best[0], raw[cut.end():].strip() if cut else rest, "words"
    return None, raw, "unclear"


def read_capital(text) -> str | None:
    """'normal', 'capital', 'keep', None (blank) or '?' (not readable)."""
    t = _norm(text)
    if not t:
        return None
    for key, words in CAP_WORDS.items():
        if t in words:
            return key
    return "?"


def capital_in_words(text) -> str | None:
    """'capital' or 'normal' when a remark says so plainly ("Always capitalised", "Need not be capitalised")."""
    t = _norm(text)
    if "capital" not in t:
        return None
    if re.search(r"\b(not|never|no need|need not|don't|doesn't|isn't|shouldn't)\b|n't\b", t):
        return "normal"
    if re.search(r"\balways\b(\s+\w+){0,2}\s+capital|\bcapital\w*\s+always\b", t):
        return "capital"
    return None


def spelling(text) -> tuple[str, str | None]:
    """(the spelling in the house style, or '', why it can't be used or None)."""
    raw = str(text or "").strip()
    if not raw:
        return "", None
    s = clean_result(harvest.house(raw))
    why = harvest.code_problem(s)
    return s, (f"the spelling “{raw}” can't be used: {why}" if why else None)


# ---------- workbooks: writing ----------

def _font(**kw):
    from openpyxl.styles import Font
    return Font(name="Calibri", size=kw.pop("size", FONT), **kw)


def _save(wb) -> bytes:
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def _dashboard_sheet(wb, pairs: list[tuple], table: list[list] | None = None):
    ws = wb.create_sheet(DASHBOARD)
    ws.append(["This sheet tells the dashboard what this file is. Please don't change it."])
    for p in pairs:
        ws.append(list(p))
    if table:
        ws.append([])
        for row in table:
            ws.append(row)
    ws.sheet_state = "hidden"
    return ws


def _table_sheet(wb, title: str, note: str, columns: list[tuple], rows: list[list], *, decide: dict | None = None,
                 info: bool = False):
    """A sheet: a note in row 1, headers in row 2 (frozen, with a filter), the rows. columns: [(header, width)];
    a header 'Ref' is hidden. decide: {header: [drop-down labels]} for yellow decision columns."""
    from openpyxl.styles import Alignment, Border, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation
    decide = decide or {}
    line = Side(style="thin", color="C8CDD3")
    box = Border(left=line, right=line, top=line, bottom=line)
    centred = {"Times", "Pages", "Page", "Row", "Rows"}
    ws = wb.create_sheet(title[:31])
    ws.append([note])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(1, len(columns) - 1))
    ws["A1"].font = _font(italic=True)
    ws["A1"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[1].height = 36
    ws.append([h for h, _ in columns])
    head_fill = PatternFill("solid", fgColor=GREY if info else BLUE)
    yellow = PatternFill("solid", fgColor=YELLOW)
    for i, (h, _) in enumerate(columns, 1):
        c = ws.cell(2, i)
        c.font = _font(bold=True)
        c.fill = yellow if h in decide else head_fill
        c.border = box
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center" if h in centred else None)
    for r in rows:
        ws.append(["" if v is None else v for v in r])
    last = ws.max_row
    for i, (h, w) in enumerate(columns, 1):
        letter = ws.cell(2, i).column_letter
        ws.column_dimensions[letter].width = w
        if h == "Ref":
            ws.column_dimensions[letter].hidden = True
        for n in range(3, last + 1):
            c = ws.cell(n, i)
            c.font = _font()
            c.border = box
            c.alignment = Alignment(wrap_text=True, vertical="top", horizontal="center" if h in centred else None)
            if h in decide:
                c.fill = yellow
        if h in decide and decide[h] and last >= 3:
            dv = DataValidation(type="list", formula1='"' + ",".join(decide[h]) + '"', allow_blank=True,
                                showErrorMessage=True, errorStyle="warning", errorTitle="Not one of the choices",
                                error="Please choose from the list. Other words are read only where the meaning is "
                                      "obvious; otherwise the row is treated as Pending (you'll see it listed before "
                                      "anything changes).")
            dv.add(f"{letter}3:{letter}{last}")
            ws.add_data_validation(dv)
    ws.freeze_panes = "A3"
    if last >= 3:
        ws.auto_filter.ref = f"A2:{ws.cell(2, len(columns)).column_letter}{last}"
    _print_wide(ws)
    ws.print_title_rows = "2:2"
    return ws


def _print_wide(ws) -> None:
    """Printed landscape, as wide as the page."""
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True


def _gives(item: dict) -> str:
    """What our list gives for a word, for people: the translator's output, with the entry's kind when it isn't
    plain (always capital, name, choices)."""
    entry = item.get("entry")
    if not entry:
        return f"{item.get('gives', '')} (made from separate words)" if item.get("gives") else "(not in our list)"
    p = harvest.parse_code(entry)
    return item.get("gives") or entry if p and p["type"] == "normal" else harvest.describe(entry)


DOC_SECTIONS = [  # (section, sheet, what it is, needs a decision)
    ("clash", "Different from our list", "In our list, but the document spells it differently", True),
    ("new", "New words", "Words with marks that are not in our list", True),
    ("unmarked", "Written without marks", "Plain in the document, but our list would add marks. Nothing changes "
                                          "unless you choose Pending", True),
    ("style", "Style only", "The same word typed differently (underline, look-alike letters): nothing to change",
     False),
    ("doubtful", "Doubtful", "Look-alike letters and PDF quirks: not compared, nothing to change", False),
    ("already", "Already in our list", "Exactly as our list gives them: nothing to change", False),
]
LIST_SECTIONS = [
    ("lnew", "New in your list", "In your list, not in ours", True),
    ("lchanged", "Changed", "In both lists, with a different result", True),
    ("lmissing", "Missing from your list", "In our list, not in yours: kept unless you choose Delete", True),
    ("problems", "Problems", "Rows of your list that couldn't be used", False),
    ("empty", "Left empty", "Rows with a plain word but no result (for example not answered yet): left out", False),
    ("unchanged", "Same as our list", "The same in both lists: nothing to change", False),
]


def refs(item: dict) -> list[str]:
    """The Ref of each row of an item: one per spelling for clashes and new words."""
    if item.get("kind") in ("clash", "new"):
        return [f"{item['id']}:{k}" for k in range(len(item.get("rows") or [item]))]
    return [item["id"]]


def report_xlsx(review: dict, info: dict, cells: dict | None = None) -> bytes:
    """The report of a review, to fill in. info: review (id), list ('trial'|'live'), list_label, files, made, by,
    status. cells: what an uploaded copy said, by Ref ({decision, capital, spelling, notes}), to show it again."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment
    cells = cells or {}
    s = review["sections"]
    doc = review["kind"] == "document"
    sections = DOC_SECTIONS if doc else LIST_SECTIONS
    wb = Workbook()
    start = wb.active
    start.title = "Start here"

    def c(ref, key, default=""):
        return (cells.get(ref) or {}).get(key, default) if ref in cells else default

    if doc:
        decision_sheets = [sheet for key, sheet, _, needs in sections if needs and s.get(key) and key != "unmarked"]
    else:
        decision_sheets = [sheet for key, sheet, _, needs in sections if needs and s.get(key)]
    files = ", ".join(info.get("files") or [])
    steps = []
    if decision_sheets:
        steps.append("Fill in the yellow column “Your decision” on " + " and ".join(f"“{x}”" for x in decision_sheets)
                     + ". Anything left blank goes to the pending list.")
    if doc:
        steps += ["“Capital”: choose Always capital for names and places; blank keeps the entry as it is (for a new "
                  "word: Normal). “Correct spelling” (optional) is used instead of the document's spelling.",
                  "“Notes” go with the word to the pending list, for the Diacriticals team."]
        if s.get("unmarked"):
            steps.append("“Written without marks” already says “Nothing to change”; choose Pending for a word the "
                         "team should look at.")
    else:
        steps.append("“Notes” go with the word to the pending list, for the Diacriticals team.")
    steps.append("Save this file and upload it on the dashboard under “Upload your decisions”. You'll see a summary "
                 "and an Apply button before anything in the list changes.")
    made = (f"Made {info.get('made', '')} by {info.get('by', '')}, compared with "
            f"{info.get('list_label', 'the list')} as it was then.")
    lines = [(f"Word check: {files}", _font(bold=True, size=16)), (made, None),
             (f"Status: {info.get('status', '')}", _font(bold=True)), ("", None),
             ("What to do", _font(bold=True, size=14))]
    lines += [(f"{n}. {text}", None) for n, text in enumerate(steps, 1)]
    lines += [("", None), ("The sheets", _font(bold=True, size=14))]
    for text, font in lines:
        start.append([text])
        start.cell(start.max_row, 1).font = font or _font()
    start.append(["Sheet", "Rows", "What it is"])
    for i in range(1, 4):
        start.cell(start.max_row, i).font = _font(bold=True)
    for key, sheet, what, _ in sections:
        start.append([sheet if s.get(key) else f"{sheet} (no sheet)", len(s.get(key, [])), what])
        for i in range(1, 4):
            start.cell(start.max_row, i).font = _font()
    start.column_dimensions["A"].width = 30
    start.column_dimensions["B"].width = 8
    start.column_dimensions["C"].width = 100
    for row in start.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=cell.column == 3, vertical="top",
                                       horizontal="center" if cell.column == 2 else None)
    _print_wide(start)

    pend_note = "Notes go with the word to the pending list."
    if doc:
        if s.get("clash"):
            rows = []
            for item in s["clash"]:
                for ref, r in zip(refs(item), item.get("rows") or [item]):
                    rows.append([r["form"], item["plain"], r["times"], r["where"], r["line"], _gives(item),
                                 c(ref, "decision"), c(ref, "capital"), c(ref, "spelling"), c(ref, "notes"), ref])
            _table_sheet(wb, "Different from our list",
                         "In our list, but the document spells it differently. “Your decision”: blank = Pending. "
                         "“Capital” blank = keep as it is. A word spelt two ways has one row per spelling: choose "
                         "“Document is right” on the right one only. " + pend_note,
                         [("Word (as in the document)", 22), ("Plain word", 16), ("Times", 7), ("Pages", 10),
                          ("Example line", 48), ("Our list gives", 24), ("Your decision", 20), ("Capital", 15),
                          ("Correct spelling", 18), ("Notes", 30), ("Ref", 8)], rows,
                         decide={"Your decision": [lab for _, lab in CHOICES["clash"]],
                                 "Capital": [lab for _, lab in CAPITALS]})
        if s.get("new"):
            rows = []
            for item in s["new"]:
                cap = CAP_LABEL["capital"] if item.get("seen") == "name" else ""
                for ref, r in zip(refs(item), item.get("rows") or [item]):
                    rows.append([r["form"], item["plain"], r["times"], r["where"], r["line"], c(ref, "decision"),
                                 c(ref, "capital", cap), c(ref, "spelling"), c(ref, "notes"), ref])
            _table_sheet(wb, "New words",
                         "Words with marks that are not in our list. “Your decision”: blank = Pending. “Capital”: "
                         "blank = Normal (Always capital is filled in for words only seen with a capital, also "
                         "mid-sentence). " + pend_note,
                         [("Word (as in the document)", 22), ("Plain word", 16), ("Times", 7), ("Pages", 10),
                          ("Example line", 52), ("Your decision", 16), ("Capital", 15), ("Correct spelling", 18),
                          ("Notes", 30), ("Ref", 8)], rows,
                         decide={"Your decision": [lab for _, lab in CHOICES["new"]],
                                 "Capital": [lab for _, lab in CAPITALS]})
        if s.get("unmarked"):
            rows = [[i["form"], i["plain"], i["times"], i["where"], i["line"], _gives(i),
                     c(i["id"], "decision", label("unmarked", "none")), c(i["id"], "notes"), i["id"]]
                    for i in s["unmarked"]]
            _table_sheet(wb, "Written without marks",
                         "Written without marks in the document (often addresses and registered names), but our list "
                         "would add marks. Nothing changes in the list for these; choose Pending for a word the team "
                         "should look at. " + pend_note,
                         [("Word (as in the document)", 22), ("Plain word", 16), ("Times", 7), ("Pages", 10),
                          ("Example line", 52), ("Our list gives", 20), ("Your decision", 20), ("Notes", 30),
                          ("Ref", 8)], rows,
                         decide={"Your decision": [lab for _, lab in CHOICES["unmarked"]]})
        nothing = label("unmarked", "none")
        if s.get("style"):
            _table_sheet(wb, "Style only", "The same word, typed differently in the document (for information).",
                         [("Word (as in the document)", 22), ("Plain word", 16), ("Times", 7), ("Pages", 10),
                          ("Example line", 52), ("Our list gives", 20), ("Difference", 26), ("Decision", 18)],
                         [[i["form"], i["plain"], i["times"], i["where"], i["line"], i["gives"], i["why"], nothing]
                          for i in s["style"]], info=True)
        if s.get("doubtful"):
            _table_sheet(wb, "Doubtful", "Look-alike letters, or lines that came out garbled from the PDF (not "
                         "compared; for information).",
                         [("Word or page", 22), ("Times", 7), ("Pages", 10), ("Line", 56), ("What it is", 46),
                          ("Decision", 18)],
                         [[i["form"], i["times"], i["where"], i["line"], i["why"], nothing] for i in s["doubtful"]],
                         info=True)
        if s.get("already"):
            _table_sheet(wb, "Already in our list", "Exactly as our list gives them (for information).",
                         [("Word (as in the document)", 22), ("Plain word", 16), ("Times", 7), ("Pages", 10),
                          ("Example line", 56), ("Our list gives", 20), ("Decision", 18)],
                         [[i["form"], i["plain"], i["times"], i["where"], i["line"], i["gives"], nothing]
                          for i in s["already"]], info=True)
    else:
        if s.get("lnew"):
            _table_sheet(wb, "New in your list", "In your list, not in ours. “Add” is filled in; change it to Pending "
                         "or Skip where needed. " + pend_note,
                         [("Plain word", 18), ("Result in your list", 26), ("Shown as", 30), ("Row", 7),
                          ("Your decision", 16), ("Notes", 30), ("Ref", 8)],
                         [[i["plain"], i["result"], harvest.describe(i["result"]), i["row"],
                           c(i["id"], "decision", label("lnew", "add")), c(i["id"], "notes"), i["id"]]
                          for i in s["lnew"]],
                         decide={"Your decision": [lab for _, lab in CHOICES["lnew"]]})
        if s.get("lchanged"):
            _table_sheet(wb, "Changed", "In both lists, with a different result. “Your decision”: blank = Pending. "
                         + pend_note,
                         [("Plain word", 18), ("Our list gives", 30), ("Your list gives", 30), ("Row", 7),
                          ("Your decision", 18), ("Notes", 30), ("Ref", 8)],
                         [[i["plain"], harvest.describe(i["entry"]), harvest.describe(i["result"]), i["row"],
                           c(i["id"], "decision"), c(i["id"], "notes"), i["id"]] for i in s["lchanged"]],
                         decide={"Your decision": [lab for _, lab in CHOICES["lchanged"]]})
        if s.get("lmissing"):
            _table_sheet(wb, "Missing from your list", "In our list, not in yours: kept unless you choose Delete.",
                         [("Plain word", 18), ("Our list gives", 30), ("Your decision", 16), ("Ref", 8)],
                         [[i["plain"], harvest.describe(i["entry"]), c(i["id"], "decision", label("lmissing", "keep")),
                           i["id"]] for i in s["lmissing"]],
                         decide={"Your decision": [lab for _, lab in CHOICES["lmissing"]]})
        if s.get("problems"):
            _table_sheet(wb, "Problems", "Rows of your list that couldn't be used (for information).",
                         [("Row", 7), ("Plain word", 20), ("Result", 26), ("Why", 50)],
                         [[i["row"], i["plain"], i["result"], i["why"]] for i in s["problems"]], info=True)
        if s.get("empty"):
            _table_sheet(wb, "Left empty", "Rows with a plain word but no result: left out (for information).",
                         [("Row", 7), ("Plain word", 24)], [[i["row"], i["plain"]] for i in s["empty"]], info=True)
        if s.get("unchanged"):
            _table_sheet(wb, "Same as our list", "The same in both lists (for information).",
                         [("Plain word", 20), ("Result", 30)], [[i["plain"], i["result"]] for i in s["unchanged"]],
                         info=True)
    _dashboard_sheet(wb, [("kind", "report"), ("review", info["review"]), ("list", info["list"]),
                          ("made", info.get("made_iso", "")), ("version", 1)])
    return _save(wb)


def pending_xlsx(items: list[dict], current: dict[str, str], info: dict) -> bytes:
    """The open pending items, to fill in: 'Your decision' (Add / Remove from pending; blank = stays pending), 'Result'
    (the suggestion, to overwrite with the agreed spelling), 'Capital', 'Notes'. info: list, made, made_iso, by,
    when(timestamp)->str. The hidden sheet keeps what each row showed (the entry then), for the check on upload."""
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    rows, snap = [], []
    when = info.get("when") or str
    for i in items:
        entry = current.get(i["plain_text"])
        sug = i["suggested"] or ""
        p = harvest.parse_code(sug) if sug else None
        result, cap = sug, ""
        if p and p["type"] == "normal":
            result = sug
        elif p and p["type"] == "capital" and "," not in p["main"]:
            result, cap = p["main"].strip(), CAP_LABEL["capital"]
        rows.append([i["plain_text"], harvest.describe(sug) if sug else "", harvest.describe(entry) if entry else
                     "(not in our list)", i["reason"], i["source"], i["page"], i["context"], when(i["added_at"]),
                     "", result, cap, i.get("notes") or "", str(i["id"])])
        snap.append([str(i["id"]), i["plain_text"], entry or "", sug])
    _table_sheet(wb, "Pending", "Words waiting for an answer. “Your decision”: Add puts “Result” in the list (change "
                 "it to the agreed spelling if needed; house style is applied), Remove from pending takes the word off "
                 "the list, blank = stays pending. “Capital” blank = Normal for a new word, or keeps the entry's form "
                 "for a change. Then upload this file on the dashboard under “Upload your decisions”.",
                 [("Plain word", 16), ("Suggested", 24), ("Our list now gives", 22), ("Why", 36), ("Source", 22),
                  ("Page", 8), ("Example line", 44), ("Added", 16), ("Your decision", 18), ("Result", 22),
                  ("Capital", 15), ("Notes", 30), ("Ref", 8)], rows,
                 decide={"Your decision": [lab for _, lab in CHOICES["pending"]], "Result": None,
                         "Capital": [lab for _, lab in CAPITALS]})
    _dashboard_sheet(wb, [("kind", "pending"), ("list", info["list"]), ("made", info.get("made_iso", "")),
                          ("version", 1)], [["ref", "plain", "entry", "suggested"]] + snap)
    return _save(wb)


def list_xlsx(entries: list[dict]) -> bytes:
    """The whole list (plain_text, diacritic_text), in the layout an offline list upload reads back."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Word list"
    ws.append(["plain_text", "diacritic_text", "Shown as"])
    for c in ws[1]:
        c.font = _font(bold=True)
    for e in sorted(entries, key=lambda e: e["plain_text"]):
        ws.append([e["plain_text"], e["diacritic_text"], harvest.describe(e["diacritic_text"])])
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 40
    ws.column_dimensions["C"].width = 40
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:C{ws.max_row}"
    return _save(wb)


# ---------- workbooks: reading back ----------

HEADERS = {"your decision": "decision", "capital": "capital", "correct spelling": "spelling", "result": "result",
           "notes": "notes", "ref": "ref", "word (as in the document)": "word", "plain word": "plain"}


def is_dashboard_file(data: bytes) -> bool:
    try:
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(data), read_only=True)
        found = DASHBOARD in wb.sheetnames
        wb.close()
        return found
    except Exception:  # noqa: BLE001
        return False


def read_filled(name: str, data: bytes) -> dict:
    """A filled report or pending list: {'kind': 'report'|'pending', 'list', 'made', 'review' (report),
    'snapshot' (pending: {ref: {plain, entry, suggested}}), 'cells': {ref: {sheet, row, word, decision, capital,
    spelling, result, notes}}}. Raises NotOurs / ReadProblem with a message for people."""
    from openpyxl import load_workbook
    low = (name or "").lower()
    if low.endswith((".xls", ".csv", ".ods", ".numbers")):
        raise harvest.ReadProblem("Please save the file as an Excel workbook (.xlsx) and upload that.")
    if not low.endswith((".xlsx", ".xlsm")):
        raise NotOurs("Please upload the filled report or pending list: an Excel file (.xlsx) downloaded from this "
                      "page.")
    try:
        wb = load_workbook(io.BytesIO(data), data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise harvest.ReadProblem("This file couldn't be opened as an Excel workbook (.xlsx).") from exc
    if DASHBOARD not in wb.sheetnames:
        raise NotOurs("This Excel file isn't a report or pending list from this page, so there are no decisions in "
                      "it to read. To compare a word list of your own with ours, upload it under “Upload a "
                      "document”.")
    meta, snapshot, table = {}, {}, False
    for row in wb[DASHBOARD].iter_rows(min_row=2, values_only=True):
        row = list(row) + [None] * 4
        if row[0] == "ref":
            table = True
            continue
        if table and row[0] is not None:
            snapshot[str(row[0]).strip()] = {"plain": str(row[1] or ""), "entry": str(row[2] or ""),
                                             "suggested": str(row[3] or "")}
        elif not table and row[0]:
            meta[str(row[0]).strip()] = row[1]
    if meta.get("kind") not in ("report", "pending") or meta.get("list") not in ("trial", "live"):
        raise NotOurs("This file's hidden “_dashboard” sheet was changed, so it can't be matched to an upload. "
                      "Please download the report again.")
    cells = {}
    for ws in wb.worksheets:
        if ws.title == DASHBOARD:
            continue
        rows = list(ws.iter_rows(values_only=True))
        head = next((n for n, r in enumerate(rows[:6]) if any(_norm(v) == "ref" for v in r)), None)
        if head is None:
            continue
        cols = {HEADERS[_norm(v)]: i for i, v in enumerate(rows[head]) if _norm(v) in HEADERS}
        for n, r in enumerate(rows[head + 1:], head + 2):
            def get(k):
                i = cols.get(k)
                v = r[i] if i is not None and i < len(r) else None
                return "" if v is None else str(v).strip()
            ref = get("ref")
            if ref and ref not in cells:
                cells[ref] = {"sheet": ws.title, "row": n, "word": get("word") or get("plain"),
                              **{k: get(k) for k in ("decision", "capital", "spelling", "result", "notes")}}
    review = meta.get("review")
    return {"kind": meta["kind"], "list": meta["list"], "made": str(meta.get("made") or ""),
            "review": int(review) if str(review or "").isdigit() else None, "snapshot": snapshot, "cells": cells}


# ---------- what the decisions mean ----------

def _items(review: dict):
    for key, items in review["sections"].items():
        for item in items:
            if item.get("kind") in CHOICES:
                yield item


def answers_from_report(review: dict, filled: dict) -> dict:
    """Each item's decision from a filled report: {'rows': {item id: {choice, option, cap, spelling, note}},
    'unclear': [...], 'words': [...] (read from someone's own words), 'missing': [words whose rows weren't in the
    file], 'cells': the cells (to show again in a re-downloaded report)}."""
    cells = filled["cells"]
    out = {"rows": {}, "unclear": [], "words": [], "missing": [], "cells": cells}
    for item in _items(review):
        kind = item["kind"]
        per_row, notes, problems = [], [], []
        item_refs = refs(item)
        for k, ref in enumerate(item_refs):
            cell = cells.get(ref)
            if cell is None:
                continue
            where = {"sheet": cell["sheet"], "row": cell["row"], "word": cell.get("word") or item["plain"]}
            key, rest, how = read_choice(kind, cell.get("decision"))
            cap = read_capital(cell.get("capital")) if kind in ("clash", "new") else None
            spell, bad = spelling(cell.get("spelling")) if kind in ("clash", "new", "unmarked") else ("", None)
            note = " ".join(x for x in (rest if how == "words" else "", cell.get("notes", "")) if x).strip()
            if note and note not in notes:
                notes.append(note)
            if how == "unclear":
                problems.append(where | {"typed": cell.get("decision"), "why": "not one of the choices"})
                continue
            if cap == "?":
                problems.append(where | {"typed": cell.get("capital"), "why": "the Capital column isn't Normal or "
                                                                                "Always capital"})
                continue
            if bad:
                problems.append(where | {"typed": cell.get("spelling"), "why": bad})
                continue
            if cap is None and kind in ("clash", "new"):
                inferred = capital_in_words(rest) or capital_in_words(cell.get("notes"))
                if inferred:
                    cap = inferred
                    how = "words"
            if spell and key in ("list", "skip", "none"):
                problems.append(where | {"typed": f"{cell.get('decision')} + {cell.get('spelling')}",
                                         "why": f"a correct spelling was given, but the decision is "
                                                f"“{label(kind, key)}”"})
                continue
            if how == "words":
                out["words"].append(where | {"typed": " · ".join(x for x in (cell.get("decision"),
                                                                             cell.get("notes")) if x),
                                             "read": label(kind, key or BLANK[kind]) + (
                                                 f" · {CAP_LABEL[cap]}" if cap in CAP_LABEL else "")})
            per_row.append({"option": k, "choice": key, "cap": cap, "spelling": spell, "where": where})
        if BLANK[kind] == "pending" and not any(r in cells for r in item_refs):
            out["missing"].append(item["plain"])     # its rows were deleted from the file: it goes to pending
        d = {"choice": BLANK[kind], "option": 0, "cap": None, "spelling": "", "note": " · ".join(notes)}
        chosen = [r for r in per_row if r["choice"] is not None]
        if problems:
            out["unclear"] += [p | {"plain": item["plain"]} for p in problems]
            d["choice"] = "pending" if BLANK[kind] == "pending" or kind == "unmarked" else BLANK[kind]
            d["unclear"] = True
        elif chosen:
            picked = {r["choice"] for r in chosen}
            if len(picked) == 1 and not (picked & {"doc", "add"} and len(chosen) > 1):
                r = chosen[0]
                d.update(choice=r["choice"], option=r["option"], cap=r["cap"], spelling=r["spelling"])
            elif "pending" in picked:
                d["choice"] = "pending"
            else:
                d["choice"] = "pending" if BLANK[kind] != "keep" else "keep"
                d["unclear"] = True
                why = ("“Document is right” was chosen on more than one spelling" if picked == {"doc"} else
                       "“Add” was chosen on more than one spelling" if picked == {"add"} else
                       "the rows of this word say different things")
                out["unclear"].append({"sheet": chosen[0]["where"]["sheet"], "row": chosen[0]["where"]["row"],
                                       "word": item["plain"], "plain": item["plain"],
                                       "typed": ", ".join(label(kind, r["choice"]) for r in chosen), "why": why})
        else:
            blank = [r for r in per_row if r["spelling"] or r["cap"]]
            if blank:     # no decision, but a spelling or capital: they go with the word to pending
                r = blank[0]
                d.update(option=r["option"], cap=r["cap"], spelling=r["spelling"])
        out["rows"][item["id"]] = d
    return out


def compose(result_text: str, cap: str | None, entry: str | None) -> str:
    """The code to store for a spelling from the pending list: Always capital / Normal as chosen; blank keeps the
    entry's form (a new word: Normal). A typed code ({{ }}) is used as it is."""
    r = clean_result(harvest.house(result_text or ""))
    if "{" in r:
        return harvest.with_type(r, cap or "keep")
    if cap == "capital":
        return clean_result("{{}}{{" + r + "}}")
    if cap == "normal" or not entry:
        return r
    return harvest.replace_result(entry, r, False)


def _empty_plan() -> dict:
    return {"add": [], "update": [], "delete": [], "pending": [], "resolve": [], "remove": [], "notes": {},
            "skipped": [], "unclear": [], "words": [], "missing": [], "stay": 0}


def _guard(out: dict) -> None:
    """A word that two rows want to change differently is left alone (listed as unclear)."""
    targets = {}
    for op in ("add", "update", "delete"):
        for c in out[op]:
            targets.setdefault(c["plain"], set()).add((op, c.get("result")))
    bad = {p for p, t in targets.items() if len(t) > 1}
    for op in ("add", "update", "delete"):
        seen, kept = set(), []
        for c in out[op]:
            if c["plain"] in bad or c["plain"] in seen:
                continue
            seen.add(c["plain"])
            kept.append(c)
        out[op] = kept
    for p in sorted(bad):
        out["unclear"].append({"sheet": "", "row": "", "word": p, "plain": p, "typed": "",
                               "why": "two rows want to change this word differently: nothing is changed"})


def plan(review: dict, answers: dict, current: dict[str, str], source: str = "") -> dict:
    """What Apply does with a report's answers, against the list as it is now: {'add': [{plain, result, item}],
    'update': [{plain, old, result, item}], 'delete': [{plain, old, item}], 'pending': [{plain, suggested, reason,
    source, page, context, notes, item}], 'resolve': [plain words confirmed as they are], 'skipped': [{plain, then,
    now, wanted}] (changed in the list since the report was made), 'unclear', 'words', 'missing'}."""
    out = _empty_plan()
    out["unclear"] = list(answers.get("unclear", []))
    out["words"] = list(answers.get("words", []))
    out["missing"] = list(answers.get("missing", []))
    rows = answers.get("rows", {})

    def change(item, then, new):
        plain = item["plain"]
        now = current.get(plain)
        if new == then:
            return
        if now != then:
            out["skipped"].append({"plain": plain, "then": then, "now": now, "wanted": new})
        elif then is None:
            out["add"].append({"plain": plain, "result": new, "item": item["id"]})
        elif new is None:
            out["delete"].append({"plain": plain, "old": then, "item": item["id"]})
        else:
            out["update"].append({"plain": plain, "old": then, "result": new, "item": item["id"]})

    def pend(item, d, suggested, reason, row=None):
        row = row or item
        out["pending"].append({"plain": item["plain"], "suggested": suggested, "reason": reason, "source": source,
                               "page": row.get("where", ""), "context": row.get("line", ""),
                               "notes": d.get("note", ""), "item": item["id"]})

    for item in _items(review):
        kind = item["kind"]
        d = rows.get(item["id"]) or {"choice": BLANK[kind], "option": 0, "cap": None, "spelling": "", "note": ""}
        choice, k, cap, spell = d["choice"], d.get("option") or 0, d.get("cap"), d.get("spelling") or ""
        entry = item.get("entry")
        options = item.get("options") or []
        k = max(0, min(k, len(options) - 1))
        row = (item.get("rows") or [None] * len(options))[k] if kind in ("clash", "new") else None
        if kind == "clash":
            capitalised = (row or item)["form"][:1].isupper()
            result = spell or options[k]
            if choice == "pending":
                suggested = harvest.with_type(harvest.replace_result(entry, result, capitalised), cap or "keep") \
                    if entry else harvest.with_type(result, cap or "keep")
                differs = ", ".join(f["form"] for f in item["forms"] if f["status"] == "differs")
                pend(item, d, suggested, f"Different from our list: the document has {differs}; our list gives "
                                         f"{item['gives']}", row)
            elif choice == "list":
                if entry and cap in ("normal", "capital"):
                    change(item, entry, harvest.with_type(entry, cap))
                out["resolve"].append(item["plain"])
            elif choice == "doc":
                if entry is None:
                    change(item, None, harvest.with_type(result, cap if cap in ("normal", "capital") else "normal"))
                else:
                    change(item, entry, harvest.with_type(harvest.replace_result(entry, result, capitalised),
                                                          cap or "keep"))
        elif kind == "new":
            result = spell or options[k]
            code = harvest.with_type(result, cap if cap in ("normal", "capital") else "normal")
            if choice == "add":
                change(item, None, code)
            elif choice == "pending":
                spelt = item.get("spellings") or []
                pend(item, d, code, "New word: not in our list" + (
                    f" (the document spells it {' and '.join(spelt)})" if len(options) > 1 else ""), row)
        elif kind == "unmarked":
            if choice == "pending":
                pend(item, d, spell or entry or "", f"Written without marks in the document ({item['form']}); "
                                                    f"our list gives {item['gives']}")
        elif kind == "lnew":
            if choice == "add":
                change(item, None, item["result"])
            elif choice == "pending":
                pend(item, d, item["result"], "New in an uploaded list", {"where": f"row {item['row']}", "line": ""})
        elif kind == "lchanged":
            if choice == "use":
                change(item, entry, item["result"])
            elif choice == "keep":
                out["resolve"].append(item["plain"])
            elif choice == "pending":
                pend(item, d, item["result"], f"Changed in an uploaded list; our list has {harvest.describe(entry)}",
                     {"where": f"row {item['row']}", "line": ""})
        elif kind == "lmissing":
            if choice == "delete":
                change(item, entry, None)
    out["resolve"] += [u["plain"] for u in review["sections"].get("unchanged", [])]
    _guard(out)
    return out


def answers_from_pending(filled: dict) -> dict:
    """The rows of a filled pending list, as stored until Apply: {'snapshot', 'cells'}."""
    return {"snapshot": filled["snapshot"], "cells": filled["cells"], "made": filled.get("made", "")}


def plan_pending(answers: dict, open_items: dict[int, dict], current: dict[str, str]) -> dict:
    """What Apply does with a filled pending list, against the list and the pending list as they are now. Like plan(),
    plus 'remove' (pending ids taken off), 'done' (pending ids answered by a change), 'notes' ({id: new notes})."""
    out = _empty_plan()
    out["done"] = []
    snapshot, cells = answers["snapshot"], answers["cells"]
    wanted = {}
    for ref, cell in cells.items():
        snap = snapshot.get(ref)
        if snap is None or not ref.isdigit():
            continue
        item = open_items.get(int(ref))
        where = {"sheet": cell["sheet"], "row": cell["row"], "word": snap["plain"], "plain": snap["plain"]}
        key, rest, how = read_choice("pending", cell.get("decision"))
        if item is None:
            if key in ("add", "remove"):
                out["skipped"].append({"plain": snap["plain"], "then": snap["entry"] or None,
                                       "now": current.get(snap["plain"]), "wanted": None,
                                       "why": "already taken off the pending list"})
            continue
        if (cell.get("notes") or "") != (item.get("notes") or ""):
            out["notes"][item["id"]] = cell.get("notes") or ""
        if how == "unclear":
            out["unclear"].append(where | {"typed": cell.get("decision"), "why": "not one of the choices"})
            out["stay"] += 1
            continue
        if how == "words":
            out["words"].append(where | {"typed": cell.get("decision"), "read": label("pending", key) or
                                         "Stays pending"})
        if key == "remove":
            out["remove"].append(item["id"])
            continue
        if key != "add":
            out["stay"] += 1
            continue
        cap = read_capital(cell.get("capital"))
        spell, bad = spelling(cell.get("result"))
        why = ("“Add” was chosen but “Result” is empty" if not spell else bad or
               ("the Capital column isn't Normal or Always capital" if cap == "?" else None))
        if why:
            out["unclear"].append(where | {"typed": cell.get("result") or cell.get("decision"), "why": why})
            out["stay"] += 1
            continue
        then = snap["entry"] or None
        code = compose(spell, cap if cap in ("normal", "capital") else None, then)
        wanted.setdefault(snap["plain"], []).append((item["id"], then, code, where))
    for plain, asks in sorted(wanted.items()):
        codes = {code for _, _, code, _ in asks}
        if len(codes) > 1:
            for _, _, code, where in asks:
                out["unclear"].append(where | {"typed": harvest.describe(code), "why": "two rows of this word give "
                                                                                      "different results"})
            out["stay"] += len(asks)
            continue
        ids = [i for i, _, _, _ in asks]
        then, code = asks[0][1], asks[0][2]
        now = current.get(plain)
        if now != then:
            out["skipped"].append({"plain": plain, "then": then, "now": now, "wanted": code})
            out["stay"] += len(asks)
        elif now == code:
            out["resolve"].append(plain)
            out["done"] += ids
        elif now is None:
            out["add"].append({"plain": plain, "result": code, "item": ids[0]})
            out["done"] += ids
        else:
            out["update"].append({"plain": plain, "old": now, "result": code, "item": ids[0]})
            out["done"] += ids
    return out


def fingerprint(p: dict) -> str:
    """What Apply will do, as a short hash: the confirm page sends it back, so nothing different is applied."""
    keep = {k: p.get(k) for k in ("add", "update", "delete", "remove", "done", "resolve")}
    keep["pending"] = [(x["plain"], x["suggested"]) for x in p.get("pending", [])]
    keep["notes"] = sorted((p.get("notes") or {}).items())
    return hashlib.sha256(json.dumps(keep, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:20]


def sentence(p: dict, *, done: bool = False) -> str:
    """'This will add 3 words, change 2, and put 130 on the pending list.' (done: 'Applied: added 3 words, …')"""
    def n(k, one, many):
        return f"{k:,} {one if k == 1 else many}"
    verbs = ("added", "changed", "deleted", "put", "took") if done else ("add", "change", "delete", "put", "take")
    parts = []
    if p["add"]:
        parts.append(f"{verbs[0]} {n(len(p['add']), 'word', 'words')}")
    if p["update"]:
        parts.append(f"{verbs[1]} {len(p['update']):,}")
    if p["delete"]:
        parts.append(f"{verbs[2]} {len(p['delete']):,}")
    if p.get("pending_new"):
        parts.append(f"{verbs[3]} {len(p['pending_new']):,} on the pending list")
    if p.get("pending_off"):
        parts.append(f"{verbs[4]} {p['pending_off']:,} off the pending list")
    if not parts:
        return "Applied: nothing in the list changed." if done else "This changes nothing in the list."
    joined = parts[0] if len(parts) == 1 else " and ".join(parts) if len(parts) == 2 else \
        ", ".join(parts[:-1]) + ", and " + parts[-1]
    return ("Applied: " if done else "This will ") + joined + "."
