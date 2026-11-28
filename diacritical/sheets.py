"""The words dashboard's Excel round trip (admin_words.py): the report of an upload, the pending list and the whole list
as .xlsx; reading a filled report or pending list back; and what Apply does with it. Pure: no database, no web.

The report: 'Start here', then the sheets that need a decision ('Different from our list', 'New words', 'Written without
marks'; for an offline list 'New in your list', 'Changed', 'Missing from your list'), then read-only sheets ('Style
only', 'Doubtful', 'Already in our list'). Empty sheets are left out. The decision columns have drop-downs (warning
style: other text is allowed; it is read where the meaning is obvious, otherwise as Pending, and listed on the confirm
page). A hidden column 'Ref' ties each row to its item, so sorting or deleting rows is harmless; a hidden sheet
'_dashboard' says what the file is (a report of review N, or the pending list with what each row showed) and for which
list (trial or live).

'Also typed as' (optional, on the sheets whose decisions keep an entry and on the pending list): other ways people type
the word, separated by commas. When the word keeps an entry, each becomes an entry with exactly the same code; a word
put on the pending list takes them with it. Columns are read by their header, so the column can also be added by hand
to a report made before it existed.

One row per plain word on 'Different from our list' and 'New words' (harvest.group_words; a review stored before is
regrouped on reading, and its old per-spelling Refs 'c3:0' are still read: item_cells, answers_from_report).
'Correct spelling' (and the pending list's 'Result'): one spelling, several separated by commas (choices, labels in
brackets allowed), or a code as stored (spellings, word_code). Filled in with an empty decision it counts as
"Document is right" / "Add".
"""
from __future__ import annotations

import hashlib
import io
import json
import re

from . import harvest
from .cleaning import clean_plain, clean_result
from .diacritics import CHOICE_COMMA

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
ALSO = "Also typed as"
NEW_SHEET = "New words"
OWN_HELP = ("To add a word that isn't in the report, type it on a new row at the bottom of “New words” (the word with "
            "its marks) and choose Add.")
CORRECT_HELP = ("“Correct spelling”: the spelling you want. Several spellings, separated by commas, become choices: "
                "Tṛtīya (masculine), Tṛtīyā (feminine). Filling it in is enough: an empty decision then counts as "
                "“Document is right” (or “Add” for a new word).")
ALSO_HELP = ("“Also typed as” (optional): other ways people type the word, separated by commas, e.g. krishna, krushna. "
             "Each gets the same result as the word.")
PLAIN_OK = re.compile(r"[a-z0-9' \-]+")      # what a plain word of the list may hold (as harvest.clean_list_rows)


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


def spellings(text) -> tuple[list[str], str | None, str | None]:
    """A 'Correct spelling' cell (or the pending list's 'Result'): (the spellings, each in the house style; a code
    typed as it is stored, '{{...}}', or None; why it can't be used, or None). Several spellings are separated by
    commas or semicolons (a comma inside a label's brackets stays: 'āratī (hindi, marathi)'); they become choices."""
    raw = str(text or "").strip()
    if not raw:
        return [], None, None
    if "{" in raw or "}" in raw:                      # typed as a code: used as it is
        code = clean_result(harvest.house(raw))
        why = harvest.code_problem(code)
        return [], (None if why else code), (f"the spelling “{raw}” can't be used: {why}" if why else None)
    out = []
    for part in CHOICE_COMMA.split(re.sub(r"[;\n\r]+", ",", raw)):
        part = part.strip().strip("\"'“”‘’").strip().rstrip(".").strip()
        if not part:
            continue
        s = clean_result(harvest.house(part))
        why = harvest.code_problem(s) or ("there is more after the brackets (separate spellings with commas)"
                                          if re.search(r"\)\s*[^\s,)]", s) else None)
        if why:
            return [], None, f"the spelling “{raw}” can't be used: {why}"
        if s not in out:
            out.append(s)
    return out, None, None


def word_code(spelled: list[str], code: str | None, cap: str | None, entry: str | None,
              capitalised: bool = False) -> str:
    """The code a decision gives a word: a typed code as it is; several spellings as choices (Always capital:
    {{}}{{a, b}}; Capital blank keeps an always-capital entry so); one spelling in the entry's own form (Capital
    blank: as the entry is, for a new word Normal)."""
    if code:
        return code
    if len(spelled) > 1:
        return harvest.several(spelled, cap, entry)
    s = spelled[0] if spelled else ""
    if entry:
        return harvest.with_type(harvest.replace_result(entry, s, capitalised), cap or "keep")
    return harvest.with_type(s, cap if cap in ("normal", "capital") else "normal")


def split_also(text, own: str = "") -> tuple[list[str], list[str]]:
    """An 'Also typed as' cell: (the other ways of typing, each a plain word in the house style, and the parts that
    can't be a plain word, as typed). Split at commas, semicolons and line breaks; empty parts, repeats and the row's
    own plain word are left out."""
    good, bad = [], []
    for part in re.split(r"[,;\n\r]+", str(text or "")):
        part = part.strip().strip(".\"'“”‘’").strip()
        v = clean_plain(part)
        if not v or v == own or v in good or part in bad:
            continue
        (good if PLAIN_OK.fullmatch(v) else bad).append(v if PLAIN_OK.fullmatch(v) else part)
    return good, bad


def also_text(text, own: str = "") -> str:
    """An 'Also typed as' cell as it is kept with a pending word: 'krishna, krushna'."""
    good, bad = split_also(text, own)
    return ", ".join(good + bad)


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
    wide = sum(w for h, w in columns[:-1]) or 60
    ws.row_dimensions[1].height = max(36, 17 * -(-len(note) // int(wide * 0.85)) + 4)
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
    """The Ref of an item's row: its id (one row per word). A report made before that had a row per spelling,
    'c3:0', 'c3:1' (harvest.old_refs); those are still read."""
    return [item["id"]]


CELL_FIELDS = (("decision", "Your decision"), ("capital", "Capital"), ("spelling", "Correct spelling"),
               ("also", ALSO), ("notes", "Notes"))


def item_cells(item: dict, cells: dict) -> dict | None:
    """What an uploaded copy said for an item, as one row ({decision, capital, spelling, also, notes}), or None.
    A report made before one row per word had a row per spelling: those rows are merged (the first value of each
    column is kept, a different one goes into Notes; 'Document is right' / 'Add' on one spelling's row writes that
    spelling in 'Correct spelling'; notes and 'Also typed as' are put together)."""
    if item["id"] in cells:
        return cells[item["id"]]
    found = sorted((k, ref) for ref, k in harvest.old_refs(item).items() if ref in cells)
    if not found:
        return None
    rows, options = item.get("rows") or [], item.get("options") or []
    out = {key: "" for key, _ in CELL_FIELDS}
    notes, also, said = [], [], []
    for k, ref in found:
        cell = cells[ref]
        form = rows[k]["form"] if k < len(rows) else item["form"]
        values = {key: str(cell.get(key) or "").strip() for key, _ in CELL_FIELDS}
        if read_choice(item["kind"], values["decision"])[0] in ("doc", "add") and not values["spelling"] \
                and len(options) > 1:
            values["spelling"] = form
        for key, header in CELL_FIELDS:
            v = values[key]
            if not v:
                continue
            if key == "notes":
                notes += [v] if v not in notes else []
            elif key == "also":
                also += [p.strip() for p in re.split(r"[,;\n\r]+", v) if p.strip() and p.strip() not in also]
            elif not out[key]:
                out[key] = v
            elif (clean_result(harvest.house(v)) if key == "spelling" else _norm(v)) != \
                    (clean_result(harvest.house(out[key])) if key == "spelling" else _norm(out[key])):
                said.append(f"the {form} row said {header}: {v}")
    out["notes"] = " · ".join(notes + said)
    out["also"] = ", ".join(also)
    return out


def _spelt(item: dict) -> str:
    """The 'Word' cell of a word's row: every spelling in the document, 'Tṛtīyā, Tṛtīya (same as our list)'."""
    parts = [r["form"] for r in item.get("rows") or []] or [item["form"]]
    for x in item.get("same") or []:
        parts.append(f"{x['form']} (same as our list)" if x.get("status") != "style only" else
                     f"{x['form']} (same as our list, typed differently: {x.get('why') or 'style only'})")
    return ", ".join(parts)


def _examples(item: dict) -> str:
    """One short example line per spelling, each on its own line."""
    spelt = (item.get("rows") or []) + (item.get("same") or [])
    width = 170 if len(spelt) <= 1 else 110
    seen, lines = set(), []
    for r in spelt or [item]:
        line = r.get("line") or ""
        if line and line not in seen:
            seen.add(line)
            lines.append(harvest.context(line, r["form"], width))
    return "\n".join(lines) or item.get("line", "")


def report_xlsx(review: dict, info: dict, cells: dict | None = None, extra: list | None = None) -> bytes:
    """The report of a review, to fill in. info: review (id), list ('trial'|'live'), list_label, files, made, by,
    status. cells: what an uploaded copy said, by Ref ({decision, capital, spelling, also, notes}), to show it again
    (also from a report made before one row per word: item_cells); extra: the rows typed in on its 'New words'
    sheet (read_filled's 'extra'), shown again at the bottom."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment
    review = harvest.regroup(review)
    cells = cells or {}
    s = review["sections"]
    doc = review["kind"] == "document"
    sections = DOC_SECTIONS if doc else LIST_SECTIONS
    wb = Workbook()
    start = wb.active
    start.title = "Start here"

    def c(ref, key, default=""):
        return (cells.get(ref) or {}).get(key, default) if ref in cells else default

    def said(item):
        """What an uploaded copy said for a word's row: c(key, default)."""
        got = item_cells(item, cells)
        return lambda key, default="": (got.get(key) or "") if got is not None else default

    if doc:
        decision_sheets = [sheet for key, sheet, _, needs in sections if needs and s.get(key) and key != "unmarked"]
    else:
        decision_sheets = [sheet for key, sheet, _, needs in sections if needs and s.get(key)]
    files = ", ".join(info.get("files") or [])
    steps = []
    if decision_sheets:
        steps.append("Fill in the yellow column “Your decision” on " + " and ".join(f"“{x}”" for x in decision_sheets)
                     + (". Each word has one row, with every spelling the document uses." if doc else ".")
                     + " Anything left blank goes to the pending list.")
    if doc:
        steps += [CORRECT_HELP,
                  "When the document spells a word more than one way, type the spelling you want in “Correct "
                  "spelling” (or several, as choices).",
                  "“Capital”: choose Always capital for names and places; blank keeps the entry as it is (for a new "
                  "word: Normal).",
                  ALSO_HELP,
                  "“Notes” go with the word to the pending list, for the Diacriticals team."]
        if s.get("new"):
            steps.append(OWN_HELP)
        if s.get("unmarked"):
            steps.append("“Written without marks” already says “Nothing to change”; choose Pending for a word the "
                         "team should look at.")
    else:
        if s.get("lnew") or s.get("lchanged"):
            steps.append(ALSO_HELP)
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
                got = said(item)
                rows.append([_spelt(item), item["plain"], item["times"], item["where"], _examples(item), _gives(item),
                             got("decision"), got("capital"), got("spelling"), got("also"), got("notes"),
                             item["id"]])
            _table_sheet(wb, "Different from our list",
                         "In our list, but the document spells it differently. One row per word, with every spelling "
                         "the document uses (“same as our list”: that spelling already matches). “Your decision”: "
                         "blank = Pending. “Correct spelling”: the spelling you want; several, separated by commas, "
                         "become choices; filling it in is enough (it counts as “Document is right”). “Capital” "
                         "blank = keep as it is. " + pend_note,
                         [("Word (as in the document)", 26), ("Plain word", 16), ("Times", 7), ("Pages", 10),
                          ("Example line", 48), ("Our list gives", 24), ("Your decision", 20), ("Capital", 15),
                          ("Correct spelling", 22), (ALSO, 24), ("Notes", 30), ("Ref", 8)], rows,
                         decide={"Your decision": [lab for _, lab in CHOICES["clash"]],
                                 "Capital": [lab for _, lab in CAPITALS], "Correct spelling": None, ALSO: None})
        if s.get("new"):
            rows = []
            for item in s["new"]:
                got = said(item)
                cap = CAP_LABEL["capital"] if item.get("seen") == "name" else ""
                rows.append([_spelt(item), item["plain"], item["times"], item["where"], _examples(item),
                             got("decision"), got("capital", cap), got("spelling"), got("also"), got("notes"),
                             item["id"]])
            for x in extra or []:
                rows.append([x.get("word"), x.get("plain"), None, None, None, x.get("decision"), x.get("capital"),
                             x.get("spelling"), x.get("also"), x.get("notes"), None])
            _table_sheet(wb, "New words",
                         "Words with marks that are not in our list, one row per word. “Your decision”: blank = "
                         "Pending. “Correct spelling”: the spelling you want; several, separated by commas, become "
                         "choices; filling it in is enough (it counts as “Add”). “Capital”: blank = Normal (Always "
                         "capital is filled in for words only seen with a capital, also mid-sentence). " + pend_note,
                         [("Word (as in the document)", 24), ("Plain word", 16), ("Times", 7), ("Pages", 10),
                          ("Example line", 52), ("Your decision", 16), ("Capital", 15), ("Correct spelling", 22),
                          (ALSO, 24), ("Notes", 30), ("Ref", 8)], rows,
                         decide={"Your decision": [lab for _, lab in CHOICES["new"]],
                                 "Capital": [lab for _, lab in CAPITALS], "Correct spelling": None, ALSO: None})
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
                          ("Your decision", 16), (ALSO, 24), ("Notes", 30), ("Ref", 8)],
                         [[i["plain"], i["result"], harvest.describe(i["result"]), i["row"],
                           c(i["id"], "decision", label("lnew", "add")), c(i["id"], "also"), c(i["id"], "notes"),
                           i["id"]] for i in s["lnew"]],
                         decide={"Your decision": [lab for _, lab in CHOICES["lnew"]], ALSO: None})
        if s.get("lchanged"):
            _table_sheet(wb, "Changed", "In both lists, with a different result. “Your decision”: blank = Pending. "
                         + pend_note,
                         [("Plain word", 18), ("Our list gives", 30), ("Your list gives", 30), ("Row", 7),
                          ("Your decision", 18), (ALSO, 24), ("Notes", 30), ("Ref", 8)],
                         [[i["plain"], harvest.describe(i["entry"]), harvest.describe(i["result"]), i["row"],
                           c(i["id"], "decision"), c(i["id"], "also"), c(i["id"], "notes"), i["id"]]
                          for i in s["lchanged"]],
                         decide={"Your decision": [lab for _, lab in CHOICES["lchanged"]], ALSO: None})
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
                          ("made", info.get("made_iso", "")), ("version", 2)])
    return _save(wb)


def pending_xlsx(items: list[dict], current: dict[str, str], info: dict) -> bytes:
    """The open pending items, to fill in: 'Your decision' (Add / Remove from pending; blank = stays pending), 'Result'
    (the suggestion, to overwrite with the agreed spelling), 'Also typed as' (what the item carries), 'Capital',
    'Notes'. info: list, made, made_iso, by,
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
        elif p and p["type"] in ("capital", "choices"):     # several spellings: written as they are typed, a, b
            result = ", ".join(harvest.split_choices(p["main"]))
            cap = CAP_LABEL["capital"] if p["type"] == "capital" else ""
        rows.append([i["plain_text"], harvest.describe(sug) if sug else "", harvest.describe(entry) if entry else
                     "(not in our list)", i["reason"], i["source"], i["page"], i["context"], when(i["added_at"]),
                     "", result, i.get("also") or "", cap, i.get("notes") or "", str(i["id"])])
        snap.append([str(i["id"]), i["plain_text"], entry or "", sug])
    _table_sheet(wb, "Pending", "Words waiting for an answer. “Your decision”: Add puts “Result” in the list (change "
                 "it to the agreed spelling if needed; house style is applied; several spellings, separated by commas, "
                 "become choices), Remove from pending takes the word off the list, blank = stays pending. “Capital” "
                 "blank = Normal for a new word, or keeps the entry's form "
                 "for a change. “Also typed as”: other ways people type the word, separated by commas; with Add each "
                 "gets the same result. Then upload this file on the dashboard under “Upload your decisions”.",
                 [("Plain word", 16), ("Suggested", 24), ("Our list now gives", 22), ("Why", 36), ("Source", 22),
                  ("Page", 8), ("Example line", 44), ("Added", 16), ("Your decision", 18), ("Result", 22),
                  (ALSO, 24), ("Capital", 15), ("Notes", 30), ("Ref", 8)], rows,
                 decide={"Your decision": [lab for _, lab in CHOICES["pending"]], "Result": None, ALSO: None,
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
           "notes": "notes", "ref": "ref", "word (as in the document)": "word", "plain word": "plain",
           # 'Also typed as', also as a column someone adds by hand to a report made before it existed
           "also typed as": "also", "also typed": "also", "also add": "also", "other spellings": "also",
           "other spelling": "also", "variants": "also", "variant": "also", "other ways of typing": "also"}
_HEAD_KEYS = {re.sub(r"[\s_\-]+", "", k): v for k, v in HEADERS.items()}


def _head(value) -> str | None:
    """What a column header means (a key of HEADERS' values), ignoring case, spaces and a remark in brackets at the
    end ('Also typed as (optional)'), or None."""
    t = _norm(value)
    for cand in (t, re.sub(r"\s*\([^)]*\)$", "", t)):
        key = _HEAD_KEYS.get(re.sub(r"[\s_\-]+", "", cand))
        if key:
            return key
    return None


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
    spelling, result, notes, also}}} ('also' is None when the sheet has no 'Also typed as' column), 'extra': [rows
    typed in by hand on a report's 'New words' sheet: {sheet, row, word, plain, decision, capital, spelling, notes,
    also}]}. Raises NotOurs / ReadProblem with a message for people."""
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
    cells, extra, first_word = {}, [], {}
    for ws in wb.worksheets:
        if ws.title == DASHBOARD:
            continue
        own_rows = meta["kind"] == "report" and _norm(ws.title) == NEW_SHEET.lower()
        rows = list(ws.iter_rows(values_only=True))
        head = next((n for n, r in enumerate(rows[:6]) if any(_norm(v) == "ref" for v in r)), None)
        if head is None:
            continue
        cols = {}
        for i, v in enumerate(rows[head]):
            key = _head(v)
            if key and key not in cols:
                cols[key] = i
        for n, r in enumerate(rows[head + 1:], head + 2):
            def get(k):
                i = cols.get(k)
                v = r[i] if i is not None and i < len(r) else None
                return "" if v is None else str(v).strip()
            ref = get("ref")
            if ref and ref not in cells:
                cells[ref] = {"sheet": ws.title, "row": n, "word": get("word") or get("plain"),
                              **{k: get(k) for k in ("decision", "capital", "spelling", "result", "notes")},
                              "also": get("also") if "also" in cols else None}
                first_word[ref] = (get("word"), get("spelling"), get("plain"))
                continue
            if not own_rows:
                continue
            # a row typed in by hand on 'New words' (no Ref), or a copied row given another word: a word of their own
            own = {"sheet": ws.title, "row": n, "word": get("word"), "plain": get("plain"),
                   **{k: get(k) for k in ("decision", "capital", "spelling", "notes")},
                   "also": get("also") if "also" in cols else None}
            if ref:
                word, spelt, plain = first_word[ref]
                if (own["word"], own["spelling"]) == (word, spelt):
                    continue                              # the same row twice
                if own["plain"] == plain:
                    own["plain"] = ""                     # copied along: made from the new word instead
            if own["word"] or own["spelling"] or own["plain"]:
                extra.append(own)
    review = meta.get("review")
    return {"kind": meta["kind"], "list": meta["list"], "made": str(meta.get("made") or ""),
            "review": int(review) if str(review or "").isdigit() else None, "snapshot": snapshot, "cells": cells,
            "extra": extra}


# ---------- what the decisions mean ----------

def _items(review: dict):
    for key, items in review["sections"].items():
        for item in items:
            if item.get("kind") in CHOICES:
                yield item


def answers_from_report(review: dict, filled: dict) -> dict:
    """Each item's decision from a filled report: {'rows': {item id: {choice, option (which of the document's
    spellings, for a decision without a correct spelling), cap, spellings (typed in 'Correct spelling'), code (a code
    typed there), spelling (the first of them), note, also (other ways of typing it), also_bad (parts of 'Also typed
    as' that can't be a plain word)}}, 'unclear': [...], 'words': [...] (read from someone's own words, or a correct
    spelling with no decision), 'missing': [words whose rows weren't in the file], 'cells': the cells (to show again
    in a re-downloaded report), 'extra': [words typed in on their own rows (_own_row)], 'extra_cells': those rows as
    typed}. A report made before one row per word (a row per spelling, Refs 'c3:0', 'c3:1') is read too."""
    review = harvest.regroup(review)
    cells = filled["cells"]
    out = {"rows": {}, "unclear": [], "words": [], "missing": [], "cells": cells}
    for item in _items(review):
        kind = item["kind"]
        per_row, notes, problems, also, also_bad = [], [], [], [], []
        options = item.get("options") or []
        if item["id"] in cells or kind not in ("clash", "new"):
            found = [(None, cells[item["id"]])] if item["id"] in cells else []
        else:                                         # a report with a row per spelling: k = that spelling
            found = [(k, cells[ref]) for k, ref in sorted((k, r) for r, k in harvest.old_refs(item).items())
                     if ref in cells]
        for k, cell in found:
            good, bad = split_also(cell.get("also"), item["plain"])   # any row of the word: they're all for it
            also += [v for v in good if v not in also]
            also_bad += [v for v in bad if v not in also_bad]
            where = {"sheet": cell["sheet"], "row": cell["row"], "word": cell.get("word") or item["plain"]}
            key, rest, how = read_choice(kind, cell.get("decision"))
            cap = read_capital(cell.get("capital")) if kind in ("clash", "new") else None
            spelled, code, bad = spellings(cell.get("spelling")) if kind in ("clash", "new", "unmarked") else \
                ([], None, None)
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
            typed = bool(spelled or code)
            if typed and key in ("list", "skip", "none"):
                problems.append(where | {"typed": f"{cell.get('decision')} + {cell.get('spelling')}",
                                         "why": f"a correct spelling was given, but the decision is "
                                                f"“{label(kind, key)}”"})
                continue
            if key is None and typed and kind in ("clash", "new"):
                key = "doc" if kind == "clash" else "add"   # a correct spelling is enough: Document is right / Add
                out["words"].append(where | {"typed": f"Correct spelling: {cell.get('spelling')} (no decision)",
                                             "read": label(kind, key) + (
                                                 f" · {CAP_LABEL[cap]}" if cap in CAP_LABEL else "")})
            elif how == "words":
                out["words"].append(where | {"typed": " · ".join(x for x in (cell.get("decision"),
                                                                             cell.get("notes")) if x),
                                             "read": label(kind, key or BLANK[kind]) + (
                                                 f" · {CAP_LABEL[cap]}" if cap in CAP_LABEL else "")})
            if k is None and key in ("doc", "add") and not typed and len(options) > 1:
                problems.append(where | {"typed": cell.get("decision"),
                                         "why": "the document spells it more than one way: type the spelling(s) you "
                                                "want in “Correct spelling”"})
                continue
            per_row.append({"option": k or 0, "choice": key, "cap": cap, "spellings": spelled, "code": code,
                            "where": where})
        if BLANK[kind] == "pending" and not found:
            out["missing"].append(item["plain"])     # its rows were deleted from the file: it goes to pending
        d = {"choice": BLANK[kind], "option": 0, "cap": None, "spelling": "", "spellings": [], "code": None,
             "note": " · ".join(notes), "also": also, "also_bad": also_bad}
        chosen = [r for r in per_row if r["choice"] is not None]

        def gives(r):                                # what a row's decision would put in the list
            return (r["choice"], r["code"] or tuple(r["spellings"] or options[r["option"]:r["option"] + 1]),
                    r["cap"])
        if problems:
            out["unclear"] += [p | {"plain": item["plain"]} for p in problems]
            d["choice"] = "pending" if BLANK[kind] == "pending" or kind == "unmarked" else BLANK[kind]
            d["unclear"] = True
        elif chosen:
            picked = {r["choice"] for r in chosen}
            if len(picked) == 1 and (not picked & {"doc", "add"} or len({gives(r) for r in chosen}) == 1):
                r = chosen[0]
                d.update(choice=r["choice"], option=r["option"], cap=r["cap"], spellings=r["spellings"],
                         code=r["code"], spelling=(r["spellings"] or [""])[0])
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
            blank = [r for r in per_row if r["spellings"] or r["code"] or r["cap"]]
            if blank:     # no decision, but a capital (or, without marks, a spelling): they go with the word to pending
                r = blank[0]
                d.update(option=r["option"], cap=r["cap"], spellings=r["spellings"], code=r["code"],
                         spelling=(r["spellings"] or [""])[0])
        out["rows"][item["id"]] = d
    out["extra"] = [_own_row(x, out) for x in filled.get("extra") or []]
    out["extra_cells"] = filled.get("extra") or []
    return out


def _own_row(x: dict, out: dict) -> dict:
    """A row typed in by hand on 'New words': {id, word, plain, choice ('add'|'pending'|'skip'), code, note, also,
    also_bad, where} or, when it can't be used, {.., why}. Decisions read from someone's own words go to
    out['words']. A filled 'Correct spelling' with no decision is Add; several spellings there (or in 'Word') become
    choices; the plain word is made from the first."""
    where = {"sheet": x["sheet"], "row": x["row"]}
    text = x.get("spelling") or x.get("word") or ""
    word = x.get("word") or x.get("spelling") or x.get("plain") or ""
    first = re.sub(r"\s*\(.*?\)", "", CHOICE_COMMA.split(re.sub(r"[;\n\r]+", ",", word))[0]).strip()
    plain = clean_plain(x["plain"]) if x.get("plain") else clean_plain(harvest.plain_typed(first))
    key, rest, how = read_choice("new", x.get("decision"))
    cap = read_capital(x.get("capital"))
    spelled, code, bad = spellings(text)
    note = " ".join(v for v in (rest if how == "words" else "", x.get("notes") or "") if v).strip()
    also, also_bad = split_also(x.get("also"), plain)
    if key is None and how != "unclear" and x.get("spelling"):
        key = "add"                                   # a correct spelling is enough
    own = {"id": f"own{x['row']}", "word": word, "plain": plain, "choice": key or "pending", "code": "", "note": note,
           "also": also, "also_bad": also_bad, "where": where}
    why = ("there is no word with its marks in the “Word” column" if not text else
           f"“{x.get('decision')}” isn't Add, Pending or Skip" if how == "unclear" else
           "the Capital column isn't Normal or Always capital" if cap == "?" else
           bad or (f"the plain word “{plain}” has letters our list can't use" if not PLAIN_OK.fullmatch(plain)
                   else None))
    if why:
        return own | {"why": why}
    if cap is None:
        cap = capital_in_words(rest) or capital_in_words(x.get("notes"))
    if how == "words":
        out["words"].append(where | {"word": word, "typed": " · ".join(v for v in (x.get("decision"), x.get("notes"))
                                                                         if v),
                                     "read": label("new", key) + (f" · {CAP_LABEL[cap]}" if cap in CAP_LABEL else "")})
    own["code"] = word_code(spelled, code, cap if cap in ("normal", "capital") else None, None)
    return own


def compose(result_text: str, cap: str | None, entry: str | None) -> str:
    """The code to store for a spelling from the pending list: Always capital / Normal as chosen; blank keeps the
    entry's form (a new word: Normal). Several spellings (separated by commas) become choices. A typed code ({{ }})
    is used as it is."""
    spelled, code, _ = spellings(result_text)
    if code is None and "{" in str(result_text or ""):
        code = clean_result(harvest.house(result_text))
    if code:
        return harvest.with_type(code, cap or "keep")
    if len(spelled) > 1:
        return harvest.several(spelled, cap, entry)
    r = spelled[0] if spelled else clean_result(harvest.house(result_text or ""))
    if cap == "capital":
        return clean_result("{{}}{{" + r + "}}")
    if cap == "normal" or not entry:
        return r
    return harvest.replace_result(entry, r, False)


def _empty_plan() -> dict:
    return {"add": [], "update": [], "delete": [], "pending": [], "resolve": [], "remove": [], "notes": {},
            "also_set": {}, "also_left": [], "own_left": [], "skipped": [], "unclear": [], "words": [], "missing": [],
            "stay": 0}


def _guard(out: dict) -> set:
    """A word that two rows want to change differently is left alone (listed as unclear). Returns those words."""
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
    return bad


def _also(out: dict, wants: list[dict], current: dict[str, str], own: set[str]) -> None:
    """The other ways of typing ('Also typed as') as adds with exactly the code their word ends up with. wants:
    [{plain, item, also: [typed words], bad: [parts that can't be a plain word], code: the word's entry after Apply
    or None, why: why not used when code is None}]. A typed word already in the list with that code needs nothing;
    with another code it is left as it is; one given for two words with different codes, or that has its own row in
    the same file, isn't added. Everything not added is listed in out['also_left'] ({plain, typed, why})."""
    left = out["also_left"]
    by_typed = {}
    for w in wants:
        for raw in w.get("bad") or []:
            left.append({"plain": w["plain"], "typed": raw,
                         "why": f"“{raw}” can't be a typed word (only letters a-z, spaces, hyphens and apostrophes), "
                                "so it wasn't added."})
        also = w.get("also") or []
        if not also:
            continue
        if w["code"] is None:
            if w.get("why"):
                left.append({"plain": w["plain"], "typed": ", ".join(also), "why": f"not used, because {w['why']}."})
            continue
        for v in also:
            if v in own:
                left.append({"plain": w["plain"], "typed": v,
                             "why": f"{v} has its own row in this file, so it wasn't added as another way of typing "
                                    f"{w['plain']}."})
            else:
                by_typed.setdefault(v, []).append(w)
    for v, ws in by_typed.items():
        first = ws[0]
        if len({w["code"] for w in ws}) > 1:
            names = sorted({w["plain"] for w in ws})
            left.append({"plain": ", ".join(names), "typed": v,
                         "why": f"{v} was given for {' and '.join(names)}, which get different results, so it wasn't "
                                "added."})
            continue
        now = current.get(v)
        if now == first["code"]:
            continue                                  # already in the list exactly so: nothing to do
        if now is not None:
            left.append({"plain": first["plain"], "typed": v,
                         "why": f"{v} is already in the list as {harvest.describe(now)}, so it was left as it is; "
                                "change it on the Mappings page if needed."})
            continue
        out["add"].append({"plain": v, "result": first["code"], "item": first["item"], "also_for": first["plain"]})


def plan(review: dict, answers: dict, current: dict[str, str], source: str = "") -> dict:
    """What Apply does with a report's answers, against the list as it is now: {'add': [{plain, result, item,
    also_for (when it is another way of typing that word)}], 'update': [{plain, old, result, item}], 'delete':
    [{plain, old, item}], 'pending': [{plain, suggested, reason, source, page, context, notes, also, item}], 'resolve':
    [plain words confirmed as they are], 'skipped': [{plain, then, now, wanted}] (changed in the list since the report
    was made), 'also_left' (other ways of typing not added, with why), 'unclear', 'words', 'missing'}."""
    out = _empty_plan()
    out["unclear"] = list(answers.get("unclear", []))
    out["words"] = list(answers.get("words", []))
    out["missing"] = list(answers.get("missing", []))
    rows = answers.get("rows", {})
    wants = []

    def change(item, then, new) -> bool:
        """Plan the change; True when the word's entry will be `new` after Apply."""
        plain = item["plain"]
        now = current.get(plain)
        if new == then:
            return now == then
        if now != then:
            out["skipped"].append({"plain": plain, "then": then, "now": now, "wanted": new})
            return False
        elif then is None:
            out["add"].append({"plain": plain, "result": new, "item": item["id"]})
        elif new is None:
            out["delete"].append({"plain": plain, "old": then, "item": item["id"]})
        else:
            out["update"].append({"plain": plain, "old": then, "result": new, "item": item["id"]})
        return True

    def pend(item, d, suggested, reason, row=None):
        row = row or item
        out["pending"].append({"plain": item["plain"], "suggested": suggested, "reason": reason, "source": source,
                               "page": row.get("where", ""), "context": row.get("line", ""),
                               "notes": d.get("note", ""), "also": ", ".join(d.get("also") or []),
                               "item": item["id"]})

    review = harvest.regroup(review)
    for item in _items(review):
        kind = item["kind"]
        d = rows.get(item["id"]) or {"choice": BLANK[kind], "option": 0, "cap": None, "spelling": "", "note": ""}
        choice, k, cap = d["choice"], d.get("option") or 0, d.get("cap")
        spelled = list(d.get("spellings") or ([d["spelling"]] if d.get("spelling") else []))
        typed_code = d.get("code")
        spell = (spelled or [""])[0]
        entry = item.get("entry")
        options = item.get("options") or []
        k = max(0, min(k, len(options) - 1))
        row = (item.get("rows") or [None] * len(options))[k] if kind in ("clash", "new") else None
        here = {"where": item.get("where", ""), "line": (row or item).get("line", "")}   # for the pending list
        # the code the word's entry ends up with (its other ways of typing get it), or why they aren't used
        want = {"plain": item["plain"], "item": item["id"], "also": d.get("also") or [], "bad": d.get("also_bad") or [],
                "code": None, "why": "the word was skipped"}
        wants.append(want)
        changed_meanwhile = f"{item['plain']} was changed in the list after this file was made"

        def keeps(then, new):
            if change(item, then, new):
                want["code"] = new
            else:
                want["why"] = changed_meanwhile
        if kind == "clash":
            capitalised = (row or item)["form"][:1].isupper()
            code = word_code(spelled or options[k:k + 1], typed_code, cap, entry, capitalised)
            if choice == "pending":
                differs = ", ".join(f["form"] for f in item["forms"] if f["status"] == "differs")
                pend(item, d, code, f"Different from our list: the document has {differs}; our list gives "
                                    f"{item['gives']}", here)
                want["why"] = None                    # they go with the word to the pending list
            elif choice == "list":
                if entry is None:
                    want["why"] = f"{item['plain']} has no entry of its own in the list"
                else:
                    keeps(entry, harvest.with_type(entry, cap) if cap in ("normal", "capital") else entry)
                out["resolve"].append(item["plain"])
            elif choice == "doc":
                keeps(entry, code)
        elif kind == "new":
            code = word_code(spelled or options[k:k + 1], typed_code, cap if cap in ("normal", "capital") else None,
                             None)
            if choice == "add":
                keeps(None, code)
            elif choice == "pending":
                spelt = item.get("spellings") or []
                pend(item, d, code, "New word: not in our list" + (
                    f" (the document spells it {' and '.join(spelt)})" if len(options) > 1 else ""), here)
                want["why"] = None
        elif kind == "unmarked":
            if choice == "pending":
                pend(item, d, typed_code or (harvest.several(spelled, None, entry) if len(spelled) > 1 else spell)
                     or entry or "", f"Written without marks in the document ({item['form']}); "
                                                    f"our list gives {item['gives']}")
                want["why"] = None
        elif kind == "lnew":
            if choice == "add":
                keeps(None, item["result"])
            elif choice == "pending":
                pend(item, d, item["result"], "New in an uploaded list", {"where": f"row {item['row']}", "line": ""})
                want["why"] = None
        elif kind == "lchanged":
            if choice == "use":
                keeps(entry, item["result"])
            elif choice == "keep":
                keeps(entry, entry)
                out["resolve"].append(item["plain"])
            elif choice == "pending":
                pend(item, d, item["result"], f"Changed in an uploaded list; our list has {harvest.describe(entry)}",
                     {"where": f"row {item['row']}", "line": ""})
                want["why"] = None
        elif kind == "lmissing":
            if choice == "delete":
                change(item, entry, None)
                want["why"] = "the word is deleted"
            else:
                keeps(entry, entry)
    in_file = {i["plain"] for i in _items(review)}
    own_words = _own_rows(out, answers.get("extra") or [], current, in_file, wants, pend)
    out["resolve"] += [u["plain"] for u in review["sections"].get("unchanged", [])]
    for p in _guard(out):
        for w in wants:
            if w["plain"] == p:
                w.update(code=None, why=f"{p} itself isn't changed (two rows want different things)")
    _also(out, wants, current, in_file | own_words)
    return out


def _own_rows(out: dict, extra: list[dict], current: dict[str, str], in_file: set[str], wants: list, pend) -> set:
    """Words typed in on their own rows of 'New words', treated like new words: Add (when the plain word isn't in
    the list yet) or Pending. Not added, and listed in out['own_left'] ({word, plain, sheet, row, why}): a plain word
    already in the list, or with a row of its own in the report, or typed on two rows with different results, and
    rows that can't be read. Returns their plain words."""
    groups = {}
    for x in extra:
        if x.get("why"):
            out["own_left"].append({"word": x["word"], "plain": x["plain"], **x["where"],
                                    "why": f"not added: {x['why']}."})
        elif x["choice"] != "skip":
            groups.setdefault(x["plain"], []).append(x)
    for plain, xs in groups.items():
        first = xs[0]
        want = {"plain": plain, "item": first["id"], "also": first["also"], "bad": first["also_bad"], "code": None,
                "why": f"{plain} itself wasn't added"}
        wants.append(want)
        now = current.get(plain)
        why = None
        if plain in in_file:
            why = f"{plain} has its own row in this report, so this row was left out; decide it there."
        elif len({(x["choice"], x["code"]) for x in xs}) > 1:
            why = f"{plain} is typed on more than one of your own rows, with different results, so none was used."
        elif first["choice"] == "pending":
            pend({"plain": plain, "id": first["id"]}, {"note": first["note"], "also": first["also"]}, first["code"],
                 f"Typed in on the report's “New words” sheet (row {first['where']['row']})")
            want["why"] = None
        elif now == first["code"]:
            why = f"{plain} is already in the list exactly like this, so nothing changes."
            want["code"] = now
        elif now is not None:
            why = (f"{plain} is already in the list as {harvest.describe(now)}, so it was left as it is; change it on "
                   "the Mappings page if needed.")
        else:
            out["add"].append({"plain": plain, "result": first["code"], "item": first["id"],
                               "own_row": f"{first['where']['sheet']}, row {first['where']['row']}"})
            want["code"] = first["code"]
        if why:
            for x in xs:
                out["own_left"].append({"word": x["word"], "plain": plain, **x["where"], "why": why})
    return set(groups)


def answers_from_pending(filled: dict) -> dict:
    """The rows of a filled pending list, as stored until Apply: {'snapshot', 'cells'}."""
    return {"snapshot": filled["snapshot"], "cells": filled["cells"], "made": filled.get("made", "")}


def plan_pending(answers: dict, open_items: dict[int, dict], current: dict[str, str]) -> dict:
    """What Apply does with a filled pending list, against the list and the pending list as they are now. Like plan(),
    plus 'remove' (pending ids taken off), 'done' (pending ids answered by a change), 'notes' ({id: new notes}),
    'also_set' ({id: its new 'Also typed as'}). An item answered Add brings its other ways of typing (the column as it
    is in the file; a file made before the column existed: what the item carries) with the same code."""
    out = _empty_plan()
    out["done"] = []
    snapshot, cells = answers["snapshot"], answers["cells"]
    wanted, wants = {}, []
    own = {snapshot[ref]["plain"] for ref in cells if ref in snapshot}
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
        typed = cell.get("also")
        if typed is None:                             # no such column in this file: what the item carries
            typed = item.get("also") or ""
        elif also_text(typed, snap["plain"]) != also_text(item.get("also"), snap["plain"]):
            out["also_set"][item["id"]] = also_text(typed, snap["plain"])
        good, bad = split_also(typed, snap["plain"])
        if how == "unclear":
            out["unclear"].append(where | {"typed": cell.get("decision"), "why": "not one of the choices"})
            out["stay"] += 1
            continue
        if how == "words":
            out["words"].append(where | {"typed": cell.get("decision"), "read": label("pending", key) or
                                         "Stays pending"})
        if key == "remove":
            out["remove"].append(item["id"])
            wants.append({"plain": snap["plain"], "item": item["id"], "also": good, "bad": [], "code": None,
                          "why": "the word was taken off the pending list"})
            continue
        if key != "add":
            out["stay"] += 1
            continue
        cap = read_capital(cell.get("capital"))
        spelled, typed_code, bad_spelling = spellings(cell.get("result"))
        why = ("“Add” was chosen but “Result” is empty" if not (spelled or typed_code or bad_spelling) else
               bad_spelling or ("the Capital column isn't Normal or Always capital" if cap == "?" else None))
        if why:
            out["unclear"].append(where | {"typed": cell.get("result") or cell.get("decision"), "why": why})
            out["stay"] += 1
            continue
        then = snap["entry"] or None
        code = compose(cell.get("result"), cap if cap in ("normal", "capital") else None, then)
        want = {"plain": snap["plain"], "item": item["id"], "also": good, "bad": bad, "code": None, "why": None}
        wants.append(want)
        wanted.setdefault(snap["plain"], []).append((item["id"], then, code, where, want))
    for plain, asks in sorted(wanted.items()):
        codes = {code for _, _, code, _, _ in asks}
        if len(codes) > 1:
            for _, _, code, where, want in asks:
                out["unclear"].append(where | {"typed": harvest.describe(code), "why": "two rows of this word give "
                                                                                      "different results"})
                want["why"] = f"{plain} itself isn't changed (its rows give different results)"
            out["stay"] += len(asks)
            continue
        ids = [i for i, _, _, _, _ in asks]
        then, code = asks[0][1], asks[0][2]
        now = current.get(plain)
        if now != then:
            out["skipped"].append({"plain": plain, "then": then, "now": now, "wanted": code})
            out["stay"] += len(asks)
            for *_, want in asks:
                want["why"] = f"{plain} was changed in the list after this file was made"
            continue
        if now == code:
            out["resolve"].append(plain)
        elif now is None:
            out["add"].append({"plain": plain, "result": code, "item": ids[0]})
        else:
            out["update"].append({"plain": plain, "old": now, "result": code, "item": ids[0]})
        out["done"] += ids
        for *_, want in asks:
            want["code"] = code
    _also(out, wants, current, own)
    return out


def fingerprint(p: dict) -> str:
    """What Apply will do, as a short hash: the confirm page sends it back, so nothing different is applied."""
    keep = {k: p.get(k) for k in ("add", "update", "delete", "remove", "done", "resolve")}
    keep["pending"] = [(x["plain"], x["suggested"], x.get("also") or "") for x in p.get("pending", [])]
    keep["notes"] = sorted((p.get("notes") or {}).items())
    keep["also_set"] = sorted((p.get("also_set") or {}).items())
    return hashlib.sha256(json.dumps(keep, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:20]


def sentence(p: dict, *, done: bool = False) -> str:
    """'This will add 5 words (2 of them other ways of typing a word), change 2, and put 130 on the pending list.'
    (done: 'Applied: added 5 words …')"""
    def n(k, one, many):
        return f"{k:,} {one if k == 1 else many}"
    verbs = ("added", "changed", "deleted", "put", "took") if done else ("add", "change", "delete", "put", "take")
    parts = []
    if p["add"]:
        total, also = len(p["add"]), sum(1 for c in p["add"] if c.get("also_for"))
        way = "other ways of typing a word"
        extra = ("" if not also else " (another way of typing a word)" if total == also == 1 else
                 f" (both {way})" if total == also == 2 else f" (all {way})" if total == also else
                 " (1 of them another way of typing a word)" if also == 1 else f" ({also:,} of them {way})")
        parts.append(f"{verbs[0]} {n(total, 'word', 'words')}{extra}")
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
