"""Words with diacritical marks in an uploaded file, compared with the Diacritical word list: what the dashboard's
report is made from (admin_words.py; the Excel side is diacritical/sheets.py). Pure functions: no database, no web.

- Reading: read_upload(name, data) tells a document (.docx, .pdf, .txt) from an offline word list (.xlsx, or a .txt
  of "plain,result" lines) and reads it. A PDF is read page by page with pypdf.
- Document review: review_document(files, mappings) finds every word with marks (and every plain word the list would
  mark) and sorts them into: different from our list (clash), new, written without marks, style only, doubtful
  (look-alike letters, PDF quirks), already in the list. Phrase entries of the list found in the text are compared.
  One item per plain word on clash and new (group_words): 'rows' are its spellings that differ from the list (the
  'options' of "Document is right" / "Add"), 'same' the spellings that match it (so they aren't also on 'already').
  regroup() puts a review stored before that into the same layout when it is read (old_refs: the Refs its rows had).
- List review: review_list(rows, problems, mappings): new, changed, missing from the file, problems, unchanged.
- Stored codes: parse_code, with_type (normal / always capital), replace_result, describe(code) for people.

The stored code of an entry (always small letters): `word` (follows the typed case), `{{a, b}}` (choices, all shown),
`{{small}}{{name}}` (typed with a capital it gives the second group: the "Name rule"), `{{}}{{x}}` (always capital).
Groups may hold comma choices themselves: `{{}}{{kārtik, kārtika}}`.
"""
from __future__ import annotations

import codecs
import io
import re
import unicodedata
from collections import Counter, defaultdict

from .cleaning import clean_plain, clean_result
from .diacritics import CHOICE_COMMA, phrase_key, translate_text

MAX_BYTES = 40 * 1024 * 1024
DOCUMENT_ENDINGS = (".docx", ".pdf", ".txt")
LIST_ENDINGS = (".xlsx",)

DEVANAGARI = re.compile("[\u0900-\u097f]")
# a word: letters with their marks, and hyphens between letters ("laḍḍū-s")
TOKEN = re.compile(r"[^\W\d_](?:[\w\u0300-\u036f]|[-\u2010\u2011](?=[^\W\d_]))*")
HYPHENS = re.compile("[\u2010\u2011]")
# letters that print like the right one but aren't: Greek Ᾱ for Ā, Greek iota for l
GREEK_LOOKALIKES = {"\u1fb9": "Ā", "\u1fb1": "ā", "\u1fd9": "Ī", "\u1fd1": "ī", "\u1fe9": "Ū", "\u1fe1": "ū",
                    "\u03b9": "l"}
LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}
_OTHER_MARKS = "[\u0300-\u0331\u0333-\u036f]*"


class ReadProblem(Exception):
    """A file that can't be read. The message is shown to the person who uploaded it."""


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_marks(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def has_marks(text: str) -> bool:
    return bool(re.search(r"[^\x00-\x7f]", text))


def house(word: str) -> str:
    """A spelling in the house style, keeping its capitals: look-alike letters replaced, an underlined pair (s̲h̲)
    made the tie (s͟h), a doubled tie made single, letters joined (NFC)."""
    for greek, latin in GREEK_LOOKALIKES.items():
        word = word.replace(greek, latin)
    t = unicodedata.normalize("NFD", HYPHENS.sub("-", word))
    t = re.sub(rf"(\w{_OTHER_MARKS})\u0332(\w{_OTHER_MARKS})\u0332", "\\1\u035f\\2", t)
    t = re.sub("\u035f+", "\u035f", t)
    return nfc(t)


def plain_typed(word: str) -> str:
    """What someone would type for a word: no marks, the same capitals."""
    for greek, latin in GREEK_LOOKALIKES.items():
        word = word.replace(greek, strip_marks(latin))
    return strip_marks(HYPHENS.sub("-", word))


def choices(output: str) -> list[str]:
    """The spellings in a translator output: "{{a (hindi),b}}" -> [a, b]; "x" -> [x]."""
    m = re.fullmatch(r"\{\{(.*)\}\}", output)
    if not m:
        return [output]
    return [re.sub(r"\s*\(.*?\)", "", x).strip() for x in CHOICE_COMMA.split(m.group(1))]


def list_letters(mappings: dict[str, str]) -> set[str]:
    """Letters a correctly typed word may use: a-z plus every letter and mark the list itself uses."""
    letters = set("abcdefghijklmnopqrstuvwxyz-")
    for code in mappings.values():
        for ch in nfc(code.lower()):
            if ch.isalpha() or unicodedata.combining(ch):
                letters.add(ch)
    return letters


# ---------- reading files ----------

def fix_pdf_text(text: str) -> str:
    """pypdf puts a space before some combining marks ("Maṅgal ̃ūru"): remove it, and join the letters (NFC)."""
    return nfc(re.sub(r"\s+(?=[\u0300-\u036f])", "", text))


def read_pdf(data: bytes) -> list[tuple[int, str]]:
    """[(page number, text)] for every page."""
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = []
        for n, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001  (one bad page doesn't stop the rest)
                text = ""
            pages.append((n, fix_pdf_text(text)))
    except ReadProblem:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ReadProblem("This file couldn't be opened as a PDF.") from exc
    if not any(re.search(r"[^\W\d_]{2}", t) for _, t in pages):
        raise ReadProblem("This PDF has no text inside (a scanned PDF): it can't be read for words.")
    return pages


def read_docx(data: bytes) -> str:
    """The text of a Word document: paragraphs and table cells, in document order."""
    import docx
    from docx.text.paragraph import Paragraph
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ReadProblem("This file couldn't be opened as a Word document (.docx).") from exc
    out, seen = [], set()

    def walk(container):
        for block in container.iter_inner_content():
            if isinstance(block, Paragraph):
                out.append(block.text)
                continue
            for row in block.rows:
                for cell in row.cells:
                    if id(cell._tc) not in seen:  # a merged cell is listed once per row/column it spans
                        seen.add(id(cell._tc))
                        walk(cell)
    walk(document)
    return "\n".join(out)


def decode_text(data: bytes) -> str:
    if data.startswith(codecs.BOM_UTF8):
        return data[3:].decode("utf-8", "replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", "replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", "replace")


def looks_like_list(text: str) -> bool:
    """True for a .txt of "plain,result" lines (the old Mappings page's download), not prose."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    def entry_like(line):
        plain, comma, result = line.partition(",")
        return (comma and plain.strip() and result.strip() and not has_marks(plain)
                and not re.search(r"\d", plain) and len(plain.split()) <= 4 and len(result.split()) <= 8)
    return sum(1 for ln in lines if entry_like(ln)) >= 0.8 * len(lines)


def read_upload(name: str, data: bytes) -> dict:
    """{'kind': 'document', 'pages': [(page or None, text)]} or {'kind': 'list', 'rows': [...], 'problems': [...]}.
    Raises ReadProblem with a message for people."""
    low = (name or "").lower()
    if len(data) > MAX_BYTES:
        raise ReadProblem(f"This file is over {MAX_BYTES // 1024 // 1024} MB.")
    if not data:
        raise ReadProblem("This file is empty.")
    if low.endswith(".pdf"):
        return {"kind": "document", "pages": read_pdf(data)}
    if low.endswith(".docx"):
        return {"kind": "document", "pages": [(None, read_docx(data))]}
    if low.endswith(".xlsx"):
        rows, problems = read_list_xlsx(data)
        return {"kind": "list", "rows": rows, "problems": problems}
    if low.endswith(".txt"):
        text = decode_text(data)
        if looks_like_list(text):
            rows, problems = read_list_lines(text)
            return {"kind": "list", "rows": rows, "problems": problems}
        return {"kind": "document", "pages": [(None, text)]}
    if low.endswith(".doc"):
        raise ReadProblem("This is an old Word file (.doc). Open it in Word and save it as .docx, then upload that.")
    if low.endswith((".xls", ".csv")):
        raise ReadProblem("Please save the list as an Excel workbook (.xlsx) and upload that.")
    raise ReadProblem("Only Word (.docx), PDF, text (.txt) and Excel (.xlsx) files can be read.")


# ---------- offline lists (.xlsx, or .txt lines "plain,result") ----------

PLAIN_HEADERS = ("plain_text", "plain word", "plain")
RESULT_HEADERS = ("team's answer", "teams answer", "answer", "diacritic_text", "result")


def _header(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\u2019", "'")).strip().lower()


def find_columns(rows: list[list]) -> tuple[int | None, int, int]:
    """(index of the header row or None, plain column, result column), from the headers in the first rows; without
    headers, the first two columns."""
    for i, row in enumerate(rows[:5]):
        heads = [_header(v) for v in row]
        plain = next((heads.index(h) for h in PLAIN_HEADERS if h in heads), None)
        result = next((heads.index(h) for h in RESULT_HEADERS if h in heads), None)
        if plain is not None and result is not None:
            return i, plain, result
    return None, 0, 1


def read_list_xlsx(data: bytes) -> tuple[list[dict], list[dict]]:
    from openpyxl import load_workbook
    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        wb.close()
    except Exception as exc:  # noqa: BLE001
        raise ReadProblem("This file couldn't be opened as an Excel workbook (.xlsx).") from exc
    head, pc, rc = find_columns(rows)
    out = []
    for n, row in enumerate(rows, 1):
        if head is not None and n <= head + 1:
            continue
        cell = lambda i: "" if i >= len(row) or row[i] is None else str(row[i])  # noqa: E731
        out.append((n, cell(pc), cell(rc)))
    return clean_list_rows(out)


def read_list_lines(text: str) -> tuple[list[dict], list[dict]]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        plain, _, result = line.partition(",")
        if n == 1 and _header(plain) in PLAIN_HEADERS:
            continue
        out.append((n, plain, result))
    return clean_list_rows(out)


def code_problem(code: str) -> str | None:
    """Why a stored code can't be used by the translator, or None."""
    if parse_code(code) is None:
        return "its {{ }} brackets don't make sense"
    if re.search(r"[^\w\u0300-\u036f\s{},()'\-.]", code):
        return "it has characters a word can't have"
    return None


NO_RESULT = "no result (left empty)"


def clean_list_rows(raw: list[tuple[int, str, str]]) -> tuple[list[dict], list[dict]]:
    """Rows of an offline list in the house style, and the rows skipped (with why). A plain word given twice keeps its
    first row."""
    rows, problems, seen = [], [], {}
    for n, plain_raw, result_raw in raw:
        plain, result = clean_plain(plain_raw), clean_result(result_raw)
        if not plain and not result:
            continue
        why = None
        if not plain:
            why = "no plain word"
        elif not result:
            why = NO_RESULT
        elif not re.fullmatch(r"[a-z0-9' \-]+", plain):
            why = "the plain word has letters our list can't use"
        elif code_problem(result):
            why = "the result: " + code_problem(result)
        elif plain in seen:
            if seen[plain]["result"] == result:
                continue
            why = f"the plain word is given again in row {seen[plain]['row']} with another result; that row is used"
        if why:
            problems.append({"row": n, "plain": plain_raw.strip(), "result": result_raw.strip(), "why": why})
            continue
        seen[plain] = {"row": n, "plain": plain, "result": result}
        rows.append(seen[plain])
    return rows, problems


# ---------- reviewing a document ----------

def split_camel(text: str, letters: set[str]) -> str:
    """PDF text sometimes runs two words together ("TempleShrīmath"): split where a small letter meets a capital."""
    out, prev_lower = [], False
    for ch in text:
        if ch.isupper() and prev_lower and ch.lower() in letters:
            out.append(" ")
        out.append(ch)
        if not unicodedata.combining(ch):
            prev_lower = ch.islower()
    return "".join(out)


def sentence_start(line: str, pos: int) -> bool:
    before = line[:pos].rstrip(" \u201c\u2018\"'(")
    return before == "" or before[-1] in ".!?:\u2022\u2013-\u2026"


_WORD = re.compile(r"[^\W\d_][\w\u0300-\u036f]*")
_ENGLISH = set("""and of in at for with from by is as his her our their on an or that this its not all was but has had
were are him them then when which what who whom during under after your my did can will shall must out up upon than
between through without within about into onto over both each such if so yet nor per via""".split())


def _should_join(a: str, b: str, known) -> bool:
    """Whether "a b" is one word that pypdf split with a space. In letter-spaced (justified) lines it puts spaces next
    to marked letters ("Sw āmījī", "Haṁsik ā", "Saṁ vit"), and after InDesign's ṭṭ ligature ("Viṭṭ al")."""
    def key(x):
        return clean_plain(plain_typed(x))
    ka, kb, kab = key(a), key(b), key(a + b)
    if not b[:1].islower():
        return False                                  # a word starting with a capital is a word of its own
    if kab in known and not (ka in known and kb in known):
        return True                                   # the joined word is in our list
    if len(strip_marks(b)) == 1 and has_marks(b):
        return True                                   # a lone marked letter: "sādhan ā"
    if len(a) == 1 and a not in "aAIO" and has_marks(b[:1]):
        return True                                   # "Y ātrā"
    if len(strip_marks(a)) <= 3 and a[-1:] in "ṁṛ" and ka not in known:
        return True                                   # "Saṁ vit", "Dṛ s͟hṭi"
    return (has_marks(b[:1]) and re.search("[bcdfghjklmnpqrstvwxyz]$", a) is not None and a.lower() not in _ENGLISH
            and ka not in known and kb not in known)  # "Nir ākāra", "adhik āra"


def join_split_words(line: str, known) -> str:
    """A PDF line with the words pypdf split put back together (see _should_join)."""
    words = list(_WORD.finditer(line))
    if len(words) < 2:
        return line
    out, pos, current = [], 0, None   # current: [start, text] of the word being built
    for m in words:
        if current and line[pos:m.start()] == " " and has_marks(current[1] + m.group(0)) \
                and _should_join(current[1], m.group(0), known):
            current[1] += m.group(0)
        else:
            if current:
                out.append(current[1])
                out.append(line[pos:m.start()])
            else:
                out.append(line[:m.start()])
            current = [m.start(), m.group(0)]
        pos = m.end()
    out.append(current[1])
    out.append(line[pos:])
    return "".join(out)


def _clean_line(line: str) -> str:
    for lig, letters in LIGATURES.items():
        line = line.replace(lig, letters)
    return nfc(line).strip()


def context(line: str, word: str, width: int = 170) -> str:
    """The line, cut down around the word when it's long."""
    if len(line) <= width:
        return line
    at = max(0, line.find(word))
    start = max(0, min(at - width // 3, len(line) - width))
    return ("…" if start else "") + line[start:start + width].strip() + ("…" if start + width < len(line) else "")


def why_odd(word: str) -> str:
    if "\u0128" in word or "\u0129" in word:
        return "Ĩ (capital I with a tilde) where l̃ (L with a tilde) is meant: they look the same in print"
    if "\u0342" in unicodedata.normalize("NFD", word):
        return "a Greek accent over the n instead of ñ: they look the same in print"
    if re.search("[\u019f\u01a9]", word):
        return "the PDF stores the joined letters 'ti'/'tt' as Ɵ/Ʃ: a PDF quirk, the printed word is fine"
    return "letters our list never uses (perhaps garbled in the PDF)"


def where(places, files) -> str:
    """Pages (and file names when there are several files) of a word's places [(file, page), ...]."""
    by_file = defaultdict(set)
    for f, p in places:
        by_file[f].add(p)
    parts = []
    for f in sorted(by_file):
        pages = ", ".join(str(p) for p in sorted(x for x in by_file[f] if x is not None))
        if len(files) > 1:
            parts.append(files[f] + (f" p. {pages}" if pages else ""))
        else:
            parts.append(pages)
    return "; ".join(p for p in parts if p)


def _row(form: str, result: str, places, names) -> dict:
    """One spelling of a word as a row of the report: the spelling as written, in the house style, how often, where,
    an example line. places: [(file, page, line, ...)]."""
    return {"form": form, "result": result, "times": len(places), "where": where([(p[0], p[1]) for p in places], names),
            "line": context(places[0][2], form) if places else ""}


def review_document(files: list[tuple[str, list[tuple[int | None, str]]]], mappings: dict[str, str]) -> dict:
    """The review of one or more documents. files: [(file name, [(page or None, text), ...])]. Returns
    {'kind': 'document', 'files': [...], 'sections': {name: [items]}, 'counts': {...}}; every item that needs a
    choice has an 'id'."""
    letters = list_letters(mappings)
    keys = set(mappings)
    names = [f for f, _ in files]
    occ = defaultdict(list)       # form -> [(file, page, line, at the start of a sentence)]
    odd = defaultdict(list)       # form -> [(file, page, line)]
    garbled = []
    # phrase entries of the list (two or more words), by their first word; where one is found, it wins over its words
    phrases = defaultdict(list)
    for k in mappings:
        if " " in k.strip():
            phrases[k.split()[0]].append(k.split())
    for first in phrases:
        phrases[first].sort(key=len, reverse=True)
    phrase_hits = {}              # the phrase as written -> {places, line, key}
    for fi, (name, pages) in enumerate(files):
        from_pdf = name.lower().endswith(".pdf")
        for page, text in pages:
            for raw in text.splitlines():
                line = _clean_line(raw)
                if not line:
                    continue
                if from_pdf:
                    line = join_split_words(line, keys)
                if DEVANAGARI.search(line):
                    latin = [m.group(0) for m in TOKEN.finditer(line)
                             if not DEVANAGARI.search(m.group(0)) and has_marks(m.group(0))]
                    if any(set(house(w).lower()) - letters for w in latin):
                        garbled.append((fi, page, line))  # an Indian-script font that doesn't copy out correctly
                        continue
                line = split_camel(line, letters)
                toks = list(TOKEN.finditer(line))
                in_phrase = set()
                if phrases:
                    low = [plain_typed(t.group(0)).lower() for t in toks]
                    i = 0
                    while i < len(toks):
                        ph = next((ph for ph in phrases.get(low[i], ()) if low[i:i + len(ph)] == ph), None)
                        if ph is None:
                            i += 1
                            continue
                        doc = " ".join(t.group(0) for t in toks[i:i + len(ph)])
                        hit = phrase_hits.setdefault(doc, {"places": [], "line": line, "key": " ".join(ph)})
                        hit["places"].append((fi, page))
                        in_phrase.update(range(i, i + len(ph)))
                        i += len(ph)
                for n, m in enumerate(toks):
                    if n in in_phrase:
                        continue
                    whole = re.sub("['\u2019]s$", "", m.group(0))
                    parts = [whole]
                    # a hyphenated word without an entry of its own is looked at part by part; "-s" stays with its word
                    if re.search("[-\u2010\u2011]", whole) and phrase_key(plain_typed(whole)) not in keys:
                        parts = re.findall("[^-\u2010\u2011]+(?:[-\u2010\u2011]s(?![^-\u2010\u2011]))?", whole)
                    for j, w in enumerate(parts):
                        if DEVANAGARI.search(w) or len(w) < 2:
                            continue
                        if not has_marks(w) and plain_typed(w).lower() not in keys:
                            continue
                        if set(house(w).lower()) - letters:
                            odd[w].append((fi, page, line))
                            continue
                        occ[w].append((fi, page, line, j == 0 and sentence_start(line, m.start())))

    already, style, unmarked, doubtful = [], [], {}, []
    clash_forms = defaultdict(list)   # plain key -> forms that differ from the list
    other_forms = defaultdict(list)   # plain key -> forms that are the same as the list, or style only
    new_groups = {}
    for w, places in sorted(occ.items(), key=lambda x: (plain_typed(x[0]).lower(), x[0])):
        typed = plain_typed(w)
        got = translate_text(typed, mappings)
        key = clean_plain(typed)
        base = {"form": w, "plain": key, "times": len(places), "where": where([(f, p) for f, p, _, _ in places], names),
                "line": context(places[0][2], w), "gives": got}
        if not has_marks(w):
            if got != typed:
                u = unmarked.setdefault(key, {"plain": key, "forms": [], "places": [], "gives": got,
                                              "line": context(places[0][2], w), "mark": w})
                u["forms"].append(w)
                u["places"] += [(f, p) for f, p, _, _ in places]
            continue
        hw = house(w)
        if got == w or w in choices(got):
            already.append(base)
            other_forms[key].append({"form": w, "times": len(places), "status": "same as our list", "gives": got})
        elif hw == got or hw in choices(got):
            nfd = unicodedata.normalize("NFD", w)
            why = ("underline instead of the tie" if "\u0332" in nfd else
                   "Greek look-alike letter" if any(g in w for g in GREEK_LOOKALIKES) else "letters stored differently")
            style.append(base | {"why": why})
            other_forms[key].append({"form": w, "times": len(places), "status": "style only", "gives": got})
        elif got == typed and key not in mappings and not any(has_marks(c) for c in choices(got)):
            g = new_groups.setdefault(key, {"forms": Counter(), "places": [], "greek": False,
                                            "by_form": defaultdict(list)})
            g["forms"][hw] += len(places)
            g["places"] += places
            g["by_form"][hw] += places
            g["greek"] |= any(x in w for x in GREEK_LOOKALIKES)
        else:
            only_capital = got.lower() == hw.lower() or hw.lower() in [c.lower() for c in choices(got)]
            clash_forms[key].append({"form": w, "times": len(places), "status": "differs", "gives": got,
                                     "places": places, "capital_only": only_capital})

    clashes = []
    for key in sorted(clash_forms):
        forms = sorted(clash_forms[key], key=lambda f: -f["times"])
        first = forms[0]
        by_result = {}               # one row per spelling (in the house style), most used first
        for f in forms:
            by_result.setdefault(clean_result(house(f["form"])), []).append(f)
        rows = [_row(fs[0]["form"], r, [p for f in fs for p in f["places"]], names) for r, fs in by_result.items()]
        places = [p for f in forms for p in f["places"]]
        clashes.append({
            "kind": "clash", "plain": key, "entry": mappings.get(key), "options": list(by_result), "rows": rows,
            "forms": [{k: f[k] for k in ("form", "times", "status", "gives")} for f in forms]
            + other_forms.get(key, []),
            "form": first["form"], "capitalised": first["form"][:1].isupper(), "gives": first["gives"],
            "times": sum(f["times"] for f in forms), "where": where([(fi, p) for fi, p, _, _ in places], names),
            "line": context(first["places"][0][2], first["form"]),
            "note": "only the capital letter differs" if all(f["capital_only"] for f in forms) else "",
        })

    # phrase entries of the list found in the text: compared as a whole
    for doc, hit in sorted(phrase_hits.items()):
        got = translate_text(plain_typed(doc), mappings)
        key = hit["key"]
        base = {"form": doc, "plain": key, "times": len(hit["places"]), "where": where(hit["places"], names),
                "line": context(hit["line"], doc), "gives": got}
        if got == doc or doc in choices(got) or house(doc) == got or house(doc) in choices(got):
            already.append(base | {"phrase": True})
        else:
            clashes.append({"kind": "clash", "plain": key, "entry": mappings.get(key),
                            "options": [clean_result(house(doc))], "phrase": True,
                            "rows": [_row(doc, clean_result(house(doc)),
                                          [(f, p, hit["line"]) for f, p in hit["places"]], names)],
                            "forms": [{"form": doc, "times": len(hit["places"]), "status": "differs", "gives": got}],
                            "form": doc, "capitalised": doc[:1].isupper(), "gives": got, "times": base["times"],
                            "where": base["where"], "line": base["line"], "note": "a phrase entry of our list"})

    new = []
    for key, g in sorted(new_groups.items()):
        places = g["places"]
        by_result = {}               # one row per spelling (in the house style), most used first
        for form, _ in g["forms"].most_common():
            by_result.setdefault(clean_result(form), []).append(form)
        rows = [_row(fs[0], r, [p for f in fs for p in g["by_form"][f]], names) for r, fs in by_result.items()]
        form = g["forms"].most_common(1)[0][0]
        small = any(not f[:1].isupper() for f in g["forms"])
        mid_capital = not small and any(not start for (_, _, _, start) in places)
        seen = "small" if small else "name" if mid_capital else "start"
        new.append({"kind": "new", "plain": key, "form": form, "options": list(by_result), "rows": rows,
                    "times": len(places), "spellings": [f for f, _ in g["forms"].most_common()],
                    "where": where([(f, p) for f, p, _, _ in places], names),
                    "line": context(places[0][2], form), "seen": seen, "greek": g["greek"]})

    unmarked_items = []
    for key, u in sorted(unmarked.items()):
        unmarked_items.append({"kind": "unmarked", "plain": key, "form": ", ".join(u["forms"]), "mark": u["mark"],
                               "forms": u["forms"], "times": len(u["places"]), "where": where(u["places"], names),
                               "line": u["line"], "gives": u["gives"], "entry": mappings.get(key)})

    for w, v in odd.items():
        doubtful.append({"form": w, "times": len(v), "where": where([(f, p) for f, p, _ in v], names),
                         "line": context(v[0][2], w), "why": why_odd(w)})
    for f, p, line in garbled:
        doubtful.append({"form": f"page {p}" if p else names[f], "times": 1, "where": where([(f, p)], names),
                         "line": line[:170], "why": "Indian-script text whose font doesn't copy out correctly"})

    sections = group_words({"clash": clashes, "new": new, "unmarked": unmarked_items, "style": style,
                            "doubtful": doubtful, "already": already})
    _number(sections)
    return {"kind": "document", "sections": sections,
            "counts": {k: len(v) for k, v in sections.items()},
            "words_with_marks": sum(1 for w in occ if has_marks(w)), "grouped": GROUPED}


# ---------- one row per plain word ----------

GROUPED = 2      # review['grouped']: the items are one per plain word (reviews made before it are regrouped on reading)


def merge_where(wheres: list[str]) -> str:
    """The pages of several where() texts together: '3, 12' + '12, 40' -> '3, 12, 40'; with file names
    ('a.pdf p. 3; b.docx') per file."""
    by_file, order = {}, []
    for text in wheres:
        for part in (p.strip() for p in str(text or "").split("; ")):
            if not part:
                continue
            if re.fullmatch(r"[\d, ]+", part):
                name, pages = "", part
            elif " p. " in part:
                name, _, pages = part.rpartition(" p. ")
            else:
                name, pages = part, ""
            if name not in by_file:
                by_file[name] = set()
                order.append(name)
            by_file[name] |= {int(x) for x in re.findall(r"\d+", pages)}
    parts = []
    for name in order:
        pages = ", ".join(str(p) for p in sorted(by_file[name]))
        parts.append(f"{name} p. {pages}" if name and pages else name or pages)
    return "; ".join(p for p in parts if p)


def old_refs(item: dict) -> dict[str, int]:
    """The Refs a report made before one row per word gave this item's spellings: {'c3:0': option 0, ...}."""
    if item.get("old_refs") is not None:
        return item["old_refs"]
    return {f"{item['id']}:{k}": k for k in range(len(item.get("rows") or [item]))} if item.get("id") else {}


def _same(item: dict, status: str) -> dict:
    """A spelling of the document that matches our list (an 'already' or 'style' item), as part of its word's row."""
    return {"form": item["form"], "times": item.get("times", 0), "where": item.get("where", ""),
            "line": item.get("line", ""), "status": status, "why": item.get("why", ""), "gives": item.get("gives", "")}


def group_words(sections: dict) -> dict:
    """One item per plain word on 'Different from our list' and 'New words': clash items with the same plain word
    (a phrase typed two ways) are merged, a new word with the plain word of a clash joins the clash, and the
    spellings that match our list ('already', 'style') of a word with a row are moved into that row ('same').
    Pure: returns new sections (the items copied); ids are kept, and each item remembers the Refs its spellings
    had in a report made before (old_refs), so old filled files can still be read."""
    import copy
    s = copy.deepcopy(sections)
    rows_by_plain, clash, new = {}, [], []

    def start(item):
        if item.get("id") and item.get("old_refs") is None:
            item["old_refs"] = old_refs(item)
        item.setdefault("rows", [])
        item.setdefault("same", [])
        return item

    def absorb(into, item):
        """item's spellings become spellings of `into` (same plain word)."""
        offset = {}
        for k, (opt, row) in enumerate(zip(item.get("options") or [], item.get("rows") or [])):
            if opt in into["options"]:
                j = into["options"].index(opt)
                r = into["rows"][j]
                r["times"] = r.get("times", 0) + row.get("times", 0)
                r["where"] = merge_where([r.get("where", ""), row.get("where", "")])
            else:
                j = len(into["options"])
                into["options"].append(opt)
                into["rows"].append(row)
            offset[k] = j
        refs = old_refs(item)
        if refs:
            into.setdefault("old_refs", {}).update({ref: offset.get(k, 0) for ref, k in refs.items()})
        into["same"] += item.get("same") or []
        into["forms"] = (into.get("forms") or []) + [f for f in item.get("forms") or []
                                                     if f.get("status") == "differs"]
        if item.get("spellings"):
            into["spellings"] = list(dict.fromkeys((into.get("spellings") or []) + item["spellings"]))
        if into.get("note") != item.get("note"):
            into["note"] = ""

    for item in s.get("clash") or []:
        if item["plain"] in rows_by_plain:
            absorb(rows_by_plain[item["plain"]], item)
        else:
            rows_by_plain[item["plain"]] = start(item)
            clash.append(item)
    for item in s.get("new") or []:
        if item["plain"] in rows_by_plain:
            into = rows_by_plain[item["plain"]]
            if into["kind"] == "clash":       # spelt as a new word somewhere, differently from our list elsewhere
                item = dict(item, forms=[{"form": r["form"], "times": r.get("times", 0), "status": "differs",
                                          "gives": r["form"]} for r in item.get("rows") or []])
            absorb(into, item)
        else:
            rows_by_plain[item["plain"]] = start(item)
            new.append(item)
    for name, status in (("already", "same as our list"), ("style", "style only")):
        kept = []
        for item in s.get(name) or []:
            into = rows_by_plain.get(item["plain"])
            if into is None:
                kept.append(item)
            elif item["form"] not in [x["form"] for x in into["same"]]:
                into["same"].append(_same(item, status))
        s[name] = kept
    for item in clash + new:
        same = item["same"]
        differing = [f for f in item.get("forms") or [] if f.get("status") == "differs"]
        item["forms"] = differing + [{"form": x["form"], "times": x["times"], "status": x["status"],
                                      "gives": x.get("gives", "")} for x in same]
        item["times"] = sum(r.get("times", 0) for r in item["rows"]) + sum(x["times"] for x in same)
        item["where"] = merge_where([r.get("where", "") for r in item["rows"]] + [x["where"] for x in same])
        if item["rows"]:
            item["form"], item["line"] = item["rows"][0]["form"], item["rows"][0].get("line", item.get("line", ""))
    s["clash"], s["new"] = clash, new
    return s


def regroup(review: dict) -> dict:
    """A stored review in the layout of one row per plain word (a review made before it is regrouped; the stored
    copy isn't changed). Reviews of lists and grouped reviews are returned as they are."""
    if not review or review.get("kind") != "document" or (review.get("grouped") or 0) >= GROUPED:
        return review
    sections = group_words(review["sections"])
    return dict(review, sections=sections, counts={k: len(v) for k, v in sections.items()}, grouped=GROUPED)


def _number(sections: dict) -> None:
    prefix = {"clash": "c", "new": "n", "unmarked": "u", "style": "s", "doubtful": "d", "already": "a",
              "lnew": "ln", "lchanged": "lc", "lmissing": "lm", "problems": "p", "empty": "le", "unchanged": "lu"}
    for name, items in sections.items():
        for i, item in enumerate(items, 1):
            item["id"] = f"{prefix[name]}{i}"


# ---------- reviewing an offline list ----------

def review_list(rows: list[dict], problems: list[dict], mappings: dict[str, str]) -> dict:
    """New, changed, missing from the file (only when the file is a whole list: it has at least half of our list's
    words; a shorter file is additions and changes), problems, unchanged."""
    new, changed, unchanged = [], [], []
    in_file = set()
    for r in rows:
        in_file.add(r["plain"])
        old = mappings.get(r["plain"])
        if old is None:
            new.append({"kind": "lnew", "plain": r["plain"], "result": r["result"], "row": r["row"]})
        elif clean_result(old) != r["result"]:
            changed.append({"kind": "lchanged", "plain": r["plain"], "result": r["result"], "entry": old,
                            "row": r["row"]})
        else:
            unchanged.append({"plain": r["plain"], "result": r["result"]})
    whole = len(in_file & set(mappings)) >= 0.5 * max(1, len(mappings))
    missing = [{"kind": "lmissing", "plain": k, "entry": v} for k, v in sorted(mappings.items())
               if k not in in_file] if whole else []
    empty = [p for p in problems if p["why"] == NO_RESULT]   # e.g. pending words the team hasn't answered yet
    sections = {"lnew": new, "lchanged": changed, "lmissing": missing,
                "problems": [p for p in problems if p["why"] != NO_RESULT], "empty": empty, "unchanged": unchanged}
    _number(sections)
    return {"kind": "list", "sections": sections, "counts": {k: len(v) for k, v in sections.items()},
            "whole_list": whole, "rows": len(rows)}


# ---------- stored codes ----------

def parse_code(code: str) -> dict | None:
    """{'type': normal|capital|name|choices, 'main': group, 'name': group} or None if the brackets don't make sense."""
    code = code or ""
    groups = re.findall(r"\{\{(.*?)\}\}", code)
    if not groups:
        return None if "{" in code or "}" in code else {"type": "normal", "main": code}
    if re.sub(r"\{\{.*?\}\}", "", code).strip() or any("{" in g or "}" in g for g in groups):
        return None
    if len(groups) == 1:
        return {"type": "choices", "main": groups[0]}
    if len(groups) == 2:
        if not groups[0].strip():
            return {"type": "capital", "main": groups[1]}
        return {"type": "name", "main": groups[0], "name": groups[1]}
    return None


def with_type(code: str, cap: str) -> str:
    """The code with its type set: cap "normal" or "capital" ("keep" or anything else leaves it)."""
    p = parse_code(code)
    if p is None or cap not in ("normal", "capital"):
        return code
    main = p["main"]
    if cap == "capital":
        return clean_result("{{}}{{" + main + "}}")
    return clean_result("{{" + main + "}}" if CHOICE_COMMA.search(main) else main)


def several(spellings: list[str], cap: str | None, entry: str | None = None) -> str:
    """Several spellings as choices, {{a, b (label)}}; with cap "capital" {{}}{{a, b}}. Without a capital chosen,
    an entry that is always capital stays so."""
    code = clean_result("{{" + ", ".join(spellings) + "}}")
    if cap not in ("normal", "capital"):
        p = parse_code(entry) if entry else None
        cap = "capital" if p and p["type"] == "capital" else "normal"
    return with_type(code, cap)


def replace_result(code: str | None, result: str, capitalised: bool) -> str:
    """The entry with this spelling in place of what our list gives for that capitalisation, keeping its type: always
    capital stays always capital; for a Name entry the group used for that capitalisation is replaced."""
    p = parse_code(code) if code else None
    if p is None or parse_code(result) is None or "{" in result:
        return result
    if p["type"] == "capital":
        return clean_result("{{}}{{" + result + "}}")
    if p["type"] == "name":
        if capitalised:
            return clean_result("{{" + p["main"] + "}}{{" + result + "}}")
        return clean_result("{{" + result + "}}{{" + p["name"] + "}}")
    return result


def split_choices(group: str) -> list[str]:
    # a comma inside a label's brackets, "āratī (hindi, marathi)", stays in its choice (as the translator)
    return [c for c in (re.sub(r"\s+", " ", x).strip() for x in CHOICE_COMMA.split(group or "")) if c]


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _capital_words(text: str) -> str:
    return re.sub(r"(?<![^\s,])([^\W\d_])", lambda m: m.group(1).upper(), text)


def describe(code: str) -> str:
    """A stored code in words: 'Kṛs͟hṇa (always capital)', 'gŏḍ / with a capital: God',
    'choice: nāmāvalī (hindi) or nāmāvaliḥ (sanskrit)'."""
    p = parse_code(code or "")
    if p is None:
        return code or ""

    def show(group, cap=None):
        cs = split_choices(group)
        if cap:
            cs = [cap(c) for c in cs]
        return cs[0] if len(cs) == 1 else "choice: " + " or ".join(cs)
    if p["type"] == "normal":
        return code
    if p["type"] == "capital":
        return f"{show(p['main'], _capital_words)} (always capital)"
    if p["type"] == "name":
        return f"{show(p['main'])} / with a capital: {show(p['name'], _capital)}"
    return show(p["main"])


def small_result(code: str) -> str:
    """The word an entry stands for, in small letters (for counting distinct diacritical words)."""
    p = parse_code(code or "")
    return (p["main"] if p else code or "").strip().lower()
