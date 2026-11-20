"""Words with diacritical marks in an uploaded file, compared with the Diacritical word list: the review behind the
dashboard's Add words page (admin_words.py). Pure functions: no database, no web.

- Reading: read_upload(name, data) tells a document (.docx, .pdf, .txt) from an offline word list (.xlsx, or a .txt
  of "plain,result" lines) and reads it. A PDF is read page by page with pypdf.
- Document review: review_document(files, mappings) finds every word with marks (and every plain word the list would
  mark) and sorts them into: different from our list (clash), new, written without marks, style only, doubtful
  (look-alike letters, PDF quirks), already in the list. Phrase entries of the list found in the text are compared.
- List review: review_list(rows, problems, mappings): new, changed, missing from the file, problems, unchanged.
- plan(review, decisions): what Accept does (adds, updates, deletes, pending items), and what is still undecided.
- Entries: the stored code (diacritic_text) <-> the plain boxes of the edit page; describe(code) for people.
- Spreadsheets: the review report, the whole list and the pending list as .xlsx.

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
from .diacritics import phrase_key, translate_text

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
    return [re.sub(r"\s*\(.*?\)", "", x).strip() for x in m.group(1).split(",")]


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
            g = new_groups.setdefault(key, {"forms": Counter(), "places": [], "greek": False, "first": None})
            g["forms"][hw] += len(places)
            g["places"] += places
            g["greek"] |= any(x in w for x in GREEK_LOOKALIKES)
        else:
            only_capital = got.lower() == hw.lower() or hw.lower() in [c.lower() for c in choices(got)]
            clash_forms[key].append({"form": w, "times": len(places), "status": "differs", "gives": got,
                                     "places": places, "capital_only": only_capital})

    clashes = []
    for key in sorted(clash_forms):
        forms = sorted(clash_forms[key], key=lambda f: -f["times"])
        first = forms[0]
        options = []
        for f in forms:
            r = clean_result(house(f["form"]))
            if r not in options:
                options.append(r)
        places = [p for f in forms for p in f["places"]]
        clashes.append({
            "kind": "clash", "plain": key, "entry": mappings.get(key), "options": options,
            "forms": [{k: f[k] for k in ("form", "times", "status", "gives")} for f in forms] + other_forms.get(key, []),
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
                            "forms": [{"form": doc, "times": len(hit["places"]), "status": "differs", "gives": got}],
                            "form": doc, "capitalised": doc[:1].isupper(), "gives": got, "times": base["times"],
                            "where": base["where"], "line": base["line"], "note": "a phrase entry of our list"})

    new = []
    for key, g in sorted(new_groups.items()):
        places = g["places"]
        options = []
        for form, _ in g["forms"].most_common():
            r = clean_result(form)
            if r not in options:
                options.append(r)
        form = g["forms"].most_common(1)[0][0]
        small = any(not f[:1].isupper() for f in g["forms"])
        mid_capital = not small and any(not start for (_, _, _, start) in places)
        seen = "small" if small else "name" if mid_capital else "start"
        new.append({"kind": "new", "plain": key, "form": form, "options": options, "times": len(places),
                    "spellings": [f for f, _ in g["forms"].most_common()],
                    "where": where([(f, p) for f, p, _, _ in places], names),
                    "line": context(places[0][2], form), "seen": seen, "greek": g["greek"],
                    "extras": suggestions(key, options[0], mappings)})

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

    sections = {"clash": clashes, "new": new, "unmarked": unmarked_items, "style": style, "doubtful": doubtful,
                "already": already}
    _number(sections)
    return {"kind": "document", "sections": sections,
            "counts": {k: len(v) for k, v in sections.items()},
            "words_with_marks": sum(1 for w in occ if has_marks(w))}


def suggestions(plain: str, result: str, mappings: dict[str, str]) -> list[dict]:
    """Shantish's old rules, offered (unticked) on a new word's row: for "laddu-s" also "laddus" (same result) and the
    singular "laddu" -> laḍḍū; for another hyphenated word also the spaced form."""
    out = []
    if re.search(r"-s$", plain) and result.endswith("-s"):
        joined, single, single_result = plain[:-2] + "s", plain[:-2], result[:-2]
        if joined not in mappings:
            out.append({"plain": joined, "result": result})
        if single and single not in mappings and has_marks(single_result):
            out.append({"plain": single, "result": single_result})
    elif "-" in plain:
        spaced = plain.replace("-", " ")
        if spaced not in mappings:
            out.append({"plain": spaced, "result": result})
    return out


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


# ---------- choices and what Accept does ----------

# The choices each kind of row offers: (value, label). Document clashes and new words get their options from the item.
NEEDS_CHOICE = ("clash", "new", "lchanged")


def choice_options(item: dict) -> list[tuple[str, str]]:
    kind = item["kind"]
    if kind == "clash":
        docs = [(f"doc:{i}", "The document is right" if len(item["options"]) == 1 else f"The document is right: {r}")
                for i, r in enumerate(item["options"])]
        return [("list", "Our list is right")] + docs + [("pending", "Pending")]
    if kind == "new":
        adds = [] if item.get("now_in_list") else [
            (f"add:{i}", "Add" if len(item["options"]) == 1 else f"Add {r}") for i, r in enumerate(item["options"])]
        return adds + [("pending", "Pending"), ("skip", "Skip")]
    if kind == "unmarked":
        return [("none", "Nothing to change"), ("pending", "Pending")]
    if kind == "lnew":
        return ([] if item.get("now_in_list") else [("add", "Add")]) + [("pending", "Pending"), ("skip", "Skip")]
    if kind == "lchanged":
        return [("keep", "Keep our list"), ("use", "Use the file"), ("pending", "Pending")]
    if kind == "lmissing":
        return [("keep", "Keep"), ("delete", "Delete")]
    return []


CAPITAL_OPTIONS = {"clash": [("keep", "Keep as it is"), ("normal", "Normal"), ("capital", "Always capital")],
                   "new": [("normal", "Normal"), ("capital", "Always capital")]}


def default_decisions(review: dict) -> dict:
    """The choices a new review starts with: rows that need a decision start empty (Always capital is preselected for
    a new word only ever seen with a capital mid-sentence); the others start with the harmless choice."""
    out = {}
    for items in review["sections"].values():
        for item in items:
            kind = item.get("kind")
            if kind == "clash":
                out[item["id"]] = {"cap": "keep"}
            elif kind == "new":
                out[item["id"]] = {"cap": "capital" if item["seen"] == "name" else "normal"}
            elif kind == "unmarked":
                out[item["id"]] = {"choice": "none"}
            elif kind == "lnew":
                out[item["id"]] = {"choice": "add"}
            elif kind == "lmissing":
                out[item["id"]] = {"choice": "keep"}
    return out


def undecided(review: dict, decisions: dict) -> list[str]:
    valid = {}
    out = []
    for items in review["sections"].values():
        for item in items:
            if item.get("kind") in NEEDS_CHOICE:
                valid = {v for v, _ in choice_options(item)}
                if (decisions.get(item["id"]) or {}).get("choice") not in valid:
                    out.append(item["id"])
    return out


def with_type(code: str, cap: str) -> str:
    """The code with its type set: cap "normal" or "capital" ("keep" leaves it)."""
    p = parse_code(code)
    if p is None or cap not in ("normal", "capital"):
        return code
    main = p["main"]
    if cap == "capital":
        return clean_result("{{}}{{" + main + "}}")
    return clean_result("{{" + main + "}}" if "," in main else main)


def replace_result(code: str | None, result: str, capitalised: bool) -> str:
    """The entry with the document's spelling in place of what our list gives for that capitalisation, keeping its
    type: always capital stays always capital; for a Name entry the group used for that capitalisation is replaced."""
    p = parse_code(code) if code else None
    if p is None:
        return result
    if p["type"] == "capital":
        return clean_result("{{}}{{" + result + "}}")
    if p["type"] == "name":
        if capitalised:
            return clean_result("{{" + p["main"] + "}}{{" + result + "}}")
        return clean_result("{{" + result + "}}{{" + p["name"] + "}}")
    return result


def plan(review: dict, decisions: dict, source: str = "") -> dict:
    """What Accept does with these choices: {'add': [{plain, result, item}], 'update': [{plain, old, result, item}],
    'delete': [{plain, old, item}], 'pending': [{plain, suggested, reason, source, page, context, item}],
    'resolve': [plain words the file confirms as they are], 'undecided': [item ids], 'clashing': [plain words two
    rows want to change differently]}."""
    out = {"add": [], "update": [], "delete": [], "pending": [], "resolve": [], "undecided": undecided(review, decisions),
           "clashing": []}
    s = review["sections"]

    def pend(item, suggested, reason):
        out["pending"].append({"plain": item["plain"], "suggested": suggested, "reason": reason, "source": source,
                               "page": item.get("where", ""), "context": item.get("line", ""), "item": item["id"]})

    for item in s.get("clash", []):
        d = decisions.get(item["id"]) or {}
        choice, cap = d.get("choice"), d.get("cap", "keep")
        entry = item.get("entry")
        if choice == "pending":
            # suggested: the document's spelling, in the entry's own form (as "The document is right" would store it)
            pend(item, with_type(replace_result(entry, item["options"][0], item.get("capitalised", False)), cap)
                 if entry else item["options"][0],
                 f"Different from our list: the document has {', '.join(f['form'] for f in item['forms'] if f['status'] == 'differs')}; "
                 f"our list gives {item['gives']}")
        elif choice == "list" and entry is not None:
            new = with_type(entry, cap)
            if new != entry:
                out["update"].append({"plain": item["plain"], "old": entry, "result": new, "item": item["id"]})
        elif choice and choice.startswith("doc:"):
            result = item["options"][int(choice[4:])]
            if entry is None:
                out["add"].append({"plain": item["plain"], "result": with_type(result, cap if cap != "keep" else "normal"),
                                   "item": item["id"]})
            else:
                new = with_type(replace_result(entry, result, item.get("capitalised", False)), cap)
                if new != entry:
                    out["update"].append({"plain": item["plain"], "old": entry, "result": new, "item": item["id"]})
    for item in s.get("new", []):
        d = decisions.get(item["id"]) or {}
        choice, cap = d.get("choice"), d.get("cap", "normal")
        if choice == "pending":
            pend(item, with_type(item["options"][0], cap), "New word: not in our list")
        elif choice and choice.startswith("add:"):
            result = item["options"][int(choice[4:])]
            out["add"].append({"plain": item["plain"], "result": with_type(result, cap), "item": item["id"]})
            for i in d.get("extras", []):
                if 0 <= i < len(item.get("extras", [])):
                    x = item["extras"][i]
                    out["add"].append({"plain": x["plain"], "result": with_type(x["result"], cap), "item": item["id"]})
    for item in s.get("unmarked", []):
        if (decisions.get(item["id"]) or {}).get("choice") == "pending":
            pend(item, item.get("entry") or "", f"Written without marks in the document ({item['form']}); "
                                                f"our list gives {item['gives']}")
    for item in s.get("lnew", []):
        choice = (decisions.get(item["id"]) or {}).get("choice")
        if choice == "add":
            out["add"].append({"plain": item["plain"], "result": item["result"], "item": item["id"]})
        elif choice == "pending":
            pend(item | {"where": f"row {item['row']}", "line": ""}, item["result"], "New in the uploaded list")
    for item in s.get("lchanged", []):
        choice = (decisions.get(item["id"]) or {}).get("choice")
        if choice == "use" and item.get("entry") is None:  # deleted from the list meanwhile
            out["add"].append({"plain": item["plain"], "result": item["result"], "item": item["id"]})
        elif choice == "use":
            out["update"].append({"plain": item["plain"], "old": item["entry"], "result": item["result"],
                                  "item": item["id"]})
        elif choice == "pending":
            pend(item | {"where": f"row {item['row']}", "line": ""}, item["result"],
                 f"Changed in the uploaded list; our list has {describe(item['entry'])}")
        elif choice == "keep":
            out["resolve"].append(item["plain"])
    for item in s.get("lmissing", []):
        if (decisions.get(item["id"]) or {}).get("choice") == "delete" and item.get("entry") is not None:
            out["delete"].append({"plain": item["plain"], "old": item["entry"], "item": item["id"]})
    out["resolve"] += [u["plain"] for u in s.get("unchanged", [])]

    targets = defaultdict(set)
    for op in ("add", "update", "delete"):
        for c in out[op]:
            targets[c["plain"]].add((op, c.get("result")))
    out["clashing"] = sorted(p for p, t in targets.items() if len(t) > 1)
    for op in ("add", "update", "delete"):  # the same change asked twice (an extra that is also its own row) is one
        seen, kept = set(), []
        for c in out[op]:
            if c["plain"] not in seen:
                seen.add(c["plain"])
                kept.append(c)
        out[op] = kept
    return out


def plan_sentence(p: dict, done: bool = False) -> str:
    """'Accept will add 3 words, change 2, and put 130 on the pending list.' (done: 'Accepted: added 3 words, …')"""
    def n(k, one, many):
        return f"{k} {one if k == 1 else many}"
    verbs = ("added", "changed", "deleted", "put") if done else ("add", "change", "delete", "put")
    parts = []
    if p["add"]:
        parts.append(f"{verbs[0]} {n(len(p['add']), 'word', 'words')}")
    if p["update"]:
        parts.append(f"{verbs[1]} {len(p['update'])}")
    if p["delete"]:
        parts.append(f"{verbs[2]} {len(p['delete'])}")
    if p["pending"]:
        parts.append(f"{verbs[3]} {len(p['pending'])} on the pending list")
    if not parts:
        return "Accepted: nothing in the list changed." if done else "Accept will change nothing in the list."
    joined = parts[0] if len(parts) == 1 else " and ".join(parts) if len(parts) == 2 else \
        ", ".join(parts[:-1]) + ", and " + parts[-1]
    return ("Accepted: " if done else "Accept will ") + joined + "."


# ---------- entries: stored code <-> plain boxes ----------

TYPES = {"normal": "Normal", "capital": "Always capital", "name": "Name (different when typed with a capital)",
         "choices": "Choices"}


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


def split_choices(group: str) -> list[str]:
    return [c for c in (re.sub(r"\s+", " ", x).strip() for x in (group or "").split(",")) if c]


def boxes_to_code(boxes: dict) -> str:
    """The stored code (house style) from the edit page's boxes; the Advanced box, when filled, wins."""
    if (boxes.get("advanced") or "").strip():
        return clean_result(boxes["advanced"])
    kind = boxes.get("type") or "normal"
    result = (boxes.get("result") or "").strip()
    if kind == "capital":
        code = "{{}}{{" + result + "}}"
    elif kind == "name":
        code = "{{" + result + "}}{{" + (boxes.get("name") or "").strip() + "}}"
    elif kind == "choices":
        code = "{{" + ", ".join(c.strip() for c in boxes.get("choices") or [] if c.strip()) + "}}"
    else:
        code = result
    return clean_result(code)


def code_to_boxes(code: str) -> dict:
    """The edit page's boxes for a stored code. 'advanced' holds the code when the boxes can't show it exactly
    (e.g. always capital with choices); then that box is what gets saved."""
    boxes = {"type": "normal", "result": "", "name": "", "choices": [], "advanced": ""}
    p = parse_code(code)
    if p is not None:
        main = split_choices(p["main"])
        if p["type"] == "normal" and "," not in code:
            boxes["result"] = code.strip()
        elif p["type"] == "capital" and len(main) == 1:
            boxes.update(type="capital", result=main[0])
        elif p["type"] == "name" and len(main) == 1 and len(split_choices(p["name"])) == 1:
            boxes.update(type="name", result=main[0], name=split_choices(p["name"])[0])
        elif p["type"] == "choices" and len(main) >= 2:
            boxes.update(type="choices", choices=main)
    if p is None or boxes_to_code(boxes) != clean_result(code):
        boxes = {"type": "normal", "result": "", "name": "", "choices": [], "advanced": code}
    return boxes


def box_problems(boxes: dict) -> list[str]:
    """What's wrong with the boxes, in plain words (empty when they can be saved)."""
    if (boxes.get("advanced") or "").strip():
        why = code_problem(clean_result(boxes["advanced"]))
        return [f"The stored code can't be used: {why}."] if why else []
    kind = boxes.get("type") or "normal"
    out = []
    if kind == "choices":
        if len([c for c in boxes.get("choices") or [] if c.strip()]) < 2:
            out.append("Choices needs at least two spellings, one per line.")
        if any(re.search("[{}]", c) for c in boxes.get("choices") or []):
            out.append("Please type the spellings without {{ }}.")
        return out
    result = (boxes.get("result") or "").strip()
    if not result:
        out.append("Please type the result.")
    for label, value in (("result", result), ("name", (boxes.get("name") or "").strip())):
        if re.search("[{},]", value):
            out.append(f"Please type the {label} without {{{{ }}}} or commas (for several spellings choose Choices).")
    if kind == "name" and not (boxes.get("name") or "").strip():
        out.append("Please type how it's spelt as a name.")
    if not out and code_problem(boxes_to_code(boxes)):
        out.append("The result has characters a word can't have.")
    return out


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _capital_words(text: str) -> str:
    return re.sub(r"(?<![^\s,])([^\W\d_])", lambda m: m.group(1).upper(), text)


def describe(code: str) -> str:
    """A stored code in words: 'Kṛs͟hṇa (always capital)', 'gŏḍ / as a name: God',
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
        return f"{show(p['main'])} / as a name: {show(p['name'], _capital)}"
    return show(p["main"])


def small_result(code: str) -> str:
    """The word an entry stands for, in small letters (for counting distinct diacritical words)."""
    p = parse_code(code or "")
    return (p["main"] if p else code or "").strip().lower()


def typed_with_capital(plain: str) -> str:
    return " ".join(_capital(w) for w in plain.split(" "))


def preview(plain: str, code: str) -> tuple[str, str]:
    """The translator's output for the plain word typed small and with a capital, with this code."""
    m = {plain: code}
    return translate_text(plain, m), translate_text(typed_with_capital(plain), m)


# ---------- spreadsheets ----------

def _sheet(wb, title: str, note: str, headers: list[str], rows: list[list], widths: list[int]):
    from openpyxl.styles import Alignment, Font, PatternFill
    ws = wb.create_sheet(title[:31])
    ws.append([note])
    ws["A1"].font = Font(italic=True)
    ws.append(headers)
    for c in ws[2]:
        c.font, c.fill = Font(bold=True), PatternFill("solid", fgColor="DDE7F3")
    for r in rows:
        ws.append(["" if v is None else v for v in r])
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(2, i).column_letter].width = w
    for row in ws.iter_rows(min_row=3):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A3"
    return ws


def _save(wb) -> bytes:
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def _label(item: dict, decision: dict) -> str:
    return dict(choice_options(item)).get((decision or {}).get("choice"), "(not decided)")


def _cap_label(item: dict, decision: dict) -> str:
    return dict(CAPITAL_OPTIONS.get(item["kind"], [])).get((decision or {}).get("cap"), "")


def report_xlsx(review: dict, decisions: dict, info: dict) -> bytes:
    """The review for the teams: a Summary sheet, then one sheet per section with what was decided, including the rows
    that change nothing (written without marks, style only, doubtful, already in the list). info: title, list,
    files, made, by, status, accepted."""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    s = review["sections"]
    p = plan(review, decisions)
    lines = [[info.get("title", "Review")], [f"Word list: {info.get('list', '')}"],
             ["Files: " + ", ".join(info.get("files") or [])], [f"Made {info.get('made', '')} by {info.get('by', '')}"],
             [f"Status: {info.get('status', '')}"], [], ["Section", "Rows", "What it means"]]
    if review["kind"] == "document":
        what = [("Different from our list", "clash", "In our list, but the document spells it differently"),
                ("New words", "new", "Words with marks that are not in our list"),
                ("Written without marks", "unmarked", "Plain in the document; our list would add marks (nothing changes)"),
                ("Style only", "style", "The same word typed differently (underline, look-alike letters): fixed automatically"),
                ("Doubtful", "doubtful", "Look-alike letters and PDF quirks: not compared"),
                ("Already in our list", "already", "Exactly as our list gives them")]
    else:
        what = [("New", "lnew", "In the file, not in our list"), ("Changed", "lchanged", "In both, with a different result"),
                ("Missing from the file", "lmissing", "In our list, not in the file (kept unless Delete was chosen)"),
                ("Problems", "problems", "Rows of the file that were skipped"),
                ("Left empty", "empty", "Rows with a plain word but no result (e.g. not answered yet): left out"),
                ("Unchanged", "unchanged", "The same in the file and in our list")]
    for label, key, meaning in what:
        lines.append([label, len(s.get(key, [])), meaning])
    accepted = str(info.get("status", "")).startswith("accepted")
    lines += [[], ["Decided", "", plan_sentence(p, done=accepted) if accepted
                   else "Not accepted yet. " + plan_sentence(p)]]
    if p["undecided"]:
        lines.append(["Not decided yet", len(p["undecided"]), ""])
    for row in lines:
        summary.append(row)
    summary["A1"].font = Font(bold=True, size=13)
    for c in summary[7]:
        c.font = Font(bold=True)
    summary.column_dimensions["A"].width = 28
    summary.column_dimensions["B"].width = 8
    summary.column_dimensions["C"].width = 90

    def dec(item):
        return decisions.get(item["id"]) or {}
    if review["kind"] == "document":
        _sheet(wb, "1 Different from our list", "In our list, but the document spells it differently.",
               ["In the document", "Plain word", "Other spellings in the document", "Times", "Pages", "Example line",
                "Our list gives", "Stored entry", "Decision", "Capital"],
               [[i["form"], i["plain"], "; ".join(f"{f['form']} ({f['status']})" for f in i["forms"][1:]), i["times"],
                 i["where"], i["line"], i["gives"], describe(i["entry"]) if i.get("entry") else "(not one entry)",
                 _label(i, dec(i)), _cap_label(i, dec(i))] for i in s["clash"]],
               [20, 16, 26, 7, 12, 50, 20, 26, 22, 14])
        _sheet(wb, "2 New words", "Words with marks that are not in our list.",
               ["In the document", "Plain word", "Times", "Pages", "Example line", "Seen as", "Decision", "Capital",
                "Also added"],
               [[i["form"], i["plain"], i["times"], i["where"], i["line"], SEEN[i["seen"]], _label(i, dec(i)),
                 _cap_label(i, dec(i)) if (dec(i).get("choice") or "").startswith(("add", "pending")) else "",
                 ", ".join(f"{i['extras'][x]['plain']} → {i['extras'][x]['result']}" for x in dec(i).get("extras", [])
                           if (dec(i).get("choice") or "").startswith("add") and x < len(i["extras"]))]
                for i in s["new"]], [20, 16, 7, 12, 50, 26, 14, 14, 24])
        _sheet(wb, "3 Written without marks", "Written without marks in the document; our list would add marks. "
               "Nothing changes in the list for these (for the teams' information).",
               ["In the document", "Plain word", "Our list gives", "Times", "Pages", "Example line", "Decision"],
               [[i["form"], i["plain"], i["gives"], i["times"], i["where"], i["line"], _label(i, dec(i))]
                for i in s["unmarked"]], [22, 16, 20, 7, 14, 60, 18])
        _sheet(wb, "4 Style only", "The same word, typed differently in the document (fixed automatically).",
               ["In the document", "Plain word", "Times", "Pages", "Example line", "Our list gives", "Difference",
                "Decision"],
               [[i["form"], i["plain"], i["times"], i["where"], i["line"], i["gives"], i["why"], "Nothing to change"]
                for i in s["style"]], [20, 16, 7, 12, 50, 20, 26, 18])
        _sheet(wb, "5 Doubtful", "Look-alike letters or lines that came out garbled from the PDF (not compared).",
               ["Word / page", "Times", "Pages", "Line", "What it is", "Decision"],
               [[i["form"], i["times"], i["where"], i["line"], i["why"], "Not compared"] for i in s["doubtful"]],
               [22, 7, 12, 60, 50, 14])
        _sheet(wb, "6 Already in our list", "Exactly as our list gives them: nothing to do.",
               ["In the document", "Plain word", "Times", "Pages", "Example line", "Our list gives"],
               [[i["form"], i["plain"], i["times"], i["where"], i["line"], i["gives"]] for i in s["already"]],
               [20, 16, 7, 12, 60, 20])
    else:
        _sheet(wb, "1 New", "In the file, not in our list.", ["Plain word", "Result in the file", "Shown as", "Row",
                                                               "Decision"],
               [[i["plain"], i["result"], describe(i["result"]), i["row"], _label(i, dec(i))] for i in s["lnew"]],
               [20, 26, 30, 7, 14])
        _sheet(wb, "2 Changed", "In both, with a different result.",
               ["Plain word", "Our list", "The file", "Row", "Decision"],
               [[i["plain"], describe(i["entry"]), describe(i["result"]), i["row"], _label(i, dec(i))]
                for i in s["lchanged"]], [20, 30, 30, 7, 16])
        _sheet(wb, "3 Missing from the file", "In our list but not in the file: kept unless Delete was chosen.",
               ["Plain word", "Our list", "Decision"],
               [[i["plain"], describe(i["entry"]), _label(i, dec(i))] for i in s["lmissing"]], [20, 30, 14])
        _sheet(wb, "4 Problems", "Rows of the file that were skipped.", ["Row", "Plain word", "Result", "Why"],
               [[i["row"], i["plain"], i["result"], i["why"]] for i in s["problems"]], [7, 20, 26, 50])
        _sheet(wb, "5 Left empty", "Rows with a plain word but no result (e.g. not answered yet): left out.",
               ["Row", "Plain word"], [[i["row"], i["plain"]] for i in s.get("empty", [])], [7, 24])
        _sheet(wb, "6 Unchanged", "The same in the file and in our list.", ["Plain word", "Result"],
               [[i["plain"], i["result"]] for i in s["unchanged"]], [20, 30])
    return _save(wb)


SEEN = {"small": "in small letters",
        "name": "always with a capital, also mid-sentence (a name or title?)",
        "start": "with a capital only at the start of a sentence"}


def list_xlsx(entries: list[dict]) -> bytes:
    """The whole list in the layout Add words accepts back (plain_text, diacritic_text)."""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "Word list"
    ws.append(["plain_text", "diacritic_text"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for e in sorted(entries, key=lambda e: e["plain_text"]):
        ws.append([e["plain_text"], e["diacritic_text"]])
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 40
    ws.freeze_panes = "A2"
    return _save(wb)


def pending_xlsx(items: list[dict], when) -> bytes:
    """The open pending items for the Diacriticals team, with an empty "Team's answer" column. The answered sheet can
    be uploaded on Add words (as an offline list)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "Pending"
    ws.append(["Plain word", "Suggested", "Shown as", "Why", "Source", "Page", "Example line", "Added",
               "Team's answer"])
    for c in ws[1]:
        c.font, c.fill = Font(bold=True), PatternFill("solid", fgColor="DDE7F3")
    for i in items:
        ws.append([i["plain_text"], i["suggested"], describe(i["suggested"] or ""), i["reason"], i["source"],
                   i["page"], i["context"], when(i["added_at"]), ""])
    for col, w in zip("ABCDEFGHI", (18, 22, 24, 40, 24, 10, 50, 18, 24)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    return _save(wb)
