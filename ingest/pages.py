"""Reading one page, and checking the read against the page's text layer.

Haiku 5.5 (thinking off) transcribes the page image into typed blocks (prompt from the spike). Where the PDF has a
clean text layer (≥ 200 characters outside legacy Devanagari fonts), the read must contain ≥ 97% of the layer's
words; if it doesn't, or the read is empty or unusable, Sonnet 5.5 reads the page again and the better read is
kept. A page that is still bad gets a review flag.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process
from unidecode import unidecode

from .llm import Spend, image_block
from .pdf import PdfFile, TextLayer

READER_MODEL = "claude-haiku-5-5"
REREAD_MODEL = "claude-sonnet-5-5"
MIN_RECALL = 0.97
READ_ATTEMPTS = 2  # the reader is asked again once if its reply isn't usable JSON

PAGE_PROMPT = """You are reading one page of {publication}.
Transcribe the page into blocks, in the order a person would read them, and return ONLY JSON:
{{"printed_page_number": str|null,
 "blocks": [{{"type": "headline"|"subheading"|"byline"|"body"|"verse"|"table"|"caption"|"list"|"sponsor_note"|"page_furniture"|"advert"|"photo",
             "text": str,
             "continued_from_previous_page": bool, "continues_on_next_page": bool}}],
 "new_articles_starting_here": [str],
 "notes": str}}
Rules:
- page_furniture = running headers/footers (magazine name, date, page number). Keep them, typed as such.
- Copy text exactly. Do not summarise, correct, or skip anything. Devanagari or Kannada text goes in its own script (Unicode).
- A table becomes a Markdown table that keeps each row's label next to its numbers.
- photo = one short description of the picture; its caption is a separate caption block.
- A headline set beside or above its article still comes first, before that article's body.
- new_articles_starting_here lists titles of articles that begin on this page (not ones continued from before).
- Return the JSON object only: no code fences, no commentary before or after it."""

BLOCK_TYPES = {"headline", "subheading", "byline", "body", "verse", "table", "caption", "list", "sponsor_note",
               "page_furniture", "advert", "photo"}


def publication_phrase(name: str, description: str = "") -> str:
    """'"The Chitrapur Sunbeam", the magazine of Shri Chitrapur Math' (the spike prompt's wording)."""
    return f'"{name}", {description}' if description else f'"{name}"'


def page_prompt(publication: str) -> str:
    return PAGE_PROMPT.format(publication=publication)


class InvalidRead(ValueError):
    pass


def parse_page_read(text: str) -> dict:
    """The reader's JSON, normalised: unknown block types become body, flags are booleans. Raises InvalidRead."""
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise InvalidRead("no JSON object in the reply")
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise InvalidRead(f"bad JSON: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("blocks"), list):
        raise InvalidRead("no blocks list")
    blocks = []
    for b in data["blocks"]:
        if not isinstance(b, dict) or not isinstance(b.get("text"), str):
            raise InvalidRead("a block without text")
        kind = str(b.get("type") or "body").strip().lower()
        blocks.append({"type": kind if kind in BLOCK_TYPES else "body", "text": b["text"],
                       "continued_from_previous_page": bool(b.get("continued_from_previous_page")),
                       "continues_on_next_page": bool(b.get("continues_on_next_page"))})
    printed = data.get("printed_page_number")
    starts = data.get("new_articles_starting_here")
    return {"printed_page_number": None if printed in (None, "") else str(printed), "blocks": blocks,
            "new_articles_starting_here": [str(s) for s in starts] if isinstance(starts, list) else [],
            "notes": str(data.get("notes") or "")}


def content_chars(read: dict | None) -> int:
    """Characters of text outside page furniture and photo descriptions."""
    if not read:
        return 0
    return sum(len(b["text"].strip()) for b in read["blocks"] if b["type"] not in ("page_furniture", "photo"))


# ---------- the check ----------

_HYPHEN = re.compile(r"[-\u00ad\u2010\u2011\x02]\s*")  # a hyphen and any line break after it: "commu-\nnity"
_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    """Words for comparing a read with the text layer: transliterated to ASCII, lowercased, hyphens joined."""
    return _WORD.findall(_HYPHEN.sub("", unidecode(text or "")).lower())


def _read_words(read: dict | None) -> set[str]:
    return set(words("\n".join(b["text"] for b in (read or {}).get("blocks", []))))


def _unmatched(layer_text: str, read: dict | None) -> tuple[int, list[str]]:
    """(number of layer words counted, the layer words not found in the read). A layer word is found if the read
    has it (how often doesn't matter: some PDFs draw text twice for a bold effect, so the layer repeats what the
    page shows once). Text layers also break words where the read doesn't ("1 st", "Shr ī" with the accented letter
    drawn separately and moved to the end of the line) and run words together ("instancesthat"), so a layer word
    also counts as found when, joined with the next one or two layer words or split in two, it gives words of the
    read; and a broken-off piece of 2+ letters counts if a longer word of the read contains it. Some fonts' text
    layers misspell what they draw (a "ti" ligature mapped to "o": "parocipants"; doubled letters for diacritics:
    "chaaturmaasa"), so a word of 6+ letters also counts when the read has one at least 80% similar. Single letters
    are left out (accent debris). Numbers must match exactly."""
    want, got = words(layer_text), _read_words(read)
    vocabulary = list(got)
    missing: list[str] = []
    counted = 0
    i = 0
    while i < len(want):
        joined = next((n for n in (3, 2) if i + n <= len(want) and "".join(want[i:i + n]) in got), None)
        if joined:
            counted += joined
            i += joined
            continue
        w = want[i]
        i += 1
        if len(w) == 1 and w.isalpha():
            continue
        counted += 1
        if w in got or any(w[:k] in got and w[k:] in got for k in range(1, len(w))):
            continue
        if not (w.isalpha() and (_fragment_of(w, vocabulary) or _misspelling_of(w, vocabulary))):
            missing.append(w)
    return counted, missing


def _fragment_of(w: str, vocabulary: list[str]) -> bool:
    """A piece (2+ letters) broken off a longer word of the read ("sw" of "swami" when the ā is drawn apart)."""
    return len(w) >= 2 and any(w in v for v in vocabulary)


def _misspelling_of(w: str, vocabulary: list[str]) -> bool:
    """A word of 6+ letters the read has in a slightly different spelling (≥ 80% similar, not identical)."""
    return len(w) >= 6 and any(v != w for v, _, _ in process.extract(w, vocabulary, scorer=fuzz.ratio,
                                                                     score_cutoff=80, limit=3))


def recall(layer_text: str, read: dict | None) -> float:
    """Share of the text layer's words (with multiplicity) found in the read."""
    total, missing = _unmatched(layer_text, read)
    return 1.0 if not total else (total - len(missing)) / total


def missing_words(layer_text: str, read: dict | None, limit: int = 12) -> list[str]:
    return [w for w, _ in Counter(_unmatched(layer_text, read)[1]).most_common(limit)]


# ---------- one page ----------

@dataclass
class Attempt:
    model: str
    reread: bool = False  # the stronger model's second opinion (else the reader's own read)
    read: dict | None = None
    error: str | None = None
    recall: float | None = None
    cost: float = 0.0
    secs: float = 0.0
    out_tokens: int = 0

    def as_dict(self) -> dict:
        return {"model": self.model, "reread": self.reread, "error": self.error, "recall": self.recall,
                "cost": round(self.cost, 6), "secs": round(self.secs, 2), "out": self.out_tokens}


@dataclass
class PageResult:
    page: int
    read: dict | None
    model: str | None
    layer_kind: str
    layer_chars: int
    legacy_chars: int
    recall: float | None
    attempts: list[Attempt] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(a.cost for a in self.attempts)

    @property
    def reread(self) -> bool:
        return any(a.reread for a in self.attempts)

    def as_dict(self) -> dict:
        return {"page": self.page, "read": self.read, "model": self.model, "layer_kind": self.layer_kind,
                "layer_chars": self.layer_chars, "legacy_chars": self.legacy_chars, "recall": self.recall,
                "attempts": [a.as_dict() for a in self.attempts], "flags": self.flags, "cost": round(self.cost, 6)}

    @classmethod
    def from_dict(cls, d: dict) -> "PageResult":
        return cls(page=d["page"], read=d.get("read"), model=d.get("model"), layer_kind=d.get("layer_kind", "none"),
                   layer_chars=d.get("layer_chars", 0), legacy_chars=d.get("legacy_chars", 0),
                   recall=d.get("recall"), flags=list(d.get("flags") or []),
                   attempts=[Attempt(model=a["model"], reread=a.get("reread", a["model"] == REREAD_MODEL),
                                     error=a.get("error"), recall=a.get("recall"), cost=a.get("cost", 0.0),
                                     secs=a.get("secs", 0.0), out_tokens=a.get("out", 0))
                             for a in d.get("attempts") or []])


async def _ask(llm, model: str, image, prompt: str, spend: Spend | None, page: int, reread: bool) -> Attempt:
    """One read of the page image: by the reader (thinking off; the model must allow that), or as the re-read (the
    model's default thinking, with room for it in max_tokens)."""
    attempt = Attempt(model=model, reread=reread)
    try:
        text, usage = await llm.complete(
            model=model, max_tokens=32000 if reread else 16000, thinking=None if reread else False,
            messages=[{"role": "user", "content": [image_block(image.data, image.media_type),
                                                   {"type": "text", "text": prompt}]}])
    except Exception as e:  # noqa: BLE001  (an API failure after the SDK's retries: record it, try the next step)
        attempt.error = f"{type(e).__name__}: {str(e)[:200]}"
        return attempt
    attempt.cost, attempt.secs, attempt.out_tokens = usage.cost, usage.secs, usage.output_tokens
    if spend is not None:
        spend.add(usage, f"page {page} {model}")
    if usage.stop_reason == "max_tokens":
        attempt.error = "the reply was cut off (max_tokens)"
        return attempt
    try:
        attempt.read = parse_page_read(text)
    except InvalidRead as e:
        attempt.error = str(e)
    return attempt


def _missing_text(read: dict | None, layer: TextLayer) -> bool:
    """No text besides headers/footers and picture descriptions, though the page has some: Devanagari in a legacy
    font, or text-layer words the read doesn't have. (A photo page whose only text is its footer is fine.)"""
    if content_chars(read) > 0:
        return False
    return layer.legacy_chars >= 50 or recall(layer.clean_text, read) < MIN_RECALL


def _usable(attempt: Attempt, layer: TextLayer) -> bool:
    """A read worth keeping without a second opinion."""
    if attempt.read is None or _missing_text(attempt.read, layer):
        return False
    return not (layer.checkable and (attempt.recall or 0) < MIN_RECALL)


def _better(a: Attempt, b: Attempt, layer: TextLayer) -> Attempt:
    """Of two reads, the one to keep: valid over invalid, then higher recall (checkable pages), then more text."""
    if (a.read is None) != (b.read is None):
        return a if a.read is not None else b
    if layer.checkable and a.recall is not None and b.recall is not None and a.recall != b.recall:
        return a if a.recall > b.recall else b
    if (content_chars(a.read) > 0) != (content_chars(b.read) > 0):
        return a if content_chars(a.read) > 0 else b
    return b  # the re-read: a stronger model looked at the same page


async def read_page(pdf: PdfFile, page_no: int, llm, publication: str, *, spend: Spend | None = None,
                    reader_model: str = READER_MODEL, reread_model: str = REREAD_MODEL) -> PageResult:
    image = await asyncio.to_thread(pdf.render, page_no)
    layer = await asyncio.to_thread(pdf.text_layer, page_no)
    prompt = page_prompt(publication)
    result = PageResult(page=page_no, read=None, model=None, layer_kind=layer.kind, layer_chars=layer.clean_chars,
                        legacy_chars=layer.legacy_chars, recall=None)

    def scored(a: Attempt) -> Attempt:
        if a.read is not None and layer.checkable:
            a.recall = round(recall(layer.clean_text, a.read), 4)
        result.attempts.append(a)
        return a

    best = None
    for _ in range(READ_ATTEMPTS):
        attempt = scored(await _ask(llm, reader_model, image, prompt, spend, page_no, reread=False))
        best = attempt if best is None else _better(best, attempt, layer)
        if attempt.read is not None:
            break  # retry only unusable replies; a low-recall read goes to the stronger model instead
    if not _usable(best, layer):
        best = _better(best, scored(await _ask(llm, reread_model, image, prompt, spend, page_no, reread=True)), layer)

    result.read, result.model, result.recall = best.read, best.model, best.recall
    result.flags = page_flags(page_no, layer, result.read, result.recall, best.error)
    return result


def page_flags(page_no: int, layer: TextLayer, read: dict | None, read_recall: float | None,
               error: str | None = None) -> list[str]:
    """Review flags for the read that was kept."""
    if read is None:
        return [f"page {page_no}: couldn't be read ({error})"]
    if layer.checkable and (read_recall or 0) < MIN_RECALL:
        miss = ", ".join(missing_words(layer.clean_text, read, 8))
        return [f"page {page_no}: the read has only {read_recall:.0%} of the PDF's text (missing e.g. {miss})"]
    if _missing_text(read, layer):
        return [f"page {page_no}: the read is empty but the page has text"]
    return []


def recheck(pdf: PdfFile, result: PageResult) -> PageResult:
    """Recompute a stored page's recall and flags with the current check (no new read)."""
    layer = pdf.text_layer(result.page)
    result.recall = round(recall(layer.clean_text, result.read), 4) if layer.checkable and result.read else None
    result.flags = page_flags(result.page, layer, result.read, result.recall,
                              next((a.error for a in reversed(result.attempts) if a.error), None))
    return result
