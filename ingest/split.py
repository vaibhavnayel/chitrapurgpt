"""Splitting an issue into articles.

Sonnet 5.5 (default effort) sees an ISSUE MAP: every content block with an id like p12.b3 (PDF page 12, the 4th
content block on it), its type, and its text (headlines in full, long blocks as start … end), plus the contents
page in full. It answers with articles as block-id ranges, never retyping text. Content blocks are numbered
contiguously per page and page furniture never gets an id, so every id the model can see is valid.

The checker reads the answer tolerantly (dash styles, spaces, single ids, an id one past the end of a page) and
reports blocks in no article, blocks in two, ranges it can't place and contents entries with no article. One repair
round sends those findings back; whatever is still wrong becomes a review flag.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz
from unidecode import unidecode

from .llm import Spend

SPLIT_MODEL = "claude-sonnet-5-5"
SKIP_TYPES = {"page_furniture"}
KINDS = ("article", "notice", "photo_page", "cover", "contents", "poem", "other")
TITLE_MATCH = 75  # rapidfuzz token_set_ratio for a contents entry and an article title

SPLIT_SYSTEM = """You split one issue of {publication} into the articles a reader would cite: each teaching,
report, announcement, notice, appeal, invitation, table of rates, poem, photo page.

You get an ISSUE MAP. Every content block on every page has an id like p12.b3 (PDF page 12, block 3), its type, and its
text: headlines in full, longer blocks as their start ... end and length. Running headers/footers are left out.
"(cont. from prev page)" / "(continues next page)" are the page reader's flags. The contents page is also given in full.

Return ONLY JSON:
{{"issue": {{"label": "e.g. August 2024", "published_date": "DD-MM-YYYY (1st of the month if no day)"}},
 "articles": [{{"title": str, "author": str|null, "kind": "article"|"notice"|"photo_page"|"cover"|"contents"|"poem"|"other",
               "toc_title": str|null, "ranges": ["p6.b0-p9.b4", "p11.b2-p11.b2"]}}],
 "excluded": [{{"range": "p40.b0-p40.b2", "why": "advert"}}],
 "notes": str}}
Rules:
- A range "pA.bX-pB.bY" means every block from pA.bX through pB.bY in issue order. Use several ranges when an article
  is interrupted (e.g. a box or a photo page in the middle of it). Use only ids that appear in the map.
- Every block must end up in exactly one article or in "excluded". Exclude only blank/decorative blocks and adverts
  with no information; keep photo captions (they record events, dates and people) as photo_page articles or as part of
  the article they illustrate.
- Use the contents page: each entry there should become an article (fill toc_title with the entry as printed). Items
  not in the contents (cover, photo pages, short notices, the contents page itself) are articles too.
- Titles are the printed headline (ASCII transliteration is fine); for photo pages, describe the event from captions.
- Articles appear in issue order.
- Take the issue's label and date from the cover and the running headers/footers (listed at the end). A double issue
  is labelled like "July-August 2026"."""


def split_system(publication: str) -> str:
    return SPLIT_SYSTEM.format(publication=publication)


# ---------- blocks and the issue map ----------

@dataclass
class Block:
    id: str          # "p12.b3"
    order: int       # position in the issue
    page: int        # PDF page
    type: str
    text: str
    cont_prev: bool = False
    cont_next: bool = False


def content_blocks(reads: dict[int, dict | None]) -> list[Block]:
    """All content blocks in issue order, numbered contiguously per page; page furniture gets no id."""
    out: list[Block] = []
    for page in sorted(reads):
        k = 0
        for b in (reads[page] or {}).get("blocks", []):
            if b["type"] in SKIP_TYPES:
                continue
            out.append(Block(f"p{page}.b{k}", len(out), page, b["type"], b["text"],
                             b.get("continued_from_previous_page", False), b.get("continues_on_next_page", False)))
            k += 1
    return out


def _short(t: str, head: int = 110, tail: int = 70) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t if len(t) <= head + tail + 20 else f"{t[:head]} ... {t[-tail:]} [{len(t)} chars]"


def issue_map(blocks: list[Block], reads: dict[int, dict | None]) -> str:
    by_page = defaultdict(list)
    for b in blocks:
        by_page[b.page].append(b)
    lines = []
    for page in sorted(reads):
        printed = (reads[page] or {}).get("printed_page_number")
        lines.append(f"=== PDF page {page} (printed page {printed}) ===")
        for b in by_page[page]:
            if b.type in ("headline", "subheading", "byline"):
                body = re.sub(r"\s+", " ", b.text).strip()
            elif b.type in ("caption", "photo", "sponsor_note"):
                body = _short(b.text, 150, 0)
            else:
                body = _short(b.text)
            flags = (" (cont. from prev page)" if b.cont_prev else "") + (" (continues next page)" if b.cont_next else "")
            lines.append(f"{b.id} [{b.type}]{flags} {body}")
    return "\n".join(lines)


# ---------- the contents page ----------

_CONTENTS = re.compile(r"\b(contents|index)\b|अनुक्रमणिका|विषय\s*सूची|ಪರಿವಿಡಿ", re.I)
_PAGE_CELL = re.compile(r"^\(?(\d{1,3})\)?$")
_DOTTED = re.compile(r"^(.*?\S)[\s.·…_\-–]*?\s(\d{1,3})$")


@dataclass
class ContentsEntry:
    title: str
    page: int | None  # printed page, as listed


def parse_contents(text: str, max_page: int | None = None) -> list[ContentsEntry]:
    """Entries of a contents page as the reader transcribed it: Markdown table rows, or "Title .... 12" lines when
    there is no table. Entries pointing past `max_page` (e.g. a PIN code read as a page number) are dropped."""
    entries = _parse_contents(text, table_only=True) or _parse_contents(text, table_only=False)
    return [e for e in entries if max_page is None or e.page is None or e.page <= max_page]


def _is_separator(line: str) -> bool:
    return line.startswith("|") and all(re.fullmatch(r":?-{2,}:?", c.strip()) or not c.strip()
                                        for c in line.strip().strip("|").split("|"))


def _parse_contents(text: str, table_only: bool) -> list[ContentsEntry]:
    entries: list[ContentsEntry] = []
    lines = [line.strip() for line in (text or "").splitlines()]
    for i, line in enumerate(lines):
        if not line:
            continue
        if line.startswith("|"):
            if _is_separator(line) or (i + 1 < len(lines) and _is_separator(lines[i + 1])):
                continue  # a separator, or the header row above one ("| Item | Page No. |")
            cells = [c.strip().strip("*").strip() for c in line.strip("|").split("|")]
            pages = [int(m.group(1)) for c in cells if (m := _PAGE_CELL.match(c))]
            words = [c for c in cells if c and not _PAGE_CELL.match(c)]
            title = " ".join(words).strip(" |—–-•·")
            if not title or (_CONTENTS.search(title) and not pages) or re.fullmatch(r"(?i)page\s*(no\.?)?", title):
                continue
            entries.append(ContentsEntry(title, pages[-1] if pages else None))
        elif table_only:
            continue
        elif (m := _DOTTED.match(line)) and not _CONTENTS.fullmatch(m.group(1).strip()):
            title = m.group(1).strip(" .·…_-–")
            if title and not re.fullmatch(r"(?i)page\s*(no\.?)?", title):
                entries.append(ContentsEntry(title, int(m.group(2))))
    return entries


def find_contents(reads: dict[int, dict | None]) -> tuple[int | None, str, list[ContentsEntry]]:
    """(PDF page, full text, entries) of the contents page: the first page mentioning "contents" whose text parses
    into at least 3 entries, else the first page mentioning it."""
    first = None
    max_page = max(reads, default=0) + max(page_offset(reads), 0) + 2
    for page in sorted(reads):
        blocks = [b for b in (reads[page] or {}).get("blocks", []) if b["type"] not in SKIP_TYPES]
        text = "\n".join(b["text"] for b in blocks)
        if not _CONTENTS.search(text):
            continue
        entries = parse_contents(text, max_page)
        if len(entries) >= 3:
            return page, text, entries
        first = first or (page, text, [])  # one or two "entries" are likely something else (a price line)
    return first or (None, "", [])


def furniture_lines(reads: dict[int, dict | None], limit: int = 10) -> list[str]:
    """The page furniture the map leaves out but the issue's date is in: the first two pages' lines (the cover's
    masthead), then the most common running headers/footers (page numbers blanked out for counting)."""
    cover, counts, example = [], Counter(), {}
    for page in sorted(reads):
        for b in (reads[page] or {}).get("blocks", []):
            if b["type"] != "page_furniture" or not b["text"].strip():
                continue
            text = re.sub(r"\s+", " ", b["text"]).strip()
            if page <= 2 and text not in cover:
                cover.append(text)
            key = re.sub(r"(?<![\d-])\d{1,3}(?![\d-])", "#", text)
            counts[key] += 1
            example.setdefault(key, text)
    common = [example[k] for k, _ in counts.most_common() if example[k] not in cover]
    return (cover + common)[:limit]


def page_offset(reads: dict[int, dict | None]) -> int:
    """printed page - PDF page, if most pages that show a number agree (else 0)."""
    diffs = Counter()
    for page, read in reads.items():
        printed = (read or {}).get("printed_page_number")
        if printed and re.fullmatch(r"\d{1,4}", printed.strip()):
            diffs[int(printed) - page] += 1
    if not diffs:
        return 0
    offset, n = diffs.most_common(1)[0]
    return offset if n >= 0.5 * sum(diffs.values()) else 0


# ---------- reading the answer ----------

class SplitError(ValueError):
    pass


def parse_split(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise SplitError("no JSON object in the reply")
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise SplitError(f"bad JSON: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
        raise SplitError("no articles list")
    articles = []
    for a in data["articles"]:
        if not isinstance(a, dict):
            continue
        ranges = a.get("ranges") or []
        if isinstance(ranges, str):
            ranges = [ranges]
        kind = str(a.get("kind") or "article").strip().lower()
        articles.append({"title": str(a.get("title") or "Untitled").strip() or "Untitled",
                         "author": (str(a["author"]).strip() or None) if a.get("author") else None,
                         "kind": kind if kind in KINDS else "other",
                         "toc_title": (str(a["toc_title"]).strip() or None) if a.get("toc_title") else None,
                         "ranges": [r if isinstance(r, str) else "-".join(map(str, r)) if isinstance(r, list)
                                    else f"{r.get('from', '')}-{r.get('to', '')}" if isinstance(r, dict) else str(r)
                                    for r in ranges]})
    excluded = []
    for x in data.get("excluded") or []:
        if isinstance(x, dict) and x.get("range"):
            excluded.append({"range": str(x["range"]), "why": str(x.get("why") or "")})
        elif isinstance(x, str):
            excluded.append({"range": x, "why": ""})
    issue = data.get("issue") if isinstance(data.get("issue"), dict) else {}
    return {"issue": {"label": str(issue.get("label") or ""), "published_date": str(issue.get("published_date") or "")},
            "articles": articles, "excluded": excluded, "notes": str(data.get("notes") or "")}


_ID = re.compile(r"p\s*(\d+)\s*[.\-_ ]?\s*b\s*(\d+)", re.I)


class BlockIndex:
    def __init__(self, blocks: list[Block]):
        self.blocks = blocks
        self.by_id = {b.id: b.order for b in blocks}
        self.pages = defaultdict(list)
        for b in blocks:
            self.pages[b.page].append(b.order)

    def resolve(self, page: int, k: int, start: bool) -> tuple[int | None, bool]:
        """(block order, adjusted?) for pPAGE.bK. An id past the end of a page (or on a page without content
        blocks) snaps to the next block for a range start, or the page's last block for a range end."""
        order = self.by_id.get(f"p{page}.b{k}")
        if order is not None:
            return order, False
        if start:
            later = [orders[0] for p, orders in self.pages.items() if p > page]
            return (min(later) if later else None), True
        if self.pages.get(page):
            return self.pages[page][-1], True
        earlier = [orders[-1] for p, orders in self.pages.items() if p < page]
        return (max(earlier) if earlier else None), True

    def expand(self, rng: str) -> tuple[list[int], bool]:
        """Block orders covered by a range string, and whether an id had to be adjusted. Raises SplitError."""
        s = re.sub(r"\s+to\s+", "-", rng.replace("–", "-").replace("—", "-"), flags=re.I)
        ids = _ID.findall(s)
        if len(ids) not in (1, 2):
            raise SplitError(f"can't read range {rng!r}")
        (pa, ka), (pb, kb) = ids[0], ids[-1]
        single = ids[0] == ids[-1]
        a, adj_a = self.resolve(int(pa), int(ka), start=not single)
        b, adj_b = self.resolve(int(pb), int(kb), start=False)
        if a is None or b is None:
            raise SplitError(f"range {rng!r} isn't in this issue")
        if a > b:
            if adj_a or adj_b:  # e.g. "p12.b9-p12.b10", both past the end of page 12: its last block
                return [b], True
            a, b = b, a  # written backwards
        return list(range(a, b + 1)), adj_a or adj_b


# ---------- checking ----------

@dataclass
class Findings:
    unassigned: list[int] = field(default_factory=list)   # block orders in no article or exclusion
    doubled: dict[int, list] = field(default_factory=dict)  # block order -> owners (article index or "x")
    bad_ranges: list[str] = field(default_factory=list)
    adjusted_ranges: list[str] = field(default_factory=list)
    contents_total: int = 0
    contents_matched: int = 0
    missing_entries: list[ContentsEntry] = field(default_factory=list)

    @property
    def problems(self) -> bool:
        return bool(self.unassigned or self.doubled or self.bad_ranges or self.missing_entries)

    def summary(self) -> dict:
        return {"unassigned": len(self.unassigned), "doubled": len(self.doubled), "bad_ranges": len(self.bad_ranges),
                "adjusted_ranges": len(self.adjusted_ranges), "contents_matched": self.contents_matched,
                "contents_total": self.contents_total, "missing_entries": [e.title for e in self.missing_entries]}


def article_blocks(article: dict, index: BlockIndex) -> list[int]:
    """The article's blocks in range order (each once); ranges that can't be read are skipped."""
    seen, out = set(), []
    for rng in article["ranges"]:
        try:
            orders, _ = index.expand(rng)
        except SplitError:
            continue
        for o in orders:
            if o not in seen:
                seen.add(o)
                out.append(o)
    return out


def _norm_title(t: str) -> str:
    return unidecode(t or "").lower()


def check_split(result: dict, blocks: list[Block], entries: list[ContentsEntry], offset: int = 0) -> Findings:
    index = BlockIndex(blocks)
    f = Findings(contents_total=len(entries))
    owners: dict[int, list] = defaultdict(list)
    for i, a in enumerate(result["articles"]):
        for rng in a["ranges"]:
            try:
                orders, adjusted = index.expand(rng)
            except SplitError:
                f.bad_ranges.append(rng)
                continue
            if adjusted:
                f.adjusted_ranges.append(rng)
            for o in orders:
                if i not in owners[o]:
                    owners[o].append(i)
    for x in result["excluded"]:
        try:
            orders, _ = index.expand(x["range"])
        except SplitError:
            f.bad_ranges.append(x["range"])
            continue
        for o in orders:
            owners[o].append("x")
    f.unassigned = [b.order for b in blocks if not owners.get(b.order)]
    f.doubled = {o: v for o, v in owners.items() if len(v) > 1}

    starts = []
    for a in result["articles"]:
        orders = article_blocks(a, index)
        starts.append(blocks[min(orders)].page if orders else None)
    titles = [_norm_title(a.get("toc_title") or a["title"]) for a in result["articles"]]
    shifts = {0, _contents_shift(entries, titles, starts, offset)}
    for e in entries:
        target = None if e.page is None else e.page - offset
        hit = any(fuzz.token_set_ratio(_norm_title(e.title), t) >= TITLE_MATCH
                  and (target is None or (s is not None and any(abs(s - target - k) <= 1 for k in shifts)))
                  for t, s in zip(titles, starts))
        if hit:
            f.contents_matched += 1
        else:
            f.missing_entries.append(e)
    return f


def _contents_shift(entries: list[ContentsEntry], titles: list[str], starts: list, offset: int) -> int:
    """Some contents pages are stale: most entries point a fixed number of pages before (or after) the article with
    their title. That shift, if at least 3 clear title matches and half of them agree on it; else 0."""
    diffs = Counter()
    for e in entries:
        if e.page is None:
            continue
        scores = [(fuzz.token_set_ratio(_norm_title(e.title), t), s) for t, s in zip(titles, starts) if s is not None]
        best = max(scores, default=(0, None))
        if best[0] >= 90:
            diffs[best[1] - (e.page - offset)] += 1
    if not diffs:
        return 0
    shift, n = diffs.most_common(1)[0]
    return shift if n >= 3 and n >= 0.5 * sum(diffs.values()) else 0


def repair_message(f: Findings, blocks: list[Block], result: dict) -> str:
    lines = ["Your split has these problems:"]
    if f.unassigned:
        lines.append("- Blocks in no article and not excluded:")
        lines += [f"  {blocks[o].id} [{blocks[o].type}] {_short(blocks[o].text, 80, 0)}" for o in f.unassigned[:80]]
        if len(f.unassigned) > 80:
            lines.append(f"  ... and {len(f.unassigned) - 80} more")
    if f.doubled:
        lines.append("- Blocks in more than one article (or both in an article and excluded):")
        for o, who in list(f.doubled.items())[:60]:
            names = ", ".join("excluded" if w == "x" else repr(result["articles"][w]["title"]) for w in who)
            lines.append(f"  {blocks[o].id}: {names}")
    if f.bad_ranges:
        lines.append("- Ranges that don't use ids from the map: " + ", ".join(repr(r) for r in f.bad_ranges[:40]))
    if f.missing_entries:
        lines.append("- Contents entries with no article (matching title, starting on or next to that page):")
        lines += [f"  {e.title!r}" + (f" (page {e.page})" if e.page is not None else "") for e in f.missing_entries]
        lines.append("  If an entry really is part of another article, or isn't in this issue, say so in notes.")
    lines.append("Fix these and return the complete corrected JSON in the same format: all articles, not only the "
                 "changed ones.")
    return "\n".join(lines)


# ---------- the split call ----------

@dataclass
class SplitOutcome:
    result: dict
    findings: Findings
    rounds: int
    cost: float
    first_findings: dict | None = None
    error: str | None = None


async def split_issue(blocks: list[Block], reads: dict[int, dict | None], llm, publication: str, *,
                      model: str = SPLIT_MODEL, spend: Spend | None = None, repair: bool = True) -> SplitOutcome:
    """Split, check, and (if needed) one repair round. Raises SplitError if no usable answer came back."""
    contents_page, contents_text, entries = find_contents(reads)
    offset = page_offset(reads)
    furniture = "\n".join(furniture_lines(reads)) or "(none)"
    user = (f"ISSUE MAP\n{issue_map(blocks, reads)}\n\nCONTENTS PAGE (full text)\n{contents_text or '(none found)'}"
            f"\n\nCOVER AND RUNNING HEADERS/FOOTERS\n{furniture}")
    # The map is cached, so a repair round reads it at a tenth of the price.
    messages = [{"role": "user", "content": [{"type": "text", "text": user, "cache_control": {"type": "ephemeral"}}]}]
    system = split_system(publication)
    cost = 0.0

    async def ask(msgs, what):
        nonlocal cost
        text, usage = await llm.complete(model=model, system=system, messages=msgs, max_tokens=48000)
        cost += usage.cost
        if spend is not None:
            spend.add(usage, what)
        if usage.stop_reason == "max_tokens":
            raise SplitError("the split was cut off (max_tokens)")
        return text

    text = await ask(messages, "split")
    result = parse_split(text)
    findings = check_split(result, blocks, entries, offset)
    outcome = SplitOutcome(result, findings, 1, cost)
    if repair and findings.problems:
        outcome.first_findings = findings.summary()
        msgs = messages + [{"role": "assistant", "content": text},
                           {"role": "user", "content": repair_message(findings, blocks, result)}]
        try:
            text2 = await ask(msgs, "split repair")
            result2 = parse_split(text2)
            findings2 = check_split(result2, blocks, entries, offset)
            if _score(findings2) <= _score(findings):
                outcome.result, outcome.findings = result2, findings2
        except SplitError as e:
            outcome.error = f"repair round failed: {e}"
        outcome.rounds, outcome.cost = 2, cost
    return outcome


def _score(f: Findings) -> tuple:
    return (len(f.bad_ranges) + len(f.unassigned) + len(f.doubled), len(f.missing_entries))
