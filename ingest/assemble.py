"""Articles from a checked split: text, pages, part splitting, summaries, the issue's source name."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date

from .llm import Spend
from .split import Block, BlockIndex, article_blocks

SUMMARY_MODEL = "claude-haiku-5-5"
MAX_ARTICLE_TOKENS = 8000  # under text-embedding-3-large's 8,191-token input; longer articles become parts
LEFT_OUT_TYPES = {"photo"}  # the reader's descriptions of pictures (captions are kept)

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
_MONTH_RE = re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
                       r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?", re.I)


# ---------- token counting ----------

_encoding = None


def count_tokens(text: str) -> int:
    """cl100k_base tokens (the embeddings model's tokenizer). Where tiktoken can't load its vocabulary (no
    network), a conservative estimate: ~4 ASCII characters per token, one token per other character."""
    global _encoding
    if _encoding is None:
        try:
            import tiktoken
            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            _encoding = False
    if _encoding:
        return len(_encoding.encode(text, disallowed_special=()))
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return math.ceil(ascii_chars / 4) + (len(text) - ascii_chars)


# ---------- articles ----------

@dataclass
class Article:
    n: int                    # 1-based, in issue order
    title: str
    kind: str
    author: str | None
    toc_title: str | None
    blocks: list[str]         # block ids, in text order
    pdf_start: int
    pdf_end: int
    start_page: int           # the pages cited (see assemble)
    end_page: int
    text: str
    part: tuple[int, int] | None = None
    summary: str | None = None
    excluded: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {"n": self.n, "title": self.title, "kind": self.kind, "author": self.author,
                "toc_title": self.toc_title, "blocks": self.blocks, "pdf_start": self.pdf_start,
                "pdf_end": self.pdf_end, "start_page": self.start_page, "end_page": self.end_page, "text": self.text,
                "part": list(self.part) if self.part else None, "summary": self.summary, "excluded": self.excluded,
                "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "Article":
        d = dict(d)
        d["part"] = tuple(d["part"]) if d.get("part") else None
        return cls(**d)


def block_text(b: Block) -> str:
    return b.text.strip()


MIN_TEXT_CHARS = 40  # shorter articles (a section divider's heading, a lone photo credit) start out excluded


def assemble(result: dict, blocks: list[Block], max_tokens: int = MAX_ARTICLE_TOKENS) -> list[Article]:
    """One Article per split article (or several parts of a long one), in issue order. Text is the blocks in range
    order, one paragraph each: headlines on their own lines, tables as the reader's Markdown, captions kept,
    photo descriptions left out. Pages are PDF page numbers: what anyone opening the PDF sees (for recent Sunbeam
    issues they are also the printed numbers; front matter in older scans isn't numbered)."""
    index = BlockIndex(blocks)
    drafts = []
    for a in result["articles"]:
        orders = article_blocks(a, index)
        if orders:
            drafts.append((min(orders), a, orders))
    drafts.sort(key=lambda d: d[0])
    out: list[Article] = []
    for _, a, orders in drafts:
        for part_orders, part in _parts([blocks[o] for o in orders], max_tokens):
            texts = [block_text(b) for b in part_orders if b.type not in LEFT_OUT_TYPES and block_text(b)]
            pages = [b.page for b in part_orders]
            title = a["title"] if part is None else f"{a['title']} (part {part[0]} of {part[1]})"
            art = Article(n=len(out) + 1, title=title, kind=a["kind"], author=a.get("author"),
                          toc_title=a.get("toc_title"), blocks=[b.id for b in part_orders],
                          pdf_start=min(pages), pdf_end=max(pages), start_page=min(pages), end_page=max(pages),
                          text="\n\n".join(texts), part=part)
            if not art.text:
                art.excluded, art.note = True, "no text (pictures only)"
            elif len(art.text) < MIN_TEXT_CHARS:
                art.excluded, art.note = True, "almost no text (e.g. a section divider)"
            out.append(art)
    return out


def _parts(blocks: list[Block], max_tokens: int):
    """The article's blocks in parts of at most ~max_tokens, split at block boundaries (a single oversized block is
    split at line breaks). Yields (blocks, (k, n)) or, for a short article, (blocks, None)."""
    units: list[tuple[Block, int]] = []
    for b in blocks:
        t = count_tokens(block_text(b)) if b.type not in LEFT_OUT_TYPES else 0
        if t <= max_tokens:
            units.append((b, t))
            continue
        chunk: list[str] = []
        for line in b.text.splitlines(keepends=True):
            if chunk and count_tokens("".join(chunk + [line])) > max_tokens:
                units.append((Block(b.id, b.order, b.page, b.type, "".join(chunk)), count_tokens("".join(chunk))))
                chunk = []
            chunk.append(line)
        if chunk:
            units.append((Block(b.id, b.order, b.page, b.type, "".join(chunk)), count_tokens("".join(chunk))))
    total = sum(t for _, t in units)
    if total <= max_tokens:
        yield [b for b, _ in units], None
        return
    n = math.ceil(total / max_tokens)
    cap = min(max_tokens, total / n * 1.15)
    parts, cur, cur_t = [], [], 0
    for b, t in units:
        if cur and cur_t + t > cap:
            parts.append(cur)
            cur, cur_t = [], 0
        cur.append(b)
        cur_t += t
    if cur:
        parts.append(cur)
    for k, p in enumerate(parts, 1):
        yield p, (k, len(parts))


def leftover_flags(findings, blocks: list[Block], result: dict) -> list[str]:
    """Review flags for what the split still gets wrong after the repair round."""
    flags = []
    if findings.unassigned:
        ids = ", ".join(blocks[o].id for o in findings.unassigned[:10])
        chars = sum(len(blocks[o].text) for o in findings.unassigned)
        flags.append(f"{len(findings.unassigned)} block(s) ({chars} characters) are in no article: {ids}"
                     + (" …" if len(findings.unassigned) > 10 else ""))
    if findings.doubled:
        ids = ", ".join(blocks[o].id for o in list(findings.doubled)[:10])
        flags.append(f"{len(findings.doubled)} block(s) are in two articles: {ids}")
    if findings.bad_ranges:
        flags.append(f"{len(findings.bad_ranges)} range(s) couldn't be placed: {', '.join(findings.bad_ranges[:5])}")
    for e in findings.missing_entries:
        where = f" (page {e.page})" if e.page is not None else ""
        flags.append(f"contents entry “{e.title}”{where} has no matching article")
    return flags


# ---------- summaries ----------

SUMMARY_PROMPT = """Summarise this item from {publication} in 1 to 3 sentences, in English. Say what it is (for example
a teaching, a report of an event, a notice or a table of rates) and its main points, names and dates. Return only the
summary.

Title: {title}

{text}"""


async def summarize(articles: list[Article], llm, publication: str, *, model: str = SUMMARY_MODEL,
                    spend: Spend | None = None, concurrency: int = 8) -> None:
    """Fills in each article's summary (skips ones that have one, so a resumed job doesn't pay twice)."""
    from .llm import gather_limited

    async def one(a: Article):
        if a.summary or a.excluded:
            return
        prompt = SUMMARY_PROMPT.format(publication=publication, title=a.title, text=a.text[:24000])
        text, usage = await llm.complete(model=model, thinking=False, max_tokens=600,
                                         messages=[{"role": "user", "content": prompt}])
        if spend is not None:
            spend.add(usage, f"summary {a.n}")
        a.summary = " ".join(text.split()) or None

    await gather_limited(concurrency, [one(a) for a in articles])


# ---------- the issue ----------

def issue_months(label: str, published_date: str = "") -> tuple[int | None, list[str]]:
    """(year, month names) of an issue from its label ("July-August 2026", "May & June 2023"), else its date."""
    months = []
    for m in _MONTH_RE.finditer(label or ""):
        name = next(x for x in MONTHS if x.lower().startswith(m.group(1).lower()))
        if name not in months:
            months.append(name)
    year = re.search(r"\b(1[89]\d\d|20\d\d)\b", label or "")
    d = parse_date(published_date)
    if not months and d:
        months = [MONTHS[d.month - 1]]
    return (int(year.group(1)) if year else (d.year if d else None)), months


def parse_date(s: str) -> date | None:
    m = re.fullmatch(r"\s*(\d{1,2})-(\d{1,2})-(\d{4})\s*", s or "")
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def source_name(prefix: str, year: int | None, months: list[str]) -> str:
    """The issue's file name in citations, in the old naming: Sunbeam-2025-February.txt,
    Sunbeam-2026-July-August.txt."""
    return f"{prefix}-{year or 'undated'}-{'-'.join(months) or 'Issue'}.txt"


def published_date_for(year: int | None, months: list[str], given: str = "") -> str:
    """DD-MM-YYYY: the split's date if it's valid, else the 1st of the issue's first month."""
    if parse_date(given):
        d = parse_date(given)
        return f"{d.day:02d}-{d.month:02d}-{d.year}"
    if year and months:
        return f"01-{MONTHS.index(months[0]) + 1:02d}-{year}"
    return ""


@dataclass
class IssueInfo:
    label: str
    published_date: str
    source: str
    year: int | None = None
    months: list[str] = field(default_factory=list)


def issue_info(result: dict, prefix: str) -> IssueInfo:
    label = result["issue"].get("label", "")
    year, months = issue_months(label, result["issue"].get("published_date", ""))
    return IssueInfo(label=label, published_date=published_date_for(year, months, result["issue"].get("published_date", "")),
                     source=source_name(prefix, year, months), year=year, months=months)


def doc_metadata(article: Article, issue: IssueInfo) -> dict:
    """Metadata in the knowledge base's format (same keys and order as the original articles). The line numbers
    meant something only for the old text-file pipeline, so they're missing values, as unknown authors are."""
    return {"source": issue.source, "published_date": issue.published_date, "title": article.title,
            "author": article.author, "summary": article.summary, "start_line": None, "end_line": None,
            "start_page": article.start_page, "end_page": article.end_page}
