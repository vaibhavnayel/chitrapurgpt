"""One issue end to end: read and check every page, split, assemble, summarise.

Where page reads are kept is up to the caller (Postgres for the worker, files for the eval), through a PageStore:
pages already stored are not read again, so a job interrupted half-way resumes where it stopped.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from . import assemble as asm
from .llm import Spend, gather_limited
from .pages import REREAD_MODEL, READER_MODEL, PageResult, read_page, recheck
from .pdf import PdfFile
from .split import SPLIT_MODEL, content_blocks, find_contents, split_issue

PAGE_CONCURRENCY = 8


@dataclass
class IssueConfig:
    publication: str                 # '"The Chitrapur Sunbeam", the magazine of Shri Chitrapur Math'
    source_prefix: str               # "Sunbeam"
    reader_model: str = READER_MODEL
    reread_model: str = REREAD_MODEL
    split_model: str = SPLIT_MODEL
    summary_model: str = asm.SUMMARY_MODEL
    concurrency: int = PAGE_CONCURRENCY
    max_article_tokens: int = asm.MAX_ARTICLE_TOKENS


class PageStore(Protocol):
    async def load(self) -> dict[int, dict]: ...
    async def save(self, result: PageResult) -> None: ...


class FilePageStore:
    """Page reads as JSON files in a directory (the eval)."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    async def load(self) -> dict[int, dict]:
        out = {}
        for f in sorted(self.dir.glob("p*.json")):
            d = json.loads(f.read_text())
            out[d["page"]] = d
        return out

    async def save(self, result: PageResult) -> None:
        (self.dir / f"p{result.page:03d}.json").write_text(json.dumps(result.as_dict(), ensure_ascii=False))


Progress = Callable[[str, dict], Awaitable[None]] | None
OnPage = Callable[[PdfFile, int], Awaitable[None]] | None


async def read_pages(pdf_path: str, store: PageStore, llm, cfg: IssueConfig, *, spend: Spend | None = None,
                     progress: Progress = None, recheck_stored: bool = False,
                     on_page: OnPage = None) -> dict[int, PageResult]:
    """Every page's checked read, reading only pages the store doesn't have yet. recheck_stored re-applies the
    current check to stored reads (the eval uses it after the check changed). on_page(pdf, page) runs after each
    page is read (the worker uploads its preview)."""
    done = {p: PageResult.from_dict(d) for p, d in (await store.load()).items()}
    with PdfFile(pdf_path) as pdf:
        if recheck_stored:
            for r in done.values():
                await asyncio.to_thread(recheck, pdf, r)
        todo = [p for p in range(1, pdf.page_count + 1) if p not in done]

        async def one(p: int):
            result = await read_page(pdf, p, llm, cfg.publication, spend=spend, reader_model=cfg.reader_model,
                                     reread_model=cfg.reread_model)
            await store.save(result)
            done[p] = result
            if on_page:
                await on_page(pdf, p)
            if progress:
                await progress("page", {"done": len(done), "total": pdf.page_count})
            return result

        await gather_limited(cfg.concurrency, [one(p) for p in todo])
        return {p: done[p] for p in range(1, pdf.page_count + 1)}


@dataclass
class IssueResult:
    pages: dict[int, PageResult]
    split: dict | None = None
    findings: dict | None = None
    first_findings: dict | None = None
    split_rounds: int = 0
    articles: list[asm.Article] = field(default_factory=list)
    issue: asm.IssueInfo | None = None
    contents_page: int | None = None
    flags: list[str] = field(default_factory=list)
    costs: dict = field(default_factory=dict)  # per stage, including stored page reads paid for earlier
    secs: dict = field(default_factory=dict)
    run_cost: float = 0.0                      # spent by this run only
    error: str | None = None

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for a in self.articles:
            if not a.excluded:
                kinds[a.kind] = kinds.get(a.kind, 0) + 1
        f = self.findings or {}
        return {"pages": len(self.pages), "articles": sum(kinds.values()), "kinds": kinds,
                "contents_matched": f.get("contents_matched"), "contents_total": f.get("contents_total"),
                "unassigned": f.get("unassigned"), "doubled": f.get("doubled"),
                "reread_pages": sum(1 for p in self.pages.values() if p.reread),
                "flags": len(self.flags), "cost": round(sum(self.costs.values()), 4),
                "secs": round(sum(self.secs.values()), 1)}


def reads_of(pages: dict[int, PageResult]) -> dict[int, dict | None]:
    return {p: r.read for p, r in pages.items()}


async def split_and_assemble(pages: dict[int, PageResult], llm, cfg: IssueConfig, *, spend: Spend | None = None,
                             result: IssueResult | None = None) -> IssueResult:
    """Split the read pages into articles and assemble them (no summaries)."""
    result = result or IssueResult(pages=pages)
    reads = reads_of(pages)
    blocks = content_blocks(reads)
    result.contents_page = find_contents(reads)[0]
    t = time.monotonic()
    outcome = await split_issue(blocks, reads, llm, cfg.publication, model=cfg.split_model, spend=spend)
    result.secs["split"] = time.monotonic() - t
    result.costs["split"] = outcome.cost
    result.split, result.findings = outcome.result, outcome.findings.summary()
    result.first_findings, result.split_rounds = outcome.first_findings, outcome.rounds
    result.articles = asm.assemble(outcome.result, blocks, cfg.max_article_tokens)
    result.issue = asm.issue_info(outcome.result, cfg.source_prefix)
    result.flags = [f for p in pages.values() for f in p.flags] + asm.leftover_flags(outcome.findings, blocks,
                                                                                       outcome.result)
    if outcome.error:
        result.flags.append(outcome.error)
    if result.contents_page is None:
        result.flags.append("no contents page found, so articles couldn't be checked against it")
    if not result.issue.year or not result.issue.months:
        result.flags.append(f"couldn't tell the issue's month from “{result.issue.label}”: check the source name")
    return result


async def process_issue(pdf_path: str, store: PageStore, llm, cfg: IssueConfig, *, spend: Spend | None = None,
                        summaries: bool = True, progress: Progress = None, recheck_stored: bool = False,
                        on_page: OnPage = None) -> IssueResult:
    t = time.monotonic()
    spend = spend if spend is not None else Spend()
    before = spend.usd
    pages = await read_pages(pdf_path, store, llm, cfg, spend=spend, progress=progress, recheck_stored=recheck_stored,
                             on_page=on_page)
    result = IssueResult(pages=pages)
    result.secs["read"] = time.monotonic() - t
    result.costs["read"] = sum(p.cost for p in pages.values())
    if progress:
        await progress("stage", {"stage": "split"})
    await split_and_assemble(pages, llm, cfg, spend=spend, result=result)
    if summaries:
        if progress:
            await progress("stage", {"stage": "summaries"})
        t = time.monotonic()
        mark = spend.usd
        await asm.summarize(result.articles, llm, cfg.publication, model=cfg.summary_model, spend=spend)
        result.secs["summaries"] = time.monotonic() - t
        result.costs["summaries"] = spend.usd - mark
    result.run_cost = spend.usd - before  # page reads loaded from the store cost nothing this time
    return result
