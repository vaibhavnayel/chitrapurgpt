"""Run the ingest pipeline (read -> check -> split -> assemble; no summaries, no publish) on local PDFs.

  .venv/bin/python eval/ingest_eval.py --budget-usd 9 ".cache/sunbeam_pdfs/20*/*.pdf"
  .venv/bin/python eval/ingest_report.py          # -> .cache/ingest-eval/REPORT.local.md

Page reads are cached under .cache/ingest-eval/reads/<issue>/ (a re-run only pays for pages not read yet); each
issue's result goes to .cache/ingest-eval/results/<issue>.json and its cost to the spend ledger (label
"ingest-eval"). An issue isn't started if the phase's ingest spend so far plus its estimated cost would pass
--budget-usd. Everything stays in .cache/ (git-ignored): the PDFs and their text are not for the public repo.
"""
import argparse
import asyncio
import glob
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from ingest.llm import AnthropicLLM, Spend  # noqa: E402
from ingest.pages import publication_phrase  # noqa: E402
from ingest.pdf import page_count  # noqa: E402
from ingest.pipeline import FilePageStore, IssueConfig, process_issue  # noqa: E402

OUT = ROOT / ".cache" / "ingest-eval"
LEDGER = ROOT / "eval" / "results" / "spend_ledger.local.jsonl"
SUNBEAM = IssueConfig(publication=publication_phrase("The Chitrapur Sunbeam", "the magazine of Shri Chitrapur Math"),
                      source_prefix="Sunbeam")


def issue_name(path: str) -> str:
    parent = Path(path).parent.name
    stem = re.sub(r"[^A-Za-z0-9]+", "-", Path(path).stem).strip("-")
    return f"{parent}-{stem}" if parent == "website" else stem


def ingest_spend(label_prefix: str = "ingest") -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(r.get("usd", 0) for r in map(json.loads, LEDGER.read_text().splitlines())
               if str(r.get("label", "")).startswith(label_prefix))


def ledger(label: str, item: str, usd: float) -> None:
    with LEDGER.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": label,
                            "id": item, "usd": round(usd, 6)}) + "\n")


async def run_one(path: str, llm, args, lock: asyncio.Lock) -> dict | None:
    name = issue_name(path)
    out = OUT / "results" / f"{name}.json"
    if out.exists() and not args.redo:
        print(f"  {name}: done already", flush=True)
        return json.loads(out.read_text())
    pages = page_count(path)
    estimate = pages * 0.0016 + 0.20
    async with lock:
        spent = ingest_spend()
        if spent + estimate > args.budget_usd:
            print(f"  {name}: skipped (ingest spend ${spent:.2f} + estimate ${estimate:.2f} > ${args.budget_usd})",
                  flush=True)
            return None
    spend = Spend()
    t = time.monotonic()
    store = FilePageStore(OUT / "reads" / name)
    try:
        result = await process_issue(path, store, llm, SUNBEAM, spend=spend, summaries=args.summaries,
                                     recheck_stored=True)
        error = None
    except Exception as e:  # noqa: BLE001
        result, error = None, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    wall = time.monotonic() - t
    ledger("ingest-eval", name, spend.usd)
    rec = {"name": name, "pdf": path, "wall_s": round(wall, 1), "run_cost": round(spend.usd, 4), "error": error}
    if result is not None:
        rec.update(summary=result.summary(), issue=vars(result.issue) if result.issue else None,
                   contents_page=result.contents_page, findings=result.findings,
                   first_findings=result.first_findings, split_rounds=result.split_rounds, flags=result.flags,
                   costs={k: round(v, 5) for k, v in result.costs.items()},
                   secs={k: round(v, 1) for k, v in result.secs.items()},
                   pages=[{k: v for k, v in p.as_dict().items() if k != "read"} for p in result.pages.values()],
                   split=result.split, articles=[a.as_dict() for a in result.articles], calls=spend.calls)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    s = rec.get("summary") or {}
    print(f"  {name}: {s.get('pages')} pages, {s.get('articles')} articles {s.get('kinds')}, contents "
          f"{s.get('contents_matched')}/{s.get('contents_total')}, unassigned {s.get('unassigned')}, doubled "
          f"{s.get('doubled')}, re-read {s.get('reread_pages')}, flags {s.get('flags')}, rounds "
          f"{rec.get('split_rounds')}, ${spend.usd:.3f} this run, {wall:.0f}s{', ERROR ' + error if error else ''}",
          flush=True)
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", help="PDF paths or globs")
    ap.add_argument("--budget-usd", type=float, required=True, help="cap on all ingest spend in the ledger")
    ap.add_argument("--issues-at-once", type=int, default=2)
    ap.add_argument("--summaries", action="store_true", help="also summarise articles (the eval doesn't need it)")
    ap.add_argument("--redo", action="store_true", help="split again even if a result exists (reads stay cached)")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    paths = sorted({p for pat in args.pdfs for p in glob.glob(pat)})
    print(f"{len(paths)} PDFs; ingest spend so far ${ingest_spend():.2f}, budget ${args.budget_usd}", flush=True)
    llm = AnthropicLLM()
    lock, sem = asyncio.Lock(), asyncio.Semaphore(args.issues_at_once)

    async def guarded(p):
        async with sem:
            return await run_one(p, llm, args, lock)

    t = time.monotonic()
    await asyncio.gather(*(guarded(p) for p in paths))
    print(f"done in {time.monotonic() - t:.0f}s; ingest spend now ${ingest_spend():.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
