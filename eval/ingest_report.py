"""Report on an ingest eval run (eval/ingest_eval.py): .cache/ingest-eval/REPORT.local.md (git-ignored).

Per issue: pages, articles by kind, contents entries matched, coverage, pages re-read by Sonnet, flags, cost,
time; totals and medians; for issues the original knowledge base already has (2023 – Jan 2025), old vs new article
counts and which contents entries the old articles lack. Page flags are recomputed from the stored reads with the
current check (no API calls). Hand-written sections (failure patterns, spot checks) come from
.cache/ingest-eval/notes.local.md if it exists.
"""
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rapidfuzz import fuzz  # noqa: E402
from unidecode import unidecode  # noqa: E402

from ingest.assemble import leftover_flags  # noqa: E402
from ingest.pages import PageResult, recheck  # noqa: E402
from ingest.pdf import PdfFile  # noqa: E402
from ingest.split import check_split, content_blocks, find_contents, page_offset  # noqa: E402

OUT = ROOT / ".cache" / "ingest-eval"
KINDS = ["article", "notice", "photo_page", "poem", "cover", "contents", "other"]


def load_results() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((OUT / "results").glob("*.json"))]


def issue_key(r: dict) -> tuple:
    issue = r.get("issue") or {}
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    m = (issue.get("months") or [None])[0]
    return (issue.get("year") or 0, months.index(m) if m in months else 0, r["name"])


def rechecked_flags(r: dict) -> tuple[int, list[str]]:
    """Page flags with the current check, from the stored reads."""
    flagged, flags = 0, []
    with PdfFile(r["pdf"]) as pdf:
        for f in sorted((OUT / "reads" / r["name"]).glob("p*.json")):
            page = recheck(pdf, PageResult.from_dict(json.loads(f.read_text())))
            if page.flags:
                flagged += 1
                flags += page.flags
    return flagged, flags


def legacy_articles() -> dict[str, list[str]]:
    out = defaultdict(list)
    for line in open(ROOT / "knowledge_base.jsonl"):
        m = json.loads(line)["metadata"]
        out[m["source"]].append(m["title"])
    return out


def legacy_source(r: dict, legacy: dict) -> str | None:
    issue = r.get("issue") or {}
    if not issue.get("year") or not issue.get("months"):
        return None
    name = f"Sunbeam-{issue['year']}-{issue['months'][0]}.txt"
    return name if name in legacy else None


def norm(t: str) -> str:
    return unidecode(t or "").lower()


def fmt_kinds(kinds: dict) -> str:
    return ", ".join(f"{kinds[k]} {k.replace('_', ' ')}" for k in KINDS if kinds.get(k))


def main():
    results = load_results()
    legacy = legacy_articles()
    rows, website = [], []
    for r in sorted(results, key=issue_key):
        s = dict(r.get("summary") or {})
        flagged, page_flags = rechecked_flags(r)
        # The split as stored, checked with the current checker.
        reads = {int(p.stem[1:]): json.loads(p.read_text())["read"] for p in (OUT / "reads" / r["name"]).glob("p*.json")}
        blocks = content_blocks(reads)
        _, _, entries = find_contents(reads)
        findings = check_split(r["split"], blocks, entries, page_offset(reads))
        s.update(contents_matched=findings.contents_matched, contents_total=findings.contents_total,
                 unassigned=len(findings.unassigned), doubled=len(findings.doubled))
        split_flags = leftover_flags(findings, blocks, r["split"]) + [
            f for f in r.get("flags") or [] if f.startswith(("repair round", "no contents page", "couldn't tell"))]
        cost = (r.get("costs") or {}).get("read", 0) + (r.get("costs") or {}).get("split", 0)
        cached = r.get("run_cost", 0) + 1e-6 < cost
        secs = r.get("secs") or {}
        excluded = sum(1 for a in r.get("articles") or [] if a["excluded"])
        row = {"name": r["name"], "label": (r.get("issue") or {}).get("label", ""),
               "source": (r.get("issue") or {}).get("source", ""), "pages": s.get("pages"),
               "articles": s.get("articles"), "excluded": excluded, "kinds": s.get("kinds") or {},
               "matched": s.get("contents_matched"), "total": s.get("contents_total"),
               "unassigned": s.get("unassigned"), "doubled": s.get("doubled"), "reread": s.get("reread_pages"),
               "page_flags": flagged, "split_flags": len(split_flags), "flags": page_flags + split_flags,
               "rounds": r.get("split_rounds"), "cost": cost, "read_s": secs.get("read"),
               "split_s": secs.get("split"), "cached": cached, "result": r}
        (website if r["name"].startswith("website-") else rows).append(row)

    def table(rs):
        lines = ["| Issue | Pages | Articles (by kind) | Excl. | Contents matched | Unassigned / double | "
                 "Re-read by Sonnet | Flags (page / split) | Split rounds | Cost | Read + split time |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in rs:
            contents = f"{x['matched']}/{x['total']}" if x["total"] else "no contents page"
            t = f"{x['read_s']:.0f}{'†' if x['cached'] else ''} + {x['split_s']:.0f} s"
            lines.append(f"| {x['label'] or x['name']} | {x['pages']} | {x['articles']} ({fmt_kinds(x['kinds'])}) | "
                         f"{x['excluded']} | {contents} | {x['unassigned']} / {x['doubled']} | {x['reread']} | "
                         f"{x['page_flags']} / {x['split_flags']} | {x['rounds']} | ${x['cost']:.3f} | {t} |")
        return "\n".join(lines)

    def med(key, rs):
        vals = [x[key] for x in rs if x[key] is not None]
        return statistics.median(vals) if vals else 0

    pages = sum(x["pages"] for x in rows)
    with_toc = [x for x in rows if x["total"]]
    matched, total = sum(x["matched"] for x in with_toc), sum(x["total"] for x in with_toc)
    fully = sum(1 for x in with_toc if x["matched"] == x["total"])
    rereads = sum(x["reread"] for x in rows)
    kinds = Counter()
    for x in rows:
        kinds.update(x["kinds"])
    totals = f"""**{len(rows)} born-digital issues** ({min(issue_key(x['result'])[0] for x in rows)}–{max(issue_key(x['result'])[0] for x in rows)}): {pages:,} pages, {sum(x['articles'] for x in rows):,} articles ({fmt_kinds(kinds)}), {sum(x['excluded'] for x in rows)} more excluded at assembly (near-empty).

| | Total | Median per issue |
|---|---|---|
| Pages | {pages:,} | {med('pages', rows)} |
| Articles | {sum(x['articles'] for x in rows):,} | {med('articles', rows)} |
| Contents entries matched | {matched}/{total} ({matched / max(total, 1):.1%}); {fully}/{len(with_toc)} issues all matched | – |
| Blocks unassigned / double-assigned | {sum(x['unassigned'] for x in rows)} / {sum(x['doubled'] for x in rows)} | 0 / 0 |
| Pages re-read by Sonnet | {rereads} ({rereads / pages:.1%} of pages) | {med('reread', rows)} |
| Pages flagged (final check) | {sum(x['page_flags'] for x in rows)} | {med('page_flags', rows)} |
| Split flags (contents entries without an article, leftovers) | {sum(x['split_flags'] for x in rows)} | {med('split_flags', rows)} |
| Issues needing the repair round | {sum(1 for x in rows if x['rounds'] == 2)}/{len(rows)} | – |
| Cost (page reads + split; no summaries) | ${sum(x['cost'] for x in rows):.2f} | ${med('cost', rows):.3f} (${sum(x['cost'] for x in rows) / pages * 60:.3f} per 60 pages) |
| Read + split time | – | {med('read_s', [x for x in rows if not x['cached']]):.0f} + {med('split_s', rows):.0f} s |
"""

    # old vs new
    cmp_lines = ["| Issue | Old articles | New articles | Contents entries the old articles lack |", "|---|---|---|---|"]
    for x in rows:
        src = legacy_source(x["result"], legacy)
        if not src:
            continue
        old_titles = legacy[src]
        reads = {int(p.stem[1:]): json.loads(p.read_text())["read"] for p in (OUT / "reads" / x["name"]).glob("p*.json")}
        _, _, entries = find_contents(reads)
        lacking = [e.title for e in entries
                   if not any(fuzz.token_set_ratio(norm(e.title), norm(t)) >= 75 for t in old_titles)]
        cmp_lines.append(f"| {x['label']} ({src}) | {len(old_titles)} | {x['articles']} | "
                         f"{len(lacking)}/{len(entries)}: {'; '.join(lacking) or '–'} |")

    flag_lines = []
    for x in rows + website:
        for f in x["flags"]:
            flag_lines.append(f"- {x['label'] or x['name']}: {f}")

    ledger = [json.loads(line) for line in open(ROOT / "eval" / "results" / "spend_ledger.local.jsonl")]
    spent = sum(r["usd"] for r in ledger if str(r.get("label", "")).startswith("ingest"))
    notes = (OUT / "notes.local.md").read_text() if (OUT / "notes.local.md").exists() else ""
    report = f"""# Ingest eval: read → check → split on {len(rows)} born-digital issues and {len(website)} website copies

Generated by `eval/ingest_report.py` from `.cache/ingest-eval/results/` (local only: these files hold magazine
text). Pipeline: Haiku 5.5 page reads (thinking off) checked against the PDF text layer (Sonnet 5.5 re-read below
97% word recall), Sonnet 5.5 split at default effort with one repair round, assembly. No summaries, no publish.
Page flags, contents matches and coverage are recomputed from the stored reads and splits with the final
check and checker (both changed during the run; "split rounds" shows what the run did). † = some page reads were cached from an
earlier run, so the read time shown is partial. Costs are what the final reads and split cost; the ingest
eval's ledger total (including earlier splits redone while tuning) is ${spent:.2f}.

## Totals

{totals}
## Per issue (born-digital)

{table(rows)}

## Website copies (stress test)

{table(website)}

## Issues in today's knowledge base: old vs new

{chr(10).join(cmp_lines)}

## All flags

{chr(10).join(flag_lines) or 'None.'}

{notes}
"""
    (OUT / "REPORT.local.md").write_text(report)
    print(f"wrote {OUT / 'REPORT.local.md'} ({len(rows)} issues + {len(website)} website copies)")


if __name__ == "__main__":
    main()
