"""Summarise eval results: latency, cost, tokens, tool use, retrieval and citation quality.

  .venv/bin/python eval/report.py eval/results/baseline.local.jsonl            # summary
  .venv/bin/python eval/report.py eval/results/baseline.local.jsonl --per-question
  .venv/bin/python eval/report.py baseline.local.jsonl new.local.jsonl         # side-by-side comparison
  .venv/bin/python eval/report.py --spend                                      # API spend across all runs
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "results" / "spend_ledger.local.jsonl"

SOURCES_HEADER = re.compile(r"(?im)^[ \t]*(?:#+[ \t]*)?\**[ \t]*(?:sources|references)[ \t]*:?[ \t]*\**[ \t]*:?[ \t]*$")
SOURCE_LINE = re.compile(r"^\s*[-*]?\s*\[(\d+)\]\s*(.+?)\s*$")
# The prescribed `[n] <title>, <document> page <start>-<end>`; a single page ("page 10") counts as start == end.
SOURCE_FORMAT = re.compile(r"^(?P<title>.+),\s+(?P<doc>[^\s,]+)\s+page\s+(?P<p1>\d+(?:\.\d+)?)(?:\s*[-–]\s*(?P<p2>\d+(?:\.\d+)?))?\s*$",
                           re.I)
# Looser parse used only to check grounding: tolerates "<document>, page", "pages", "pp." and a missing ".txt".
SOURCE_LENIENT = re.compile(r"^(?P<title>.+),\s*(?P<doc>[^\s,]+?),?\s+(?:pages?|pp?\.)\s*(?P<p1>\d+(?:\.\d+)?)(?:\s*[-–]\s*(?P<p2>\d+(?:\.\d+)?))?",
                            re.I)


def parse_source_line(text: str):
    return SOURCE_FORMAT.match(text) or SOURCE_LENIENT.match(text)


def _doc(name) -> str:
    return re.sub(r"\.txt$", "", _norm(name))
INLINE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")


def _expand(token: str) -> list[int]:
    out = []
    for part in re.split(r"\s*,\s*", token):
        if re.search(r"[–-]", part):
            a, b = (int(x) for x in re.split(r"\s*[–-]\s*", part)[:2])
            out.extend(range(a, b + 1) if b >= a and b - a < 50 else [a, b])
        elif part.strip():
            out.append(int(part))
    return out


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _page(p) -> str:
    try:
        return str(int(float(p)))
    except (TypeError, ValueError):
        return str(p)


def split_sources(answer: str) -> tuple[str, str, bool]:
    """(body, sources block, english_header). Uses the last "sources:" header; otherwise a trailing block of
    `[n] ...` lines, optionally under a header line in another language (e.g. "**ಮೂಲಗಳು:**")."""
    headers = list(SOURCES_HEADER.finditer(answer))
    if headers:
        return answer[: headers[-1].start()], answer[headers[-1].end():], True
    lines = answer.rstrip().splitlines()
    i = len(lines)
    while i > 0 and (SOURCE_LINE.match(lines[i - 1]) or not lines[i - 1].strip()):
        i -= 1
    if i == len(lines):
        return answer, "", False
    j = i - 1 if i > 0 and lines[i - 1].strip().strip("*#").strip().endswith(":") else i
    return "\n".join(lines[:j]), "\n".join(lines[i:]), False


def analyze_citations(answer: str, seen_docs: list[dict]) -> dict:
    """Citation checks against the system prompt's rules and against the docs the agent actually saw."""
    answer = answer or ""
    body, tail, english_header = split_sources(answer)
    lines = [m for m in (SOURCE_LINE.match(line) for line in tail.splitlines()) if m]
    indices = [int(m.group(1)) for m in lines]
    inline = sorted({i for m in INLINE.finditer(body) for i in _expand(m.group(1))})
    strict = [SOURCE_FORMAT.match(m.group(2)) for m in lines]
    parsed = [parse_source_line(m.group(2)) for m in lines]
    seen_sources = {_doc(d.get("source")) for d in seen_docs}
    seen_titles = {(_doc(d.get("source")), _norm(d.get("title"))) for d in seen_docs}
    seen_pages = {(_doc(d.get("source")), _page(d.get("start_page")), _page(d.get("end_page"))) for d in seen_docs}
    keys = [(_norm(p["title"]), _doc(p["doc"])) if p else _norm(m.group(2)) for p, m in zip(parsed, lines)]
    doc_ok = [bool(p) and _doc(p["doc"]) in seen_sources for p in parsed]
    title_ok = [bool(p) and (_doc(p["doc"]), _norm(p["title"])) in seen_titles for p in parsed]
    pages_ok = [bool(p) and (_doc(p["doc"]), _page(p["p1"]), _page(p["p2"] or p["p1"])) in seen_pages for p in parsed]
    n = len(lines)
    return {
        "has_sources": n > 0,
        "english_sources_header": english_header,
        "n_sources": n,
        "n_inline": len(inline),
        "sequential": n > 0 and indices == list(range(1, n + 1)),
        "duplicates": n - len(set(keys)),
        "format_ok": n > 0 and all(strict),
        "inline_in_sources": set(inline) <= set(indices),
        "sources_cited_inline": set(indices) <= set(inline),
        "doc_grounded": n > 0 and all(doc_ok),
        "title_grounded": n > 0 and all(title_ok),
        "pages_grounded": n > 0 and all(pages_ok),
        "frac_doc_grounded": (sum(doc_ok) / n) if n else None,
        "float_pages": bool(re.search(r"page\s+\d+\.0\s*[-–]", tail)),
    }


def pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):.0f}% ({sum(values)}/{len(values)})" if values else "n/a"


def q(values: list[float], p: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def fmt(v, nd=1) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def turn_metrics(rec: dict) -> list[dict]:
    rows, seen = [], []
    for t in rec["turns"]:
        for tc in t.get("tool_calls") or []:
            seen.extend(tc.get("kept_docs") or [])
        tools = t.get("tool_calls") or []
        readers = [g for tc in tools for g in tc.get("llm_groups") or [] if g.get("schema") != "Queries"]
        row = {
            "id": rec["id"], "turn": t["turn"], "category": rec.get("category"), "error": bool(t.get("error")),
            "latency_s": t.get("latency_s"), "visible_s": t.get("time_to_visible_answer_s"),
            "cost_usd": t.get("cost_usd") or 0.0,
            "n_tool_calls": len(tools),
            "queries_per_call": [len(tc.get("queries") or []) for tc in tools],
            "retrieved_per_call": [sum(len(r.get("docs") or []) for r in tc.get("retrieved") or []) for tc in tools],
            "unique_per_call": [len({(d.get("source"), d.get("title"), _page(d.get("start_page")))
                                     for r in tc.get("retrieved") or [] for d in r.get("docs") or []}) for tc in tools],
            "kept_per_call": [len(tc.get("kept_docs") or []) for tc in tools],
            "dup_kept_per_call": [len(tc.get("kept_docs") or []) - len({(d.get("source"), d.get("title"))
                                                                         for d in tc.get("kept_docs") or []}) for tc in tools],
            "reader_calls": sum(g["calls"] for g in readers),
            "reader_errors": sum(g["errors"] for g in readers),
            "tool_latency": [tc.get("latency_s") for tc in tools],
            "retrieval_wall": [tc.get("retrieval_wall_s") for tc in tools],
            "reader_wall": [g.get("wall_s") for g in readers],
            "final_answer_tags": t.get("final_answer_tags"),
            "streamed_equals_saved": t.get("streamed_equals_saved"),
            "answer_chars": len(t.get("answer") or ""),
            "is_command": "command_output" in (t.get("answer_tags") or []),
            "llm_calls": t.get("llm_calls") or [],
            "cit": analyze_citations(t.get("answer") or "", seen),
        }
        rows.append(row)
    return rows


def summarize(recs: list[dict]) -> dict:
    turns = [m for r in recs for m in turn_metrics(r)]
    llm_turns = [m for m in turns if not m["is_command"]]
    searched = [m for m in llm_turns if m["n_tool_calls"] and not m["error"]]
    by_model = defaultdict(lambda: Counter())
    for m in llm_turns:
        for c in m["llm_calls"]:
            k = by_model[c.get("model_canonical") or ("(errored call)" if c.get("error") else "?")]
            k["calls"] += 1
            for f in ("input_tokens", "output_tokens", "cache_read", "reasoning_tokens"):
                k[f] += c.get(f) or 0
            k["cost_millicents"] += round((c.get("cost_usd") or 0) * 1e5)
    qlat = [r["latency_s"] for r in recs if not all(m["is_command"] for m in turn_metrics(r))]
    qcost = [r["cost_usd"] for r in recs if not all(m["is_command"] for m in turn_metrics(r))]
    calls = [x for m in llm_turns for x in m["queries_per_call"]]
    flat = lambda key: [x for m in llm_turns for x in m[key] if x is not None]  # noqa: E731
    return {
        "questions": len(recs), "turns": len(turns), "llm_turns": len(llm_turns),
        "errors": sum(m["error"] for m in turns),
        "latency_q": {"mean": statistics.mean(qlat) if qlat else None, "p50": q(qlat, .5), "p90": q(qlat, .9), "max": max(qlat, default=None)},
        "latency_turn": {"p50": q([m["latency_s"] for m in llm_turns], .5), "p90": q([m["latency_s"] for m in llm_turns], .9)},
        "visible_turn": {"p50": q([m["visible_s"] for m in llm_turns], .5), "p90": q([m["visible_s"] for m in llm_turns], .9)},
        "cost_q": {"mean": statistics.mean(qcost) if qcost else None, "p50": q(qcost, .5), "p90": q(qcost, .9),
                   "max": max(qcost, default=None), "total": sum(qcost)},
        "by_model": {k: dict(v, cost_usd=v["cost_millicents"] / 1e5) for k, v in by_model.items()},
        "tool_calls_per_turn": {"mean": statistics.mean([m["n_tool_calls"] for m in llm_turns]) if llm_turns else None,
                                "max": max((m["n_tool_calls"] for m in llm_turns), default=None),
                                "zero": sum(1 for m in llm_turns if not m["n_tool_calls"])},
        "per_search_call": {
            "calls": len(calls),
            "queries_mean": statistics.mean(calls) if calls else None,
            "retrieved_mean": statistics.mean(flat("retrieved_per_call")) if calls else None,
            "unique_mean": statistics.mean(flat("unique_per_call")) if calls else None,
            "kept_mean": statistics.mean(flat("kept_per_call")) if calls else None,
            "dup_kept_total": sum(flat("dup_kept_per_call")),
            "latency_p50": q(flat("tool_latency"), .5), "latency_p90": q(flat("tool_latency"), .9),
            "retrieval_p50": q(flat("retrieval_wall"), .5), "reader_wall_p50": q(flat("reader_wall"), .5),
        },
        "reader_calls": sum(m["reader_calls"] for m in llm_turns),
        "reader_errors": sum(m["reader_errors"] for m in llm_turns),
        "final_answer_tags": pct([bool(m["final_answer_tags"]) for m in llm_turns if not m["error"]]),
        "streamed_equals_saved": pct([m["streamed_equals_saved"] for m in llm_turns if m["streamed_equals_saved"] is not None]),
        "citations (turns that searched)": {
            "has_sources": pct([m["cit"]["has_sources"] for m in searched]),
            "sequential_from_1": pct([m["cit"]["sequential"] for m in searched if m["cit"]["n_sources"]]),
            "no_duplicate_sources": pct([m["cit"]["duplicates"] == 0 for m in searched if m["cit"]["n_sources"]]),
            "format_ok": pct([m["cit"]["format_ok"] for m in searched if m["cit"]["n_sources"]]),
            "inline_refs_in_sources": pct([m["cit"]["inline_in_sources"] for m in searched if m["cit"]["n_sources"]]),
            "all_sources_cited_inline": pct([m["cit"]["sources_cited_inline"] for m in searched if m["cit"]["n_sources"]]),
            "all_sources_doc_grounded": pct([m["cit"]["doc_grounded"] for m in searched if m["cit"]["n_sources"]]),
            "all_sources_title_grounded": pct([m["cit"]["title_grounded"] for m in searched if m["cit"]["n_sources"]]),
            "all_sources_pages_grounded": pct([m["cit"]["pages_grounded"] for m in searched if m["cit"]["n_sources"]]),
            "float_page_artifacts": sum(m["cit"]["float_pages"] for m in searched),
            "sources_per_answer_mean": statistics.mean([m["cit"]["n_sources"] for m in searched]) if searched else None,
        },
        "kb_docs_mutated_per_question_mean": statistics.mean([r.get("kb_docs_mutated") or 0 for r in recs]) if recs else None,
    }


def print_summary(name: str, s: dict) -> None:
    print(f"\n## {name}\n")
    print(f"- questions {s['questions']} ({s['turns']} turns, {s['llm_turns']} LLM turns), errors {s['errors']}")
    lq, lt, vt = s["latency_q"], s["latency_turn"], s["visible_turn"]
    print(f"- latency per question: mean {fmt(lq['mean'])}s, p50 {fmt(lq['p50'])}s, p90 {fmt(lq['p90'])}s, max {fmt(lq['max'])}s")
    print(f"- latency per turn: p50 {fmt(lt['p50'])}s, p90 {fmt(lt['p90'])}s; first visible answer text: p50 {fmt(vt['p50'])}s, p90 {fmt(vt['p90'])}s")
    c = s["cost_q"]
    print(f"- cost per question: mean ${fmt(c['mean'], 4)}, p50 ${fmt(c['p50'], 4)}, p90 ${fmt(c['p90'], 4)}, max ${fmt(c['max'], 4)}, total ${fmt(c['total'], 3)}")
    for model, v in sorted(s["by_model"].items(), key=lambda kv: -kv[1]["cost_usd"]):
        print(f"  - {model}: {v['calls']} calls, in {v['input_tokens']:,} (cache read {v['cache_read']:,}), out {v['output_tokens']:,}"
              f" (reasoning {v['reasoning_tokens']:,}), ${v['cost_usd']:.3f}")
    t = s["tool_calls_per_turn"]
    print(f"- search calls per LLM turn: mean {fmt(t['mean'], 2)}, max {t['max']}, turns with none {t['zero']}")
    p = s["per_search_call"]
    print(f"- per search call ({p['calls']}): {fmt(p['queries_mean'])} queries, {fmt(p['retrieved_mean'])} docs retrieved,"
          f" {fmt(p['unique_mean'])} unique, {fmt(p['kept_mean'])} kept after reading; duplicate articles kept: {p['dup_kept_total']}")
    print(f"  - search latency p50 {fmt(p['latency_p50'])}s / p90 {fmt(p['latency_p90'])}s; retrieval p50 {fmt(p['retrieval_p50'])}s; reading p50 {fmt(p['reader_wall_p50'])}s")
    print(f"- reader calls {s['reader_calls']}, reader errors {s['reader_errors']}")
    print(f"- <final_answer> tags present: {s['final_answer_tags']}; streamed text == saved: {s['streamed_equals_saved']}")
    print("- citations:")
    for k, v in s["citations (turns that searched)"].items():
        print(f"  - {k}: {fmt(v, 2) if isinstance(v, float) else v}")
    print(f"- knowledge-base docs left modified per question (shared-state bug): {fmt(s['kb_docs_mutated_per_question_mean'], 1)}")


def per_question(recs: list[dict]) -> None:
    print("\n| id | category | turns | latency s | cost $ | searches | kept | sources | seq | grounded | tags | error |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        ms = turn_metrics(r)
        last = ms[-1]
        print(f"| {r['id']} | {r.get('category')} | {len(ms)} | {r['latency_s']:.1f} | {r['cost_usd']:.3f} | "
              f"{sum(m['n_tool_calls'] for m in ms)} | {sum(sum(m['kept_per_call']) for m in ms)} | {last['cit']['n_sources']} | "
              f"{'y' if last['cit']['sequential'] else '-'} | {fmt(last['cit']['frac_doc_grounded'], 2)} | "
              f"{'y' if last['final_answer_tags'] else '-'} | {'ERR' if any(m['error'] for m in ms) else ''} |")


def compare(a_name: str, a: list[dict], b_name: str, b: list[dict]) -> None:
    print_summary(a_name, summarize(a))
    print_summary(b_name, summarize(b))
    bi = {r["id"]: r for r in b}
    print(f"\n| id | latency {a_name} | latency {b_name} | cost {a_name} | cost {b_name} | sources | grounded |")
    print("|---|---|---|---|---|---|---|")
    for r in a:
        o = bi.get(r["id"])
        if not o:
            continue
        ca, cb = turn_metrics(r)[-1]["cit"], turn_metrics(o)[-1]["cit"]
        print(f"| {r['id']} | {r['latency_s']:.1f} | {o['latency_s']:.1f} | {r['cost_usd']:.3f} | {o['cost_usd']:.3f} | "
              f"{ca['n_sources']} → {cb['n_sources']} | {fmt(ca['frac_doc_grounded'], 2)} → {fmt(cb['frac_doc_grounded'], 2)} |")


def spend() -> None:
    rows = [json.loads(line) for line in LEDGER.open()] if LEDGER.exists() else []
    by_label = defaultdict(float)
    notes = {}
    for r in rows:
        by_label[r["label"]] += r["usd"]
        if r.get("note"):
            notes[r["label"]] = r["note"]
    for label, usd in sorted(by_label.items()):
        print(f"{label:<20} ${usd:.3f}" + (f"  [{notes[label]}]" if label in notes else ""))
    print(f"{'TOTAL':<20} ${sum(by_label.values()):.3f}  ({sum(1 for r in rows if r['id'] != 'SUPERSEDED')} question runs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--per-question", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--spend", action="store_true")
    args = ap.parse_args()
    if args.spend:
        spend()
        return
    if len(args.files) == 2:
        compare(Path(args.files[0]).stem, load(args.files[0]), Path(args.files[1]).stem, load(args.files[1]))
        return
    recs = load(args.files[0])
    s = summarize(recs)
    if args.json:
        print(json.dumps(s, indent=1, default=str))
        return
    print_summary(Path(args.files[0]).stem, s)
    if args.per_question:
        per_question(recs)


if __name__ == "__main__":
    main()
