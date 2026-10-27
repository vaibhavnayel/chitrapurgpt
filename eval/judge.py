"""Blind pairwise judgement of two eval runs with claude-opus-5-5: answer quality and faithfulness to the passages
each stack actually retrieved. Response order is randomised per question (seeded) and the judge never sees which
stack wrote what. Each judgement's cost goes to the spend ledger.

  .venv/bin/python eval/judge.py A.local.jsonl B.local.jsonl --out eval/results/judge_A_vs_B.local.jsonl
  .venv/bin/python eval/judge.py ... --summary-only      # re-print the summary of an existing --out file
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import dotenv  # noqa: E402

dotenv.load_dotenv(HERE.parent / ".env")

from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from pricing import cost_usd  # noqa: E402
from report import SOURCE_LINE, parse_source_line, split_sources  # noqa: E402

# One doc in retrievers.format_docs output (the text the agent saw).
DOC_RE_CONTENT = re.compile(r"(?ms)^(\d+)\. (?P<title>.*?)\n Metadata:\n(?P<meta>\{.*?\})\nContent:\n(?P<content>.*?)\n-{100}$")

JUDGE_MODEL = "claude-opus-5-5"
EVIDENCE_CHARS = 24_000
LEDGER = HERE / "results" / "spend_ledger.local.jsonl"

RUBRIC = """You are grading two responses from ChitrapurGPT, a research assistant for the Chitrapur Saraswat community.
Its rules: answer community questions only from passages retrieved from a knowledge base of magazine articles (the
"Chitrapur Sunbeam"), never from its own knowledge; cite sources Wikipedia-style ([1], [2]) with a final
"sources:" list in the form "[n] <title>, <document> page <start>-<end>", numbered from 1 with no duplicates; it may
translate and transliterate; it may read, translate or explain files the user attaches; if the knowledge base
doesn't cover something it should say so.

Each response is shown with the passages ITS OWN search retrieved (the two may have retrieved different passages;
they are summaries/quotes extracted from the articles). Judge each response against its own passages.

Score each response from 1 to 5 on:
- faithfulness: every community fact is supported by that response's passages; citations point to passages that
  support the claim; nothing is invented; an honest "not found" is faithful. Facts that are general knowledge
  rather than community facts (e.g. a transliteration, a translation, a general definition the question asked to
  be rewritten) don't count against it.
- quality: answers what was asked, completely and clearly, in the language/script/length requested, with correct
  citation formatting.
Then pick the better response overall (faithfulness matters most, then quality), or "tie" if they're equally good.
The responses are labelled 1 and 2 in random order; their length, style and order must not sway you."""


class Verdict(BaseModel):
    analysis: str = Field(description="brief comparison of the two responses: the key facts, any unsupported claims or citation problems")
    faithfulness_1: int = Field(description="1-5")
    faithfulness_2: int = Field(description="1-5")
    quality_1: int = Field(description="1-5")
    quality_2: int = Field(description="1-5")
    winner: str = Field(description='"1", "2" or "tie"')


def evidence(rec: dict, turn_index: int, answer: str) -> str:
    """Passages this stack saw up to and including the judged turn: cited sources first, capped."""
    docs = []
    for t in rec["turns"][: turn_index + 1]:
        for tc in t.get("tool_calls") or []:
            for m in DOC_RE_CONTENT.finditer(tc.get("output") or ""):
                docs.append((m.group("title"), m.group("meta"), m.group("content")))
    _, tail, _ = split_sources(answer or "")
    cited = set()
    for line in tail.splitlines():
        mm = SOURCE_LINE.match(line)
        p = parse_source_line(mm.group(2)) if mm else None
        if p:
            cited.add(re.sub(r"\.txt$", "", p["doc"].lower()))
    ordered = sorted(docs, key=lambda d: not any(c in d[1].lower() for c in cited))
    out, total, seen = [], 0, set()
    for title, meta, content in ordered:
        try:
            md = ast.literal_eval(meta)
        except (ValueError, SyntaxError):
            md = {}
        key = (title, md.get("source"), content[:200])
        if key in seen:
            continue
        seen.add(key)
        block = (f"- {title} | {md.get('source', '?')} | pages {md.get('start_page', '?')}-{md.get('end_page', '?')}"
                 f" | published {md.get('published_date', '?')}\n{content.strip()}\n")
        if total + len(block) > EVIDENCE_CHARS:
            out.append(f"[... {len(ordered) - len(out)} more passages not shown]")
            break
        out.append(block)
        total += len(block)
    return "\n".join(out) if out else "(no knowledge-base search was made)"


def conversation(rec: dict) -> str:
    qs = rec.get("questions") or []
    if len(qs) == 1:
        return f"User's message:\n{qs[0]}"
    return "Conversation (only the last turn is graded; earlier user turns for context):\n" + "\n".join(
        f"Turn {i + 1} (user): {q}" for i, q in enumerate(qs))


async def judge_one(llm, rec_a: dict, rec_b: dict, rng: random.Random) -> dict:
    ta, tb = rec_a["turns"][-1], rec_b["turns"][-1]
    sides = [("A", rec_a, ta), ("B", rec_b, tb)]
    if rng.random() < 0.5:
        sides.reverse()
    body = [conversation(rec_a)]
    for i, (_, rec, t) in enumerate(sides, 1):
        body.append(f"\n===== Response {i}: retrieved passages =====\n{evidence(rec, len(rec['turns']) - 1, t.get('answer') or '')}")
        body.append(f"\n===== Response {i}: answer shown to the user =====\n{t.get('answer') or '(no answer: the turn failed)'}")
    out = await llm.ainvoke([SystemMessage(content=RUBRIC), HumanMessage(content="\n".join(body))])
    v: Verdict = out["parsed"]
    usage = out["raw"].usage_metadata or {}
    cost = cost_usd(JUDGE_MODEL, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                    (usage.get("input_token_details") or {}).get("cache_read", 0),
                    (usage.get("input_token_details") or {}).get("cache_creation", 0)) or 0.0
    order = [s[0] for s in sides]  # order[0] is the stack shown as Response 1
    score = {order[0]: (v.faithfulness_1, v.quality_1), order[1]: (v.faithfulness_2, v.quality_2)}
    winner = "tie" if v.winner.strip() not in ("1", "2") else order[int(v.winner.strip()) - 1]
    return {"id": rec_a["id"], "category": rec_a.get("category"), "order": order, "winner": winner,
            "faithfulness": {k: s[0] for k, s in score.items()}, "quality": {k: s[1] for k, s in score.items()},
            "analysis": v.analysis, "cost_usd": round(cost, 5)}


def summary(rows: list[dict], a: str, b: str) -> None:
    n = len(rows)
    wins = {k: sum(r["winner"] == k for r in rows) for k in ("A", "B", "tie")}
    print(f"\n{n} questions judged ({a} = A, {b} = B): A wins {wins['A']}, B wins {wins['B']}, ties {wins['tie']}")
    for metric in ("faithfulness", "quality"):
        ma = sum(r[metric]["A"] for r in rows) / n
        mb = sum(r[metric]["B"] for r in rows) / n
        print(f"  mean {metric}: A {ma:.2f}, B {mb:.2f}")
    print(f"  judge cost ${sum(r['cost_usd'] for r in rows):.3f}")
    for r in rows:
        if r["winner"] == "A":
            print(f"  A better on {r['id']} ({r['category']}): F {r['faithfulness']} Q {r['quality']}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.summary_only:
        summary([json.loads(line) for line in open(args.out)], Path(args.a).stem, Path(args.b).stem)
        return
    a = {r["id"]: r for r in map(json.loads, open(args.a)) if r.get("turns")}
    b = {r["id"]: r for r in map(json.loads, open(args.b)) if r.get("turns")}
    ids = [i for i in a if i in b and (not args.ids or i in args.ids.split(","))
           and "command_output" not in (a[i]["turns"][-1].get("answer_tags") or [])]
    rng = random.Random(args.seed)
    rngs = {i: random.Random(rng.random()) for i in ids}
    llm = ChatAnthropic(model=JUDGE_MODEL, max_tokens=16000, effort="medium").with_structured_output(
        Verdict, method="json_schema", include_raw=True).with_retry()
    sem = asyncio.Semaphore(args.concurrency)
    rows = []

    async def one(i):
        async with sem:
            row = await judge_one(llm, a[i], b[i], rngs[i])
            rows.append(row)
            with LEDGER.open("a") as f:
                f.write(json.dumps({"ts": "", "label": "judge", "id": i, "usd": row["cost_usd"]}) + "\n")
            print(f"  {i:<30} winner={row['winner']:<4} F={row['faithfulness']} Q={row['quality']} ${row['cost_usd']:.3f}", flush=True)

    await asyncio.gather(*(one(i) for i in ids))
    rows.sort(key=lambda r: ids.index(r["id"]))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary(rows, Path(args.a).stem, Path(args.b).stem)


if __name__ == "__main__":
    asyncio.run(main())
