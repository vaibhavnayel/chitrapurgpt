"""Summarise the LangSmith trace shape of an eval run: how many root traces each turn produced and what is
nested under them. Reads the dev project named in .env (LANGCHAIN_PROJECT); read-only.

  .venv/bin/python eval/langsmith_traces.py --label baseline [--question real-01] [--hours 24]
"""
import argparse
import datetime as dt
import os
from collections import Counter, defaultdict
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(ROOT / ".env")

from langsmith import Client  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--question")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--max-questions", type=int, default=5)
    args = ap.parse_args()

    client = Client()
    project = os.environ.get("LANGCHAIN_PROJECT") or os.environ.get("LANGSMITH_PROJECT")
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=args.hours)
    flt = f'and(eq(metadata_key, "eval_label"), eq(metadata_value, "{args.label}"))'
    roots = list(client.list_runs(project_name=project, is_root=True, start_time=since, filter=flt))
    by_turn = defaultdict(list)
    for r in roots:
        md = (r.extra or {}).get("metadata", {})
        if args.question and md.get("question_id") != args.question:
            continue
        by_turn[(md.get("question_id"), md.get("turn"))].append(r)
    print(f"project={project} label={args.label}: {len(roots)} root runs over {len(by_turn)} turns")
    for (qid, turn), rs in sorted(by_turn.items(), key=lambda kv: str(kv[0]))[: args.max_questions]:
        print(f"\n{qid} turn {turn}: {len(rs)} root trace(s): {Counter(r.name for r in rs).most_common()}")
        for r in rs[:3]:
            children = list(client.list_runs(project_name=project, trace_id=r.trace_id, start_time=since))
            kinds = Counter((c.run_type, c.name) for c in children)
            print(f"  trace {r.name}: {len(children)} runs; " + ", ".join(f"{n}x {k[0]}:{k[1]}" for k, n in kinds.most_common(12)))


if __name__ == "__main__":
    main()
