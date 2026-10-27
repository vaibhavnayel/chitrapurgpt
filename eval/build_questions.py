"""Build the eval question set: eval/questions.local.jsonl (git-excluded).

Real user questions never go into committed files. `questions/real_selection.json` (committed) holds only a
12-hex-char SHA-256 prefix of each chosen question plus a category label; the text is looked up in
`real_questions.local.jsonl`. Edge cases written for the eval are in `questions/edge_cases.jsonl`.

  python eval/build_questions.py                 # write eval/questions.local.jsonl
  python eval/build_questions.py --pick 3 17 42  # helper: print hash prefixes for rows of the real file
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE / "real_questions.local.jsonl"
SELECTION = HERE / "questions" / "real_selection.json"
EDGE = HERE / "questions" / "edge_cases.jsonl"
OUT = HERE / "questions.local.jsonl"


def qhash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", type=int, nargs="*", help="row numbers in real_questions.local.jsonl")
    args = ap.parse_args()
    rows = [json.loads(line) for line in REAL.open()]

    if args.pick:
        for i in args.pick:
            print(json.dumps({"hash": qhash(rows[i]["question"]), "turn_idx": rows[i]["turn_idx"]}))
        return

    by_hash = {qhash(r["question"]): r for r in rows}
    selection = json.loads(SELECTION.read_text())
    out, missing = [], []
    for n, item in enumerate(selection["questions"], 1):
        row = by_hash.get(item["hash"])
        if row is None:
            missing.append(item["hash"])
            continue
        out.append({"id": f"real-{n:02d}", "source": "real", "category": item["category"],
                    "turns": [{"content": row["question"]}]})
    for line in EDGE.open():
        if line.strip():
            out.append({"source": "edge", **json.loads(line)})
    with OUT.open("w") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} questions to {OUT.relative_to(HERE.parent)}"
          f" ({sum(q['source'] == 'real' for q in out)} real, {sum(q['source'] == 'edge' for q in out)} edge)")
    if missing:
        sys.exit(f"{len(missing)} selected questions not found in {REAL.name}: {missing}")


if __name__ == "__main__":
    main()
