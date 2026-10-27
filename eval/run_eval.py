"""Run eval questions through the app and record answers, tool calls, retrieval, latency, tokens and cost.

It drives the real Chainlit handlers in app.py (on_chat_start / on_message / on_chat_resume) inside a
Chainlit HTTP context whose emitter records every step and message, i.e. the production code path minus
the websocket. Nothing in app code is patched, so the same harness measures any stack behind app.py.

  .venv/bin/python eval/build_questions.py
  .venv/bin/python eval/run_eval.py --label baseline --out eval/results/baseline.local.jsonl --budget-usd 8
  .venv/bin/python eval/report.py eval/results/baseline.local.jsonl

Options: --ids real-01,edge-followup  --limit N  --skip-requires uploads  --concurrency N (default 1; >1
skews latency).  Every question's cost is appended to eval/results/spend_ledger.local.jsonl, so total API
spend across all runs is `report.py --spend`.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import datetime as dt
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "eval"
sys.path[:0] = [str(HERE), str(ROOT)]
os.chdir(ROOT)  # Chainlit reads .chainlit/config.toml and the app reads knowledge_base.jsonl from here

import dotenv  # noqa: E402

dotenv.load_dotenv(ROOT / ".env")
for _var in ("DATABASE_URL", "LITERAL_API_KEY", "BUCKET_NAME"):  # never let an eval touch a real DB or bucket
    os.environ.pop(_var, None)
# app.py registers @cl.oauth_callback, which Chainlit refuses without a provider configured. The eval never signs
# anyone in (handlers are called directly, with no user), so local placeholders are enough.
os.environ.setdefault("OAUTH_GOOGLE_CLIENT_ID", "eval-placeholder")
os.environ.setdefault("OAUTH_GOOGLE_CLIENT_SECRET", "eval-placeholder")
os.environ.setdefault("CHAINLIT_AUTH_SECRET", "eval-placeholder-" + "x" * 48)

from instrument import Collector, RecordingEmitter, collecting, steps_from_events  # noqa: E402
from pricing import EMBEDDING_PRICES  # noqa: E402


def _route_pinecone_through_env_proxy() -> None:
    """Local sandboxes reach the internet only through HTTPS_PROXY. Every other client here (httpx, requests)
    honours it, but the Pinecone SDK uses its own urllib3 pool and only takes an explicit proxy_url, so pass
    it one. Eval-only: production has direct egress and never imports this file."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not proxy:
        return
    from urllib.parse import unquote, urlsplit

    import pinecone
    import urllib3

    parts = urlsplit(proxy)
    bare = f"{parts.scheme}://{parts.hostname}:{parts.port}"
    headers = (urllib3.make_headers(proxy_basic_auth=f"{unquote(parts.username)}:{unquote(parts.password or '')}")
               if parts.username else None)
    original_init = pinecone.Pinecone.__init__

    def init_with_proxy(self, *args, **kwargs):
        kwargs.setdefault("proxy_url", bare)
        if headers:
            kwargs.setdefault("proxy_headers", headers)
        original_init(self, *args, **kwargs)

    pinecone.Pinecone.__init__ = init_with_proxy
    EVAL_SHIMS.append("pinecone-env-proxy")


def _embed_without_local_tokenizer_if_unavailable() -> None:
    """OpenAIEmbeddings tokenizes queries locally with tiktoken (to split texts over 8191 tokens) and tiktoken
    downloads its vocabulary on first use. Where that download is blocked, send the query text instead: the
    API tokenizes it with the same cl100k vocabulary, so a short query gets the same embedding."""
    try:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")
        return
    except Exception:  # noqa: BLE001
        pass
    from langchain_openai import OpenAIEmbeddings

    original_init = OpenAIEmbeddings.__init__

    def init_without_tokenizer(self, *args, **kwargs):
        kwargs.setdefault("check_embedding_ctx_length", False)
        original_init(self, *args, **kwargs)

    OpenAIEmbeddings.__init__ = init_without_tokenizer
    EVAL_SHIMS.append("embeddings-no-local-tiktoken")


EVAL_SHIMS: list[str] = []
_route_pinecone_through_env_proxy()
_embed_without_local_tokenizer_if_unavailable()

LEDGER = HERE / "results" / "spend_ledger.local.jsonl"
PACKAGES = ["chainlit", "langchain", "langchain-core", "langchain-anthropic", "langchain-openai", "langgraph",
            "deepagents", "anthropic", "openai", "langsmith", "pinecone"]
DOC_RE = re.compile(r"(?ms)^(\d+)\. (.*?)\n Metadata:\n(\{.*?\})\nContent:\n(.*?)\n-{100}$")


# ---------- helpers ----------

def stack_info(label: str) -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return {
        "label": label,
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0],
        "packages": versions,
        "model_env": {k: v for k, v in os.environ.items() if k.startswith("CGPT_")},
        "langsmith_project": os.environ.get("LANGCHAIN_PROJECT") or os.environ.get("LANGSMITH_PROJECT"),
        "eval_shims": EVAL_SHIMS,
    }


def parse_formatted_docs(text: str | None) -> list[dict]:
    """Parse retrievers.format_docs output (the text the agent sees) back into per-doc records."""
    out = []
    for m in DOC_RE.finditer(text or ""):
        try:
            meta = ast.literal_eval(m.group(3))
        except (ValueError, SyntaxError):
            meta = {}
        out.append({
            "idx": int(m.group(1)), "title": m.group(2), "source": meta.get("source"),
            "start_page": meta.get("start_page"), "end_page": meta.get("end_page"),
            "float_pages": isinstance(meta.get("start_page"), float),
            "content_chars": len(m.group(4)),
        })
    return out


def span(runs: list[dict]) -> float | None:
    runs = [r for r in runs if r.get("end")]
    return round(max(r["end"] for r in runs) - min(r["start"] for r in runs), 3) if runs else None


def llm_record(r: dict, role: str, keep_text: bool) -> dict:
    rec = {k: r.get(k) for k in ("model_canonical", "schema", "input_tokens", "output_tokens", "cache_read",
                                 "cache_write", "reasoning_tokens", "cost_usd", "error", "n_input_messages")}
    rec["role"] = role
    rec["latency_s"] = round(r["end"] - r["start"], 3) if r.get("end") else None
    if r.get("first_token") and r.get("start"):
        rec["ttft_s"] = round(r["first_token"] - r["start"], 3)
    if keep_text:
        rec["text"] = r.get("text")
        rec["tool_calls"] = r.get("tool_calls")
    return rec


def group_llms(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for r in runs:
        key = (r.get("model_canonical") or r.get("model"), r.get("schema"))
        g = groups.setdefault(key, {"model": key[0], "schema": key[1], "calls": 0, "errors": 0, "input_tokens": 0,
                                    "output_tokens": 0, "cache_read": 0, "reasoning_tokens": 0, "cost_usd": 0.0,
                                    "_runs": []})
        g["calls"] += 1
        g["errors"] += bool(r.get("error"))
        for k in ("input_tokens", "output_tokens", "cache_read", "reasoning_tokens"):
            g[k] += r.get(k) or 0
        g["cost_usd"] += r.get("cost_usd") or 0.0
        g["_runs"].append(r)
    for g in groups.values():
        g["wall_s"] = span(g.pop("_runs"))
        g["cost_usd"] = round(g["cost_usd"], 6)
    return list(groups.values())


def summarize_turn(collector: Collector, events: list, mark: int, turn: int, t0: float, t1: float,
                   error: str | None) -> dict:
    # --- what the UI received ---
    steps = steps_from_events(events, mark)
    by_id = {sid: s for sid, s in steps.items()}
    ui_steps = []
    for sid, s in steps.items():
        depth, p = 0, by_id.get(s.get("parentId") or "")
        while p is not None:
            depth += 1
            p = by_id.get(p.get("parentId") or "")
        parent = by_id.get(s.get("parentId") or "")
        ui_steps.append({
            "name": s.get("name"), "type": s.get("type"), "depth": depth,
            "parent": parent.get("name") if parent else None,
            "input_chars": len(s.get("input") or ""), "output_chars": len(s.get("output") or ""),
            "is_error": bool(s.get("isError")), "deleted": s["deleted"], "tags": s.get("tags"),
            "t_rel_s": round(s["first_t"] - t0, 3),
        })
    assistant = [s for s in steps.values() if s.get("type") == "assistant_message" and not s["deleted"]]
    answer = "\n\n".join(s.get("output") or "" for s in assistant)
    visible_t = [s["stream_t"] or s["first_t"] for s in assistant]
    streamed = [s["streamed"] for s in assistant if s["streamed"] is not None]

    # --- what the LLM stack did ---
    llm_runs = collector.turn_runs(turn, "llm")
    llms = []
    for r in llm_runs:
        tool = collector.enclosing_tool(r)
        role = "agent" if tool is None else f"in_tool:{tool['name']}"
        llms.append(llm_record(r, role, keep_text=tool is None))
    agent_runs = [r for r in llm_runs if collector.enclosing_tool(r) is None and not r.get("error")]
    final_llm = next((r for r in reversed(agent_runs) if not r.get("tool_calls")), None)
    raw_final = final_llm.get("text") if final_llm else None

    tools = []
    for t in collector.turn_runs(turn, "tool"):
        outer, leaves = collector.split_retrievers(collector.in_tool(t, "retriever"))
        inner_llms = collector.in_tool(t, "llm")
        tools.append({
            "name": t["name"], "input": t.get("input"), "error": t.get("error"),
            "latency_s": round(t["end"] - t["start"], 3) if t.get("end") else None,
            "t_start_rel_s": round(t["start"] - t0, 3),
            "queries": [r.get("query") for r in outer],
            "retrieval_wall_s": span(outer),
            "retrieved": [{"retriever": r["name"], "query": r.get("query"), "docs": r.get("docs") or [],
                           "latency_s": round(r["end"] - r["start"], 3) if r.get("end") else None,
                           "error": r.get("error")} for r in leaves],
            "llm_groups": group_llms(inner_llms),
            "kept_docs": parse_formatted_docs(t.get("output")),
            "output_chars": len(t.get("output") or ""),
            "output": t.get("output"),
        })
    n_embed = sum(1 for t in tools for r in t["retrieved"] if "vector" in (r["retriever"] or "").lower())
    embed_cost = sum(len(r["query"] or "") / 4 for t in tools for r in t["retrieved"]
                     if "vector" in (r["retriever"] or "").lower()) * EMBEDDING_PRICES["text-embedding-3-large"] / 1e6
    llm_cost = sum(r.get("cost_usd") or 0 for r in llm_runs)
    unknown = sorted({r.get("model") or "?" for r in llm_runs if r.get("cost_usd") is None and not r.get("error")})
    return {
        "turn": turn,
        "error": error,
        "latency_s": round(t1 - t0, 3),
        "time_to_visible_answer_s": round(min(visible_t) - t0, 3) if visible_t else None,
        "answer": answer,
        "answer_tags": sorted({tag for s in assistant for tag in (s.get("tags") or [])}),
        "n_assistant_messages": len(assistant),
        "streamed_equals_saved": (streamed[-1] == assistant[-1].get("output")) if streamed else None,
        "raw_final_llm_text": raw_final,
        "final_answer_tags": bool(raw_final and "<final_answer>" in raw_final and "</final_answer>" in raw_final),
        "ui_steps": ui_steps,
        "llm_calls": llms,
        "tool_calls": tools,
        "n_embedding_queries": n_embed,
        "cost_usd": round(llm_cost + embed_cost, 6),
        "unpriced_models": unknown,
    }


def summarize_history(messages) -> list[dict] | None:
    if not isinstance(messages, list):
        return None
    out = []
    for m in messages:
        content = getattr(m, "content", m)
        out.append({"type": getattr(m, "type", type(m).__name__), "chars": len(str(content)),
                    "tool_calls": len(getattr(m, "tool_calls", None) or []),
                    "multimodal": isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") not in ("text", "tool_use", "tool_result", "thinking")
                        for b in content)})
    return out


def thread_dict(thread_id: str, events: list) -> dict:
    """What Chainlit's data layer would return to on_chat_resume for this thread."""
    steps = [s for s in steps_from_events(events).values() if not s["deleted"]]
    return {"id": thread_id, "name": None, "createdAt": None, "metadata": {}, "tags": [], "userId": None,
            "userIdentifier": None, "elements": [],
            "steps": [{k: s.get(k) for k in ("id", "name", "type", "parentId", "input", "output", "tags",
                                             "createdAt", "start", "end", "isError")} for s in steps]}


class KBFreshness:
    """steps.contextualize_docs overwrites page_content on the shared FuzzyMatchRetriever documents (see
    docs/PARITY.md). Restore them before each question so every question sees a freshly deployed process,
    and count how many documents a question left modified."""

    def __init__(self) -> None:
        retriever = getattr(sys.modules.get("steps"), "fuzzy_retriever", None)
        self.docs = getattr(retriever, "documents", None) or []
        self.snapshot = [d.page_content for d in self.docs]

    def restore(self) -> None:
        for d, content in zip(self.docs, self.snapshot):
            d.page_content = content

    def mutated(self) -> int:
        return sum(1 for d, content in zip(self.docs, self.snapshot) if d.page_content != content)


# ---------- running ----------

def new_context(thread_id: str, events: list):
    from chainlit.context import init_http_context
    ctx = init_http_context(thread_id=thread_id)
    ctx.emitter = RecordingEmitter(ctx.session, events)
    return ctx


def make_elements(files: list[dict], tmpdir: Path) -> list:
    """Mirror chainlit.emitter.ChainlitEmitter.process_message for uploaded files."""
    from chainlit.element import Element
    elements = []
    for f in files:
        src = ROOT / f["path"]
        eid = str(uuid.uuid4())
        dst = tmpdir / eid
        shutil.copyfile(src, dst)
        elements.append(Element.from_dict({
            "id": eid, "name": src.name, "path": str(dst), "chainlitKey": eid, "display": "inline",
            "type": Element.infer_type_from_mime(f["mime"]), "mime": f["mime"],
        }))
    return elements


async def run_question(q: dict, app, label: str, kb: KBFreshness | None, timeout_s: float) -> dict:
    import chainlit as cl
    from chainlit.chat_context import chat_context
    from langsmith.run_helpers import tracing_context

    events: list = []
    thread_id = str(uuid.uuid4())
    new_context(thread_id, events)
    collector = Collector()
    if kb:
        kb.restore()
    tmpdir = Path(tempfile.mkdtemp(prefix="cgpt-eval-"))
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    await app.on_chat_start()
    sent: dict[int, object] = {}
    turns = []
    for i, turn in enumerate(q["turns"]):
        collector.turn = i
        if turn.get("resume_before"):
            thread = thread_dict(thread_id, events)
            new_context(thread_id, events)
            await app.on_chat_resume(thread)
            for step in thread["steps"]:  # what Chainlit does after on_chat_resume
                if "message" in (step["type"] or ""):
                    chat_context.add(cl.Message.from_dict(step))
        mark = len(events)
        if turn.get("edit_of") is not None:
            # What chainlit.socket.edit_message does: drop later messages, update this one, re-run on_message.
            target_id = sent[turn["edit_of"]].id
            msg = None
            for m in chat_context.get():
                if msg is not None:
                    await m.remove()
                if m.id == target_id:
                    m.content = turn["content"]
                    await m.update()
                    msg = m
        else:
            msg = cl.Message(content=turn["content"], author="User", type="user_message",
                             command=turn.get("command"))
            await msg.send()
            if turn.get("files"):
                msg.elements = make_elements(turn["files"], tmpdir)
        sent[i] = msg
        error = None
        t0 = time.perf_counter()
        with collecting(collector), tracing_context(tags=["eval", label],
                                                    metadata={"eval_label": label, "question_id": q["id"],
                                                              "turn": i}):
            try:
                await asyncio.wait_for(app.on_message(msg), timeout=timeout_s)
            except Exception:  # noqa: BLE001
                error = traceback.format_exc(limit=8)
        t1 = time.perf_counter()
        try:
            turns.append(summarize_turn(collector, events, mark, i, t0, t1, error))
        except Exception:  # noqa: BLE001  (a harness bug must not lose the run)
            turns.append({"turn": i, "error": error, "harness_error": traceback.format_exc(limit=5),
                          "latency_s": round(t1 - t0, 3), "cost_usd": round(sum(
                              r.get("cost_usd") or 0 for r in collector.turn_runs(i, "llm")), 6),
                          "tool_calls": [], "answer": None})
    history = summarize_history(cl.user_session.get("messages"))
    shutil.rmtree(tmpdir, ignore_errors=True)
    return {
        "id": q["id"], "source": q.get("source"), "category": q.get("category"), "label": label,
        "started_at": started,
        "questions": [t.get("content") for t in q["turns"]],
        "turns": turns,
        "history_after": history,
        "kb_docs_mutated": kb.mutated() if kb else None,
        "latency_s": round(sum(t["latency_s"] for t in turns), 3),
        "cost_usd": round(sum(t["cost_usd"] for t in turns), 6),
        "error": next((t["error"] for t in turns if t["error"]), None),
    }


def load_questions(args) -> list[dict]:
    qs = [json.loads(line) for line in Path(args.questions).open() if line.strip()]
    if args.ids:
        wanted = args.ids.split(",")
        qs = [q for q in qs if q["id"] in wanted]
    if args.skip_requires:
        skip = set(args.skip_requires.split(","))
        qs = [q for q in qs if not skip & set(q.get("requires", []))]
    if args.source:
        qs = [q for q in qs if q.get("source") == args.source]
    if args.exclude_done and Path(args.out).exists():
        done = {json.loads(line)["id"] for line in Path(args.out).open() if line.strip()}
        qs = [q for q in qs if q["id"] not in done]
    return qs[: args.limit] if args.limit else qs


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", default=str(HERE / "questions.local.jsonl"))
    ap.add_argument("--ids")
    ap.add_argument("--source", choices=["real", "edge"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-requires")
    ap.add_argument("--budget-usd", type=float, default=None, help="stop starting new questions past this spend")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=900)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--exclude-done", action="store_true", help="skip ids already in --out (resume a run)")
    args = ap.parse_args()

    questions = load_questions(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    import app  # noqa: F401  (builds the retrievers; same import the Chainlit server does)
    import_s = time.perf_counter() - t
    info = stack_info(args.label)
    info["app_import_s"] = round(import_s, 1)
    print(f"[{args.label}] app imported in {import_s:.1f}s; running {len(questions)} questions", flush=True)
    kb = KBFreshness() if args.concurrency == 1 else None

    spent = 0.0
    sem = asyncio.Semaphore(args.concurrency)
    mode = "a" if (args.append or args.exclude_done) else "w"
    with open(args.out, mode) as out:
        async def one(q):
            nonlocal spent
            async with sem:
                if args.budget_usd is not None and spent >= args.budget_usd:
                    print(f"  skip {q['id']}: budget {args.budget_usd} reached", flush=True)
                    return
                try:
                    rec = await run_question(q, app, args.label, kb, args.timeout_s)
                except Exception:  # noqa: BLE001
                    print(f"  {q['id']}: HARNESS ERROR\n{traceback.format_exc(limit=6)}", flush=True)
                    return
                rec["stack"] = info
                spent += rec["cost_usd"]
                out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                out.flush()
                with LEDGER.open("a") as ledger:
                    ledger.write(json.dumps({"ts": rec["started_at"], "label": args.label, "id": q["id"],
                                             "usd": rec["cost_usd"]}) + "\n")
                n_tools = sum(len(tt["tool_calls"]) for tt in rec["turns"])
                print(f"  {q['id']:<30} {rec['latency_s']:7.1f}s  ${rec['cost_usd']:.4f}  tools={n_tools}"
                      f"  {'ERROR ' + rec['error'].strip().splitlines()[-1][:120] if rec['error'] else ''}"
                      f"  (run total ${spent:.3f})", flush=True)

        await asyncio.gather(*(asyncio.create_task(one(q)) for q in questions))
    print(f"[{args.label}] done: ${spent:.3f} this run", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
