"""Stack-agnostic instrumentation for eval runs.

- `Collector` is a LangChain callback handler. It is attached to *every* LangChain/LangGraph run made while
  `collecting(collector)` is active, through langchain-core's configure hook, so the app code needs no
  changes: LLM calls (model, tokens, cost, latency, output), tool calls, retriever queries and documents,
  and the parent/child links between them.
- `RecordingEmitter` replaces Chainlit's no-op HTTP emitter and records every step, message, streamed token
  and element the app sends to the UI, with timestamps.
"""
from __future__ import annotations

import contextlib
import contextvars
import time
from typing import Any

from chainlit.emitter import BaseChainlitEmitter
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from pricing import canonical_model, cost_usd

_collector_var: contextvars.ContextVar = contextvars.ContextVar("cgpt_eval_collector", default=None)
register_configure_hook(_collector_var, inheritable=True)
# Run id of the tool whose body is executing. Set in on_tool_start, which runs (inline) in the context that the
# tool body's context is copied from, so LLM/retriever runs inside the tool see it even when something like
# langsmith's @traceable breaks the LangChain parent_run_id chain.
_current_tool: contextvars.ContextVar = contextvars.ContextVar("cgpt_eval_current_tool", default=None)


@contextlib.contextmanager
def collecting(collector: "Collector"):
    token = _collector_var.set(collector)
    try:
        yield collector
    finally:
        _collector_var.reset(token)


def _doc_meta(doc) -> dict:
    m = getattr(doc, "metadata", {}) or {}
    return {
        "title": m.get("title"),
        "source": m.get("source"),
        "start_page": m.get("start_page"),
        "end_page": m.get("end_page"),
        "chars": len(getattr(doc, "page_content", "") or ""),
    }


def _message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _structured_schema_name(kwargs: dict, metadata: dict | None) -> str | None:
    """Name of the structured-output schema a chat model call is forced to produce, if any."""
    params = kwargs.get("invocation_params") or {}
    for source in (kwargs, params, (metadata or {})):
        fmt = source.get("ls_structured_output_format")
        if isinstance(fmt, dict):
            schema = fmt.get("schema") or {}
            if isinstance(schema, dict):
                name = schema.get("title") or schema.get("name") or (schema.get("function") or {}).get("name")
                if name:
                    return name
    # Not every langchain-core version forwards ls_structured_output_format; the request itself says it too.
    choice = params.get("tool_choice")
    if isinstance(choice, dict):
        name = choice.get("name") or (choice.get("function") or {}).get("name")
        if name:
            return name
    out_fmt = ((params.get("output_config") or {}).get("format") or {})  # Anthropic native structured output
    if isinstance(out_fmt, dict) and isinstance(out_fmt.get("schema"), dict) and out_fmt["schema"].get("title"):
        return out_fmt["schema"]["title"]
    fmt = params.get("response_format")
    if isinstance(fmt, dict):
        return (fmt.get("json_schema") or {}).get("name")
    if isinstance(fmt, type):
        return fmt.__name__
    return None


class Collector(BaseCallbackHandler):
    run_inline = True
    raise_error = False

    def __init__(self) -> None:
        super().__init__()
        self.runs: dict[str, dict] = {}
        self.turn = 0

    # ---- bookkeeping ----
    def _start(self, run_id, parent_run_id, kind: str, name: str | None, **extra) -> None:
        self.runs[str(run_id)] = {
            "id": str(run_id),
            "parent": str(parent_run_id) if parent_run_id else None,
            "tool_ctx": _current_tool.get(),
            "kind": kind,
            "name": name,
            "turn": self.turn,
            "start": time.perf_counter(),
            "end": None,
            "error": None,
            **extra,
        }

    def _end(self, run_id, **extra) -> None:
        run = self.runs.get(str(run_id))
        if run is not None:
            run["end"] = time.perf_counter()
            run.update(extra)

    def _error(self, error, run_id) -> None:
        self._end(run_id, error=f"{type(error).__name__}: {error}"[:500])

    # ---- chains (only needed for the parent links) ----
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name") or ((serialized or {}).get("id") or ["?"])[-1]
        self._start(run_id, parent_run_id, "chain", name)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._error(error, run_id)

    # ---- LLMs ----
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        params = kwargs.get("invocation_params") or {}
        model = (metadata or {}).get("ls_model_name") or params.get("model") or params.get("model_name")
        n_in = sum(len(batch) for batch in messages)
        self._start(run_id, parent_run_id, "llm", kwargs.get("name") or (serialized or {}).get("name"),
                    model=model, model_canonical=canonical_model(model),
                    schema=_structured_schema_name(kwargs, metadata), n_input_messages=n_in,
                    ls_provider=(metadata or {}).get("ls_provider"))

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        params = kwargs.get("invocation_params") or {}
        self._start(run_id, parent_run_id, "llm", kwargs.get("name"),
                    model=(metadata or {}).get("ls_model_name") or params.get("model"),
                    schema=_structured_schema_name(kwargs, metadata))

    def on_llm_new_token(self, token, *, run_id, **kwargs):
        run = self.runs.get(str(run_id))
        if run is not None and run.get("first_token") is None:
            run["first_token"] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage, text, tool_calls, model = {}, "", [], None
        try:
            gen = response.generations[0][0]
            message = getattr(gen, "message", None)
            if message is not None:
                usage = dict(getattr(message, "usage_metadata", None) or {})
                text = _message_text(message)
                tool_calls = [{"name": tc.get("name"), "args": tc.get("args")} for tc in (message.tool_calls or [])]
                model = (message.response_metadata or {}).get("model_name") or (message.response_metadata or {}).get("model")
            else:
                text = gen.text
        except (IndexError, AttributeError):
            pass
        if not usage and response.llm_output:
            tu = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
            usage = {"input_tokens": tu.get("prompt_tokens") or tu.get("input_tokens") or 0,
                     "output_tokens": tu.get("completion_tokens") or tu.get("output_tokens") or 0}
        in_details = usage.get("input_token_details") or {}
        out_details = usage.get("output_token_details") or {}
        run = self.runs.get(str(run_id)) or {}
        model = run.get("model") or model
        tokens = {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_read": int(in_details.get("cache_read") or 0),
            "cache_write": int(in_details.get("cache_creation") or 0),
            "reasoning_tokens": int(out_details.get("reasoning") or 0),
        }
        self._end(run_id, model=model, model_canonical=canonical_model(model), text=text, tool_calls=tool_calls,
                  cost_usd=cost_usd(model, tokens["input_tokens"], tokens["output_tokens"],
                                    tokens["cache_read"], tokens["cache_write"]),
                  **tokens)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._error(error, run_id)

    # ---- tools ----
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, inputs=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name")
        self._start(run_id, parent_run_id, "tool", name, input=inputs if inputs is not None else input_str,
                    prev_tool_ctx=_current_tool.get())
        _current_tool.set(str(run_id))

    def _leave_tool(self, run_id) -> None:
        run = self.runs.get(str(run_id)) or {}
        if _current_tool.get() == str(run_id):
            _current_tool.set(run.get("prev_tool_ctx"))

    def on_tool_end(self, output, *, run_id, **kwargs):
        content = getattr(output, "content", output)
        if not isinstance(content, str):
            content = str(content)
        self._end(run_id, output=content)
        self._leave_tool(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._error(error, run_id)
        self._leave_tool(run_id)

    # ---- retrievers ----
    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name") or ((serialized or {}).get("id") or ["?"])[-1]
        self._start(run_id, parent_run_id, "retriever", name, query=query)

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self._end(run_id, docs=[_doc_meta(d) for d in documents])

    def on_retriever_error(self, error, *, run_id, **kwargs):
        self._error(error, run_id)

    # ---- queries over the recorded runs ----
    def ancestor(self, run: dict, kind: str) -> dict | None:
        parent = self.runs.get(run.get("parent") or "")
        while parent is not None:
            if parent["kind"] == kind:
                return parent
            parent = self.runs.get(parent.get("parent") or "")
        return None

    def enclosing_tool(self, run: dict) -> dict | None:
        return self.ancestor(run, "tool") or self.runs.get(run.get("tool_ctx") or "")

    def in_tool(self, tool: dict, kind: str) -> list[dict]:
        """Runs of `kind` executed inside `tool`, in start order."""
        out = [r for r in self.runs.values() if r["kind"] == kind and (self.enclosing_tool(r) or {}).get("id") == tool["id"]]
        return sorted(out, key=lambda r: r["start"])

    @staticmethod
    def split_retrievers(rets: list[dict]) -> tuple[list[dict], list[dict]]:
        """(outer, leaves). Retrievers called inside another retriever (e.g. HybridRetriever -> fuzzy + vector)
        don't always get it as LangChain parent, so nest by time containment with the same query."""
        def contains(a, b):
            return (a is not b and a.get("query") == b.get("query") and a["start"] <= b["start"]
                    and b.get("end") and a.get("end") and b["end"] <= a["end"])
        outer = [r for r in rets if not any(contains(o, r) for o in rets)]
        leaves = [r for r in rets if not any(contains(r, o) for o in rets)]
        return outer, leaves

    def turn_runs(self, turn: int, kind: str) -> list[dict]:
        return sorted((r for r in self.runs.values() if r["turn"] == turn and r["kind"] == kind),
                      key=lambda r: r["start"])


class RecordingEmitter(BaseChainlitEmitter):
    """Chainlit emitter that records what the UI would receive."""

    def __init__(self, session, events: list | None = None) -> None:
        super().__init__(session)
        self.events = events if events is not None else []

    def _rec(self, kind: str, **data: Any) -> None:
        self.events.append({"t": time.perf_counter(), "kind": kind, **data})

    async def send_step(self, step_dict):
        self._rec("send_step", step=dict(step_dict))

    async def update_step(self, step_dict):
        self._rec("update_step", step=dict(step_dict))

    async def delete_step(self, step_dict):
        self._rec("delete_step", step=dict(step_dict))

    async def stream_start(self, step_dict):
        self._rec("stream_start", step=dict(step_dict))

    async def send_token(self, id: str, token: str, is_sequence=False, is_input=False):
        self._rec("send_token", id=id, token=token, is_sequence=is_sequence, is_input=is_input)

    async def send_element(self, element_dict):
        self._rec("send_element", element={k: element_dict.get(k) for k in ("id", "name", "type", "mime", "forId")})

    async def set_commands(self, commands):
        self._rec("set_commands", commands=[c.get("id") for c in commands])

    async def set_modes(self, modes):
        self._rec("set_modes", modes=[m.to_dict() for m in modes])

    async def emit(self, event, data):
        self._rec("emit", event=event)


def steps_from_events(events: list, since: int = 0) -> dict[str, dict]:
    """Latest state of each step/message first seen at or after event index `since`, in first-seen order."""
    steps: dict[str, dict] = {}
    seen_before = {e["step"]["id"] for e in events[:since] if "step" in e}
    for e in events[since:]:
        if "step" in e:
            sid = e["step"]["id"]
            if sid in seen_before:
                continue
            entry = steps.setdefault(sid, {"first_t": e["t"], "deleted": False, "streamed": None, "stream_t": None})
            if e["kind"] == "delete_step":
                entry["deleted"] = True
                continue
            if e["kind"] == "stream_start":
                entry["streamed"] = e["step"].get("output") or ""
                entry["stream_t"] = e["t"]
            entry.update({k: v for k, v in e["step"].items()})
        elif e["kind"] == "send_token" and e["id"] in steps and not e.get("is_input"):
            entry = steps[e["id"]]
            entry["streamed"] = e["token"] if e["is_sequence"] else (entry["streamed"] or "") + e["token"]
    return steps
