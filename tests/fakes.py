"""Offline fakes for testing the LLM stack: no network, no API keys, no vector database.

Import this before `steps` (or `app`): it points retrievers' loaders at a tiny in-memory knowledge base so that
importing steps doesn't need the vector database or index the 30 MB knowledge_base.jsonl.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.setdefault("OAUTH_GOOGLE_CLIENT_ID", "test-client")
os.environ.setdefault("OAUTH_GOOGLE_CLIENT_SECRET", "test-secret")
os.environ.setdefault("CHAINLIT_AUTH_SECRET", "t" * 64)
os.environ.pop("DATABASE_URL", None)

from langchain_core.callbacks import CallbackManagerForRetrieverRun  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

KB = [
    Document(page_content="The Rathotsava at Shirali is held in Chaitra. The chariot is pulled by devotees.",
             metadata={"source": "Sunbeam-1976-April.txt", "published_date": "01-04-1976",
                       "title": "Shri Chitrapur Rathotsava", "author": None, "summary": "Rathotsava",
                       "start_line": 1, "end_line": 9, "start_page": 19, "end_page": 25}),
    Document(page_content="Kojagiri Pournima is celebrated at the Math with milk and moonlight.",
             metadata={"source": "Sunbeam-2010-October.txt", "published_date": "01-10-2010",
                       "title": "Kojagiri", "author": "Editor", "summary": "Kojagiri",
                       "start_line": 3, "end_line": 7, "start_page": 4, "end_page": 5}),
    Document(page_content="Hari Jagara is an all-night vigil of bhajans.",
             metadata={"source": "Sunbeam-2005-November.txt", "published_date": "01-11-2005",
                       "title": "Hari Jagara", "author": None, "summary": "Hari Jagara",
                       "start_line": 2, "end_line": 4, "start_page": 10, "end_page": 10}),
]


KB_IDS = ["sunbeam"] * len(KB)  # the knowledge base of each KB document (tests may extend both lists)


class FakeVectorStore:
    """Records the search_kwargs it was asked for (e.g. a kb_id filter) and honours the filter."""

    def __init__(self):
        self.calls: list[dict] = []

    def as_retriever(self, **kwargs):
        search_kwargs = kwargs.get("search_kwargs") or {}
        self.calls.append(search_kwargs)
        kbs = (search_kwargs.get("filter") or {}).get("kb_id")
        docs = [d for d, k in zip(KB, KB_IDS) if kbs is None or k in kbs]
        return KeywordRetriever(documents=docs, k=5)


class KeywordRetriever(BaseRetriever):
    """Returns the KB docs that share a word with the query."""
    documents: list
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        words = {w.lower() for w in query.split()}
        hits = [d for d in self.documents if words & {w.lower().strip(".,") for w in d.page_content.split()}]
        return [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in hits][: self.k]


def use_kb(docs: list[Document] | None = None, kb_ids: list[str] | None = None) -> None:
    """Put a knowledge base in the in-memory index (and the fake vector store): the tiny KB by default."""
    import kb_index
    KB[:] = docs if docs is not None else _ORIGINAL_KB
    KB_IDS[:] = kb_ids if kb_ids is not None else ["sunbeam"] * len(KB)
    kb_index.INDEX.set_documents([(f"fake-{n}", k, d) for n, (d, k) in enumerate(zip(KB, KB_IDS))])


# Import steps against the tiny knowledge base (no database), then put the real vector store loader back.
import retrievers  # noqa: E402

_ORIGINAL_KB = list(KB)
use_kb()
_real_loader = retrievers.load_vector_store
retrievers.load_vector_store = lambda: FakeVectorStore()
try:
    import steps  # noqa: E402,F401
finally:
    retrievers.load_vector_store = _real_loader


def tool_call(query: str, call_id: str | None = None) -> dict:
    return {"name": "search_knowledge_base", "args": {"research_query": query}, "id": call_id or f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call"}


class ScriptedChatModel(BaseChatModel):
    """Plays back scripted replies (AIMessages, or callables taking the request's messages). Streams text in
    small chunks so tag-splitting is exercised; tool calls come in the last chunk. Records every request."""
    script: list
    model: str = "scripted"
    chunk_size: int = 3
    _requests: list = PrivateAttr(default_factory=list)
    _bound_tools: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _get_ls_params(self, stop=None, **kwargs) -> dict:
        return {"ls_provider": "fake", "ls_model_name": self.model, "ls_model_type": "chat"}

    def bind_tools(self, tools, **kwargs):
        self._bound_tools = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self

    @property
    def requests(self) -> list[list[BaseMessage]]:
        return self._requests

    @property
    def bound_tools(self) -> list[str]:
        return self._bound_tools

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self._requests.append(list(messages))
        if not self.script:
            raise AssertionError("ScriptedChatModel ran out of replies")
        item = self.script.pop(0)
        reply = item(messages) if callable(item) else item
        if isinstance(reply, Exception):
            raise reply
        return reply

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self._next(messages)
        blocks = reply.content if isinstance(reply.content, list) else [{"type": "text", "text": reply.content}]
        for index, block in enumerate(blocks):
            body_key = "text" if block["type"] == "text" else "thinking"
            body = block.get(body_key, "")
            for i in range(0, max(len(body), 1), self.chunk_size):
                piece = body[i:i + self.chunk_size]
                content = piece if block["type"] == "text" else [{"type": "thinking", "thinking": piece, "index": index}]
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
                if run_manager and block["type"] == "text":
                    await run_manager.on_llm_new_token(piece, chunk=chunk)
                yield chunk
                await asyncio.sleep(0)
        if reply.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                for i, tc in enumerate(reply.tool_calls)]))


class FakeStructured:
    """Stands in for `<llm>.with_structured_output(Schema)`: `fn(messages)` returns the parsed object or raises.
    Tracks how many calls are in flight at once."""

    def __init__(self, fn: Callable[[list], Any], delay: float = 0.0):
        self.fn, self.delay = fn, delay
        self.in_flight = self.max_in_flight = self.calls = 0

    async def ainvoke(self, messages, config=None, **kwargs):
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return self.fn(messages)
        finally:
            self.in_flight -= 1

    def as_runnable(self):
        return RunnableLambda(lambda m: self.fn(m), afunc=self.ainvoke)
