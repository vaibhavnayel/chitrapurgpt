"""Offline checks for the eval instrumentation (no network, no API keys).

  .venv/bin/python -m pytest eval/test_instrument.py -q
"""
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import chainlit as cl  # noqa: E402
from chainlit.context import init_http_context  # noqa: E402
from langchain_core.callbacks import CallbackManagerForRetrieverRun  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langsmith import traceable  # noqa: E402

from instrument import Collector, RecordingEmitter, collecting, steps_from_events  # noqa: E402
from run_eval import parse_formatted_docs  # noqa: E402


class Leaf(BaseRetriever):
    tag: str

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return [Document(page_content=f"{self.tag}:{query}", metadata={"title": self.tag, "source": "x.txt",
                                                                       "start_page": 1, "end_page": 2})]


class Hybrid(BaseRetriever):
    a: BaseRetriever
    b: BaseRetriever

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return self.a.invoke(query) + self.b.invoke(query)


retriever = Hybrid(a=Leaf(tag="fuzzy"), b=Leaf(tag="vector"))


@traceable
@cl.step(name="search query generator")
async def make_queries(q: str) -> list[str]:
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="q1")]))
    await llm.ainvoke("write queries")
    return [q + " a", q + " b"]


@tool
@cl.step(name="knowledge base search engine")
@traceable
async def search(research_query: str) -> str:
    """Search."""
    queries = await make_queries(research_query)
    docs = retriever.batch(queries)
    return "\n".join(d.page_content for batch in docs for d in batch)


def test_tool_attribution_steps_and_retrievers():
    async def run():
        events = []
        ctx = init_http_context()
        ctx.emitter = RecordingEmitter(ctx.session, events)
        collector = Collector()
        agent = GenericFakeChatModel(messages=iter([AIMessage(content="final")]))
        with collecting(collector):
            await agent.ainvoke("hi")
            out = await search.ainvoke({"research_query": "kodi"})
        return collector, events, out

    collector, events, out = asyncio.run(run())
    llms = collector.turn_runs(0, "llm")
    roles = [(collector.enclosing_tool(r) or {}).get("name") for r in llms]
    assert roles == [None, "search"], roles
    tool_run = collector.turn_runs(0, "tool")[0]
    outer, leaves = collector.split_retrievers(collector.in_tool(tool_run, "retriever"))
    assert [r["query"] for r in outer] == ["kodi a", "kodi b"]
    assert sorted(r["name"] for r in leaves) == ["Leaf"] * 4
    steps = list(steps_from_events(events).values())
    names = [(s["name"], next((p["name"] for p in steps if p["id"] == s.get("parentId")), None)) for s in steps]
    assert names == [("knowledge base search engine", None),
                     ("search query generator", "knowledge base search engine")], names
    assert "fuzzy:kodi a" in out


def test_streamed_text_reconstruction():
    async def run():
        events = []
        ctx = init_http_context()
        ctx.emitter = RecordingEmitter(ctx.session, events)
        msg = cl.Message(content="")
        for tok in ["Hello", " wor", "ld"]:
            await msg.stream_token(tok)
        await msg.send()
        return events

    steps = list(steps_from_events(asyncio.run(run())).values())
    assert steps[0]["streamed"] == "Hello world" == steps[0]["output"]


def test_parse_formatted_docs_matches_retrievers_format_docs():
    from retrievers import format_docs
    docs = [Document(page_content="abc", metadata={"title": "T1", "source": "S.txt", "start_page": 3, "end_page": 4}),
            Document(page_content="d\ne", metadata={"title": "T2", "source": "S2.txt", "start_page": 5.0, "end_page": 6.0})]
    parsed = parse_formatted_docs(format_docs(docs))
    assert [(p["title"], p["source"], p["start_page"], p["float_pages"]) for p in parsed] == [
        ("T1", "S.txt", 3, False), ("T2", "S2.txt", 5.0, True)]
