"""Offline tests for the LLM stack in steps.py and the chat handlers in app.py (fake models, no network).

  .venv/bin/python -m pytest tests/test_agent.py -q
"""
import asyncio
import random
import re
from datetime import datetime

import pytest

import fakes
from fakes import KB, FakeStructured, ScriptedChatModel, tool_call

import chainlit as cl  # noqa: E402
from chainlit.chat_context import chat_context  # noqa: E402
from chainlit.context import init_http_context  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

import kb  # noqa: E402
import retrievers  # noqa: E402
import steps  # noqa: E402
from instrument import RecordingEmitter, steps_from_events  # noqa: E402


# ---------- helpers ----------

def chainlit_context():
    events = []
    ctx = init_http_context()
    ctx.emitter = RecordingEmitter(ctx.session, events)
    return events


def answer(text: str) -> AIMessage:
    return AIMessage(content=text)


def passages_for(messages):
    """Fake reader: 'relevant' unless the doc is about Hari Jagara."""
    doc = messages[-1].content
    if "Hari Jagara" in doc:
        return steps.RelevantPassages(reasoning="", passages_in_context=["no relevant information found"])
    title = re.search(r"\d+\. (.*)\n", doc).group(1)
    return steps.RelevantPassages(reasoning="", passages_in_context=[f'"{title}" says: quoted passage'])


SUNBEAM = {"id": "sunbeam", "name": "The Chitrapur Sunbeam", "description": "the magazine of Shri Chitrapur Math"}
KSA = {"id": "ksa", "name": "Kanara Saraswat", "description": "the magazine of the Kanara Saraswat Association"}
USER_KBS = [SUNBEAM]  # what kb.enabled_kbs answers for the signed-in user (no database in these tests)


@pytest.fixture(autouse=True)
def fake_tool_models(monkeypatch):
    queries = FakeStructured(lambda m: steps.Queries(reasoning="", queries=[m[-1].content.split(": ", 1)[1]]))
    reader = FakeStructured(passages_for, delay=0.01)
    monkeypatch.setattr(steps, "query_llm", lambda: queries)
    monkeypatch.setattr(steps, "reader_llm", lambda: reader)
    monkeypatch.setattr(steps, "_agents", {})
    steps._reader_semaphores.clear()

    async def enabled_kbs(email):
        return [dict(k) for k in USER_KBS]

    monkeypatch.setattr(kb, "enabled_kbs", enabled_kbs)
    USER_KBS[:] = [SUNBEAM]
    return queries, reader


def use_agent(monkeypatch, script, mode=steps.DETAILED, **kw):
    model = ScriptedChatModel(script=script, **kw)
    agents = dict(steps._agents)
    agents[mode] = steps.build_agent(model, mode=mode)
    monkeypatch.setattr(steps, "_agents", agents)
    return model


def user_message(text, **kw):
    return cl.Message(content=text, author="User", type="user_message", **kw)


async def send(app, text, **kw):
    msg = user_message(text, **kw)
    await msg.send()
    await app.on_message(msg)
    return msg


def assistant_outputs(events):
    return [s["output"] for s in steps_from_events(events).values()
            if s.get("type") == "assistant_message" and not s["deleted"]]


@pytest.fixture
def app():
    import app as app_module
    return app_module


# ---------- final-answer extraction ----------

CASES = [
    "Let me draft.\n<final_answer>\nThe answer [1].\n\nsources:\n[1] T, D page 1-2\n</final_answer>",
    "no tags at all, the whole reply is the answer",
    "I'll put it in <final_answer> tags. Draft...\n<final_answer>real answer</final_answer>",  # tag in reasoning
    "<final_answer>first</final_answer> and <final_answer>second</final_answer>",
    "opened but <final_answer> never closed",
    "<final_answer></final_answer>",
    "a < b and </final_ans not a tag <final_answer>x < y </final_answer",
]


@pytest.mark.parametrize("text", CASES)
def test_extractor_streams_exactly_what_the_regex_shows(text):
    for seed in range(30):
        rng = random.Random(seed)
        cuts = sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(1, 12))))
        pieces = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]
        extractor = steps.FinalAnswerExtractor()
        streamed = "".join(extractor.feed(p) for p in pieces)
        expected = steps.final_answer_text(text)
        if steps._FINAL_ANSWER.search(text):
            assert streamed == expected, (pieces, streamed)
        else:
            assert expected == text
            assert expected.startswith(streamed) or "<final_answer>" in text  # unclosed: partial stream, replaced at the end


def test_sources_written_after_the_closing_tag_are_kept():
    text = ("Draft.\n<final_answer>\nIt was revived in 1973 [1].\n</final_answer>\n\nsources:\n"
            "[1] The Revival of Rathotsava, Sunbeam-1973-April.txt page 6-13\n")
    assert steps.final_answer_text(text) == ("\nIt was revived in 1973 [1].\n\nsources:\n"
                                             "[1] The Revival of Rathotsava, Sunbeam-1973-April.txt page 6-13\n")
    inside = "<final_answer>A [1]\n\nsources:\n[1] T, D.txt page 1-2\n</final_answer>\nsources:\n[1] X, Y.txt page 3-4"
    assert steps.final_answer_text(inside) == "A [1]\n\nsources:\n[1] T, D.txt page 1-2\n", "sources inside win"
    assert steps.final_answer_text("<final_answer>A</final_answer> trailing words") == "A"

    async def run():
        events = chainlit_context()
        stream = steps.AnswerStream()
        await stream.feed(AIMessage(content=text, id="c1"))
        await stream.finish(steps.final_answer_text(text))
        (msg,) = [s for s in steps_from_events(events).values() if s.get("type") == "assistant_message"]
        return msg

    msg = asyncio.run(run())
    assert msg["output"].endswith("page 6-13\n") and msg["output"].startswith(msg["streamed"].rstrip())


def test_answer_stream_saved_equals_displayed():
    async def run(chunks, final_text):
        events = chainlit_context()
        stream = steps.AnswerStream()
        for call_id, text in chunks:
            await stream.feed(AIMessage(content=text, id=call_id))
        await stream.finish(steps.final_answer_text(final_text))
        (msg,) = [s for s in steps_from_events(events).values() if s.get("type") == "assistant_message"]
        return msg

    full = "thinking...<final_ans" + "wer>Hello wor" + "ld</final_answer>"
    msg = asyncio.run(run([("c1", "thinking...<final_ans"), ("c1", "wer>Hello wor"), ("c1", "ld</final_answer>")], full))
    assert msg["streamed"] == "Hello world" == msg["output"]

    msg = asyncio.run(run([("c1", "no tags here")], "no tags here"))  # fallback: nothing streamed, whole reply shown
    assert msg["streamed"] is None and msg["output"] == "no tags here"

    msg = asyncio.run(run([("c1", "<final_answer>half")], "<final_answer>half"))  # unclosed: replaced by whole reply
    assert msg["output"] == "<final_answer>half"

    # an earlier model call wrote a final answer, then searched again and wrote a new one
    msg = asyncio.run(run([("c1", "<final_answer>old</final_answer>"), ("c2", "<final_answer>new</final_answer>")],
                          "<final_answer>new</final_answer>"))
    assert msg["streamed"] == "new" == msg["output"]


# ---------- the search tool ----------

def test_search_tool_output_and_shared_docs_untouched(fake_tool_models):
    async def run():
        chainlit_context()
        return await steps.search_knowledge_base.ainvoke({"research_query": "Rathotsava Kojagiri Hari Jagara"})

    before = [d.page_content for d in steps.fuzzy_retriever.documents]
    out = asyncio.run(run())
    kept = [d for d in KB if d.metadata["title"] != "Hari Jagara"]
    # byte-identical to what retrievers.format_docs produced before (PARITY 4.10), in retrieval order
    expected = retrievers.format_docs([
        type(d)(page_content=f'Here are the relevant passages from this document: "{d.metadata["title"]}" says: quoted passage',
                metadata=d.metadata) for d in kept])
    assert out == expected
    assert [d.page_content for d in steps.fuzzy_retriever.documents] == before, "shared KB docs must not be rewritten"


def test_reader_failure_skips_the_doc(monkeypatch):
    def flaky(messages):
        if "Kojagiri" in messages[-1].content:
            raise RuntimeError("reader exploded")
        return passages_for(messages)

    reader = FakeStructured(flaky)
    monkeypatch.setattr(steps, "reader_llm", lambda: reader)

    async def run():
        chainlit_context()
        return await steps.contextualize_docs([KB[0], KB[1]], "rathotsava")

    kept = asyncio.run(run())
    assert [d.metadata["title"] for d in kept] == ["Shri Chitrapur Rathotsava"]


def test_query_writer_failure_falls_back_to_the_question(monkeypatch):
    def truncated(messages):
        raise ValueError("Failed to parse Queries from completion {\"reasoning\": \"...\"}")

    monkeypatch.setattr(steps, "query_llm", lambda: FakeStructured(truncated))

    async def run():
        chainlit_context()
        return await steps.search_knowledge_base.ainvoke({"research_query": "Kojagiri"})

    out = asyncio.run(run())
    assert "0. Kojagiri\n" in out, "the search still ran, with the research query as the only query"


def test_global_reader_cap_across_parallel_searches(monkeypatch):
    monkeypatch.setattr(steps.models, "READER_CONCURRENCY", 3)
    steps._reader_semaphores.clear()
    reader = FakeStructured(passages_for, delay=0.02)
    monkeypatch.setattr(steps, "reader_llm", lambda: reader)
    docs = [type(KB[0])(page_content=f"doc {i}", metadata={**KB[0].metadata, "title": f"T{i}"}) for i in range(12)]

    async def run():
        chainlit_context()
        await asyncio.gather(steps.contextualize_docs(docs, "a"), steps.contextualize_docs(docs, "b"))

    asyncio.run(run())
    assert reader.calls == 24 and reader.max_in_flight == 3


# ---------- the agent through app.py ----------

def test_one_tool_our_prompt_todays_date_and_steps(app, monkeypatch):
    model = use_agent(monkeypatch, [
        AIMessage(content="I'll search.", tool_calls=[tool_call("Rathotsava", "c1"), tool_call("Kojagiri", "c2")]),
        answer("Draft.<final_answer>Both [1][2].</final_answer>"),
        answer("<final_answer>Second turn.</final_answer>"),
    ])

    class Day1(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 27, 9, 0)

    class Day2(Day1):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 28, 9, 0)

    async def run():
        events = chainlit_context()
        await app.on_chat_start()
        monkeypatch.setattr(steps, "datetime", Day1)
        await send(app, "Tell me about Rathotsava and Kojagiri")
        monkeypatch.setattr(steps, "datetime", Day2)
        await send(app, "And again?")
        return events, cl.user_session.get("messages")

    events, history = asyncio.run(run())
    assert model.bound_tools == ["search_knowledge_base"], "no filesystem, task or todo tools"
    system = [r[0].content for r in model.requests]
    assert all(s == steps.system_prompt(datetime(2026, 10, 27)) for s in system[:2])
    assert system[2] == steps.system_prompt(datetime(2026, 10, 28)), "the date is today's at each turn"

    # two parallel searches -> two engine steps, each with its own two children, in order
    tree = steps_from_events(events)
    engines = [s for s in tree.values() if s["name"] == "knowledge base search engine"]
    assert len(engines) == 2 and all(s.get("parentId") is None for s in engines)
    for engine in engines:
        children = [s["name"] for s in tree.values() if s.get("parentId") == engine["id"]]
        assert children == ["search query generator", "document analysis"], children
    assert {e["input"] for e in engines} == {'{\n    "research_query": "Rathotsava"\n}',
                                             '{\n    "research_query": "Kojagiri"\n}'}
    assert assistant_outputs(events) == ["Both [1][2].", "Second turn."]

    assert [type(m).__name__ for m in history] == [
        "HumanMessage", "AIMessage", "ToolMessage", "ToolMessage", "AIMessage",
        "HumanMessage", "AIMessage"], "each answer once; tool results kept for later turns"
    assert history[4].content == "Both [1][2]." and history[-1].content == "Second turn."
    second_request = model.requests[2]
    assert any(isinstance(m, ToolMessage) and "Rathotsava" in m.content for m in second_request)


def test_mid_turn_failure_keeps_the_chat_usable(app, monkeypatch):
    def boom(messages):
        raise RuntimeError("Anthropic is down")

    model = use_agent(monkeypatch, [
        answer("<final_answer>First.</final_answer>"),
        AIMessage(content="<final_answer>half an ans", tool_calls=[tool_call("Rathotsava")]),
        boom,
        answer("<final_answer>Recovered.</final_answer>"),
    ])

    async def run():
        events = chainlit_context()
        await app.on_chat_start()
        await send(app, "one")
        await send(app, "two")  # fails half-way: the user is told to try again
        after_failure = list(cl.user_session.get("messages"))
        await send(app, "three")
        return events, after_failure

    events, after_failure = asyncio.run(run())
    assert [type(m).__name__ for m in after_failure] == ["HumanMessage", "AIMessage"], "failed turn not kept"
    assert assistant_outputs(events) == ["First.", app.ANSWER_FAILED_MESSAGE, "Recovered."], \
        "the half-streamed answer was removed and replaced by the failure notice"
    last_request = model.requests[-1]
    assert [m.content for m in last_request if isinstance(m, HumanMessage)] == ["one", "three"]
    assert not any(getattr(m, "tool_calls", None) for m in last_request), "no dangling tool call reaches the model"


def test_edit_truncates_history(app, monkeypatch):
    model = use_agent(monkeypatch, [answer("<final_answer>Kojagiri is…</final_answer>"),
                                    answer("<final_answer>Later.</final_answer>"),
                                    answer("<final_answer>Hari Jagara is…</final_answer>")])

    async def run():
        chainlit_context()
        await app.on_chat_start()
        first = await send(app, "What is Kojagiri?")
        await send(app, "Tell me more")
        # what chainlit.socket.edit_message does: remove later messages, update this one, re-run on_message
        later = False
        for m in chat_context.get():
            if later:
                await m.remove()
            if m.id == first.id:
                m.content = "What is Hari Jagara?"
                await m.update()
                later = True
        await app.on_message(first)
        return cl.user_session.get("messages")

    history = asyncio.run(run())
    assert [m.content for m in model.requests[-1] if isinstance(m, HumanMessage)] == ["What is Hari Jagara?"]
    assert [m.content for m in history] == ["What is Hari Jagara?", "Hari Jagara is…"]


def test_commands_unchanged_and_marked(app, monkeypatch):
    use_agent(monkeypatch, [])

    async def run():
        events = chainlit_context()
        await app.on_chat_start()
        await send(app, "rathotsava AND shirali", command="Fuzzy Search")
        return events, cl.user_session.get("messages")

    events, history = asyncio.run(run())
    (out,) = [s for s in steps_from_events(events).values() if s.get("type") == "assistant_message"]
    import search_engine
    assert out["output"] == search_engine.search_knowledge_base("rathotsava AND shirali", exact=False)
    assert out["tags"] == ["command_output"] and out["metadata"] == steps.COMMAND_OUTPUT
    assert history == [], "commands never reach the agent's history"


# ---------- reopened chats ----------

def thread(*steps_, elements=()):
    return {"id": "t", "steps": list(steps_), "elements": list(elements)}


def step(sid, type_, output, **kw):
    return {"id": sid, "type": type_, "output": output, "metadata": kw.pop("metadata", {}), "isError": kw.pop("isError", False), **kw}


def test_history_from_thread_skips_commands_errors_and_notes():
    legacy_command = "\n1. Festivals (Source: Sunbeam-2007-March.txt)\nSnippet: ...rathotsava...\n\n"
    history = steps.history_from_thread(thread(
        step("u1", "user_message", "Who is Swamiji?"),
        step("r1", "run", "tool step output, ignored"),
        step("a1", "assistant_message", "Swamiji is… [1]"),
        step("u2", "user_message", "rathotsava AND shirali"),
        step("a2", "assistant_message", "1. ...", metadata=steps.COMMAND_OUTPUT),
        step("u3", "user_message", "temple"),
        step("a3", "assistant_message", legacy_command),  # command output from before it was marked
        step("u4", "user_message", "zzzz"),
        step("a4", "assistant_message", ""),               # a command with no matches
        step("u5", "user_message", "Translate this"),
        step("a5", "assistant_message", 'I couldn\'t read "x.bin"…', metadata=steps.UPLOAD_NOTE),
        step("a6", "assistant_message", "Here is the translation…"),
        step("u6", "user_message", "And this?"),
        step("a7", "assistant_message", "rate limit exceeded", isError=True),
        step("a8", "assistant_message", "Sorry, something went wrong…", metadata=steps.ANSWER_FAILED),
        step("u7", "user_message", ""),
        step("u8", "user_message", "   "),  # empty, no saved file: dropped (the API rejects empty text)
        elements=[{"forId": "u5", "name": "note.pdf", "mime": "application/pdf"},
                  {"forId": "u7", "name": "photo.jpg", "mime": "image/jpeg"}],
    ))
    assert [(type(m).__name__, m.content) for m in history] == [
        ("HumanMessage", "Who is Swamiji?"),
        ("AIMessage", "Swamiji is… [1]"),
        ("HumanMessage", "Translate this\n\n[The user attached: note.pdf (application/pdf). "
                         "The file isn't available in this reopened chat.]"),
        ("AIMessage", "Here is the translation…"),
        ("HumanMessage", "And this?"),
        ("HumanMessage", "[The user attached: photo.jpg (image/jpeg). The file isn't available in this reopened chat.]"),
    ]
    assert history[0].id == "u1", "ids kept, so an edit after reopening still truncates correctly"


def test_resume_then_continue(app, monkeypatch):
    model = use_agent(monkeypatch, [answer("<final_answer>You asked about Swamiji.</final_answer>")])

    async def run():
        chainlit_context()
        await app.on_chat_resume(thread(step("u1", "user_message", "Who is Swamiji?"),
                                        step("a1", "assistant_message", "Swamiji is… [1]")))
        await send(app, "What did I ask?")

    asyncio.run(run())
    assert [(type(m).__name__, m.content) for m in model.requests[0][1:]] == [
        ("HumanMessage", "Who is Swamiji?"), ("AIMessage", "Swamiji is… [1]"), ("HumanMessage", "What did I ask?")]


def test_model_call_cap_gives_a_friendly_answer(app, monkeypatch):
    model = ScriptedChatModel(script=[AIMessage(content="searching", tool_calls=[tool_call(f"q{i}")]) for i in range(20)])
    monkeypatch.setattr(steps, "_agents", {steps.DETAILED: steps.build_agent(model, model_call_limit=2)})

    async def run():
        events = chainlit_context()
        await app.on_chat_start()
        await send(app, "everything about everything")
        return events, cl.user_session.get("messages")

    events, history = asyncio.run(run())
    assert assistant_outputs(events) == [steps.OUT_OF_STEPS]
    assert history[-1].content == steps.OUT_OF_STEPS and len(model.requests) == 2


def test_history_before():
    h = [HumanMessage(content="a", id="1"), AIMessage(content="b"), HumanMessage(content="c", id="2"), AIMessage(content="d")]
    assert steps.history_before(h, "2") == h[:2]
    assert steps.history_before(h, "1") == []
    assert steps.history_before(h, "new") == h


def test_thinking_blocks_never_shown(app, monkeypatch):
    use_agent(monkeypatch, [AIMessage(content=[
        {"type": "thinking", "thinking": "maybe <final_answer>wrong</final_answer>"},
        {"type": "text", "text": "Draft. <final_answer>right</final_answer>"}])])

    async def run():
        events = chainlit_context()
        await app.on_chat_start()
        await send(app, "q")
        return events

    events = asyncio.run(run())
    (msg,) = [s for s in steps_from_events(events).values() if s.get("type") == "assistant_message"]
    assert msg["streamed"] == "right" == msg["output"]


# ---------- answer modes (Detailed / Quick picker) ----------

def test_answer_mode_resolution():
    from types import SimpleNamespace as NS
    assert steps.answer_mode(NS(modes={"answer": "quick"})) == steps.QUICK
    assert steps.answer_mode(NS(modes={"answer": "detailed"})) == steps.DETAILED
    for modes in (None, {}, {"answer": "bogus"}, {"answer": None}, {"other": "quick"}):
        assert steps.answer_mode(NS(modes=modes)) == steps.DETAILED, modes
    assert steps.answer_mode(NS()) == steps.DETAILED, "messages without a modes attribute"


def test_prompt_per_mode():
    day = datetime(2026, 10, 27)
    detailed = steps.system_prompt(day)
    assert detailed == steps.SYSTEM_PROMPT.replace("{today}", "October 27, 2026"), "Detailed prompt unchanged"
    assert steps.system_prompt(day, mode=steps.DETAILED) == detailed
    quick = steps.system_prompt(day, mode=steps.QUICK)
    assert steps.QUICK_INSTRUCTION not in detailed and quick.count(steps.QUICK_INSTRUCTION) == 1
    assert quick.replace("\n" + steps.QUICK_INSTRUCTION, "") == detailed, "the instruction is the only difference"
    assert "Don't use your own knowledge to answer the question.\n" + steps.QUICK_INSTRUCTION in quick


def test_effort_per_mode_and_one_cached_agent_per_mode(monkeypatch):
    import models
    assert steps.ANSWER_MODE_EFFORT == {steps.DETAILED: models.MAIN_EFFORT, steps.QUICK: "medium"}
    # Detailed's supervisor is configured exactly as main_llm() was before answer modes existed.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fields = ("model", "reasoning_effort", "max_tokens", "streaming", "thinking", "temperature")
    before = models.chat_model(models.MAIN_MODEL, max_tokens=32000, effort=models.MAIN_EFFORT, streaming=True)
    detailed_llm = models.main_llm(effort=steps.ANSWER_MODE_EFFORT[steps.DETAILED])
    assert {f: getattr(detailed_llm, f) for f in fields} == {f: getattr(before, f) for f in fields}
    assert models.main_llm(effort=steps.ANSWER_MODE_EFFORT[steps.QUICK]).reasoning_effort == "medium"

    built = []

    def fake_main_llm(effort=None):
        built.append(effort)
        return ScriptedChatModel(script=[])

    monkeypatch.setattr(steps.models, "main_llm", fake_main_llm)
    detailed = steps.get_agent()
    assert steps.get_agent(steps.DETAILED) is detailed
    quick = steps.get_agent(steps.QUICK)
    assert steps.get_agent(steps.QUICK) is quick and quick is not detailed
    assert built == [models.MAIN_EFFORT, "medium"], "built once per mode, lazily"


class ConfigSpy:
    """Wraps a compiled agent and records the config each turn runs it with (what LangSmith sees)."""

    def __init__(self, agent):
        self.agent, self.configs = agent, []

    def astream(self, *args, config=None, **kwargs):
        self.configs.append(config)
        return self.agent.astream(*args, config=config, **kwargs)


def test_quick_turn_uses_the_quick_agent(app, monkeypatch):
    detailed_model = use_agent(monkeypatch, [answer("<final_answer>Long answer [1].</final_answer>")])
    quick_model = use_agent(monkeypatch, [answer("<final_answer>Short answer.</final_answer>")], mode=steps.QUICK)
    spies = {mode: ConfigSpy(agent) for mode, agent in steps._agents.items()}
    monkeypatch.setattr(steps, "_agents", spies)

    async def run():
        events = chainlit_context()
        await app.on_chat_start()
        await send(app, "q1", modes={"answer": "quick"})
        await send(app, "q2")  # no modes: Detailed
        return events

    events = asyncio.run(run())
    answers = [s for s in steps_from_events(events).values() if s.get("type") == "assistant_message"]
    assert [(a["output"], a["metadata"]) for a in answers] == [
        ("Short answer.", {"cgpt_answer_mode": "quick"}), ("Long answer [1].", {"cgpt_answer_mode": "detailed"})]
    assert answers[0]["streamed"] == answers[0]["output"], "quick answers stream and save like detailed ones"
    (quick_request,) = quick_model.requests
    assert quick_request[0].content == steps.system_prompt(mode=steps.QUICK)
    (detailed_request,) = detailed_model.requests
    assert detailed_request[0].content == steps.system_prompt()
    (quick_config,) = spies[steps.QUICK].configs
    assert quick_config["metadata"]["answer_mode"] == "quick" and quick_config["tags"] == ["answer_mode:quick"]
    (detailed_config,) = spies[steps.DETAILED].configs
    assert detailed_config["metadata"]["answer_mode"] == "detailed" and detailed_config["tags"] == ["answer_mode:detailed"]
    assert quick_config["run_name"] == detailed_config["run_name"] == "ChitrapurGPT turn"


def test_commands_ignore_the_mode(app, monkeypatch):
    quick_model = use_agent(monkeypatch, [], mode=steps.QUICK)

    async def run():
        events = chainlit_context()
        await app.on_chat_start()
        await send(app, "rathotsava", command="Exact Search", modes={"answer": "quick"})
        return events

    events = asyncio.run(run())
    (out,) = [s for s in steps_from_events(events).values() if s.get("type") == "assistant_message"]
    import search_engine
    assert out["output"] == search_engine.search_knowledge_base("rathotsava", exact=True)
    assert out["metadata"] == steps.COMMAND_OUTPUT and quick_model.requests == []


def test_edit_keeps_the_mode(app, monkeypatch):
    quick_model = use_agent(monkeypatch, [answer("<final_answer>A</final_answer>"),
                                          answer("<final_answer>B</final_answer>")], mode=steps.QUICK)

    async def run():
        chainlit_context()
        await app.on_chat_start()
        first = await send(app, "What is Kojagiri?", modes={"answer": "quick"})
        for m in chat_context.get():  # what chainlit.socket.edit_message does: same Message object, new content
            if m.id == first.id:
                m.content = "What is Hari Jagara?"
                await m.update()
        await app.on_message(first)

    asyncio.run(run())
    assert len(quick_model.requests) == 2, "the edited message re-ran with the mode it was sent with"


def test_modes_offered_on_start_and_resume(app):
    async def run():
        events = chainlit_context()
        await app.on_chat_start()
        await app.on_chat_resume(thread())
        return [e["modes"] for e in events if e["kind"] == "set_modes"]

    offered = asyncio.run(run())
    assert len(offered) == 2 and offered[0] == offered[1]
    (mode,) = offered[0]
    assert mode["id"] == "answer" and mode["name"] == "Answer"
    assert [(o["id"], o["name"], o["default"]) for o in mode["options"]] == [("detailed", "Detailed", True),
                                                                            ("quick", "Quick", False)]
    assert all(o["icon"] and o["description"] for o in mode["options"])


# ---------- collections (knowledge bases) ----------

KSA_DOC = fakes.Document(
    page_content="The KSA golden jubilee in Mumbai ended with a Rathotsava tableau.",
    metadata={"source": "KSA-2099-March.txt", "published_date": "01-03-2099", "title": "KSA Golden Jubilee",
              "author": None, "summary": "Jubilee", "start_line": None, "end_line": None, "start_page": 3,
              "end_page": 4})


@pytest.fixture
def two_kbs():
    fakes.use_kb([*fakes._ORIGINAL_KB, KSA_DOC], ["sunbeam"] * len(fakes._ORIGINAL_KB) + ["ksa"])
    yield
    fakes.use_kb()


def test_search_covers_only_the_users_collections(two_kbs):
    def search(kbs):
        async def go():
            chainlit_context()
            token = kb.SEARCH_KBS.set(kbs)
            try:
                return await steps.search_knowledge_base.ainvoke({"research_query": "Rathotsava"})
            finally:
                kb.SEARCH_KBS.reset(token)
        return asyncio.run(go())

    store = steps.vector_db
    both = search((SUNBEAM, KSA))
    assert "Sunbeam-1976-April.txt" in both and "KSA-2099-March.txt" in both
    assert "filter" not in store.calls[-1], "every collection with articles: an unfiltered search"
    sunbeam_only = search((SUNBEAM,))
    assert "Sunbeam-1976-April.txt" in sunbeam_only and "KSA-2099-March.txt" not in sunbeam_only
    assert store.calls[-1]["filter"] == {"kb_id": ["sunbeam"]}
    ksa_only = search((KSA,))
    assert "KSA-2099-March.txt" in ksa_only and "Sunbeam" not in ksa_only
    assert search(()) == "", "no collection: nothing to search"


def test_unfiltered_while_sunbeam_is_the_only_collection():
    steps.vector_db.calls.clear()
    steps.search_retriever(["sunbeam"])
    assert steps.vector_db.calls == [{"k": 5, "fetch_k": 15}], "exactly the search as it always was"
    assert steps.search_retriever(["sunbeam"]).fuzzy_retriever.kb_ids is None


def test_prompt_names_collections_except_for_sunbeam_alone():
    now = datetime(2026, 10, 28)
    base = steps.system_prompt(now)
    assert steps.system_prompt(now, kbs=(SUNBEAM,)) == base, "Sunbeam alone: the prompt as before"
    line = steps.collections_line((SUNBEAM, KSA))
    assert line == ("The knowledge base holds articles from these collections: The Chitrapur Sunbeam (the magazine "
                    "of Shri Chitrapur Math); Kanara Saraswat (the magazine of the Kanara Saraswat Association). The "
                    "source file name of each search result shows which collection it comes from.")
    both = steps.system_prompt(now, kbs=(SUNBEAM, KSA))
    assert line in both and both.replace(line + "\n", "") == base
    assert "Kanara Saraswat" in steps.system_prompt(now, mode=steps.QUICK, kbs=(KSA,))


def test_turn_and_commands_search_the_users_collections(app, monkeypatch, two_kbs):
    seen = []

    async def respond(history, message):
        seen.append(kb.SEARCH_KBS.get())
        return history

    monkeypatch.setattr(steps, "respond", respond)

    async def run():
        events = chainlit_context()
        USER_KBS[:] = [KSA]
        await send(app, "Rathotsava")
        await send(app, "Rathotsava", command="Exact Search")
        USER_KBS[:] = []
        await send(app, "anything")
        return assistant_outputs(events)

    outputs = asyncio.run(run())
    assert seen == [(KSA,)] and kb.SEARCH_KBS.get() is None, "set for the turn only"
    assert "KSA-2099-March.txt" in outputs[0] and "Sunbeam" not in outputs[0], "the command searched KSA only"
    assert outputs[-1] == app.NO_COLLECTIONS_MESSAGE
    history = steps.history_from_thread(thread(step("u", "user_message", "anything"),
                                               step("a", "assistant_message", app.NO_COLLECTIONS_MESSAGE,
                                                    metadata=dict(steps.NO_COLLECTIONS))))
    assert [type(m).__name__ for m in history] == ["HumanMessage"], "the notice isn't an answer"
