"""A failed answer must not break the rest of the chat.

If answering fails half-way (e.g. an AI service has no credit), the half-finished tool call used
to stay in the chat history, and every later message in that chat failed with
"tool_use ids were found without tool_result blocks". Run: python tests/test_on_message.py
"""
import asyncio
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.update({
    "CHAINLIT_AUTH_SECRET": "x" * 64,
    "OAUTH_GOOGLE_CLIENT_ID": "test-client",
    "OAUTH_GOOGLE_CLIENT_SECRET": "test-secret",
    "INGEST_WORKER": "0",
})
for _var in ("DATABASE_URL", "VECTOR_DATABASE_URL"):  # no database here: the user's collections are stubbed
    os.environ.pop(_var, None)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

calls = []


async def fake_respond(history, message):
    """Behaves like steps.respond: works on its own list and returns the new history (or raises)."""
    calls.append([type(m).__name__ for m in history])
    messages = [*history, HumanMessage(content=message.content, id=message.id)]
    messages.append(AIMessage(content="", tool_calls=[{"name": "search_knowledge_base", "args": {"research_query": "q"}, "id": "t1"}]))
    if fake_respond.fail:
        raise RuntimeError("Error code: 429 - You have no credits remaining")
    messages.append(ToolMessage(content="docs", tool_call_id="t1"))
    messages.append(AIMessage(content="the answer"))
    await said_append("the answer")  # steps.respond sends (streams) the answer itself
    return messages


said = []


async def said_append(content):
    said.append(content)


async def main():
    # Stub the LLM stack here, not at import time: pytest imports this file while collecting tests, and a
    # module-level stub would replace `steps` for every other test.
    steps = types.ModuleType("steps")
    steps.respond = fake_respond
    steps.history_before = lambda messages, message_id: list(messages)
    steps.COMMAND_OUTPUT = {"cgpt_kind": "command_output"}
    steps.ANSWER_FAILED = {"cgpt_kind": "answer_failed"}
    steps.NO_COLLECTIONS = {"cgpt_kind": "no_collections"}
    sys.modules["steps"] = steps
    os.chdir(tempfile.mkdtemp())  # Chainlit writes its default config files into the working directory
    import app

    collections = [{"id": "sunbeam", "name": "The Chitrapur Sunbeam", "description": ""}]

    async def user_collections():
        return list(collections)

    app.user_collections = user_collections
    session = {"messages": [SystemMessage(content="system")], "user": None}

    app.cl = types.SimpleNamespace(
        user_session=types.SimpleNamespace(get=session.get, set=session.__setitem__),
        Message=lambda content, **kw: types.SimpleNamespace(send=lambda: said_append(content)),
    )
    msg = lambda text: types.SimpleNamespace(content=text, command=None, id=text)  # noqa: E731

    fake_respond.fail = True
    await app.on_message(msg("first question"))
    assert said == [app.ANSWER_FAILED_MESSAGE], said
    assert [type(m).__name__ for m in session["messages"]] == ["SystemMessage"], "failed attempt must leave no trace"

    fake_respond.fail = False
    await app.on_message(msg("second question"))
    assert said[-1] == "the answer", said
    assert calls[-1] == ["SystemMessage"], calls[-1]  # no dangling tool call sent to the AI
    names = [type(m).__name__ for m in session["messages"]]
    assert names[:2] == ["SystemMessage", "HumanMessage"] and "ToolMessage" in names
    assert isinstance(session["messages"][1], HumanMessage) and session["messages"][1].content == "second question"

    # No collection switched on: a clear message, the agent isn't run, the history is unchanged.
    collections.clear()
    before, n_calls = list(session["messages"]), len(calls)
    await app.on_message(msg("third question"))
    assert said[-1] == app.NO_COLLECTIONS_MESSAGE and len(calls) == n_calls and session["messages"] == before
    print("on_message: all checks passed")


if __name__ == "__main__":
    asyncio.run(main())
