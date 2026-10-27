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
})

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

calls = []


async def fake_respond(messages):
    """Behaves like steps.respond_to_user_message: grows `messages` in place while it works."""
    calls.append([type(m).__name__ for m in messages])
    messages.append(AIMessage(content="", tool_calls=[{"name": "search_knowledge_base", "args": {"research_query": "q"}, "id": "t1"}]))
    if fake_respond.fail:
        raise RuntimeError("Error code: 429 - You have no credits remaining")
    messages.append(ToolMessage(content="docs", tool_call_id="t1"))
    messages.append(AIMessage(content="the answer"))
    return messages[-1]


steps = types.ModuleType("steps")
steps.respond_to_user_message = fake_respond
sys.modules["steps"] = steps


async def main():
    os.chdir(tempfile.mkdtemp())  # Chainlit writes its default config files into the working directory
    import app

    session = {"messages": [SystemMessage(content="system")], "user": None}
    said = []

    async def send(content):
        said.append(content)

    app.cl = types.SimpleNamespace(
        user_session=types.SimpleNamespace(get=session.get, set=session.__setitem__),
        Message=lambda content, **kw: types.SimpleNamespace(send=lambda: send(content)),
    )
    msg = lambda text: types.SimpleNamespace(content=text, command=None)  # noqa: E731

    fake_respond.fail = True
    await app.on_message(msg("first question"))
    assert said == [app.ANSWER_FAILED_MESSAGE], said
    assert [type(m).__name__ for m in session["messages"]] == ["SystemMessage"], "failed attempt must leave no trace"

    fake_respond.fail = False
    await app.on_message(msg("second question"))
    assert said[-1] == "the answer", said
    assert calls[-1] == ["SystemMessage", "HumanMessage"], calls[-1]  # no dangling tool call sent to the AI
    names = [type(m).__name__ for m in session["messages"]]
    assert names[:2] == ["SystemMessage", "HumanMessage"] and "ToolMessage" in names
    assert isinstance(session["messages"][1], HumanMessage) and session["messages"][1].content == "second question"
    print("on_message: all checks passed")


if __name__ == "__main__":
    asyncio.run(main())
