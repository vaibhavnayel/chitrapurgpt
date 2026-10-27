import chainlit as cl
from chainlit.types import ThreadDict

# The LLM stack (agent, search tool, history helpers). Used as `steps.x` at call time; tests/test_admin.py stubs it.
import steps
from search_engine import search_knowledge_base
import logging

# Google sign-in on Render PR previews goes through the live site's callback (see preview_oauth.py).
from chainlit.server import app as chainlit_server_app
import preview_oauth
preview_oauth.install(chainlit_server_app)

# Admin dashboard at /admin (see admin.py), and idle tracking on PR previews (see preview_idle.py).
import access
import admin
import preview_idle
admin.install(chainlit_server_app)
preview_idle.install(chainlit_server_app)

ANSWER_FAILED_MESSAGE = "Sorry, something went wrong while preparing the answer. Please try again in a few minutes."
NO_ACCESS_MESSAGE = "Sorry, you no longer have access to ChitrapurGPT. If you think this is a mistake, please contact the person who invited you."

@cl.oauth_callback
async def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict[str, str],
    default_user: cl.User) -> cl.User | None:

    # The sign-in list is in the database now (managed at /admin, see access.py).
    logging.info(f"oauth_callback: {default_user}")
    try:
        allowed = await access.is_allowed(default_user.identifier)
    except Exception:
        logging.exception("Couldn't check the sign-in list")
        return
    if allowed:
        logging.info(f"user {default_user.identifier} in whitelist")
        return default_user

    logging.info(f"user {default_user.identifier} not in whitelist")
    return


async def still_allowed() -> bool:
    """Signed-in users stay signed in for days, so re-check the list when they use the app.
    If they were removed, tell them and stop. If the database can't be reached, let them carry on."""
    user = cl.user_session.get("user")
    if user is None:
        return True
    try:
        if await access.is_allowed(user.identifier):
            return True
    except Exception:
        logging.exception("Couldn't check the sign-in list")
        return True
    logging.info(f"user {user.identifier} was removed from the whitelist, refusing")
    await cl.Message(content=NO_ACCESS_MESSAGE).send()
    return False
    
commands = [
    {"id": "Exact Search", "icon": "crosshair", "description": "Exact search across all documents. 'sans' will not match 'sanskrit'."},
    {"id": "Fuzzy Search", "icon": "search", "description": "Fuzzy search across all documents. 'sans' will match 'sanskar' and 'sanskrit'."},
]

# The agent's system prompt (with today's date, per turn) is in steps.py.
@cl.on_chat_start
async def on_chat_start():
    if not await still_allowed():
        return
    cl.user_session.set("messages", [])
    await cl.context.emitter.set_commands(commands)

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    if not await still_allowed():
        return
    cl.user_session.set("messages", steps.history_from_thread(thread))
    await cl.context.emitter.set_commands(commands)


@cl.on_message
async def on_message(message: cl.Message):
    if not await still_allowed():
        return
    if message.command == "Fuzzy Search":
        search_results = search_knowledge_base(message.content, exact=False)
        await cl.Message(content=search_results, tags=["command_output"], metadata=steps.COMMAND_OUTPUT).send()
    elif message.command == "Exact Search":
        search_results = search_knowledge_base(message.content, exact=True)
        await cl.Message(content=search_results, tags=["command_output"], metadata=steps.COMMAND_OUTPUT).send()
    else:
        # An edited message comes back with its original id: answer it as if what followed it never happened.
        history = steps.history_before(cl.user_session.get("messages") or [], message.id)
        try:
            new_history = await steps.respond(history, message)
        except Exception:
            # Only a finished turn is kept, so a failure half-way (e.g. an AI service is down) can't leave a
            # half-finished tool call in the history and break every later message in this chat.
            logging.exception("Couldn't answer the message")
            await cl.Message(content=ANSWER_FAILED_MESSAGE).send()
            return
        cl.user_session.set("messages", new_history)
