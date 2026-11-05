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

# Collections of articles each user searches (kb.py), the picker for people with several (kb_picker.py), and the
# worker that processes uploaded issues (ingest/).
import kb
import kb_picker
from ingest import jobs

ANSWER_FAILED_MESSAGE = "Sorry, something went wrong while preparing the answer. Please try again in a few minutes."
NO_ACCESS_MESSAGE = "Sorry, you no longer have access to ChitrapurGPT. If you think this is a mistake, please contact the person who invited you."
NONE_TICKED_MESSAGE = ("No collection is ticked, so I have nothing to search. Press **Collections** (the gear button "
                       "in the message box), tick the collections you want, and press Confirm.")
NO_COLLECTIONS_MESSAGE = ("Sorry, no collection of articles is switched on for you at the moment, so I have nothing to "
                          "search. Please ask the person who invited you (an admin) to switch one on.")


@cl.on_app_startup
async def on_app_startup():
    # Processes uploaded magazine issues in the background (ingest/jobs.py).
    jobs.start_worker()
    try:  # a video-list update that was being applied when the app last stopped can be applied again
        from ingest import sources
        await sources.mark_interrupted()
    except Exception:  # noqa: BLE001  (the database may be unreachable; the chat must still start)
        logging.exception("Couldn't check for interrupted collection updates")


async def user_collections() -> list[dict]:
    """The knowledge bases this user's questions search (switched per person at /admin). If the database can't be
    reached, Sunbeam (the default collection) rather than nothing."""
    user = cl.user_session.get("user")
    try:
        return await kb.enabled_kbs(user.identifier if user else None)
    except Exception:
        logging.exception("Couldn't read the user's collections; searching Sunbeam")
        return [{key: kb.SUNBEAM[key] for key in ("id", "name", "description")}]

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
    await cl.context.emitter.set_modes(steps.answer_modes())  # the Detailed / Quick picker
    await offer_collections_picker()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    if not await still_allowed():
        return
    cl.user_session.set("messages", steps.history_from_thread(thread))
    await cl.context.emitter.set_commands(commands)
    await cl.context.emitter.set_modes(steps.answer_modes())
    await offer_collections_picker()  # with the choice saved in this chat


# ---------- choosing collections (for people with more than one; see kb_picker.py) ----------

async def offer_collections_picker():
    kbs = await user_collections()
    if kb_picker.offered(kbs):
        await send_collections_picker(kbs, kb_picker.from_settings(kbs, cl.user_session.get("chat_settings")))


async def send_collections_picker(kbs: list[dict], selected: set[str]):
    cl.user_session.set("kb_selected", sorted(selected))
    await cl.ChatSettings(kb_picker.widgets(kbs, selected)).send()  # also saves the values as this chat's settings


@cl.on_settings_update
async def on_settings_update(settings: dict):
    kbs = await user_collections()
    if not kb_picker.offered(kbs):
        return
    previous = set(cl.user_session.get("kb_selected") or [k["id"] for k in kbs])
    selected = kb_picker.update(kbs, settings, previous)
    await send_collections_picker(kbs, selected)  # shows the tidied-up boxes (e.g. "All" ticked when every box is)
    if selected:
        await cl.context.emitter.send_toast(f"Your questions in this chat will search {kb_picker.describe(kbs, selected)}.")
    else:
        await cl.context.emitter.send_toast("No collection is ticked. Tick at least one to search.", "warning")


async def collections_to_search(kbs: list[dict]) -> list[dict]:
    """The user's collections narrowed to the ones ticked in this chat (everything if they have only one)."""
    if not kb_picker.offered(kbs):
        return kbs
    selected = kb_picker.from_settings(kbs, cl.user_session.get("chat_settings"))
    return [k for k in kbs if k["id"] in selected]


@cl.on_message
async def on_message(message: cl.Message):
    if not await still_allowed():
        return
    collections = await user_collections()
    if not collections:
        await cl.Message(content=NO_COLLECTIONS_MESSAGE, metadata=dict(steps.NO_COLLECTIONS)).send()
        return
    collections = await collections_to_search(collections)
    if not collections:
        await cl.Message(content=NONE_TICKED_MESSAGE, metadata=dict(steps.NO_COLLECTIONS)).send()
        return
    token = kb.SEARCH_KBS.set(tuple(collections))  # read by the search tool and the system prompt (steps.py)
    try:
        await answer(message, [c["id"] for c in collections])
    finally:
        kb.SEARCH_KBS.reset(token)


async def answer(message: cl.Message, kb_ids: list[str]):
    if message.command == "Fuzzy Search":
        search_results = search_knowledge_base(message.content, exact=False, kb_ids=kb_ids)
        await cl.Message(content=search_results, tags=["command_output"], metadata=dict(steps.COMMAND_OUTPUT)).send()
    elif message.command == "Exact Search":
        search_results = search_knowledge_base(message.content, exact=True, kb_ids=kb_ids)
        await cl.Message(content=search_results, tags=["command_output"], metadata=dict(steps.COMMAND_OUTPUT)).send()
    else:
        # An edited message comes back with its original id: answer it as if what followed it never happened.
        history = steps.history_before(cl.user_session.get("messages") or [], message.id)
        try:
            new_history = await steps.respond(history, message)
        except Exception:
            # Only a finished turn is kept, so a failure half-way (e.g. an AI service is down) can't leave a
            # half-finished tool call in the history and break every later message in this chat.
            logging.exception("Couldn't answer the message")
            await cl.Message(content=ANSWER_FAILED_MESSAGE, metadata=dict(steps.ANSWER_FAILED)).send()
            return
        cl.user_session.set("messages", new_history)
