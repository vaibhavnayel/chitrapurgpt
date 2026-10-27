"""ChitrapurGPT's LLM stack (see docs/AGENT_DESIGN.md).

A Deep Agents main agent with one tool, `search_knowledge_base`. The tool is a fixed pipeline:
query writer -> hybrid retrieval -> dedupe -> per-document readers -> `format_docs`. The readers are plain
structured-output calls behind one process-wide concurrency cap, not subagents. The UI shows only the text the
agent puts inside <final_answer></final_answer>, streamed as it is written.

app.py imports this module as `steps` and reaches into it at call time (tests/test_admin.py stubs it).
"""
from __future__ import annotations

import asyncio
import logging
import re
import weakref
from collections import defaultdict
from datetime import datetime

import chainlit as cl
from deepagents import (FilesystemMiddleware, GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent,
                        register_harness_profile)
from langchain.agents.middleware import ModelCallLimitMiddleware, dynamic_prompt
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import models
import uploads
from retrievers import (FuzzyMatchRetriever, HybridRetriever, deduplicate_docs, format_docs, load_docs_from_jsonl,
                        load_vector_store)

logger = logging.getLogger(__name__)

fuzzy_retriever = FuzzyMatchRetriever(documents=load_docs_from_jsonl("knowledge_base.jsonl"), k=5)
vector_db_retriever = load_vector_store().as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})
retriever = HybridRetriever(fuzzy_retriever=fuzzy_retriever, vector_db_retriever=vector_db_retriever)

# Marks on assistant messages that aren't answers, so a reopened chat leaves them out of the agent's history
# (step metadata is saved to the database; tags and commands aren't).
COMMAND_OUTPUT = {"cgpt_kind": "command_output"}
UPLOAD_NOTE = {"cgpt_kind": "upload_note"}

SYSTEM_PROMPT = """
You are a helpful research assistant who can answer questions about the chitrapur saraswat religious community by thoroughly studying magazine articles in your knowledge base.
you have access to a search_knowledge_base tool that can search the knowledge base to get relevant magazine articles.
You may use this tool multiple times to get more information. When you need several independent searches, you can run them at the same time.
You are also allowed to help with translations and transliterations.
The user may attach files (images, PDFs, spreadsheets, documents). You may read, translate, transliterate, summarise or explain an attached file directly. Facts about the chitrapur saraswat community must still come from the knowledge base, with citations in the usual format. Refer to an attachment by its file name; don't list it under sources.
When responding to the user, add citations to the sources you used to answer in wikipedia format. sources should not be duplicated. citation indices should be sequential starting at 1.
Write each line of the sources list exactly like the example: [n] <title>, <document> page <start>-<end>, where <document> is the source file name from the search results (for example Sunbeam-1996-August.txt), with no comma before "page".
Before you give your answer to the user, think about your reasoning step by step and draft your answer, making sure citation formatting is correct.
Put your final answer in <final_answer></final_answer> tags. Don't forget to close the tags.
Only answer questions with information from the knowledge base. Don't use your own knowledge to answer the question.

Today's date is {today}.

Here is some relevant information about the chitrapur saraswat samaj:
- the main holy site of the chitrapur saraswat samaj is the chitrapur math.
- the religious leader of the chitrapur saraswat samaj is called swamiji.

<citation example>
After looking at the knowledge base, I found the following information:
more information goes here...

<final_answer>
The first president of the chitrapur saraswat samaj was <name> [1] and was appointed on <date> [2]. Additionally, the second president was <name> [1].

sources:
[1] <title of article 1>, <document 1> page <start page number>-<end page number>
[2] <title of article 2>, <document 2> page <start page number>-<end page number>
</final_answer>

</citation_example>
    """


def system_prompt(now: datetime | None = None) -> str:
    """Built for every model call, so the date is today's, not the deploy's."""
    return SYSTEM_PROMPT.replace("{today}", (now or datetime.now()).strftime("%B %d, %Y"))


# ---------------- the search tool ----------------

class Queries(BaseModel):
    reasoning: str = Field(description="the reasoning for the queries")
    queries: list[str] = Field(description="a list of queries to search the knowledge base")


QUERY_PROMPT = """
You are a research assistant whose task it is to look at the question and parse it into a list of queries to search the knowledge base.
Try to make queries short and concise. They will be search with a mix of exact match, fuzzy match and vector search.
    """


@cl.step(name="search query generator")
async def generate_search_queries(research_instructions: str) -> list[str]:
    messages = [
        SystemMessage(content=QUERY_PROMPT),
        HumanMessage(content=f"here is the question: {research_instructions}"),
    ]
    response = await query_llm().ainvoke(messages)
    logger.info(" generated queries: %s", response.queries)
    return response.queries


class RelevantPassages(BaseModel):
    reasoning: str = Field(description="reasoning for the compressed document")
    passages_in_context: list[str] = Field(description="summarised passages with quotes that are relevant to the query. If there is no relevant information, say 'no relevant information found'")


READER_PROMPT = """
Your task is to extract information relevant to the query from the following document.
Your extractions should accurately summarise the document in the context of the query while also quoting the passage verbatim.
Make sure that the quoted passages are talking about the same topic, person or place as the query.
Do not make up answers or include information that is not present in the document.
If you find that the document is not relevant to the query, the passage should say "no relevant information found".
It is also essential that you don't miss any important information and don't include any information that is not present in the document, otherwise the user will not get a complete answer to their question.
The query is: {query}
    """

# One cap on reader calls in flight for the whole process, shared by parallel searches and users (rate limits).
_reader_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def reader_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _reader_semaphores.get(loop)
    if sem is None:
        sem = _reader_semaphores[loop] = asyncio.Semaphore(models.READER_CONCURRENCY)
    return sem


def query_llm():
    return models.query_llm().with_structured_output(Queries, method="json_schema")


def reader_llm():
    return models.reader_llm().with_structured_output(RelevantPassages, method="json_schema").with_retry()


@cl.step(name="document analysis")
async def contextualize_docs(docs: list[Document], query: str) -> list[Document]:
    prompt = READER_PROMPT.replace("{query}", query)
    llm = reader_llm()

    async def read(doc: Document):
        messages = [SystemMessage(content=prompt), HumanMessage(content=f"Here is the document: {format_docs([doc])}")]
        async with reader_semaphore():
            try:
                return await llm.ainvoke(messages)
            except Exception:  # noqa: BLE001  (one unreadable doc must not fail the search)
                logger.exception("reader failed for %r; skipping it", doc.metadata.get("title"))
                return None

    results = await asyncio.gather(*(read(doc) for doc in docs))
    contextualized_docs = []
    for compressed_doc, doc in zip(results, docs):
        if compressed_doc and compressed_doc.passages_in_context and ("no relevant information" not in compressed_doc.passages_in_context[0]):
            # A new Document: the retrieved ones are shared with the fuzzy retriever for the life of the process.
            contextualized_docs.append(Document(
                page_content=f"Here are the relevant passages from this document: {'\n'.join(compressed_doc.passages_in_context)}",
                metadata=doc.metadata))
    logger.info("number of contextualized docs: %d", len(contextualized_docs))
    return contextualized_docs


@tool
@cl.step(name="knowledge base search engine")
async def search_knowledge_base(research_query: str) -> str:
    """
    Search the knowledge base to get relevant magazine articles.
    The query should be a question that you want to answer.
    """

    queries = await generate_search_queries(research_query)

    # search the knowledge base (the fuzzy matcher is CPU-bound, so keep it off the event loop)
    docs = await asyncio.to_thread(retriever.batch, queries)
    docs = deduplicate_docs(docs)

    #contextual compression
    contextualized_docs = await contextualize_docs(docs, research_query)

    return format_docs(contextualized_docs)


# ---------------- the agent ----------------

@dynamic_prompt
def _todays_system_prompt(request) -> str:
    return system_prompt()


def _model_key(model: BaseChatModel) -> str:
    provider = (model._get_ls_params() or {}).get("ls_provider")
    name = getattr(model, "model_name", None) or getattr(model, "model", None)
    return f"{provider}:{name}"


def build_agent(model: BaseChatModel | None = None, tools: list | None = None, model_call_limit: int = 15):
    """The main agent. Deep Agents provides the loop, parallel tool calls, repair of dangling tool calls (a turn
    that died mid-way can't wedge the chat) and Anthropic prompt caching. Its filesystem tools, general-purpose
    subagent and summarisation are switched off: the model sees one tool and exactly our system prompt."""
    model = model or models.main_llm()
    register_harness_profile(_model_key(model), HarnessProfile(
        excluded_tools=frozenset({"read_file"}),
        excluded_middleware=frozenset({"SummarizationMiddleware"}),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ))
    return create_deep_agent(
        model=model,
        tools=tools if tools is not None else [search_knowledge_base],
        name="chitrapurgpt",
        middleware=[
            # Filesystem scaffolding can't be removed. Keep only read_file (hidden above) and turn off the
            # eviction that would swap large tool results or messages for a file pointer the model can't read.
            FilesystemMiddleware(tools=["read_file"], tool_token_limit_before_evict=None,
                                 human_message_token_limit_before_evict=None),
            _todays_system_prompt,
            # A safety net: the old loop had no cap. Real turns use far fewer (at most 9 seen in the eval).
            ModelCallLimitMiddleware(run_limit=model_call_limit),
        ],
    )


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


# ---------------- final answer extraction and streaming ----------------

OPEN, CLOSE = "<final_answer>", "</final_answer>"
_FINAL_ANSWER = re.compile(r'<final_answer>(.*?)</final_answer>', re.DOTALL)


def final_answer_text(text: str) -> str:
    """What the user sees: the inside of the first <final_answer>…</final_answer>, else the whole reply."""
    match = _FINAL_ANSWER.search(text)
    return match.group(1) if match else text


class FinalAnswerExtractor:
    """Incremental version of final_answer_text for one model call's text deltas: returns the newly available
    text inside the first <final_answer>, holding back anything that could be the start of a split tag."""

    def __init__(self) -> None:
        self.buf = ""
        self.state = "before"  # before -> inside -> done

    def feed(self, delta: str) -> str:
        if self.state == "done" or not delta:
            return ""
        self.buf += delta
        if self.state == "before":
            i = self.buf.find(OPEN)
            if i == -1:
                self.buf = self.buf[-(len(OPEN) - 1):]
                return ""
            self.buf = self.buf[i + len(OPEN):]
            self.state = "inside"
        j = self.buf.find(CLOSE)
        if j != -1:
            out, self.buf, self.state = self.buf[:j], "", "done"
            return out
        keep = next((k for k in range(min(len(CLOSE) - 1, len(self.buf)), 0, -1) if CLOSE.startswith(self.buf[-k:])), 0)
        out = self.buf[:len(self.buf) - keep]
        self.buf = self.buf[len(self.buf) - keep:]
        return out


def _chunk_text(chunk) -> str:
    """Text blocks only: thinking and tool-use blocks are never shown."""
    text = getattr(chunk, "text", None)
    return str(text) if text is not None else ""


class AnswerStream:
    """Streams the main model's final-answer text into one Chainlit message. finish() makes the message equal
    final_answer_text(<the last reply>), so what's saved is always what's shown."""

    def __init__(self) -> None:
        self.msg: cl.Message | None = None
        self._call = None
        self._extractor = FinalAnswerExtractor()
        self._streaming_call = None

    async def feed(self, chunk) -> None:
        call = getattr(chunk, "id", None) or self._call
        if call != self._call:  # a new model call
            self._call, self._extractor = call, FinalAnswerExtractor()
        out = self._extractor.feed(_chunk_text(chunk))
        if not out:
            return
        if self.msg is None:
            self.msg = cl.Message(content="")
        # A later model call writing a new final answer replaces the earlier one.
        await self.msg.stream_token(out, is_sequence=self._streaming_call not in (None, call))
        self._streaming_call = call

    async def finish(self, text: str) -> cl.Message:
        if self.msg is None:
            self.msg = cl.Message(content=text)
        else:
            self.msg.content = text
        await self.msg.send()
        return self.msg

    async def discard(self) -> None:
        """The turn failed: don't leave a half-streamed answer on screen that was never saved."""
        if self.msg is not None and self.msg.streaming:
            await self.msg.remove()


# ---------------- one turn ----------------

async def respond(history: list[BaseMessage], message: cl.Message) -> list[BaseMessage]:
    """Run the agent on `message` after `history`; stream and send the answer; return the new history.
    If the turn fails, the exception propagates (Chainlit shows it) and the caller keeps the old history."""
    content, notes = await asyncio.to_thread(uploads.to_content, message.content, list(message.elements or []))
    for note in notes:
        await cl.Message(content=note, metadata=dict(UPLOAD_NOTE)).send()
    human = HumanMessage(content=content, id=message.id)
    answer = AnswerStream()
    state = None
    try:
        async for mode, data in get_agent().astream(
                {"messages": [*history, human]},
                stream_mode=["messages", "values"],
                config={"run_name": "ChitrapurGPT turn", "recursion_limit": 60,
                        "metadata": {"thread_id": getattr(cl.context.session, "thread_id", None)}}):
            if mode == "messages":
                chunk, meta = data
                if meta.get("langgraph_node") == "model":
                    await answer.feed(chunk)
            elif mode == "values":
                state = data
    except BaseException:
        await answer.discard()
        raise
    messages = list(state["messages"])
    final = messages[-1]
    text = final_answer_text(_chunk_text(final))
    if not isinstance(final, AIMessage) or text.startswith("Model call limits exceeded"):
        text = OUT_OF_STEPS  # the safety cap on model calls stopped the turn
        final = AIMessage(content=text)
        messages.append(final)
    await answer.finish(text)
    final.content = text  # later turns see each earlier answer, not its drafts (as before)
    return messages


OUT_OF_STEPS = ("Sorry, I couldn't finish researching this question within my search limit. "
                "Please try asking about one part of it at a time.")


# ---------------- history: edits and reopened chats ----------------

def history_before(messages: list[BaseMessage], message_id: str | None) -> list[BaseMessage]:
    """When the user edits a message, Chainlit re-runs it with its original id: drop it and everything after it."""
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage) and message_id and m.id == message_id:
            return list(messages[:i])
    return list(messages)


# An Exact/Fuzzy Search result: "\n1. <title> (Source: <file>)\nSnippet: ..." (search_engine.py)
_COMMAND_OUTPUT_SHAPE = re.compile(r"^\n\d+\. .*\(Source: [^\n]*\)\nSnippet: ")


def history_from_thread(thread: dict) -> list[BaseMessage]:
    """Agent history for a reopened chat, from its saved user and assistant messages (no tool calls or results,
    as before). Leaves out errors, upload notes, and Exact/Fuzzy Search requests with their results."""
    attachments = defaultdict(list)
    for element in thread.get("elements") or []:
        attachments[element.get("forId")].append(element)
    history: list[BaseMessage] = []
    for step in thread.get("steps") or []:
        kind = (step.get("metadata") or {}).get("cgpt_kind")
        output = step.get("output") or ""
        if step.get("type") == "user_message":
            files = attachments.get(step.get("id"))
            if files:
                output = f"{output}\n\n{uploads.resume_note(files)}" if output else uploads.resume_note(files)
            if not output.strip():
                continue  # e.g. an attachment-only message whose file record is gone: the API rejects empty text
            history.append(HumanMessage(content=output, id=step.get("id")))
        elif step.get("type") == "assistant_message":
            if step.get("isError") or kind == UPLOAD_NOTE["cgpt_kind"]:
                continue
            if kind == COMMAND_OUTPUT["cgpt_kind"] or _COMMAND_OUTPUT_SHAPE.match(output) or not output.strip():
                if history and isinstance(history[-1], HumanMessage):
                    history.pop()  # the command itself
                continue
            history.append(AIMessage(content=output, id=step.get("id")))
    return history
