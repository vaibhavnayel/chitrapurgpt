# Agent design: ChitrapurGPT on LangChain Deep Agents

Status: **as built (Phase B)**. Code: `steps.py` (agent, tool, streaming, history), `models.py` (model settings),
`uploads.py` (attachments), `app.py` (Chainlit handlers), `ingest.py`. Parity requirements and every behaviour change
are in `docs/PARITY.md`. Old-vs-new measurements: §13 below; full report in `eval/REPORT.local.md` (local only).

APIs below were checked against the installed source:
- deepagents 0.7.32
- langchain 1.4.7, langchain-core 1.7.0, langgraph 1.2.18, langgraph-prebuilt 1.1.1, langchain-anthropic 1.7.6
- Chainlit 2.9.6
- the Anthropic models API for `claude-sonnet-5-5` / `claude-haiku-5-5` capabilities

Where the build differs from the Phase A proposal, the text says so ("Built:").

File references are given as `pkg:path:line`.

## 1. Summary

```
Chainlit on_message ──► deep agent (Sonnet 5.5, adaptive thinking)          one LangSmith trace per turn
                          │  tools: [search_knowledge_base]   (parallel calls allowed)
                          │
                          ├─ search_knowledge_base(research_query)   ← plain async tool, not a subagent
                          │     @cl.step "knowledge base search engine"
                          │       ├─ @cl.step "search query generator"  (CGPT_QUERY_MODEL, structured Queries)
                          │       ├─ HybridRetriever (unchanged; run off the event loop)
                          │       ├─ dedupe (unchanged)
                          │       └─ @cl.step "document analysis": per-doc Haiku 5.5 readers,
                          │            structured RelevantPassages, global semaphore, failures skipped
                          │     returns retrievers.format_docs(kept)  (byte-identical text)
                          │
                          └─ streamed tokens ──► <final_answer> extractor ──► cl.Message.stream_token
```

- **One agent, one tool, no subagents.**
- The per-document reading stays a deterministic fan-out inside the tool.
- Deep Agents provides the agent loop, parallel tool execution, dangling-tool-call repair, prompt caching and
  safe handling of unsupported content.
- Everything else Deep Agents offers is switched off, because it either duplicates our tool or would change what
  the agent sees.

## 2. What is an agent, what is a subagent, what stays a function

| Piece | Choice | Why |
|---|---|---|
| Main agent | `create_deep_agent(...)`, model `CGPT_MAIN_MODEL` (default `claude-sonnet-5-5`) | Replaces `respond_to_user_message` / `handle_tool_call` / the `while tool_calls` loop. |
| `search_knowledge_base` | **Plain async LangChain tool** (same name, arg and description) | It's a fixed pipeline (queries → retrieve → dedupe → read → format). The agent only needs its output, and that output must be the exact `format_docs` text the citations depend on (PARITY 4.10). A subagent returns only its last AI message text (deepagents `middleware/subagents.py:747-787`), so the formatted metadata would have to be re-emitted verbatim by an LLM. That costs tokens, and the LLM can drop or alter fields. |
| Query writer | Plain structured-output call inside the tool (`CGPT_QUERY_MODEL`, default = main model) | One deterministic call; no loop needed. Kept for parity. Phase B will also evaluate `claude-haiku-5-5` here (cheaper, faster); the default changes only if the eval shows no loss. |
| Per-document readers | **Plain structured-output calls** (`CGPT_READER_MODEL`, default `claude-haiku-5-5`), `asyncio.gather` behind one process-wide `asyncio.Semaphore(CGPT_READER_CONCURRENCY=40)` | See below. |
| General-purpose subagent | **Disabled** | It would duplicate the tool and let the main agent delegate whole research tasks to a context that can't see the citations. |

**Why readers are not subagents.** The baseline reads 38 unique docs per search, with up to 8 searches per question:
1,367 reader calls over 15 questions, about 91 per question. As subagents, each read would be:
- a `task` tool call the main model has to *write*: Sonnet output tokens, plus a tool-call round trip per batch;
- its own agent loop, with the subagent middleware stack (filesystem layer, prompt caching, unsupported content);
- a free-text answer that we'd have to parse back into `RelevantPassages`.

The reading has no decisions in it: every retrieved doc must be read exactly once with the same prompt. So a subagent
adds cost, latency and nondeterminism and gains nothing. Inside the tool, the readers still trace as child runs of the
tool in LangSmith, so observability doesn't suffer.

**Where subagents could earn their keep later (not proposed now):**
- A "researcher" subagent for the exhaustive questions ("every mention of…", "all yatras since 1997"): it runs many
  searches in its own context and returns a condensed, cited digest. The main context would stay small. But
  citations would then pass through a second LLM, so it needs its own eval.

## 3. Deep Agents built-ins: what's on and off

deepagents 0.7.32 main-agent stack, in order (`graph.py:907-988`):
1. Filesystem
2. SubAgent (only if subagents exist)
3. ParentNotifier
4. Summarization
5. PatchToolCalls
6. **user middleware**
7. Anthropic prompt caching
8. HITL (only if configured)
9. UnsupportedContent
10. ToolExclusion

| Built-in | Decision | Reason / mechanism |
|---|---|---|
| Planning / `write_todos` | n/a | Removed from deepagents in 0.7.0 (CHANGELOG). |
| Virtual filesystem (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`) | **Off, as far as the API allows** | `FilesystemMiddleware` is required scaffolding and can't be excluded (`graph.py:282` `_REQUIRED_MIDDLEWARE_NAMES`; `profiles/harness/harness_profiles.py:54-75`). Pass `FilesystemMiddleware(tools=["read_file"], tool_token_limit_before_evict=None, human_message_token_limit_before_evict=None)` (same `.name`, so it replaces the default in place) and hide `read_file` with `excluded_tools`. **The eviction settings matter:** by default any tool result over 20,000 tokens × 4 chars (`middleware/filesystem.py:1898`) is replaced by a preview plus a file pointer. That would silently cut the docs text that citations depend on. Uploaded text over 50k tokens in a HumanMessage would be evicted the same way. |
| `task` tool + general-purpose subagent | **Off** | `HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False))` and no `subagents=` (then `SubAgentMiddleware` is never added). |
| Summarization | **Off** | `excluded_middleware={"SummarizationMiddleware"}`. Sonnet 5.5 has a 1M-token window, so the default trigger (85% of the window) never fires in real chats; the longest real thread is a few dozen turns. When it did fire it would make an extra Sonnet call, and with no checkpointer it would re-summarise every turn. Without it, every turn gets the full history, as today. |
| Anthropic prompt caching | **On** (5-minute TTL) | Caches the system prompt, the tool definition and the conversation prefix. Later turns and later loop iterations re-read cached tokens at $0.10/MTok instead of $2. Matters most for long threads and big uploads. |
| PatchToolCalls | **On** | Repairs `tool_use` without a `tool_result` in the incoming history. This is the fix for PARITY 3.6 (a chat stuck after a mid-turn exception). |
| UnsupportedContent | **On** | Replaces blocks the model can't take with a text placeholder instead of an API error. Our upload code never sends non-PDF file blocks, so this is only a safety net. |
| Skills, memory, permissions, HITL, async subagents | Off (not configured) | Only added when their arguments are passed. |

Configuration (the profile is process-global, so it's registered once at import, keyed to the configured model):

```python
register_harness_profile(f"anthropic:{MAIN_MODEL}", HarnessProfile(
    excluded_tools=frozenset({"read_file"}),
    excluded_middleware=frozenset({"SummarizationMiddleware"}),
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
))
agent = create_deep_agent(
    model=main_llm, tools=[search_knowledge_base], name="chitrapurgpt",
    middleware=[
        FilesystemMiddleware(tools=["read_file"], tool_token_limit_before_evict=None,
                             human_message_token_limit_before_evict=None),
        system_prompt_for_today,                      # @dynamic_prompt, see §5
        ModelCallLimitMiddleware(run_limit=15),       # safety net; today there is no cap
    ],
)
```

Built: `steps.build_agent(model=None, tools=None, model_call_limit=15)`. The profile is registered under the key
deepagents derives from the model instance, `"{ls_provider}:{model}"`, so it also applies to the fake models in the
unit tests and to whatever `CGPT_MAIN_MODEL` is. When the cap stops a turn, the middleware's "Model call limits
exceeded" text is replaced with a friendly sorry-message (`steps.OUT_OF_STEPS`). The eval never came near the cap.
A unit test asserts the model is bound to exactly one tool, `search_knowledge_base`, and sees exactly
`system_prompt(today)`.

Result: the model sees exactly one tool and exactly our system prompt. No deepagents base prompt is added for
unprofiled models (`graph.py:363-382`; built-in profiles exist only for `claude-sonnet-4-6`, `claude-haiku-4-5`,
`claude-opus-4-7`).

> **Note for Vaibhav (not re-opening the decision):** once all of that is switched off, `create_deep_agent` is
> langchain's `create_agent` + `PatchToolCallsMiddleware` + prompt caching, with a hidden, inert filesystem layer and
> a beta, process-global profile registry. The design keeps Deep Agents as decided. If the hidden layer ever gets in
> the way, dropping to `create_agent` with those two middleware is a ~10-line change and the rest of this design
> carries over.

## 4. Models and settings (all env vars)

| Env var | Default | Used for |
|---|---|---|
| `CGPT_MAIN_MODEL` | `claude-sonnet-5-5` | main agent |
| `CGPT_QUERY_MODEL` | `$CGPT_MAIN_MODEL` | query writer |
| `CGPT_READER_MODEL` | `claude-haiku-5-5` | per-doc readers |
| `CGPT_INGEST_MODEL` | `claude-sonnet-5-5` | `ingest.py` article-boundary extraction |
| `CGPT_MAIN_EFFORT` | by model family: `claude-opus-*` → `high`, `claude-sonnet-*` → `xhigh` | supervisor thinking effort (optional override) |
| `CGPT_READER_EFFORT` | `xhigh` | reader thinking effort (`off` disables) |
| `CGPT_READER_CONCURRENCY` | `40` | global cap on concurrent reader calls (same number as today's per-search cap) |

Built: all of these live in `models.py`, with `chat_model(model, max_tokens, effort, thinking)` choosing the right
sampling settings per model family. Production needs no new env vars.

**Thinking configuration (Vaibhav's decision, replacing "let the eval pick low or medium"):**
- The supervisor runs at `xhigh` on Sonnet (the default) and at `high` if pointed at Opus. Only Sonnet 5.5 was
  benchmarked; Vaibhav chose not to spend eval budget on an Opus supervisor.
- Readers (Haiku 5.5) think at `xhigh`. The proposal had their thinking off.
- `max_tokens` covers the thinking:
  - supervisor 32,000, always streamed, which also lifts the Anthropic SDK's cap on non-streaming requests;
  - readers 20,000, under that cap.
- The query writer stays at `low`, and `ingest.py` is unchanged.
- The superseded low/medium runs are kept in §13 as data points.

OpenAI stays only for `text-embedding-3-large` (the vector index; Pinecone until Phase C, now pgvector, §14).
Readers move from gpt-5-mini to Haiku 5.5, so chat LLMs are Anthropic-only.

**Sampling and thinking (PARITY 11).** `temperature=0` can't be kept on the main model. langchain-anthropic 1.7.6
raises `ValueError` for `claude-sonnet-5-5` when `temperature` ≠ 1, `top_k` is set, `top_p` ≠ 1, or `thinking` is
`disabled`/`enabled` (`chat_models.py:1810-1846`). The models API agrees: adaptive thinking only.

Built, also: **Haiku 5.5 rejects `temperature` too.** The API returns 400 "`temperature` is deprecated for this
model", which the proposal didn't anticipate. So `temperature=0` is now set only for Claude 4.x and older (if
someone points an env var at one); 5.x models get thinking settings instead.

The plan:
- Main agent: `ChatAnthropic(model=…, effort=CGPT_MAIN_EFFORT, max_tokens=16000)`.
  - No temperature.
  - `max_tokens` doubles from 8,000 because thinking tokens count against it. The visible answer budget stays at
    least what it is today; baseline final answers are ≤ ~1k tokens.
  - The system prompt keeps "think step by step and draft, then `<final_answer>`". Whether the model drafts in thinking
    blocks or in text, the visible output has the same shape, and the extractor (§6) handles both.
  - Phase B evaluates `effort=low` vs `medium` on the question set and picks the cheaper one unless citations or
    answers degrade.
- Query writer on Sonnet 5.5: forced `tool_choice` is rejected (`chat_models.py:1155-1159`), so use
  `with_structured_output(Queries, method="json_schema")` (native structured outputs; supported per the models API).
  Built: with `effort="low"` (writing a few short queries needs little thought, and it keeps each search fast).
- Readers on Haiku 5.5: proposed with thinking disabled. Built: thinking on at `xhigh` (see above), no
  temperature, `max_tokens=20000`, `with_structured_output(RelevantPassages, method="json_schema")`, `.with_retry()`
  as today. Checked live: a reader call takes 3–5 s and ~400–800 output tokens including thinking.
- Built: both the query writer and the readers are created with `disable_streaming=True, tags=["nostream"]`.
  LangGraph's message streaming would otherwise make every chat model in the graph stream internally, readers
  included, and the tag keeps their tokens out of the UI stream.
- Checked against the real models: `Queries` and `RelevantPassages` (both with a `reasoning` field) parse fine with
  native structured output on Sonnet 5.5 and Haiku 5.5. A field named `scratchpad` makes Sonnet 5.5 stop with
  `stop_reason="refusal"` and no output (see §9).

## 5. The tool, step by step (parity with `steps.py`)

```python
@tool
@cl.step(name="knowledge base search engine")
async def search_knowledge_base(research_query: str) -> str:
    """Search the knowledge base to get relevant magazine articles.
    The query should be a question that you want to answer."""
    queries = await generate_search_queries(research_query)            # @cl.step "search query generator"
    docs = await asyncio.to_thread(retriever.batch, queries)           # same HybridRetriever, off the loop
    docs = deduplicate_docs(docs)                                      # unchanged
    kept = await contextualize_docs(docs, research_query)              # @cl.step "document analysis"
    return format_docs(kept)                                           # unchanged retrievers.format_docs
```

- Prompts, schemas (`Queries`, `RelevantPassages`) and the keep filter (first passage, case-sensitive
  `"no relevant information"`) are copied verbatim.
- `contextualize_docs` rewrites **copies** of the docs (fixes PARITY 4.9).
- Each read runs `async with READER_SEMAPHORE:` and is wrapped in `try/except`: a doc that still fails after retries
  is logged and skipped (PARITY 10.2).
- `format_docs`, `deduplicate_docs`, `FuzzyMatchRetriever` and `HybridRetriever` are reused unchanged from
  `retrievers.py`. The Pinecone wrapper was too, until Phase C replaced it with `vector_store.py` (§14).
- Module name: the agent code stays in `steps.py`, and `app.py` switches to `import steps` (module access at call
  time). `tests/test_admin.py` already replaces `sys.modules["steps"]` with a stub, so the admin test keeps working
  with no edit and without the vector database.

**System prompt per turn (PARITY 1.7/12).** The prompt text moves to `steps.py` as a template, and a `@dynamic_prompt`
middleware (langchain `agents/middleware/types.py:1690`) fills in today's date on every model call. The session
history no longer starts with a `SystemMessage`.

## 6. Chainlit integration: streaming, steps, history

### 6.1 One turn

```python
async def on_message(message):
    if not await still_allowed(): return
    if message.command in COMMANDS: ...unchanged...; return
    history = steps.history_before(cl.user_session.get("messages"), message.id)   # edit-message truncation (§6.4)
    human = HumanMessage(content=await uploads.to_content(message), id=message.id)
    answer = steps.AnswerStream()            # owns the cl.Message and the <final_answer> extractor
    async for mode, data in steps.agent.astream(
            {"messages": [*history, human]},
            stream_mode=["messages", "values"],
            config={"run_name": "ChitrapurGPT turn", "recursion_limit": 60,
                    "metadata": {"thread_id": cl.context.session.thread_id}}):
        if mode == "messages":
            chunk, meta = data
            if meta.get("langgraph_node") == "model":       # main agent only (readers are nostream anyway)
                await answer.feed(chunk)
        else:
            state = data
    final_ai = state["messages"][-1]
    text = steps.parse_final_answer_text(final_ai.text)      # the same regex as today
    await answer.finish(text)                                # displayed == saved, guaranteed
    final_ai.content = text                                  # PARITY 2.6
    cl.user_session.set("messages", state["messages"])       # appended once (fixes PARITY 3.2)
```

Built: the turn lives in `steps.respond(history, message)`, so `app.on_message` is three lines:
1. `history = steps.history_before(...)`
2. `new_history = await steps.respond(history, message)`
3. `cl.user_session.set("messages", new_history)`

The session is only updated once the turn has finished. If the turn raises, a half-streamed answer is removed from
the screen (it was never saved), Chainlit shows the error, and history stays at the previous turn. That's the fix
for the "stuck chat" bug, tested with a model failure in the middle of a tool loop. Upload notes, if any, are sent
before the answer.

### 6.2 Streaming only the final answer

`stream_mode="messages"` yields `(AIMessageChunk, metadata)` for every chat-model call in the graph. The metadata
includes `langgraph_node`, `langgraph_step`, `checkpoint_ns` and `ls_model_name` (langgraph
`pregel/_algo.py:1005-1016`, `pregel/_messages.py:150-157`). We keep chunks from node `"model"` (the main agent's
node in `create_agent`). Readers and the query writer run in node `"tools"` and are also tagged `nostream`, which
LangGraph drops (`constants.py:24`).

`AnswerStream.feed(chunk)` takes `chunk.text`, i.e. text blocks only, with no thinking or tool_use. It runs a small
state machine per model call (a new call starts when the chunk `id` changes):
- outside `<final_answer>`: buffer (a tag can be split across chunks, so hold back up to 13 chars) and stream nothing;
- inside: stream to `cl.Message.stream_token`, stop at `</final_answer>`.

A model call that ends in tool calls usually has no tags, so the reasoning text before a search is never shown, same
as today. If a later call streams a new final answer, the message is reset (`is_sequence=True`).

`finish(text)` computes the answer exactly as today (`parse_final_answer` regex on the last AI message's text):
- tags present: `text` equals what was streamed, so just `send()`;
- tags missing, or opened but never closed: today's fallback is the whole reply, so the message content is replaced
  with it and sent.

Either way the persisted `assistant_message` output equals what's on screen. The harness checks this
(`streamed_equals_saved`), and unit tests replay recorded raw outputs from the baseline through the extractor.

Expected effect: the answer starts appearing as soon as Sonnet begins writing `<final_answer>`, instead of after the
whole turn. That saves roughly the final call's answer-writing time (baseline: ~10–18 s for long answers).

### 6.3 Steps

Steps stay `@cl.step` decorators inside the tool rather than being rebuilt from LangGraph events:
- the step inputs and outputs come out byte-for-byte as today;
- nesting comes for free from Chainlit's `local_steps` context var.

LangGraph runs each tool call as its own asyncio task, which copies the context (Chainlit's context var and
`local_steps` included). So parallel searches each get their own "knowledge base search engine" step with their own
"search query generator" and "document analysis" children. A Phase B unit test runs two searches concurrently against
the recording emitter and checks the trees (PARITY 5.2). No step is created for the main model calls (none today).

### 6.4 History, resume, edits

- **Memory:** no checkpointer. The full message list (AI messages with `tool_calls`, ToolMessages, the new
  HumanMessage) is passed in every turn and the final state's list is saved back into `cl.user_session["messages"]`.
  This is the same model as today, and LangGraph supports it (no required state keys; the reducer dedupes by id).
  AI messages are kept exactly as returned (thinking blocks + signatures) because Anthropic requires them inside a
  tool-use loop. Only the final AI message's content is replaced by the extracted answer, as today.
- **Resume:** rebuild from `user_message`/`assistant_message` steps as today, setting `HumanMessage(id=step["id"])`.
  Skip `isError` steps and Exact/Fuzzy Search pairs (PARITY 3.4). New command output carries
  `metadata={"cgpt_kind": "command_output"}` (step metadata is persisted, tags aren't); old threads are recognised by
  the command output's shape. Built (`steps.history_from_thread`), also skipping:
  - empty assistant outputs (a command with no matches), together with the command before them;
  - upload notes;
  - user messages that would be empty (the API rejects empty text).
- **Edits (PARITY 9):** Chainlit re-runs `on_message` with the edited message's original id.
  `history_before(messages, id)` cuts the list just before the HumanMessage with that id, so the agent sees what the
  user sees.

## 7. Uploads

The UI accepts any file (20 max, 500 MB each). Production so far: jpeg, png, pdf, xlsx and 2× octet-stream. In
`uploads.to_content(message)`, each element becomes a content block, attachments first, then the user's text:

| Type (by MIME, then extension, then magic bytes via `filetype`, already a Chainlit dependency) | Sent as | Cap |
|---|---|---|
| jpeg, png, gif, webp | `{"type": "image", "base64", "mime_type"}` (langchain standard block → Anthropic image) | Resized to ≤ 1568 px long edge with Pillow, which is what the API would do anyway, so no quality loss; keeps requests small. Max 10 images per message. |
| PDF | `{"type": "file", "base64", "mime_type": "application/pdf", "extras": {"title": name}}` → Anthropic document block (text + page images) | Built: native if ≤ 100 pages and ≤ 10 MB, and ≤ 15 MB of native PDFs per message (to stay well under the 32 MB request limit); otherwise extracted text via `pypdf`, with a note to the user. Password-protected PDFs get a note. |
| any file over 50 MB | not read; note to agent and user | n/a |
| xlsx | text: per sheet, `## Sheet <name>` + rows as CSV (`openpyxl`, read-only, values only) | 100k chars per file, 200k per message, with a "(truncated)" note |
| docx | text from `word/document.xml` (stdlib `zipfile`), paragraphs + table cells | same |
| csv, txt, md, json, html/xml, and octet-stream that decodes as UTF-8 | text | same |
| anything else, or any failure | a text block for the agent (`[Attachment "x.bin" (application/octet-stream, 1.2 MB) could not be read: unsupported type]`) **and** a short visible note to the user (`cl.Message`, `metadata={"cgpt_kind": "upload_note"}`, skipped on resume) | n/a |

New dependencies: `pillow` and `openpyxl` (pinned).

**System prompt addition** (keeping everything else):
> The user may attach files (images, PDFs, spreadsheets, documents). You may read, translate, transliterate,
> summarise or explain an attached file directly. Facts about the Chitrapur Saraswat community must still come
> from the knowledge base, with citations in the usual format. Refer to an attachment by its file name; don't list
> it under sources.

**On resume** (decided): attachments are **not** re-sent. Chainlit deletes `.files/<session>` at session end, and
unless a storage client is configured the `Element` row has no file. A resumed thread's HumanMessage gets a note:
`[The user attached: note.pdf (application/pdf). The file isn't available in this reopened chat.]`. The agent then
knows a file existed and can ask for a re-upload, and its earlier answer (e.g. the translation) is still in history.
Within a live session, attachments stay in history for follow-ups ("now translate page 2"). Prompt caching makes
re-sending them cheap ($0.10/MTok after the first turn).

## 8. LangSmith

One `astream` call is one root run (`run_name="ChitrapurGPT turn"`). The model, tools and middleware nodes nest under
it; the query writer and reader runs nest under each tool run. deepagents adds `ls_integration="deepagents"`. Env vars
are unchanged (`LANGCHAIN_TRACING`, `LANGCHAIN_PROJECT`, `LANGCHAIN_API_KEY`). The `@traceable` decorators are
removed. Retrieval is moved off the event loop with `asyncio.to_thread`, which copies context, so the retriever runs
nest too. Phase B checks this with `eval/langsmith_traces.py --label <run>`. The baseline already has 1 root per
turn (`respond_to_user_message`, 170–810 runs each), so the requirement is to keep that, now with the graph's
model/tools structure, and to lose the "No indexed run ID" tracer warnings the current reader fan-out logs.

## 9. `ingest.py`

Only the model changes: `models.chat_model(CGPT_INGEST_MODEL, max_tokens=20000)` (no temperature on Sonnet 5.5;
20,000 stays under the Anthropic SDK's limit for a non-streaming call),
`.with_structured_output(ArticleBoundaries, method="json_schema", include_raw=True).with_retry()`, same prompt. The
cost print switches to `usage_metadata`.

Built, one forced schema change: Sonnet 5.5 returns `stop_reason="refusal"` with no output whenever the structured
output schema has a field named **`scratchpad`**. The original `ArticleBoundaries` has one, described as "space for
thinking and reasoning". This was isolated with probes: the same schema without the field, or with it renamed,
works; keeping the name with a neutral description still refuses. The field is now `notes`. Nothing reads it, and the
article boundaries themselves are unchanged.

Tests: `tests/test_ingest.py` is offline (a fake model; `main()` runs in a temp dir with `add_to_vector_store`
stubbed, and the real `knowledge_base.jsonl` is hash-checked as untouched). `tests/ingest_live_check.py` runs the
real model on a short two-article sample (about $0.01; it extracted cover, contents, editorial and the Rathotsava
article with the right author and dates).

## 10. Dependencies (Phase B)

deepagents 0.7.32 needs:
- `langchain>=1.4.7`
- `langchain-core>=1.7.0`
- `langchain-anthropic>=1.7.6` (which needs `anthropic>=0.120,<2`)
- `langsmith>=0.14.12`
- `langchain-google-genai>=4.4.2` (a hard dependency even though we don't use Gemini)

So these pins move together:

| package | before | built (exact pins in `requirements.txt`) |
|---|---|---|
| deepagents | n/a | 0.7.32 |
| langchain | 1.0.2 | 1.4.7 |
| langchain-core | 1.6.10 (transitive) | 1.7.0 (pinned) |
| langgraph | 1.0.10 (transitive) | 1.2.18 (pinned) |
| langchain-anthropic | 1.0.0 | 1.7.6 |
| anthropic | transitive | 0.125.0 (pinned) |
| langchain-openai | 1.0.1 | 1.7.3 (embeddings only) |
| openai | transitive | 2.54.0 (pinned; see below) |
| langchain-community | 0.4.1 | 0.4.2 (`ingest.py`'s PDF loader) |
| langchain-google-genai | 3.0.0 | 4.4.2 |
| langsmith | 0.4.38 | 0.14.12 |
| pillow, openpyxl | n/a | 13.0.0, 3.1.5 |

Built:
- `uv pip compile requirements.txt --python-version 3.12` resolves cleanly, and the dev venv was synced to that
  resolution, so the tests ran against what a fresh install gets.
- `openai` is pinned to 2.54.0 because a fresh resolve picks 3.x. That moves to `httpx2` + `truststore`, which
  verifies TLS through the OS trust store; it failed in the sandbox (`OSStatus -26276`) where 2.x worked. Pinning the
  tested version keeps the embeddings client as it was. Moving to 3.x later is a one-line change worth a separate
  test on Render.

- Chainlit stays at 2.9.6: it doesn't pin langchain (only in its `tests` extra).
- `langchain-groq` and `rank_bm25` are unused. They're left alone unless they block the resolve.
- Admin and preview tests are re-run after the upgrade.

## 11. Latency and cost: expected (Phase A) vs measured (Phase B)

**Measured** on the same 14 questions, old stack vs the shipped configuration (Sonnet 5.5 at `xhigh`, Haiku 5.5
readers at `xhigh`, full hybrid retrieval). Details in §13.

| | Old stack | New stack | |
|---|---|---|---|
| Latency per question | mean 114 s, p50 117 s, p90 208 s | mean 90 s, p50 79 s, p90 177 s | −21% mean |
| Time to first visible answer text | p50 117 s (= whole turn) | p50 66 s | streams, but late: `xhigh` thinking and searching take most of the turn |
| Cost per question | mean $0.37 | mean $0.33 | −12% |
| Searches per question | 2.5 (serial) | 4.5 (parallel) | the `xhigh` supervisor searches more |
| Blind judge (Opus 5.5) | 1 win | 13 wins | faithfulness 4.71 vs 3.86, quality 4.57 vs 3.43 |

The Phase A estimates below assumed readers without thinking and a low/medium supervisor. A superseded run of that
configuration on the 30 real questions cost $0.06 and took 24 s per question, against $0.46 and 117 s for the shipped
`xhigh` configuration: about 7× cheaper and 5× faster. It's not a clean comparison, because that run had fuzzy-only
retrieval and the pre-fix prompt. Vaibhav chose `xhigh` for quality. §13 has the numbers.

Phase A analysis of the old stack (§13; 15 questions, 35 searches). Per search: 6.1 queries, 61 docs retrieved, 38
unique, 13.7 kept.
- Retrieval p50 6.8 s, synchronous.
- Reading p50 19.6 s: gpt-5-mini, 74% of its output tokens are reasoning.
- Search total p50 29 s, p90 48 s.
- Agent LLM calls are short (1.5–4.5 s) except the final answer.
- Searches are strictly sequential (`parallel_tool_calls=False`); 2.3 per turn on average, up to 8. One real
  question made 8 searches in 247 s, each a reformulation of the same question.

Phase A estimates (superseded by the measurements above):

| | Baseline (measured) | New stack (estimate, Phase B measures) | Driver |
|---|---|---|---|
| Reader cost per search | $0.087 (gpt-5-mini: ~71k in, ~35k out of which ~26k reasoning) | ≈ $0.01–0.015 | Haiku 5.5 at $0.10/$0.50 per MTok, thinking off: same ~71k in, ~9k visible out |
| Main agent + query writer per question | $0.14 (Sonnet 4.5; ~39k input tokens per question, as tool outputs pile up in the loop) | ≈ $0.06–0.15 | Sonnet 5.5 is cheaper per token ($2/$10) and prompt caching makes the re-sent loop prefix $0.10/MTok, but adaptive thinking adds output. Biggest uncertainty: `effort`. |
| Cost per question | mean $0.37, p50 $0.39, p90 $0.60, max $0.95 | ≈ $0.10–0.19 (about 50–70% lower) | readers dominate today ($3.05 of $5.17) |
| Search latency | p50 29 s, p90 48 s | ≈ 15–20 s | Haiku without reasoning reads a doc in a few seconds, so reading wall time ≈ the slowest doc; retrieval unchanged (6.8 s) but off the event loop; the query writer may get slower with thinking |
| Multi-search questions | strictly serial | independent searches in parallel | the 8-search question could take ~3 rounds |
| Turn latency | p50 117 s, p90 185 s | ≈ 40–50% lower | the two rows above |
| Time to first visible answer text | = full turn (p50 117 s) | full turn minus answer-writing time | streaming |

### Risks (Phase A list, with what Phase B found)

Outcomes:
- Risk 1: tags 46/47 turns; one forced prompt fix (the sources line format) and one extraction fix (sources written
  after `</final_answer>`).
- Risk 2: grounding went up (100% vs 92%), and 17 docs are kept per search vs 13.7.
- Risk 3: all tests pass on the new pins.
- Risk 5: 0 reader errors in 14,491 calls.
- Risk 6: proven by unit tests and the E2E.
- Risk 8: unchanged.
- New risk, cost tail: "every mention / complete list" questions trigger up to 30 searches ($1.91, 449 s) at `xhigh`.

1. **Answer style and determinism change with Sonnet 5.5.** There's no temperature control and adaptive thinking is
   always on. Mitigation: the same prompt; eval comparison of tag compliance, citation format/grounding, sources per
   answer, and side-by-side reading of the edge cases.
2. **Haiku 5.5 reader quality vs gpt-5-mini with reasoning.** It could miss passages or keep noise. The keep filter is
   a case-sensitive exact phrase: if Haiku writes "No relevant information found." (capital N), irrelevant docs are
   kept. Mitigation: eval "kept per search" (baseline 13.7) and grounded-citation rates vs baseline. In the baseline,
   0 of 479 kept docs had a "no relevant information" first passage in any case. If Haiku capitalises the phrase,
   propose (and flag) a case-insensitive check.
3. **Dependency jump.** langchain 1.0 → 1.4, core 1.6 → 1.7, langsmith 0.4 → 0.14, google-genai 3 → 4. Risk to
   Chainlit 2.9.6 and to our `PineconeVectorStore` wrapper (it implements langchain-core's `VectorStore`).
   Mitigation: full test suite + eval + preview.
4. **deepagents churn and beta APIs.** Releases are frequent (0.7.29 → 0.7.32 in days) and the profile registry is
   beta and process-global. Mitigation: exact pins; a unit test asserting the model sees exactly one tool, our
   system prompt, and no `task`/filesystem tools.
5. **Parallel searches multiply reader load.** 3 parallel searches × ~27 docs ≈ 80 Haiku calls. The global semaphore
   (40) bounds concurrency, but Anthropic rate limits for Haiku 5.5 on this account are unknown. Mitigation: SDK
   retries + `with_retry`; a failed doc is skipped rather than failing the turn.
6. **Context propagation into LangGraph tool tasks** (Chainlit steps nesting, LangSmith nesting) is expected from
   asyncio semantics but must be proven. Unit test in Phase B.
7. **Big uploads in history.** A 100-page PDF is ~200k tokens and is re-sent every turn of that session. Prompt
   caching cuts that to ~$0.02 per later turn, but latency grows. The caps in §7 bound it.
8. **The eval runs with environment shims** (`eval/sandbox_shims.py`). Embeddings skip the local tiktoken length
   check because the tokenizer download host is blocked here. Until Phase C, the Pinecone SDK was also given the
   sandbox's HTTPS proxy explicitly; that shim went with Pinecone. Shims don't change app behaviour and are recorded
   in each result's `stack.eval_shims`.

## 12. Environment notes

Phase B resolution of the Phase A blockers:
- **Packages:** installed with `uv pip … -p .venv`.
- **Postgres:** a local Postgres 15 was provided; `tests/test_admin.py` passes.
- **OpenAI:** credit stayed exhausted and the key was removed. For the local eval only, embeddings
  (`text-embedding-3-large`, the same model as the Pinecone index) come from an OpenAI-compatible endpoint set
  through the standard `OPENAI_BASE_URL`. Nothing in the app names a provider.
  - The endpoint was unreachable from the sandbox for part of the session. Runs from that period used fuzzy-only
    retrieval and are marked superseded.
  - The final run and the E2E use full hybrid retrieval.
- **Budget:** raised to $40 for both phases.

The Phase A text follows.

1. **OpenAI credit.** The OpenAI account behind `.env` returns `429 insufficient_quota` for chat *and* embeddings.
   The new stack still needs embeddings for every Pinecone search, so Phase B can't run any search until credit is
   added. If this is the same key as production (the `.env` says it was copied from Render), live ChitrapurGPT
   searches are failing now too.
2. **Installing packages.** `uv venv` panics inside this sandbox (macOS `system-configuration` "Attempted to create a
   NULL object"), and `uv pip install` / `pip install` need an approval that a non-interactive run can't give.
   deepagents was therefore verified from its GitHub source (main, `version = "0.7.32"`, the same as the latest
   PyPI release), not an installed copy. Phase B needs installs pre-approved, or a venv with the new pins prepared
   outside the sandbox.
3. **Postgres for `tests/test_admin.py`.** `initdb` fails here because the sandbox denies System V shared memory
   (`shmget`). The admin test has to run outside this sandbox (e.g. by Hermes).
4. **Budget.** Spend so far is $5.80 of the ~$20 for both phases ($3.50 OpenAI, $2.30 Anthropic). Finishing the
   baseline on all 30 real questions + resume + edit would cost ≈ $9 more at the measured $0.30–0.37 per question.
   Phase B's full new-stack run is ≈ $2–4 for 43 questions, plus iteration on subsets.

## 13. Results

### 13.1 New stack (shipped configuration), 2026-10-27

All 45 questions (30 real, 15 edge, including two with uploads). Sonnet 5.5 supervisor at `xhigh`, Haiku 5.5 readers
at `xhigh`, full hybrid retrieval. Answers are as displayed by the shipped extraction: 3 of the 47 turns were
re-extracted from their recorded raw output after the sources-after-the-tag fix. Run with
`eval/run_eval.py --label new-xhigh`.

| Metric | New stack (45 questions) | Old stack (15 questions, §13.2) |
|---|---|---|
| Errors | 0 | 0 |
| Latency per question | mean 110 s, p50 91 s, p90 190 s, max 449 s | mean 114 s, p50 117 s, p90 208 s |
| First visible answer text (per turn) | p50 72 s, p90 184 s | = turn: p50 117 s |
| Cost per question | mean $0.42, p50 $0.31, p90 $0.86, max $1.91 | mean $0.37, p50 $0.39, max $0.95 |
| Cost split | Haiku readers $10.54 (14,491 calls, 47.8M in, 11.5M out of which 7.7M thinking); Sonnet $8.06 (393 calls, 5.1M in of which 2.8M cache reads) | gpt-5-mini $3.05; Sonnet 4.5 $2.12 |
| Searches per turn | mean 5.2, max 30 (parallel) | mean 2.3, max 8 (serial) |
| Per search | 8.3 queries, 83 retrieved → 57 unique → 17.2 kept; p50 32 s | 6.1 queries, 61 → 38 → 13.7; p50 29 s |
| Reader errors | 0 of 14,491 | 1 of 1,367 |
| `<final_answer>` tags | 46/47 (the other: no tags, whole reply shown, which is correct) | 14/15 |
| Streamed text == saved | 46/46 | n/a (no streaming) |
| Sources list where citations exist | 43/43 (2 answers need none: transliteration, out-of-KB) | 13/13 |
| Sequential from 1 / no duplicates | 43/43 / 42/43 | 12/13 / 11/13 |
| Exact sources-line format | 43/43 | 12/13 |
| Every cited document and title among docs the agent saw | 43/43 | 12/13 |
| Page ranges match the retrieved metadata | 35/43 | 12/13 |
| KB docs rewritten per question | 0 | 18.5 |
| LangSmith | 1 root trace per turn ("ChitrapurGPT turn") | 1 per turn |

The 8 page-range mismatches are all on knowledge-base entries whose page fields are really line numbers (e.g.
`start_page 857, end_page 1077` for a children's column). The model shortens the impossible range to one page.
That's a data issue in `knowledge_base.jsonl`, not a citation error.

**Blind pairwise judge** (`eval/judge.py`, claude-opus-5-5), on the 14 LLM questions both stacks answered. Each answer
is shown next to the passages its own stack retrieved, in random order.
- The new stack wins **13**, the old stack wins **1** (`real-02`: the new answer added a claim backed only by an uncited
  passage, plus the KB page-range issue above).
- Mean faithfulness 4.71 vs 3.86; mean quality 4.57 vs 3.43. Judge cost $1.55.
- One verdict (`edge-out-of-kb`) was re-run after fixing the judge's evidence text. It had told the judge "no search
  was made" for a search that kept nothing, which made a truthful "I searched and found nothing" look fabricated.

**Supervisor effort.** Fixed by Vaibhav at `xhigh` (Sonnet). Earlier same-retrieval data points, all superseded:
- `low` vs `medium` on 15 questions, fuzzy-only: judge 8–7, faithfulness 4.33 vs 4.47, quality 4.20 vs 4.00.
  `low` was 13% faster and 19% cheaper and grounded better (100% vs 80%).
- An Opus 5.5 supervisor (`high` by the model-family rule) was not benchmarked, by Vaibhav's choice.

**E2E** (`tests/e2e/run_e2e.py`, real retrieval, the shipped configuration): 12/12 checks pass:
- streaming, saved == displayed (socket and database), persisted step names and nesting;
- PDF and image uploads used in the answer;
- Exact/Fuzzy Search unchanged, edit truncation, resume with memory, `/admin` renders the thread.

### 13.2 Baseline numbers (old stack, 2026-10-27)

`eval/report.py eval/results/baseline.local.jsonl`.

**Coverage is partial.** The OpenAI account ran out of credit during the run (`429 insufficient_quota`, from both the
reader and the embedding endpoints). The baseline therefore has:
- 4 real questions (`real-01`…`real-04`);
- 10 LLM edge cases: translate, transliterate, out-of-KB, in-domain unanswerable, multi-part, 2 spelling variants,
  follow-up, Hindi, Kannada;
- the Fuzzy Search command case.

Not captured: `edge-resume`, `edge-edit`, `real-05`…`real-30`, and the two upload cases (skipped on purpose: the
current stack ignores files). Phase B didn't extend it: the old stack's readers need OpenAI chat, the OpenAI key was
removed, and the $40 budget went to the new-stack run, which at `xhigh` costs more than planned. The sample leans towards heavy edge questions, so per-question cost and latency are
higher than a typical real question. The four real ones averaged $0.34 and 101 s, with one 8-search outlier.

| Metric | Value |
|---|---|
| Latency per question | mean 114 s, p50 117 s, p90 208 s, max 247 s |
| Time to first visible answer text | = turn latency (no streaming): p50 117 s, p90 185 s |
| Cost per question | mean $0.369, p50 $0.391, p90 $0.595, max $0.953 |
| Cost split | gpt-5-mini readers $3.05 (1,366 calls; 2.48M in, 1.22M out of which 0.90M reasoning); Sonnet 4.5 $2.12 (85 calls; 580k in, 25k out) |
| Searches per LLM turn | mean 2.3, max 8; 2 turns with none (transliteration, out-of-KB) |
| Per search | 6.1 queries; 61 retrieved → 38 unique → 13.7 kept; latency p50 29 s / p90 48 s; retrieval p50 6.8 s; reading p50 19.6 s |
| Tool output size | p50 16k chars, max 104k; 2 of 35 over 80k chars |
| Reader errors | 1 of 1,367 (absorbed by the retry) |
| `<final_answer>` tags | 14/15 (the out-of-KB answer had none; whole reply shown) |
| Citations (13 answers with sources) | sources present 13/13; sequential from 1: 12/13; no duplicates: 11/13; format: 12/13; inline refs ⊆ sources: 13/13; every source cited inline: 12/13; every source's document, title and pages among the docs the agent saw: 12/13; 6.6 sources per answer |
| Same article kept twice in one search | 20 (fuzzy vs Pinecone copies, PARITY 4.5) |
| KB docs rewritten per question (PARITY 4.9) | 18.5 |
| LangSmith | 1 root trace per turn |

Answers by hand:
- **Out-of-KB:** declined correctly.
- **In-domain unanswerable:** said the figures weren't found and reported what it did find, with citations.
- **Follow-up:** used the first turn's context.
- **Hindi/Kannada:** answered in the question's script with citations; the source-list format deviated (PARITY 1.9).
- **Transliteration:** done without a search.

## 14. Vector search: Pinecone → Postgres + pgvector (Phase C)

**Why:**
- Pinecone's free plan caps data read at 1 GB a month. The eval runs used it up, so every query returned 429,
  including on the live site.
- The new stack runs about 2.4× as many vector queries per question, so the plan was too small anyway.
- Vaibhav decided to move the index into the Postgres the app already uses (`DATABASE_URL`; Render Postgres 16 with
  pgvector 0.8.0) and drop Pinecone.

**What replaced what:**

| | Pinecone (before) | pgvector (now) |
|---|---|---|
| Store | `pinecone_vectorstore_wrapper.PineconeVectorStore` | `vector_store.PgVectorStore` (a LangChain `VectorStore`, sync, psycopg 3 pool) |
| Data | index `chitrapur-gpt`, dotproduct, 3072 dims | table `kb_article(id text pk, content text, metadata jsonb, embedding halfvec(3072))` |
| Index | managed | HNSW `halfvec_ip_ops`, `m = 32, ef_construction = 200` (46 MB; ~10 s to build locally) |
| Query | `top_k = 15` with values, then local MMR | `ORDER BY embedding <#> q LIMIT 15` with the vectors, then **the same MMR function** (moved over unchanged) |
| Retriever | `as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})` | unchanged |
| Metadata | Pinecone's return: None → "Unknown", non-scalars → str, ints → floats | stored exactly that way (`pinecone_style_metadata`), so `format_docs` output is byte-identical |
| Ids | `<source>-<title>` | same |
| Ingest | `add_to_vector_store` lists all ids, upserts new docs | same behaviour via `existing_ids()` + upsert |
| Embeddings | `text-embedding-3-large` | same model; all 5,924 articles re-embedded through the same LangChain path |
| Config | `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | `VECTOR_DATABASE_URL` or else `DATABASE_URL` (production: nothing new), `CGPT_VECTOR_EF_SEARCH` (default 400) |

Design points:
- **`halfvec`:** pgvector indexes `vector` only up to 2,000 dimensions; `halfvec` (16-bit floats) indexes up to 4,000.
  MMR runs on the half-precision vectors, against the full-precision query embedding as before.
- **Search settings per connection:** the pool's configure callback runs `SET hnsw.ef_search = 400` and
  `SET enable_seqscan = off` on the store's connections only. At ef_search 400 the planner otherwise chooses an
  exact sequential scan: its cost model ignores detoasting the halfvec column. That scan takes ~430 ms locally,
  worse on Render's small database, and competes with chat traffic. A test asserts the plan is
  `Index Scan using kb_article_embedding_hnsw`.
- **Single source of truth:** `vector_store.DDL`, `pinecone_style_metadata()` and `doc_id()` are imported by
  `scripts/load_vectors.py`, which builds and loads the table (idempotently, from a saved embedding matrix or by
  re-embedding).
- **Failing loudly:** with neither `VECTOR_DATABASE_URL` nor `DATABASE_URL`, the store raises at import. The eval
  and E2E harnesses (which never get `DATABASE_URL`) refuse to run unless `VECTOR_DATABASE_URL` points at a local
  database.

**Parity** (measured by Hermes on a local copy):
- Replaying 2,333 recorded Pinecone searches, 82–85% return the identical top 5. Every difference is a near-tie inside
  the top 30 (median similarity gap 0.005). The noise floor between two Pinecone-era embedding runs is 98–99%, so
  the gap is small model-serving drift. Vaibhav accepted this.
- HNSW at ef_search 400 against exact float32 search (300 recorded queries): recall@15 0.999, and 99% give the
  identical final MMR top 5, at p50 9 ms. ef_search 100 gives 89% and 200 gives 96%.
- Tests: `tests/test_vector_store.py`:
  - MMR picks equal the moved function's;
  - `Document` shape;
  - Pinecone-style metadata through `format_docs`;
  - ef_search applied, sequential scans off on store connections only, HNSW in the plan;
  - upsert idempotence, and ingest skipping existing ids;
  - a vector hit from the local copy formats byte-identically to a recorded Pinecone-era tool output (an article
    from the knowledge base, `tests/fixtures/pinecone_vector_hit.txt`).
- The E2E ran with real retrieval against the local copy.

**Not changed:** the fuzzy half, dedupe, readers, `format_docs`, and the agent. The benchmark on the new index is
Hermes's to run.
