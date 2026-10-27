# Parity inventory: current LLM stack → Deep Agents

Everything the current stack does that the rebuild must keep (or deliberately change). It was built from the brief's
list, then checked against the code (`app.py`, `steps.py`, `retrievers.py`, `pinecone_vectorstore_wrapper.py`,
`search_engine.py`, `ingest.py`, `admin.py`), Chainlit 2.9.6's source, and an instrumented run of the current
stack (`eval/`, baseline in `eval/results/baseline.local.jsonl`).

**Status column**
- **keep**: strict parity.
- **upgrade**: one of the approved upgrades (streaming, parallel searches, uploads).
- **fix**: a bug in the current code that the rebuild should fix. Each one is flagged in the "Decisions" section at
  the end so Vaibhav can veto it.

**How it's tested**
- **eval**: the harness in `eval/` (`run_eval.py` + `report.py`), run against the baseline and the new stack.
- **unit**: an offline pytest (no network, fake models), in `tests/test_agent.py`, `tests/test_uploads.py` and
  `tests/test_ingest.py`.
- **existing**: the tests already in `tests/`.
- **manual**: on the Render PR preview.

Test status before any change (Phase A): `pytest tests/` passes (7 tests), `tests/test_preview_idle.py` and
`tests/test_preview_oauth.py` pass. `tests/test_admin.py` couldn't run in the Phase A sandbox (no Postgres); in
Phase B it runs against a local Postgres and passes, before and after the change.

## Phase B: where each item is tested now

Approved: fixes 1, 2, 3, 4, 5, 7, 8 and 9; kept as before: 6 (dedupe key) and 10 (full article text in the
"document analysis" step).

| Items | Covered by |
|---|---|
| 1 system prompt, 12 date | `tests/test_agent.py::test_one_tool_our_prompt_todays_date_and_steps`: the model sees exactly `steps.system_prompt(today)` and the date changes between turns on different days. Eval: citation checks. |
| 2 final answer, streaming | `test_extractor_streams_exactly_what_the_regex_shows`: 7 cases × 30 random chunkings each, including split tags, missing tags, a tag mentioned in the reasoning, two answers and an unclosed tag. Also `test_answer_stream_saved_equals_displayed`, `test_thinking_blocks_never_shown`. E2E: "displayed == final message", "saved message == displayed". Eval: `streamed_equals_saved`, time to first visible text. |
| 3.1, 3.2 memory (answer stored once) | `test_one_tool_our_prompt_todays_date_and_steps` (history = human, ai, tool, tool, ai, human, ai; tool results reach the next turn) |
| 3.3, 3.4 resume | `test_history_from_thread_skips_commands_errors_and_notes`, `test_resume_then_continue`; E2E "thread resumed", "agent remembers after resume" |
| 3.6 stuck chat | `test_mid_turn_failure_keeps_the_chat_usable` (the model fails mid tool loop; the half-streamed answer is removed, history keeps only finished turns, the next turn works) |
| 4 search tool | `test_search_tool_output_and_shared_docs_untouched` (output byte-identical to `retrievers.format_docs`; shared docs untouched), `test_reader_failure_skips_the_doc`, `test_global_reader_cap_across_parallel_searches`; eval per-search metrics and `kb_docs_mutated` |
| 5 steps, `/admin` | `test_one_tool_our_prompt_todays_date_and_steps` (two parallel searches → two engine steps, each with "search query generator" then "document analysis"); E2E "steps persisted with names and nesting", "/admin renders the new-stack thread"; `tests/test_admin.py` |
| 6 LangSmith | `eval/langsmith_traces.py --label <run>`: one root per turn, "ChitrapurGPT turn" |
| 7 commands | `test_commands_unchanged_and_marked`; E2E "Exact Search unchanged", "Fuzzy Search unchanged"; `tests/test_search_engine*.py` |
| 8 gating | `tests/test_admin.py` (unchanged; its `steps` stub still works because `app.py` uses `steps.x` at call time) |
| 9 edit | `test_edit_truncates_history`; E2E "edit truncates history" |
| 10 reader failures | `test_reader_failure_skips_the_doc` |
| 11 sampling | `models.py`; eval comparison of effort `low` vs `medium` (see `docs/AGENT_DESIGN.md` §13) |
| 13 ingest | `tests/test_ingest.py` (offline, temp dir, upsert stubbed), `tests/ingest_live_check.py` |
| 16 uploads | `tests/test_uploads.py` (each type, oversize, unsupported, damaged, missing, caps); E2E "PDF upload used", "image upload used"; eval `edge-upload-*` |
| safety cap (new) | `test_model_call_cap_gives_a_friendly_answer` |
| query writer fallback (new) | `test_query_writer_failure_falls_back_to_the_question` |
| sources after `</final_answer>` (new) | `test_sources_written_after_the_closing_tag_are_kept` |
| model settings | `tests/test_models.py` (effort by model family, reader thinking, overrides, older models keep temperature 0) |

Changes beyond the PARITY decisions, all found by measuring the new stack and listed in `eval/REPORT.local.md`:
- **1.9:** one prompt sentence restating the sources line format (Sonnet 5.5 added a comma before "page").
- **2.1:** if the extracted final answer has no sources list and one follows `</final_answer>` directly, it is kept
  (Sonnet 5.5 at `xhigh` did this in 3 of 47 turns).
- **10.3:** the query writer retries and then falls back to searching with the question itself (Sonnet 5.5 once
  returned truncated structured output there).
- **11:** thinking effort by model family (Sonnet `xhigh`, Opus `high`); readers think at `xhigh`. Vaibhav's choice.
- **13.1:** `ArticleBoundaries.scratchpad` renamed `notes` (Sonnet 5.5 refuses schemas with a `scratchpad` field).

Outcomes on the full eval and the E2E are in `docs/AGENT_DESIGN.md` §13.

---

## 1. System prompt (`app.py: system_prompt`)

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 1.1 | Role: research assistant for the Chitrapur Saraswat community that "thoroughly studies magazine articles in the knowledge base". | keep | unit: prompt contains the clause |
| 1.2 | Tool use: may call `search_knowledge_base` several times. | keep (the prompt also allows parallel calls) | eval: search calls per turn |
| 1.3 | Translations and transliterations are allowed. | keep | eval: `edge-translate`, `edge-transliterate`, `real` transliteration question |
| 1.4 | Wikipedia-style citations; no duplicated sources; indices sequential from 1. | keep | eval: `report.py` citation checks (sequential, duplicates, inline ⊆ sources) |
| 1.5 | "Think step by step and draft your answer" first, then `<final_answer>…</final_answer>` ("don't forget to close the tags"). | keep (see §2 and §11) | eval: `final_answer_tags` rate |
| 1.6 | Answer only from the knowledge base; don't use own knowledge. | keep | eval: `edge-out-of-kb`, `edge-in-domain-unanswerable` (judged by reading the answers) |
| 1.7 | "Today's date is …". Computed once **at import**, so it goes stale after a deploy. | **fix**: compute per turn (§12) | unit: date in the prompt equals today |
| 1.8 | Community facts: the main holy site is the Chitrapur Math; the religious leader is called Swamiji. | keep | unit |
| 1.9 | Citation example: the `sources:` block format `[n] <title>, <document> page <start>-<end>`, inside the final answer. Baseline compliance (13 answers with sources): sequential 12/13, no duplicates 11/13, format 12/13, every source's document/title/pages found among the docs the agent saw 12/13. The misses are the Hindi answer (sources block translated: "पृष्ठ", `.txt` dropped, 29 sources with repeats), the Kannada answer (indices `1,2,3,6,10,12,17,20,0`), and one English answer listing the same article as [2], [5] and [7]. | keep (same prompt, so the same compliance or better) | eval: `format_ok` (regex on every sources line), grounding of title/document/pages in the docs the agent saw |
| 1.10 | New: may use uploaded files (translate/explain them), but community facts still come from the knowledge base with citations. | upgrade | eval: `edge-upload-*`; manual |

## 2. Final-answer extraction and display (`steps.parse_final_answer`, `app.on_message`)

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 2.1 | Show only the text inside the **first** `<final_answer>…</final_answer>` (non-greedy regex, DOTALL). Surrounding newlines are kept (answers start with `\n`). | keep | unit: streaming extractor gives the same output as the regex on recorded raw outputs from the baseline |
| 2.2 | Tags missing → show the whole reply. Seen in the baseline: the out-of-KB question was answered with no tags and no search (1/15 answers), and the whole reply was shown. | keep | unit; eval |
| 2.3 | Opening tag present but no closing tag → today the regex fails, so the **whole reply** (reasoning included) is shown. | keep the fallback; the streaming version can't know up front, so it streams from `<final_answer>` and, if the turn ends unclosed, replaces the message with the whole reply (same final result). | unit |
| 2.4 | The saved message (`assistant_message` step output) equals the displayed text. | keep, and must still hold with streaming | eval: `streamed_equals_saved`; unit |
| 2.5 | The message appears all at once at the end of the turn (no streaming). Baseline time-to-first-visible-text = full turn latency. | **upgrade**: stream only the final-answer text as it's written | eval: `time_to_visible_answer_s` |
| 2.6 | The extracted text also **replaces** the final AIMessage's content in session history, so later turns see only the final answer of earlier turns, not their drafts. | keep | unit |
| 2.7 | Author "Assistant" (`config.ui.name`), type `assistant_message`, no tags. | keep | eval: `ui_steps` |

## 3. Conversation memory

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 3.1 | `cl.user_session["messages"]` holds `[SystemMessage, Human, AI(tool_calls), Tool, …, AI(final)]`. Later turns see earlier tool calls and **full tool results** (the formatted docs). | keep | eval: `edge-followup`, `history_after` |
| 3.2 | **Bug:** the final AIMessage is appended twice (`respond_to_user_message` appends it, then `on_message` appends the returned object again). Every later turn sends the previous answer twice. Confirmed in the baseline: in `edge-followup` the history ends `…, ai(611), ai(611), human, …, ai(3503), ai(3503)`. | **fix** | eval: `history_after` |
| 3.3 | Resume (`on_chat_resume`): history = fresh system prompt + every `user_message` / `assistant_message` step's `output`, in order. No tool calls or tool results. | keep | eval: `edge-resume` (not captured on the baseline because the OpenAI credit ran out; behaviour here is from the code); unit for the history builder |
| 3.4 | **Bug on resume:** Chainlit's data layer persists neither `tags` nor `command` for steps (the `INSERT INTO "Step"` has neither column). So on resume, Exact/Fuzzy Search requests and their output (often 100k+ chars; 262 matches = 105k chars in one eval run) come back as ordinary Human/AI turns and are sent to the LLM. Error messages (`ErrorMessage` is `assistant_message` with `isError`) come back as AI turns too. In a live session neither is in history. | **fix**: on resume skip `isError` steps, and skip command pairs. Detect them by the `\n{i}. {title} (Source: …)\nSnippet:` output shape, or better, mark command output in step `metadata`, which *is* persisted. | unit |
| 3.5 | A user message carrying only files (no text) becomes an empty HumanMessage on resume. | keep, with a note (see §16 uploads) | unit |
| 3.6 | **Bug:** if anything raises mid-turn (e.g. a reader failure, §10), history keeps an AIMessage with `tool_use` but no `tool_result`. Every later turn in that chat then fails with an API 400, and the chat is stuck. | **fix** (deepagents `PatchToolCallsMiddleware`, or only committing a turn's messages to the session when the turn finishes) | unit: inject a tool exception, then send another message |

## 4. The search tool (`steps.search_knowledge_base` and helpers)

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 4.1 | Tool name `search_knowledge_base`, one arg `research_query: str`, description "Search the knowledge base to get relevant magazine articles. The query should be a question that you want to answer." | keep | unit: tool schema snapshot |
| 4.2 | Query writer: Sonnet 4.5 (`temperature=0`, no `max_tokens`, so langchain-anthropic's default 64,000), structured output `Queries{reasoning, queries}`, system prompt verbatim ("…parse it into a list of queries… short and concise… exact match, fuzzy match and vector search"), user message `here is the question: {q}`. No retry. Baseline: ~4 queries per search. | keep (model → `CGPT_QUERY_MODEL`, default the main model) | eval: queries per search; unit: prompt snapshot |
| 4.3 | Retrieval per query: `HybridRetriever` = `FuzzyMatchRetriever(k=5)` results **then** Pinecone MMR (`k=5, fetch_k=15, lambda_mult=0.5`, MMR computed locally with cosine similarity in `pinecone_vectorstore_wrapper.py`), index `chitrapur-gpt`, namespace `""`, `text-embedding-3-large`. Fuzzy scoring: content (partial_ratio + token overlap + double-metaphone token_set_ratio)/3 × 0.9 + metadata (title/summary/description fuzzy or phonetic max) × 0.1, threshold 30, sorted by (phonetic, token overlap, ratio). | keep exactly (same objects reused) | unit: same docs for fixed queries on the fuzzy side; vector side spot-checked in eval |
| 4.4 | `retriever.batch(queries)` is **synchronous** inside the async tool, so it blocks the event loop (every user's session) for the whole retrieval: baseline p50 ≈ 6 s per search. | **fix**: `await asyncio.to_thread(...)` or `abatch`, same results | eval: retrieval wall time |
| 4.5 | Dedupe on `(page_content, str(metadata))`, keeping first-seen order (fuzzy hits of query 1, vector hits of query 1, fuzzy hits of query 2, …). The fuzzy copy (JSONL: int pages, `author: None`) and the Pinecone copy (float pages `3.0`, `author: "Unknown"`) of the **same article** never compare equal, so both can be read and both shown. | keep the key (changing it changes what the agent sees); flagged. Baseline: 61 docs retrieved → 38 unique per search; 20 cases of the same article kept twice in one search (35 searches); 132 of 479 kept docs were Pinecone copies (float pages). | eval: unique vs retrieved per search; duplicate articles kept |
| 4.6 | Reader: one call per doc, prompt verbatim (query interpolated into the system prompt; doc as `format_docs([doc])` in the user message), structured output `RelevantPassages{reasoning, passages_in_context}`, gpt-5-mini, `max_tokens=8000`, `with_retry()` (3 attempts, exponential jitter), `abatch(max_concurrency=40)` per search. `temperature=0` is **silently dropped** by langchain-openai for gpt-5 models (they run at temperature 1 with medium reasoning; baseline ≈ 80% of reader output tokens are reasoning). | keep prompt/schema/filter; model → `CGPT_READER_MODEL` (Haiku 5.5); the concurrency cap becomes **global** across parallel searches | eval: reader calls, kept per search, cost; unit: prompt snapshot |
| 4.7 | Filter: keep a doc iff the result is not None, `passages_in_context` is non-empty, and the **first** passage doesn't contain `"no relevant information"` (case-sensitive substring). Baseline: 0 of 479 kept docs had a first passage saying "no relevant information" in any case, so gpt-5-mini always used the exact lowercase phrase. | keep exactly; watch whether Haiku keeps to the phrase | unit (fake reader outputs incl. "No relevant information found" capitalised, which today is *kept*); eval |
| 4.8 | Kept docs get `page_content = "Here are the relevant passages from this document: " + "\n".join(passages)`. | keep | unit |
| 4.9 | **Bug:** 4.8 writes into the `Document` objects that `FuzzyMatchRetriever` holds for the whole process. After a search, those articles' text **is** the extract written for someone else's question. A later search (any user) that fuzzy-matches the article sends the reader the old extract instead of the article, and the dedupe in 4.5 then also fails between the fuzzy and vector copies. Matching itself still uses the cached original text. Baseline: 18.5 docs changed per question on average (the harness restores them between questions so each question sees a fresh process). In production this accumulates for the life of the process. | **fix**: copy docs before rewriting | eval: `kb_docs_mutated`; unit |
| 4.10 | Size: baseline tool output p50 16k chars, max 104k; 2 of 35 searches were over 80k chars (relevant to deepagents' default tool-output eviction, see AGENT_DESIGN §3). Text the agent sees (`format_docs`): for each kept doc, `"{i}. {title}\n Metadata:\n{pformat(metadata)}\nContent:\n{page_content}\n" + "-"*100`, joined by `"\n"`, `i` from **0**. `pformat` sorts keys and wraps at 80 columns. Metadata includes `source` (e.g. `Sunbeam-2008-October.txt`), `published_date`, `title`, `author`, `summary`, `start_line`, `end_line`, `start_page`, `end_page`. Pinecone docs show floats (`start_page: 11.0`) and `author: 'Unknown'`. Citations are built from this text. | keep **byte-for-byte** | unit: new tool output == `retrievers.format_docs` on the same docs; eval: `kept_docs` parse, citation grounding |
| 4.11 | No docs survive → the tool returns `""`. The agent sees an empty tool result. | keep | unit |

## 5. Chainlit steps and the admin viewer

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 5.1 | Per search call: step **"knowledge base search engine"** (type `undefined`, top level), with children **"search query generator"** then **"document analysis"**. Inputs/outputs come from `@cl.step`. Engine: input `{"research_query": …}`, output = the formatted docs. Query generator: output = JSON list of queries. Document analysis: input = `str()` of the docs (full article text, 130–165k chars per search in the baseline) + query; output = `str()` of the kept docs. | keep names, types and nesting. Keep contents in the same shape (see Decisions on the 160k-char input). | eval: `ui_steps` tree signature per search; unit with the recording emitter |
| 5.2 | Parallel searches (upgrade) must each get their own "knowledge base search engine" step with its own two children. Today `@cl.step` finds the parent through a context var, which works per asyncio task. | upgrade | unit: two concurrent searches give two correctly-nested trees |
| 5.3 | `cot = "full"`: steps are visible in the UI and persisted (`Step` table, `parentId`). Old threads must look the same as new ones. | keep | manual: resume an old thread and a new one on the preview |
| 5.4 | `/admin` chat viewer shows **only** `user_message` and `assistant_message` steps (`admin.py`, SQL filter), ordered by `startTime, createdAt`. Tool steps never appear there. So for `/admin`, parity means one `assistant_message` per turn whose output is the displayed answer. | keep | existing: `tests/test_admin.py`; eval: `n_assistant_messages == 1` |
| 5.5 | Exceptions from `on_message` become an `ErrorMessage` (author "Error", the exception text) via Chainlit's `wrap_user_function`. | keep | unit |

## 6. LangSmith tracing

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 6.1 | `@traceable` on `respond_to_user_message`, `handle_tool_call`, `search_knowledge_base`, `generate_search_queries`, `contextualize_docs`. Enabled by `LANGCHAIN_TRACING=true` + `LANGCHAIN_PROJECT` (langsmith reads `LANGSMITH_*`/`LANGCHAIN_*` `TRACING_V2` or `TRACING`). | keep the env vars | eval: `eval/langsmith_traces.py` |
| 6.2 | Baseline: **one root trace per turn** (`respond_to_user_message`), with tool, query writer, retriever and reader runs nested (170–810 runs per trace). Measured on all baseline turns; a turn shows 2 roots only when the harness ran it twice. LangChainTracer still logs many `No indexed run ID` warnings during the reader fan-out (`with_retry` + `abatch` under `@traceable`), but the reader runs counted in the traces match the harness's counts. | keep one trace per turn; the new root is the LangGraph run (`ChitrapurGPT turn`) with model/tools nodes, and the warnings should go | eval: `langsmith_traces.py --label <run>` shows 1 root per turn |

## 7. Exact Search / Fuzzy Search commands

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 7.1 | `message.command in {"Fuzzy Search", "Exact Search"}` → `search_engine.search_knowledge_base(content, exact=…)` (reloads the 30 MB JSONL each call; boolean AND/OR/parentheses), sent as `cl.Message(content, tags=["command_output"])`. Not added to LLM history. Commands are registered on chat start and resume (`set_commands`). | keep unchanged (no edits to `search_engine.py`) | existing: `tests/test_search_engine*.py`; eval: `edge-command-fuzzy` checks the tag, no LLM calls, no history change |

## 8. Access gating and unrelated modules

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 8.1 | `still_allowed()` runs first in `on_chat_start`, `on_chat_resume` and `on_message`. If the user was removed: send `NO_ACCESS_MESSAGE` and stop. DB unreachable → allow. No user (no auth) → allow. | keep | existing: `tests/test_admin.py` (calls `app.still_allowed`) |
| 8.2 | `oauth_callback`, `access.py`, `admin.py`, `preview_oauth.py`, `preview_idle.py` are installed at import of `app.py`. | keep untouched | existing tests. **Note:** `tests/test_admin.py` stubs the `steps` module (`sys.modules["steps"]` with `respond_to_user_message`) so `app` imports without Pinecone. Phase B must keep `app.py` importable that way, or update the stub to whatever module builds the retrievers. That's a test-only change. |

## 9. Editing a message (`edit_message = true`)

Chainlit's `edit_message` handler (`chainlit/socket.py`):
1. Removes every message after the edited one from the UI and DB (`message.remove()`).
2. Updates the edited message's content.
3. Calls `on_message(orig_message)` with the **same message id**.

The app ignores all of this: `cl.user_session["messages"]` still holds the old question, the old answer and its tool
results, and the edited question is appended after them. The LLM therefore sees a conversation that the user no
longer sees. Also, the handler runs outside a task, so "stop" can't cancel it, and an exception in it isn't turned
into an ErrorMessage.

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 9.1 | Edited message re-runs with stale history. | **fix**: store the Chainlit message id on each HumanMessage. When `on_message` receives an id already in history, truncate history to just before it, then run. | eval: `edge-edit` (`history_after` shows the old Q/A; not captured on the baseline because the OpenAI credit ran out, so this is from the code); unit |

## 10. Retries and reader failures

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 10.1 | SDK-level retries: langchain-anthropic `max_retries=2`, openai SDK default 2. Readers add `with_retry()`: 3 attempts, retrying only the failed items of the batch. | keep (SDK retries + one retry layer on readers) | unit |
| 10.2 | **The brief assumes "a failed doc is skipped, not fatal". That's not what happens today.** If a reader still fails after its retries, `RunnableRetry._abatch` re-raises, so `abatch` raises, the tool raises, the agent loop raises, the user gets an ErrorMessage, and the chat is stuck (3.6). Only a `None` parse result is skipped. Seen for real during the baseline: when the OpenAI account ran out of credit mid-question (`real-05`), 32 of 72 reader calls failed and the whole turn failed instead of skipping those docs (`eval/results/baseline_failed_openai_quota.local.jsonl`). In normal operation the baseline had 1 transient reader error in 1,367 calls, absorbed by the retry. | **fix to the intended behaviour**: a failed doc is logged and skipped | unit: a fake reader that raises for one doc |
| 10.3 | The query writer and the main agent have no retry beyond the SDK's. | keep | n/a |

## 11. Sampling, thinking and token limits

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 11.1 | Main agent: Sonnet 4.5, `temperature=0`, `max_tokens=8000`, no extended thinking. Reasoning happens in visible text before `<final_answer>`. | **changed** (forced): langchain-anthropic 1.7.6 raises `ValueError` for `claude-sonnet-5-5` with any `temperature` ≠ 1, any `top_k`, `top_p` ≠ 1, or `thinking` `disabled`/`enabled`. Sonnet 5.5 always uses adaptive thinking (models API: `thinking.types.disabled.supported = false`). Plan: no temperature, `effort` set explicitly, `max_tokens` raised so thinking doesn't eat the answer budget (see AGENT_DESIGN). The prompt keeps "draft, then `<final_answer>`", so the visible output keeps the same shape whether the model drafts in thinking or in text. | eval: `final_answer_tags` rate, answer length, citation metrics vs baseline |
| 11.2 | Readers: gpt-5-mini already ignores `temperature=0` (see 4.6). | Haiku 5.5: leave temperature unset, thinking off (supported on Haiku 5.5), `max_tokens=8000` | eval |
| 11.3 | No cap on agent iterations (`while tool_calls`). | keep "no small cap", but add a generous safety limit (e.g. 12 model calls) so a loop can't run forever | unit |

## 12. Today's date

Computed at import (`datetime.now()` in the module-level f-string), so it is the deploy date, not today. **Fix**:
build the system prompt per turn (and on resume). Flagged.

## 13. `ingest.py`

| # | Behaviour | Status | Tested |
|---|---|---|---|
| 13.1 | Article-boundary extraction uses `claude-3-5-sonnet-latest`, which is **retired** (not in the models list). Structured output `ArticleBoundaries{scratchpad, article_boundaries[ArticleBoundary]}` with `include_raw=True`, `temperature=0`, `max_tokens=8000`, `with_retry()`. | **fix**: model from `CGPT_INGEST_MODEL` (default `claude-sonnet-5-5`); same schema and prompt. Sonnet 5.5 rules apply: no temperature, and forced tool choice is rejected, so use native structured output (`method="json_schema"`). Raise `max_tokens`: a 100-article issue needs more than 8k output tokens with thinking. | unit: run `extract_article_boundaries` on a short sample text to a temp dir with `add_to_vector_store` stubbed |
| 13.2 | `print_cost` uses `get_openai_callback()` (counts only OpenAI calls, so it prints zeros for Claude) and hard-coded $3/$15 prices. | fix the cost print using `usage_metadata` (cosmetic) | unit |
| 13.3 | `main()` only runs the "move articles to KB + upsert" half; the extraction half is commented out. Not an agent. | keep as is | n/a |

## 14. Existing tests

See "Current test status" at the top. Phase B must keep all of them passing (`test_admin.py` wherever Postgres can run).

## 15. Other behaviour found while reading the code

| # | Behaviour | Status |
|---|---|---|
| 15.1 | `steps.py` imports `langchain_google_genai` and `ChatGoogleGenerativeAI` but doesn't use them; `rank_bm25` and `langchain-groq` are in requirements but unused. deepagents hard-depends on `langchain-google-genai>=4.4.2` anyway. | Remove the unused imports in the files we touch. Leave requirements pins we don't need to move. |
| 15.2 | Retrievers and Pinecone are built at import of `steps.py` (fuzzy index over 5,924 docs, ~15 s at startup). | keep (build once at import) |
| 15.3 | The final answer often starts with `\n` (the text right after `<final_answer>`). | keep (2.1) |
| 15.4 | Structured-output calls inside the tool show up in LangSmith under nested `RunnableSequence`s; readers are 60–90 LLM runs per turn. | fine |
| 15.5 | `chainlit.md` (the readme shown in the UI) mentions a `/Keyword Search` command that doesn't exist. | out of scope, mention only |
| 15.6 | Chainlit stores uploads under `.files/<session>/` and deletes them when the session ends. Without a storage client (`BUCKET_NAME`/S3/GCS/Azure env vars) the `Element` row is saved without the file, so files can't be re-read on resume. | design decision for uploads (AGENT_DESIGN §Uploads) |
| 15.7 | `cl.Message(content=search_results)` for a command with no matches sends an empty message. | keep |

## 16. Uploaded files (upgrade; current behaviour for reference)

Today `on_message` ignores `message.elements`: the agent never sees uploads, and the eval's upload questions are
skipped on the baseline for that reason. New behaviour, tested by `edge-upload-pdf` and `edge-upload-image-csv` plus
unit tests per type, is specified in `docs/AGENT_DESIGN.md`.

---

## Decisions to confirm (bug fixes beyond strict parity)

1. **Duplicate final AIMessage in history (3.2)**: fix.
2. **Commands and error messages replayed into LLM history on resume (3.4)**: fix by skipping them. Command output
   gets marked in step metadata so it can be recognised later; old threads are detected by output shape.
3. **Chat stuck after a mid-turn exception (3.6)**: fix.
4. **Sync retrieval blocking the event loop (4.4)**: fix (same results, off the loop).
5. **Shared-document mutation (4.9)**: fix. This changes what later searches see, to what was always intended.
6. **Dedupe key (4.5)**: *keep* as is for parity, though it lets the fuzzy and Pinecone copies of an article both
   through. Changing it is a separate quality decision.
7. **Edit message (9.1)**: fix by truncating history at the edited message.
8. **Reader failure skips the doc (10.2)**: implement what the brief describes (it isn't today's behaviour).
9. **Date per turn (12)**: fix.
10. **"document analysis" step input (5.1)**: keep storing the full article text (~160k chars per search) in the
    `Step` table for parity, or store a short summary (titles + sources) to save DB space. Recommendation: keep it for
    now, since it's what admins debugging a thread would expect to see.
