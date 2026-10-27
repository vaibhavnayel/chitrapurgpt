# Eval harness

Measures the chat stack end to end on a fixed question set: answers, citations, tool calls and search queries,
retrieved and kept documents, Chainlit steps, latency, tokens and cost. It was built to compare the current
hand-written agent with the Deep Agents rebuild (`docs/AGENT_DESIGN.md`), and works unchanged for both because it
drives `app.py`, not the stack's internals.

## How it works

- `run_eval.py` imports `app.py` and, for each question, calls its Chainlit handlers (`on_chat_start`,
  `on_message`, `on_chat_resume`) inside a Chainlit HTTP context. That's the production code path minus the
  websocket. Nothing in the app is patched.
- `instrument.RecordingEmitter` replaces Chainlit's no-op emitter. It records every step, message, streamed token and
  element the UI would receive. That gives step names and nesting, the displayed answer, time to first visible text,
  and whether the streamed text equals the saved message.
- `instrument.Collector` is a LangChain callback handler, attached to every LangChain/LangGraph run through
  langchain-core's configure hook. It records LLM calls (model, tokens incl. cache and reasoning, cost via
  `pricing.py`, latency), tool calls, retriever queries and documents. Runs are attributed to the enclosing tool even
  when `@traceable` breaks LangChain's parent chain.
- Multi-turn questions support `resume_before` (rebuild the thread the way Chainlit's data layer would and call
  `on_chat_resume`), `edit_of` (do what Chainlit's `edit_message` handler does), `command` (Exact/Fuzzy Search) and
  `files` (attachments, built the way Chainlit builds uploaded elements).
- The current stack rewrites the shared fuzzy-retriever documents (see `docs/PARITY.md` 4.9). The harness restores
  them before each question, so every question sees a freshly started process, and records how many a question
  changed (`kb_docs_mutated`).
- Every question's cost goes to `results/spend_ledger.local.jsonl`, so the API spend across all runs is
  `report.py --spend`.

## Privacy

Real user questions and all outputs stay in git-excluded `*.local.*` files:
- `real_questions.local.jsonl`, `questions.local.jsonl` and `results/*.local.jsonl` are never committed.
- `questions/real_selection.json` holds only 12-hex-char hashes of the chosen questions plus category labels.
- `questions/edge_cases.jsonl` holds questions written for the eval.

The harness unsets `DATABASE_URL`/`BUCKET_NAME` so it can never touch a real database. It sets placeholder OAuth
variables because Chainlit refuses to register `@cl.oauth_callback` without them; no one signs in.

## Running

```sh
.venv/bin/python eval/build_questions.py              # -> eval/questions.local.jsonl (30 real + 15 edge)
.venv/bin/python -m pytest eval/test_instrument.py -q # offline checks of the instrumentation
.venv/bin/python eval/run_eval.py --label baseline --out eval/results/baseline.local.jsonl \
    --skip-requires uploads --budget-usd 9.5          # options: --ids a,b --source real|edge --limit N --exclude-done
.venv/bin/python eval/report.py eval/results/baseline.local.jsonl --per-question
.venv/bin/python eval/report.py eval/results/baseline.local.jsonl eval/results/new.local.jsonl   # compare
.venv/bin/python eval/report.py --spend
.venv/bin/python eval/langsmith_traces.py --label baseline   # trace shape in the dev LangSmith project
```

Run with `--concurrency 1` (the default) when latency matters. The current stack's retrieval blocks the event loop,
so concurrent questions distort each other's timings.

## Environment shims (sandbox only)

`run_eval.py` applies two shims when the environment needs them, and lists them in each record's
`stack.eval_shims`:
- `pinecone-env-proxy`: the Pinecone SDK ignores `HTTPS_PROXY`; where egress must go through a proxy, the harness
  passes it to `Pinecone(...)` explicitly.
- `embeddings-no-local-tiktoken`: where tiktoken can't download its vocabulary, `OpenAIEmbeddings` sends query text
  instead of locally tokenized ids. The API tokenizes with the same vocabulary, so short queries embed the same.

## Metrics (`report.py`)

- **Latency:** per question and per turn, plus time to the first visible answer text.
- **Cost:** per question and per model (tokens, cache reads, reasoning tokens).
- **Search behaviour:** searches per turn; per search: queries, docs retrieved, unique docs, docs kept after
  reading, duplicate articles kept, and retrieval and reading wall time; reader calls and errors.
- **Answer extraction:** `<final_answer>` tag compliance; streamed text == saved message.
- **Citations** (turns that searched):
  - a sources block is present;
  - indices sequential from 1, with no duplicate sources;
  - every sources line matches `[n] <title>, <document> page <start>-<end>`;
  - inline references ⊆ sources, and every source cited inline;
  - every source's document, title and pages appear among the docs the agent actually saw (grounding).
- Edge cases (out-of-KB refusals, translations, follow-ups, resume, edit) are also read by hand.
