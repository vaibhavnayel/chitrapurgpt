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

Vector search goes to `VECTOR_DATABASE_URL`, a local copy of the `kb_article` index (see the main README). The harness
exits if it's unset or isn't on localhost, rather than silently running without vector search. Each record notes
which copy it searched (`stack.vector_db`) and the `hnsw.ef_search` used.

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

Run with `--concurrency 1` (the default) when latency matters. The old stack's retrieval blocks the event loop,
so concurrent questions distort each other's timings.

**Measuring the old stack.** The pre-Deep-Agents code and its exact dependency set can be run by the same harness
from the same venv:
```sh
git archive 7a090ba | tar -x -C .cache/oldstack && echo 7a090ba > .cache/oldstack/COMMIT
uv pip install -p .venv --target .cache/olddeps -r <old freeze>     # old package versions, outside the venv
.venv/bin/python eval/run_eval.py --label baseline --app-dir .cache/oldstack --site .cache/olddeps ...
```

**Blind pairwise judge.** `judge.py A B --out ...` shows `claude-opus-5-5` each question with both answers in
random order, each next to the passages its own stack retrieved (cited ones first, 24k chars per side), and asks
for faithfulness and quality scores (1-5) and a winner. Its cost goes to the spend ledger.

`check_openai.py` makes one one-word embeddings call to check the embeddings endpoint can serve requests (it
honours `OPENAI_BASE_URL`).

`--modes answer=quick` sends the chat composer's Quick answer mode with every message (the default is Detailed,
as in the app). Records note the modes sent (`modes_sent`, `stack.modes`) and the mode each answer was saved with
(`turns[].answer_mode`).

`--no-vector` runs with fuzzy retrieval only, for when the embeddings endpoint is unreachable. Such runs are marked
`no-vector-retrieval` in `stack.eval_shims` and aren't comparable with full runs.

Superseded runs are marked in the spend ledger with a `note` (`report.py --spend` shows it).

## Environment shims (sandbox only)

`sandbox_shims.py` (used by `run_eval.py` and the E2E server) applies a shim when the environment needs it; it's
listed in each record's `stack.eval_shims`:
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
