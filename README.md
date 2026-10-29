# Chitrapur GPT

A conversational AI assistant for the Chitrapur Saraswat community. It answers questions by researching a knowledge
base of ~5,900 magazine articles (Chitrapur Sunbeam), with citations. Built on Chainlit and LangChain Deep Agents.

## Features

- Chat interface powered by Chainlit, with Google sign-in and an allow-list managed at `/admin`
- An agent that searches the knowledge base (several searches at once when useful), reads each retrieved article
  for the question, and answers with Wikipedia-style citations; the answer streams as it is written
- A Detailed / Quick answer picker in the chat composer: Quick gives a shorter answer sooner, from fewer searches
- Hybrid search: fuzzy and phonetic matching over the articles plus vector search in Postgres (pgvector)
- Translation and transliteration, and reading files you attach (images, PDFs, Word, Excel, CSV, text)
- Exact Search and Fuzzy Search commands for boolean keyword search (no LLM)
- Admins add magazine issues (PDF) at `/admin/kb`: every page is read, the issue is split into articles and
  summarised, and an admin reviews it before publishing (see `docs/INGEST.md`)
- Collections (e.g. Sunbeam, later KSA) that admins switch on or off per person; each person's questions search
  only their collections

## How it works

See `docs/AGENT_DESIGN.md` for the design, `docs/PARITY.md` for the behaviour it keeps, and `docs/INGEST.md` for
adding issues and collections.

- `steps.py`: the main agent (Deep Agents), its one tool `search_knowledge_base` (query writer → hybrid retrieval →
  dedupe → per-document readers → formatted passages), final-answer streaming, and chat history (edits, reopened
  chats)
- `models.py`: which model does what (all env settings, see below)
- `uploads.py`: turns attached files into content the agent can read
- `retrievers.py`: the fuzzy and hybrid retrievers; `vector_store.py`: vector search in Postgres + pgvector
- `kb.py`: collections and who searches which; `kb_index.py`: the articles in memory (loaded from Postgres) for
  fuzzy search and the commands
- `ingest/`: turning an uploaded PDF into articles (page reading and checking, splitting, assembly), the background
  worker and publishing, and R2 storage
- `scripts/load_vectors.py`: builds and loads the vector index
- `search_engine.py`: the Exact/Fuzzy Search commands
- `access.py`, `admin.py`, `admin_kb.py`: who may sign in, the admin dashboard, and its collections and uploads
- `preview_oauth.py`, `preview_idle.py`: Google sign-in and idle tracking on Render PR previews

## Models and settings

Chat models are Anthropic; OpenAI is used only for the `text-embedding-3-large` embeddings the vector index was
built with. Every model is an env setting. Production (Render) sets `CGPT_MAIN_EFFORT=high` and
`CGPT_READER_EFFORT=medium` (since 27 Oct 2026, for cost and speed); everything else runs on the defaults below.
To go back to the code defaults (both `xhigh`), delete those two env vars on Render and redeploy.

| Env var | Default | Used for |
|---|---|---|
| `CGPT_MAIN_MODEL` | `claude-sonnet-5-5` | the agent (supervisor) |
| `CGPT_MAIN_EFFORT` | by model: `claude-opus-*` → `high`, `claude-sonnet-*` → `xhigh` | the agent's thinking effort (optional override) |
| `CGPT_DETAILED_EFFORT` | `CGPT_MAIN_EFFORT` (or its by-model default) | the agent's thinking effort for Detailed answers |
| `CGPT_QUICK_EFFORT` | `medium` | the agent's thinking effort for Quick answers |
| `CGPT_QUERY_MODEL` | the main model (thinking effort `low`) | writing search queries |
| `CGPT_READER_MODEL` | `claude-haiku-5-5` | reading each retrieved article |
| `CGPT_READER_EFFORT` | `xhigh` | the readers' thinking effort (`off` turns their thinking off) |
| `CGPT_READER_CONCURRENCY` | `40` | most reader calls in flight at once, for the whole process |
| `CGPT_INGEST_READER_MODEL` | `claude-haiku-5-5` (thinking off) | adding issues: reading each page, article summaries |
| `CGPT_INGEST_MODEL` | `claude-sonnet-5-5` (default effort) | adding issues: re-reading pages that fail the check, splitting the issue into articles |

**Answer modes.** The chat composer's Answer picker offers Detailed (preselected) and Quick. Detailed is the agent
exactly as configured above. Quick runs the agent at `CGPT_QUICK_EFFORT`, with one extra prompt line asking for a
concise answer, searching only as much as it needs and citing sources in the same format. The readers, the query
writer and everything else are the same in both. With production's settings, Detailed runs at `high` and Quick at
`medium`. To change either effort, set its env var on Render and redeploy. Which option is preselected is set in
`steps.answer_modes()`. Each answer records its mode: the `answer_mode` LangSmith metadata and tag, and
`cgpt_answer_mode` in the saved message's metadata. Chainlit doesn't save the picker's state, so editing a message
in a reopened chat answers it in Detailed.

Sonnet 5.5 and Haiku 5.5 don't accept `temperature`; it is only set (to 0) for older models. The OpenAI client
honours the standard `OPENAI_BASE_URL` if embeddings are served from an OpenAI-compatible endpoint.

## Setup

1. Clone the repository and install dependencies (Python 3.12):
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory:
```
ANTHROPIC_API_KEY = your-anthropic-api-key
OPENAI_API_KEY = your-openai-api-key
LANGCHAIN_API_KEY = your-langsmith-api-key
LANGCHAIN_TRACING = "true"
LANGCHAIN_PROJECT = your-langsmith-project
VECTOR_DATABASE_URL = postgresql://postgres@127.0.0.1:55433/cgpt_vec   # local copy of the vector index
```
Sign-in and chat history also need `DATABASE_URL`, `CHAINLIT_AUTH_SECRET`, `OAUTH_GOOGLE_CLIENT_ID`,
`OAUTH_GOOGLE_CLIENT_SECRET`, `CHAINLIT_URL` and `ADMIN_EMAILS` (set on Render). Adding issues needs Cloudflare R2:
`R2_ACCOUNT_ID` (or `R2_ENDPOINT`), `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, plus a CORS rule on the
bucket (`docs/INGEST.md`, Operations). Tests need `requirements-dev.txt`.

## Vector index (Postgres + pgvector)

The articles live in the app's own Postgres, in table `kb_article`, which both halves of the hybrid search read
(fuzzy search and the commands from a copy in memory, `kb_index.py`):
- one row per article, id `<source>-<title>` (published issues: `<source>#<n>`), and its collection (`kb_id`);
- the article text, its metadata (jsonb), and a `text-embedding-3-large` embedding as `halfvec(3072)`.
  pgvector indexes `vector` only up to 2,000 dimensions, `halfvec` up to 4,000.
- `knowledge_base.jsonl` stays in the repo; the app reads it once, on the first start after the collections change,
  to record the original articles' order (so fuzzy-search ties break as before).
- An HNSW index (`halfvec_ip_ops`, `m = 32, ef_construction = 200`). Search is by inner product (the embeddings are
  unit-length): the 15 nearest, then MMR picks 5.
- **Chunks** (`kb_chunk`, `chunking.py`): articles of 300+ words are also embedded in ~250-word overlapping windows,
  each with the article title on top, stored at 1,024 dimensions. Each search then adds up to 2 more whole articles
  whose chunks best match the query. That finds passages buried deep inside long articles: on 160 such questions, 90%
  -> 94% found (embeddings alone 29% -> 71%), nothing the plain search found is lost, about 12% more articles read.
  New issues get chunks when published; `scripts/build_chunks.py` adds any missing. `CGPT_CHUNK_SEARCH=0` turns it off.
- The table, index and metadata format are defined once, in `vector_store.py`.

Settings:
- **Which database:** `VECTOR_DATABASE_URL` if set, else `DATABASE_URL`. Production uses `DATABASE_URL`, so it needs
  no new env var.
- **Search breadth:** `CGPT_VECTOR_EF_SEARCH` (default 400). Lower is faster but less exact: at 400, 99% of searches
  give the same final top 5 as an exact search.
- The store's own connections also turn off sequential scans, so Postgres always uses the index. Without that it
  sometimes picks an exact scan that is ~50× slower.

Build and load (idempotent: creates the `vector` extension, table and index, then upserts every article in
`knowledge_base.jsonl`):
```bash
.venv/bin/python scripts/load_vectors.py --dsn-env VECTOR_DATABASE_URL --embeddings .cache/kb_embeddings.npz
```
- Without `--embeddings` it re-embeds all ~5,900 articles, roughly $1 of embeddings.
- Locally, a pgvector Postgres 16 container serves as the copy (`VECTOR_DATABASE_URL` above). Production loads into
  the Render database the same way.
- New issues are added from `/admin/kb` (`docs/INGEST.md`); the old offline `ingest.py` is gone.
- The app adds its collections tables and columns at startup (idempotent; no existing row is rewritten). Running it
  against your local copy therefore migrates that copy; the tests and the E2E use throwaway databases.

Pinecone is no longer used; its env vars were removed from Render on 27 Oct 2026.

## Running Locally

```bash
chainlit run app.py
```

The chat interface will be available at `http://localhost:8000`

## Tests

Offline (no network, no API keys; fake models; `pip install -r requirements-dev.txt`):
```bash
.venv/bin/python -m pytest tests/ eval/test_instrument.py -q
.venv/bin/python tests/test_preview_idle.py
.venv/bin/python tests/test_preview_oauth.py
.venv/bin/python tests/test_on_message.py
TEST_DATABASE_URL=postgresql://postgres@localhost:5432/cgpt_test .venv/bin/python tests/test_admin.py   # throwaway DB, wiped
```
`tests/test_vector_store.py`, `tests/test_kb.py` and `tests/test_ingest_jobs.py` need `VECTOR_DATABASE_URL`
pointing at the local pgvector copy. They only read the copy, and work in throwaway databases they create on the
same server and drop afterwards (`test_kb.py` copies the 5,924 articles in the old layout to test the migration and
that the database gives exactly the jsonl's articles). Without it, those tests are skipped.
`tests/test_ingest_jobs.py` runs a local S3 server (moto) in place of R2.

With real models (small API cost):
```bash
E2E_DATABASE_URL=postgresql://postgres@localhost:5432/cgpt_e2e .venv/bin/python tests/e2e/run_e2e.py   # ~$0.5
```
The E2E test starts a real Chainlit server on a throwaway local Postgres, signs in with a locally minted JWT and
drives it over socket.io (streaming, saved messages, steps, uploads, commands, edit, Detailed / Quick answer modes,
resume, `/admin`, per-person collections, and adding an issue: upload → job → review → publish → searchable only
with that collection). Articles live in a throwaway copy of the local index (`VECTOR_DATABASE_URL`, required); R2 is
a local S3 server. The issue check needs a 4-page PDF in `.cache/` (instructions in the file) and is skipped
without it. Add `--no-llm` to check only the server, sign-in, socket, database plumbing and collection switches.

## Evaluation

`eval/` measures the stack end to end on a fixed question set (answers, citation checks, searches and retrieved
documents, latency including time to the first streamed token, tokens and cost), compares two runs, and has a
blind pairwise judge. See `eval/README.md`. Real user questions and results stay in git-ignored `*.local.*` files.

```bash
.venv/bin/python eval/build_questions.py
.venv/bin/python eval/run_eval.py --label new --out eval/results/new.local.jsonl   # --modes answer=quick: Quick answers
.venv/bin/python eval/report.py eval/results/baseline.local.jsonl eval/results/new.local.jsonl
.venv/bin/python eval/judge.py eval/results/baseline.local.jsonl eval/results/new.local.jsonl --out eval/results/judge.local.jsonl
```

Adding issues has its own eval: read → check → split (no publish) on a folder of PDFs, with a report
(`.cache/ingest-eval/REPORT.local.md`; the PDFs and their text stay local):
```bash
.venv/bin/python eval/ingest_eval.py --budget-usd 9 ".cache/sunbeam_pdfs/20*/*.pdf"
.venv/bin/python eval/ingest_report.py
```

## Admin dashboard

Admins (comma-separated emails in the `ADMIN_EMAILS` env var) sign in normally, then open `/admin`.
Removing someone only stops them signing in; their chats are kept. The People page also has one button per
collection per person (✓ on, ✗ off); **Collections & uploads** (`/admin/kb`) is where issues are added, reviewed
and published, and collections are created (`docs/INGEST.md`).

## Deployment

Render web service: build `pip install -r requirements.txt`, start
`chainlit run app.py -h --port $PORT --host 0.0.0.0`, `PYTHON_VERSION=3.12.0`. It deploys from `main`, and every
pull request gets a preview deploy.
