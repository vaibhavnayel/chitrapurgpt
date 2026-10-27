# Chitrapur GPT

A conversational AI assistant for the Chitrapur Saraswat community. It answers questions by researching a knowledge
base of ~5,900 magazine articles (Chitrapur Sunbeam), with citations. Built on Chainlit and LangChain Deep Agents.

## Features

- Chat interface powered by Chainlit, with Google sign-in and an allow-list managed at `/admin`
- An agent that searches the knowledge base (several searches at once when useful), reads each retrieved article
  for the question, and answers with Wikipedia-style citations; the answer streams as it is written
- Hybrid search: fuzzy and phonetic matching over the articles plus vector search in Postgres (pgvector)
- Translation and transliteration, and reading files you attach (images, PDFs, Word, Excel, CSV, text)
- Exact Search and Fuzzy Search commands for boolean keyword search (no LLM)

## How it works

See `docs/AGENT_DESIGN.md` for the design and `docs/PARITY.md` for the behaviour it keeps.

- `steps.py`: the main agent (Deep Agents), its one tool `search_knowledge_base` (query writer → hybrid retrieval →
  dedupe → per-document readers → formatted passages), final-answer streaming, and chat history (edits, reopened
  chats)
- `models.py`: which model does what (all env settings, see below)
- `uploads.py`: turns attached files into content the agent can read
- `retrievers.py`: the fuzzy and hybrid retrievers; `vector_store.py`: vector search in Postgres + pgvector
- `scripts/load_vectors.py`: builds and loads the vector index
- `search_engine.py`: the Exact/Fuzzy Search commands
- `ingest.py`: offline import of new magazine issues into the knowledge base
- `access.py`, `admin.py`: who may sign in, and the admin dashboard
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
| `CGPT_QUERY_MODEL` | the main model (thinking effort `low`) | writing search queries |
| `CGPT_READER_MODEL` | `claude-haiku-5-5` | reading each retrieved article |
| `CGPT_READER_EFFORT` | `xhigh` | the readers' thinking effort (`off` turns their thinking off) |
| `CGPT_READER_CONCURRENCY` | `40` | most reader calls in flight at once, for the whole process |
| `CGPT_INGEST_MODEL` | `claude-sonnet-5-5` | `ingest.py` article-boundary extraction |

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
`OAUTH_GOOGLE_CLIENT_SECRET`, `CHAINLIT_URL` and `ADMIN_EMAILS` (set on Render).

## Vector index (Postgres + pgvector)

The vector half of the hybrid search lives in the app's own Postgres, in table `kb_article`:
- one row per article, id `<source>-<title>`;
- the article text, its metadata (jsonb), and a `text-embedding-3-large` embedding as `halfvec(3072)`.
  pgvector indexes `vector` only up to 2,000 dimensions, `halfvec` up to 4,000.
- An HNSW index (`halfvec_ip_ops`, `m = 32, ef_construction = 200`). Search is by inner product (the embeddings are
  unit-length): the 15 nearest, then MMR picks 5.
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
- `ingest.py` adds new articles through the same store and skips ids already present.

Pinecone is no longer used; its env vars were removed from Render on 27 Oct 2026.

## Running Locally

```bash
chainlit run app.py
```

The chat interface will be available at `http://localhost:8000`

## Tests

Offline (no network, no API keys; fake models):
```bash
.venv/bin/python -m pytest tests/ eval/test_instrument.py -q --ignore=tests/test_admin.py --ignore=tests/e2e
.venv/bin/python tests/test_preview_idle.py
.venv/bin/python tests/test_preview_oauth.py
TEST_DATABASE_URL=postgresql://postgres@localhost:5432/cgpt_test .venv/bin/python tests/test_admin.py   # throwaway DB, wiped
```
`tests/test_vector_store.py` needs `VECTOR_DATABASE_URL` pointing at the local pgvector copy. It reads the copy and
tests the store in a throwaway database it creates on the same server and drops afterwards. Without it, those tests
are skipped.

With real models (small API cost):
```bash
.venv/bin/python tests/ingest_live_check.py      # ingest.py against CGPT_INGEST_MODEL (~$0.01)
E2E_DATABASE_URL=postgresql://postgres@localhost:5432/cgpt_e2e .venv/bin/python tests/e2e/run_e2e.py   # ~$0.5
```
The E2E test starts a real Chainlit server on a throwaway local Postgres, signs in with a locally minted JWT and
drives it over socket.io (streaming, saved messages, steps, uploads, commands, edit, resume, `/admin`). Vector search
goes to the local copy (`VECTOR_DATABASE_URL`, required). Add `--no-llm` to check only the server, sign-in, socket
and database plumbing.

## Evaluation

`eval/` measures the stack end to end on a fixed question set (answers, citation checks, searches and retrieved
documents, latency including time to the first streamed token, tokens and cost), compares two runs, and has a
blind pairwise judge. See `eval/README.md`. Real user questions and results stay in git-ignored `*.local.*` files.

```bash
.venv/bin/python eval/build_questions.py
.venv/bin/python eval/run_eval.py --label new --out eval/results/new.local.jsonl
.venv/bin/python eval/report.py eval/results/baseline.local.jsonl eval/results/new.local.jsonl
.venv/bin/python eval/judge.py eval/results/baseline.local.jsonl eval/results/new.local.jsonl --out eval/results/judge.local.jsonl
```

## Admin dashboard

Admins (comma-separated emails in the `ADMIN_EMAILS` env var) sign in normally, then open `/admin`.
Removing someone only stops them signing in; their chats are kept.

## Deployment

Render web service: build `pip install -r requirements.txt`, start
`chainlit run app.py -h --port $PORT --host 0.0.0.0`, `PYTHON_VERSION=3.12.0`. It deploys from `main`, and every
pull request gets a preview deploy.
