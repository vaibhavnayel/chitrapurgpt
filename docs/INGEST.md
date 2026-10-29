# Adding magazine issues, and collections

Admins add magazine issues (PDF) to the knowledge base from `/admin/kb`; the whole PDF → articles → search pipeline
runs in the web service. Articles belong to collections (knowledge bases) such as "Sunbeam" and, later, "KSA"; admins
switch collections on or off per person. Code: `ingest/` (pipeline, worker, storage), `kb.py` (collections and who
searches what), `kb_index.py` (articles in memory), `admin_kb.py` (the admin pages).

## What an admin does

1. **Collections & uploads** (`/admin/kb`): pick a collection (Sunbeam by default) and a PDF, press *Upload and
   process*. The browser checks the file (PDF, ≤ 250 MB), computes its SHA-256 and uploads it straight to Cloudflare
   R2; the app then queues a job. If the same file was uploaded before, the admin is asked to confirm.
2. The job page shows progress (fetching, reading pages *n/N*, splitting, summaries), then the **review**:
   - up top: issue name and date, collection, pages, articles by kind, contents entries matched (e.g. 22/22),
     blocks in no article / in two, pages re-read by Sonnet, flags, cost, processing time;
   - the flags, and the splitter's notes (they explain e.g. a contents page with wrong page numbers);
   - the article list (title, kind, pages, length); each article opens beside its page images;
   - edits: rename an article, exclude/include it, fix the issue's name, date and citation file name;
   - actions: **Publish**, **Re-run split** (keeps the page reads), **Re-run whole job**, **Stop**, **Delete**.
3. **Publish** puts the articles into search for everyone with that collection switched on. Publishing again
   (after edits) replaces them; so does publishing a newer upload of the same issue (the older upload is marked
   "Replaced"). Deleting a published upload removes its articles from search and its files from R2.
4. **Collections**: create one (id, name, description, citation prefix, default on/off); people you haven't switched
   get its default. Switch collections per person on the **People** page (one button per collection).

The original 5,924 articles are untouched: they keep their text, metadata and ids, and stay in Sunbeam.

**An issue the original knowledge base already has** (it covers Sunbeam up to January 2025). Example: uploading
January 2025. The new articles get the same citation file name, `Sunbeam-2025-January.txt`, and new ids
(`Sunbeam-2025-January.txt#001` …), so they can't overwrite the 14 original articles of that issue. The review page
warns that the originals already exist and that publishing adds the new articles *alongside* them. After publishing,
both sets are searchable: the eval's January 2025 split has 27 articles, including the 3 contents entries the
original import lacks. But articles both sets have (e.g. the editorial) are now in search twice, and can take two of
a search's slots. Nothing removes the originals. Uploading the same issue again later replaces the first upload's
published articles (marked "Replaced"), never the originals. Whether to re-ingest 2023 – January 2025, and whether to
replace the originals when doing so, is an open decision (it would need an explicit "replace the original articles of
this issue" step).

## The pipeline (one job per uploaded PDF)

| Step | What happens | Model |
|---|---|---|
| Upload | Presigned PUT to R2 (`<kb>/<yyyy-mm or upload date>/<sha256>.pdf`); the app checks size, `%PDF-` header and that the object arrived | – |
| Worker | One asyncio task in the web process claims jobs from Postgres (`FOR UPDATE SKIP LOCKED`), one at a time, heartbeat every 20 s | – |
| Download | To a temp file; SHA-256 must match the upload; 1–400 pages | – |
| Read pages | Each page rendered at 1568 px (PNG, JPEG if > 3.5 MB) and transcribed into typed blocks (prompt from the spike); 8 pages at a time; an unusable reply is retried once. Each page's read is saved as soon as it's done, with its cost | Haiku 5.5, thinking off (`CGPT_INGEST_READER_MODEL`) |
| Check pages | Where the PDF has a clean text layer (≥ 200 characters outside legacy Devanagari fonts: AkrutiDev*, SanskritText, Shree*, Kruti*, Shivaji…), the read must contain ≥ 97% of the layer's words; else (or if the read is empty or unusable) Sonnet reads the page again and the better read is kept. Still bad → a review flag | Sonnet 5.5, default effort (`CGPT_INGEST_MODEL`) |
| Split | An issue map (every content block with an id `p12.b3`, numbered contiguously per page, page furniture left out), the contents page in full, and the cover's and running headers/footers' lines (for the date). The answer is articles as block-id ranges. A tolerant checker reports blocks in no article or two, ranges it can't place, and contents entries without an article; one repair round sends those findings back. Leftovers → flags | Sonnet 5.5, default effort (`CGPT_INGEST_MODEL`) |
| Assemble | Article text = the blocks in range order, one paragraph each (headlines on their own lines, tables as Markdown, captions kept, the reader's picture descriptions left out). Over ~8,000 tokens → "Title (part k of n)", split at block boundaries, never dropped. Under 40 characters (a section divider) → excluded to start with | – |
| Summaries | 1–3 sentences per article | Haiku 5.5, thinking off |
| Review | `/admin/kb/jobs/<id>`; page previews (1200 px JPEG) were stored in R2 while reading, or are rendered on demand | – |
| Publish | Embed (text-embedding-3-large, as the original articles; an article that is mostly in an Indian script is embedded by its English title and summary instead, so English questions find it, see `embedding_text`), also embed the chunks of each long article (`chunking.py`), then in **one transaction** delete any rows from an earlier publish of the same issue (their chunks go with them) and insert the new articles into `kb_article` and their chunks into `kb_chunk`; refresh the in-memory index | OpenAI embeddings |

**Metadata** of a published article has exactly the original keys: `source` (e.g. `Sunbeam-2025-February.txt`,
`Sunbeam-2026-July-August.txt`; other collections use their own prefix), `published_date` (DD-MM-YYYY), `title`,
`author`, `summary`, `start_page`/`end_page` (PDF page numbers: what anyone opening the PDF sees; for recent Sunbeam
issues they're the printed numbers too), and `start_line`/`end_line` as missing values (they only meant something
in the old text-file pipeline). Ids are `<source>#<n>` (e.g. `Sunbeam-2025-February.txt#007`), so they can never
collide with each other or with the original `<source>-<title>` ids.

**Resuming.** A job left "running" by a process that stopped (restart, redeploy) has a stale heartbeat (3 min) and is
claimed again; its saved page reads aren't read (or paid for) twice. A deploy's overlap is safe: the old process keeps
its job's heartbeat fresh until it exits.

**Memory** stays flat: the PDF is on disk, pages are rendered one at a time (pdfium is serialised by a lock), and at
most 8 page images are in flight.

## Collections and search

- Postgres is the single source of truth for articles: fuzzy search (the agent's tool) and the Exact/Fuzzy Search
  commands read `kb_article`, loaded into memory at startup (~13 s for 5,924 articles, as the jsonl took) and
  refreshed after a publish or delete (only new articles are pre-processed). The worker also notices changes made by
  another process within a minute. `knowledge_base.jsonl` stays in the repo; at runtime it's read once, by the first
  startup after this change, to record the original articles' order (`kb_legacy_order`), so fuzzy-search ties break
  exactly as before.
- Tables (all idempotent at startup, safe on the live database; no existing row is rewritten, so the HNSW index
  is untouched): `knowledge_base` (id, name, description, source_prefix, default_on), `kb_user_access` (email,
  kb_id, enabled, set_by, updated_at; no row = the collection's default), `kb_article.kb_id` (original rows:
  `sunbeam`), `kb_article.seq`, `kb_article.issue_id`, `kb_legacy_order`, and the job tables `ingest_job`,
  `ingest_page`, `ingest_article`.
- Each message searches the person's switched-on collections: vector search filters `kb_id` (pgvector 0.8's
  `hnsw.iterative_scan = strict_order` keeps the k nearest; there's deliberately no btree on `kb_id`, which made the
  planner choose an exact scan), fuzzy search and the commands skip other collections. When a person's collections
  include every collection that has articles (everyone, while Sunbeam is the only one), nothing is filtered: the
  search is exactly the one before this change.
- The agent's prompt names the person's collections, except for Sunbeam alone (then it's unchanged).
- Someone with no collection switched on gets a clear message instead of an answer.

## Costs and times (eval on 40 born-digital issues, 2021–2026; `.cache/ingest-eval/REPORT.local.md`)

| | Typical issue (60 pages) |
|---|---|
| Page reads + checks | ~$0.04 (Haiku ~$0.0007 a page; 1.4% of pages re-read by Sonnet, fewer with the final check) |
| Split (+ a repair round in 5 of 40 issues) | ~$0.10 (max $0.27 for a 113-page issue) |
| Summaries + embeddings | ~$0.01 + < $0.001 |
| **Total** | **~$0.16**; time ~35 s reading + ~25 s splitting + summaries, i.e. 1–2 minutes |

Quality (40 issues): 519/524 contents entries matched to an article (the 5 others are contents pages with wrong
page numbers; the articles exist), no block left out or doubled, 2 pages flagged by the final check (text-layer
defects, not reading errors), 1,124 articles vs the old pipeline's 10–15 per issue (it lost notices, appeals,
invitations and rate tables).

## Operations

**Env vars** (Render):

| Var | Needed for | Notes |
|---|---|---|
| `R2_ACCOUNT_ID` | uploads | or set `R2_ENDPOINT` directly |
| `R2_ENDPOINT` | uploads | optional; default `https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com` (use the `.eu.` endpoint for an EU bucket) |
| `R2_BUCKET` | uploads | |
| `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | uploads | an R2 API token with Object Read & Write on that bucket only |
| `CGPT_INGEST_READER_MODEL` | – | default `claude-haiku-5-5` (must allow thinking to be switched off) |
| `CGPT_INGEST_MODEL` | – | default `claude-sonnet-5-5` |
| `INGEST_WORKER` | – | `0` off, `1` on. Default: on, except on PR previews (`IS_PULL_REQUEST=true`), which share the live database |

Without the R2 vars the app works as before and `/admin/kb` explains that uploads need R2 (collections and
per-person switches still work).

**One-time setup** (production):
1. Create the R2 bucket and an API token scoped to it (Object Read & Write).
2. Set the bucket's CORS policy so the admin page can upload to it from the browser (a presigned URL is the
   permission; CORS only lets the browser make the request):
   ```json
   [{"AllowedOrigins": ["https://<the site's origin>"], "AllowedMethods": ["PUT"],
     "AllowedHeaders": ["content-type"], "MaxAgeSeconds": 3600}]
   ```
   Add PR-preview origins too if uploads should work there (they use the live database, so normally don't).
3. Set the R2 env vars on Render and deploy. The first start adds the new tables and columns (seconds) and records
   the original articles' order from `knowledge_base.jsonl`.

**When something goes wrong**: a failed job shows its error on its page (e.g. "This file couldn't be opened as a
PDF"); *Re-run whole job* tries again, *Delete* removes it. A job stuck "running" after a crash resumes by itself
within 3 minutes. Every API call's cost is added to the job (`cost_usd`).

**Single instance.** Render runs one instance. A second one would also run a worker (safe: jobs are claimed with
`SKIP LOCKED` and heartbeats), and its in-memory index catches up within a minute of a publish.

**PR previews use the live database.** Their worker is off by default, so a preview never takes live jobs. An issue
uploaded on a preview is processed by the live site once it runs this code. Publishing from a preview's review page
publishes to the live knowledge base, as every other admin action there does. To try the whole flow on a preview,
set `INGEST_WORKER=1` on it and expect real (live) results.

## Tests and the eval

- `tests/test_ingest_pipeline.py` (offline, fake LLM, generated PDFs): page check and Sonnet fallback, contiguous
  ids and the issue map, contents parsing, range expansion and the checker, the repair round, assembly and parts.
- `tests/test_kb.py` (local pgvector, throwaway database with the 5,924 originals in the old layout): the migration
  rewrites no row; the database's articles equal `knowledge_base.jsonl` (text, metadata, order) and give identical
  command and fuzzy results; filtered vector search keeps recall and uses the HNSW index; transactional re-publish;
  per-collection filtering in all three search paths; access defaults.
- `tests/test_ingest_jobs.py` (throwaway database + a local S3 server, moto): presigned upload, the worker end to
  end, resume after a crash, re-runs, publish, delete, bad uploads.
- `tests/test_admin.py`: per-person collection switches on the People page.
- `tests/e2e/run_e2e.py`: upload → job → review → publish → searchable for a person with that collection and not
  for one without, on the real local server (needs a 4-page fixture PDF in `.cache/`, see the file).
- `eval/ingest_eval.py` + `eval/ingest_report.py`: read → check → split on a folder of PDFs, with the report
  (local only).
