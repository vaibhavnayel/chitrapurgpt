# test_admin.py is a standalone script (run it with TEST_DATABASE_URL set: `python tests/test_admin.py`). Importing it
# under pytest replaces the `steps` module other tests patch, so keep plain `pytest tests/` from collecting it.
# The E2E server (tests/e2e/) is also run directly: `python tests/e2e/run_e2e.py`.
collect_ignore = ["test_admin.py", "test_kb_admin.py"]
collect_ignore_glob = ["e2e/*"]

# The Exact/Fuzzy Search commands read articles from the in-memory index, which the app loads from Postgres. The
# old search-engine scripts in this folder search at import time, so give them the original articles (from the
# jsonl, no database). tests/fakes.py swaps in a tiny knowledge base for the agent tests.
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kb_index  # noqa: E402
from retrievers import load_docs_from_jsonl  # noqa: E402

if not kb_index.INDEX.loaded:
    kb_index.INDEX.set_documents([(f"legacy-{n}", "sunbeam", d) for n, d in
                                  enumerate(load_docs_from_jsonl(str(Path(__file__).resolve().parents[1]
                                                                     / "knowledge_base.jsonl")))])
