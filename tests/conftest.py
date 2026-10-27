# test_admin.py is a standalone script (run it with TEST_DATABASE_URL set: `python tests/test_admin.py`). Importing it
# under pytest replaces the `steps` module other tests patch, so keep plain `pytest tests/` from collecting it.
# The E2E server (tests/e2e/) is also run directly: `python tests/e2e/run_e2e.py`.
collect_ignore = ["test_admin.py"]
collect_ignore_glob = ["e2e/*"]
