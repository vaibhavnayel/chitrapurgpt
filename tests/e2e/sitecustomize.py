# Loaded by Python at startup for the E2E server only (run_e2e.py puts this directory on its PYTHONPATH and sets
# CGPT_E2E_SHIMS=1). Applies the eval sandbox's network shims; see eval/sandbox_shims.py. Never used in production.
import os
import sys
from pathlib import Path

if os.environ.get("CGPT_E2E_SHIMS") == "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
    import sandbox_shims

    sandbox_shims.apply()
