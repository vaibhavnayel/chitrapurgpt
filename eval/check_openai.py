"""One tiny embeddings call (a single word, < $0.000001) to check the OpenAI account can serve requests.
Exit code 0 = OK, 2 = out of credit (429 insufficient_quota / credit exhausted), 1 = other error.

  .venv/bin/python eval/check_openai.py
"""
import sys
from pathlib import Path

import dotenv
import openai

dotenv.load_dotenv(Path(__file__).resolve().parents[1] / ".env")

try:
    openai.OpenAI().embeddings.create(model="text-embedding-3-large", input="ok")
    print("openai: OK")
except openai.RateLimitError as e:
    body = str(e)
    print("openai: 429 " + ("out of credit" if ("quota" in body or "credit" in body) else "rate limited"), "-", body[:160])
    sys.exit(2)
except Exception as e:  # noqa: BLE001
    print("openai: error", type(e).__name__, str(e)[:160])
    sys.exit(1)
