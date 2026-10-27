"""Live check (Anthropic API, ~$0.02): ingest.extract_article_boundaries on a short two-article sample with the
configured CGPT_INGEST_MODEL returns the ArticleBoundaries structure. Nothing is written anywhere.

  .venv/bin/python tests/ingest_live_check.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ingest  # noqa: E402
import models  # noqa: E402

SAMPLE = """THE CHITRAPUR SUNBEAM
April 1976
Vol. 12 No. 4
------- PAGE END -------
CONTENTS
Editorial ........................ 2
Shri Chitrapur Rathotsava ........ 3
------- PAGE END -------
EDITORIAL
The month of Chaitra brings the Rathotsava to Shirali once again. Devotees from Bombay,
Bangalore and Mangalore gather at the Math.
------- PAGE END -------
SHRI CHITRAPUR RATHOTSAVA
by K. Rao
The Rathotsava began on Chaitra Shukla Dashami. On the sixth day the chariot was pulled
through the village by hundreds of devotees, and Swamiji gave His ashirvachan.
The celebrations ended with the Avabhritha snana.
------- PAGE END -------
"""


async def main():
    print("model:", models.INGEST_MODEL)
    boundaries = await ingest.extract_article_boundaries(SAMPLE)
    for b in boundaries:
        print(f"  {b.title!r} by {b.author!r}, {b.published_date}, lines {b.start_line}-{b.end_line}, pages {b.start_page}-{b.end_page}")
    titles = [b.title.lower() for b in boundaries]
    assert any("rathotsava" in t for t in titles), titles
    assert all(isinstance(b, ingest.ArticleBoundary) for b in boundaries)
    print("ingest live check: OK")


if __name__ == "__main__":
    asyncio.run(main())
