"""Turning an uploaded magazine issue (PDF) into knowledge-base articles. See docs/INGEST.md.

  pdf.py        rendering pages and reading their text layer (pypdfium2)
  llm.py        the Anthropic calls, with per-call cost
  pages.py      reading each page (Haiku) and checking it against the text layer (Sonnet re-read if needed)
  split.py      the issue map, splitting it into articles (Sonnet) and checking the split (one repair round)
  assemble.py   article text from the blocks, part splitting, summaries
  pipeline.py   one issue end to end, independent of where page reads are stored
  storage.py    the originals and page previews in Cloudflare R2
  jobs.py       ingest jobs in Postgres, the background worker, publishing
"""
