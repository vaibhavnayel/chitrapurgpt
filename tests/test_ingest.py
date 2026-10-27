"""Offline test of ingest.py: article-boundary extraction with a fake model, then main() writing a knowledge base in a
temp directory with the Pinecone upsert stubbed. The real knowledge_base.jsonl and index are never touched.

  .venv/bin/python -m pytest tests/test_ingest.py -q
(tests/ingest_live_check.py runs the extraction against the real model.)
"""
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402

import ingest  # noqa: E402
import models  # noqa: E402

TEXT = "Cover\n------- PAGE END -------\nRathotsava report\nThe chariot was pulled.\n------- PAGE END -------\n"


class FakeLLM:
    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append(("with_structured_output", schema, kwargs))
        return self

    def with_retry(self):
        return self

    async def ainvoke(self, messages):
        self.calls.append(("ainvoke", messages))
        parsed = ingest.ArticleBoundaries(notes="", article_boundaries=[
            ingest.ArticleBoundary(title="Cover Page", author=None, summary="cover", published_date="01-04-1976",
                                   start_line=0, end_line=1, start_page=1, end_page=1),
            ingest.ArticleBoundary(title="Rathotsava report", author="Editor", summary="report",
                                   published_date="01-04-1976", start_line=2, end_line=4, start_page=2, end_page=2)])
        return {"parsed": parsed, "raw": AIMessage(content="", usage_metadata={
            "input_tokens": 100, "output_tokens": 50, "total_tokens": 150})}


def test_extract_uses_configured_model_and_same_schema(monkeypatch):
    fake = FakeLLM()
    seen = {}

    def chat_model(model, **kwargs):
        seen.update(model=model, **kwargs)
        return fake

    monkeypatch.setattr(models, "chat_model", chat_model)
    boundaries = asyncio.run(ingest.extract_article_boundaries(TEXT))
    assert seen["model"] == models.INGEST_MODEL == "claude-sonnet-5-5"
    assert fake.calls[0][1] is ingest.ArticleBoundaries
    assert fake.calls[0][2] == {"method": "json_schema", "include_raw": True}
    system, human = fake.calls[1][1]
    assert "You are a PDF parser" in system.content and human.content.startswith("1. Cover")
    docs = ingest.parse_articles(TEXT, boundaries, "documents/txts/Sunbeam-1976-April.txt")
    assert [d.metadata["title"] for d in docs] == ["Cover Page", "Rathotsava report"]
    assert docs[1].metadata["source"] == "Sunbeam-1976-April.txt" and "chariot" in docs[1].page_content


def test_main_writes_temp_kb_and_stubbed_upsert(tmp_path, monkeypatch):
    real_kb = ROOT / "knowledge_base.jsonl"
    before = hashlib.sha256(real_kb.read_bytes()).hexdigest()
    articles = tmp_path / "documents" / "articles"
    articles.mkdir(parents=True)
    for i in range(3):
        doc = ingest.Document(page_content=f"article {i} " * 5, metadata={"source": f"S{i}.txt", "title": f"T{i}"})
        ingest.save_doc_to_json(doc, str(articles / f"S{i}.txt-T{i}.json"))
    upserts = []
    monkeypatch.setattr(ingest, "add_to_vector_store", lambda batch: upserts.append(batch))

    class Enc:
        def encode(self, text):
            return text.split()

    monkeypatch.setattr(ingest, "get_encoding", lambda name: Enc())
    monkeypatch.chdir(tmp_path)
    asyncio.run(ingest.main())
    kb = [json.loads(line) for line in (tmp_path / "knowledge_base.jsonl").open()]
    assert sorted(d["metadata"]["title"] for d in kb) == ["T0", "T1", "T2"]
    assert sorted(d.metadata["title"] for batch in upserts for d in batch) == ["T0", "T1", "T2"]
    os.chdir(ROOT)
    assert hashlib.sha256(real_kb.read_bytes()).hexdigest() == before
