"""Offline tests for models.py: which model and thinking settings each role gets, and the env overrides.

  .venv/bin/python -m pytest tests/test_models.py -q
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

ENV = ("CGPT_MAIN_MODEL", "CGPT_MAIN_EFFORT", "CGPT_QUERY_MODEL", "CGPT_READER_MODEL", "CGPT_READER_EFFORT",
       "CGPT_READER_CONCURRENCY", "CGPT_INGEST_MODEL")


@pytest.fixture
def models(monkeypatch):
    def load(**env):
        for k in ENV:
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        import models as m
        return importlib.reload(m)
    yield load
    for k in ENV:
        os.environ.pop(k, None)
    import models as m
    importlib.reload(m)


def test_production_defaults(models):
    m = models()
    main, reader, query = m.main_llm(), m.reader_llm(), m.query_llm()
    assert (main.model, main.reasoning_effort, main.max_tokens, main.streaming) == ("claude-sonnet-5-5", "xhigh", 32000, True)
    assert main.temperature is None
    assert (reader.model, reader.reasoning_effort, reader.max_tokens) == ("claude-haiku-5-5", "xhigh", 20000)
    assert reader.thinking is None or reader.thinking.get("type") != "disabled", "readers think"
    assert reader.disable_streaming and "nostream" in reader.tags and reader.temperature is None
    assert (query.model, query.reasoning_effort) == ("claude-sonnet-5-5", "low"), "query writer unchanged"
    assert m.READER_CONCURRENCY == 40 and m.INGEST_MODEL == "claude-sonnet-5-5"


def test_supervisor_effort_follows_the_model_family(models):
    assert models(CGPT_MAIN_MODEL="claude-opus-5-5").main_llm().reasoning_effort == "high"
    assert models(CGPT_MAIN_MODEL="claude-sonnet-5").main_llm().reasoning_effort == "xhigh"
    assert models(CGPT_MAIN_MODEL="claude-haiku-5-5").main_llm().reasoning_effort is None
    m = models(CGPT_MAIN_MODEL="claude-opus-5-5")
    assert m.QUERY_MODEL == "claude-opus-5-5", "the query writer follows the main model by default"


def test_overrides(models):
    m = models(CGPT_MAIN_EFFORT="medium", CGPT_READER_EFFORT="high", CGPT_READER_CONCURRENCY="8")
    assert m.main_llm().reasoning_effort == "medium"
    assert m.reader_llm().reasoning_effort == "high" and m.READER_CONCURRENCY == 8
    off = models(CGPT_READER_EFFORT="off").reader_llm()
    assert off.thinking == {"type": "disabled"} and off.reasoning_effort is None


def test_older_models_keep_temperature_zero(models):
    m = models(CGPT_MAIN_MODEL="claude-sonnet-4-5")
    main = m.main_llm()
    assert main.temperature == 0 and main.reasoning_effort is None, "no effort on models that don't support it"
