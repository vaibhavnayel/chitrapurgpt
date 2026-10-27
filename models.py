"""Chat models used by ChitrapurGPT. Every model is an env setting; the defaults are the production choices.

  CGPT_MAIN_MODEL         main agent (supervisor)            claude-sonnet-5-5
  CGPT_MAIN_EFFORT        its thinking effort                by model family: claude-opus-* -> high,
                                                             claude-sonnet-* -> xhigh (so xhigh by default)
  CGPT_QUERY_MODEL        search query writer                CGPT_MAIN_MODEL (thinking effort low)
  CGPT_READER_MODEL       per-document readers               claude-haiku-5-5
  CGPT_READER_EFFORT      readers' thinking effort           xhigh ("off" turns their thinking off)
  CGPT_READER_CONCURRENCY max reader calls in flight         40   (whole process, across parallel searches)
  CGPT_INGEST_MODEL       ingest.py article boundaries       claude-sonnet-5-5

OpenAI is only used for the text-embedding-3-large embeddings the Pinecone index was built with (retrievers.py).
"""
import os

from langchain_anthropic import ChatAnthropic


def supervisor_effort(model: str) -> str | None:
    """Thinking effort for the main agent when CGPT_MAIN_EFFORT isn't set."""
    if model.startswith("claude-opus-"):
        return "high"
    if model.startswith("claude-sonnet-"):
        return "xhigh"
    return None


MAIN_MODEL = os.environ.get("CGPT_MAIN_MODEL", "claude-sonnet-5-5")
MAIN_EFFORT = os.environ.get("CGPT_MAIN_EFFORT") or supervisor_effort(MAIN_MODEL)
QUERY_MODEL = os.environ.get("CGPT_QUERY_MODEL", MAIN_MODEL)
READER_MODEL = os.environ.get("CGPT_READER_MODEL", "claude-haiku-5-5")
READER_EFFORT = os.environ.get("CGPT_READER_EFFORT", "xhigh")
READER_CONCURRENCY = int(os.environ.get("CGPT_READER_CONCURRENCY", "40"))
INGEST_MODEL = os.environ.get("CGPT_INGEST_MODEL", "claude-sonnet-5-5")

# Claude 4.x and older take temperature=0 (the old setting). Newer models reject `temperature` (Sonnet 5.5 and
# Haiku 5.5 return 400 "temperature is deprecated for this model") and use thinking + `effort` instead.
_TAKES_TEMPERATURE = ("claude-3", "claude-sonnet-4", "claude-opus-4", "claude-haiku-4")
# Models whose thinking can't be switched off (adaptive only).
_ALWAYS_THINKS = ("claude-sonnet-5-5", "claude-opus-5", "claude-fable-5")


def chat_model(model: str, *, max_tokens: int, effort: str | None = None, thinking: bool = True,
               **kwargs) -> ChatAnthropic:
    """A ChatAnthropic with the right sampling settings for `model`.
    effort: thinking effort, for models that support it. thinking=False switches thinking off where the model
    allows it (e.g. Haiku 5.5); models that always think ignore it."""
    if model.startswith(_TAKES_TEMPERATURE):
        kwargs["temperature"] = 0
    elif not thinking and not model.startswith(_ALWAYS_THINKS):
        kwargs["thinking"] = {"type": "disabled"}
    elif effort:
        kwargs["effort"] = effort
    return ChatAnthropic(model=model, max_tokens=max_tokens, **kwargs)


def main_llm() -> ChatAnthropic:
    # Thinking tokens count against max_tokens, so leave room for xhigh thinking plus a long answer. Always
    # streamed (the UI streams it anyway), which also lifts the SDK's cap on non-streaming max_tokens.
    return chat_model(MAIN_MODEL, max_tokens=32000, effort=MAIN_EFFORT, streaming=True)


def query_llm() -> ChatAnthropic:
    # Writing a few short search queries needs little thought; low effort keeps each search fast.
    return chat_model(QUERY_MODEL, max_tokens=8000, effort="low", disable_streaming=True,
                      tags=["nostream"])


def reader_llm() -> ChatAnthropic:
    # Thinking counts against max_tokens: 20,000 leaves xhigh thinking room for one article's extraction and stays
    # under the Anthropic SDK's limit for a non-streaming call (~21k).
    off = READER_EFFORT.lower() in ("", "off", "none")
    return chat_model(READER_MODEL, max_tokens=20000, effort=None if off else READER_EFFORT, thinking=not off,
                      disable_streaming=True, tags=["nostream"])
