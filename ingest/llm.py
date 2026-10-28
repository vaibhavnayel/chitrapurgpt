"""The Anthropic calls ingest makes, with token usage and cost per call.

Plain Anthropic SDK (no LangChain): every call is one message with optional image, streamed (long outputs and
thinking would otherwise trip the SDK's non-streaming time limit). The SDK retries 429/5xx/overloaded itself.
Tests pass a fake with the same `complete()` signature.
"""
from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field

# USD per million tokens: (input, cache write 5 min, cache read, output). Same numbers as eval/pricing.py
# (tests/test_ingest_pipeline.py checks they agree).
PRICES = {
    "claude-sonnet-5-5": (2.00, 2.50, 0.10, 10.00),
    "claude-haiku-5-5": (0.10, 0.125, 0.01, 0.50),
}
# Haiku 5.5 bills the whole request at a higher tier when the prompt is over 100k tokens.
LONG_PROMPT = {"claude-haiku-5-5": (100_000, (0.50, 0.625, 0.05, 2.50))}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0) -> float:
    """`input_tokens` is the uncached part, as the Anthropic API reports it."""
    p_in, p_write, p_read, p_out = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    if model in LONG_PROMPT and input_tokens + cache_write + cache_read > LONG_PROMPT[model][0]:
        p_in, p_write, p_read, p_out = LONG_PROMPT[model][1]
    return (input_tokens * p_in + cache_write * p_write + cache_read * p_read + output_tokens * p_out) / 1e6


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write: int = 0
    cache_read: int = 0
    cost: float = 0.0
    secs: float = 0.0
    stop_reason: str | None = None

    def as_dict(self) -> dict:
        return {"model": self.model, "in": self.input_tokens, "out": self.output_tokens,
                "cache_write": self.cache_write, "cache_read": self.cache_read, "cost": round(self.cost, 6),
                "secs": round(self.secs, 2), "stop": self.stop_reason}


@dataclass
class Spend:
    """Running total for a job or an eval run."""
    usd: float = 0.0
    calls: list[dict] = field(default_factory=list)

    def add(self, usage: Usage, what: str) -> None:
        self.usd += usage.cost
        self.calls.append({"what": what, **usage.as_dict()})


def image_block(data: bytes, media_type: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                        "data": base64.b64encode(data).decode()}}


class AnthropicLLM:
    def __init__(self, client=None, *, timeout: float = 600.0, max_retries: int = 4):
        if client is None:
            import anthropic
            client = anthropic.AsyncAnthropic(timeout=timeout, max_retries=max_retries)
        self.client = client

    async def complete(self, *, model: str, messages: list[dict], system: str | None = None,
                       max_tokens: int = 16000, thinking: bool | None = None) -> tuple[str, Usage]:
        """One call. thinking=False switches thinking off (Haiku 5.5); None leaves the model's default (Sonnet 5.5
        thinks adaptively at its default effort). Returns the reply's text and its usage."""
        kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if thinking is False:
            kwargs["thinking"] = {"type": "disabled"}
        t = time.monotonic()
        async with self.client.messages.stream(**kwargs) as stream:
            msg = await stream.get_final_message()
        u = msg.usage
        usage = Usage(model=model, input_tokens=u.input_tokens or 0, output_tokens=u.output_tokens or 0,
                      cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
                      cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                      secs=time.monotonic() - t, stop_reason=msg.stop_reason)
        usage.cost = cost_usd(model, usage.input_tokens, usage.output_tokens, usage.cache_write, usage.cache_read)
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return text, usage


async def gather_limited(limit: int, coros) -> list:
    """Run coroutines with at most `limit` in flight, keeping order."""
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))
