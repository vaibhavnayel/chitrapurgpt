"""USD prices per million tokens, used to cost eval runs.

Sources (checked 2026-10-27):
  https://platform.claude.com/docs/en/about-claude/pricing
  https://developers.openai.com/api/docs/pricing
Cache writes are costed at the 5-minute TTL rate (what automatic/ephemeral caching uses).
"""

# model prefix -> (input, cache_write_5m, cache_read, output)
PRICES = {
    "claude-sonnet-5-5": (2.00, 2.50, 0.10, 10.00),
    "claude-sonnet-5": (2.00, 2.50, 0.20, 10.00),
    "claude-sonnet-4-5": (3.00, 3.75, 0.30, 15.00),
    "claude-haiku-5-5": (0.10, 0.125, 0.01, 0.50),
    "claude-haiku-4-5": (1.00, 1.25, 0.10, 5.00),
    "claude-opus-5-5": (5.00, 6.25, 0.25, 25.00),
    "gpt-5-mini": (0.25, 0.25, 0.025, 2.00),
}
# Haiku 5.5 bills the whole request at a higher tier when the prompt (incl. cache reads/writes) is over 100k.
LONG_PROMPT_PRICES = {
    "claude-haiku-5-5": (100_000, (0.50, 0.625, 0.05, 2.50)),
}
EMBEDDING_PRICES = {"text-embedding-3-large": 0.13}


def canonical_model(name: str | None) -> str | None:
    """'claude-sonnet-4-5-20250929' -> 'claude-sonnet-4-5', 'gpt-5-mini-2025-08-07' -> 'gpt-5-mini'."""
    if not name:
        return name
    name = name.split(":", 1)[-1]  # "anthropic:claude-..." style ids
    for prefix in sorted(PRICES, key=len, reverse=True):
        if name == prefix or name.startswith(prefix + "-"):
            return prefix
    return name


def cost_usd(model: str | None, input_tokens: int, output_tokens: int,
             cache_read: int = 0, cache_write: int = 0) -> float | None:
    """`input_tokens` is the total prompt size including cached tokens (LangChain's usage_metadata convention)."""
    key = canonical_model(model)
    if key not in PRICES:
        return None
    p_in, p_write, p_read, p_out = PRICES[key]
    if key in LONG_PROMPT_PRICES and input_tokens > LONG_PROMPT_PRICES[key][0]:
        p_in, p_write, p_read, p_out = LONG_PROMPT_PRICES[key][1]
    uncached = max(input_tokens - cache_read - cache_write, 0)
    return (uncached * p_in + cache_write * p_write + cache_read * p_read + output_tokens * p_out) / 1e6
