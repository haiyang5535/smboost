"""Per-model $/1M-token pricing for external-API baseline runs.

All prices are **USD per 1,000,000 tokens**, split into input and output.
These are reference list prices as of 2026-04-24; batch / cached pricing is
intentionally ignored here — baseline runs are small (N<=100 typical) and
live-priced.

Claude Sonnet 4.6: published at https://www.anthropic.com/pricing ($3/$15 per 1M).
GPT-4o: OpenAI list pricing ($2.50/$10 per 1M at 2025 rate card).
Llama-3-70B via OpenRouter: Meta's open weights, OpenRouter aggregates
pricing from hosted inference providers; the commonly-quoted rate is
around $0.59/$0.79 per 1M for llama-3-70b-instruct.
"""
from __future__ import annotations


# Canonical $/1M-token rates. Keys are case-insensitive on lookup.
PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # OpenRouter (Llama-3-70B-Instruct — commonly-quoted blended rate)
    "meta-llama/llama-3-70b-instruct": {"input": 0.59, "output": 0.79},
    # Short alias accepted by the CLI / runner
    "llama-3-70b": {"input": 0.59, "output": 0.79},
}


def get_rates(model: str) -> dict[str, float]:
    """Return {"input": <$/1M>, "output": <$/1M>} for ``model``.

    Raises KeyError with a helpful message if the model isn't registered —
    we deliberately don't silently return 0.0, because a zero cost would hide
    the fact that a new model was added without a pricing entry.
    """
    key = model.lower()
    if key in PRICING:
        return PRICING[key]
    # Fall back to any case-insensitive match on registered keys.
    for k, v in PRICING.items():
        if k.lower() == key:
            return v
    raise KeyError(
        f"No pricing entry for model {model!r}. "
        f"Add it to benchmarks/external/pricing.py:PRICING "
        f"(known: {sorted(PRICING.keys())})."
    )


def estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    """Return estimated USD cost for one request.

    in_tokens and out_tokens are token counts (not k-tokens). The PRICING
    rates are $/1M, so we divide by 1_000_000.
    """
    rates = get_rates(model)
    return (in_tokens * rates["input"] + out_tokens * rates["output"]) / 1_000_000.0
