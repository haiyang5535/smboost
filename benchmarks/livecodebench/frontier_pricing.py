from __future__ import annotations

# USD per 1M tokens, snapshotted 2026-04-19. Update Day 12 before frontier runs.
_PRICES = {
    "gpt-4o":           {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":      {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-5":  {"input": 3.00, "output": 15.00},
    "claude-opus-4-5":    {"input": 15.00, "output": 75.00},
}


def calc_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    p = _PRICES[model_name]
    return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]
