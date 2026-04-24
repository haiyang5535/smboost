"""Prompt templates for GSM8K.

Small-model CoT-inducing format: elicit step-by-step reasoning and require
the final numeric answer on a trailing ``#### <integer>`` line so the scorer
has a stable anchor.
"""
from __future__ import annotations


_COT_TEMPLATE = (
    "Solve the following math word problem.\n"
    "Think step by step, then state the final numeric answer on a new line "
    "prefixed with '#### ' (four hash marks, one space, then the integer).\n"
    "\n"
    "Q: {question}\n"
    "A: Let's think step by step. "
)


def build_prompt(question: str) -> str:
    """Return the CoT-inducing prompt string for a GSM8K question."""
    return _COT_TEMPLATE.format(question=question)


def raw_question(task: dict) -> str:
    """Expose the raw question for callers doing guided-decoding.

    Agent 1 (GUIDED_DECODING) will take the raw question and pair it with a
    grammar forcing "#### <integer>" at the end. Keeping this helper keeps
    the interface stable.
    """
    return task["question"]
