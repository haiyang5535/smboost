"""Run one GSM8K-style task through the C5 self-consistency harness.

Prerequisites: a running OpenAI-compatible server on
``SMBOOST_OPENAI_BASE_URL`` (default ``http://127.0.0.1:8000/v1``).
The Quickstart in the README sets up a llama.cpp server with Qwen 2.5
2B Q4_K_M; any model the server exposes works as long as the env vars
point at it.

Usage:
    python3 examples/single_task.py

What this script demonstrates:
    1. Constructing a C5 (self-consistency) HarnessAgent via the
       same ``build_condition`` factory the benchmark uses.
    2. Calling ``agent.run(...)`` on a single math word problem.
    3. Inspecting the trace, the final answer, and the per-run stats
       (retry count, total wall-clock latency).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the in-repo package importable when running from a source checkout.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default to the same server URL the README Quickstart starts.
os.environ.setdefault("SMBOOST_LLM_BACKEND", "server")
os.environ.setdefault("SMBOOST_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
os.environ.setdefault("SMBOOST_OPENAI_API_KEY", "sk-no-key")
os.environ.setdefault("SMBOOST_OPENAI_MAX_TOKENS", "512")

from benchmarks.conditions import build_condition  # noqa: E402
from benchmarks.gsm8k.prompt import build_prompt  # noqa: E402

QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she "
    "sold half as many clips in May. How many clips did Natalia sell "
    "altogether in April and May?"
)
EXPECTED_ANSWER = "72"


def main() -> int:
    agent = build_condition(
        condition="C5",
        model="qwen3.5:2b",
        task_graph_kind="completion",
        bench="gsm8k",
    )

    result = agent.run(
        build_prompt(QUESTION),
        task_metadata={
            "testtype": "gsm8k",
            "task_id": "demo/0",
            "question": QUESTION,
            "expected_answer": EXPECTED_ANSWER,
        },
        task_id="demo/0",
        condition="C5",
    )

    print("=" * 60)
    print("Question:", QUESTION)
    print("Expected:", EXPECTED_ANSWER)
    print("=" * 60)

    # Pull the model's final generate output from the trace.
    final_generate = ""
    for step in result.trace:
        if step.node == "generate":
            final_generate = step.output
    print("Generate output (truncated):")
    print(final_generate[:600])

    print("=" * 60)
    print(f"Retries: {result.stats.retry_count}")
    print(f"Total wall-clock latency: {result.stats.total_latency_s:.2f}s")
    print(f"Status: {result.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
