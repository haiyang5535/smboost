"""GSM8K raw-mode runner.

Mirrors ``benchmarks/humaneval_plus/runner.py`` / ``benchmarks/bfcl/runner.py``
structure: hits the OpenAI-compatible server via ``_CompatibleChatOpenAI``
(keeps ``max_tokens`` working on llama.cpp).

Harness modes (C1/C4/C5/C6) are dispatched from
``benchmarks/gates/runner.py`` through the shared ``build_condition`` path
with ``task_graph_kind="completion"`` and ``task_metadata`` carrying
``{"testtype": "gsm8k", "expected_answer": "..."}`` for grounded-verify
hooks.
"""
from __future__ import annotations

import os
import time
from typing import Any

from langchain_core.messages import HumanMessage

from benchmarks.gsm8k.prompt import build_prompt
from benchmarks.gsm8k.scorer import score


def _make_raw_llm(model: str, *, temperature: float, max_tokens: int):
    """Indirection so tests can patch. Uses the ``_CompatibleChatOpenAI``
    subclass so ``max_tokens`` survives langchain-openai 1.1.x payload
    rewriting against llama.cpp.
    """
    from smboost.llm.runtime import _CompatibleChatOpenAI

    return _CompatibleChatOpenAI(
        model=model,
        base_url=os.environ.get("SMBOOST_OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("SMBOOST_OPENAI_API_KEY", "sk-no-key"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def run_baseline(
    tasks: list[dict[str, Any]],
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> list[dict[str, Any]]:
    """Run each task through raw ChatOpenAI. Returns row dicts.

    Each row carries:
        - ``task_id``, ``model``, ``mode="raw"``
        - ``completion``: raw text the model produced
        - ``expected_answer``: canonical integer string from the dataset
        - ``passed``: 1 iff scorer matched; else 0
        - ``retries`` (0 for raw), ``latency_s``
        - ``bench="gsm8k"``, ``failure_bucket`` hint

    Callers that want harness modes should go through
    ``benchmarks.gates.runner`` which builds a ``HarnessAgent``.
    """
    llm = _make_raw_llm(model, temperature=temperature, max_tokens=max_tokens)
    rows: list[dict[str, Any]] = []
    for t in tasks:
        prompt = build_prompt(t["question"])
        start = time.monotonic()
        raw = llm.invoke([HumanMessage(content=prompt)]).content or ""
        latency_s = round(time.monotonic() - start, 3)
        passed = score(raw, t["expected_answer"])
        rows.append(
            {
                "task_id": t["task_id"],
                "model": model,
                "mode": "raw",
                "completion": raw,
                "expected_answer": t["expected_answer"],
                "passed": 1 if passed else 0,
                # HE+ columns kept as 0 so the shared CSV schema stays aligned;
                # downstream aggregators key off bench="gsm8k" rather than
                # these fields.
                "passed_heval": 0,
                "passed_heval_plus": 0,
                "retries": 0,
                "latency_s": latency_s,
                "bench": "gsm8k",
                "failure_bucket": "PASS" if passed else (
                    "no_numeric_answer" if not raw.strip() else "wrong_answer"
                ),
            }
        )
    return rows
