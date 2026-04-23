"""BFCL raw + harness runners with AST-free call matching.

Raw mode: ChatOpenAI against the llama.cpp server, expects a JSON object like
    {"name": "fn", "arguments": {"arg1": "...", ...}}

Harness mode: uses build_condition to build a HarnessAgent with a
ToolCallingTaskGraph bound to the task's function list.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage

from benchmarks.conditions import build_condition


_JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)


def match_function_call(
    predicted: dict[str, Any] | None,
    ground_truth: list[dict[str, Any]] | None,
) -> bool:
    """True if predicted exactly matches any ground-truth call (name + args)."""
    if not predicted or not ground_truth:
        return False
    for gt in ground_truth:
        if predicted.get("name") != gt.get("name"):
            continue
        pa = predicted.get("arguments") or {}
        ga = gt.get("arguments") or {}
        # String-compare after JSON-ifying for order-stable equality
        if json.dumps(pa, sort_keys=True) == json.dumps(ga, sort_keys=True):
            return True
    return False


def _parse_raw_call(raw: str) -> dict[str, Any] | None:
    """Extract the first JSON object from raw LLM output; return None if unparseable."""
    if not raw:
        return None
    match = _JSON_BLOB.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _make_raw_llm(model: str):
    """Indirection for tests to patch.

    Uses the same env-var-aware factory as the harness path
    (SMBOOST_OPENAI_BASE_URL / _API_KEY / SMBOOST_LLM_BACKEND), so a BFCL raw
    run on a non-default port doesn't silently hit `localhost:8000`.
    """
    from smboost.llm.runtime import get_default_llm_factory

    return get_default_llm_factory()(model)


def _format_raw_prompt(task: dict[str, Any]) -> str:
    fns = json.dumps(task["functions"], indent=2)
    return (
        "Given these available functions:\n"
        f"{fns}\n\n"
        f"Answer the user's question by emitting a single JSON object of the form "
        f'{{"name": "<fn>", "arguments": {{...}}}}. No prose.\n\n'
        f"Question: {task['question']}"
    )


def run_bfcl_raw(tasks: list[dict], model: str) -> list[dict]:
    """Run each task through raw ChatOpenAI. Returns row dicts."""
    llm = _make_raw_llm(model)
    rows: list[dict] = []
    for t in tasks:
        start = time.monotonic()
        raw = llm.invoke([HumanMessage(content=_format_raw_prompt(t))]).content or ""
        latency_s = round(time.monotonic() - start, 3)
        predicted = _parse_raw_call(raw)
        passed = match_function_call(predicted, t.get("ground_truth") or [])
        rows.append(
            {
                "task_id": t["task_id"],
                "category": t["category"],
                "mode": "raw",
                "model": model,
                "predicted": predicted,
                "passed": 1 if passed else 0,
                "retries": 0,
                "latency_s": latency_s,
                "failure_bucket": "PASS" if passed
                else ("malformed_output" if predicted is None else "wrong_call"),
                "raw_output": raw[:500],
            }
        )
    return rows


def run_bfcl_harness(
    tasks: list[dict], condition: str, model: str
) -> list[dict]:
    """Run each task through build_condition(condition, tool_calling). Returns row dicts.

    Note: BFCL's function list is per-task, so we build one agent per task (cheap —
    HarnessAgent init is just graph construction; the LLM backend is shared).
    """
    rows: list[dict] = []
    for t in tasks:
        agent = build_condition(
            condition=condition,
            model=model,
            task_graph_kind="tool_calling",
            tools=t["functions"],  # BFCL functions already in tool-schema shape
        )
        result = agent.run(t["question"])
        predicted = _parse_raw_call(result.output or "")
        passed = match_function_call(predicted, t.get("ground_truth") or [])
        rows.append(
            {
                "task_id": t["task_id"],
                "category": t["category"],
                "mode": condition,
                "model": model,
                "predicted": predicted,
                "passed": 1 if passed else 0,
                "retries": result.stats.retry_count,
                "latency_s": result.stats.total_latency_s,
                "failure_bucket": "PASS" if passed
                else ("malformed_output" if predicted is None else "wrong_call"),
            }
        )
    return rows
