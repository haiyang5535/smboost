"""Run raw external-API baselines against the same task sets we use for the
local small-model gate.

CSV shape (matches `benchmarks/gates/runner.py` + adds `cost_usd`)::

    bench, task_id, model, mode, passed, passed_heval, passed_heval_plus,
    retries, latency_s, cost_usd

``mode`` is always ``"raw"`` — harness-mode comparisons for frontier models
aren't part of the YC pitch; the whole point of this runner is "frontier raw
vs 2B + harness". For HumanEval+, ``passed`` mirrors ``passed_heval_plus``
(same convention as `_run_humaneval_raw` in the gate runner).

Supported benches:
- ``humaneval_plus``  — reuses ``benchmarks.humaneval_plus.simple_eval`` /
  ``evalplus_eval`` via the same resolver the gate runner uses.
- ``bfcl_simple`` / ``bfcl_multi`` — reuses
  ``benchmarks.bfcl.runner._format_raw_prompt`` + ``_parse_raw_call`` +
  ``match_function_call``.
- ``gsm8k`` — guarded import; Agent 3 is building the bench module in a
  parallel worktree, so we try/except ImportError and emit a clear skip
  message. After merge, ``benchmarks.gsm8k.runner`` is expected to expose
  ``load_gsm8k_tasks(n) -> list[dict]`` plus ``score_response(task,
  response) -> int``.
"""
from __future__ import annotations

import os
import re
from typing import Any

from .clients import make_client
from .pricing import estimate_cost  # noqa: F401 — re-exported for callers


# ---------------------------------------------------------------------------
# HumanEval+ (code completion)
# ---------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>\n*", re.DOTALL)
_FENCED_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _clean_completion(raw: str) -> str:
    """Strip <think> blocks and unwrap fenced python blocks.

    Same shape as `benchmarks.run_humaneval.clean_completion` — duplicated so
    this module doesn't transitively import `langchain_openai`, which isn't
    needed for a pure external-API run. (Importing `run_humaneval` pulls in
    `from langchain_openai import ChatOpenAI` at module load time.)
    """
    cleaned = _THINK_BLOCK.sub("", raw)
    fenced = _FENCED_CODE.search(cleaned)
    if fenced:
        return fenced.group(1)
    return cleaned


def _evaluate_humaneval_dual(results: list[dict]):
    """Same resolver as the gate runner uses — keeps scoring identical."""
    if os.getenv("SMBOOST_USE_EVALPLUS_SUBSET"):
        from benchmarks.humaneval_plus.evalplus_eval import evaluate_subset
        return evaluate_subset(results)
    try:
        from benchmarks.humaneval_plus import simple_eval
        return simple_eval.evaluate_base_subset(results)
    except ImportError:
        from benchmarks.humaneval_plus.runner import evaluate_dual
        return evaluate_dual(results)


def _run_humaneval_plus(
    tasks: list[dict],
    model: str,
    max_tokens: int,
) -> list[dict]:
    client = make_client(model)

    completions: list[dict] = []
    for t in tasks:
        out = client.complete(
            prompt=t["prompt"], max_tokens=max_tokens, temperature=0.0
        )
        completions.append(
            {
                "task_id": t["task_id"],
                "completion": _clean_completion(out["content"]),
                "latency_s": out["latency_s"],
                "cost_usd": out["cost_usd"],
                "input_tokens": out["input_tokens"],
                "output_tokens": out["output_tokens"],
            }
        )

    eval_out = _evaluate_humaneval_dual(completions)

    by_tid = {c["task_id"]: c for c in completions}
    rows: list[dict] = []
    for r in eval_out.rows:
        raw = by_tid.get(r["task_id"], {})
        rows.append(
            {
                "bench": "humaneval_plus",
                "task_id": r["task_id"],
                "model": model,
                "mode": "raw",
                "passed": r["passed_heval_plus"],
                "passed_heval": r["passed_heval"],
                "passed_heval_plus": r["passed_heval_plus"],
                "retries": 0,
                "latency_s": raw.get("latency_s", 0),
                "cost_usd": raw.get("cost_usd", 0.0),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# BFCL simple / multiple_function (tool calling)
# ---------------------------------------------------------------------------


def _run_bfcl(
    tasks: list[dict],
    model: str,
    max_tokens: int,
    bench_label: str,
) -> list[dict]:
    """Reuses BFCL's raw prompt formatter + call parser + ground-truth match.

    `bench_label` is the final CSV `bench` column (``bfcl_simple`` /
    ``bfcl_multi``); the task's ``category`` field might disagree with the
    user-supplied bench when the loader's upstream naming drifts, so we
    respect the caller's label for CSV-side consistency with the other
    benches.
    """
    from benchmarks.bfcl.runner import (
        _format_raw_prompt,
        _parse_raw_call,
        match_function_call,
    )

    client = make_client(model)
    rows: list[dict] = []
    for t in tasks:
        prompt = _format_raw_prompt(t)
        out = client.complete(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        predicted = _parse_raw_call(out["content"])
        passed = match_function_call(predicted, t.get("ground_truth") or [])
        rows.append(
            {
                "bench": bench_label,
                "task_id": t["task_id"],
                "model": model,
                "mode": "raw",
                "passed": 1 if passed else 0,
                "passed_heval": 0,
                "passed_heval_plus": 0,
                "retries": 0,
                "latency_s": out["latency_s"],
                "cost_usd": out["cost_usd"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# GSM8K (math word problems) — Agent 3 dependency
# ---------------------------------------------------------------------------


def _run_gsm8k(
    tasks: list[dict],
    model: str,
    max_tokens: int,
) -> list[dict]:
    """Best-effort GSM8K run.

    Agent 3 is building `benchmarks.gsm8k.runner` in a parallel worktree.
    We guard the import so this runner is usable before that lands, and
    duck-type the scoring helper: we look for a few common entry-point
    names rather than pinning one.
    """
    try:
        from benchmarks.gsm8k import runner as gsm8k_runner  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "benchmarks.gsm8k is not yet available in this worktree "
            "(Agent 3 is building it in parallel). Re-run after merge, "
            "or pick a different --bench."
        ) from exc

    # Pick whatever scoring fn the gsm8k module ended up exposing. The
    # parallel agent's spec mentions `score_response(task, text) -> int`; we
    # also accept a `grade_answer` alias so a naming nit doesn't block us.
    score_fn = (
        getattr(gsm8k_runner, "score_response", None)
        or getattr(gsm8k_runner, "grade_answer", None)
    )
    if score_fn is None:
        raise ImportError(
            "benchmarks.gsm8k.runner exists but exposes neither "
            "`score_response` nor `grade_answer`. Cannot score."
        )

    client = make_client(model)
    rows: list[dict] = []
    for t in tasks:
        prompt = t.get("prompt") or t.get("question", "")
        out = client.complete(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        passed = score_fn(t, out["content"])
        rows.append(
            {
                "bench": "gsm8k",
                "task_id": t.get("task_id", t.get("id", "?")),
                "model": model,
                "mode": "raw",
                "passed": int(bool(passed)),
                "passed_heval": 0,
                "passed_heval_plus": 0,
                "retries": 0,
                "latency_s": out["latency_s"],
                "cost_usd": out["cost_usd"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_external_baseline(
    bench: str,
    tasks: list[dict],
    model: str,
    max_tokens: int = 1024,
) -> list[dict[str, Any]]:
    """Run ``tasks`` through ``model`` as raw no-harness completions.

    Returns rows in the standard gate CSV shape with an added ``cost_usd``
    column. The caller (``cli.py`` or a notebook) is responsible for
    persisting the rows — we don't write CSV here so unit tests can assert
    on in-memory shape.
    """
    if bench == "humaneval_plus":
        return _run_humaneval_plus(tasks, model, max_tokens)
    if bench == "bfcl_simple":
        return _run_bfcl(tasks, model, max_tokens, bench_label="bfcl_simple")
    if bench == "bfcl_multi":
        return _run_bfcl(tasks, model, max_tokens, bench_label="bfcl_multi")
    if bench == "gsm8k":
        return _run_gsm8k(tasks, model, max_tokens)
    raise ValueError(
        f"unknown bench {bench!r}. "
        "Supported: humaneval_plus, bfcl_simple, bfcl_multi, gsm8k."
    )
