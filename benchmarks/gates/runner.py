"""Execute one gate: load N tasks for a benchmark, run them under each
(model, condition) config, return flat row list suitable for the criteria
module.

HumanEval+ scoring path:
  - If env var `SMBOOST_USE_EVALPLUS_SUBSET=1` is set, use the real
    HumanEval+ scorer at
    `benchmarks.humaneval_plus.evalplus_eval.evaluate_subset`, which pads
    the submission to 164 problems internally and gives correct per-task
    `passed_heval_plus` values from the ~80x plus-tests.
  - Otherwise prefer the subprocess-based
    `benchmarks.humaneval_plus.simple_eval.evaluate_base_subset` (base
    HumanEval only; avoids evalplus multiprocessing issues on some
    Python/macOS configurations). `passed_heval_plus` mirrors `passed_heval`.
  - Fall back to `runner.evaluate_dual` if `simple_eval` isn't importable;
    this path assumes the caller has submitted completions for all 164
    HumanEval problems.
See `docs/superpowers/research/2026-04-23-evalplus-he-plus-fix.md`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _evaluate_humaneval_dual(results: list[dict]):
    """Resolve the HumanEval+ evaluator at call-time so tests can patch the
    underlying `simple_eval.evaluate_base_subset` (or `runner.evaluate_dual`)
    at their source module. Selection order matches the module docstring."""
    if os.getenv("SMBOOST_USE_EVALPLUS_SUBSET"):
        from benchmarks.humaneval_plus.evalplus_eval import evaluate_subset
        return evaluate_subset(results)
    try:
        from benchmarks.humaneval_plus import simple_eval
        return simple_eval.evaluate_base_subset(results)
    except ImportError:
        from benchmarks.humaneval_plus.runner import evaluate_dual
        return evaluate_dual(results)


@dataclass
class GateConfig:
    name: str
    bench: str                              # "humaneval_plus" | "bfcl_simple" | "bfcl_multi"
    n: int
    configs: list[tuple[str, str]] = field(default_factory=list)  # [(model, "raw"|"C1"|...)]


def _load_tasks_for_bench(bench: str, n: int) -> list[dict]:
    if bench == "humaneval_plus":
        from benchmarks.tasks import load_humaneval_tasks
        return load_humaneval_tasks(n)
    if bench == "bfcl_simple":
        from benchmarks.bfcl.loader import load_bfcl_tasks
        return load_bfcl_tasks(category="simple", n=n)
    if bench == "bfcl_multi":
        from benchmarks.bfcl.loader import load_bfcl_tasks
        return load_bfcl_tasks(category="multiple_function", n=n)
    raise ValueError(f"unknown bench: {bench!r}")


def _run_humaneval_raw(tasks: list[dict], model: str) -> list[dict]:
    from benchmarks.run_humaneval import run_baseline

    results = run_baseline(tasks, model)
    eval_out = _evaluate_humaneval_dual(results)
    return [
        {
            "task_id": r["task_id"],
            "model": model,
            "mode": "raw",
            "passed": r["passed_heval_plus"],
            "passed_heval": r["passed_heval"],
            "passed_heval_plus": r["passed_heval_plus"],
            "retries": 0,
            "latency_s": next(
                (x["latency_s"] for x in results if x["task_id"] == r["task_id"]), 0
            ),
            "bench": "humaneval_plus",
        }
        for r in eval_out.rows
    ]


def _run_humaneval_harness(
    tasks: list[dict], condition: str, model: str
) -> list[dict]:
    """Run through a C1-C6 harness condition. Returns row dicts (HE+ scored).

    The task_metadata dict plumbed into ``agent.run(...)`` uses
    ``testtype="humaneval"`` which routes through
    ``src/smboost/tasks/completion.py::_verify_grounded``'s HumanEval branch:
    it re-runs ``prompt + completion + test + check(entry_point)`` in a
    subprocess so the verify step matches the final ground-truth pass.
    Without this, ``task_metadata`` was empty and verify degraded to
    ``_verify_ast_only`` (always passes on a first-shot generation), making
    the whole harness inert on HumanEval gate data.
    """
    from benchmarks.conditions import build_condition
    from benchmarks.run_humaneval import clean_completion

    results: list[dict] = []
    for t in tasks:
        agent = build_condition(
            condition=condition, model=model, task_graph_kind="completion"
        )
        task_metadata = {
            "testtype": "humaneval",
            "task_id": t["task_id"],
            "entry_point": t.get("entry_point", ""),
            "prompt": t["prompt"],
            "test": t.get("test", ""),
        }
        run = agent.run(
            t["prompt"],
            task_metadata=task_metadata,
            task_id=t["task_id"],
            condition=condition,
        )
        generate_output = ""
        for step in run.trace:
            if step.node == "generate":
                generate_output = step.output
        results.append(
            {
                "task_id": t["task_id"],
                "completion": clean_completion(generate_output),
                "latency_s": run.stats.total_latency_s,
                "retries": run.stats.retry_count,
            }
        )
    eval_out = _evaluate_humaneval_dual(results)
    return [
        {
            "task_id": r["task_id"],
            "model": model,
            "mode": condition,
            "passed": r["passed_heval_plus"],
            "passed_heval": r["passed_heval"],
            "passed_heval_plus": r["passed_heval_plus"],
            "retries": next((x["retries"] for x in results if x["task_id"] == r["task_id"]), 0),
            "latency_s": next((x["latency_s"] for x in results if x["task_id"] == r["task_id"]), 0),
            "bench": "humaneval_plus",
        }
        for r in eval_out.rows
    ]


def _run_bfcl_raw_fn(tasks: list[dict], model: str) -> list[dict]:
    from benchmarks.bfcl.runner import run_bfcl_raw

    return [{**r, "bench": f"bfcl_{r['category']}"} for r in run_bfcl_raw(tasks, model)]


def _run_bfcl_harness_fn(tasks: list[dict], condition: str, model: str) -> list[dict]:
    from benchmarks.bfcl.runner import run_bfcl_harness

    return [
        {**r, "bench": f"bfcl_{r['category']}"}
        for r in run_bfcl_harness(tasks, condition, model)
    ]


def run_gate(cfg: GateConfig) -> list[dict]:
    tasks = _load_tasks_for_bench(cfg.bench, cfg.n)
    rows: list[dict] = []
    for model, mode in cfg.configs:
        if cfg.bench == "humaneval_plus":
            if mode == "raw":
                rows.extend(_run_humaneval_raw(tasks, model))
            else:
                rows.extend(_run_humaneval_harness(tasks, mode, model))
        elif cfg.bench.startswith("bfcl_"):
            if mode == "raw":
                rows.extend(_run_bfcl_raw_fn(tasks, model))
            else:
                rows.extend(_run_bfcl_harness_fn(tasks, mode, model))
        else:
            raise ValueError(f"unknown bench in GateConfig: {cfg.bench!r}")
    return rows
