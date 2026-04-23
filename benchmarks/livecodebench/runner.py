from __future__ import annotations
import csv
import os
import time
from pathlib import Path
from typing import Callable

from benchmarks.livecodebench.sandbox import run as sandbox_run


HEADER = [
    "condition", "model", "seed", "task_id", "passed", "duration_ms",
    "retries", "fallback_triggered", "grounded_verify_result", "memory_hits",
    "usd_cost", "wall_clock_ms", "max_tokens", "failure_bucket",
]


def _current_max_tokens() -> str:
    for name in ("SMBOOST_OPENAI_MAX_TOKENS", "SMBOOST_LOCAL_MAX_TOKENS"):
        value = os.getenv(name)
        if value:
            return value
    return ""


def _last_generate_output(result) -> str:
    return next(
        (s.output or "" for s in reversed(getattr(result, "trace", [])) if s.node == "generate"),
        "",
    )


def _ground_truth_eval(result, task: dict) -> dict:
    completion = _last_generate_output(result)
    if not completion.strip():
        return {
            "passed": False,
            "traceback": "",
            "stdout": "",
            "stderr": "",
            "duration_ms": 0,
        }
    return sandbox_run(
        completion,
        task.get("test_code", ""),
        task.get("entry_point", ""),
    )


def _classify_failure(*, passed: bool, traceback: str) -> str:
    if passed:
        return "PASS"
    if not traceback:
        return "no_verify_or_empty_generate"
    if "got ''" in traceback:
        return "empty_output"
    if "SyntaxError" in traceback or "IndentationError" in traceback:
        return "syntax_truncation"
    if "OutputMismatch" in traceback or "Output mismatch" in traceback or "AssertionError" in traceback:
        return "logic_or_output_mismatch"
    return "other_runtime"


def _already_done(shard_path: Path) -> set[str]:
    if not shard_path.exists():
        return set()
    done = set()
    with open(shard_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["task_id"])
    return done


def run_one_cell(
    condition_name: str,
    agent_factory: Callable,
    tasks: list[dict],
    *,
    model: str,
    seed: int,
    shard_path: Path,
    memory_log_path: Path | None,
) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    done = _already_done(shard_path)
    write_header = not shard_path.exists() or shard_path.stat().st_size == 0

    agent = agent_factory(model, seed)
    if memory_log_path is not None:
        agent.set_memory_log(memory_log_path)
    try:
        with open(shard_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            if write_header:
                writer.writeheader()
            for task in tasks:
                if task["task_id"] in done:
                    continue
                pre_hits = agent._memory.hits if (agent._memory is not None) else 0
                start = time.monotonic()
                result = agent.run(task["prompt"], task_metadata=task)
                wall = int((time.monotonic() - start) * 1000)
                post_hits = agent._memory.hits if (agent._memory is not None) else 0
                hits_this_task = post_hits - pre_hits
                last_verify = next(
                    (s for s in reversed(result.trace) if s.node == "verify"), None,
                ) if result.trace else None
                gv = (last_verify.output[:120] if last_verify else "").replace("\n", " ")
                eval_result = _ground_truth_eval(result, task)
                passed = 1 if eval_result["passed"] else 0
                writer.writerow({
                    "condition": condition_name,
                    "model": model,
                    "seed": seed,
                    "task_id": task["task_id"],
                    "passed": passed,
                    "duration_ms": int(result.stats.total_latency_s * 1000),
                    "retries": result.stats.retry_count,
                    "fallback_triggered": result.stats.fallback_triggers,
                    "grounded_verify_result": gv,
                    "memory_hits": hits_this_task,
                    "usd_cost": 0.0,
                    "wall_clock_ms": wall,
                    "max_tokens": _current_max_tokens(),
                    "failure_bucket": _classify_failure(
                        passed=bool(passed),
                        traceback=eval_result.get("traceback", ""),
                    ),
                })
                f.flush()
    finally:
        agent.close_memory()
