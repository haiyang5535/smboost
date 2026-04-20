from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Callable


HEADER = [
    "condition", "model", "seed", "task_id", "passed", "duration_ms",
    "retries", "fallback_triggered", "grounded_verify_result", "memory_hits",
    "usd_cost", "wall_clock_ms",
]


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
                passed = 1 if result.status == "success" else 0
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
                })
                f.flush()
    finally:
        agent.close_memory()
