"""Day-8 canary: run C1 vs C4 on 10 LCB Hard tasks, both models, 1 seed.
Abort sprint if C1 does not beat C4 by >=5pp on qwen3.5:9b."""
from __future__ import annotations
import csv
from pathlib import Path

from benchmarks.livecodebench.matrix import run_matrix


def _pass_rate(csv_path: Path, condition: str, model: str) -> float:
    passed, total = 0, 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["condition"] == condition and row["model"] == model:
                total += 1
                passed += int(row["passed"])
    return passed / total if total else 0.0


def main():
    out = run_matrix(
        conditions=["C1", "C4"],
        models=["qwen3.5:4b", "qwen3.5:9b"],
        seeds=[0],
        n_tasks=10,
        ollama_concurrency=2,
    )
    c1_9b = _pass_rate(out, "C1", "qwen3.5:9b")
    c4_9b = _pass_rate(out, "C4", "qwen3.5:9b")
    gap = c1_9b - c4_9b
    print(f"9b: C1={c1_9b:.2%}  C4={c4_9b:.2%}  gap={gap:+.2%}")
    if gap < 0.05:
        raise SystemExit(
            f"CANARY FAIL: C1 only beats C4 by {gap:+.2%} on 9b (<5pp). "
            "Investigate grounded verify + memory wiring before burning full matrix."
        )
    print("canary ok")


if __name__ == "__main__":
    main()
