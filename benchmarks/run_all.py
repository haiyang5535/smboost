#!/usr/bin/env python3
"""Run baseline, smboost, and upper_bound for all specified model sizes."""
from __future__ import annotations
import argparse
import subprocess
import sys


DEFAULT_MODELS = ["qwen3.5:0.8b", "qwen3.5:4b", "qwen3.5:9b"]
DEFAULT_MODES = ["baseline", "smboost", "upper_bound"]


def run_one(model: str, mode: str, n_tasks: int, out: str) -> bool:
    """Run a single model/mode combination. Returns True on success."""
    cmd = [
        sys.executable, "-m", "benchmarks.run_humaneval",
        "--model", model,
        "--mode", mode,
        "--n-tasks", str(n_tasks),
        "--out", out,
    ]
    print(f"\n{'='*60}")
    print(f"  model={model}  mode={mode}  n_tasks={n_tasks}")
    print(f"{'='*60}")
    return subprocess.run(cmd).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full multi-model SMBoost benchmark suite"
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        metavar="MODEL",
        help=f"Ollama model tags to benchmark (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        choices=DEFAULT_MODES,
        help="Modes to run (default: all three)",
    )
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument(
        "--out", default="benchmarks/results/results.csv",
        help="CSV output path (appended, same file for all runs)",
    )
    args = parser.parse_args()

    total = len(args.models) * len(args.modes)
    done = 0
    failed: list[str] = []

    for model in args.models:
        for mode in args.modes:
            done += 1
            print(f"\n[{done}/{total}]", end=" ")
            ok = run_one(model, mode, args.n_tasks, args.out)
            if not ok:
                failed.append(f"{model}/{mode}")

    print(f"\n{'='*60}")
    print(f"Complete: {done - len(failed)}/{total} succeeded")
    if failed:
        print(f"Failed runs: {', '.join(failed)}")
        sys.exit(1)
    print("All runs succeeded!")
    print(f"\nNext step — export to dashboard:")
    print(f"  python benchmarks/export_results.py --n-tasks {args.n_tasks}")


if __name__ == "__main__":
    main()
