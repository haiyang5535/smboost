from __future__ import annotations
import argparse

from benchmarks.livecodebench.matrix import run_matrix


def main():
    p = argparse.ArgumentParser(description="SMBoost LiveCodeBench ablation runner")
    p.add_argument("mode", choices=["ablation"])
    p.add_argument("--conditions", default="C1,C2,C3,C4")
    p.add_argument("--models", default="qwen3.5:4b,qwen3.5:9b")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-tasks", type=int, default=50)
    p.add_argument("--ollama-concurrency", type=int, default=2)
    args = p.parse_args()

    out = run_matrix(
        conditions=args.conditions.split(","),
        models=args.models.split(","),
        seeds=[int(s) for s in args.seeds.split(",")],
        n_tasks=args.n_tasks,
        ollama_concurrency=args.ollama_concurrency,
    )
    print(f"merged: {out}")


if __name__ == "__main__":
    main()
