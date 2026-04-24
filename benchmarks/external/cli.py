"""CLI entrypoint for external-API baseline runs.

Example::

    python3 -m benchmarks.external.cli \\
        --model claude-sonnet-4-6 \\
        --bench humaneval_plus \\
        --n 10 \\
        --out-csv /tmp/ext_smoke.csv

Cost safety: external API calls cost real money. Default ``--n`` is 10; runs
above 30 require ``--confirm-cost`` so an overnight fat-finger doesn't
silently burn $100. We print an estimated cost ceiling before starting
based on a rough input+output token budget per task.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .pricing import estimate_cost
from .runner import run_external_baseline


CSV_FIELDS = [
    "bench",
    "task_id",
    "model",
    "mode",
    "passed",
    "passed_heval",
    "passed_heval_plus",
    "retries",
    "latency_s",
    "cost_usd",
]


def _load_tasks(bench: str, n: int) -> list[dict]:
    """Same task-loading contract as `benchmarks.gates.runner._load_tasks_for_bench`.

    gsm8k is guarded because Agent 3's module may not exist in this worktree
    yet.
    """
    if bench == "humaneval_plus":
        from benchmarks.tasks import load_humaneval_tasks
        return load_humaneval_tasks(n)
    if bench == "bfcl_simple":
        from benchmarks.bfcl.loader import load_bfcl_tasks
        return load_bfcl_tasks(category="simple", n=n)
    if bench == "bfcl_multi":
        from benchmarks.bfcl.loader import load_bfcl_tasks
        return load_bfcl_tasks(category="multiple_function", n=n)
    if bench == "gsm8k":
        try:
            from benchmarks.gsm8k.loader import load_gsm8k_tasks  # type: ignore[import-not-found]
            return load_gsm8k_tasks(n)
        except ImportError:
            try:
                from benchmarks.gsm8k import runner as gsm8k_runner  # type: ignore[import-not-found]
                loader = getattr(gsm8k_runner, "load_gsm8k_tasks", None)
                if loader is None:
                    raise ImportError
                return loader(n)
            except ImportError as exc:
                raise SystemExit(
                    "benchmarks.gsm8k is not yet available in this worktree "
                    "(Agent 3 dependency). Pick --bench humaneval_plus, "
                    "bfcl_simple, or bfcl_multi."
                ) from exc
    raise ValueError(f"unknown bench {bench!r}")


def _estimate_run_cost(model: str, n: int, max_tokens: int) -> float:
    """Rough $ ceiling for a run.

    Assume ~1K input tokens (prompt + few-shot header) and ``max_tokens``
    output tokens per task — intentionally pessimistic so the number printed
    to the user is an upper bound, not a lower bound.
    """
    per_task = estimate_cost(model, in_tokens=1000, out_tokens=max_tokens)
    return per_task * n


def _write_rows(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmarks.external.cli",
        description=(
            "Run a raw external-API baseline (Claude Sonnet 4.6 / GPT-4o / "
            "Llama-3-70B via OpenRouter) against SMBoost's gate task sets, "
            "producing CSV rows in the same shape as the local gate runner."
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        help=(
            "Model id. claude-sonnet-4-6, gpt-4o, gpt-4o-mini, or "
            "meta-llama/llama-3-70b-instruct (short alias: llama-3-70b)."
        ),
    )
    p.add_argument(
        "--bench",
        required=True,
        choices=["humaneval_plus", "bfcl_simple", "bfcl_multi", "gsm8k"],
        help="Which task set to run.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of tasks (default 10). Above 30 requires --confirm-cost.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Per-request max_tokens ceiling (default 1024).",
    )
    p.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path (appends if the file already has a header).",
    )
    p.add_argument(
        "--confirm-cost",
        action="store_true",
        help="Required when --n > 30 to acknowledge real-$ spend.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.n > 30 and not args.confirm_cost:
        print(
            f"[external.cli] refusing to run n={args.n} without --confirm-cost. "
            "Re-run with --confirm-cost to acknowledge real-$ spend.",
            file=sys.stderr,
        )
        return 2

    ceiling = _estimate_run_cost(args.model, args.n, args.max_tokens)
    print(
        f"[external.cli] model={args.model} bench={args.bench} n={args.n} "
        f"max_tokens={args.max_tokens}  ~upper-bound cost ${ceiling:.4f}"
    )

    tasks = _load_tasks(args.bench, args.n)
    rows = run_external_baseline(
        bench=args.bench,
        tasks=tasks,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    out_path = Path(args.out_csv)
    _write_rows(rows, out_path)

    passed = sum(int(r.get("passed", 0)) for r in rows)
    total_cost = sum(float(r.get("cost_usd", 0.0)) for r in rows)
    print(
        f"[external.cli] wrote {len(rows)} rows to {out_path}  "
        f"passed={passed}/{len(rows)}  total_cost=${total_cost:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
