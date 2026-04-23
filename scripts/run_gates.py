"""CLI orchestrator for G1 -> G2 -> G3 -> G4.

Usage:
    # G1 + G2 (HumanEval+) — Day 1. Operator swaps server between 2B and 9B as prompted.
    python3 scripts/run_gates.py --stage G1 --out-csv benchmarks/results/gate_g1.csv
    python3 scripts/run_gates.py --stage G2 --out-csv benchmarks/results/gate_g2.csv

    # G3 (BFCL) — Day 2 AM
    python3 scripts/run_gates.py --stage G3 --out-csv benchmarks/results/gate_g3.csv

    # G4 (widen to 50) — Day 2 PM
    python3 scripts/run_gates.py --stage G4 --out-csv benchmarks/results/gate_g4.csv

    # All four stages + final report
    python3 scripts/run_gates.py --stage all --report-date 2026-04-24
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.gates.runner import run_gate, GateConfig
from benchmarks.gates.criteria import (
    evaluate_g1, evaluate_g2, evaluate_g3, evaluate_g4, GateResult,
)
from benchmarks.gates.report import render_gate_report


def _row_header() -> list[str]:
    return [
        "bench", "task_id", "model", "mode", "passed",
        "passed_heval", "passed_heval_plus",
        "retries", "latency_s",
    ]


def _append_rows(out_csv: Path, rows: list[dict]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    header = _row_header()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _cast_rows(rows: list[dict]) -> list[dict]:
    """CSV returns strings; convert passed/retries/latency to numbers."""
    out = []
    for r in rows:
        r = dict(r)
        r["passed"] = int(r.get("passed", 0) or 0)
        r["retries"] = int(r.get("retries", 0) or 0)
        try:
            r["latency_s"] = float(r.get("latency_s", 0) or 0.0)
        except ValueError:
            r["latency_s"] = 0.0
        out.append(r)
    return out


STAGE_SPECS = {
    "G1": GateConfig(
        name="G1_capability_floor",
        bench="humaneval_plus",
        n=20,
        configs=[("qwen3.5:2b", "raw"), ("qwen3.5:9b", "raw")],
    ),
    "G2": GateConfig(
        name="G2_harness_lift",
        bench="humaneval_plus",
        n=20,
        configs=[("qwen3.5:2b", "C1"), ("qwen3.5:2b", "C4")],
    ),
    "G3": GateConfig(
        name="G3_bfcl_sanity",
        bench="bfcl_simple",
        n=30,
        configs=[("qwen3.5:2b", "raw"), ("qwen3.5:2b", "C1")],
    ),
    "G4": GateConfig(
        # G4 reuses G1+G2 rows for the first 20 tasks; this stage ADDS tasks 20..49.
        name="G4_widen_confidence",
        bench="humaneval_plus",
        n=50,
        configs=[
            ("qwen3.5:2b", "raw"),
            ("qwen3.5:2b", "C1"),
            ("qwen3.5:9b", "raw"),
        ],
    ),
}


def run_stage(stage: str, out_csv: Path, model_filter: str | None = None) -> list[dict]:
    cfg = STAGE_SPECS[stage]
    if model_filter:
        filtered = [c for c in cfg.configs if c[0] == model_filter]
        if not filtered:
            print(f"[{stage}] no configs match model_filter={model_filter!r}; "
                  f"all configs: {cfg.configs}", flush=True)
            return []
        cfg = GateConfig(name=cfg.name, bench=cfg.bench, n=cfg.n, configs=filtered)
    print(f"[{stage}] running {cfg.bench} n={cfg.n} configs={cfg.configs}", flush=True)
    rows = run_gate(cfg)
    _append_rows(out_csv, rows)
    print(f"[{stage}] wrote {len(rows)} rows -> {out_csv}", flush=True)
    return rows


def evaluate_stage(stage: str, rows: list[dict]) -> GateResult:
    rows = _cast_rows(rows)
    if stage == "G1":
        return evaluate_g1(rows)
    if stage == "G2":
        # G2 criterion needs G1's raw rows too for the lift comparison
        return evaluate_g2(rows)
    if stage == "G3":
        return evaluate_g3(rows)
    if stage == "G4":
        return evaluate_g4(rows)
    raise ValueError(f"unknown stage: {stage!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["G1", "G2", "G3", "G4", "all"])
    p.add_argument("--out-csv", default=None,
                   help="Path to write raw rows; default benchmarks/results/gate_<STAGE>.csv")
    p.add_argument("--report-date", default=date.today().isoformat())
    p.add_argument(
        "--report-out",
        default=None,
        help="Markdown report path; default docs/overnight/<DATE>-gate-results.md",
    )
    p.add_argument(
        "--model-filter", default=None,
        help="Only run the configs matching this model (e.g. qwen3.5:2b or qwen3.5:9b). "
             "Used when the operator needs to swap servers between models — run the stage "
             "once per model with the matching GGUF loaded.",
    )
    args = p.parse_args()

    report_out = Path(args.report_out) if args.report_out \
        else Path(f"docs/overnight/{args.report_date}-gate-results.md")

    results: list[GateResult] = []

    stages = ["G1", "G2", "G3", "G4"] if args.stage == "all" else [args.stage]

    for stage in stages:
        out_csv = Path(args.out_csv) if (args.out_csv and args.stage != "all") \
            else Path(f"benchmarks/results/gate_{stage}.csv")

        rows = run_stage(stage, out_csv, model_filter=args.model_filter)

        if args.model_filter:
            # Partial run (operator is splitting G1/G4 across server swaps).
            # Skip criterion evaluation on partial data; defer to a later full evaluation.
            print(f"[{stage}] partial run with --model-filter={args.model_filter!r}; "
                  "skipping criterion evaluation until all configs have been run",
                  flush=True)
            continue

        if stage == "G2":
            # G2 criterion needs G1's raw rows for the lift ratio
            g1_rows = _load_csv(Path("benchmarks/results/gate_G1.csv"))
            combined = rows + g1_rows
            result = evaluate_stage(stage, combined)
        elif stage == "G4":
            # G4 reuses G1+G2 and adds new tasks; combine
            g1_rows = _load_csv(Path("benchmarks/results/gate_G1.csv"))
            g2_rows = _load_csv(Path("benchmarks/results/gate_G2.csv"))
            combined = rows + g1_rows + g2_rows
            result = evaluate_stage(stage, combined)
        else:
            result = evaluate_stage(stage, rows)

        results.append(result)
        verdict = "PASS" if result.passed else "FAIL"
        print(f"[{stage}] {verdict}  metrics={result.metrics}  failures={result.failed_checks}",
              flush=True)

        if not result.passed:
            print(f"[{stage}] FAILED — stopping orchestrator; see pivot options in report", flush=True)
            break

    md = render_gate_report(results, run_date=args.report_date)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(md)
    print(f"Report: {report_out}", flush=True)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
