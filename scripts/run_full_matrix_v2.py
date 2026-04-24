"""Post-merge full-matrix runner for YC sprint (P2+ plan).

Runs the full model × condition × bench matrix and produces a consolidated
CSV plus a Markdown evaluation report (G1a-G5a verdicts).

Designed to be crash-safe (resume by re-running — it skips cells already
present in the target CSV) and tolerant of missing modules (benches /
conditions from sub-agents may not all be merged at runtime).

Example:
  python3 scripts/run_full_matrix_v2.py \\
      --out-csv benchmarks/results/full_matrix_v2.csv \\
      --models qwen3.5:2b,qwen3.5:0.8b \\
      --modes raw,C1,C5,C6 \\
      --benches gsm8k,bfcl_simple,humaneval_plus \\
      --n 30
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HEADER = [
    "bench", "task_id", "model", "mode", "passed",
    "passed_heval", "passed_heval_plus",
    "retries", "latency_s",
    "cost_usd",
]


def _existing_cells(csv_path: Path) -> set[tuple[str, str, str]]:
    """Return the (bench, model, mode) triples already present in the CSV."""
    if not csv_path.exists():
        return set()
    seen: set[tuple[str, str, str]] = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            seen.add((row.get("bench", ""), row.get("model", ""), row.get("mode", "")))
    return seen


def _merge_csv(src_rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in src_rows:
            w.writerow(r)


def _read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _server_base_url_for(model: str) -> str:
    """Map a model id to its llama.cpp server base_url.

    Expects 2B on :8000 and 0.8B on :8001 (see CLAUDE.md).
    Override via SMBOOST_OPENAI_BASE_URL if set.
    """
    if os.environ.get("SMBOOST_OPENAI_BASE_URL"):
        return os.environ["SMBOOST_OPENAI_BASE_URL"]
    if "0.8b" in model.lower():
        return "http://127.0.0.1:8001/v1"
    return "http://127.0.0.1:8000/v1"


def run_local_cell(bench: str, model: str, mode: str, n: int,
                   tmp_csv: Path) -> list[dict]:
    """Run one (bench, model, mode) cell via scripts/run_gates_0_8b.py.

    Returns the freshly written rows.
    """
    if bench == "humaneval_plus":
        stage = "HE"
    elif bench == "bfcl_simple":
        stage = "BFCL"
    elif bench == "gsm8k":
        stage = "GSM8K"
    else:
        raise ValueError(f"unknown bench {bench!r}")

    env = dict(os.environ)
    env["SMBOOST_LLM_BACKEND"] = "server"
    env["SMBOOST_OPENAI_BASE_URL"] = _server_base_url_for(model)
    env.setdefault("SMBOOST_OPENAI_API_KEY", "sk-no-key")
    env.setdefault("SMBOOST_OPENAI_MAX_TOKENS", "512")

    tmp_csv.unlink(missing_ok=True)
    cmd = [
        sys.executable, "-u", "scripts/run_gates_0_8b.py",
        "--stage", stage,
        "--model", model,
        "--out-csv", str(tmp_csv),
        "--n", str(n),
        "--modes", mode,
    ]
    print(f"[matrix] RUN bench={bench} model={model} mode={mode} n={n}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT))
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        print(f"[matrix] FAIL cell returned {proc.returncode} after {elapsed:.1f}s",
              flush=True)
        return []
    print(f"[matrix] OK cell done in {elapsed:.1f}s", flush=True)
    return _read_csv(tmp_csv)


def run_external_cell(bench: str, model: str, n: int,
                      tmp_csv: Path) -> list[dict]:
    """Run one (bench, external_model, raw) cell via benchmarks.external.cli.

    External runs are always mode=raw (no harness on paid APIs).
    """
    tmp_csv.unlink(missing_ok=True)
    cmd = [
        sys.executable, "-m", "benchmarks.external.cli",
        "--model", model,
        "--bench", bench,
        "--n", str(n),
        "--out-csv", str(tmp_csv),
    ]
    print(f"[matrix] RUN EXTERNAL bench={bench} model={model} n={n}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        print(f"[matrix] FAIL external cell returned {proc.returncode} after {elapsed:.1f}s",
              flush=True)
        return []
    print(f"[matrix] OK external cell done in {elapsed:.1f}s", flush=True)
    return _read_csv(tmp_csv)


def evaluate_and_report(consolidated_csv: Path, report_md: Path,
                        cli_smoke_passed: bool = False,
                        demo_trace_present: bool = False) -> None:
    """Read consolidated CSV, compute G1a-G5a, write markdown report."""
    from benchmarks.gates.criteria import (
        evaluate_g1a, evaluate_g2a, evaluate_g3a, evaluate_g4a, evaluate_g5a,
    )
    rows = _read_csv(consolidated_csv)

    gates = [
        ("G1a_capability_floor", evaluate_g1a(rows)),
        ("G2a_harness_lift_math", evaluate_g2a(rows)),
        ("G3a_structured_output", evaluate_g3a(rows)),
        ("G4a_external_parity", evaluate_g4a(rows)),
        ("G5a_demo_readiness", evaluate_g5a(rows, cli_smoke_passed=cli_smoke_passed,
                                            demo_trace_present=demo_trace_present)),
    ]

    lines = ["# Full-matrix YC Gate Report\n"]
    lines.append(f"Source: `{consolidated_csv.as_posix()}`  ({len(rows)} rows)\n")
    lines.append("## Gate verdicts\n")
    for name, r in gates:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"- **{name}**: {status}")
        for k, v in r.metrics.items():
            lines.append(f"  - {k}: `{v}`")
        if not r.passed:
            for f in r.failed_checks:
                lines.append(f"  - ✗ {f}")
        lines.append("")

    # Summary table
    from collections import defaultdict
    agg: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "pass": 0, "ret": 0.0, "lat": 0.0})
    for row in rows:
        k = (row.get("bench", ""), row.get("model", ""), row.get("mode", ""))
        agg[k]["n"] += 1
        agg[k]["pass"] += int(row.get("passed", 0) or 0)
        agg[k]["ret"] += float(row.get("retries", 0) or 0)
        try:
            agg[k]["lat"] += float(row.get("latency_s", 0) or 0)
        except (TypeError, ValueError):
            pass

    lines.append("## Per-cell summary\n")
    lines.append("| bench | model | mode | n | pass | rate | ret_avg | lat_avg |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (bench, model, mode), v in sorted(agg.items()):
        rate = v["pass"] / v["n"] * 100 if v["n"] else 0.0
        lines.append(f"| {bench} | {model} | {mode} | {v['n']} | {v['pass']} | "
                     f"{rate:.1f}% | {v['ret']/v['n']:.2f} | {v['lat']/v['n']:.1f}s |")

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n")
    print(f"[matrix] report → {report_md}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-csv", default="benchmarks/results/full_matrix_v2.csv")
    p.add_argument("--report-md", default="docs/overnight/2026-05-02-yc-full-matrix.md")
    p.add_argument("--models", default="qwen3.5:2b,qwen3.5:0.8b",
                   help="comma-separated local model ids")
    p.add_argument("--external-models", default="",
                   help="comma-separated external model ids (claude-sonnet-4-6,gpt-4o,...)")
    p.add_argument("--modes", default="raw,C1,C5,C6",
                   help="comma-separated modes for local models")
    p.add_argument("--benches", default="gsm8k,bfcl_simple,humaneval_plus")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--n-external", type=int, default=30)
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="skip cells already present in --out-csv (default on)")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--report-only", action="store_true",
                   help="skip runs, only regenerate the report from existing CSV")
    p.add_argument("--cli-smoke", action="store_true",
                   help="mark G5a CLI smoke as passed (caller verified externally)")
    p.add_argument("--demo-trace", action="store_true",
                   help="mark G5a demo trace as present (caller verified externally)")
    args = p.parse_args()

    out_csv = Path(args.out_csv)
    report_md = Path(args.report_md)

    if args.report_only:
        evaluate_and_report(out_csv, report_md,
                            cli_smoke_passed=args.cli_smoke,
                            demo_trace_present=args.demo_trace)
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    external = [m.strip() for m in args.external_models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    benches = [b.strip() for b in args.benches.split(",") if b.strip()]

    existing = _existing_cells(out_csv) if args.skip_existing else set()
    if existing:
        print(f"[matrix] {len(existing)} cells already present; skipping those", flush=True)

    tmp_csv = Path("/tmp/matrix_cell.csv")

    # Local model cells
    for bench in benches:
        for model in models:
            for mode in modes:
                if (bench, model, mode) in existing:
                    print(f"[matrix] SKIP {bench}/{model}/{mode} (already in CSV)",
                          flush=True)
                    continue
                rows = run_local_cell(bench, model, mode, args.n, tmp_csv)
                if rows:
                    _merge_csv(rows, out_csv)

    # External model cells (raw only)
    for bench in benches:
        for model in external:
            if (bench, model, "raw") in existing:
                print(f"[matrix] SKIP EXTERNAL {bench}/{model}/raw", flush=True)
                continue
            rows = run_external_cell(bench, model, args.n_external, tmp_csv)
            if rows:
                _merge_csv(rows, out_csv)

    evaluate_and_report(out_csv, report_md,
                        cli_smoke_passed=args.cli_smoke,
                        demo_trace_present=args.demo_trace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
