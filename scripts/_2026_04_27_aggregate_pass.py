"""Aggregator for the 2026-04-27 improvement pass.

Reads (each is optional — missing files are treated as empty):
  benchmarks/results/full_matrix_v2.csv      GSM8K headline matrix
  benchmarks/results/he_n50_2b.csv           HumanEval+ n=50 confirm — A1.5
  benchmarks/results/a2_phi3_gsm8k_n20.csv   Phi-3 GSM8K pilot — A2
  benchmarks/results/a2_phi3_gsm8k_n50.csv   Phi-3 GSM8K confirm — A2.5

Writes a consolidated table to stdout and to
  docs/overnight/2026-04-27-aggregate.md
covering all (bench, model, mode) cells with n, pass, lift vs. raw, and
median latency. Also writes the merged headline CSV
  benchmarks/results/full_matrix_v2_with_he.csv
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent.parent
HEADER = [
    "bench", "task_id", "model", "mode", "passed",
    "passed_heval", "passed_heval_plus",
    "retries", "latency_s",
]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _merge_he_rows(rows: list[dict]) -> list[dict]:
    """Tag bench column for HE+ rows that may have been written without it."""
    out = []
    for r in rows:
        if not r.get("bench"):
            r = {**r, "bench": "humaneval_plus"}
        out.append(r)
    return out


def main() -> int:
    res = ROOT / "benchmarks/results"
    matrix_csv = res / "full_matrix_v2.csv"
    he_n50_csv = res / "he_n50_2b.csv"
    a2_n20_csv = res / "a2_phi3_gsm8k_n20.csv"
    a2_n50_csv = res / "a2_phi3_gsm8k_n50.csv"

    rows = _read(matrix_csv)
    rows += _merge_he_rows(_read(he_n50_csv))
    rows += _read(a2_n20_csv)
    rows += _read(a2_n50_csv)

    # Aggregate
    agg = defaultdict(lambda: {"n": 0, "pass": 0, "lat": []})
    for r in rows:
        if not r.get("bench"):
            continue
        k = (r["bench"], r["model"], r["mode"])
        agg[k]["n"] += 1
        agg[k]["pass"] += int(r.get("passed", 0) or 0)
        try:
            agg[k]["lat"].append(float(r.get("latency_s", 0) or 0))
        except (TypeError, ValueError):
            pass

    # Lift table
    raws = {(b, m): v for (b, m, mode), v in agg.items() if mode == "raw"}

    lines = [
        "# 2026-04-27 Improvement Pass — Consolidated Aggregate",
        "",
        "| bench | model | mode | n | pass | rate | lat_med | lift vs raw |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (bench, model, mode), v in sorted(agg.items()):
        rate = v["pass"] / v["n"] * 100 if v["n"] else 0.0
        lat_med = median(v["lat"]) if v["lat"] else 0.0
        if mode == "raw":
            lift = "—"
        else:
            r = raws.get((bench, model))
            if r and r["pass"]:
                lift_ratio = (v["pass"] / v["n"]) / (r["pass"] / r["n"])
                lift = f"{lift_ratio:.2f}×"
            else:
                lift = "n/a"
        lines.append(
            f"| {bench} | {model} | {mode} | {v['n']} | {v['pass']} | "
            f"{rate:.1f}% | {lat_med:.1f}s | {lift} |"
        )

    out_md = ROOT / "docs/overnight/2026-04-27-aggregate.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[aggregate] wrote {out_md}")

    # Merged CSV (informational; ignored from VCS)
    merged_csv = ROOT / "benchmarks/results/full_matrix_v2_with_he.csv"
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {merged_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
