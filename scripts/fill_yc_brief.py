"""Auto-fill the YC submission brief's TBD numbers from the consolidated matrix CSV.

Reads ``benchmarks/results/full_matrix_v2.csv`` (or whatever ``--csv`` points
at), aggregates per (model, mode) on bench=gsm8k, and rewrites the headline
table in ``docs/overnight/2026-05-02-yc-submission-brief.md``.

Idempotent — safe to re-run after each matrix cell completes.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _aggregate(csv_path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("bench") != "gsm8k":
                continue
            k = (row.get("model", ""), row.get("mode", ""))
            agg[k][0] += int(row.get("passed", 0) or 0)
            agg[k][1] += 1
    return {k: (v[0], v[1]) for k, v in agg.items()}


def _fmt(rate: tuple[int, int] | None) -> str:
    if rate is None or rate[1] == 0:
        return "TBD"
    return f"{rate[0]}/{rate[1]} = {rate[0]/rate[1]*100:.1f}%"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="benchmarks/results/full_matrix_v2.csv")
    p.add_argument("--brief", default="docs/overnight/2026-05-02-yc-submission-brief.md")
    args = p.parse_args()

    csv_path = ROOT / args.csv
    brief_path = ROOT / args.brief

    if not brief_path.exists():
        print(f"brief not found: {brief_path}", file=sys.stderr)
        return 1

    agg = _aggregate(csv_path)
    raw_2b = agg.get(("qwen3.5:2b", "raw"))
    c1_2b = agg.get(("qwen3.5:2b", "C1"))
    c5_2b = agg.get(("qwen3.5:2b", "C5"))
    raw_08 = agg.get(("qwen3.5:0.8b", "raw"))
    c5_08 = agg.get(("qwen3.5:0.8b", "C5"))

    def _lift(raw, harnessed):
        if not raw or not harnessed or raw[1] == 0 or harnessed[1] == 0:
            return "TBD"
        rr = raw[0] / raw[1]
        hr = harnessed[0] / harnessed[1]
        if rr == 0:
            return f"{hr*100:.1f}pp absolute"
        return f"{hr/rr:.2f}×"

    new_table = "\n".join([
        "| Model | Bench | raw | C1 | C5 | Lift (C5/raw) |",
        "|---|---|---|---|---|---|",
        f"| Qwen 2.5 2B  | GSM8K | {_fmt(raw_2b)} | {_fmt(c1_2b)} | {_fmt(c5_2b)} | {_lift(raw_2b, c5_2b)} |",
        f"| Qwen 2.5 0.8B | GSM8K | {_fmt(raw_08)} | — | {_fmt(c5_08)} | {_lift(raw_08, c5_08)} |",
    ])

    text = brief_path.read_text()
    pattern = re.compile(
        r"\| Model \| Bench \| raw \| C1 \| C5 \| Lift \(C5/raw\) \|.*?\| Qwen 2\.5 0\.8B \|[^\n]*\|",
        re.DOTALL,
    )
    if not pattern.search(text):
        print("table marker not found in brief", file=sys.stderr)
        return 1
    text = pattern.sub(new_table, text)
    brief_path.write_text(text)
    print(f"[fill_yc_brief] updated {brief_path}")
    print(f"  raw_2b={_fmt(raw_2b)}, C1_2b={_fmt(c1_2b)}, C5_2b={_fmt(c5_2b)}")
    print(f"  raw_0.8b={_fmt(raw_08)}, C5_0.8b={_fmt(c5_08)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
