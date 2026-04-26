#!/usr/bin/env bash
# Merge per-server matrix CSVs into the consolidated full_matrix_v2.csv,
# regenerate the gate report, and auto-fill the YC brief.
#
# Idempotent: safe to re-run after each cell finishes.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT=benchmarks/results/full_matrix_v2.csv
A=benchmarks/results/full_matrix_2b.csv
B=benchmarks/results/full_matrix_0.8b.csv

rm -f "$OUT"

# Take the header from whichever exists first
if [ -f "$A" ]; then
  head -1 "$A" > "$OUT"
elif [ -f "$B" ]; then
  head -1 "$B" > "$OUT"
else
  echo "no matrix CSVs found yet" >&2
  exit 1
fi

# Append data rows from each
[ -f "$A" ] && tail -n +2 "$A" >> "$OUT"
[ -f "$B" ] && tail -n +2 "$B" >> "$OUT"

echo "[finalize] merged → $OUT ($(wc -l < "$OUT") lines)"

# Regenerate report
python3 scripts/run_full_matrix_v2.py \
  --report-only \
  --out-csv "$OUT" \
  --report-md docs/overnight/2026-05-02-yc-full-matrix.md \
  --cli-smoke --demo-trace

# Auto-fill brief
python3 scripts/fill_yc_brief.py --csv "$OUT"

echo "[finalize] done"
