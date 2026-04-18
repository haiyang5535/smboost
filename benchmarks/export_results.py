from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def load_results(csv_path: Path) -> list[dict]:
    """Read CSV, return one row per mode (last occurrence wins). Raises on missing/empty."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    seen: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            mode = row["mode"]
            avg_retries_raw = row["avg_retries"]
            seen[mode] = {
                "mode": mode,
                "passAt1": float(row["pass@1"]),
                "avgRetries": float(avg_retries_raw) if avg_retries_raw not in ("-", "") else None,
                "avgLatency": float(row["avg_latency_s"]),
            }

    if not seen:
        raise ValueError("No results found in CSV (file exists but has no data rows)")

    mode_order = ["baseline", "smboost", "upper_bound"]
    ordered = [seen[m] for m in mode_order if m in seen]
    ordered += [seen[m] for m in seen if m not in mode_order]
    return ordered


def _model_from_csv(csv_path: Path) -> str:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row["model"]
    return "unknown"


def render_js(rows: list[dict], model: str, n_tasks: int, generated_at: str) -> str:
    """Return benchmark_data.js content as a string."""
    payload = {
        "generatedAt": generated_at,
        "model": model,
        "nTasks": n_tasks,
        "rows": rows,
    }
    return f"window.SMBOOST_BENCHMARK = {json.dumps(payload, indent=2)};\n"


def export_results(csv_path: Path, out_path: Path, n_tasks: int) -> None:
    rows = load_results(csv_path)
    model = _model_from_csv(csv_path)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    js = render_js(rows, model=model, n_tasks=n_tasks, generated_at=generated_at)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(js)
    print(f"Written: {out_path}  ({len(rows)} modes, model={model})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark CSV to dashboard JS")
    parser.add_argument("--csv", default="benchmarks/results/results.csv")
    parser.add_argument("--out", default="frontend/benchmark_data.js")
    parser.add_argument("--n-tasks", type=int, default=50)
    args = parser.parse_args()
    export_results(Path(args.csv), Path(args.out), args.n_tasks)


if __name__ == "__main__":
    main()
