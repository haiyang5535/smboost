from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def load_results(csv_path: Path) -> list[dict]:
    """Read CSV, return [{model, rows}] grouped by model (first-appearance order).
    Within each model, last occurrence per mode wins.
    Raises FileNotFoundError if missing, ValueError if no data rows.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    by_model: dict[str, dict[str, dict]] = {}
    model_order: list[str] = []

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            mode = row["mode"]
            avg_retries_raw = row["avg_retries"]
            if model not in by_model:
                by_model[model] = {}
                model_order.append(model)
            by_model[model][mode] = {
                "mode": mode,
                "passAt1": float(row["pass@1"]),
                "avgRetries": float(avg_retries_raw) if avg_retries_raw not in ("-", "") else None,
                "avgLatency": float(row["avg_latency_s"]),
            }

    if not by_model:
        raise ValueError("No results found in CSV (file exists but has no data rows)")

    mode_order = ["baseline", "smboost", "upper_bound"]
    result = []
    for model in model_order:
        seen = by_model[model]
        rows = [seen[m] for m in mode_order if m in seen]
        rows += [seen[m] for m in seen if m not in mode_order]
        result.append({"model": model, "rows": rows})
    return result


def render_js(model_groups: list[dict], n_tasks: int, generated_at: str) -> str:
    """Return benchmark_data.js content as a string."""
    payload = {
        "generatedAt": generated_at,
        "nTasks": n_tasks,
        "models": model_groups,
    }
    return f"window.SMBOOST_BENCHMARK = {json.dumps(payload, indent=2)};\n"


def export_results(csv_path: Path, out_path: Path, n_tasks: int) -> None:
    model_groups = load_results(csv_path)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    js = render_js(model_groups, n_tasks=n_tasks, generated_at=generated_at)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(js)
    models_str = ", ".join(g["model"] for g in model_groups)
    print(f"Written: {out_path}  ({len(model_groups)} models: {models_str})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark CSV to dashboard JS")
    parser.add_argument("--csv", default="benchmarks/results/results.csv",
                        help="path to results CSV")
    parser.add_argument("--out", default="frontend/benchmark_data.js",
                        help="output JS file path")
    parser.add_argument("--n-tasks", type=int, default=50)
    args = parser.parse_args()
    export_results(Path(args.csv), Path(args.out), args.n_tasks)


if __name__ == "__main__":
    main()
