"""Overnight full ablation: 3 models × 2 budgets × 4 conditions × 3 seeds × 50 tasks.

Run order: 0.8b → 2b → 4b (sequential server swaps).
Waits for CANARY_PID to exit before starting.
Resume-safe: skips already-done tasks via shard resume semantics.

Usage:
    nohup python3 -u scripts/overnight_full_matrix.py > /tmp/overnight.log 2>&1 &
"""
from __future__ import annotations
import csv
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.livecodebench.matrix import run_matrix, FINAL_DIR, SHARD_DIR

CANARY_PID = 0  # 0 = no wait

# Security (F15): previously this script unconditionally SIGKILL'd any process
# bound to port 8000 and any "llama_cpp.server"/"llama-server" process on the
# box. That's dangerous on shared dev machines (FastAPI defaults to 8000,
# other llama servers, other operators' work). The kill/start-server flow is
# now gated behind an explicit env var.
_MANAGE_SERVER_ENV = "SMBOOST_OVERNIGHT_MANAGE_SERVER"


def _manage_server_enabled() -> bool:
    return os.environ.get(_MANAGE_SERVER_ENV, "").strip() not in ("", "0", "false", "False")

MODELS = [
    ("0.8b", "qwen3.5:0.8b", _ROOT / "models/Qwen3.5-0.8B-Q4_K_M.gguf"),
    ("2b",   "qwen3.5:2b",   _ROOT / "models/Qwen3.5-2B-Q4_K_M.gguf"),
    ("4b",   "qwen3.5:4b",   _ROOT / "models/Qwen3.5-4B-Q4_K_M.gguf"),
]

# Start from this model tag (skips earlier ones); set to None to run all
START_FROM = None
CONDITIONS = ["C1", "C2", "C3", "C4"]
SEEDS = [0, 1, 2]
BUDGETS = [64, 160]
N_TASKS = 50
SERVER_URL = "http://localhost:8000/v1/models"


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _wait_for_canary(pid: int) -> None:
    if not _pid_alive(pid):
        _log(f"canary pid={pid} already done")
        return
    _log(f"waiting for canary pid={pid} to finish ...")
    while _pid_alive(pid):
        time.sleep(10)
    _log("canary done — reading gate report")
    try:
        log = Path("/tmp/canary_run.log").read_text()
        for line in log.splitlines():
            if any(k in line for k in ("CANARY REPORT", "GATE:", "4b:", "9b:", "gap=")):
                _log(f"  canary: {line.strip()}")
    except FileNotFoundError:
        pass


def _kill_server() -> None:
    if not _manage_server_enabled():
        _log(
            f"refusing to kill processes on port 8000 without {_MANAGE_SERVER_ENV}=1"
        )
        return
    subprocess.run(["pkill", "-9", "-f", "llama_cpp.server"], check=False)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
    for _ in range(15):
        out = subprocess.run(["lsof", "-ti", ":8000"], capture_output=True, text=True)
        pids = out.stdout.split()
        if not pids:
            break
        subprocess.run(["kill", "-9"] + pids, check=False)
        time.sleep(1)
    time.sleep(1)


def _start_server(gguf: Path) -> bool:
    _kill_server()
    _log(f"starting server: {gguf.name}")
    subprocess.Popen(
        [sys.executable, "-m", "llama_cpp.server",
         "--model", str(gguf),
         "--port", "8000",
         "--n_gpu_layers", "-1",
         "--chat_format", "qwen",
         "--host", "127.0.0.1"],
        stdout=open(f"/tmp/llama_server_{gguf.stem}.log", "w"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 240
    dots = 0
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(SERVER_URL, timeout=3)
            _log(f"server ready: {gguf.name}")
            return True
        except Exception:
            time.sleep(5)
            dots += 1
            if dots % 6 == 0:
                _log(f"  still loading {gguf.name} ...")
    _log(f"ERROR: server startup timeout for {gguf.name}")
    return False


def _budget_matches(row: dict[str, str], max_tokens: int) -> bool:
    return str(row.get("max_tokens", "")).strip() == str(max_tokens)


def _pass_rate(csv_path: Path, condition: str, model: str, max_tokens: int) -> tuple[float, int]:
    passed, total = 0, 0
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if (
                    row["condition"] == condition
                    and row["model"] == model
                    and _budget_matches(row, max_tokens)
                ):
                    total += 1
                    passed += int(row["passed"])
    except FileNotFoundError:
        pass
    return (passed / total if total else 0.0), total


def _avg_field(csv_path: Path, condition: str, model: str, max_tokens: int, field: str) -> float:
    vals: list[float] = []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if (
                    row["condition"] == condition
                    and row["model"] == model
                    and _budget_matches(row, max_tokens)
                ):
                    try:
                        vals.append(float(row[field]))
                    except (ValueError, KeyError):
                        pass
    except FileNotFoundError:
        pass
    return sum(vals) / len(vals) if vals else 0.0


def _dominant_failure_bucket(csv_path: Path, condition: str, model: str, max_tokens: int) -> str:
    counts: dict[str, int] = {}
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if (
                    row["condition"] == condition
                    and row["model"] == model
                    and _budget_matches(row, max_tokens)
                ):
                    bucket = row.get("failure_bucket", "")
                    if not bucket:
                        continue
                    counts[bucket] = counts.get(bucket, 0) + 1
    except FileNotFoundError:
        return "—"
    if not counts:
        return "—"
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _generate_report(csv_path: Path, skipped: list[str]) -> str:
    lines = [
        "# Overnight Full Ablation Report",
        f"",
        f"Matrix CSV: `{csv_path.name}`",
        f"Conditions: C1=full, C2=-grounded_verify, C3=-session_memory, C4=plain_langgraph_retry",
        f"Tasks: {N_TASKS}  Seeds: {SEEDS}  Budgets: {BUDGETS}",
        "",
        "## Matched Budget Results",
        "",
    ]
    for budget in BUDGETS:
        lines += [
            f"### Budget {budget}",
            "",
            "| Model | C1 | C2 | C3 | C4 | C1 vs C4 Δ | Avg Latency C1 | Avg Retries C1 | Dominant Failure Bucket C1 |",
            "|-------|----|----|----|----|------------|----------------|----------------|----------------------------|",
        ]
        for tag, model_name, _ in MODELS:
            rates = {cond: _pass_rate(csv_path, cond, model_name, budget) for cond in CONDITIONS}
            if all(n == 0 for _, n in rates.values()):
                lines.append(f"| {tag} | — | — | — | — | — | — | — | — |")
                continue
            c1, n1 = rates["C1"]
            c2, _ = rates["C2"]
            c3, _ = rates["C3"]
            c4, n4 = rates["C4"]
            delta = c1 - c4
            lat_s = _avg_field(csv_path, "C1", model_name, budget, "duration_ms") / 1000
            retries = _avg_field(csv_path, "C1", model_name, budget, "retries")
            bucket = _dominant_failure_bucket(csv_path, "C1", model_name, budget)
            lines.append(
                f"| {tag} | {c1:.1%} ({n1}t) | {c2:.1%} | {c3:.1%} | {c4:.1%} ({n4}t) | "
                f"**{delta:+.1%}** | {lat_s:.1f}s | {retries:.2f} | {bucket} |"
            )
        lines.append("")

    lines += ["## Ablation Breakdown", ""]
    for budget in BUDGETS:
        lines.append(f"**Budget {budget}**")
        for tag, model_name, _ in MODELS:
            c1, n1 = _pass_rate(csv_path, "C1", model_name, budget)
            c2, _ = _pass_rate(csv_path, "C2", model_name, budget)
            c3, _ = _pass_rate(csv_path, "C3", model_name, budget)
            c4, _ = _pass_rate(csv_path, "C4", model_name, budget)
            if n1 == 0:
                continue
            lines.append(
                f"- {tag}: grounded_verify adds {c1-c2:+.1%}, session_memory adds {c1-c3:+.1%}, "
                f"full harness vs plain retry is {c1-c4:+.1%}"
            )
        lines.append("")

    if skipped:
        lines += ["", "## Skipped Models", ""]
        for s in skipped:
            lines.append(f"- {s}")

    return "\n".join(lines)


def main() -> None:
    _log("=" * 60)
    _log("OVERNIGHT FULL ABLATION MATRIX")
    _log(f"Models: {[t for t,_,_ in MODELS]}")
    _log(f"Budgets: {BUDGETS}")
    _log(f"Conditions: {CONDITIONS}")
    _log(f"Seeds: {SEEDS}  Tasks/cell: {N_TASKS}")
    _log("=" * 60)

    # Security (F15): refuse to run the kill-and-relaunch server lifecycle
    # without explicit operator consent. The script swaps models by SIGKILLing
    # anything on port 8000, which is dangerous on shared dev machines.
    if not _manage_server_enabled():
        _log(
            f"ERROR: this script manages the llama.cpp server lifecycle "
            f"(pkill llama_cpp.server, kill $(lsof -ti :8000), relaunch per model). "
            f"Refusing to run without {_MANAGE_SERVER_ENV}=1."
        )
        _log(
            "To run it: start the server yourself for each model, or re-run with "
            f"{_MANAGE_SERVER_ENV}=1 to let this script manage it."
        )
        sys.exit(2)

    if CANARY_PID:
        _wait_for_canary(CANARY_PID)

    results_csv: Path | None = None
    skipped: list[str] = []
    start_total = time.monotonic()

    models = MODELS
    if START_FROM:
        idx = next((i for i, (t, _, __) in enumerate(MODELS) if t == START_FROM), 0)
        models = MODELS[idx:]
        _log(f"Resuming from model: {START_FROM} (skipping {MODELS[:idx]})")

    for tag, model_name, gguf in models:
        if not gguf.exists():
            _log(f"SKIP {tag}: GGUF not found: {gguf}")
            skipped.append(f"{tag}: GGUF missing")
            continue

        _log(f"{'='*50}")
        _log(f"MODEL: {tag} ({model_name})")
        _log(f"{'='*50}")

        ok = _start_server(gguf)
        if not ok:
            skipped.append(f"{tag}: server startup timeout or OOM")
            continue

        for budget in BUDGETS:
            n_cells = len(CONDITIONS) * len(SEEDS)
            _log(
                f"running budget={budget} with {n_cells} cells "
                f"({len(CONDITIONS)} conds × {len(SEEDS)} seeds × {N_TASKS} tasks)"
            )
            os.environ["SMBOOST_LLM_BACKEND"] = "server"
            os.environ["SMBOOST_OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"
            os.environ["SMBOOST_OPENAI_API_KEY"] = "sk-no-key"
            os.environ["SMBOOST_OPENAI_MAX_TOKENS"] = str(budget)
            t0 = time.monotonic()
            try:
                results_csv = run_matrix(
                    conditions=CONDITIONS,
                    models=[model_name],
                    seeds=SEEDS,
                    n_tasks=N_TASKS,
                    ollama_concurrency=2,
                    max_tokens=budget,
                )
                elapsed = time.monotonic() - t0
                _log(f"DONE {tag} budget={budget} in {elapsed/60:.1f} min — csv: {results_csv.name}")
            except Exception as exc:
                _log(f"ERROR {tag} budget={budget}: {exc}")
                skipped.append(f"{tag}@{budget}: run_matrix error: {exc}")

    total_elapsed = time.monotonic() - start_total
    _log(f"{'='*60}")
    _log(f"ALL MODELS COMPLETE in {total_elapsed/3600:.2f}h")
    _log(f"{'='*60}")

    # Final merged CSV is the last result (contains all shards)
    if results_csv is None:
        # Try finding latest
        csvs = sorted(FINAL_DIR.glob("livecodebench_hard_matrix_*.csv"),
                      key=lambda p: p.stat().st_mtime)
        results_csv = csvs[-1] if csvs else None

    if results_csv and results_csv.exists():
        report = _generate_report(results_csv, skipped)
        print("\n" + report, flush=True)
        report_path = results_csv.with_suffix(".md")
        report_path.write_text(report)
        _log(f"Report saved: {report_path}")
    else:
        _log("No results CSV — all models skipped or failed.")

    print("\n" + "=" * 60, flush=True)
    print("OVERNIGHT MISSION COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
