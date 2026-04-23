"""Day-8 canary: run C1 vs C4 on 10 LCB Hard tasks, both models, 1 seed.

Sequential mode: run 4b first, then restart server with 9b, then run 9b.
Gate: C1 must beat C4 by >=5pp on qwen3.5:9b.

Usage:
    python scripts/canary_day8.py                        # sequential: 4b then 9b
    python scripts/canary_day8.py --model 4b             # 4b only
    python scripts/canary_day8.py --model 9b             # 9b only (server must already be running 9b)
    python scripts/canary_day8.py --skip-restart         # run whatever model is loaded, skip restart
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running as `python3 scripts/canary_day8.py`
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.livecodebench.matrix import run_matrix

MODEL_PATHS = {
    "4b": Path("models/Qwen3.5-4B-Q4_K_M.gguf"),
    "9b": Path("models/Qwen3.5-9B-Q4_K_M.gguf"),
}
MODEL_NAMES = {
    "4b": "qwen3.5:4b",
    "9b": "qwen3.5:9b",
}
SERVER_URL = "http://localhost:8000/v1/models"

def _pass_rate(csv_path: Path, condition: str, model: str) -> float:
    passed, total = 0, 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["condition"] == condition and row["model"] == model:
                total += 1
                passed += int(row["passed"])
    return passed / total if total else 0.0


def _wait_server(timeout: int = 60) -> bool:
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(SERVER_URL, timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def _restart_server(gguf_path: Path) -> None:
    print(f"\n--- restarting server with {gguf_path.name} ---")
    # Kill any process using port 8000 (covers both llama_cpp.server and llama-server binary)
    subprocess.run(["pkill", "-9", "-f", "llama_cpp.server"], check=False)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
    # Wait for port to clear
    for _ in range(10):
        result = subprocess.run(
            ["lsof", "-ti", ":8000"], capture_output=True, text=True
        )
        if not result.stdout.strip():
            break
        subprocess.run(["kill", "-9"] + result.stdout.split(), check=False)
        time.sleep(1)
    time.sleep(1)
    subprocess.Popen(
        [
            sys.executable, "-m", "llama_cpp.server",
            "--model", str(gguf_path),
            "--port", "8000",
            "--n_gpu_layers", "-1",
            "--chat_format", "qwen",
            "--host", "127.0.0.1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("waiting for server to come up", end="", flush=True)
    ok = _wait_server(timeout=120)
    print()
    if not ok:
        raise RuntimeError(f"llama-server failed to start with {gguf_path}")
    print(f"server ready ({gguf_path.name})")


def _run_one_model(tag: str, gguf_path: Path | None, restart: bool) -> tuple[float, float, Path]:
    model_name = MODEL_NAMES[tag]
    if restart and gguf_path is not None:
        _restart_server(gguf_path)
    print(f"\n=== running {tag} ({model_name}) ===")
    out = run_matrix(
        conditions=["C1", "C4"],
        models=[model_name],
        seeds=[0],
        n_tasks=10,
        ollama_concurrency=2,
    )
    c1 = _pass_rate(out, "C1", model_name)
    c4 = _pass_rate(out, "C4", model_name)
    return c1, c4, out


def _print_report(results: dict[str, tuple[float, float]]) -> None:
    print("\n" + "=" * 50)
    print("CANARY REPORT — Day 8")
    print("=" * 50)
    gate_ok = True
    for tag in ("4b", "9b"):
        if tag not in results:
            continue
        c1, c4 = results[tag]
        gap = c1 - c4
        status = "PASS" if gap >= 0.05 else "FAIL"
        if tag == "9b" and gap < 0.05:
            gate_ok = False
        print(f"  {tag}: C1={c1:.1%}  C4={c4:.1%}  gap={gap:+.1%}  [{status}]")
    print("=" * 50)
    if "9b" in results:
        if gate_ok:
            print("GATE: PASS — proceed to full matrix (Days 9-10)")
        else:
            c1, c4 = results["9b"]
            gap = c1 - c4
            print(
                f"GATE: FAIL — C1 only beats C4 by {gap:+.1%} on 9b (<5pp). "
                "Investigate grounded verify + memory wiring before full matrix."
            )
    else:
        print("NOTE: 9b not run — gate decision deferred")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["4b", "9b", "both"], default="both",
        help="which model to run (default: both, sequential)",
    )
    parser.add_argument(
        "--skip-restart", action="store_true",
        help="skip llama-server restart; use whatever is currently loaded",
    )
    args = parser.parse_args()

    tags = ["4b", "9b"] if args.model == "both" else [args.model]
    results: dict[str, tuple[float, float]] = {}

    for i, tag in enumerate(tags):
        gguf = MODEL_PATHS[tag]
        # Only restart if we're managing the server (not skipping) and either:
        # - it's the first model (ensure correct model is loaded), or
        # - switching to a different model
        do_restart = not args.skip_restart
        c1, c4, _ = _run_one_model(tag, gguf, restart=do_restart)
        results[tag] = (c1, c4)
        print(f"{tag}: C1={c1:.1%}  C4={c4:.1%}  gap={c1-c4:+.1%}")

    _print_report(results)

    if "9b" in results:
        c1, c4 = results["9b"]
        if c1 - c4 < 0.05:
            sys.exit(1)


if __name__ == "__main__":
    main()
