"""One-off driver for gate reruns (Phase 1 / Phase 3 of overnight orchestration).

Reuses benchmarks/gates/runner.py machinery but runs one (model, mode) at a
time so we persist to CSV after each mode finishes. Crash-safe: if the
process dies mid-C4, we still have raw + C1 on disk.

Stages:
    - HE     : raw + C1 + C4 on HumanEval+
    - BFCL   : raw + C1 on BFCL simple
    - GSM8K  : raw + C1 on GSM8K math (harness sweet zone; C5/C6 added once
               those agents merge)

Notes
-----
* Raw HumanEval mode ignored `max_tokens` until 2026-04-23. When the
  underlying local model is small and verbose (0.8B especially), completions
  can run away to the n_ctx limit, taking minutes per task. We now pass
  `max_tokens` (env `SMBOOST_OPENAI_MAX_TOKENS`, default 512) into the raw
  `ChatOpenAI` call via a monkey-patch of `run_humaneval.run_baseline`.
* GSM8K raw mode already honors `SMBOOST_OPENAI_MAX_TOKENS` via
  `benchmarks.gsm8k.runner._make_raw_llm`, so no patch needed there.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "SSL_CERT_FILE" not in os.environ:
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass


HEADER = [
    "bench", "task_id", "model", "mode", "passed",
    "passed_heval", "passed_heval_plus",
    "retries", "latency_s",
]


def _install_raw_max_tokens_patch(max_tokens: int) -> None:
    """Monkey-patch benchmarks.run_humaneval.run_baseline so it pins
    ChatOpenAI.max_tokens. Otherwise small-model raw completions run to
    n_ctx, which is catastrophically slow."""
    from benchmarks import run_humaneval as _rh

    _original_run_baseline = _rh.run_baseline  # noqa: F841 (kept for debug)

    def run_baseline_with_cap(tasks, model, temperature: float = 0.0):
        # Use the _CompatibleChatOpenAI subclass so `max_tokens` survives the
        # langchain-openai 1.1.x payload rewrite (plain ChatOpenAI silently
        # renames it to `max_completion_tokens`, which llama.cpp ignores —
        # completions then run away to n_ctx and each call takes ~10min on
        # 0.8B).
        from smboost.llm.runtime import _CompatibleChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = _CompatibleChatOpenAI(
            model=model,
            base_url=os.environ.get("SMBOOST_OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("SMBOOST_OPENAI_API_KEY", "sk-no-key"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        results = []
        for task in tasks:
            start = time.monotonic()
            raw = llm.invoke([HumanMessage(content=task["prompt"])]).content or ""
            results.append({
                "task_id": task["task_id"],
                "completion": _rh.clean_completion(raw),
                "latency_s": round(time.monotonic() - start, 3),
                "retries": 0,
            })
        return results

    _rh.run_baseline = run_baseline_with_cap
    print(f"[driver] patched run_humaneval.run_baseline with max_tokens={max_tokens}", flush=True)


def _append_rows(out_csv: Path, rows: list[dict]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _summary(rows: list[dict], label: str) -> None:
    n = len(rows)
    passed = sum(int(r.get("passed", 0)) for r in rows)
    avg_ret = (sum(int(r.get("retries", 0)) for r in rows) / n) if n else 0.0
    avg_lat = (sum(float(r.get("latency_s", 0.0) or 0.0) for r in rows) / n) if n else 0.0
    print(f"[summary] {label}: {passed}/{n} = {passed/n*100 if n else 0:.1f}%  "
          f"avg_retries={avg_ret:.2f}  avg_latency_s={avg_lat:.1f}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["HE", "BFCL", "GSM8K"],
                   help="HE = raw+C1+C4 on HumanEval; BFCL = raw+C1 on BFCL simple; "
                        "GSM8K = raw+C1 on GSM8K math")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--model", default="qwen3.5:0.8b")
    p.add_argument("--modes", default=None,
                   help="Comma-separated modes to run (default: raw,C1,C4 for HE; "
                        "raw,C1 for BFCL; raw,C1 for GSM8K)")
    p.add_argument("--n", type=int, default=None,
                   help="Task count (default: 20 HE, 30 BFCL, 50 GSM8K)")
    p.add_argument("--raw-max-tokens", type=int, default=None,
                   help="If set, pin ChatOpenAI.max_tokens for raw mode (default: env SMBOOST_OPENAI_MAX_TOKENS or 512)")
    args = p.parse_args()

    from benchmarks.gates.runner import run_gate, GateConfig

    if args.stage == "HE":
        default_modes = ["raw", "C1", "C4"]
        bench = "humaneval_plus"
        default_n = 20
    elif args.stage == "BFCL":
        default_modes = ["raw", "C1"]
        bench = "bfcl_simple"
        default_n = 30
    else:  # GSM8K
        # Master plan: C5/C6 added once those agents merge. For this agent's
        # slice we ship raw + C1, the same pattern BFCL uses pre-merge.
        default_modes = ["raw", "C1"]
        bench = "gsm8k"
        default_n = 50

    modes = args.modes.split(",") if args.modes else default_modes
    n = args.n if args.n is not None else default_n

    raw_cap = args.raw_max_tokens
    if raw_cap is None:
        try:
            raw_cap = int(os.environ.get("SMBOOST_OPENAI_MAX_TOKENS", "512"))
        except ValueError:
            raw_cap = 512

    # Only patch for HE (BFCL + GSM8K raw runners already honor max_tokens
    # via their own `_make_raw_llm` indirection).
    if args.stage == "HE" and "raw" in modes:
        _install_raw_max_tokens_patch(raw_cap)

    print(f"[driver] stage={args.stage} model={args.model} bench={bench} n={n} modes={modes} "
          f"out={args.out_csv} raw_max_tokens={raw_cap}", flush=True)

    out_csv = Path(args.out_csv)
    overall_start = time.monotonic()

    for mode in modes:
        cfg = GateConfig(
            name=f"{args.stage}_{mode}",
            bench=bench,
            n=n,
            configs=[(args.model, mode)],
        )
        t0 = time.monotonic()
        print(f"[driver] === starting mode={mode} ===", flush=True)
        rows = run_gate(cfg)
        elapsed = time.monotonic() - t0
        _append_rows(out_csv, rows)
        _summary(rows, f"{args.model}/{mode}")
        print(f"[driver] === mode={mode} done in {elapsed:.1f}s; "
              f"appended {len(rows)} rows -> {out_csv} ===", flush=True)

    total = time.monotonic() - overall_start
    print(f"[driver] all modes done in {total:.1f}s ({total/60:.1f}m)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
