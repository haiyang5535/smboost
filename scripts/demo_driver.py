"""SMBoost YC demo driver.

Runs a canonical math-word-problem task end-to-end through the harness and
writes a JSONL trace to ``frontend/demo_trace.jsonl`` for the real-time
trace visualizer at ``frontend/index.html``.

Usage
-----
    # default: pick first available condition from preferred list, use 2B server
    python3 scripts/demo_driver.py

    # explicit condition (falls back if the requested condition is not yet
    # registered — e.g. C5/C6 land only after Agents 2 & 4 merge):
    python3 scripts/demo_driver.py --condition C1
    python3 scripts/demo_driver.py --condition C5

    # with the llama.cpp server already running on :8000
    SMBOOST_LLM_BACKEND=server \
    SMBOOST_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    SMBOOST_OPENAI_API_KEY=sk-no-key \
    python3 scripts/demo_driver.py --condition C1

If the harness cannot be run (e.g. server down, import fails), the driver
writes a *canned* trace derived from a known-good run so the frontend still
renders a compelling demo. This behaviour can be forced via ``--canned``.

Output
------
Prints one line:
    demo task: <task_id>, condition: <C?>, result: <pass|fail>, duration: <Xs>

Trace JSONL is written to ``frontend/demo_trace.jsonl`` (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


DEMO_TASK_ID = "demo_janet_eggs"

DEMO_TASK_PROMPT = """\
Janet owns 3 chickens. Each chicken lays 2 eggs per day.
Every day, Janet eats 2 eggs for breakfast and sells the remaining eggs at
the farmer's market for $3 per egg. How much money does Janet make in one
week (7 days)?

Write a Python function:

    def solve() -> int:
        ...

that returns the total dollars Janet makes in one week.
"""

DEMO_TASK_METADATA = {
    "task_id": DEMO_TASK_ID,
    "testtype": "functional",
    "entry_point": "solve",
    "test_cases": [{"input": "", "output": "84"}],
}

PREFERRED_CONDITIONS = ["C5", "C6", "C1"]

_MODEL_MAP = {
    "0.8b": "qwen3.5:0.8b",
    "2b": "qwen3.5:2b",
}


def _pick_available_condition(requested: str | None) -> tuple[str, object] | None:
    """Return (condition_id, build_condition_callable) or None if import fails."""
    try:
        from benchmarks.conditions import CONDITION_NAMES, build_condition
    except Exception as exc:
        print(f"[demo_driver] cannot import benchmarks.conditions: {exc}", file=sys.stderr)
        return None

    if requested:
        if requested in CONDITION_NAMES:
            return requested, build_condition
        print(f"[demo_driver] condition {requested!r} not registered; "
              f"known: {CONDITION_NAMES}", file=sys.stderr)
        # fall through to auto-pick

    for cand in PREFERRED_CONDITIONS:
        if cand in CONDITION_NAMES:
            return cand, build_condition
    # Last resort: first registered
    if CONDITION_NAMES:
        return CONDITION_NAMES[0], build_condition
    return None


def _run_live(model: str, condition: str, build_condition, trace_path: Path) -> dict:
    """Run a real harness invocation against the configured LLM backend."""
    # Ensure trace file is fresh
    if trace_path.exists():
        trace_path.unlink()
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    agent = build_condition(condition, model=model, task_graph_kind="completion")
    # HarnessAgent writes JSONL traces when given trace_log_path at construction
    # time. Our factory doesn't expose that, so set it on the returned object.
    agent.trace_log_path = trace_path

    t0 = time.monotonic()
    try:
        result = agent.run(
            DEMO_TASK_PROMPT,
            task_metadata=DEMO_TASK_METADATA,
            task_id=DEMO_TASK_ID,
            condition=condition,
        )
        duration = time.monotonic() - t0
        return {
            "passed": result.status == "success",
            "duration": duration,
            "retries": result.stats.retry_count,
            "mode": "live",
        }
    finally:
        try:
            agent.close_memory()
        except Exception:
            pass


def _write_canned(trace_path: Path, model: str, condition: str) -> dict:
    """Emit a realistic-looking trace without invoking the LLM.

    Used when the server is unreachable or the user passes ``--canned``.
    Mirrors the exact JSONL shape that ``HarnessAgent._emit_*`` writes.
    """
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = "demo" + str(int(time.time()))[-8:]
    now = time.time()

    lines = []
    # run_start
    lines.append({
        "schema_version": 1,
        "run_id": run_id,
        "task_id": DEMO_TASK_ID,
        "model": model,
        "condition": condition,
        "task": DEMO_TASK_PROMPT,
        "event": "run_start",
    })

    def step(idx, node, output, passed, retry_count, shrinkage_level, verify_tb=None):
        return {
            "run_id": run_id,
            "task_id": DEMO_TASK_ID,
            "model": model,
            "condition": condition,
            "step_idx": idx,
            "node": node,
            "entry_ts": now + idx * 0.9,
            "exit_ts": now + idx * 0.9 + 0.4,
            "retry_count": retry_count,
            "shrinkage_level": shrinkage_level,
            "scorer_confidence": None,
            "input": {"prompt": None, "budget": None},
            "output": {"code": output, "trunc": False},
            "verify": {
                "kind": None,
                "passed": passed,
                "traceback": verify_tb,
            },
            "fallback_triggered": False,
            "schema_version": 1,
        }

    # Attempt 1: wrong (forgot to subtract the 2 eggs Janet eats)
    attempt1_code = (
        "def solve() -> int:\n"
        "    eggs_per_day = 3 * 2\n"
        "    return eggs_per_day * 3 * 7\n"
    )
    lines.append(step(0, "generate", attempt1_code, None, 0, 0))
    lines.append(step(
        1, "verify", None, False, 0, 0,
        verify_tb=(
            "Traceback (most recent call last):\n"
            '  File "/tmp/demo.py", line 5, in <module>\n'
            "    assert solve() == 84\n"
            "AssertionError"
        ),
    ))

    # Attempt 2: correct
    attempt2_code = (
        "def solve() -> int:\n"
        "    eggs_per_day = 3 * 2\n"
        "    eaten = 2\n"
        "    sold = eggs_per_day - eaten\n"
        "    per_day = sold * 3\n"
        "    return per_day * 7\n"
    )
    lines.append(step(2, "generate", attempt2_code, None, 1, 0))
    lines.append(step(3, "verify", None, True, 1, 0))

    # summary
    wall_ms = int((time.time() - now) * 1000) + 3600
    lines.append({
        "run_id": run_id,
        "event": "summary",
        "passed": True,
        "retries": 1,
        "wall_ms": wall_ms,
        "final_code": attempt2_code,
    })

    with trace_path.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj, ensure_ascii=False))
            fh.write("\n")

    return {"passed": True, "duration": 3.6, "retries": 1, "mode": "canned"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", default=None, help="Condition id (C1..C6). Default: auto-pick preferred.")
    p.add_argument("--model", default="2b", help="Model size tag (0.8b | 2b) or full model id. Default: 2b.")
    p.add_argument("--trace", default=str(_ROOT / "frontend" / "demo_trace.jsonl"),
                   help="Output JSONL trace path. Default: frontend/demo_trace.jsonl")
    p.add_argument("--canned", action="store_true",
                   help="Skip the LLM server and write a canned trace (still useful for the demo UI).")
    args = p.parse_args()

    model = _MODEL_MAP.get(args.model, args.model)
    trace_path = Path(args.trace)

    picked = _pick_available_condition(args.condition)
    if picked is None:
        print("[demo_driver] could not import conditions; using canned trace", file=sys.stderr)
        outcome = _write_canned(trace_path, model, args.condition or "C1")
        cond = args.condition or "C1"
    else:
        cond, build_condition = picked
        if args.canned:
            outcome = _write_canned(trace_path, model, cond)
        else:
            try:
                outcome = _run_live(model, cond, build_condition, trace_path)
            except Exception as exc:
                print(f"[demo_driver] live run failed ({type(exc).__name__}: {exc}); "
                      f"falling back to canned trace", file=sys.stderr)
                outcome = _write_canned(trace_path, model, cond)

    result_txt = "pass" if outcome["passed"] else "fail"
    print(
        f"demo task: {DEMO_TASK_ID}, "
        f"condition: {cond}, "
        f"result: {result_txt}, "
        f"duration: {outcome['duration']:.1f}s, "
        f"mode: {outcome['mode']}, "
        f"trace: {trace_path}"
    )
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
