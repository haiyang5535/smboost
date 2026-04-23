"""
C1-only probe: grounded verify, real pass rate, 10 tasks, model from argv.
Usage: python3 scripts/c1_probe.py 0.8b
"""
import sys, time
import os
from collections import Counter
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

_MODEL_MAP = {"0.8b": "qwen3.5:0.8b", "2b": "qwen3.5:2b", "4b": "qwen3.5:4b"}
from benchmarks.livecodebench.loader import load_livecodebench_tasks
from benchmarks.livecodebench.conditions import CONDITIONS


def _normalize_backend() -> str:
    return os.getenv("SMBOOST_LLM_BACKEND", "server").lower()


def _banner(backend: str, model: str, tasks: int) -> str:
    return f"BACKEND={backend}  MODEL={model}  TASKS={tasks}"


def _classify_failure(full_verify_output: str) -> str:
    if "got ''" in full_verify_output:
        return "empty_output"
    if "SyntaxError" in full_verify_output or "IndentationError" in full_verify_output:
        return "syntax_truncation"
    if "OutputMismatch" in full_verify_output or "AssertionError" in full_verify_output:
        return "logic_or_output_mismatch"
    return "other_runtime"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    size = args[0] if len(args) > 0 else "0.8b"
    model = _MODEL_MAP.get(size, size)
    n = int(args[1]) if len(args) > 1 else 10
    backend = _normalize_backend()

    tasks = load_livecodebench_tasks(n=n)
    agent = CONDITIONS["C1"](model, seed=0)

    passed = failed = 0
    failure_buckets = Counter()
    infra_failure = False

    print(_banner(backend, model, n), flush=True)
    for i, task in enumerate(tasks):
        t0 = time.monotonic()
        retry_count = 0
        last_verify = ""
        try:
            result = agent.run(task["prompt"], task_metadata=task)
            elapsed = time.monotonic() - t0
            ok = result.status == "success"
            retry_count = result.stats.retry_count
            if ok:
                passed += 1
                failure_buckets["PASS"] += 1
            else:
                failed += 1
            last_verify = next(
                (s.output or "" for s in reversed(result.trace) if s.node == "verify"),
                "",
            )
            if not ok:
                failure_buckets[_classify_failure(last_verify)] += 1
        except Exception as exc:
            elapsed = time.monotonic() - t0
            failed += 1
            infra_failure = True
            last_verify = f"{type(exc).__name__}: {exc}"
            failure_buckets["other_runtime"] += 1
            ok = False
        print(
            f"[{i+1:2d}/{n}] {task['task_id']:12s} {task['testtype']:10s} "
            f"{'PASS' if ok else 'FAIL'} "
            f"retries={retry_count} lat={elapsed:.0f}s  {last_verify[:120]!r}",
            flush=True,
        )

    agent.close_memory()
    print(f"\nRESULT  pass={passed}/{n} ({100*passed//n}%)  fail={failed}/{n}  model={model}")
    print(f"FAILURE_BUCKETS={dict(failure_buckets)}")
    return 1 if infra_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
