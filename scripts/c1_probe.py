"""
C1-only probe: grounded verify, real pass rate, 10 tasks, model from argv.
Usage: python3 scripts/c1_probe.py 0.8b
"""
import sys, time
sys.path.insert(0, "src")

_MODEL_MAP = {"0.8b": "qwen3.5:0.8b", "2b": "qwen3.5:2b", "4b": "qwen3.5:4b"}
size = sys.argv[1] if len(sys.argv) > 1 else "0.8b"
MODEL = _MODEL_MAP.get(size, size)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10

from benchmarks.livecodebench.loader import load_livecodebench_tasks
from benchmarks.livecodebench.conditions import CONDITIONS

tasks = load_livecodebench_tasks(n=N)
agent = CONDITIONS["C1"](MODEL, seed=0)

passed = failed = 0
for i, task in enumerate(tasks):
    t0 = time.monotonic()
    result = agent.run(task["prompt"], task_metadata=task)
    elapsed = time.monotonic() - t0
    ok = result.status == "success"
    if ok:
        passed += 1
    else:
        failed += 1
    last_verify = next(
        (s.output[:120] for s in reversed(result.trace) if s.node == "verify"), ""
    )
    print(
        f"[{i+1:2d}/{N}] {task['task_id']:12s} {task['testtype']:10s} "
        f"{'PASS' if ok else 'FAIL'} "
        f"retries={result.stats.retry_count} lat={elapsed:.0f}s  {last_verify!r}",
        flush=True,
    )

agent.close_memory()
print(f"\nRESULT  pass={passed}/{N} ({100*passed//N}%)  fail={failed}/{N}  model={MODEL}")
