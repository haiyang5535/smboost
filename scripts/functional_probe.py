"""Quick probe: functional tasks only, C1, to validate stub prompt fix."""
import sys, time
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
_MODEL_MAP = {"0.8b": "qwen3.5:0.8b", "2b": "qwen3.5:2b"}
size = sys.argv[1] if len(sys.argv) > 1 else "0.8b"
MODEL = _MODEL_MAP.get(size, size)

from benchmarks.livecodebench.loader import load_livecodebench_tasks
from benchmarks.livecodebench.conditions import CONDITIONS

tasks = [t for t in load_livecodebench_tasks() if t["testtype"] == "functional"]
agent = CONDITIONS["C1"](MODEL, seed=0)
passed = 0
for i, task in enumerate(tasks):
    t0 = time.monotonic()
    result = agent.run(task["prompt"], task_metadata=task)
    elapsed = time.monotonic() - t0
    ok = result.status == "success"
    if ok: passed += 1
    last_verify = next(
        (s.output[:150] for s in reversed(result.trace) if s.node == "verify"), ""
    )
    print(f"[{i+1:2d}/{len(tasks)}] {task['task_id']:12s} {'PASS' if ok else 'FAIL'} "
          f"retries={result.stats.retry_count} lat={elapsed:.0f}s  {last_verify!r}", flush=True)
agent.close_memory()
print(f"\nRESULT  pass={passed}/{len(tasks)} ({100*passed//len(tasks) if tasks else 0}%)")
