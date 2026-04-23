"""
Pipeline smoke test: 1 stdin + 1 functional task, full trace printed.
Usage:
  python3 scripts/smoke_pipeline.py 0.8b
  python3 scripts/smoke_pipeline.py 2b
Requires llama.cpp server on localhost:8000 serving the specified model.
"""
import os
import sys, textwrap
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

_MODEL_MAP = {
    "0.8b": "qwen3.5:0.8b",
    "2b":   "qwen3.5:2b",
    "4b":   "qwen3.5:4b",
    "9b":   "qwen3.5:9b",
}

size = sys.argv[1] if len(sys.argv) > 1 else "0.8b"
MODEL = _MODEL_MAP.get(size, size)
BACKEND = os.getenv("SMBOOST_LLM_BACKEND", "server")

from benchmarks.livecodebench.loader import load_livecodebench_tasks
from benchmarks.livecodebench.conditions import CONDITIONS

tasks = load_livecodebench_tasks()
stdin_task = next(t for t in tasks if t["testtype"] == "stdin")
func_task  = next(t for t in tasks if t["testtype"] == "functional")


def run_and_print(label: str, task: dict):
    print(f"\n{'='*70}")
    print(f"  {label}  |  task_id={task['task_id']}  testtype={task['testtype']}  model={MODEL}")
    print(f"{'='*70}")
    print(f"[PROMPT first 400 chars]\n{task['prompt'][:400]}")
    if task.get("entry_point"):
        print(f"\n[ENTRY_POINT]\n{task['entry_point']}")
    print()

    # C4: no scorer/memory/shrinkage — fastest path for pipeline validation
    agent = CONDITIONS["C4"](MODEL, seed=0)
    result = agent.run(task["prompt"], task_metadata=task)

    print(f"[RESULT]  status={result.status}  retries={result.stats.retry_count}  "
          f"fallbacks={result.stats.fallback_triggers}  latency={result.stats.total_latency_s:.1f}s")
    print()

    for i, step in enumerate(result.trace):
        node_ok = "✓" if step.passed else "✗"
        print(f"  [{node_ok}] step {i+1}  node={step.node}  model={step.model}")
        out = (step.output or "(empty)").strip()
        print(textwrap.indent(out[:1000], "      "))
        print()

    agent.close_memory()


print(f"backend={BACKEND} model={MODEL} tasks={len(tasks)}")
run_and_print("STDIN", stdin_task)
run_and_print("FUNCTIONAL", func_task)
