import pytest

from benchmarks.livecodebench.conditions import CONDITIONS
from benchmarks.livecodebench.loader import load_livecodebench_tasks


@pytest.mark.integration
def test_c1_completes_3_tasks_on_4b():
    tasks = load_livecodebench_tasks(n=3)
    assert len(tasks) == 3
    agent = CONDITIONS["C1"]("qwen3.5:4b", 0)
    results = []
    try:
        for t in tasks:
            r = agent.run(t["prompt"], task_metadata=t)
            results.append(r)
    finally:
        agent.close_memory()
    assert len(results) == 3
    assert all(r.status in ("success", "failed") for r in results)
