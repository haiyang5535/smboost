from pathlib import Path
from unittest.mock import MagicMock

from benchmarks.livecodebench.runner import run_one_cell


def _fake_agent_factory(_model, _seed):
    agent = MagicMock()
    agent._memory = None

    def run(task, task_metadata=None):
        result = MagicMock()
        result.status = "success"
        result.output = "def f(): return 1"
        result.trace = []
        result.stats.retry_count = 0
        result.stats.fallback_triggers = 0
        result.stats.total_latency_s = 0.5
        return result

    agent.run.side_effect = run
    agent.close_memory = MagicMock()
    agent.set_memory_log = MagicMock()
    return agent


def test_runner_writes_one_row_per_task(tmp_path):
    tasks = [{"task_id": "lcb_1", "prompt": "p", "testtype": "functional",
              "test_cases": [], "entry_point": "f", "difficulty": "hard",
              "source": "unit", "canonical_solution": ""}]
    shard = tmp_path / "shard.csv"
    run_one_cell("C1", _fake_agent_factory, tasks, model="qwen3.5:4b", seed=0,
                 shard_path=shard, memory_log_path=None)
    lines = shard.read_text().strip().splitlines()
    assert len(lines) == 2  # header + 1 row
    assert "lcb_1" in lines[1]


def test_runner_resumes_and_skips_done_tasks(tmp_path):
    tasks = [
        {"task_id": "lcb_1", "prompt": "p", "testtype": "functional",
         "test_cases": [], "entry_point": "f", "difficulty": "hard",
         "source": "unit", "canonical_solution": ""},
        {"task_id": "lcb_2", "prompt": "p", "testtype": "functional",
         "test_cases": [], "entry_point": "f", "difficulty": "hard",
         "source": "unit", "canonical_solution": ""},
    ]
    shard = tmp_path / "shard.csv"
    shard.write_text(
        "condition,model,seed,task_id,passed,duration_ms,retries,fallback_triggered,"
        "grounded_verify_result,memory_hits,usd_cost,wall_clock_ms\n"
        "C1,qwen3.5:4b,0,lcb_1,1,100,0,0,PASS,0,0.0,100\n"
    )
    run_one_cell("C1", _fake_agent_factory, tasks, model="qwen3.5:4b", seed=0,
                 shard_path=shard, memory_log_path=None)
    lines = shard.read_text().strip().splitlines()
    assert len(lines) == 3  # header + resumed + new
    assert any("lcb_2" in ln for ln in lines)


def test_runner_records_memory_hits(tmp_path):
    class AgentWithMem:
        def __init__(self):
            class Mem:
                hits = 0
            self._memory = Mem()

        def set_memory_log(self, p):
            pass

        def close_memory(self):
            pass

        def run(self, task, task_metadata=None):
            self._memory.hits += 2
            result = MagicMock()
            result.status = "success"
            result.trace = []
            result.stats.retry_count = 0
            result.stats.fallback_triggers = 0
            result.stats.total_latency_s = 0.1
            return result

    def factory(m, s):
        return AgentWithMem()

    tasks = [{"task_id": "lcb_1", "prompt": "p", "testtype": "functional",
              "test_cases": [], "entry_point": "f", "difficulty": "hard",
              "source": "unit", "canonical_solution": ""}]
    shard = tmp_path / "shard.csv"
    run_one_cell("C1", factory, tasks, model="qwen3.5:4b", seed=0,
                 shard_path=shard, memory_log_path=None)
    line = shard.read_text().strip().splitlines()[1]
    assert ",2," in line  # memory_hits column value is 2
