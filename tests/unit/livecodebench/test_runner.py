from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from benchmarks.livecodebench.runner import run_one_cell
from smboost.harness.state import StepOutput


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
    assert "max_tokens" in lines[0]
    assert "failure_bucket" in lines[0]


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


def test_runner_records_budget_and_failure_bucket(tmp_path, monkeypatch):
    monkeypatch.setenv("SMBOOST_OPENAI_MAX_TOKENS", "128")

    class AgentWithVerifyFailure:
        _memory = None

        def set_memory_log(self, p):
            pass

        def close_memory(self):
            pass

        def run(self, task, task_metadata=None):
            result = MagicMock()
            result.status = "failed"
            result.trace = [
                MagicMock(node="generate", output="import sys", passed=True),
                MagicMock(node="verify", output="FAIL: SyntaxError: invalid syntax", passed=False),
            ]
            result.stats.retry_count = 3
            result.stats.fallback_triggers = 0
            result.stats.total_latency_s = 0.2
            return result

    tasks = [{"task_id": "lcb_1", "prompt": "p", "testtype": "stdin",
              "test_cases": [], "entry_point": "", "difficulty": "hard",
              "source": "unit", "canonical_solution": ""}]
    shard = tmp_path / "shard.csv"
    with patch("benchmarks.livecodebench.runner.sandbox_run") as sandbox_run:
        sandbox_run.return_value = {
            "passed": False,
            "traceback": "SyntaxError: invalid syntax",
            "stdout": "",
            "stderr": "",
            "duration_ms": 5,
        }
        run_one_cell("C1", lambda m, s: AgentWithVerifyFailure(), tasks, model="qwen3.5:2b", seed=0,
                     shard_path=shard, memory_log_path=None)
    header, row = shard.read_text().strip().splitlines()
    assert "max_tokens" in header
    assert "failure_bucket" in header
    assert ",128," in row
    assert "syntax_truncation" in row


def test_runner_uses_ground_truth_sandbox_for_pass_column(tmp_path):
    class ParseOnlyAgent:
        _memory = None

        def set_memory_log(self, p):
            pass

        def close_memory(self):
            pass

        def run(self, task, task_metadata=None):
            result = MagicMock()
            result.status = "success"
            result.output = "PASS"
            result.trace = [
                StepOutput(node="generate", model="qwen3.5:0.8b",
                           output="import sys\nprint(1)\n", confidence=1.0, passed=True),
                StepOutput(node="verify", model="qwen3.5:0.8b",
                           output="PASS", confidence=1.0, passed=True),
            ]
            result.stats.retry_count = 0
            result.stats.fallback_triggers = 0
            result.stats.total_latency_s = 0.1
            return result

    tasks = [{"task_id": "lcb_1", "prompt": "p", "testtype": "stdin",
              "test_code": '[{\"testtype\":\"stdin\",\"input\":\"\",\"output\":\"2\\n\"}]',
              "test_cases": [], "entry_point": "", "difficulty": "hard",
              "source": "unit", "canonical_solution": ""}]
    shard = tmp_path / "shard.csv"

    with patch("benchmarks.livecodebench.runner.sandbox_run") as sandbox_run:
        sandbox_run.return_value = {
            "passed": False,
            "traceback": "Output mismatch. Expected 2, got 1",
            "stdout": "1\n",
            "stderr": "",
            "duration_ms": 5,
        }
        run_one_cell("C4", lambda m, s: ParseOnlyAgent(), tasks, model="qwen3.5:0.8b", seed=0,
                     shard_path=shard, memory_log_path=None)

    header, row = shard.read_text().strip().splitlines()
    assert "passed" in header
    assert row.split(",")[4] == "0"
    assert "logic_or_output_mismatch" in row


def test_runner_evaluates_last_generate_output_instead_of_result_output(tmp_path):
    class AgentWithPassString:
        _memory = None

        def set_memory_log(self, p):
            pass

        def close_memory(self):
            pass

        def run(self, task, task_metadata=None):
            result = MagicMock()
            result.status = "success"
            result.output = "PASS"
            result.trace = [
                StepOutput(node="generate", model="qwen3.5:2b",
                           output="import sys\nprint(2)\n", confidence=1.0, passed=True),
                StepOutput(node="verify", model="qwen3.5:2b",
                           output="PASS", confidence=1.0, passed=True),
            ]
            result.stats.retry_count = 0
            result.stats.fallback_triggers = 0
            result.stats.total_latency_s = 0.1
            return result

    tasks = [{"task_id": "lcb_1", "prompt": "p", "testtype": "stdin",
              "test_code": '[{\"testtype\":\"stdin\",\"input\":\"\",\"output\":\"2\\n\"}]',
              "test_cases": [], "entry_point": "", "difficulty": "hard",
              "source": "unit", "canonical_solution": ""}]
    shard = tmp_path / "shard.csv"

    with patch("benchmarks.livecodebench.runner.sandbox_run") as sandbox_run:
        sandbox_run.return_value = {
            "passed": True,
            "traceback": "",
            "stdout": "2\n",
            "stderr": "",
            "duration_ms": 5,
        }
        run_one_cell("C2", lambda m, s: AgentWithPassString(), tasks, model="qwen3.5:2b", seed=0,
                     shard_path=shard, memory_log_path=None)

    sandbox_run.assert_called_once()
    assert sandbox_run.call_args.args[0] == "import sys\nprint(2)\n"
