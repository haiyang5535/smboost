from unittest.mock import MagicMock, patch
from smboost import HarnessAgent, InvariantSuite
from smboost.harness.result import HarnessResult
from smboost.harness.state import StepOutput

def _make_final_state(status="success"):
    step = StepOutput(node="plan", model="qwen3.5:2b", output="done", confidence=1.0, passed=True)
    return {
        "task": "test",
        "model": "qwen3.5:2b",
        "fallback_chain": ["qwen3.5:2b", "qwen3.5:8b"],
        "step_outputs": [step],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 1,
        "status": status,
        "final_output": "done",
    }

def test_harness_agent_init_with_defaults():
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
    )
    assert agent.model == "qwen3.5:2b"
    assert agent.fallback_chain == ["qwen3.5:2b"]

def test_harness_agent_init_with_fallback_chain():
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        fallback_chain=["qwen3.5:2b", "qwen3.5:8b"],
    )
    assert agent.fallback_chain == ["qwen3.5:2b", "qwen3.5:8b"]

def test_run_returns_harness_result():
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
    )
    with patch.object(agent._harness, "invoke", return_value=_make_final_state("success")):
        result = agent.run("write hello world")

    assert isinstance(result, HarnessResult)
    assert result.status == "success"
    assert result.output == "done"
    assert len(result.trace) == 1
    assert result.stats.model_used == "qwen3.5:2b"

def test_run_failed_result_has_failed_status():
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
    )
    failed_state = _make_final_state("failed")
    failed_state["final_output"] = None
    with patch.object(agent._harness, "invoke", return_value=failed_state):
        result = agent.run("impossible task")

    assert result.status == "failed"
    assert result.output == ""

def test_run_stats_retry_count_from_state():
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
    )
    state = _make_final_state("success")
    state["retry_count"] = 2
    state["fallback_index"] = 0
    with patch.object(agent._harness, "invoke", return_value=state):
        result = agent.run("some task")

    assert result.stats.retry_count == 2


def test_run_stats_fallback_triggers_from_state():
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        fallback_chain=["qwen3.5:2b", "qwen3.5:8b"],
    )
    state = _make_final_state("success")
    state["retry_count"] = 0
    state["fallback_index"] = 1
    state["model"] = "qwen3.5:8b"
    with patch.object(agent._harness, "invoke", return_value=state):
        result = agent.run("some task")

    assert result.stats.fallback_triggers == 1
