from smboost.harness.state import HarnessState, StepOutput

def test_step_output_fields():
    s = StepOutput(node="plan", model="qwen3.5:2b", output="ok", confidence=1.0, passed=True)
    assert s.node == "plan"
    assert s.confidence == 1.0
    assert s.passed is True

def test_harness_state_is_dict_like():
    state: HarnessState = {
        "task": "write a function",
        "model": "qwen3.5:2b",
        "fallback_chain": ["qwen3.5:2b", "qwen3.5:8b"],
        "step_outputs": [],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 0,
        "status": "running",
        "final_output": None,
    }
    assert state["task"] == "write a function"
    assert state["status"] == "running"
