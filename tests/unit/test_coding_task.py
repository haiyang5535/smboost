from unittest.mock import MagicMock, patch
from smboost.tasks.coding import CodingTaskGraph

def _make_state(step_outputs=None):
    from smboost.harness.state import HarnessState
    return {
        "task": "write hello world to /tmp/hw.py",
        "model": "qwen3.5:2b",
        "fallback_chain": ["qwen3.5:2b"],
        "step_outputs": step_outputs or [],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 0,
        "shrinkage_level": 0,
        "status": "running",
        "final_output": None,
    }

def test_node_names_are_plan_execute_verify():
    tg = CodingTaskGraph()
    assert tg.node_names == ["plan", "execute", "verify"]

def test_get_node_fn_returns_callable_for_each_node():
    tg = CodingTaskGraph()
    for name in tg.node_names:
        fn = tg.get_node_fn(name)
        assert callable(fn)

def test_plan_node_calls_llm_and_returns_content():
    tg = CodingTaskGraph()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="1. Write file\n2. Verify")
    fn = tg.get_node_fn("plan")
    result = fn(_make_state(), mock_llm)
    assert "Write file" in result
    mock_llm.invoke.assert_called_once()

def test_verify_node_calls_llm_and_returns_content():
    from smboost.harness.state import StepOutput
    step = StepOutput(node="execute", model="qwen3.5:2b", output="file written", confidence=1.0, passed=True)
    tg = CodingTaskGraph()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="PASS: file was created")
    fn = tg.get_node_fn("verify")
    result = fn(_make_state(step_outputs=[step]), mock_llm)
    assert "PASS" in result


import pytest

@pytest.mark.parametrize("level,expected_shorter", [
    (1, True),
    (2, True),
    (3, True),
])
def test_plan_prompt_shrinks_with_level(level, expected_shorter):
    tg = CodingTaskGraph()
    mock_llm = MagicMock()
    captured = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="plan")
    )

    state_l0 = _make_state()
    state_ln = {**_make_state(), "shrinkage_level": level}

    tg.get_node_fn("plan")(state_l0, mock_llm)
    prompt_l0 = captured[0]
    captured.clear()

    tg.get_node_fn("plan")(state_ln, mock_llm)
    prompt_ln = captured[0]

    assert len(prompt_ln) < len(prompt_l0)


@pytest.mark.parametrize("level,expected_shorter", [
    (1, True),
    (2, True),
    (3, True),
])
def test_verify_prompt_shrinks_with_level(level, expected_shorter):
    from smboost.harness.state import StepOutput
    step = StepOutput(node="execute", model="qwen3.5:2b", output="output", confidence=1.0, passed=True)
    tg = CodingTaskGraph()
    mock_llm = MagicMock()
    captured = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="PASS")
    )

    state_l0 = _make_state(step_outputs=[step])
    state_ln = {**_make_state(step_outputs=[step]), "shrinkage_level": level}

    tg.get_node_fn("verify")(state_l0, mock_llm)
    prompt_l0 = captured[0]
    captured.clear()

    tg.get_node_fn("verify")(state_ln, mock_llm)
    prompt_ln = captured[0]

    assert len(prompt_ln) < len(prompt_l0)
