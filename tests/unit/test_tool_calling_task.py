from unittest.mock import MagicMock
from langchain_core.tools import tool
from smboost.tasks.tool_calling import ToolCallingTaskGraph

@tool
def mock_tool(query: str) -> str:
    """A mock tool for testing."""
    return f"result for {query}"

def _make_state(step_outputs=None):
    return {
        "task": "look up the weather",
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

def test_node_names_are_plan_dispatch_verify():
    tg = ToolCallingTaskGraph(tools=[mock_tool])
    assert tg.node_names == ["plan", "dispatch", "verify"]

def test_get_node_fn_returns_callable_for_each_node():
    tg = ToolCallingTaskGraph(tools=[mock_tool])
    for name in tg.node_names:
        assert callable(tg.get_node_fn(name))

def test_plan_node_calls_llm():
    tg = ToolCallingTaskGraph(tools=[mock_tool])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="1. Call mock_tool")
    result = tg.get_node_fn("plan")(_make_state(), mock_llm)
    assert "mock_tool" in result or result  # content returned
    mock_llm.invoke.assert_called_once()

def test_verify_node_calls_llm():
    from smboost.harness.state import StepOutput
    step = StepOutput(node="dispatch", model="qwen3.5:2b", output="result found", confidence=1.0, passed=True)
    tg = ToolCallingTaskGraph(tools=[mock_tool])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="PASS")
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), mock_llm)
    assert result == "PASS"


import pytest

@pytest.mark.parametrize("level", [1, 2, 3])
def test_plan_prompt_shrinks_with_level(level):
    tg = ToolCallingTaskGraph(tools=[mock_tool])
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


@pytest.mark.parametrize("level", [1, 2, 3])
def test_verify_prompt_shrinks_with_level(level):
    from smboost.harness.state import StepOutput
    step = StepOutput(node="dispatch", model="qwen3.5:2b", output="result found", confidence=1.0, passed=True)
    tg = ToolCallingTaskGraph(tools=[mock_tool])
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
