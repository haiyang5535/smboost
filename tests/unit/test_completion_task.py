from unittest.mock import MagicMock

import pytest

from smboost.harness.state import StepOutput
from smboost.tasks.completion import CompletionTaskGraph, _clean


def _make_state(step_outputs=None, task="def add(a, b):\n    ", shrinkage_level=0):
    return {
        "task": task,
        "model": "qwen3.5:0.8b",
        "fallback_chain": ["qwen3.5:0.8b"],
        "step_outputs": step_outputs or [],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 0,
        "shrinkage_level": shrinkage_level,
        "status": "running",
        "final_output": None,
    }


def test_node_names_are_generate_verify():
    tg = CompletionTaskGraph()
    assert tg.node_names == ["generate", "verify"]


def test_get_node_fn_returns_callable_for_each_node():
    tg = CompletionTaskGraph()
    for name in tg.node_names:
        assert callable(tg.get_node_fn(name))


def test_get_node_fn_raises_on_unknown_node():
    tg = CompletionTaskGraph()
    with pytest.raises(ValueError):
        tg.get_node_fn("nonexistent")


def test_generate_node_calls_llm_and_returns_cleaned_content():
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="    return a + b")
    result = tg.get_node_fn("generate")(_make_state(), mock_llm)
    assert result == "    return a + b"
    mock_llm.invoke.assert_called_once()


def test_generate_node_strips_think_block():
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="<think>plan</think>\n    return a + b")
    result = tg.get_node_fn("generate")(_make_state(), mock_llm)
    assert "<think>" not in result
    assert "return a + b" in result


def test_generate_node_unwraps_fenced_code():
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="```python\n    return a + b\n```")
    result = tg.get_node_fn("generate")(_make_state(), mock_llm)
    assert "```" not in result
    assert "return a + b" in result


def test_verify_node_returns_pass_on_parseable_code():
    step = StepOutput(node="generate", model="qwen3.5:0.8b",
                      output="def f(a, b):\n    return a + b", confidence=1.0, passed=True)
    tg = CompletionTaskGraph()
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result == "PASS"


def test_verify_node_returns_fail_on_syntax_error():
    step = StepOutput(node="generate", model="qwen3.5:0.8b",
                      output="def f(a, b):\n    return a +", confidence=1.0, passed=True)
    tg = CompletionTaskGraph()
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result.startswith("FAIL")


def test_verify_node_returns_fail_on_empty_completion():
    step = StepOutput(node="generate", model="qwen3.5:0.8b",
                      output="   \n", confidence=1.0, passed=True)
    tg = CompletionTaskGraph()
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result.startswith("FAIL")


def test_verify_node_does_not_call_llm():
    step = StepOutput(node="generate", model="qwen3.5:0.8b",
                      output="    return a + b", confidence=1.0, passed=True)
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    tg.get_node_fn("verify")(_make_state(step_outputs=[step]), mock_llm)
    mock_llm.invoke.assert_not_called()


@pytest.mark.parametrize("level", [1, 2, 3])
def test_generate_prompt_shrinks_with_level(level):
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    captured = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="x")
    )
    long_task = "def foo(a, b, c):\n    " + ("# comment\n    " * 100)
    tg.get_node_fn("generate")(_make_state(task=long_task), mock_llm)
    prompt_l0 = captured[0]
    captured.clear()
    tg.get_node_fn("generate")(
        _make_state(task=long_task, shrinkage_level=level), mock_llm
    )
    prompt_ln = captured[0]
    # higher levels either use a different wrapping or truncate
    assert prompt_ln != prompt_l0


def test_clean_strips_think_and_unwraps_fence():
    assert _clean("<think>x</think>\n    return 1") == "    return 1"
    assert _clean("```python\nreturn 1\n```") == "return 1\n"
