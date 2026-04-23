from unittest.mock import MagicMock

import pytest

from smboost.harness.state import StepOutput
from smboost.tasks.completion import CompletionTaskGraph, _clean
from smboost.tasks.stdin_skeletons import shrink_small_model_stdin_problem


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


@pytest.mark.parametrize(
    "model,shrinkage_level",
    [
        ("qwen3.5:0.8b", 0),
        ("qwen3.5:0.8b", 1),
        ("qwen3.5:0.8b", 2),
        ("qwen3.5:0.8b", 3),
        ("qwen3.5:2b", 0),
        ("qwen3.5:2b", 1),
        ("qwen3.5:2b", 2),
        ("qwen3.5:2b", 3),
    ],
)
def test_generate_node_small_model_stdin_uses_skeleton_prompt(model, shrinkage_level):
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    captured = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="import sys\n\ndef solve():\n    pass")
    )
    state = _make_state(task="problem text", shrinkage_level=shrinkage_level)
    state["model"] = model
    state["task_metadata"] = {"testtype": "stdin", "task_id": "abc301_e"}

    tg.get_node_fn("generate")(state, mock_llm)

    prompt = captured[0]
    assert "Output only Python code." in prompt
    assert "Start with `import sys` on the first line." in prompt
    assert "def solve():" in prompt
    if shrinkage_level <= 1:
        assert 'sys.stdout.write("\\n".join(out))' in prompt
        assert "sys.stdin.read" in prompt
        assert "Solve the competitive-programming problem below." in prompt
        assert "Use this exact program skeleton and replace only the parsing and logic." in prompt
        assert "Keep `solve()` and the final `sys.stdout.write` pattern." in prompt
        assert "Do not explain the approach." in prompt
        assert "If you output any text before the code, the answer is wrong." in prompt
        assert "# parse tokens from data" in prompt
        assert "# append each answer as a string into out" in prompt
    else:
        assert r"sys.stdout.write('\n'.join(out))" in prompt
        assert "sys.stdin.buffer.read().split()" in prompt
        assert "Use this compact skeleton." in prompt
        assert "# parse tokens from data" not in prompt
        assert "# append each answer as a string into out" not in prompt
        assert "Python stdin/stdout solution:" not in prompt


def test_generate_node_small_model_stdin_compact_prompt_shrinks_problem_text():
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    captured = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="import sys\n\ndef solve():\n    pass")
    )
    task = "problem text " * 400
    state = _make_state(task=task, shrinkage_level=2)
    state["model"] = "qwen3.5:2b"
    state["task_metadata"] = {"testtype": "stdin", "task_id": "abc301_e"}

    tg.get_node_fn("generate")(state, mock_llm)

    prompt_l2 = captured[-1]
    captured.clear()
    state["shrinkage_level"] = 3
    tg.get_node_fn("generate")(state, mock_llm)
    prompt_l3 = captured[-1]

    assert len(prompt_l3) < len(prompt_l2)


def test_generate_node_small_model_stdin_compact_prompt_keeps_io_spec():
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    captured = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="import sys\n\ndef solve():\n    pass")
    )
    intro = "problem intro " * 90
    task = (
        intro
        + "\n\nInput\nThe first line contains t.\nEach test case contains n and then n integers.\n\n"
        + "Output\nPrint one answer per test case.\n"
    )
    state = _make_state(task=task, shrinkage_level=2)
    state["model"] = "qwen3.5:2b"
    state["task_metadata"] = {"testtype": "stdin", "task_id": "1899_b"}

    tg.get_node_fn("generate")(state, mock_llm)

    prompt = captured[0]
    assert "The first line contains t." in prompt
    assert "Each test case contains n and then n integers." in prompt
    assert "Print one answer per test case." in prompt


def test_shrink_small_model_stdin_problem_prefers_text_near_input():
    problem = (
        ("story " * 80)
        + "\n\nLoading happens with consecutive groups of size k. "
        + "Find the maximum absolute difference between truck sums.\n\n"
        + "Input\nThe first line contains t.\nEach test case contains n and then n integers.\n\n"
        + "Output\nPrint one answer per test case.\n"
    )

    shrunk = shrink_small_model_stdin_problem(problem, max_chars=220)

    assert "Find the maximum absolute difference between truck sums." in shrunk
    assert "The first line contains t." in shrunk
    assert "story story story story story story story story story story" not in shrunk


def test_generate_node_small_model_stdin_compact_prompt_pushes_short_code():
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    captured = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="import sys\n\ndef solve():\n    pass")
    )
    state = _make_state(task="problem text", shrinkage_level=2)
    state["model"] = "qwen3.5:2b"
    state["task_metadata"] = {"testtype": "stdin", "task_id": "1899_b"}

    tg.get_node_fn("generate")(state, mock_llm)

    prompt = captured[0]
    assert "Keep the code very short." in prompt
    assert "Use short variable names." in prompt
    assert "Do not add blank lines." in prompt
    assert "Return a complete program." in prompt
    assert "Never output only statements or fragments." in prompt
    assert "Do not leave out empty." in prompt
    assert "Append answer strings to out." in prompt
    assert "sys.stdin.buffer.read().split()" in prompt


def test_generate_node_small_model_stdin_uses_compact_prompt_when_budget_is_tight(monkeypatch):
    monkeypatch.setenv("SMBOOST_OPENAI_MAX_TOKENS", "64")
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    captured = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="import sys\ndef solve():\n    pass")
    )
    state = _make_state(task="problem text", shrinkage_level=0)
    state["model"] = "qwen3.5:2b"
    state["task_metadata"] = {"testtype": "stdin", "task_id": "1899_b"}

    tg.get_node_fn("generate")(state, mock_llm)

    prompt = captured[0]
    assert "Use this compact skeleton." in prompt
    assert "Use this exact program skeleton and replace only the parsing and logic." not in prompt


def test_generate_node_small_model_stdin_uses_more_aggressive_compact_prompt_after_verify_failure(monkeypatch):
    monkeypatch.setenv("SMBOOST_OPENAI_MAX_TOKENS", "64")
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    captured = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="import sys\ndef solve():\n    pass")
    )
    task = (
        ("problem intro " * 180)
        + "\n\nInput\nThe first line contains t.\nEach test case contains n and then n integers.\n\n"
        + "Output\nPrint one answer per test case.\n\n"
        + ("Note\n" + ("extra detail " * 120))
    )

    state = _make_state(task=task, shrinkage_level=1)
    state["model"] = "qwen3.5:2b"
    state["task_metadata"] = {"testtype": "stdin", "task_id": "1899_b"}
    tg.get_node_fn("generate")(state, mock_llm)
    prompt_without_verify_failure = captured[-1]

    captured.clear()
    retry_state = _make_state(
        task=task,
        shrinkage_level=1,
        step_outputs=[
            StepOutput(
                node="generate",
                model="qwen3.5:2b",
                output="import sys\ndef solve():\n    n",
                confidence=1.0,
                passed=True,
            ),
            StepOutput(
                node="verify",
                model="qwen3.5:2b",
                output="FAIL: NameError: name 'n' is not defined",
                confidence=0.0,
                passed=False,
            ),
        ],
    )
    retry_state["model"] = "qwen3.5:2b"
    retry_state["task_metadata"] = {"testtype": "stdin", "task_id": "1899_b"}

    tg.get_node_fn("generate")(retry_state, mock_llm)
    prompt_after_verify_failure = captured[-1]

    assert "The first line contains t." in prompt_after_verify_failure
    assert "Print one answer per test case." in prompt_after_verify_failure
    assert len(prompt_after_verify_failure) < len(prompt_without_verify_failure)


def test_generate_node_large_model_stdin_keeps_freeform_prompt():
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    captured = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="print(1)")
    )
    state = _make_state(task="problem text")
    state["model"] = "qwen3.5:9b"
    state["task_metadata"] = {"testtype": "stdin", "task_id": "abc301_e"}

    tg.get_node_fn("generate")(state, mock_llm)

    prompt = captured[0]
    assert prompt.startswith(
        "/no_think\n\nSolve the following competitive programming problem. "
        "Write a complete Python program that reads from stdin and writes to stdout. "
        "Output only the Python code, no markdown fences, no explanation.\n\n"
    )
    assert prompt.endswith("problem text")


def test_generate_node_small_model_stdin_rejects_prose_only_output():
    tg = CompletionTaskGraph()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="To solve this problem, we parse the input and compare all possible splits."
    )
    state = _make_state(task="problem text")
    state["model"] = "qwen3.5:2b"
    state["task_metadata"] = {"testtype": "stdin", "task_id": "1899_b"}

    result = tg.get_node_fn("generate")(state, mock_llm)

    assert result == ""


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
