from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from benchmarks.gsm8k.runner import run_baseline


_SAMPLE_TASK = {
    "task_id": "gsm8k/0",
    "question": "If Alice has 3 apples and buys 4 more, how many apples total?",
    "expected_answer": "7",
}


def test_run_baseline_scores_correct_answer():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(
        content="Alice starts with 3, adds 4. 3+4=7. #### 7"
    )

    with patch("benchmarks.gsm8k.runner._make_raw_llm", return_value=fake_llm):
        rows = run_baseline([_SAMPLE_TASK], model="qwen3.5:2b")

    assert len(rows) == 1
    r = rows[0]
    assert r["task_id"] == "gsm8k/0"
    assert r["mode"] == "raw"
    assert r["bench"] == "gsm8k"
    assert r["passed"] == 1
    assert r["failure_bucket"] == "PASS"
    assert r["retries"] == 0
    assert r["model"] == "qwen3.5:2b"
    assert r["expected_answer"] == "7"


def test_run_baseline_marks_wrong_answer_as_failed():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="#### 42")

    with patch("benchmarks.gsm8k.runner._make_raw_llm", return_value=fake_llm):
        rows = run_baseline([_SAMPLE_TASK], model="qwen3.5:2b")

    r = rows[0]
    assert r["passed"] == 0
    assert r["failure_bucket"] == "wrong_answer"


def test_run_baseline_handles_empty_completion():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="")

    with patch("benchmarks.gsm8k.runner._make_raw_llm", return_value=fake_llm):
        rows = run_baseline([_SAMPLE_TASK], model="qwen3.5:2b")

    r = rows[0]
    assert r["passed"] == 0
    assert r["failure_bucket"] == "no_numeric_answer"


def test_run_baseline_passes_max_tokens_to_llm_factory():
    """Caller-supplied max_tokens must reach _make_raw_llm so llama.cpp
    actually caps completion length (otherwise small-model raw runs can
    hit n_ctx and each call takes minutes)."""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="#### 7")

    factory = MagicMock(return_value=fake_llm)
    with patch("benchmarks.gsm8k.runner._make_raw_llm", side_effect=factory):
        run_baseline([_SAMPLE_TASK], model="qwen3.5:2b", max_tokens=256, temperature=0.0)

    factory.assert_called_once()
    kwargs = factory.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 256


def test_run_baseline_returns_row_per_task():
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        AIMessage(content="#### 7"),
        AIMessage(content="#### 100"),
    ]
    tasks = [
        {"task_id": "gsm8k/0", "question": "Q1", "expected_answer": "7"},
        {"task_id": "gsm8k/1", "question": "Q2", "expected_answer": "100"},
    ]

    with patch("benchmarks.gsm8k.runner._make_raw_llm", return_value=fake_llm):
        rows = run_baseline(tasks, model="qwen3.5:2b")

    assert len(rows) == 2
    assert rows[0]["task_id"] == "gsm8k/0"
    assert rows[1]["task_id"] == "gsm8k/1"
    assert all(r["passed"] == 1 for r in rows)


def test_run_baseline_prompt_includes_question():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="#### 7")

    with patch("benchmarks.gsm8k.runner._make_raw_llm", return_value=fake_llm):
        run_baseline([_SAMPLE_TASK], model="qwen3.5:2b")

    # The HumanMessage passed to invoke should include the task's question text.
    call = fake_llm.invoke.call_args
    messages = call.args[0]
    assert len(messages) == 1
    prompt_text = messages[0].content
    assert _SAMPLE_TASK["question"] in prompt_text
    assert "step by step" in prompt_text.lower()
