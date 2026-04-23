from __future__ import annotations

from unittest.mock import patch

from benchmarks.humaneval_plus.runner import (
    evaluate_dual,
    HumanEvalPlusResult,
)


def test_evaluate_dual_returns_both_scores():
    """Result shape is {task_id: [ {base_status, plus_status, ...}, ... ]}
    — one dict per attempt. See research doc
    docs/superpowers/research/2026-04-23-evalplus-he-plus-fix.md.
    """
    results = [
        {"task_id": "HumanEval/0", "completion": "    return 1"},
        {"task_id": "HumanEval/1", "completion": "    return 2"},
    ]

    fake_return = {
        "HumanEval/0": [{"base_status": "pass", "plus_status": "fail"}],
        "HumanEval/1": [{"base_status": "pass", "plus_status": "pass"}],
    }

    with patch(
        "benchmarks.humaneval_plus.runner._evalplus_evaluate",
        return_value=fake_return,
    ):
        out = evaluate_dual(results)

    assert isinstance(out, HumanEvalPlusResult)
    assert out.pass_at_1_base == 1.0        # 2/2 pass on base
    assert out.pass_at_1_plus == 0.5        # 1/2 pass on plus
    assert len(out.rows) == 2
    assert out.rows[0]["passed_heval"] == 1
    assert out.rows[0]["passed_heval_plus"] == 0
    assert out.rows[1]["passed_heval"] == 1
    assert out.rows[1]["passed_heval_plus"] == 1


def test_evaluate_dual_handles_empty_completions():
    results = [{"task_id": "HumanEval/0", "completion": ""}]
    fake_return = {
        "HumanEval/0": [{"base_status": "fail", "plus_status": "fail"}]
    }

    with patch(
        "benchmarks.humaneval_plus.runner._evalplus_evaluate",
        return_value=fake_return,
    ):
        out = evaluate_dual(results)

    assert out.pass_at_1_base == 0.0
    assert out.pass_at_1_plus == 0.0


def test_evaluate_dual_missing_task_in_map_treated_as_fail():
    """Guard: if evalplus's output is missing a task_id we submitted,
    score it as fail rather than crashing with KeyError.
    """
    results = [
        {"task_id": "HumanEval/0", "completion": "    return 1"},
        {"task_id": "HumanEval/99", "completion": "    return 1"},
    ]
    fake_return = {
        "HumanEval/0": [{"base_status": "pass", "plus_status": "pass"}],
        # HumanEval/99 absent
    }

    with patch(
        "benchmarks.humaneval_plus.runner._evalplus_evaluate",
        return_value=fake_return,
    ):
        out = evaluate_dual(results)

    assert out.pass_at_1_base == 0.5
    assert out.pass_at_1_plus == 0.5
    assert out.rows[1]["passed_heval"] == 0
    assert out.rows[1]["passed_heval_plus"] == 0


def test_evaluate_dual_multiple_attempts_per_task():
    """If the caller submits the same task_id twice, the evalmap has two
    entries and we pair them in order (attempt 0, attempt 1).
    """
    results = [
        {"task_id": "HumanEval/0", "completion": "    return 1"},
        {"task_id": "HumanEval/0", "completion": "    return 2"},
    ]
    fake_return = {
        "HumanEval/0": [
            {"base_status": "pass", "plus_status": "pass"},
            {"base_status": "fail", "plus_status": "fail"},
        ],
    }

    with patch(
        "benchmarks.humaneval_plus.runner._evalplus_evaluate",
        return_value=fake_return,
    ):
        out = evaluate_dual(results)

    assert out.pass_at_1_base == 0.5
    assert out.pass_at_1_plus == 0.5
    assert out.rows[0]["passed_heval"] == 1
    assert out.rows[1]["passed_heval"] == 0
