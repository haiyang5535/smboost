from __future__ import annotations

from unittest.mock import patch

from benchmarks.humaneval_plus.runner import (
    evaluate_dual,
    HumanEvalPlusResult,
)


def test_evaluate_dual_returns_both_scores():
    results = [
        {"task_id": "HumanEval/0", "completion": "    return 1"},
        {"task_id": "HumanEval/1", "completion": "    return 2"},
    ]

    fake_return = {
        "HumanEval/0": {"base": {"passed": True}, "plus": {"passed": False}},
        "HumanEval/1": {"base": {"passed": True}, "plus": {"passed": True}},
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
    fake_return = {"HumanEval/0": {"base": {"passed": False}, "plus": {"passed": False}}}

    with patch(
        "benchmarks.humaneval_plus.runner._evalplus_evaluate",
        return_value=fake_return,
    ):
        out = evaluate_dual(results)

    assert out.pass_at_1_base == 0.0
    assert out.pass_at_1_plus == 0.0
