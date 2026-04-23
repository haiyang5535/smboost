"""Smoke tests for the subprocess-based HumanEval fallback evaluator.

These tests actually spawn python3 subprocesses (one per completion) so they
are slower than the mock-based tests elsewhere, but cheap enough (~1-2s
total) to keep in the unit suite.
"""
from __future__ import annotations

import pytest

from benchmarks.humaneval_plus.simple_eval import evaluate_base_subset


_CORRECT_HEVAL_0 = (
    "    for idx, elem in enumerate(numbers):\n"
    "        for idx2, elem2 in enumerate(numbers):\n"
    "            if idx != idx2:\n"
    "                distance = abs(elem - elem2)\n"
    "                if distance < threshold:\n"
    "                    return True\n"
    "    return False\n"
)


def test_correct_heval_0_completion_passes():
    out = evaluate_base_subset(
        [{"task_id": "HumanEval/0", "completion": _CORRECT_HEVAL_0}]
    )
    assert out.pass_at_1_base == 1.0
    assert out.rows[0]["passed_heval"] == 1


def test_trivially_wrong_completion_fails():
    out = evaluate_base_subset(
        [{"task_id": "HumanEval/0", "completion": "    return True\n"}]
    )
    assert out.pass_at_1_base == 0.0
    assert out.rows[0]["passed_heval"] == 0


def test_unknown_task_id_scored_as_failed():
    out = evaluate_base_subset(
        [{"task_id": "HumanEval/9999", "completion": "pass"}]
    )
    assert out.rows[0]["passed_heval"] == 0
    assert out.pass_at_1_base == 0.0
