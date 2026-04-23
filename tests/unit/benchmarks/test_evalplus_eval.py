"""Tests for `benchmarks.humaneval_plus.evalplus_eval`.

The real-call tests spawn evalplus workers and should each finish in
well under 10s on an M-series mac per the research doc (~3s end-to-end).
If one starts running long, tag it `@pytest.mark.slow` and document.
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from benchmarks.humaneval_plus.evalplus_eval import (
    HumanEvalPlusResult,
    evaluate_subset,
)


# The canonical HumanEval/0 solution (body only — prompt is prepended inside
# evalplus via problems[task_id]["prompt"] + sample["completion"]).
HEVAL_0_CORRECT_COMPLETION = (
    "    for i, a in enumerate(numbers):\n"
    "        for j, b in enumerate(numbers):\n"
    "            if i != j and abs(a - b) < threshold:\n"
    "                return True\n"
    "    return False\n"
)


@pytest.mark.slow
def test_pad_to_164_all_empty_returns_164_failures() -> None:
    """Submitting an empty subset should still run evalplus cleanly — the
    runner pads 164 empty completions, each fails, and we return 0 rows
    (because the caller didn't ask for any).
    """
    out = evaluate_subset([])
    assert isinstance(out, HumanEvalPlusResult)
    assert out.pass_at_1_base == 0.0
    assert out.pass_at_1_plus == 0.0
    # Caller submitted zero tasks, so they get zero rows back; padding is
    # an internal implementation detail.
    assert out.rows == []


@pytest.mark.slow
def test_pad_with_one_correct_heval_0_returns_one_base_and_plus_pass() -> None:
    """The canonical HumanEval/0 completion passes both base and plus
    tests; padding the rest as empty should not affect that.
    """
    out = evaluate_subset(
        [{"task_id": "HumanEval/0", "completion": HEVAL_0_CORRECT_COMPLETION}]
    )
    assert isinstance(out, HumanEvalPlusResult)
    assert len(out.rows) == 1
    assert out.rows[0]["task_id"] == "HumanEval/0"
    assert out.rows[0]["passed_heval"] == 1, out
    assert out.rows[0]["passed_heval_plus"] == 1, out
    # With one real submission, pass@1 is 1/1 on both suites.
    assert out.pass_at_1_base == 1.0
    assert out.pass_at_1_plus == 1.0


@pytest.mark.slow
def test_pad_correctly_excludes_unsubmitted_tasks_from_return_rows() -> None:
    """Padding must be invisible to the caller: submit 2 tasks, get 2 rows
    (not 164).
    """
    out = evaluate_subset(
        [
            {"task_id": "HumanEval/0", "completion": HEVAL_0_CORRECT_COMPLETION},
            {"task_id": "HumanEval/1", "completion": ""},
        ]
    )
    assert len(out.rows) == 2
    task_ids_back = [r["task_id"] for r in out.rows]
    assert task_ids_back == ["HumanEval/0", "HumanEval/1"]
    # HumanEval/0 passes; HumanEval/1 with empty completion fails.
    assert out.rows[0]["passed_heval"] == 1
    assert out.rows[1]["passed_heval"] == 0
    assert out.rows[1]["passed_heval_plus"] == 0


def test_evaluate_subset_mocks_work_for_shape_regression(tmp_path) -> None:
    """Shape-regression guard: the parser must read
    `[{"base_status": "pass"|"fail", "plus_status": ...}, ...]` — one dict
    per attempt per task. Mocks both the evalplus call indirection and the
    problem-id loader so we don't need a live evalplus dataset.
    """
    fake_eval_map = {
        "HumanEval/0": [{"base_status": "pass", "plus_status": "pass"}],
        "HumanEval/1": [{"base_status": "pass", "plus_status": "fail"}],
        "HumanEval/2": [{"base_status": "fail", "plus_status": "fail"}],
    }

    with (
        patch(
            "benchmarks.humaneval_plus.evalplus_eval._evalplus_evaluate",
            return_value=fake_eval_map,
        ),
        patch(
            "benchmarks.humaneval_plus.evalplus_eval._load_problem_task_ids",
            return_value=["HumanEval/0", "HumanEval/1", "HumanEval/2"],
        ),
    ):
        out = evaluate_subset(
            [
                {"task_id": "HumanEval/0", "completion": "..."},
                {"task_id": "HumanEval/1", "completion": "..."},
                {"task_id": "HumanEval/2", "completion": "..."},
            ]
        )

    assert len(out.rows) == 3
    assert out.rows[0]["passed_heval"] == 1
    assert out.rows[0]["passed_heval_plus"] == 1
    assert out.rows[1]["passed_heval"] == 1
    assert out.rows[1]["passed_heval_plus"] == 0
    assert out.rows[2]["passed_heval"] == 0
    assert out.rows[2]["passed_heval_plus"] == 0
    assert out.pass_at_1_base == pytest.approx(2 / 3)
    assert out.pass_at_1_plus == pytest.approx(1 / 3)


def test_evaluate_subset_parser_handles_missing_entry_as_fail() -> None:
    """If evalplus's eval_map is missing a task_id we submitted (e.g.
    upstream warn-and-skip), we must score it as fail rather than
    crashing with KeyError.
    """
    fake_eval_map = {
        "HumanEval/0": [{"base_status": "pass", "plus_status": "pass"}],
        # HumanEval/1 absent
    }

    with (
        patch(
            "benchmarks.humaneval_plus.evalplus_eval._evalplus_evaluate",
            return_value=fake_eval_map,
        ),
        patch(
            "benchmarks.humaneval_plus.evalplus_eval._load_problem_task_ids",
            return_value=["HumanEval/0", "HumanEval/1"],
        ),
    ):
        out = evaluate_subset(
            [
                {"task_id": "HumanEval/0", "completion": "..."},
                {"task_id": "HumanEval/1", "completion": "..."},
            ]
        )

    assert out.rows[1]["passed_heval"] == 0
    assert out.rows[1]["passed_heval_plus"] == 0


def test_evaluate_subset_ssl_cert_file_gets_set() -> None:
    """Importing the module must populate SSL_CERT_FILE so evalplus's
    HTTPS-download of HumanEvalPlus.jsonl.gz succeeds on Python 3.13 +
    macOS. We just import the module and check the env var is set.
    """
    import benchmarks.humaneval_plus.evalplus_eval  # noqa: F401

    assert os.environ.get("SSL_CERT_FILE"), (
        "SSL_CERT_FILE should be populated at module import time"
    )
    # It should point at a readable PEM (we don't require an exact match to
    # certifi — the env var may have been pre-set by the operator).
    assert os.path.isfile(os.environ["SSL_CERT_FILE"])
