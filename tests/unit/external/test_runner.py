"""Unit tests for benchmarks/external/runner.py and cli.py.

We mock the client layer so nothing hits a real API. Assertions focus on:
- cost estimation math
- CSV row shape (matches the gate runner schema plus cost_usd)
- CLI argument surface (help prints, refuses big-n without --confirm-cost)
- gsm8k graceful degradation when Agent 3's module is missing
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.external import pricing
from benchmarks.external.cli import CSV_FIELDS, build_parser, main
from benchmarks.external.runner import run_external_baseline


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def test_estimate_cost_known_rate():
    # 1M input tokens of GPT-4o at $2.50 + 1M output tokens at $10 -> $12.50
    assert pricing.estimate_cost("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.50)


def test_estimate_cost_claude_sonnet_4_6():
    # Sanity-check the pitch-relevant model: 10k in + 2k out on Sonnet 4.6.
    # 10_000 * $3 / 1M + 2_000 * $15 / 1M = $0.03 + $0.03 = $0.06
    cost = pricing.estimate_cost("claude-sonnet-4-6", 10_000, 2_000)
    assert cost == pytest.approx(0.06)


def test_estimate_cost_llama_short_alias_and_long_alias_agree():
    cost_short = pricing.estimate_cost("llama-3-70b", 1000, 500)
    cost_long = pricing.estimate_cost(
        "meta-llama/llama-3-70b-instruct", 1000, 500
    )
    assert cost_short == pytest.approx(cost_long)


def test_estimate_cost_unknown_model_raises():
    with pytest.raises(KeyError) as excinfo:
        pricing.estimate_cost("mystery-v9000", 100, 100)
    assert "No pricing entry" in str(excinfo.value)


def test_estimate_cost_is_case_insensitive():
    a = pricing.estimate_cost("GPT-4O", 100, 50)
    b = pricing.estimate_cost("gpt-4o", 100, 50)
    assert a == b


# ---------------------------------------------------------------------------
# run_external_baseline output shape
# ---------------------------------------------------------------------------


_HUMANEVAL_TASK = {
    "task_id": "HumanEval/0",
    "prompt": "def add(a, b):\n    ",
    "entry_point": "add",
    "test": "def check(c):\n    assert c(1, 2) == 3\n",
}

_BFCL_TASK = {
    "task_id": "simple_0",
    "category": "simple",
    "question": "Get weather in SF",
    "functions": [
        {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ],
    "ground_truth": [{"name": "get_weather", "arguments": {"city": "SF"}}],
}


def test_humaneval_plus_rows_match_gate_schema():
    fake_client = MagicMock()
    fake_client.complete.return_value = {
        "content": "return a + b",
        "latency_s": 0.5,
        "cost_usd": 0.001,
        "input_tokens": 20,
        "output_tokens": 10,
    }
    # Fake the evalplus resolver so we don't need evalplus / human_eval installed.
    from benchmarks.humaneval_plus.runner import HumanEvalPlusResult
    fake_eval = HumanEvalPlusResult(
        pass_at_1_base=1.0,
        pass_at_1_plus=1.0,
        rows=[
            {
                "task_id": "HumanEval/0",
                "completion": "return a + b",
                "passed_heval": 1,
                "passed_heval_plus": 1,
            }
        ],
    )
    with patch(
        "benchmarks.external.runner.make_client", return_value=fake_client
    ), patch(
        "benchmarks.external.runner._evaluate_humaneval_dual",
        return_value=fake_eval,
    ):
        rows = run_external_baseline(
            bench="humaneval_plus",
            tasks=[_HUMANEVAL_TASK],
            model="claude-sonnet-4-6",
            max_tokens=128,
        )

    assert len(rows) == 1
    row = rows[0]
    # Every CSV_FIELDS key must be present so the CLI's DictWriter doesn't
    # silently blank a column.
    for k in CSV_FIELDS:
        assert k in row, f"missing field {k!r}"
    assert row["bench"] == "humaneval_plus"
    assert row["mode"] == "raw"
    assert row["passed"] == 1
    assert row["passed_heval_plus"] == 1
    assert row["cost_usd"] == 0.001
    assert row["model"] == "claude-sonnet-4-6"


def test_bfcl_simple_row_shape_and_pass():
    fake_client = MagicMock()
    fake_client.complete.return_value = {
        "content": '{"name": "get_weather", "arguments": {"city": "SF"}}',
        "latency_s": 0.4,
        "cost_usd": 0.0005,
        "input_tokens": 50,
        "output_tokens": 20,
    }
    with patch("benchmarks.external.runner.make_client", return_value=fake_client):
        rows = run_external_baseline(
            bench="bfcl_simple",
            tasks=[_BFCL_TASK],
            model="gpt-4o",
            max_tokens=64,
        )
    assert len(rows) == 1
    r = rows[0]
    assert r["bench"] == "bfcl_simple"
    assert r["passed"] == 1
    assert r["cost_usd"] == 0.0005
    # passed_heval* cols are always present but zero for non-HE benches
    assert r["passed_heval"] == 0
    assert r["passed_heval_plus"] == 0


def test_bfcl_simple_row_fails_on_wrong_arg():
    fake_client = MagicMock()
    fake_client.complete.return_value = {
        "content": '{"name": "get_weather", "arguments": {"city": "NYC"}}',
        "latency_s": 0.4,
        "cost_usd": 0.0005,
        "input_tokens": 50,
        "output_tokens": 20,
    }
    with patch("benchmarks.external.runner.make_client", return_value=fake_client):
        rows = run_external_baseline(
            bench="bfcl_simple",
            tasks=[_BFCL_TASK],
            model="gpt-4o",
            max_tokens=64,
        )
    assert rows[0]["passed"] == 0


def test_runner_rejects_unknown_bench():
    with pytest.raises(ValueError) as excinfo:
        run_external_baseline(
            bench="mystery_bench",
            tasks=[],
            model="claude-sonnet-4-6",
            max_tokens=32,
        )
    assert "unknown bench" in str(excinfo.value)


def test_runner_gsm8k_raises_import_error_when_module_missing():
    """If Agent 3's gsm8k module isn't present, we must surface a clear error
    rather than silently passing or obscurely crashing."""
    # Ensure no stale module hanging around from a prior test
    sys.modules.pop("benchmarks.gsm8k", None)
    sys.modules.pop("benchmarks.gsm8k.runner", None)

    # Force ImportError by ensuring the submodule can't be found.
    with patch.dict(sys.modules, {"benchmarks.gsm8k": None}):
        with pytest.raises(ImportError) as excinfo:
            run_external_baseline(
                bench="gsm8k",
                tasks=[{"task_id": "g1", "question": "1+1"}],
                model="gpt-4o",
                max_tokens=32,
            )
    assert "gsm8k" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_parser_help_includes_all_benches(capsys):
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "humaneval_plus" in out
    assert "bfcl_simple" in out
    assert "bfcl_multi" in out
    assert "gsm8k" in out


def test_cli_refuses_big_n_without_confirm_cost(capsys):
    rc = main(
        [
            "--model", "claude-sonnet-4-6",
            "--bench", "humaneval_plus",
            "--n", "100",
            "--out-csv", "/tmp/should_not_be_written.csv",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "--confirm-cost" in err


def test_cli_allows_big_n_with_confirm_cost(tmp_path, monkeypatch):
    """Smoke: with --confirm-cost, the CLI proceeds past the cost gate
    and invokes the runner. We mock _load_tasks + run_external_baseline so
    no network traffic happens."""
    out_csv = tmp_path / "rows.csv"

    fake_rows = [
        {
            "bench": "humaneval_plus",
            "task_id": "HumanEval/0",
            "model": "claude-sonnet-4-6",
            "mode": "raw",
            "passed": 1,
            "passed_heval": 1,
            "passed_heval_plus": 1,
            "retries": 0,
            "latency_s": 0.5,
            "cost_usd": 0.001,
        }
    ]
    with patch(
        "benchmarks.external.cli._load_tasks",
        return_value=[{"task_id": "HumanEval/0", "prompt": "x"}],
    ), patch(
        "benchmarks.external.cli.run_external_baseline", return_value=fake_rows
    ):
        rc = main(
            [
                "--model", "claude-sonnet-4-6",
                "--bench", "humaneval_plus",
                "--n", "31",
                "--out-csv", str(out_csv),
                "--confirm-cost",
            ]
        )
    assert rc == 0
    content = out_csv.read_text()
    # Header + one row
    assert "bench,task_id,model,mode" in content.splitlines()[0]
    assert "HumanEval/0" in content
    assert "cost_usd" in content.splitlines()[0]


def test_cli_writes_rows_in_canonical_field_order(tmp_path):
    """CSV header order must match CSV_FIELDS — downstream tooling
    (pandas concat of gate runs and external runs) depends on it."""
    out_csv = tmp_path / "rows.csv"
    fake_rows = [
        {
            "bench": "bfcl_simple",
            "task_id": "simple_0",
            "model": "gpt-4o",
            "mode": "raw",
            "passed": 1,
            "passed_heval": 0,
            "passed_heval_plus": 0,
            "retries": 0,
            "latency_s": 0.4,
            "cost_usd": 0.0005,
        }
    ]
    with patch(
        "benchmarks.external.cli._load_tasks",
        return_value=[{"task_id": "simple_0"}],
    ), patch(
        "benchmarks.external.cli.run_external_baseline", return_value=fake_rows
    ):
        rc = main(
            [
                "--model", "gpt-4o",
                "--bench", "bfcl_simple",
                "--n", "1",
                "--out-csv", str(out_csv),
            ]
        )
    assert rc == 0
    header = out_csv.read_text().splitlines()[0].split(",")
    assert header == CSV_FIELDS
