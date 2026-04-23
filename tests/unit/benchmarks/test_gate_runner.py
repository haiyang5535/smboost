from __future__ import annotations

from unittest.mock import patch, MagicMock

from benchmarks.gates.runner import run_gate, GateConfig


def _fake_humaneval_tasks(n):
    return [
        {"task_id": f"HEval/{i}", "prompt": f"def f{i}():\n    ", "entry_point": f"f{i}"}
        for i in range(n)
    ]


def _fake_raw_rows(n, model, pass_rate, bench):
    n_pass = int(n * pass_rate)
    rows = []
    for i in range(n):
        rows.append({
            "task_id": f"HEval/{i}" if bench == "humaneval_plus" else f"simple_{i}",
            "model": model, "mode": "raw", "passed": 1 if i < n_pass else 0,
            "retries": 0, "latency_s": 1.0, "bench": bench,
        })
    return rows


def _fake_harness_rows(n, model, cond, pass_rate, bench):
    n_pass = int(n * pass_rate)
    rows = []
    for i in range(n):
        rows.append({
            "task_id": f"HEval/{i}" if bench == "humaneval_plus" else f"simple_{i}",
            "model": model, "mode": cond, "passed": 1 if i < n_pass else 0,
            "retries": 1, "latency_s": 2.0, "bench": bench,
        })
    return rows


def test_run_gate_dispatches_humaneval_raw_and_harness_correctly():
    cfg = GateConfig(
        name="G2_test",
        bench="humaneval_plus",
        n=20,
        configs=[
            ("qwen3.5:2b", "raw"),
            ("qwen3.5:2b", "C1"),
            ("qwen3.5:2b", "C4"),
        ],
    )
    tasks = _fake_humaneval_tasks(20)

    with patch("benchmarks.gates.runner._load_tasks_for_bench", return_value=tasks), \
         patch("benchmarks.gates.runner._run_humaneval_raw",
               side_effect=lambda tasks, model: _fake_raw_rows(20, model, 0.40, "humaneval_plus")), \
         patch("benchmarks.gates.runner._run_humaneval_harness",
               side_effect=lambda tasks, cond, model:
                   _fake_harness_rows(20, model, cond, 0.70 if cond == "C1" else 0.50, "humaneval_plus")):
        rows = run_gate(cfg)

    assert len(rows) == 60  # 20 tasks x 3 configs
    modes = {r["mode"] for r in rows}
    assert modes == {"raw", "C1", "C4"}


def test_run_gate_dispatches_bfcl():
    cfg = GateConfig(
        name="G3_test",
        bench="bfcl_simple",
        n=30,
        configs=[("qwen3.5:2b", "raw"), ("qwen3.5:2b", "C1")],
    )
    tasks = [{"task_id": f"simple_{i}", "category": "simple",
              "question": "q", "functions": [], "ground_truth": []} for i in range(30)]

    with patch("benchmarks.gates.runner._load_tasks_for_bench", return_value=tasks), \
         patch("benchmarks.gates.runner._run_bfcl_raw_fn",
               side_effect=lambda tasks, model: _fake_raw_rows(30, model, 0.30, "bfcl_simple")), \
         patch("benchmarks.gates.runner._run_bfcl_harness_fn",
               side_effect=lambda tasks, cond, model: _fake_harness_rows(30, model, cond, 0.60, "bfcl_simple")):
        rows = run_gate(cfg)

    assert len(rows) == 60
    modes = {r["mode"] for r in rows}
    assert modes == {"raw", "C1"}
