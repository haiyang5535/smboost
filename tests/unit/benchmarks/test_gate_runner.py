from __future__ import annotations

from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from benchmarks.gates.runner import run_gate, GateConfig
from benchmarks.humaneval_plus.runner import HumanEvalPlusResult


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


def test_run_humaneval_harness_real_construction_two_tasks():
    """Smoke test that really builds HarnessAgent + CompletionTaskGraph for
    the humaneval_plus/C4 path, using a canned llm_factory.

    This is the mirror-image of the mocked dispatch test above: here
    `build_condition` is NOT patched, so any construction-time regression
    (invariant-suite wiring, task-graph wiring, llm-factory contract) will
    surface.  C4 is used because it runs the AST-only verify, which is
    deterministic and needs no subprocess sandbox.

    The only boundary that stays mocked is :func:`_evalplus_evaluate` — we
    don't want to invoke the real evalplus pipeline in a unit test.
    """
    canned_completion = "    return a + b\n"

    def _fake_llm_factory(_model):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content=canned_completion)
        return llm

    tasks = [
        {
            "task_id": "HEval/add",
            "prompt": (
                "def add(a: int, b: int) -> int:\n"
                '    """Return a + b."""\n'
            ),
            "entry_point": "add",
        },
        {
            "task_id": "HEval/add2",
            "prompt": (
                "def add2(a: int, b: int) -> int:\n"
                '    """Return a + b."""\n'
            ),
            "entry_point": "add2",
        },
    ]

    # Stub the subset-tolerant HE evaluator (the gate runner's current path —
    # see benchmarks/humaneval_plus/simple_eval.py).  Real contract shape:
    # HumanEvalPlusResult(pass_at_1_base, pass_at_1_plus, rows=[{task_id, completion, passed_heval, passed_heval_plus}, ...]).
    fake_result = HumanEvalPlusResult(
        pass_at_1_base=0.5,
        pass_at_1_plus=0.5,
        rows=[
            {"task_id": "HEval/add", "completion": canned_completion,
             "passed_heval": 1, "passed_heval_plus": 1},
            {"task_id": "HEval/add2", "completion": canned_completion,
             "passed_heval": 0, "passed_heval_plus": 0},
        ],
    )

    cfg = GateConfig(
        name="G_real_smoke",
        bench="humaneval_plus",
        n=2,
        configs=[("qwen3.5:2b", "C4")],
    )

    with patch(
        "benchmarks.gates.runner._load_tasks_for_bench", return_value=tasks
    ), patch(
        "benchmarks.conditions.get_benchmark_llm_factory",
        return_value=_fake_llm_factory,
    ), patch(
        "benchmarks.humaneval_plus.simple_eval.evaluate_base_subset",
        return_value=fake_result,
    ):
        rows = run_gate(cfg)

    assert len(rows) == 2
    # Both tasks ran through real HarnessAgent + CompletionTaskGraph + evaluate_dual.
    by_id = {r["task_id"]: r for r in rows}
    assert by_id["HEval/add"]["passed"] == 1
    assert by_id["HEval/add2"]["passed"] == 0
    for r in rows:
        assert r["mode"] == "C4"
        assert r["bench"] == "humaneval_plus"
        assert r["model"] == "qwen3.5:2b"


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
