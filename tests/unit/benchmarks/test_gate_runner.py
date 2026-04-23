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


def test_default_llm_factory_uses_temperature_zero(monkeypatch):
    """Determinism: the default benchmark factory must pin temperature=0 on the OpenAI-
    compatible backend so repeated C1/C4/etc. runs don't flip 1-3 tasks from sampling
    jitter on n=20 gates."""
    # Clear the lru_cache so a fresh factory gets constructed with our patched backend.
    from smboost.llm import runtime as runtime_mod

    runtime_mod._cached_openai_factory.cache_clear()
    runtime_mod._cached_local_factory.cache_clear()

    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "server")
    monkeypatch.setenv("SMBOOST_OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("SMBOOST_OPENAI_API_KEY", "sk-no-key")
    monkeypatch.setenv("SMBOOST_OPENAI_MAX_TOKENS", "512")

    factory = runtime_mod.get_default_llm_factory()
    llm = factory("qwen3.5:2b")
    # _CompatibleChatOpenAI subclasses ChatOpenAI which exposes `temperature`
    # as a pydantic field.
    assert llm.temperature == 0.0


def test_default_llm_factory_local_backend_pins_temperature_zero(monkeypatch):
    """Mirror of the OpenAI-backend test for the local llama.cpp backend — the
    _LocalLlamaChat the factory vends should carry temperature=0 so
    create_chat_completion runs deterministically."""
    import sys
    from types import SimpleNamespace

    from smboost.llm import runtime as runtime_mod

    runtime_mod._cached_openai_factory.cache_clear()
    runtime_mod._cached_local_factory.cache_clear()

    class FakeLlama:
        def __init__(self, model_path: str, **_kwargs):
            pass

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "local")
    monkeypatch.setenv("SMBOOST_MODEL_DIR", "models")
    monkeypatch.setenv("SMBOOST_LOCAL_MAX_TOKENS", "64")

    factory = runtime_mod.get_default_llm_factory()
    llm = factory("qwen3.5:2b")
    assert llm.temperature == 0.0


def test_run_baseline_uses_temperature_zero():
    """run_baseline must construct ChatOpenAI with temperature=0 so the raw baseline
    doesn't drift across re-runs on n=20 gates."""
    import benchmarks.run_humaneval as rh

    captured = {}

    class FakeChatOpenAI:
        def __init__(self, *, model, base_url, api_key, temperature, **_kwargs):
            captured["model"] = model
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["temperature"] = temperature

        def invoke(self, _messages):
            return AIMessage(content="    return 1\n")

    with patch.object(rh, "ChatOpenAI", FakeChatOpenAI):
        rh.run_baseline(
            [{"task_id": "HumanEval/0", "prompt": "def f():\n    "}],
            model="qwen3.5:2b",
        )

    assert captured["temperature"] == 0.0
    assert captured["model"] == "qwen3.5:2b"


def test_run_humaneval_harness_passes_task_metadata_to_agent():
    """Regression: _run_humaneval_harness previously called agent.run(prompt) with no
    task_metadata, causing _verify_grounded to degrade to AST-only verify and the whole
    harness (grounded_verify/session_memory/shrinkage/scorer) to go inert on gate data.
    See docs/superpowers/research/2026-04-23-c1-lift-debug.md.
    """
    from benchmarks.gates.runner import _run_humaneval_harness

    fake_agent = MagicMock()
    fake_run = MagicMock()
    fake_run.trace = []
    fake_run.stats = MagicMock(total_latency_s=1.0, retry_count=0)
    fake_agent.run.return_value = fake_run

    tasks = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def foo(x):\n    ",
            "entry_point": "foo",
            "test": "def check(candidate):\n    assert candidate(1) == 1\n",
        }
    ]

    fake_eval = MagicMock()
    fake_eval.rows = [
        {
            "task_id": "HumanEval/0",
            "completion": "",
            "passed_heval": 1,
            "passed_heval_plus": 1,
        }
    ]

    # build_condition is imported inside the function, so patch at its origin.
    with patch(
        "benchmarks.conditions.build_condition", return_value=fake_agent
    ), patch(
        "benchmarks.humaneval_plus.simple_eval.evaluate_base_subset",
        return_value=fake_eval,
    ):
        _run_humaneval_harness(tasks, condition="C1", model="qwen3.5:2b")

    assert fake_agent.run.called, "agent.run was not invoked"
    call = fake_agent.run.call_args
    kwargs = call.kwargs
    # prompt passed positionally
    assert call.args[0] == "def foo(x):\n    "
    # task_metadata, task_id, condition kwargs plumbed through
    assert "task_metadata" in kwargs
    md = kwargs["task_metadata"]
    assert md["testtype"] == "humaneval"
    assert md["entry_point"] == "foo"
    assert md["test"] == "def check(candidate):\n    assert candidate(1) == 1\n"
    assert md["prompt"] == "def foo(x):\n    "
    assert md["task_id"] == "HumanEval/0"
    assert kwargs.get("task_id") == "HumanEval/0"
    assert kwargs.get("condition") == "C1"


def test_run_humaneval_harness_passes_task_id_and_condition():
    """agent.run must receive task_id + condition so trace records are tagged correctly."""
    from benchmarks.gates.runner import _run_humaneval_harness

    fake_agent = MagicMock()
    fake_run = MagicMock()
    fake_run.trace = []
    fake_run.stats = MagicMock(total_latency_s=0.5, retry_count=2)
    fake_agent.run.return_value = fake_run

    tasks = [
        {
            "task_id": "HumanEval/42",
            "prompt": "def bar():\n    ",
            "entry_point": "bar",
            "test": "def check(c):\n    pass\n",
        }
    ]

    fake_eval = MagicMock()
    fake_eval.rows = [
        {
            "task_id": "HumanEval/42",
            "completion": "",
            "passed_heval": 0,
            "passed_heval_plus": 0,
        }
    ]

    with patch(
        "benchmarks.conditions.build_condition", return_value=fake_agent
    ), patch(
        "benchmarks.humaneval_plus.simple_eval.evaluate_base_subset",
        return_value=fake_eval,
    ):
        rows = _run_humaneval_harness(tasks, condition="C3", model="qwen3.5:2b")

    kwargs = fake_agent.run.call_args.kwargs
    assert kwargs.get("task_id") == "HumanEval/42"
    assert kwargs.get("condition") == "C3"
    # Retries and latency from the fake run propagate into the row for scoring.
    assert rows[0]["retries"] == 2
    assert rows[0]["latency_s"] == 0.5


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
