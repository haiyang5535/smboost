"""Unit tests for SelfConsistencyTaskGraph (C5).

Covers:
  * Construction + TaskGraph protocol conformance.
  * Majority voting: 3/5 agree -> that answer wins; all-invalid -> first sample.
  * All three built-in verifiers with simple fixtures (no live LLM).
  * build_condition("C5") wires a SelfConsistencyTaskGraph.
"""
from __future__ import annotations

import itertools
from unittest.mock import MagicMock

import pytest

from smboost.harness.self_consistency_graph import (
    SelfConsistencyTaskGraph,
    execute_program_verifier,
    majority_vote,
    run_tests_verifier,
    tool_call_valid_verifier,
)
from smboost.harness.state import StepOutput


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_state(task: str = "what is 2+2?", metadata: dict | None = None) -> dict:
    return {
        "task": task,
        "task_metadata": metadata or {},
        "model": "qwen3.5:2b",
        "fallback_chain": ["qwen3.5:2b"],
        "step_outputs": [],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 0,
        "shrinkage_level": 0,
        "status": "running",
        "final_output": None,
    }


def _mock_llm_returning(contents: list[str]):
    """Return a MagicMock whose .invoke(...) yields contents in order, cycling."""
    cycle = itertools.cycle(contents)

    def _invoke(_messages, **_kwargs):
        msg = MagicMock()
        msg.content = next(cycle)
        return msg

    llm = MagicMock()
    llm.invoke.side_effect = _invoke
    return llm


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_construction_requires_verifier():
    with pytest.raises(ValueError, match="verifier"):
        SelfConsistencyTaskGraph(verifier=None)  # type: ignore[arg-type]


def test_construction_requires_positive_n_samples():
    with pytest.raises(ValueError, match="n_samples"):
        SelfConsistencyTaskGraph(verifier=lambda s, m: {"valid": True}, n_samples=0)


def test_node_names_match_completion_shape():
    tg = SelfConsistencyTaskGraph(verifier=lambda s, m: {"valid": True})
    assert tg.node_names == ["generate", "verify"]


def test_get_node_fn_returns_callable_for_each_node():
    tg = SelfConsistencyTaskGraph(verifier=lambda s, m: {"valid": True})
    for name in tg.node_names:
        assert callable(tg.get_node_fn(name))


def test_get_node_fn_raises_on_unknown_node():
    tg = SelfConsistencyTaskGraph(verifier=lambda s, m: {"valid": True})
    with pytest.raises(ValueError, match="dispatch"):
        tg.get_node_fn("dispatch")


# ---------------------------------------------------------------------------
# majority_vote primitive
# ---------------------------------------------------------------------------


def test_majority_vote_picks_most_common_valid_answer():
    samples = ["s1", "s2", "s3", "s4", "s5"]
    results = [
        {"valid": True, "extracted_answer": 42},
        {"valid": True, "extracted_answer": 41},
        {"valid": True, "extracted_answer": 42},
        {"valid": True, "extracted_answer": 42},
        {"valid": True, "extracted_answer": 41},
    ]
    winning, answer, valid, votes = majority_vote(samples, results)
    assert answer == 42
    assert valid == 5
    assert votes == 3
    # Winning sample is the earliest one tagged with 42.
    assert winning == "s1"


def test_majority_vote_ignores_invalid_samples():
    samples = ["bad1", "good1", "bad2", "good2", "good3"]
    results = [
        {"valid": False, "extracted_answer": 99},
        {"valid": True, "extracted_answer": 7},
        {"valid": False, "extracted_answer": 99},
        {"valid": True, "extracted_answer": 7},
        {"valid": True, "extracted_answer": 7},
    ]
    winning, answer, valid, votes = majority_vote(samples, results)
    assert answer == 7
    assert valid == 3
    assert votes == 3
    assert winning == "good1"


def test_majority_vote_all_invalid_returns_first_sample():
    samples = ["s1", "s2", "s3"]
    results = [
        {"valid": False, "extracted_answer": None},
        {"valid": False, "extracted_answer": None},
        {"valid": False, "extracted_answer": None},
    ]
    winning, answer, valid, votes = majority_vote(samples, results)
    assert winning == "s1"
    assert answer is None
    assert valid == 0
    assert votes == 0


def test_majority_vote_empty_samples_returns_empty():
    winning, answer, valid, votes = majority_vote([], [])
    assert winning == ""
    assert answer is None
    assert valid == 0
    assert votes == 0


def test_majority_vote_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        majority_vote(["a", "b"], [{"valid": True}])


def test_majority_vote_tiebreak_prefers_earliest_index():
    # Two buckets of size 2; first seen wins (stable).
    samples = ["a_first", "b_first", "a_second", "b_second"]
    results = [
        {"valid": True, "extracted_answer": "A"},
        {"valid": True, "extracted_answer": "B"},
        {"valid": True, "extracted_answer": "A"},
        {"valid": True, "extracted_answer": "B"},
    ]
    winning, answer, _, votes = majority_vote(samples, results)
    assert votes == 2
    assert answer == "A"
    assert winning == "a_first"


def test_majority_vote_handles_unhashable_answer():
    samples = ["s1", "s2", "s3"]
    results = [
        {"valid": True, "extracted_answer": {"k": 1}},
        {"valid": True, "extracted_answer": {"k": 2}},
        {"valid": True, "extracted_answer": {"k": 1}},
    ]
    winning, _, valid, votes = majority_vote(samples, results)
    assert valid == 3
    assert votes == 2
    assert winning == "s1"


# ---------------------------------------------------------------------------
# generate node integration: 3/5 majority
# ---------------------------------------------------------------------------


def test_generate_node_returns_majority_voted_sample_with_mocked_llm():
    llm = _mock_llm_returning([
        "answer is 42",
        "the answer: 41",
        "I get 42",
        "it must be 42",
        "maybe 41",
    ])

    # verifier: mark valid iff completion contains a number; extract that number.
    def _vf(completion: str, meta: dict):
        import re
        m = re.search(r"\d+", completion)
        if not m:
            return {"valid": False, "extracted_answer": None}
        return {"valid": True, "extracted_answer": int(m.group())}

    tg = SelfConsistencyTaskGraph(verifier=_vf, n_samples=5)
    state = _make_state()

    out = tg.get_node_fn("generate")(state, llm)
    assert "42" in out  # winning sample contains 42
    # Verify node should say PASS since valid_count > 0
    verify_out = tg.get_node_fn("verify")(state, llm)
    assert verify_out.startswith("PASS")
    assert "answer=42" in verify_out


def test_generate_all_invalid_falls_through_to_first_sample():
    llm = _mock_llm_returning(["junk1", "junk2", "junk3"])

    def _always_invalid(_c: str, _m: dict):
        return {"valid": False, "extracted_answer": None, "trace": "nope"}

    tg = SelfConsistencyTaskGraph(verifier=_always_invalid, n_samples=3)
    state = _make_state()

    out = tg.get_node_fn("generate")(state, llm)
    assert out == "junk1"
    verify_out = tg.get_node_fn("verify")(state, llm)
    assert verify_out.startswith("FAIL")


def test_generate_with_n_samples_one_skips_thread_pool():
    llm = _mock_llm_returning(["only one"])

    def _vf(c: str, _m: dict):
        return {"valid": True, "extracted_answer": c}

    tg = SelfConsistencyTaskGraph(verifier=_vf, n_samples=1)
    out = tg.get_node_fn("generate")(_make_state(), llm)
    assert out == "only one"


def test_generate_tolerates_verifier_exception():
    llm = _mock_llm_returning(["x"])

    def _boom(_c: str, _m: dict):
        raise RuntimeError("verifier broke")

    tg = SelfConsistencyTaskGraph(verifier=_boom, n_samples=2)
    out = tg.get_node_fn("generate")(_make_state(), llm)
    # Should not raise; should fall through to first sample.
    assert out == "x"
    verify_out = tg.get_node_fn("verify")(_make_state(), llm)
    assert verify_out.startswith("FAIL")


def test_verify_before_generate_returns_fail():
    tg = SelfConsistencyTaskGraph(verifier=lambda s, m: {"valid": True})
    out = tg.get_node_fn("verify")(_make_state(), MagicMock())
    assert out.startswith("FAIL")
    assert "before generate" in out


# ---------------------------------------------------------------------------
# built-in verifier: tool_call_valid_verifier
# ---------------------------------------------------------------------------


def test_tool_call_valid_verifier_accepts_valid_json_call():
    completion = '{"name": "get_weather", "arguments": {"city": "SF"}}'
    r = tool_call_valid_verifier(completion, {})
    assert r["valid"] is True
    assert r["extracted_answer"] == ("get_weather", (("city", '"SF"'),))


def test_tool_call_valid_verifier_strips_code_fence():
    completion = '```json\n{"name": "f", "arguments": {}}\n```'
    r = tool_call_valid_verifier(completion, {})
    assert r["valid"] is True


def test_tool_call_valid_verifier_rejects_non_json():
    r = tool_call_valid_verifier("definitely not json", {})
    assert r["valid"] is False
    assert "JSONDecodeError" in r["trace"]


def test_tool_call_valid_verifier_rejects_missing_name():
    r = tool_call_valid_verifier('{"arguments": {}}', {})
    assert r["valid"] is False


def test_tool_call_valid_verifier_rejects_non_dict_arguments():
    r = tool_call_valid_verifier('{"name": "f", "arguments": [1, 2]}', {})
    assert r["valid"] is False


def test_tool_call_valid_verifier_checks_against_declared_tools():
    meta = {"tools": [{"name": "allowed_fn", "parameters": {}}]}
    good = tool_call_valid_verifier(
        '{"name": "allowed_fn", "arguments": {}}', meta
    )
    bad = tool_call_valid_verifier(
        '{"name": "unknown_fn", "arguments": {}}', meta
    )
    assert good["valid"] is True
    assert bad["valid"] is False
    assert "not in declared tools" in bad["trace"]


def test_tool_call_valid_verifier_accepts_wrapped_schema_declarations():
    meta = {"tools": [
        {"type": "function", "function": {"name": "send_email", "parameters": {}}},
    ]}
    r = tool_call_valid_verifier(
        '{"name": "send_email", "arguments": {"to": "x@y"}}', meta
    )
    assert r["valid"] is True


# ---------------------------------------------------------------------------
# built-in verifier: execute_program_verifier
# ---------------------------------------------------------------------------


def test_execute_program_verifier_runs_program_and_extracts_number():
    completion = "```python\nprint(42)\n```"
    r = execute_program_verifier(completion, {})
    assert r["valid"] is True
    assert r["extracted_answer"] == 42


def test_execute_program_verifier_extracts_last_number_when_many():
    completion = "print('setup:', 1)\nprint('final:', 99)\n"
    r = execute_program_verifier(completion, {})
    assert r["valid"] is True
    assert r["extracted_answer"] == 99


def test_execute_program_verifier_handles_float_output():
    completion = "print(3.14)\n"
    r = execute_program_verifier(completion, {})
    assert r["valid"] is True
    assert r["extracted_answer"] == pytest.approx(3.14)


def test_execute_program_verifier_marks_crash_invalid():
    completion = "raise RuntimeError('boom')\n"
    r = execute_program_verifier(completion, {})
    assert r["valid"] is False
    assert r["extracted_answer"] is None


def test_execute_program_verifier_marks_empty_stdout_invalid():
    completion = "x = 1\n"  # runs fine but prints nothing
    r = execute_program_verifier(completion, {})
    assert r["valid"] is False


def test_execute_program_verifier_rejects_empty_completion():
    r = execute_program_verifier("", {})
    assert r["valid"] is False


# ---------------------------------------------------------------------------
# built-in verifier: run_tests_verifier
# ---------------------------------------------------------------------------


def test_run_tests_verifier_passes_with_functional_fixture():
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    meta = {
        "entry_point": "add",
        "test_cases": [
            {"input": [1, 2], "output": 3},
            {"input": [10, 20], "output": 30},
        ],
    }
    r = run_tests_verifier(completion, meta)
    assert r["valid"] is True
    # extracted_answer is the normalized code (so matching solutions group)
    assert "def add" in r["extracted_answer"]


def test_run_tests_verifier_fails_with_wrong_implementation():
    completion = "def add(a, b):\n    return a - b\n"
    meta = {
        "entry_point": "add",
        "test_cases": [{"input": [1, 2], "output": 3}],
    }
    r = run_tests_verifier(completion, meta)
    assert r["valid"] is False


def test_run_tests_verifier_passes_with_humaneval_shape():
    # HumanEval-shape meta: prompt+test+entry_point
    meta = {
        "entry_point": "identity",
        "prompt": "def identity(x):\n    ",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(1) == 1\n"
            "    assert candidate(42) == 42\n"
        ),
    }
    completion = "return x\n"
    r = run_tests_verifier(completion, meta)
    assert r["valid"] is True


def test_run_tests_verifier_rejects_empty_completion():
    r = run_tests_verifier("", {"entry_point": "f", "test_cases": [{"input": 1, "output": 1}]})
    assert r["valid"] is False


def test_run_tests_verifier_rejects_meta_without_fixture():
    # No test_cases / prompt / test => verifier returns invalid (can't judge).
    r = run_tests_verifier("def f(): pass\n", {"entry_point": "f"})
    assert r["valid"] is False
    assert "missing" in r["trace"]


# ---------------------------------------------------------------------------
# integration with build_condition
# ---------------------------------------------------------------------------


def test_build_condition_C5_returns_self_consistency_task_graph():
    from benchmarks.conditions import build_condition

    agent = build_condition("C5", model="qwen3.5:2b", task_graph_kind="completion")
    # The agent's inner HarnessGraph._task_graph should be a SelfConsistencyTaskGraph
    assert isinstance(agent._harness._task_graph, SelfConsistencyTaskGraph)


def test_build_condition_C5_picks_tool_call_verifier_for_emit_only():
    from benchmarks.conditions import build_condition

    agent = build_condition(
        "C5",
        model="qwen3.5:2b",
        task_graph_kind="emit_only_tool_calling",
        tools=[],
    )
    tg = agent._harness._task_graph
    assert isinstance(tg, SelfConsistencyTaskGraph)
    assert tg._verifier is tool_call_valid_verifier


def test_build_condition_C5_picks_run_tests_verifier_for_completion():
    from benchmarks.conditions import build_condition

    agent = build_condition("C5", model="qwen3.5:2b", task_graph_kind="completion")
    tg = agent._harness._task_graph
    assert tg._verifier is run_tests_verifier


# Keep a breadcrumb for the orchestrator: referencing StepOutput here ensures
# an accidental import-removal would fail a test rather than silently break
# downstream trace emission that relies on the shared dataclass contract.
def test_step_output_dataclass_still_importable():
    s = StepOutput(node="generate", model="qwen3.5:2b", output="x", confidence=1.0, passed=True)
    assert s.node == "generate"


# ---------------------------------------------------------------------------
# raw-anchored sampling (sample 0 deterministic, others diverse)
# ---------------------------------------------------------------------------


def test_sample_parallel_uses_raw_llm_for_sample_zero_only():
    """Sample 0 must use the cached deterministic LLM; samples 1..n-1 use the
    diverse sibling. This prevents drift on weak-signal tasks where majority
    vote ties to first sample by index — without anchoring, that "first" is
    a random temp=0.7 sample, losing the model's strongest belief.
    """
    raw_calls: list[str] = []
    diverse_calls: list[str] = []

    def _make_recording_llm(record: list[str], content: str):
        def _invoke(_messages, **_kwargs):
            record.append(content)
            msg = MagicMock()
            msg.content = content
            return msg
        m = MagicMock()
        m.invoke.side_effect = _invoke
        m.temperature = 0.0
        m.model = "qwen3.5:2b"
        return m

    raw_llm = _make_recording_llm(raw_calls, "RAW#### 1")
    diverse_llm = _make_recording_llm(diverse_calls, "DIV#### 2")

    tg = SelfConsistencyTaskGraph(
        verifier=lambda s, m: {"valid": True, "extracted_answer": s},
        n_samples=4,
        sampling_temperature=0.7,
    )
    # Bypass _sampling_llm sibling construction by injecting our diverse llm.
    tg._sampling_llm = lambda _llm: diverse_llm  # type: ignore[assignment]

    samples = tg._sample_parallel(_make_state(), raw_llm)

    assert len(samples) == 4
    # Sample 0 must come from raw_llm exactly once.
    assert len(raw_calls) == 1
    # Samples 1..3 (3 of them) must come from diverse_llm.
    assert len(diverse_calls) == 3
    # Order matters: sample 0 is the raw one, samples 1..3 are diverse.
    assert samples[0] == "RAW#### 1"
    assert all(s == "DIV#### 2" for s in samples[1:])


def test_majority_vote_tie_breaks_to_sample_zero():
    """When all samples have unique answers (no consensus), the winner falls
    back to sample 0 — which under our raw-anchored sampling is the
    deterministic raw output, so C5 ≥ raw on weak-signal tasks.
    """
    from smboost.harness.self_consistency_graph import majority_vote

    samples = ["raw_answer_1", "div_answer_2", "div_answer_3", "div_answer_4", "div_answer_5"]
    results = [{"valid": True, "extracted_answer": i} for i in range(5)]
    winner, answer, valid_count, vote_count = majority_vote(samples, results)
    assert winner == "raw_answer_1"
    assert answer == 0
    assert valid_count == 5
    assert vote_count == 1
