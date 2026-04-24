"""Unit tests for RealToolTaskGraph (condition C6).

Strategy: mocked LLM that returns canned plan + canned verify. The sandbox
and memory are REAL (they're cheap and fast) — the tests verify that the
graph correctly wires LLM output → tool execution → verifier decision.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from smboost.harness.conditions import build_condition
from smboost.harness.real_tool_graph import RealToolTaskGraph
from smboost.harness.state import StepOutput
from smboost.tools import MemoryStore, PythonSandbox


def _make_state(step_outputs=None, task="What is 2+2?", task_metadata=None):
    return {
        "task": task,
        "task_metadata": task_metadata or {},
        "model": "qwen3.5:2b",
        "fallback_chain": ["qwen3.5:2b"],
        "step_outputs": step_outputs or [],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 0,
        "shrinkage_level": 0,
        "status": "running",
        "final_output": None,
    }


def _step(node: str, output: str) -> StepOutput:
    return StepOutput(
        node=node, model="qwen3.5:2b", output=output, confidence=1.0, passed=True,
    )


# --- construction / TaskGraph contract --------------------------------------


def test_node_names_are_plan_call_verify():
    g = RealToolTaskGraph()
    assert g.node_names == ["plan", "call", "verify"]


def test_get_node_fn_returns_callable_for_each_node():
    g = RealToolTaskGraph()
    for name in g.node_names:
        assert callable(g.get_node_fn(name))


def test_get_node_fn_raises_on_unknown_node():
    g = RealToolTaskGraph()
    with pytest.raises(ValueError):
        g.get_node_fn("dispatch")


def test_default_construction_works_without_args():
    g = RealToolTaskGraph()
    assert isinstance(g._sandbox, PythonSandbox)
    assert isinstance(g._memory, MemoryStore)
    assert g._n_iterations >= 1


def test_construction_accepts_injected_sandbox_and_memory():
    sb = PythonSandbox()
    mem = MemoryStore()
    g = RealToolTaskGraph(sandbox=sb, memory=mem, n_iterations=7)
    assert g._sandbox is sb
    assert g._memory is mem
    assert g._n_iterations == 7


# --- planner ----------------------------------------------------------------


def test_planner_invokes_llm_and_returns_content():
    g = RealToolTaskGraph()
    mock_llm = MagicMock()
    canned = '{"steps": [{"action": "code", "payload": "result = 4"}]}'
    mock_llm.invoke.return_value = MagicMock(content=canned)
    out = g.get_node_fn("plan")(_make_state(), mock_llm)
    assert out == canned
    mock_llm.invoke.assert_called_once()


def test_planner_prompt_includes_task_text():
    g = RealToolTaskGraph()
    mock_llm = MagicMock()
    captured: list[str] = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="{}")
    )
    g.get_node_fn("plan")(_make_state(task="Compute 6*7"), mock_llm)
    assert "Compute 6*7" in captured[0]


def test_planner_prompt_includes_memory_keys_when_present():
    mem = MemoryStore()
    mem.set("previous_result", 42)
    g = RealToolTaskGraph(memory=mem)
    mock_llm = MagicMock()
    captured: list[str] = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="{}")
    )
    g.get_node_fn("plan")(_make_state(), mock_llm)
    assert "previous_result" in captured[0]


# --- caller (tool execution) ------------------------------------------------


def test_call_executes_code_step_via_sandbox():
    g = RealToolTaskGraph()
    plan = json.dumps({
        "steps": [{"action": "code", "payload": "result = 2 + 2"}]
    })
    state = _make_state(step_outputs=[_step("plan", plan)])
    raw = g.get_node_fn("call")(state, MagicMock())
    obj = json.loads(raw)
    assert len(obj["observations"]) == 1
    obs = obj["observations"][0]
    assert obs["ok"] is True
    assert obs["output"] == "4"  # repr(4) from `result` channel


def test_call_executes_memory_set_and_get():
    g = RealToolTaskGraph()
    plan = json.dumps({
        "steps": [
            {"action": "memory_set", "payload": 'x=42'},
            {"action": "memory_get", "payload": "x"},
        ]
    })
    state = _make_state(step_outputs=[_step("plan", plan)])
    raw = g.get_node_fn("call")(state, MagicMock())
    obj = json.loads(raw)
    assert len(obj["observations"]) == 2
    assert obj["observations"][0]["ok"] is True
    assert obj["observations"][1]["ok"] is True
    assert obj["observations"][1]["output"] == "42"


def test_call_answer_action_terminates_early():
    g = RealToolTaskGraph()
    plan = json.dumps({
        "steps": [
            {"action": "code", "payload": "print('x')"},
            {"action": "answer", "payload": "42"},
            # this step must not execute:
            {"action": "code", "payload": "1/0"},
        ]
    })
    state = _make_state(step_outputs=[_step("plan", plan)])
    raw = g.get_node_fn("call")(state, MagicMock())
    obj = json.loads(raw)
    assert obj["answer"] == "42"
    # At most 2 observations were collected (answer terminated)
    assert len(obj["observations"]) == 2


def test_call_handles_plan_with_invalid_json():
    g = RealToolTaskGraph()
    state = _make_state(step_outputs=[_step("plan", "not json at all")])
    raw = g.get_node_fn("call")(state, MagicMock())
    obj = json.loads(raw)
    assert obj["observations"] == []
    assert "parse error" in (obj.get("error") or "")


def test_call_handles_missing_plan_step():
    g = RealToolTaskGraph()
    state = _make_state()
    raw = g.get_node_fn("call")(state, MagicMock())
    obj = json.loads(raw)
    assert obj["observations"] == []
    assert obj["answer"] is None


def test_call_handles_plan_without_steps_list():
    g = RealToolTaskGraph()
    plan = json.dumps({"not_steps": []})
    state = _make_state(step_outputs=[_step("plan", plan)])
    raw = g.get_node_fn("call")(state, MagicMock())
    obj = json.loads(raw)
    assert obj["observations"] == []


def test_call_extracts_plan_from_fenced_json_block():
    g = RealToolTaskGraph()
    plan_raw = (
        "sure, here is the plan:\n"
        "```json\n"
        '{"steps": [{"action": "code", "payload": "result = 9"}]}\n'
        "```\n"
    )
    state = _make_state(step_outputs=[_step("plan", plan_raw)])
    raw = g.get_node_fn("call")(state, MagicMock())
    obj = json.loads(raw)
    assert obj["observations"][0]["ok"] is True
    assert obj["observations"][0]["output"] == "9"


def test_call_does_not_invoke_llm():
    """The call node executes tools; it should never call the LLM."""
    g = RealToolTaskGraph()
    plan = json.dumps({"steps": [{"action": "answer", "payload": "x"}]})
    state = _make_state(step_outputs=[_step("plan", plan)])
    mock_llm = MagicMock()
    g.get_node_fn("call")(state, mock_llm)
    mock_llm.invoke.assert_not_called()


def test_call_unknown_action_records_error():
    g = RealToolTaskGraph()
    plan = json.dumps({"steps": [{"action": "fly_to_moon", "payload": ""}]})
    state = _make_state(step_outputs=[_step("plan", plan)])
    raw = g.get_node_fn("call")(state, MagicMock())
    obj = json.loads(raw)
    assert obj["observations"][0]["ok"] is False
    assert "unknown action" in (obj["observations"][0].get("error") or "")


# --- verifier ---------------------------------------------------------------


def test_verify_pass_when_expected_answer_matches():
    g = RealToolTaskGraph()
    call_out = json.dumps({
        "observations": [{"step": {"action": "answer"}, "ok": True, "output": "42"}],
        "answer": "42",
    })
    state = _make_state(
        step_outputs=[_step("plan", "{}"), _step("call", call_out)],
        task_metadata={"expected_answer": "42"},
    )
    result = g.get_node_fn("verify")(state, MagicMock())
    assert result == "PASS"


def test_verify_fail_when_expected_answer_differs():
    g = RealToolTaskGraph()
    call_out = json.dumps({
        "observations": [{"step": {"action": "answer"}, "ok": True, "output": "41"}],
        "answer": "41",
    })
    state = _make_state(
        step_outputs=[_step("plan", "{}"), _step("call", call_out)],
        task_metadata={"expected_answer": "42"},
    )
    result = g.get_node_fn("verify")(state, MagicMock())
    assert result.startswith("FAIL:")
    assert "42" in result


def test_verify_fail_when_no_call_output():
    g = RealToolTaskGraph()
    state = _make_state()
    result = g.get_node_fn("verify")(state, MagicMock())
    assert result.startswith("FAIL:")


def test_verify_fail_when_call_output_not_json():
    g = RealToolTaskGraph()
    state = _make_state(step_outputs=[_step("call", "garbage")])
    result = g.get_node_fn("verify")(state, MagicMock())
    assert result.startswith("FAIL:")


def test_verify_with_no_expected_answer_asks_llm():
    g = RealToolTaskGraph()
    call_out = json.dumps({
        "observations": [{"step": {"action": "answer"}, "ok": True, "output": "hello"}],
        "answer": "hello",
    })
    state = _make_state(step_outputs=[_step("call", call_out)])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="PASS")
    result = g.get_node_fn("verify")(state, mock_llm)
    assert result == "PASS"
    mock_llm.invoke.assert_called_once()


def test_verify_with_no_expected_answer_llm_fails():
    g = RealToolTaskGraph()
    call_out = json.dumps({
        "observations": [{"ok": True, "output": "wrong"}],
        "answer": "wrong",
    })
    state = _make_state(step_outputs=[_step("call", call_out)])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="FAIL: answer is wrong")
    result = g.get_node_fn("verify")(state, mock_llm)
    assert result.startswith("FAIL:")


def test_verify_short_circuits_on_explicit_error_field():
    g = RealToolTaskGraph()
    call_out = json.dumps({
        "observations": [],
        "answer": None,
        "error": "plan parse error: foo",
    })
    state = _make_state(
        step_outputs=[_step("call", call_out)],
        task_metadata={"expected_answer": "anything"},
    )
    result = g.get_node_fn("verify")(state, MagicMock())
    assert result.startswith("FAIL:")
    assert "parse error" in result


# --- end-to-end smoke -------------------------------------------------------


def test_build_condition_c6_returns_real_tool_graph():
    g = build_condition("C6")
    assert isinstance(g, RealToolTaskGraph)


def test_build_condition_c6_accepts_n_iterations_kwarg():
    g = build_condition("C6", n_iterations=5)
    assert isinstance(g, RealToolTaskGraph)
    assert g._n_iterations == 5


def test_build_condition_c6_ignores_unknown_kwargs():
    # Keep this permissive per merge-note: we silently ignore extras.
    g = build_condition("C6", totally_unknown=True)
    assert isinstance(g, RealToolTaskGraph)


def test_build_condition_c5_soft_missing_raises_not_implemented():
    # On this worktree, Agent 2's C5 module is not present, so C5 should
    # surface a NotImplementedError (not a plain ImportError) so the error
    # message is actionable for downstream callers.
    try:
        from smboost.harness import self_consistency_graph  # noqa: F401
        has_c5 = True
    except Exception:
        has_c5 = False
    if has_c5:
        # If Agent 2 has already been merged at build time, C5 must succeed.
        g = build_condition("C5")
        assert g is not None
    else:
        with pytest.raises(NotImplementedError):
            build_condition("C5")


def test_build_condition_c1_returns_completion_graph():
    from smboost.tasks.completion import CompletionTaskGraph
    g = build_condition("C1")
    assert isinstance(g, CompletionTaskGraph)


def test_build_condition_unknown_condition_raises():
    with pytest.raises(ValueError):
        build_condition("C99")


def test_end_to_end_mock_planner_plus_real_sandbox_produces_pass():
    """Smoke: planner returns a canned 2-step plan, caller executes for real
    via the sandbox, verifier sees the expected answer → PASS.

    We drive the three nodes manually (no HarnessGraph) to keep the test
    independent of harness retry logic.
    """
    g = RealToolTaskGraph()

    # Canned planner: compute 6*7, then answer 42.
    plan_json = json.dumps({
        "steps": [
            {"action": "code", "payload": "result = 6 * 7"},
            {"action": "answer", "payload": "42"},
        ]
    })
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=plan_json)

    state = _make_state(
        task="What is 6 * 7?",
        task_metadata={"expected_answer": "42"},
    )

    # Node 1: plan
    plan_out = g.get_node_fn("plan")(state, mock_llm)
    state["step_outputs"] = state["step_outputs"] + [_step("plan", plan_out)]

    # Node 2: call (no LLM involvement)
    call_out = g.get_node_fn("call")(state, MagicMock())
    state["step_outputs"] = state["step_outputs"] + [_step("call", call_out)]

    call_obj = json.loads(call_out)
    assert call_obj["answer"] == "42"
    # both steps succeeded:
    assert all(obs["ok"] for obs in call_obj["observations"])

    # Node 3: verify (no LLM needed since expected_answer is supplied)
    verify_out = g.get_node_fn("verify")(state, MagicMock())
    assert verify_out == "PASS"


def test_end_to_end_planner_wrong_answer_verifier_fails():
    g = RealToolTaskGraph()

    plan_json = json.dumps({
        "steps": [{"action": "answer", "payload": "41"}]
    })
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=plan_json)

    state = _make_state(
        task="What is 6 * 7?",
        task_metadata={"expected_answer": "42"},
    )
    plan_out = g.get_node_fn("plan")(state, mock_llm)
    state["step_outputs"] = state["step_outputs"] + [_step("plan", plan_out)]
    call_out = g.get_node_fn("call")(state, MagicMock())
    state["step_outputs"] = state["step_outputs"] + [_step("call", call_out)]

    verify_out = g.get_node_fn("verify")(state, MagicMock())
    assert verify_out.startswith("FAIL:")


def test_grammar_fallback_when_module_absent():
    """If smboost.llm.grammar is not present (Agent 1 not merged), the graph
    should still construct and operate. The _try_import_grammar_helpers path
    swallows any ImportError and leaves grammar=None."""
    g = RealToolTaskGraph()
    # Either the grammar module is present (then these are not None) or
    # absent (then both are None). Either state is fine; what matters is that
    # the graph still runs.
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"steps": [{"action": "answer", "payload": "ok"}]}'
    )
    out = g.get_node_fn("plan")(_make_state(), mock_llm)
    assert "steps" in out
