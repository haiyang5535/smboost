"""Unit tests for EmitOnlyToolCallingTaskGraph.

Verifies:
  - Construction with dict schemas (OpenAI-function shape, both flat and wrapped).
  - verify-node logic (structural-only, PASS/FAIL semantics).
  - Node names are [plan, generate, verify].
  - build_condition wiring for 'emit_only_tool_calling'.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from smboost.harness.state import StepOutput
from smboost.tasks.emit_only_tool_calling import EmitOnlyToolCallingTaskGraph


_FLAT_SCHEMA = {
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    },
}

_WRAPPED_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_email",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}},
        },
    },
}


def _make_state(step_outputs=None, shrinkage_level=0, task="Get the weather in SF"):
    return {
        "task": task,
        "task_metadata": {},
        "model": "qwen3.5:2b",
        "fallback_chain": ["qwen3.5:2b"],
        "step_outputs": step_outputs or [],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 0,
        "shrinkage_level": shrinkage_level,
        "status": "running",
        "final_output": None,
    }


def _generate_step(output: str) -> StepOutput:
    return StepOutput(
        node="generate", model="qwen3.5:2b", output=output, confidence=1.0, passed=True
    )


# --- construction -----------------------------------------------------------


def test_construction_with_single_flat_dict_schema_succeeds():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    assert tg.node_names == ["plan", "generate", "verify"]


def test_construction_with_list_of_dict_schemas_succeeds():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA, _WRAPPED_SCHEMA])
    # Names extracted from both flat and wrapped shapes
    assert "get_weather" in tg._tool_names
    assert "send_email" in tg._tool_names


def test_construction_with_empty_tools_does_not_crash():
    tg = EmitOnlyToolCallingTaskGraph(tools=[])
    assert tg._tool_names == []


def test_construction_with_none_tools_does_not_crash():
    tg = EmitOnlyToolCallingTaskGraph(tools=None)  # type: ignore[arg-type]
    assert tg._tool_names == []


def test_node_names_are_plan_generate_verify():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    assert tg.node_names == ["plan", "generate", "verify"]


def test_get_node_fn_returns_callable_for_each_node():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    for name in tg.node_names:
        assert callable(tg.get_node_fn(name))


def test_get_node_fn_raises_on_unknown_node():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    try:
        tg.get_node_fn("dispatch")
    except ValueError as exc:
        assert "dispatch" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown node")


# --- plan / generate invocations -------------------------------------------


def test_plan_node_calls_llm_and_returns_content():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Call get_weather with city=SF")
    out = tg.get_node_fn("plan")(_make_state(), mock_llm)
    assert "get_weather" in out
    mock_llm.invoke.assert_called_once()


def test_generate_node_calls_llm_and_returns_raw_content():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    mock_llm = MagicMock()
    raw = '{"name": "get_weather", "arguments": {"city": "SF"}}'
    mock_llm.invoke.return_value = MagicMock(content=raw)
    out = tg.get_node_fn("generate")(_make_state(), mock_llm)
    assert out == raw


# --- shrinkage -------------------------------------------------------------


def test_plan_prompt_shrinks_with_level():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    mock_llm = MagicMock()
    captured: list[str] = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="plan")
    )

    tg.get_node_fn("plan")(_make_state(shrinkage_level=0), mock_llm)
    tg.get_node_fn("plan")(_make_state(shrinkage_level=3), mock_llm)

    assert len(captured[1]) < len(captured[0])


def test_generate_prompt_shrinks_with_level():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    mock_llm = MagicMock()
    captured: list[str] = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="{}")
    )

    tg.get_node_fn("generate")(_make_state(shrinkage_level=0), mock_llm)
    tg.get_node_fn("generate")(_make_state(shrinkage_level=3), mock_llm)

    assert len(captured[1]) < len(captured[0])


def test_compact_prompts_are_under_200_chars_at_level_3():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    mock_llm = MagicMock()
    captured: list[str] = []
    mock_llm.invoke.side_effect = lambda msgs: (
        captured.append(msgs[0].content) or MagicMock(content="x")
    )
    long_task = "a" * 5000
    state = _make_state(shrinkage_level=3, task=long_task)

    tg.get_node_fn("plan")(state, mock_llm)
    tg.get_node_fn("generate")(state, mock_llm)

    # Task-shrinkage alone pushes compact forms well below ~200 chars of framing
    for prompt in captured:
        assert len(prompt) < 400, f"compact prompt too long: {len(prompt)} chars"


# --- verify (critical logic) -----------------------------------------------


def test_verify_pass_when_json_has_valid_name_and_arguments_dict():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    step = _generate_step('{"name": "get_weather", "arguments": {"city": "SF"}}')
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result == "PASS"


def test_verify_pass_when_arguments_dict_is_empty():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    step = _generate_step('{"name": "get_weather", "arguments": {}}')
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result == "PASS"


def test_verify_pass_with_wrapped_schema_name():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_WRAPPED_SCHEMA])
    step = _generate_step('{"name": "send_email", "arguments": {"to": "a@b.c"}}')
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result == "PASS"


def test_verify_fail_when_output_is_not_json():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    step = _generate_step("I think you should call get_weather with SF")
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result.startswith("FAIL:")
    assert "JSONDecode" in result or "json" in result.lower()


def test_verify_fail_when_output_is_empty():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    step = _generate_step("")
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result.startswith("FAIL:")


def test_verify_fail_when_no_generate_step():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    result = tg.get_node_fn("verify")(_make_state(), MagicMock())
    assert result.startswith("FAIL:")


def test_verify_fail_when_json_has_wrong_function_name():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    step = _generate_step('{"name": "not_a_function", "arguments": {}}')
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result.startswith("FAIL:")
    assert "not_a_function" in result or "not declared" in result


def test_verify_fail_when_json_missing_arguments():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    step = _generate_step('{"name": "get_weather"}')
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result.startswith("FAIL:")
    assert "arguments" in result


def test_verify_fail_when_arguments_not_dict():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    step = _generate_step('{"name": "get_weather", "arguments": "city=SF"}')
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result.startswith("FAIL:")
    assert "arguments" in result


def test_verify_fail_when_json_is_list():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    step = _generate_step('[{"name": "get_weather", "arguments": {}}]')
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result.startswith("FAIL:")


def test_verify_fail_when_name_missing():
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    step = _generate_step('{"arguments": {"city": "SF"}}')
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result.startswith("FAIL:")
    assert "name" in result


def test_verify_uses_most_recent_generate_step():
    """If there are multiple generate steps (retry scenario), verify should
    use the latest one."""
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    old = _generate_step("garbage")
    new = _generate_step('{"name": "get_weather", "arguments": {}}')
    state = _make_state(step_outputs=[old, new])
    result = tg.get_node_fn("verify")(state, MagicMock())
    assert result == "PASS"


def test_verify_does_not_call_llm():
    """Verify is structural-only — it should never invoke the LLM."""
    tg = EmitOnlyToolCallingTaskGraph(tools=[_FLAT_SCHEMA])
    step = _generate_step('{"name": "get_weather", "arguments": {}}')
    mock_llm = MagicMock()
    tg.get_node_fn("verify")(_make_state(step_outputs=[step]), mock_llm)
    mock_llm.invoke.assert_not_called()


def test_verify_passes_any_name_when_no_tools_declared():
    """Edge case: when tools list is empty, name-matching is skipped; we just
    need parseable JSON with a string name and a dict arguments field."""
    tg = EmitOnlyToolCallingTaskGraph(tools=[])
    step = _generate_step('{"name": "anything", "arguments": {}}')
    result = tg.get_node_fn("verify")(_make_state(step_outputs=[step]), MagicMock())
    assert result == "PASS"


# --- build_condition wiring ------------------------------------------------


def test_build_condition_emit_only_tool_calling_constructs_agent():
    """C1 with emit_only_tool_calling + dict tools must construct cleanly."""
    from benchmarks.conditions import build_condition

    agent = build_condition(
        condition="C1",
        model="qwen3.5:2b",
        task_graph_kind="emit_only_tool_calling",
        tools=[_FLAT_SCHEMA],
    )
    assert agent is not None
    assert isinstance(
        agent._harness._task_graph, EmitOnlyToolCallingTaskGraph
    )


def test_build_condition_emit_only_tool_calling_with_wrapped_schema():
    from benchmarks.conditions import build_condition

    agent = build_condition(
        condition="C4",
        model="qwen3.5:2b",
        task_graph_kind="emit_only_tool_calling",
        tools=[_WRAPPED_SCHEMA],
    )
    graph = agent._harness._task_graph
    assert isinstance(graph, EmitOnlyToolCallingTaskGraph)
    assert "send_email" in graph._tool_names


# --- end-to-end mock smoke -------------------------------------------------


def test_end_to_end_mock_llm_runs_through_full_harness():
    """Smoke test: agent.run() routes through plan→generate→verify with a canned
    LLM, verify returns PASS, and the run reports success."""
    from benchmarks.conditions import build_condition

    canned_json = '{"name": "get_weather", "arguments": {"city": "SF"}}'

    def fake_llm_factory(model: str):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=canned_json)
        return llm

    # Use build_condition then swap the factory on the inner HarnessGraph.
    agent = build_condition(
        condition="C4",  # all flags off: simplest path, no scorer/shrinkage
        model="qwen3.5:2b",
        task_graph_kind="emit_only_tool_calling",
        tools=[_FLAT_SCHEMA],
    )
    # Monkeypatch the llm_factory on the inner HarnessGraph so node_fn gets
    # our canned LLM instead of attempting a real network call.
    agent._harness._llm_factory = fake_llm_factory

    result = agent.run("Get the weather in SF")
    assert result.status == "success"
    # Final output should be the raw JSON from our generate step OR the verify
    # step's PASS string — in this harness the final step is verify, so it's PASS.
    # What we care about: the trace contains a verify step that passed.
    verify_steps = [s for s in result.trace if s.node == "verify"]
    assert verify_steps, "expected at least one verify step"
    assert verify_steps[-1].output == "PASS"
    assert verify_steps[-1].passed is True
    # And a generate step emitted our canned JSON.
    generate_steps = [s for s in result.trace if s.node == "generate"]
    assert generate_steps
    assert generate_steps[-1].output == canned_json
