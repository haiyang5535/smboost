from __future__ import annotations

from unittest.mock import patch, MagicMock

from benchmarks.bfcl.runner import (
    run_bfcl_raw,
    run_bfcl_harness,
    match_function_call,
)


_SAMPLE_TASK = {
    "task_id": "simple_0",
    "category": "simple",
    "question": "Get the weather in SF",
    "functions": [
        {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ],
    "ground_truth": [{"name": "get_weather", "arguments": {"city": "SF"}}],
}


def test_match_function_call_exact():
    call = {"name": "get_weather", "arguments": {"city": "SF"}}
    assert match_function_call(call, _SAMPLE_TASK["ground_truth"]) is True


def test_match_function_call_wrong_name():
    call = {"name": "weather", "arguments": {"city": "SF"}}
    assert match_function_call(call, _SAMPLE_TASK["ground_truth"]) is False


def test_match_function_call_wrong_arg_value():
    call = {"name": "get_weather", "arguments": {"city": "NYC"}}
    assert match_function_call(call, _SAMPLE_TASK["ground_truth"]) is False


def test_run_bfcl_raw_returns_parsed_call():
    fake_llm = MagicMock()
    # Raw LLM emits a JSON blob describing the call
    fake_llm.invoke.return_value.content = '{"name": "get_weather", "arguments": {"city": "SF"}}'

    with patch("benchmarks.bfcl.runner._make_raw_llm", return_value=fake_llm):
        rows = run_bfcl_raw(tasks=[_SAMPLE_TASK], model="qwen3.5:2b")

    assert len(rows) == 1
    r = rows[0]
    assert r["task_id"] == "simple_0"
    assert r["mode"] == "raw"
    assert r["passed"] == 1
    assert r["predicted"]["name"] == "get_weather"


def test_run_bfcl_raw_handles_malformed_json():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value.content = "I think you should call get_weather with city SF"

    with patch("benchmarks.bfcl.runner._make_raw_llm", return_value=fake_llm):
        rows = run_bfcl_raw(tasks=[_SAMPLE_TASK], model="qwen3.5:2b")

    r = rows[0]
    assert r["passed"] == 0
    assert r["failure_bucket"] == "malformed_output"


def test_run_bfcl_harness_uses_tool_calling_graph():
    """Harness mode uses ToolCallingTaskGraph via build_condition('C1', ...)."""
    fake_agent = MagicMock()
    fake_result = MagicMock()
    fake_result.output = '{"name": "get_weather", "arguments": {"city": "SF"}}'
    fake_result.stats.total_latency_s = 1.2
    fake_result.stats.retry_count = 0
    fake_agent.run.return_value = fake_result

    with patch("benchmarks.bfcl.runner.build_condition", return_value=fake_agent):
        rows = run_bfcl_harness(
            tasks=[_SAMPLE_TASK], condition="C1", model="qwen3.5:2b"
        )

    assert len(rows) == 1
    r = rows[0]
    assert r["mode"] == "C1"
    assert r["passed"] == 1
    assert r["retries"] == 0


def test_run_bfcl_harness_real_construction_with_dict_schemas():
    """Integration-style probe: really construct the graph BFCL would use.

    BFCL ships function schemas as bare dicts (``{"name": ..., "parameters":
    ...}``), not ``StructuredTool`` instances. Previously
    ``ToolCallingTaskGraph.__init__`` did ``{t.name: t for t in tools}`` and
    crashed on dicts (Finding F2).

    With the F2 fix, ``run_bfcl_harness`` now routes through
    ``EmitOnlyToolCallingTaskGraph``, which accepts bare dict schemas and
    emits — rather than executes — the tool call. This test verifies the
    real construction path no longer crashes.
    """
    from benchmarks.conditions import build_condition

    agent = build_condition(
        condition="C1",
        model="qwen3.5:2b",
        task_graph_kind="emit_only_tool_calling",
        tools=_SAMPLE_TASK["functions"],  # list[dict], not list[StructuredTool]
    )
    # Construction alone is the regression check; we shouldn't reach an LLM.
    assert agent is not None
    from smboost.tasks.emit_only_tool_calling import EmitOnlyToolCallingTaskGraph
    assert isinstance(agent._harness._task_graph, EmitOnlyToolCallingTaskGraph)
