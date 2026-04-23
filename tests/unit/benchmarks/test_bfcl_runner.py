from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

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


@pytest.mark.xfail(
    reason=(
        "Finding F2: ToolCallingTaskGraph can't consume BFCL dict schemas; "
        "needs EmitOnlyToolCallingTaskGraph.  When that lands, this becomes "
        "an xpass and should be un-marked."
    ),
    strict=False,
    raises=AttributeError,
)
def test_run_bfcl_harness_real_construction_with_dict_schemas():
    """Integration-style probe: really construct the graph BFCL would use.

    BFCL ships function schemas as bare dicts (``{"name": ..., "parameters":
    ...}``), not ``StructuredTool`` instances.  ``ToolCallingTaskGraph.__init__``
    does ``{t.name: t for t in tools}`` and therefore crashes on dicts.

    The previous unit test above patches ``build_condition`` away, so the
    crash never surfaces.  This test calls the real ``build_condition``.  It is
    marked xfail (AttributeError) so the regression is visible in the test
    report — when F2 is fixed (EmitOnlyToolCallingTaskGraph accepting dict
    schemas), this test will xpass and the marker should be removed.
    """
    from benchmarks.conditions import build_condition

    # Currently raises:
    #     AttributeError: 'dict' object has no attribute 'name'
    build_condition(
        condition="C1",
        model="qwen3.5:2b",
        task_graph_kind="tool_calling",
        tools=_SAMPLE_TASK["functions"],  # list[dict], not list[StructuredTool]
    )
