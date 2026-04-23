"""Contract tests for the shared JSONL trace schema.

These tests lock down the field names, nesting, and JSON-serializability of
``TraceStepRecord`` + the two run-level markers.  Anyone adding a new key
should update the expected-keys assertion below *and* bump
``SCHEMA_VERSION`` if they rename/remove/re-type an existing field.
"""
from __future__ import annotations

import json

from smboost.harness.trace_schema import (
    SCHEMA_VERSION,
    TraceRunStart,
    TraceRunSummary,
    TraceStepInput,
    TraceStepOutput,
    TraceStepRecord,
    TraceStepVerify,
)


# Every key that must appear in a TraceStepRecord JSON line.  If this list
# changes, bump SCHEMA_VERSION.
_STEP_RECORD_TOP_KEYS = {
    "schema_version",
    "run_id",
    "task_id",
    "model",
    "condition",
    "step_idx",
    "node",
    "entry_ts",
    "exit_ts",
    "retry_count",
    "shrinkage_level",
    "scorer_confidence",
    "input",
    "output",
    "verify",
    "fallback_triggered",
}


def _build_step_record(**overrides) -> TraceStepRecord:
    defaults = dict(
        run_id="run-abc",
        task_id="HEval/0",
        model="qwen3.5:2b",
        condition="C1",
        step_idx=0,
        node="generate",
        entry_ts=1000.0,
        exit_ts=1000.5,
        retry_count=0,
        shrinkage_level=0,
        scorer_confidence=0.8,
        input=TraceStepInput(prompt="def f():", budget=4096),
        output=TraceStepOutput(code="def f():\n    return 1", trunc=False),
        verify=TraceStepVerify(kind="ast", passed=True, traceback=None),
        fallback_triggered=False,
    )
    defaults.update(overrides)
    return TraceStepRecord(**defaults)


def test_trace_step_record_has_expected_top_level_keys():
    rec = _build_step_record()
    d = rec.to_json_dict()
    assert set(d.keys()) == _STEP_RECORD_TOP_KEYS


def test_trace_step_record_schema_version_is_current():
    rec = _build_step_record()
    assert rec.to_json_dict()["schema_version"] == SCHEMA_VERSION


def test_trace_step_record_nested_input_keys():
    d = _build_step_record().to_json_dict()
    assert set(d["input"].keys()) == {"prompt", "budget"}


def test_trace_step_record_nested_output_keys():
    d = _build_step_record().to_json_dict()
    assert set(d["output"].keys()) == {"code", "trunc"}


def test_trace_step_record_nested_verify_keys():
    d = _build_step_record().to_json_dict()
    assert set(d["verify"].keys()) == {"kind", "passed", "traceback"}


def test_trace_step_record_is_json_serializable():
    d = _build_step_record().to_json_dict()
    # Round-trip so any non-JSON-friendly value raises.
    raw = json.dumps(d)
    reloaded = json.loads(raw)
    assert reloaded["run_id"] == "run-abc"
    assert reloaded["input"]["budget"] == 4096
    assert reloaded["verify"]["passed"] is True


def test_trace_step_record_optional_fields_default_to_none():
    rec = _build_step_record(task_id=None, condition=None, scorer_confidence=None)
    d = rec.to_json_dict()
    assert d["task_id"] is None
    assert d["condition"] is None
    assert d["scorer_confidence"] is None


def test_trace_step_record_is_frozen():
    import dataclasses
    rec = _build_step_record()
    try:
        rec.step_idx = 99  # type: ignore[misc]
    except (dataclasses.FrozenInstanceError, AttributeError):
        return
    raise AssertionError("TraceStepRecord should be frozen")


def test_run_start_marker_shape():
    rec = TraceRunStart(
        run_id="r1", model="qwen3.5:2b", condition="C1", task_id="HEval/0",
        task="Write f()",
    )
    d = rec.to_json_dict()
    assert d["kind"] == "run_start"
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["run_id"] == "r1"
    # Round-trip JSON.
    assert json.loads(json.dumps(d))["task"] == "Write f()"


def test_run_summary_marker_shape():
    rec = TraceRunSummary(
        run_id="r1", model="qwen3.5:2b", status="success",
        step_count=3, retry_count=1, fallback_triggers=0, total_latency_s=1.2,
        condition="C1", task_id="HEval/0",
    )
    d = rec.to_json_dict()
    assert d["kind"] == "run_summary"
    assert d["status"] == "success"
    assert d["schema_version"] == SCHEMA_VERSION
    # Round-trip JSON.
    reloaded = json.loads(json.dumps(d))
    assert reloaded["step_count"] == 3
    assert reloaded["total_latency_s"] == 1.2
