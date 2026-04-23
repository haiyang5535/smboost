"""Tests for JSONL trace emission from HarnessAgent.run(...).

Per spec §5.1/§5.2 of docs/superpowers/specs/2026-04-23-yc-sprint-pivot-design.md,
each run with `trace_log_path` set should append one `run_start` line, one JSON
line per trace step, and one `summary` line (emitted last, even on exception).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from smboost import HarnessAgent, InvariantSuite
from smboost.harness.state import StepOutput


def _step(node: str = "plan", output: str = "done", passed: bool = True, confidence: float = 1.0) -> StepOutput:
    return StepOutput(node=node, model="qwen3.5:2b", output=output, confidence=confidence, passed=passed)


def _final_state(status: str = "success", step_outputs=None, final_output: str | None = "done"):
    return {
        "task": "test",
        "task_metadata": {},
        "model": "qwen3.5:2b",
        "fallback_chain": ["qwen3.5:2b"],
        "step_outputs": step_outputs if step_outputs is not None else [_step()],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 1,
        "shrinkage_level": 0,
        "status": status,
        "final_output": final_output,
    }


def _read_lines(path: Path) -> list[dict]:
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_no_emission_when_trace_log_path_is_none(tmp_path: Path):
    """Default behavior: no kwarg means no file is created, no behavior change."""
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
    )
    trace_file = tmp_path / "should_not_exist.jsonl"
    with patch.object(agent._harness, "invoke", return_value=_final_state()):
        result = agent.run("write hello world")

    assert result.status == "success"
    assert not trace_file.exists()
    # The tmp_path directory should still be empty (no files created by run)
    assert list(tmp_path.iterdir()) == []


def test_schema_version_marker_is_first_line(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        trace_log_path=trace_file,
    )
    with patch.object(agent._harness, "invoke", return_value=_final_state()):
        agent.run("task", task_id="HEval/42", condition="C1")

    lines = _read_lines(trace_file)
    assert lines, "trace file should have at least one line"
    first = lines[0]
    assert first.get("schema_version") == 1
    assert first.get("event") == "run_start"
    assert first.get("task_id") == "HEval/42"
    assert first.get("condition") == "C1"
    assert first.get("model") == "qwen3.5:2b"
    assert "run_id" in first and isinstance(first["run_id"], str) and first["run_id"]


def test_one_json_line_per_trace_step(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        trace_log_path=trace_file,
    )
    steps = [
        _step(node="plan", output="plan output", passed=True, confidence=0.8),
        _step(node="execute", output="code", passed=True, confidence=0.9),
        _step(node="verify", output="PASS", passed=True, confidence=1.0),
    ]
    with patch.object(agent._harness, "invoke", return_value=_final_state(step_outputs=steps)):
        result = agent.run("task")

    lines = _read_lines(trace_file)
    # First is run_start, last is summary. Middle lines correspond to steps.
    middle = lines[1:-1]
    assert len(middle) == len(result.trace) == 3

    for idx, (emitted, step) in enumerate(zip(middle, result.trace)):
        assert emitted.get("event") is None or emitted.get("event") == "step"
        assert emitted["step_idx"] == idx
        assert emitted["node"] == step.node
        assert emitted["scorer_confidence"] == step.confidence
        assert "output" in emitted
        assert "run_id" in emitted


def test_summary_line_is_last_and_reflects_stats(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        trace_log_path=trace_file,
    )
    state = _final_state(status="success", final_output="final code")
    state["retry_count"] = 2
    with patch.object(agent._harness, "invoke", return_value=state):
        result = agent.run("task")

    lines = _read_lines(trace_file)
    last = lines[-1]
    assert last["event"] == "summary"
    assert last["passed"] is True
    assert last["retries"] == result.stats.retry_count == 2
    assert "wall_ms" in last and isinstance(last["wall_ms"], int)
    assert "final_code" in last
    assert "run_id" in last


def test_failed_status_reflected_in_summary(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        trace_log_path=trace_file,
    )
    state = _final_state(status="failed", final_output=None)
    with patch.object(agent._harness, "invoke", return_value=state):
        agent.run("task")

    lines = _read_lines(trace_file)
    assert lines[-1]["event"] == "summary"
    assert lines[-1]["passed"] is False


def test_two_runs_append_to_same_file(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        trace_log_path=trace_file,
    )
    with patch.object(agent._harness, "invoke", return_value=_final_state()):
        agent.run("task1", task_id="T1")
        agent.run("task2", task_id="T2")

    lines = _read_lines(trace_file)
    run_starts = [i for i, line in enumerate(lines) if line.get("event") == "run_start"]
    summaries = [i for i, line in enumerate(lines) if line.get("event") == "summary"]

    assert len(run_starts) == 2
    assert len(summaries) == 2
    # Order: run_start_1 < summary_1 < run_start_2 < summary_2
    assert run_starts[0] < summaries[0] < run_starts[1] < summaries[1]
    # Distinct run_ids
    assert lines[run_starts[0]]["run_id"] != lines[run_starts[1]]["run_id"]
    # task_ids preserved per run
    assert lines[run_starts[0]]["task_id"] == "T1"
    assert lines[run_starts[1]]["task_id"] == "T2"


def test_large_output_is_truncated(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        trace_log_path=trace_file,
    )
    huge = "x" * 10_000
    steps = [_step(node="execute", output=huge, passed=True)]
    state = _final_state(step_outputs=steps, final_output=huge)
    with patch.object(agent._harness, "invoke", return_value=state):
        agent.run("task")

    lines = _read_lines(trace_file)
    step_line = lines[1]  # after run_start
    emitted_code = step_line["output"]["code"]
    assert len(emitted_code) <= 2100  # ~2000 cap plus small ellipsis marker slack
    assert emitted_code.endswith("...[truncated]") or "...[truncated]" in emitted_code

    summary = lines[-1]
    assert len(summary["final_code"]) <= 2100
    assert "...[truncated]" in summary["final_code"]


def test_exception_in_run_still_emits_summary(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        trace_log_path=trace_file,
    )

    def _boom(_state):
        raise RuntimeError("llm exploded")

    with patch.object(agent._harness, "invoke", side_effect=_boom):
        with pytest.raises(RuntimeError):
            agent.run("task", task_id="T1", condition="C1")

    assert trace_file.exists()
    lines = _read_lines(trace_file)
    assert lines, "should have emitted at least run_start before the exception"
    assert lines[0]["event"] == "run_start"
    assert lines[-1]["event"] == "summary"
    assert lines[-1]["passed"] is False


def test_task_id_and_condition_flow_through_every_line(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        trace_log_path=trace_file,
    )
    steps = [
        _step(node="plan", output="p", passed=True),
        _step(node="execute", output="e", passed=True),
    ]
    with patch.object(agent._harness, "invoke", return_value=_final_state(step_outputs=steps)):
        agent.run("task", task_id="HEval/0", condition="C1")

    lines = _read_lines(trace_file)
    # Summary line has the minimal schema; but per-step and run_start lines must carry identifiers.
    # Every line except the summary must include task_id/condition. Summary carries run_id.
    for line in lines[:-1]:
        assert line["task_id"] == "HEval/0"
        assert line["condition"] == "C1"
        assert line["model"] == "qwen3.5:2b"

    # All lines share run_id
    run_ids = {line["run_id"] for line in lines}
    assert len(run_ids) == 1


def test_task_id_and_condition_default_to_null(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        trace_log_path=trace_file,
    )
    with patch.object(agent._harness, "invoke", return_value=_final_state()):
        agent.run("task")

    lines = _read_lines(trace_file)
    assert lines[0]["task_id"] is None
    assert lines[0]["condition"] is None


def test_backwards_compatible_no_new_required_args():
    """HarnessAgent(model, invariants) still works without trace_log_path."""
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
    )
    # Unchanged attributes
    assert agent.model == "qwen3.5:2b"
    assert agent.fallback_chain == ["qwen3.5:2b"]


def test_step_records_carry_passed_field(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        trace_log_path=trace_file,
    )
    steps = [
        _step(node="plan", output="p", passed=False, confidence=0.1),
        _step(node="plan", output="p2", passed=True, confidence=0.9),
    ]
    with patch.object(agent._harness, "invoke", return_value=_final_state(step_outputs=steps)):
        agent.run("task")

    lines = _read_lines(trace_file)
    step_lines = lines[1:-1]
    assert step_lines[0]["verify"]["passed"] is False
    assert step_lines[1]["verify"]["passed"] is True
