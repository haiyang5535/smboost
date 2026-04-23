"""Shared contract for JSONL trace emission.

A ``HarnessAgent.run(...)`` may optionally write one JSON-line per step to a
trace log.  The downstream pieces that read those files — the gate runner's
summary logic, the (future) offline React trace viewer — must not re-derive
the record shape by hand.

This module is the single source of truth.  Every dict written into the
trace JSONL stream should be produced by one of these dataclasses via
``to_json_dict()``.

Schema versioning
-----------------
``schema_version`` is bumped whenever a field is renamed, removed, or changes
semantics.  Adding a new optional field is backward-compatible and does not
require a bump.

Current version: **1**.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TraceStepInput:
    """Prompt + budget snapshot at node entry."""
    prompt: str | None = None
    budget: int | None = None


@dataclass(frozen=True, slots=True)
class TraceStepOutput:
    """Generated code + truncation flag at node exit."""
    code: str | None = None
    trunc: bool = False


@dataclass(frozen=True, slots=True)
class TraceStepVerify:
    """Verify-node result — kind is e.g. ``"ast"`` / ``"grounded"``."""
    kind: str | None = None
    passed: bool | None = None
    traceback: str | None = None


@dataclass(frozen=True, slots=True)
class TraceStepRecord:
    """One JSONL line per LangGraph node execution.

    Field list comes straight from the original inline dict literal in
    ``agent.py::_emit_step``.  Do not change key names without bumping
    :data:`SCHEMA_VERSION`.

    ``entry_ts`` and ``exit_ts`` are passed through unchanged; they are
    currently often the same wall-clock value (placeholder until the
    ``harness/graph.py::_execute_step`` wrapper is added).  That's a separate
    refactor — this schema does not try to paper over it.
    """
    run_id: str
    model: str
    step_idx: int
    node: str
    entry_ts: float
    exit_ts: float
    retry_count: int
    shrinkage_level: int
    input: TraceStepInput
    output: TraceStepOutput
    verify: TraceStepVerify
    fallback_triggered: bool
    task_id: str | None = None
    condition: str | None = None
    scorer_confidence: float | None = None
    schema_version: int = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for :func:`json.dumps`.

        The nested ``input``/``output``/``verify`` sub-dataclasses are
        flattened to dicts; booleans, None, and primitives are preserved.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TraceRunStart:
    """Emitted once per :meth:`HarnessAgent.run` call, before any step."""
    run_id: str
    model: str
    condition: str | None = None
    task_id: str | None = None
    task: str | None = None
    kind: str = "run_start"
    schema_version: int = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TraceRunSummary:
    """Emitted once per :meth:`HarnessAgent.run` call, after the last step."""
    run_id: str
    model: str
    status: str  # "success" | "failed"
    step_count: int
    retry_count: int
    fallback_triggers: int
    total_latency_s: float
    condition: str | None = None
    task_id: str | None = None
    kind: str = "run_summary"
    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "SCHEMA_VERSION",
    "TraceRunStart",
    "TraceRunSummary",
    "TraceStepInput",
    "TraceStepOutput",
    "TraceStepRecord",
    "TraceStepVerify",
]
