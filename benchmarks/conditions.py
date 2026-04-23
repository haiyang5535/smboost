"""Shared 6-condition HarnessAgent factory for any benchmark that needs C1-C6 ablation.

Usage:
    agent = build_condition("C1", model="qwen3.5:2b", task_graph_kind="completion")
    agent = build_condition("C3", model="qwen3.5:2b", task_graph_kind="tool_calling", tools=[my_tool])
"""
from __future__ import annotations

from typing import Any

from smboost import HarnessAgent, InvariantSuite
from smboost.llm.runtime import get_benchmark_llm_factory
from smboost.tasks.completion import CompletionTaskGraph
from smboost.tasks.emit_only_tool_calling import EmitOnlyToolCallingTaskGraph
from smboost.tasks.tool_calling import ToolCallingTaskGraph


CONDITION_NAMES = ("C1", "C2", "C3", "C4", "C5", "C6")

# Flag table for each condition. C4 is the floor baseline (plain LangGraph retry).
_FLAGS: dict[str, dict[str, bool]] = {
    "C1": dict(grounded_verify=True,  session_memory=True,  shrinkage_enabled=True,  scorer_enabled=True),
    "C2": dict(grounded_verify=False, session_memory=True,  shrinkage_enabled=True,  scorer_enabled=True),
    "C3": dict(grounded_verify=True,  session_memory=False, shrinkage_enabled=True,  scorer_enabled=True),
    "C4": dict(grounded_verify=False, session_memory=False, shrinkage_enabled=False, scorer_enabled=False),
    "C5": dict(grounded_verify=True,  session_memory=True,  shrinkage_enabled=False, scorer_enabled=True),
    "C6": dict(grounded_verify=True,  session_memory=True,  shrinkage_enabled=True,  scorer_enabled=False),
}


def build_condition(
    condition: str,
    *,
    model: str,
    task_graph_kind: str,
    tools: list[Any] | None = None,
) -> HarnessAgent:
    flags = _FLAGS[condition]  # raises KeyError for unknown condition

    if task_graph_kind == "completion":
        task_graph = CompletionTaskGraph(grounded_verify=flags["grounded_verify"])
        invariants = InvariantSuite.completion()
    elif task_graph_kind == "tool_calling":
        task_graph = ToolCallingTaskGraph(tools=tools or [])
        invariants = InvariantSuite.tool_calling()
    elif task_graph_kind == "emit_only_tool_calling":
        task_graph = EmitOnlyToolCallingTaskGraph(tools=tools or [])
        invariants = InvariantSuite.emit_only_tool_calling()
    else:
        raise ValueError(
            f"unknown task_graph_kind: {task_graph_kind!r}; "
            "must be 'completion', 'tool_calling', or 'emit_only_tool_calling'"
        )

    return HarnessAgent(
        model=model,
        invariants=invariants,
        fallback_chain=[],
        task_graph=task_graph,
        grounded_verify=flags["grounded_verify"],
        session_memory=flags["session_memory"],
        shrinkage_enabled=flags["shrinkage_enabled"],
        scorer_enabled=flags["scorer_enabled"],
        llm_factory=get_benchmark_llm_factory(),
    )
