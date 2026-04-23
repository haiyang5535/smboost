from __future__ import annotations

import pytest

from benchmarks.conditions import build_condition, CONDITION_NAMES


def test_all_six_condition_names_registered():
    assert CONDITION_NAMES == ("C1", "C2", "C3", "C4", "C5", "C6")


def test_c1_full_has_all_four_flags_on():
    agent = build_condition("C1", model="qwen3.5:2b", task_graph_kind="completion")
    assert agent.grounded_verify is True
    assert agent.session_memory is True
    assert agent.shrinkage_enabled is True
    assert agent.scorer_enabled is True


def test_c4_plain_has_all_four_flags_off():
    agent = build_condition("C4", model="qwen3.5:2b", task_graph_kind="completion")
    assert agent.grounded_verify is False
    assert agent.session_memory is False
    assert agent.shrinkage_enabled is False
    assert agent.scorer_enabled is False


def test_c5_minus_shrinkage_keeps_others_on():
    agent = build_condition("C5", model="qwen3.5:2b", task_graph_kind="completion")
    assert agent.shrinkage_enabled is False
    assert agent.grounded_verify is True
    assert agent.session_memory is True
    assert agent.scorer_enabled is True


def test_c6_minus_scorer_keeps_others_on():
    agent = build_condition("C6", model="qwen3.5:2b", task_graph_kind="completion")
    assert agent.scorer_enabled is False
    assert agent.grounded_verify is True
    assert agent.session_memory is True
    assert agent.shrinkage_enabled is True


def test_tool_calling_kind_wires_tool_calling_task_graph():
    agent = build_condition("C1", model="qwen3.5:2b", task_graph_kind="tool_calling", tools=[])
    # Task graph is stored on the inner HarnessGraph as ._task_graph
    from smboost.tasks.tool_calling import ToolCallingTaskGraph
    assert isinstance(agent._harness._task_graph, ToolCallingTaskGraph)


def test_unknown_condition_raises():
    with pytest.raises(KeyError):
        build_condition("C99", model="qwen3.5:2b", task_graph_kind="completion")


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="task_graph_kind"):
        build_condition("C1", model="qwen3.5:2b", task_graph_kind="not_a_kind")
