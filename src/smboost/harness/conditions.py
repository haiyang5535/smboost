"""Task-graph factory for the harness conditions (C1..C6).

This module is the *graph-level* condition registry. It returns a bare
``TaskGraph`` — not a full ``HarnessAgent``. Callers that want a full agent
should use ``benchmarks.conditions.build_condition`` instead, which wraps
these graphs in the ablation flag matrix.

Responsibilities are deliberately narrow:

    build_condition("C6") -> RealToolTaskGraph(...)

Why a second factory?
    - ``benchmarks/conditions.py`` is the *benchmark-facing* factory: it returns
      a fully-configured ``HarnessAgent`` with an invariant suite and feature
      flags (``grounded_verify``, ``session_memory``, etc.). That's the right
      shape for the GSM8K / HumanEval+ runners.
    - ``src/smboost/harness/conditions.py`` (this file) is the *harness-facing*
      factory: a thin dispatcher that maps a condition id to the actual
      ``TaskGraph`` subclass. It is what demo scripts, notebooks, and new
      experimental runners want when they already have their own invariant
      suite + flag setup and just need the graph.

Merge note (Agent 2 + Agent 4)
------------------------------
Two parallel branches add conditions here:

    - Agent 2: C5 -> SelfConsistencyTaskGraph
    - Agent 4: C6 -> RealToolTaskGraph   (this file's initial content)

To keep merges trivial:
    * The condition dispatch is a flat if/elif chain in ``build_condition``.
    * Each condition has its own try-import at the top so the other agent's
      module absence does not break imports here.
    * Both agents add their branch in a clearly-commented block.
"""
from __future__ import annotations

from typing import Any

from smboost.tasks.base import TaskGraph
from smboost.tasks.completion import CompletionTaskGraph
from smboost.tasks.emit_only_tool_calling import EmitOnlyToolCallingTaskGraph

# --- Agent 4 (C6): RealToolTaskGraph ----------------------------------------
# Import is unconditional because this module ships with Agent 4's branch.
from smboost.harness.real_tool_graph import RealToolTaskGraph

# --- Agent 2 (C5): SelfConsistencyTaskGraph ---------------------------------
# Soft import: Agent 2's module may not yet be merged when this file is loaded
# on Agent 4's branch in isolation.
try:
    from smboost.harness.self_consistency_graph import (  # type: ignore[import-not-found]
        SelfConsistencyTaskGraph,
    )
    _HAS_C5 = True
except Exception:  # pragma: no cover - import-time environment variation
    SelfConsistencyTaskGraph = None  # type: ignore[assignment,misc]
    _HAS_C5 = False


CONDITION_NAMES: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6")


def build_condition(condition: str, **kwargs: Any) -> TaskGraph:
    """Return the ``TaskGraph`` for ``condition``.

    Keyword arguments are forwarded to the underlying graph constructor where
    applicable — e.g. ``tools=[...]`` for emit-only tool calling, or
    ``n_iterations=3`` for the real-tool graph.

    Unknown kwargs are silently ignored for conditions that don't consume them,
    to keep this dispatcher permissive.
    """
    if condition in {"C1", "C2", "C3", "C4"}:
        # C1..C4 are completion-style today. Feature-flag ablation
        # (grounded_verify / scorer / etc.) lives in benchmarks/conditions.py —
        # here we just hand back the graph.
        grounded = kwargs.get("grounded_verify", condition in {"C1", "C3"})
        return CompletionTaskGraph(grounded_verify=grounded)

    # --- Agent 2 branch (C5) ------------------------------------------------
    if condition == "C5":
        if not _HAS_C5 or SelfConsistencyTaskGraph is None:
            # Soft-miss: let the caller discover on their own branch that C5
            # hasn't been merged yet. Don't hard-fail in Agent 4's isolation.
            raise NotImplementedError(
                "C5 (SelfConsistencyTaskGraph) is not available in this build. "
                "It is authored by the parallel Agent 2 workstream; ensure that "
                "worktree is merged before invoking build_condition('C5')."
            )
        allowed = {"n_samples", "verifier", "verifier_kind", "base_task_graph"}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        # Agent 2's SelfConsistencyTaskGraph requires verifier=; if the caller
        # didn't supply one, default to run_tests_verifier (code-tests path).
        # Benchmark-level dispatch (benchmarks/conditions.py) picks a smarter
        # verifier based on task_graph_kind; this src-level fallback is for
        # bare build_condition("C5") calls (tests, notebooks, smoke scripts).
        if "verifier" not in filtered:
            from smboost.harness.self_consistency_graph import run_tests_verifier
            filtered["verifier"] = run_tests_verifier
        return SelfConsistencyTaskGraph(**filtered)  # type: ignore[misc]

    # --- Agent 4 branch (C6) ------------------------------------------------
    if condition == "C6":
        allowed = {"sandbox", "memory", "n_iterations", "sandbox_timeout_s"}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return RealToolTaskGraph(**filtered)

    # --- compatibility escape hatch -----------------------------------------
    # Allow callers to request emit-only tool calling via a non-standard id so
    # tests and BFCL demos don't need to import from tasks.* directly.
    if condition in {"emit_only_tool_calling", "bfcl"}:
        tools = kwargs.get("tools") or []
        return EmitOnlyToolCallingTaskGraph(tools=tools)

    raise ValueError(
        f"Unknown condition {condition!r}. Valid: {CONDITION_NAMES}"
    )


__all__ = ["CONDITION_NAMES", "build_condition"]
