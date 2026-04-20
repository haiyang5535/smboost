from __future__ import annotations
from typing import Callable

from smboost import HarnessAgent, InvariantSuite
from smboost.tasks.completion import CompletionTaskGraph


def _build(model: str, seed: int, *, grounded: bool, memory: bool,
           shrinkage: bool, scorer: bool) -> HarnessAgent:
    return HarnessAgent(
        model=model,
        invariants=InvariantSuite.completion(),
        fallback_chain=[],  # no cross-model fallback; keep each cell pure
        task_graph=CompletionTaskGraph(grounded_verify=grounded),
        grounded_verify=grounded,
        session_memory=memory,
        shrinkage_enabled=shrinkage,
        scorer_enabled=scorer,
    )


def _c1(model: str, seed: int) -> HarnessAgent:
    return _build(model, seed, grounded=True, memory=True, shrinkage=True, scorer=True)


def _c2(model: str, seed: int) -> HarnessAgent:
    return _build(model, seed, grounded=False, memory=True, shrinkage=True, scorer=True)


def _c3(model: str, seed: int) -> HarnessAgent:
    return _build(model, seed, grounded=True, memory=False, shrinkage=True, scorer=True)


def _c4(model: str, seed: int) -> HarnessAgent:
    return _build(model, seed, grounded=False, memory=False, shrinkage=False, scorer=False)


CONDITIONS: dict[str, Callable[[str, int], HarnessAgent]] = {
    "C1": _c1,
    "C2": _c2,
    "C3": _c3,
    "C4": _c4,
}
