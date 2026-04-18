from __future__ import annotations
from typing import TYPE_CHECKING

from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from smboost.harness.state import HarnessState, StepOutput
from smboost.scorer import RobustnessScorer

if TYPE_CHECKING:
    from smboost.invariants.suite import InvariantSuite
    from smboost.tasks.base import TaskGraph


class HarnessGraph:
    def __init__(
        self,
        task_graph: TaskGraph,
        invariant_suite: InvariantSuite,
        max_retries: int = 3,
        scorer: RobustnessScorer | None = None,
    ):
        self._task_graph = task_graph
        self._suite = invariant_suite
        self._max_retries = max_retries
        self._scorer = scorer or RobustnessScorer()
        self._compiled = self._build()

    def _build(self):
        g = StateGraph(HarnessState)
        g.add_node("execute_step", self._execute_step)
        g.set_entry_point("execute_step")
        g.add_conditional_edges(
            "execute_step",
            self._route,
            {"execute_step": "execute_step", END: END},
        )
        return g.compile()

    def _execute_step(self, state: HarnessState) -> dict:
        retry_count = state["retry_count"]
        fallback_index = state["fallback_index"]
        current_model = state["model"]
        shrinkage_level = state["shrinkage_level"]

        if retry_count >= self._max_retries:
            next_fi = fallback_index + 1
            if next_fi >= len(state["fallback_chain"]):
                return {"status": "failed", "model": current_model}
            fallback_index = next_fi
            current_model = state["fallback_chain"][fallback_index]
            retry_count = 0
            shrinkage_level = 0

        node_name = self._task_graph.node_names[state["current_node_index"]]
        entry_invs, exit_invs = self._suite.node_invariants.get(node_name, ([], []))

        if not all(inv(state, None) for inv in entry_invs):
            return {
                "retry_count": retry_count + 1,
                "model": current_model,
                "fallback_index": fallback_index,
                "status": "running",
            }

        llm = ChatOllama(model=current_model)
        node_fn = self._task_graph.get_node_fn(node_name)
        try:
            output = node_fn(state, llm)
            node_exception = None
        except Exception as exc:
            output = str(exc)
            node_exception = str(exc)

        passed = node_exception is None and all(inv(state, output) for inv in exit_invs)

        if not passed:
            best_output, confidence = self._scorer.score(node_fn, state, llm)
            step = StepOutput(
                node=node_name,
                model=current_model,
                output=best_output,
                confidence=confidence,
                passed=False,
            )
            new_shrinkage = shrinkage_level + (1 if confidence < self._scorer.threshold else 0)

            if new_shrinkage > 3:
                next_fi = fallback_index + 1
                if next_fi >= len(state["fallback_chain"]):
                    return {
                        "step_outputs": state["step_outputs"] + [step],
                        "status": "failed",
                        "model": current_model,
                    }
                return {
                    "step_outputs": state["step_outputs"] + [step],
                    "retry_count": 0,
                    "model": state["fallback_chain"][next_fi],
                    "fallback_index": next_fi,
                    "shrinkage_level": 0,
                    "status": "running",
                }

            return {
                "step_outputs": state["step_outputs"] + [step],
                "retry_count": retry_count + 1,
                "model": current_model,
                "fallback_index": fallback_index,
                "shrinkage_level": new_shrinkage,
                "status": "running",
            }

        step = StepOutput(
            node=node_name,
            model=current_model,
            output=output,
            confidence=1.0,
            passed=True,
        )
        next_index = state["current_node_index"] + 1
        is_done = next_index >= len(self._task_graph.node_names)
        return {
            "step_outputs": state["step_outputs"] + [step],
            "retry_count": 0,
            "current_node_index": next_index,
            "final_output": output,
            "model": current_model,
            "fallback_index": fallback_index,
            "shrinkage_level": 0,
            "status": "success" if is_done else "running",
        }

    def _route(self, state: HarnessState) -> str:
        if state["status"] in ("success", "failed"):
            return END
        return "execute_step"

    def invoke(self, state: HarnessState) -> HarnessState:
        try:
            return self._compiled.invoke(state)
        except GraphRecursionError:
            return {**state, "status": "failed"}
