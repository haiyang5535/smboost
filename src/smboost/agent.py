from __future__ import annotations
import time
from typing import TYPE_CHECKING

from smboost.harness.graph import HarnessGraph
from smboost.harness.result import HarnessResult, RunStats
from smboost.harness.state import HarnessState
from smboost.tasks.coding import CodingTaskGraph

if TYPE_CHECKING:
    from smboost.invariants.suite import InvariantSuite
    from smboost.tasks.base import TaskGraph


class HarnessAgent:
    def __init__(
        self,
        model: str,
        invariants: InvariantSuite,
        scorer: str = "adaptive",
        fallback_chain: list[str] | None = None,
        max_retries: int = 3,
        task_graph: TaskGraph | None = None,
    ):
        self.model = model
        self.scorer = scorer
        self.fallback_chain = fallback_chain or [model]
        self._harness = HarnessGraph(
            task_graph=task_graph or CodingTaskGraph(),
            invariant_suite=invariants,
            max_retries=max_retries,
        )

    def run(self, task: str) -> HarnessResult:
        initial: HarnessState = {
            "task": task,
            "model": self.model,
            "fallback_chain": self.fallback_chain,
            "step_outputs": [],
            "retry_count": 0,
            "fallback_index": 0,
            "current_node_index": 0,
            "status": "running",
            "final_output": None,
        }

        start = time.monotonic()
        final = self._harness.invoke(initial)
        elapsed = round(time.monotonic() - start, 3)

        steps = final["step_outputs"]

        return HarnessResult(
            output=final["final_output"] or "",
            trace=steps,
            stats=RunStats(
                retry_count=final["retry_count"],
                fallback_triggers=final["fallback_index"],
                total_latency_s=elapsed,
                model_used=final["model"],
            ),
            status=final["status"],  # type: ignore[arg-type]  # HarnessGraph only terminates at "success"/"failed"
        )
