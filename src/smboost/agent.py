from __future__ import annotations
import time
from typing import Callable
from typing import TYPE_CHECKING

from smboost.harness.graph import HarnessGraph
from smboost.harness.result import HarnessResult, RunStats
from smboost.harness.state import HarnessState
from smboost.memory.session import SessionMemory
from smboost.scorer import RobustnessScorer
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
        scorer_threshold: float = 0.6,
        grounded_verify: bool = True,
        session_memory: bool = True,
        shrinkage_enabled: bool = True,
        scorer_enabled: bool = True,
        llm_factory: Callable[[str], object] | None = None,
    ):
        self.model = model
        self.scorer = scorer
        self.fallback_chain = fallback_chain or [model]
        self.grounded_verify = grounded_verify
        self.session_memory = session_memory
        self.shrinkage_enabled = shrinkage_enabled
        self.scorer_enabled = scorer_enabled
        self._memory = SessionMemory() if session_memory else None
        self._harness = HarnessGraph(
            task_graph=task_graph or CodingTaskGraph(),
            invariant_suite=invariants,
            max_retries=max_retries,
            scorer=RobustnessScorer(threshold=scorer_threshold) if scorer_enabled else None,
            shrinkage_enabled=shrinkage_enabled,
            llm_factory=llm_factory,
        )

    def run(self, task: str, task_metadata: dict | None = None) -> HarnessResult:
        from smboost.tasks import completion as comp_mod

        initial: HarnessState = {
            "task": task,
            "task_metadata": task_metadata or {},
            "model": self.model,
            "fallback_chain": self.fallback_chain,
            "step_outputs": [],
            "retry_count": 0,
            "fallback_index": 0,
            "current_node_index": 0,
            "shrinkage_level": 0,
            "status": "running",
            "final_output": None,
        }

        start = time.monotonic()
        token = comp_mod._ACTIVE_MEMORY.set(self._memory if self.session_memory else None)
        try:
            final = self._harness.invoke(initial)
        finally:
            comp_mod._ACTIVE_MEMORY.reset(token)
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
            status=final["status"],  # type: ignore[arg-type]
        )

    def set_memory_log(self, log_path) -> None:
        if self._memory is not None and log_path is not None:
            from pathlib import Path
            self._memory._log_fh = open(Path(log_path), "w")
            self._memory._log_path = Path(log_path)

    def close_memory(self) -> None:
        if self._memory is not None:
            self._memory.close()
