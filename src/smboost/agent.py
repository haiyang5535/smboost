from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Callable
from typing import TYPE_CHECKING

from smboost.harness.graph import HarnessGraph
from smboost.harness.result import HarnessResult, RunStats
from smboost.harness.state import HarnessState, StepOutput
from smboost.harness.trace_schema import (
    TraceStepInput,
    TraceStepOutput,
    TraceStepRecord,
    TraceStepVerify,
)
from smboost.llm.runtime import get_default_llm_factory
from smboost.memory.session import SessionMemory
from smboost.scorer import RobustnessScorer
from smboost.tasks.coding import CodingTaskGraph

if TYPE_CHECKING:
    from smboost.invariants.suite import InvariantSuite
    from smboost.tasks.base import TaskGraph


_MAX_STR_LEN = 2000
_TRUNC_MARKER = "...[truncated]"


def _truncate(s: str | None) -> str | None:
    if s is None:
        return None
    if len(s) <= _MAX_STR_LEN:
        return s
    return s[:_MAX_STR_LEN] + _TRUNC_MARKER


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
        trace_log_path: Path | str | None = None,
    ):
        self.model = model
        self.scorer = scorer
        self.fallback_chain = fallback_chain or [model]
        self.grounded_verify = grounded_verify
        self.session_memory = session_memory
        self.shrinkage_enabled = shrinkage_enabled
        self.scorer_enabled = scorer_enabled
        self.trace_log_path = Path(trace_log_path) if trace_log_path is not None else None
        self._memory = SessionMemory() if session_memory else None
        # When no explicit factory is provided, fall back to the env-var-aware
        # default. This respects SMBOOST_OPENAI_BASE_URL / _API_KEY / _MAX_TOKENS
        # (and SMBOOST_LLM_BACKEND=local) instead of hard-coding localhost:8000.
        effective_factory = llm_factory if llm_factory is not None else get_default_llm_factory()
        self._harness = HarnessGraph(
            task_graph=task_graph or CodingTaskGraph(),
            invariant_suite=invariants,
            max_retries=max_retries,
            scorer=RobustnessScorer(threshold=scorer_threshold) if scorer_enabled else None,
            shrinkage_enabled=shrinkage_enabled,
            llm_factory=effective_factory,
        )

    def run(
        self,
        task: str,
        task_metadata: dict | None = None,
        task_id: str | None = None,
        condition: str | None = None,
    ) -> HarnessResult:
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

        run_id = uuid.uuid4().hex
        wall_start = time.monotonic()

        # Open the trace file (append mode) for the duration of this run, if configured.
        trace_fh = None
        if self.trace_log_path is not None:
            self.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
            trace_fh = open(self.trace_log_path, "a", encoding="utf-8")
            self._emit_run_start(trace_fh, run_id, task_id, condition)

        final: HarnessState | None = None
        exc_raised: BaseException | None = None
        start = time.monotonic()
        token = comp_mod._ACTIVE_MEMORY.set(self._memory if self.session_memory else None)
        try:
            try:
                final = self._harness.invoke(initial)
            except BaseException as exc:  # re-raised in outer finally via exc_raised
                exc_raised = exc
                raise
        finally:
            comp_mod._ACTIVE_MEMORY.reset(token)
            elapsed = round(time.monotonic() - start, 3)
            if trace_fh is not None:
                try:
                    steps = final["step_outputs"] if final is not None else []
                    for idx, step in enumerate(steps):
                        self._emit_step(trace_fh, run_id, task_id, condition, idx, step, final)
                    passed = final is not None and final.get("status") == "success"
                    retries = final["retry_count"] if final is not None else 0
                    final_code = final.get("final_output") if final is not None else None
                    wall_ms = int((time.monotonic() - wall_start) * 1000)
                    self._emit_summary(trace_fh, run_id, passed, retries, wall_ms, final_code)
                finally:
                    trace_fh.close()

        assert final is not None  # if we got here without exc, invoke returned
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

    @staticmethod
    def _write_json_line(fh, obj: dict) -> None:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str))
        fh.write("\n")
        fh.flush()

    def _emit_run_start(
        self,
        fh,
        run_id: str,
        task_id: str | None,
        condition: str | None,
    ) -> None:
        self._write_json_line(fh, {
            "schema_version": 1,
            "run_id": run_id,
            "task_id": task_id,
            "model": self.model,
            "condition": condition,
            "event": "run_start",
        })

    def _emit_step(
        self,
        fh,
        run_id: str,
        task_id: str | None,
        condition: str | None,
        step_idx: int,
        step: StepOutput,
        final_state: HarnessState | None,
    ) -> None:
        now_ts = time.time()
        record = TraceStepRecord(
            run_id=run_id,
            task_id=task_id,
            model=step.model if getattr(step, "model", None) else self.model,
            condition=condition,
            step_idx=step_idx,
            node=step.node,
            entry_ts=now_ts,
            exit_ts=now_ts,
            retry_count=final_state["retry_count"] if final_state is not None else None,
            shrinkage_level=final_state["shrinkage_level"] if final_state is not None else None,
            scorer_confidence=step.confidence,
            input=TraceStepInput(prompt=None, budget=None),
            output=TraceStepOutput(
                code=_truncate(step.output),
                trunc=bool(step.output) and len(step.output) > _MAX_STR_LEN,
            ),
            verify=TraceStepVerify(kind=None, passed=step.passed, traceback=None),
            fallback_triggered=(
                final_state is not None and final_state.get("fallback_index", 0) > 0
            ),
        )
        self._write_json_line(fh, record.to_json_dict())

    @staticmethod
    def _emit_summary(
        fh,
        run_id: str,
        passed: bool,
        retries: int,
        wall_ms: int,
        final_code: str | None,
    ) -> None:
        HarnessAgent._write_json_line(fh, {
            "run_id": run_id,
            "event": "summary",
            "passed": bool(passed),
            "retries": int(retries),
            "wall_ms": int(wall_ms),
            "final_code": _truncate(final_code) if final_code is not None else None,
        })

    def set_memory_log(self, log_path) -> None:
        if self._memory is not None and log_path is not None:
            from pathlib import Path
            self._memory._log_fh = open(Path(log_path), "w")
            self._memory._log_path = Path(log_path)

    def close_memory(self) -> None:
        if self._memory is not None:
            self._memory.close()
