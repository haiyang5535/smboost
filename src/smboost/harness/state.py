from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, TypedDict


@dataclass
class StepOutput:
    node: str
    model: str
    output: str
    confidence: float
    passed: bool


class HarnessState(TypedDict):
    task: str
    task_metadata: dict
    model: str
    fallback_chain: list[str]
    step_outputs: list[StepOutput]
    retry_count: int
    fallback_index: int
    current_node_index: int
    shrinkage_level: int
    status: Literal["running", "success", "failed"]
    final_output: str | None
