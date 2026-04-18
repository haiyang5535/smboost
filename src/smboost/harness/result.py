from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from smboost.harness.state import StepOutput


@dataclass
class RunStats:
    retry_count: int
    fallback_triggers: int
    total_latency_s: float
    model_used: str


@dataclass
class HarnessResult:
    output: str
    trace: list[StepOutput]
    stats: RunStats
    status: Literal["success", "failed"]
