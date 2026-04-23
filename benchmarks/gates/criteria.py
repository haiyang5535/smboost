"""Pure pass/fail evaluators for gates G1-G4. No I/O, no LLM calls.

Row shape (minimum): {"model": str, "mode": str, "passed": 0 | 1}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateResult:
    name: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    failed_checks: list[str] = field(default_factory=list)


def pass_rate(rows: list[dict], *, model: str, mode: str) -> float:
    matching = [r for r in rows if r.get("model") == model and r.get("mode") == mode]
    if not matching:
        return 0.0
    return sum(int(r.get("passed", 0)) for r in matching) / len(matching)


def evaluate_g1(rows: list[dict]) -> GateResult:
    """G1 — Capability floor.
    raw 2B >= 30% AND raw 9B >= 60% AND raw_9B - raw_2B >= 25pp.
    """
    r2 = pass_rate(rows, model="qwen3.5:2b", mode="raw")
    r9 = pass_rate(rows, model="qwen3.5:9b", mode="raw")
    metrics = {"raw_2b_pass_rate": r2, "raw_9b_pass_rate": r9, "gap_pp": (r9 - r2) * 100}

    failed: list[str] = []
    if r2 < 0.30:
        failed.append(f"raw_2b < 30% (got {r2:.1%})")
    if r9 < 0.60:
        failed.append(f"raw_9b < 60% (got {r9:.1%})")
    if (r9 - r2) < 0.25:
        failed.append(f"gap < 25pp (got {(r9 - r2) * 100:.1f}pp)")

    return GateResult(
        name="G1_capability_floor",
        passed=not failed,
        metrics=metrics,
        failed_checks=failed,
    )


def evaluate_g2(rows: list[dict]) -> GateResult:
    """G2 — Harness lift.
    C1 >= 1.5x raw 2B AND C1 - C4 >= 10pp (both on 2B).
    """
    raw = pass_rate(rows, model="qwen3.5:2b", mode="raw")
    c1 = pass_rate(rows, model="qwen3.5:2b", mode="C1")
    c4 = pass_rate(rows, model="qwen3.5:2b", mode="C4")
    lift = (c1 / raw) if raw > 0 else float("inf")
    metrics = {"raw_2b": raw, "c1_2b": c1, "c4_2b": c4, "lift_ratio": lift}

    failed: list[str] = []
    if raw > 0 and c1 < 1.5 * raw:
        failed.append(f"C1 < 1.5x raw (lift={lift:.2f}x)")
    if raw == 0 and c1 < 0.30:
        failed.append(f"raw=0 and C1 < 30% absolute ({c1:.1%})")
    if (c1 - c4) < 0.10:
        failed.append(f"C1 - C4 < 10pp (got {(c1 - c4) * 100:.1f}pp)")

    return GateResult(
        name="G2_harness_lift",
        passed=not failed,
        metrics=metrics,
        failed_checks=failed,
    )


def evaluate_g3(rows: list[dict]) -> GateResult:
    """G3 — BFCL sanity.
    raw 2B >= 20% AND C1 >= 1.5x raw 2B (both on BFCL simple).
    """
    raw = pass_rate(rows, model="qwen3.5:2b", mode="raw")
    c1 = pass_rate(rows, model="qwen3.5:2b", mode="C1")
    lift = (c1 / raw) if raw > 0 else float("inf")
    metrics = {"raw_2b_bfcl": raw, "c1_2b_bfcl": c1, "lift_ratio": lift}

    failed: list[str] = []
    if raw < 0.20:
        failed.append(f"raw_2b BFCL < 20% (got {raw:.1%})")
    if raw > 0 and c1 < 1.5 * raw:
        failed.append(f"C1 BFCL < 1.5x raw (lift={lift:.2f}x)")

    return GateResult(
        name="G3_bfcl_sanity",
        passed=not failed,
        metrics=metrics,
        failed_checks=failed,
    )


def evaluate_g4(rows: list[dict]) -> GateResult:
    """G4 — Widen confidence.
    On 50 tasks: G1 criteria hold AND C1 >= 1.5x raw 2B still holds.
    """
    # Reuse G1's capability checks
    g1 = evaluate_g1(rows)
    # And the harness-lift check on 2B
    raw = pass_rate(rows, model="qwen3.5:2b", mode="raw")
    c1 = pass_rate(rows, model="qwen3.5:2b", mode="C1")
    lift = (c1 / raw) if raw > 0 else float("inf")

    failed = list(g1.failed_checks)
    if raw > 0 and c1 < 1.5 * raw:
        failed.append(f"C1 @50 < 1.5x raw (lift={lift:.2f}x)")

    metrics = dict(g1.metrics)
    metrics.update({"c1_2b_at_50": c1, "lift_ratio_at_50": lift, "n": len(rows)})

    return GateResult(
        name="G4_widen_confidence",
        passed=not failed,
        metrics=metrics,
        failed_checks=failed,
    )
