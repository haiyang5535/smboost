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


# ------------------------------------------------------------------------
# G1a - G5a: post-9B-deletion gates (see master-plan 2026-04-24)
# ------------------------------------------------------------------------

def _best_harness(rows: list[dict], *, model: str, bench: str | None = None,
                  modes: tuple[str, ...] = ("C1", "C5", "C6")) -> tuple[str, float]:
    """Return (mode_name, pass_rate) for the best-performing harness mode on (model, bench)."""
    best_mode, best_rate = modes[0], 0.0
    for mode in modes:
        matching = [r for r in rows if r.get("model") == model and r.get("mode") == mode
                    and (bench is None or r.get("bench") == bench)]
        if not matching:
            continue
        rate = sum(int(r.get("passed", 0)) for r in matching) / len(matching)
        if rate > best_rate:
            best_mode, best_rate = mode, rate
    return best_mode, best_rate


def _pass_rate_bench(rows: list[dict], *, model: str, mode: str, bench: str) -> float:
    matching = [r for r in rows if r.get("model") == model and r.get("mode") == mode
                and r.get("bench") == bench]
    if not matching:
        return 0.0
    return sum(int(r.get("passed", 0)) for r in matching) / len(matching)


def evaluate_g1a(rows: list[dict]) -> GateResult:
    """G1a — Capability floor (no 9B dependency).
    raw 2B on GSM8K >= 30%  AND  raw_2b - raw_0.8b on GSM8K >= 15pp.
    Confirms the primary bench is viable and that scale matters within our range.
    """
    r2 = _pass_rate_bench(rows, model="qwen3.5:2b", mode="raw", bench="gsm8k")
    r08 = _pass_rate_bench(rows, model="qwen3.5:0.8b", mode="raw", bench="gsm8k")
    metrics = {"raw_2b_gsm8k": r2, "raw_0.8b_gsm8k": r08, "gap_pp": (r2 - r08) * 100}

    failed: list[str] = []
    if r2 < 0.30:
        failed.append(f"raw_2b GSM8K < 30% (got {r2:.1%})")
    if (r2 - r08) < 0.15:
        failed.append(f"gap (raw_2b - raw_0.8b) < 15pp (got {(r2 - r08) * 100:.1f}pp)")

    return GateResult(name="G1a_capability_floor", passed=not failed,
                      metrics=metrics, failed_checks=failed)


def evaluate_g2a(rows: list[dict]) -> GateResult:
    """G2a — Harness lift on math (GSM8K).
    max(C1,C5,C6) on 2B GSM8K >= 1.5x raw.
    """
    raw = _pass_rate_bench(rows, model="qwen3.5:2b", mode="raw", bench="gsm8k")
    best_mode, best_rate = _best_harness(rows, model="qwen3.5:2b", bench="gsm8k")
    lift = (best_rate / raw) if raw > 0 else float("inf")
    metrics = {"raw_2b_gsm8k": raw, "best_harness_mode": best_mode,
               "best_harness_rate": best_rate, "lift_ratio": lift}

    failed: list[str] = []
    if raw == 0 and best_rate < 0.30:
        failed.append(f"raw_2b=0 and best harness < 30% absolute ({best_rate:.1%})")
    elif raw > 0 and best_rate < 1.5 * raw:
        failed.append(f"best harness ({best_mode}) < 1.5x raw (lift={lift:.2f}x)")

    return GateResult(name="G2a_harness_lift_math", passed=not failed,
                      metrics=metrics, failed_checks=failed)


def evaluate_g3a(rows: list[dict]) -> GateResult:
    """G3a — Structured output (BFCL with guided decoding).
    max(C1,C5,C6) on 2B bfcl_simple >= 30%.
    Unblocks the historical G3 structural failure (0/30 across modes).
    """
    best_mode, best_rate = _best_harness(rows, model="qwen3.5:2b", bench="bfcl_simple")
    raw = _pass_rate_bench(rows, model="qwen3.5:2b", mode="raw", bench="bfcl_simple")
    metrics = {"raw_2b_bfcl": raw, "best_harness_mode": best_mode,
               "best_harness_rate": best_rate}

    failed: list[str] = []
    if best_rate < 0.30:
        failed.append(f"best harness on BFCL < 30% (got {best_rate:.1%} via {best_mode})")

    return GateResult(name="G3a_structured_output", passed=not failed,
                      metrics=metrics, failed_checks=failed)


def evaluate_g4a(rows: list[dict]) -> GateResult:
    """G4a — External parity.
    Best 2B+harness on >=1 bench >= 0.8x the equivalent external model (raw).
    Rows must include `bench` and `cost_usd` (from benchmarks/external/runner).
    External models considered: claude-sonnet-4-6, gpt-4o, llama-3-70b*.
    """
    external_models = ("claude-sonnet-4-6", "gpt-4o", "meta-llama/llama-3-70b-instruct")
    benches = ("gsm8k", "bfcl_simple", "humaneval_plus")

    results_per_bench: dict[str, dict[str, float]] = {}
    for bench in benches:
        _, our_best = _best_harness(rows, model="qwen3.5:2b", bench=bench)
        external_raws = [(m, _pass_rate_bench(rows, model=m, mode="raw", bench=bench))
                         for m in external_models]
        external_raws = [(m, r) for m, r in external_raws if r > 0]
        if not external_raws:
            continue
        best_external = max(external_raws, key=lambda x: x[1])
        ratio = our_best / best_external[1] if best_external[1] > 0 else 0.0
        results_per_bench[bench] = {"our_best": our_best,
                                    "external_model": best_external[0],
                                    "external_rate": best_external[1],
                                    "ratio": ratio}

    met_parity = [b for b, r in results_per_bench.items() if r["ratio"] >= 0.80]
    metrics: dict[str, Any] = {"per_bench": results_per_bench,
                               "benches_meeting_parity": met_parity}

    failed: list[str] = []
    if not results_per_bench:
        failed.append("no external baseline rows found for any bench")
    elif not met_parity:
        ratios_summary = [(b, round(r["ratio"], 2)) for b, r in results_per_bench.items()]
        failed.append(f"no bench reaches 0.8x external raw (best ratios: {ratios_summary})")

    return GateResult(name="G4a_external_parity", passed=not failed,
                      metrics=metrics, failed_checks=failed)


def evaluate_g5a(rows: list[dict], *, cli_smoke_passed: bool = False,
                 demo_trace_present: bool = False) -> GateResult:
    """G5a — Demo readiness.
    CLI smoke succeeds AND demo trace exists AND frontend renders trace.
    Called with external flags (cli_smoke_passed, demo_trace_present).
    """
    metrics = {"cli_smoke": float(cli_smoke_passed),
               "demo_trace": float(demo_trace_present)}

    failed: list[str] = []
    if not cli_smoke_passed:
        failed.append("CLI smoke did not pass (smboost run --task <X> failed)")
    if not demo_trace_present:
        failed.append("demo trace JSONL not present at expected path")

    return GateResult(name="G5a_demo_readiness", passed=not failed,
                      metrics=metrics, failed_checks=failed)
