from __future__ import annotations

from benchmarks.gates.criteria import GateResult
from benchmarks.gates.report import render_gate_report


def test_report_renders_pass_header():
    results = [
        GateResult(name="G1_capability_floor", passed=True,
                   metrics={"raw_2b_pass_rate": 0.40, "raw_9b_pass_rate": 0.75, "gap_pp": 35.0},
                   failed_checks=[]),
    ]
    md = render_gate_report(results, run_date="2026-04-24")

    assert "# Gate Results — 2026-04-24" in md
    assert "G1_capability_floor" in md
    assert "PASS" in md
    assert "40.0%" in md
    assert "75.0%" in md


def test_report_renders_fail_with_failed_checks():
    results = [
        GateResult(name="G1_capability_floor", passed=False,
                   metrics={"raw_2b_pass_rate": 0.15, "raw_9b_pass_rate": 0.70, "gap_pp": 55.0},
                   failed_checks=["raw_2b < 30% (got 15.0%)"]),
    ]
    md = render_gate_report(results, run_date="2026-04-24")

    assert "FAIL" in md
    assert "raw_2b < 30%" in md


def test_report_includes_pivot_recommendation_on_fail():
    results = [
        GateResult(name="G2_harness_lift", passed=False,
                   metrics={"raw_2b": 0.40, "c1_2b": 0.45, "c4_2b": 0.42, "lift_ratio": 1.12},
                   failed_checks=["C1 < 1.5x raw (lift=1.12x)"]),
    ]
    md = render_gate_report(results, run_date="2026-04-24")

    assert "Pivot" in md or "pivot" in md
    assert "0.8B hero" in md or "reliability metric" in md
