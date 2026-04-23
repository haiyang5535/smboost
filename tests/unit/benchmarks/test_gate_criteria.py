from __future__ import annotations

from benchmarks.gates.criteria import (
    pass_rate,
    evaluate_g1,
    evaluate_g2,
    evaluate_g3,
    evaluate_g4,
    GateResult,
)


def _rows(n_total: int, n_passed: int, model: str, mode: str):
    return (
        [{"model": model, "mode": mode, "passed": 1} for _ in range(n_passed)]
        + [{"model": model, "mode": mode, "passed": 0} for _ in range(n_total - n_passed)]
    )


def test_pass_rate_basic():
    rows = _rows(20, 7, "qwen3.5:2b", "raw")
    assert pass_rate(rows, model="qwen3.5:2b", mode="raw") == 7 / 20


def test_pass_rate_zero_when_no_matches():
    rows = _rows(20, 7, "qwen3.5:2b", "raw")
    assert pass_rate(rows, model="qwen3.5:9b", mode="raw") == 0.0


def test_g1_passes_when_criteria_met():
    rows = (
        _rows(20, 8, "qwen3.5:2b", "raw")      # 40% raw 2B  >= 30%
        + _rows(20, 15, "qwen3.5:9b", "raw")   # 75% raw 9B  >= 60%, gap 35pp >= 25pp
    )
    result = evaluate_g1(rows)
    assert result.passed is True
    assert "capability_floor" in result.name.lower()


def test_g1_fails_when_raw_2b_below_30pct():
    rows = (
        _rows(20, 4, "qwen3.5:2b", "raw")      # 20% raw 2B  < 30%
        + _rows(20, 15, "qwen3.5:9b", "raw")
    )
    result = evaluate_g1(rows)
    assert result.passed is False
    assert "raw_2b" in " ".join(result.failed_checks).lower()


def test_g1_fails_when_gap_too_small():
    rows = (
        _rows(20, 10, "qwen3.5:2b", "raw")     # 50% raw 2B  OK
        + _rows(20, 13, "qwen3.5:9b", "raw")   # 65% raw 9B  gap 15pp < 25pp
    )
    result = evaluate_g1(rows)
    assert result.passed is False


def test_g2_passes_when_c1_lifts_and_beats_c4():
    rows = (
        _rows(20, 8, "qwen3.5:2b", "raw")      # 40% raw
        + _rows(20, 14, "qwen3.5:2b", "C1")    # 70% C1 (1.75x raw)
        + _rows(20, 10, "qwen3.5:2b", "C4")    # 50% C4 (C1 - C4 = 20pp)
    )
    result = evaluate_g2(rows)
    assert result.passed is True


def test_g2_fails_when_c1_not_enough_lift():
    rows = (
        _rows(20, 10, "qwen3.5:2b", "raw")     # 50% raw
        + _rows(20, 12, "qwen3.5:2b", "C1")    # 60% C1 (only 1.2x raw)
        + _rows(20, 11, "qwen3.5:2b", "C4")
    )
    result = evaluate_g2(rows)
    assert result.passed is False


def test_g3_passes_on_bfcl_when_criteria_met():
    rows = (
        _rows(30, 9, "qwen3.5:2b", "raw")      # 30% raw BFCL (>= 20% threshold)
        + _rows(30, 18, "qwen3.5:2b", "C1")    # 60% C1 (2x raw, threshold 1.5x)
    )
    result = evaluate_g3(rows)
    assert result.passed is True


def test_g4_passes_when_g1_g2_criteria_still_hold_on_50():
    rows = (
        _rows(50, 20, "qwen3.5:2b", "raw")     # 40%
        + _rows(50, 35, "qwen3.5:2b", "C1")    # 70%
        + _rows(50, 38, "qwen3.5:9b", "raw")   # 76%
    )
    result = evaluate_g4(rows)
    assert result.passed is True


def test_gate_result_has_reporting_fields():
    rows = _rows(20, 5, "qwen3.5:2b", "raw") + _rows(20, 15, "qwen3.5:9b", "raw")
    result = evaluate_g1(rows)
    assert hasattr(result, "passed")
    assert hasattr(result, "metrics")
    assert hasattr(result, "failed_checks")
    assert "raw_2b_pass_rate" in result.metrics
    assert "raw_9b_pass_rate" in result.metrics
