import csv

from scripts.overnight_full_matrix import _avg_field, _dominant_failure_bucket, _pass_rate


def _write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition", "model", "seed", "task_id", "passed", "duration_ms",
                "retries", "fallback_triggered", "grounded_verify_result", "memory_hits",
                "usd_cost", "wall_clock_ms", "max_tokens", "failure_bucket",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_pass_rate_filters_by_budget(tmp_path):
    csv_path = tmp_path / "matrix.csv"
    _write_rows(
        csv_path,
        [
            {"condition": "C1", "model": "qwen3.5:2b", "seed": 0, "task_id": "a", "passed": 1, "duration_ms": 1000,
             "retries": 1, "fallback_triggered": 0, "grounded_verify_result": "PASS", "memory_hits": 0,
             "usd_cost": 0.0, "wall_clock_ms": 1100, "max_tokens": 64, "failure_bucket": "PASS"},
            {"condition": "C1", "model": "qwen3.5:2b", "seed": 0, "task_id": "b", "passed": 0, "duration_ms": 1200,
             "retries": 3, "fallback_triggered": 0, "grounded_verify_result": "FAIL", "memory_hits": 0,
             "usd_cost": 0.0, "wall_clock_ms": 1300, "max_tokens": 64, "failure_bucket": "syntax_truncation"},
            {"condition": "C1", "model": "qwen3.5:2b", "seed": 0, "task_id": "c", "passed": 1, "duration_ms": 900,
             "retries": 0, "fallback_triggered": 0, "grounded_verify_result": "PASS", "memory_hits": 0,
             "usd_cost": 0.0, "wall_clock_ms": 1000, "max_tokens": 160, "failure_bucket": "PASS"},
        ],
    )

    assert _pass_rate(csv_path, "C1", "qwen3.5:2b", 64) == (0.5, 2)
    assert _pass_rate(csv_path, "C1", "qwen3.5:2b", 160) == (1.0, 1)


def test_avg_field_filters_by_budget(tmp_path):
    csv_path = tmp_path / "matrix.csv"
    _write_rows(
        csv_path,
        [
            {"condition": "C1", "model": "qwen3.5:2b", "seed": 0, "task_id": "a", "passed": 1, "duration_ms": 1000,
             "retries": 1, "fallback_triggered": 0, "grounded_verify_result": "PASS", "memory_hits": 0,
             "usd_cost": 0.0, "wall_clock_ms": 1100, "max_tokens": 64, "failure_bucket": "PASS"},
            {"condition": "C1", "model": "qwen3.5:2b", "seed": 0, "task_id": "b", "passed": 0, "duration_ms": 3000,
             "retries": 3, "fallback_triggered": 0, "grounded_verify_result": "FAIL", "memory_hits": 0,
             "usd_cost": 0.0, "wall_clock_ms": 3100, "max_tokens": 160, "failure_bucket": "logic_or_output_mismatch"},
        ],
    )

    assert _avg_field(csv_path, "C1", "qwen3.5:2b", 64, "duration_ms") == 1000.0
    assert _avg_field(csv_path, "C1", "qwen3.5:2b", 160, "duration_ms") == 3000.0


def test_dominant_failure_bucket_filters_by_budget(tmp_path):
    csv_path = tmp_path / "matrix.csv"
    _write_rows(
        csv_path,
        [
            {"condition": "C1", "model": "qwen3.5:2b", "seed": 0, "task_id": "a", "passed": 0, "duration_ms": 1000,
             "retries": 1, "fallback_triggered": 0, "grounded_verify_result": "FAIL", "memory_hits": 0,
             "usd_cost": 0.0, "wall_clock_ms": 1100, "max_tokens": 64, "failure_bucket": "syntax_truncation"},
            {"condition": "C1", "model": "qwen3.5:2b", "seed": 0, "task_id": "b", "passed": 0, "duration_ms": 1200,
             "retries": 3, "fallback_triggered": 0, "grounded_verify_result": "FAIL", "memory_hits": 0,
             "usd_cost": 0.0, "wall_clock_ms": 1300, "max_tokens": 64, "failure_bucket": "syntax_truncation"},
            {"condition": "C1", "model": "qwen3.5:2b", "seed": 0, "task_id": "c", "passed": 0, "duration_ms": 2000,
             "retries": 3, "fallback_triggered": 0, "grounded_verify_result": "FAIL", "memory_hits": 0,
             "usd_cost": 0.0, "wall_clock_ms": 2100, "max_tokens": 160, "failure_bucket": "logic_or_output_mismatch"},
        ],
    )

    assert _dominant_failure_bucket(csv_path, "C1", "qwen3.5:2b", 64) == "syntax_truncation"
    assert _dominant_failure_bucket(csv_path, "C1", "qwen3.5:2b", 160) == "logic_or_output_mismatch"
