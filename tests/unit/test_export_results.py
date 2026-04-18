from __future__ import annotations
import textwrap
from pathlib import Path
import pytest
from benchmarks.export_results import load_results, render_js


# ---------- load_results ----------

def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "results.csv"
    p.write_text(textwrap.dedent(content).lstrip())
    return p


def test_load_results_three_modes(tmp_path):
    csv = _write_csv(tmp_path, """
        model,mode,pass@1,avg_retries,avg_latency_s
        qwen3.5:2b,baseline,0.54,-,2.10
        qwen3.5:2b,smboost,0.74,1.2,3.80
        qwen3.5:2b,upper_bound,0.91,-,1.90
    """)
    rows = load_results(csv)
    assert len(rows) == 3
    assert rows[0] == {"mode": "baseline",    "passAt1": 0.54, "avgRetries": None, "avgLatency": 2.10}
    assert rows[1] == {"mode": "smboost",     "passAt1": 0.74, "avgRetries": 1.2,  "avgLatency": 3.80}
    assert rows[2] == {"mode": "upper_bound", "passAt1": 0.91, "avgRetries": None, "avgLatency": 1.90}


def test_load_results_uses_last_row_per_mode(tmp_path):
    # If baseline was run twice, use the most recent row
    csv = _write_csv(tmp_path, """
        model,mode,pass@1,avg_retries,avg_latency_s
        qwen3.5:2b,baseline,0.50,-,2.00
        qwen3.5:2b,baseline,0.60,-,1.80
    """)
    rows = load_results(csv)
    assert len(rows) == 1
    assert rows[0]["passAt1"] == 0.60


def test_load_results_missing_file():
    with pytest.raises(FileNotFoundError):
        load_results(Path("/nonexistent/results.csv"))


def test_load_results_empty_csv(tmp_path):
    csv = _write_csv(tmp_path, "model,mode,pass@1,avg_retries,avg_latency_s\n")
    with pytest.raises(ValueError, match="No results"):
        load_results(csv)


def test_load_results_single_mode(tmp_path):
    csv = _write_csv(tmp_path, """
        model,mode,pass@1,avg_retries,avg_latency_s
        qwen3.5:2b,baseline,0.54,-,2.10
    """)
    rows = load_results(csv)
    assert len(rows) == 1
    assert rows[0]["mode"] == "baseline"
    assert rows[0]["passAt1"] == 0.54


# ---------- render_js ----------

def test_render_js_structure():
    rows = [
        {"mode": "baseline",    "passAt1": 0.54, "avgRetries": None, "avgLatency": 2.10},
        {"mode": "smboost",     "passAt1": 0.74, "avgRetries": 1.2,  "avgLatency": 3.80},
    ]
    js = render_js(rows, model="qwen3.5:2b", n_tasks=50, generated_at="2026-04-18T12:00:00")
    assert "window.SMBOOST_BENCHMARK" in js
    assert '"qwen3.5:2b"' in js
    assert "50" in js
    assert "2026-04-18T12:00:00" in js
    assert '"baseline"' in js
    assert '"smboost"' in js
    assert "0.54" in js
    assert "null" in js        # avgRetries for baseline
    assert "1.2" in js


def test_render_js_is_valid_assignment():
    rows = [{"mode": "baseline", "passAt1": 0.54, "avgRetries": None, "avgLatency": 2.10}]
    js = render_js(rows, model="qwen3.5:2b", n_tasks=50, generated_at="2026-04-18T12:00:00")
    assert js.startswith("window.SMBOOST_BENCHMARK")
    assert js.rstrip().endswith(";")
