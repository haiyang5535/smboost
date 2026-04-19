from __future__ import annotations
import textwrap
from pathlib import Path
import pytest
from benchmarks.export_results import load_results, render_js


def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "results.csv"
    p.write_text(textwrap.dedent(content).lstrip())
    return p


# ---------- load_results ----------

def test_load_results_single_model_three_modes(tmp_path):
    csv = _write_csv(tmp_path, """
        model,mode,pass@1,avg_retries,avg_latency_s
        qwen3.5:0.8b,baseline,0.30,-,1.50
        qwen3.5:0.8b,smboost,0.50,1.2,2.10
        qwen3.5:0.8b,upper_bound,0.65,-,3.50
    """)
    groups = load_results(csv)
    assert len(groups) == 1
    assert groups[0]["model"] == "qwen3.5:0.8b"
    rows = groups[0]["rows"]
    assert len(rows) == 3
    assert rows[0] == {"mode": "baseline",    "passAt1": 0.30, "avgRetries": None, "avgLatency": 1.50}
    assert rows[1] == {"mode": "smboost",     "passAt1": 0.50, "avgRetries": 1.2,  "avgLatency": 2.10}
    assert rows[2] == {"mode": "upper_bound", "passAt1": 0.65, "avgRetries": None, "avgLatency": 3.50}


def test_load_results_multi_model(tmp_path):
    csv = _write_csv(tmp_path, """
        model,mode,pass@1,avg_retries,avg_latency_s
        qwen3.5:0.8b,baseline,0.30,-,1.50
        qwen3.5:0.8b,smboost,0.50,1.2,2.10
        qwen3.5:4b,baseline,0.52,-,2.30
        qwen3.5:4b,smboost,0.72,1.1,3.80
    """)
    groups = load_results(csv)
    assert len(groups) == 2
    assert groups[0]["model"] == "qwen3.5:0.8b"
    assert groups[1]["model"] == "qwen3.5:4b"
    assert len(groups[0]["rows"]) == 2
    assert len(groups[1]["rows"]) == 2
    assert groups[1]["rows"][0]["passAt1"] == 0.52


def test_load_results_uses_last_row_per_mode(tmp_path):
    csv = _write_csv(tmp_path, """
        model,mode,pass@1,avg_retries,avg_latency_s
        qwen3.5:0.8b,baseline,0.30,-,1.50
        qwen3.5:0.8b,baseline,0.35,-,1.40
    """)
    groups = load_results(csv)
    assert len(groups) == 1
    assert groups[0]["rows"][0]["passAt1"] == 0.35


def test_load_results_missing_file():
    with pytest.raises(FileNotFoundError):
        load_results(Path("/nonexistent/results.csv"))


def test_load_results_empty_csv(tmp_path):
    csv = _write_csv(tmp_path, "model,mode,pass@1,avg_retries,avg_latency_s\n")
    with pytest.raises(ValueError, match="No results"):
        load_results(csv)


# ---------- render_js ----------

def test_render_js_structure():
    groups = [
        {
            "model": "qwen3.5:0.8b",
            "rows": [
                {"mode": "baseline", "passAt1": 0.30, "avgRetries": None, "avgLatency": 1.5},
                {"mode": "smboost",  "passAt1": 0.50, "avgRetries": 1.2,  "avgLatency": 2.1},
            ],
        }
    ]
    js = render_js(groups, n_tasks=50, generated_at="2026-04-18T12:00:00")
    assert "window.SMBOOST_BENCHMARK" in js
    assert '"models"' in js
    assert "qwen3.5:0.8b" in js
    assert "50" in js
    assert "2026-04-18T12:00:00" in js
    assert '"baseline"' in js
    assert "null" in js
    assert "1.2" in js


def test_render_js_is_valid_assignment():
    groups = [{"model": "qwen3.5:0.8b", "rows": [
        {"mode": "baseline", "passAt1": 0.30, "avgRetries": None, "avgLatency": 1.5}
    ]}]
    js = render_js(groups, n_tasks=50, generated_at="2026-04-18T12:00:00")
    assert js.startswith("window.SMBOOST_BENCHMARK")
    assert js.rstrip().endswith(";")


# ---------- export_results ----------

def test_export_results_writes_file(tmp_path):
    csv_path = _write_csv(tmp_path, """
        model,mode,pass@1,avg_retries,avg_latency_s
        qwen3.5:0.8b,baseline,0.30,-,1.50
    """)
    out_path = tmp_path / "out" / "benchmark_data.js"
    from benchmarks.export_results import export_results
    export_results(csv_path, out_path, n_tasks=50)
    content = out_path.read_text()
    assert "window.SMBOOST_BENCHMARK" in content
    assert "qwen3.5:0.8b" in content
    assert '"models"' in content
    assert out_path.exists()
