from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.bfcl.loader import load_bfcl_tasks, BFCL_CATEGORIES


def test_load_bfcl_tasks_returns_normalized_shape(tmp_path: Path):
    # Seed a fake cached data file with BFCL's known JSONL-ish shape
    fake_simple = tmp_path / "simple.jsonl"
    fake_simple.write_text(
        json.dumps({
            "id": "simple_0",
            "question": "Get the weather in SF",
            "function": [{"name": "get_weather", "parameters": {"type": "object",
                                                                 "properties": {"city": {"type": "string"}}}}],
        }) + "\n" +
        json.dumps({
            "id": "simple_1",
            "question": "What's 2+2?",
            "function": [{"name": "calc", "parameters": {"type": "object",
                                                          "properties": {"expr": {"type": "string"}}}}],
        }) + "\n"
    )

    with patch("benchmarks.bfcl.loader._ensure_cached", return_value=fake_simple):
        tasks = load_bfcl_tasks(category="simple", n=2)

    assert len(tasks) == 2
    t0 = tasks[0]
    # Normalized shape keys we agree on
    assert set(t0.keys()) >= {"task_id", "category", "question", "functions"}
    assert t0["task_id"] == "simple_0"
    assert t0["category"] == "simple"
    assert t0["question"] == "Get the weather in SF"
    assert isinstance(t0["functions"], list)
    assert t0["functions"][0]["name"] == "get_weather"


def test_bfcl_categories_constant():
    assert "simple" in BFCL_CATEGORIES
    assert "multiple_function" in BFCL_CATEGORIES


def test_load_bfcl_tasks_rejects_unknown_category():
    import pytest
    with pytest.raises(ValueError, match="unknown category"):
        load_bfcl_tasks(category="not_real", n=1)


def test_load_bfcl_tasks_respects_n(tmp_path: Path):
    fake = tmp_path / "simple.jsonl"
    rows = [
        json.dumps({"id": f"simple_{i}", "question": f"q{i}",
                    "function": [{"name": "f", "parameters": {"type": "object", "properties": {}}}]})
        for i in range(10)
    ]
    fake.write_text("\n".join(rows) + "\n")

    with patch("benchmarks.bfcl.loader._ensure_cached", return_value=fake):
        tasks = load_bfcl_tasks(category="simple", n=3)

    assert len(tasks) == 3
    assert tasks[-1]["task_id"] == "simple_2"
