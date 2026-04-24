from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.gsm8k.loader import load_tasks, _parse_final_answer


def test_parse_final_answer_canonical_integer():
    assert _parse_final_answer(
        "Janet's ducks lay 16 eggs per day... #### 18"
    ) == "18"


def test_parse_final_answer_with_commas_is_stripped():
    assert _parse_final_answer("... #### 1,200") == "1200"


def test_parse_final_answer_negative():
    assert _parse_final_answer("... #### -7") == "-7"


def test_parse_final_answer_missing_marker_returns_empty():
    assert _parse_final_answer("No marker here") == ""


def test_load_tasks_normalizes_shape(tmp_path: Path, monkeypatch):
    """Loader should read cached JSONL and emit {task_id, question, expected_answer}."""
    fake_cache = tmp_path / "gsm8k_test.jsonl"
    fake_cache.write_text(
        json.dumps({
            "question": "If Alice has 3 apples and buys 4 more, how many does she have?",
            "answer": "3 + 4 = 7\n#### 7",
        }) + "\n" +
        json.dumps({
            "question": "Bob has 10 marbles and loses half. How many remain?",
            "answer": "10 / 2 = 5\n#### 5",
        }) + "\n"
    )

    with patch("benchmarks.gsm8k.loader._ensure_cached", return_value=fake_cache):
        tasks = load_tasks(n=2)

    assert len(tasks) == 2
    assert tasks[0]["task_id"] == "gsm8k/0"
    assert tasks[0]["question"].startswith("If Alice")
    assert tasks[0]["expected_answer"] == "7"
    assert tasks[1]["task_id"] == "gsm8k/1"
    assert tasks[1]["expected_answer"] == "5"


def test_load_tasks_respects_n(tmp_path: Path):
    fake = tmp_path / "gsm8k_test.jsonl"
    rows = [
        json.dumps({"question": f"q{i}", "answer": f"#### {i}"})
        for i in range(10)
    ]
    fake.write_text("\n".join(rows) + "\n")

    with patch("benchmarks.gsm8k.loader._ensure_cached", return_value=fake):
        tasks = load_tasks(n=3)

    assert len(tasks) == 3
    assert tasks[-1]["task_id"] == "gsm8k/2"
    assert tasks[-1]["expected_answer"] == "2"


def test_load_tasks_handles_comma_separated_ground_truth(tmp_path: Path):
    fake = tmp_path / "gsm8k_test.jsonl"
    fake.write_text(json.dumps({
        "question": "Big number problem",
        "answer": "Lots of work... #### 1,250",
    }) + "\n")

    with patch("benchmarks.gsm8k.loader._ensure_cached", return_value=fake):
        tasks = load_tasks(n=1)

    assert tasks[0]["expected_answer"] == "1250"


def test_download_and_cache_invokes_datasets(tmp_path: Path, monkeypatch):
    """Smoke: _download_and_cache should call datasets.load_dataset and write JSONL."""
    import benchmarks.gsm8k.loader as loader_mod

    # Point cache at a tmp dir so we don't pollute the real cache.
    monkeypatch.setattr(loader_mod, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(loader_mod, "_CACHE_FILE", tmp_path / "gsm8k_test.jsonl")

    fake_rows = [
        {"question": "Q1", "answer": "thinking... #### 42"},
        {"question": "Q2", "answer": "thinking... #### 9"},
    ]

    def fake_load_dataset(repo, config, split):
        assert repo == "openai/gsm8k"
        assert config == "main"
        assert split == "test"
        return fake_rows

    import sys
    import types
    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    out = loader_mod._download_and_cache()
    assert out == tmp_path / "gsm8k_test.jsonl"
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["question"] == "Q1"
    assert "#### 42" in first["answer"]
