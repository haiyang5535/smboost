from unittest.mock import patch, MagicMock


def _fake_problems():
    return {
        "HumanEval/0": {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n    ",
            "test": "assert add(1, 2) == 3",
            "entry_point": "add",
        },
        "HumanEval/1": {
            "task_id": "HumanEval/1",
            "prompt": "def sub(a, b):\n    ",
            "test": "assert sub(3, 1) == 2",
            "entry_point": "sub",
        },
    }


def test_load_humaneval_tasks_returns_list_of_dicts():
    with patch("benchmarks.tasks.read_problems", return_value=_fake_problems()):
        from benchmarks.tasks import load_humaneval_tasks
        tasks = load_humaneval_tasks(n=1)

    assert len(tasks) == 1
    assert "task_id" in tasks[0]
    assert "prompt" in tasks[0]


def test_load_humaneval_tasks_returns_all_when_n_is_none():
    with patch("benchmarks.tasks.read_problems", return_value=_fake_problems()):
        from benchmarks.tasks import load_humaneval_tasks
        tasks = load_humaneval_tasks()

    assert len(tasks) == 2


def test_baseline_runner_calls_llm_per_task():
    with patch("benchmarks.tasks.read_problems", return_value=_fake_problems()):
        from benchmarks.tasks import load_humaneval_tasks
        tasks = load_humaneval_tasks(n=2)

    from benchmarks.run_humaneval import run_baseline
    mock_llm_cls = MagicMock()
    mock_llm_cls.return_value.invoke.return_value = MagicMock(content="    return a + b")

    with patch("benchmarks.run_humaneval.ChatOpenAI", mock_llm_cls):
        results = run_baseline(tasks, model="qwen3.5:2b")

    assert len(results) == 2
    assert all("task_id" in r and "completion" in r and "latency_s" in r for r in results)
    assert mock_llm_cls.return_value.invoke.call_count == 2


# ---------- clean_completion ----------

def test_clean_completion_strips_think_block_preserving_indent():
    from benchmarks.run_humaneval import clean_completion
    raw = "<think>\nreasoning here\n</think>\n\n    return a + b"
    assert clean_completion(raw) == "    return a + b"


def test_clean_completion_strips_multiple_think_blocks():
    from benchmarks.run_humaneval import clean_completion
    raw = "<think>first</think>\n<think>second</think>\nreturn 1"
    assert clean_completion(raw) == "return 1"


def test_clean_completion_unwraps_python_code_fence():
    from benchmarks.run_humaneval import clean_completion
    raw = "```python\ndef f(x):\n    return x + 1\n```"
    assert "def f(x):" in clean_completion(raw)
    assert "```" not in clean_completion(raw)


def test_clean_completion_unwraps_bare_fence():
    from benchmarks.run_humaneval import clean_completion
    raw = "```\nreturn 42\n```"
    assert clean_completion(raw).strip() == "return 42"


def test_clean_completion_think_then_fenced_code():
    from benchmarks.run_humaneval import clean_completion
    raw = "<think>planning</think>\n```python\nreturn a + b\n```"
    assert clean_completion(raw).strip() == "return a + b"


def test_clean_completion_passthrough_when_clean():
    from benchmarks.run_humaneval import clean_completion
    raw = "    return a + b"
    assert clean_completion(raw) == "    return a + b"


def test_clean_completion_handles_empty_string():
    from benchmarks.run_humaneval import clean_completion
    assert clean_completion("") == ""
