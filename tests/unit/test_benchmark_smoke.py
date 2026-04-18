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

    with patch("benchmarks.run_humaneval.ChatOllama", mock_llm_cls):
        results = run_baseline(tasks, model="qwen3.5:2b")

    assert len(results) == 2
    assert all("task_id" in r and "completion" in r and "latency_s" in r for r in results)
    assert mock_llm_cls.return_value.invoke.call_count == 2
