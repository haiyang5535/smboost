from __future__ import annotations

from human_eval.data import read_problems


def load_humaneval_tasks(n: int | None = None) -> list[dict]:
    """Load HumanEval problems. Returns list of problem dicts with task_id and prompt."""
    problems = list(read_problems().values())
    if n is not None:
        problems = problems[:n]
    return problems
