"""Subprocess-based HumanEval evaluator, subset-tolerant.

Used as a fallback for environments where evalplus's multiprocessing sandbox
or human_eval's `Manager()` can't run — observed on Python 3.13 + macOS
Python.org installer, where `setrlimit(RLIMIT_AS, ...)` in a subprocess
either raises `ValueError` or crashes the multiprocessing Manager.

This evaluator runs each (prompt + completion + test + check call) as a
plain `subprocess.run(...)` with a `preexec_fn` that best-effort applies
rlimits in the child. If a limit can't be set, we continue without it
(same pattern the existing LCB sandbox uses).

Only scores the BASE HumanEval tests. HumanEval+ stricter tests are NOT
evaluated here — callers that want the +plus scores must use the evalplus
path (or pad all 164 tasks to work around evalplus's all-or-nothing mode).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmarks.humaneval_plus.runner import HumanEvalPlusResult


_TIMEOUT_SEC = 10


def _set_limits() -> None:
    """Best-effort rlimit setter for the subprocess child (post-fork, pre-exec)."""
    try:
        import resource
    except ImportError:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_TIMEOUT_SEC, _TIMEOUT_SEC))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(
            resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024)
        )
    except (ValueError, OSError):
        pass


def _build_script(problem: dict, completion: str) -> str:
    return (
        problem["prompt"]
        + completion
        + "\n\n"
        + problem["test"]
        + "\n\n"
        + f"check({problem['entry_point']})\n"
    )


def _check_one(problem: dict, completion: str) -> bool:
    script = _build_script(problem, completion)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=_TIMEOUT_SEC + 2,
            preexec_fn=_set_limits if sys.platform != "win32" else None,
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    return proc.returncode == 0


def evaluate_base_subset(results: list[dict]) -> HumanEvalPlusResult:
    """Score a subset of HumanEval completions against BASE HumanEval tests.

    Returns HumanEvalPlusResult with `pass_at_1_plus` mirrored from base —
    the +plus tests are NOT evaluated here, so callers should treat the two
    as the same number.
    """
    from human_eval.data import read_problems

    problems = read_problems()

    rows: list[dict] = []
    passed_base = 0
    for r in results:
        tid = r["task_id"]
        completion = r["completion"]
        problem = problems.get(tid)
        passed = _check_one(problem, completion) if problem is not None else False
        pb = 1 if passed else 0
        passed_base += pb
        rows.append(
            {
                "task_id": tid,
                "completion": completion,
                "passed_heval": pb,
                "passed_heval_plus": pb,
            }
        )

    n = max(len(results), 1)
    rate = passed_base / n
    return HumanEvalPlusResult(
        pass_at_1_base=rate,
        pass_at_1_plus=rate,
        rows=rows,
    )
