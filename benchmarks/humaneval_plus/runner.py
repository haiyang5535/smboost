"""HumanEval+ evaluator that emits both base HumanEval and HumanEval+ pass rates
from the same set of generated completions.

HumanEval+ (via `evalplus`) reuses the 164 HumanEval problems with ~80x more tests
per problem, so we score the same completion against both test suites.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HumanEvalPlusResult:
    pass_at_1_base: float
    pass_at_1_plus: float
    rows: list[dict[str, Any]]  # per-task: task_id, passed_heval, passed_heval_plus


def _evalplus_evaluate(sample_file: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Indirection so tests can patch. Returns evalplus's per-task result map.

    Real call shape:
        from evalplus.evaluate import evaluate
        return evaluate(dataset="humaneval", samples=sample_file, ...)

    evalplus returns { task_id: {"base": {"passed": bool, ...},
                                  "plus": {"passed": bool, ...}} }
    """
    from evalplus.evaluate import evaluate  # lazy import so unit tests can patch

    return evaluate(
        dataset="humaneval",
        samples=sample_file,
        parallel=1,
        i_just_wanna_run=True,
    )


def evaluate_dual(results: list[dict]) -> HumanEvalPlusResult:
    """Given a list of {task_id, completion} dicts, return pass@1 for both
    HumanEval (base) and HumanEval+ (plus), plus per-task rows.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in results:
            json.dump({"task_id": r["task_id"], "completion": r["completion"]}, f)
            f.write("\n")
        sample_file = f.name

    eval_map = _evalplus_evaluate(sample_file)

    rows: list[dict[str, Any]] = []
    passed_base = 0
    passed_plus = 0
    for r in results:
        tid = r["task_id"]
        entry = eval_map.get(tid, {"base": {"passed": False}, "plus": {"passed": False}})
        pb = 1 if entry["base"]["passed"] else 0
        pp = 1 if entry["plus"]["passed"] else 0
        passed_base += pb
        passed_plus += pp
        rows.append(
            {
                "task_id": tid,
                "completion": r["completion"],
                "passed_heval": pb,
                "passed_heval_plus": pp,
            }
        )

    n = max(len(results), 1)
    return HumanEvalPlusResult(
        pass_at_1_base=passed_base / n,
        pass_at_1_plus=passed_plus / n,
        rows=rows,
    )


def save_rows_csv(rows: list[dict], out_path: Path) -> None:
    """Write per-task rows to CSV with stable header."""
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["task_id", "passed_heval", "passed_heval_plus", "completion"]
        )
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "task_id": r["task_id"],
                    "passed_heval": r["passed_heval"],
                    "passed_heval_plus": r["passed_heval_plus"],
                    "completion": r.get("completion", ""),
                }
            )
