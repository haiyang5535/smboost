"""HumanEval+ evaluator that emits both base HumanEval and HumanEval+ pass rates
from the same set of generated completions.

HumanEval+ (via `evalplus`) reuses the 164 HumanEval problems with ~80x more tests
per problem, so we score the same completion against both test suites.

NOTE: `evaluate_dual` here assumes the caller has submitted completions for
ALL 164 HumanEval problems. For subset submissions (e.g. the G1/G2 gates
that only score N=20 tasks), use `benchmarks.humaneval_plus.evalplus_eval.
evaluate_subset` — it handles the 164-task padding upstream evalplus
requires. See `docs/superpowers/research/2026-04-23-evalplus-he-plus-fix.md`.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HumanEvalPlusResult:
    pass_at_1_base: float
    pass_at_1_plus: float
    rows: list[dict[str, Any]]  # per-task: task_id, passed_heval, passed_heval_plus


def _evalplus_evaluate(sample_file: str) -> dict[str, list[dict[str, Any]]]:
    """Indirection so tests can patch. Returns evalplus's per-task result map.

    evalplus's `evaluate(...)` returns None (not the result map) — the actual
    per-task results are written to `<sample_file>_eval_results.json`. Shape:

        { task_id: [ {"base_status": "pass"|"fail",
                      "plus_status": "pass"|"fail",
                      "solution": ..., ...}, ... ] }

    Each task_id maps to a list (one dict per attempt/completion_id).
    """
    from evalplus.evaluate import evaluate  # lazy import so unit tests can patch

    evaluate(
        dataset="humaneval",
        samples=sample_file,
        parallel=1,
        i_just_wanna_run=True,
    )

    result_path = sample_file.replace(".jsonl", "_eval_results.json")
    if not os.path.isfile(result_path):
        try:
            import evalplus  # noqa: F401

            version = getattr(evalplus, "__version__", "unknown")
        except Exception:
            version = "unknown"
        raise RuntimeError(
            f"evalplus did not produce the expected eval-results file. "
            f"Looked for {result_path!r}. evalplus version: {version}."
        )

    with open(result_path) as f:
        full = json.load(f)
    return full.get("eval", {})


def evaluate_dual(results: list[dict]) -> HumanEvalPlusResult:
    """Given a list of {task_id, completion} dicts, return pass@1 for both
    HumanEval (base) and HumanEval+ (plus), plus per-task rows.

    This function assumes the caller covers all 164 HumanEval problems (or
    has monkey-patched `_evalplus_evaluate` for a test). For subset
    submissions use `evalplus_eval.evaluate_subset`.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in results:
            json.dump({"task_id": r["task_id"], "completion": r["completion"]}, f)
            f.write("\n")
        sample_file = f.name

    eval_map = _evalplus_evaluate(sample_file)

    # Pair each caller entry with the matching completion_id in eval_map[tid].
    # Preserves multi-sample ordering: if the same task_id shows up twice in
    # `results`, we read entries[0] then entries[1].
    per_tid_seen: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    passed_base = 0
    passed_plus = 0
    for r in results:
        tid = r["task_id"]
        idx = per_tid_seen.get(tid, 0)
        per_tid_seen[tid] = idx + 1

        entries = eval_map.get(tid, [])
        if idx < len(entries):
            entry = entries[idx]
        else:
            entry = {"base_status": "fail", "plus_status": "fail"}
        pb = 1 if entry.get("base_status") == "pass" else 0
        pp = 1 if entry.get("plus_status") == "pass" else 0
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
