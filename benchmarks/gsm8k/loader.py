"""GSM8K task loader.

Downloads the ``openai/gsm8k`` (``main`` config) **test** split from
HuggingFace Datasets on first call, caches as
``benchmarks/gsm8k/data/gsm8k_test.jsonl`` (gitignored), and exposes
``load_tasks(n)`` returning our normalized task shape:

    {"task_id": "gsm8k/<idx>", "question": "<str>", "expected_answer": "<int-as-str>"}

The expected_answer is the canonical final numeric answer parsed from the
dataset's "#### <N>" marker on the ``answer`` field.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_CACHE_DIR = Path("benchmarks/gsm8k/data")
_CACHE_FILE = _CACHE_DIR / "gsm8k_test.jsonl"

# GSM8K canonically places the final answer after a "#### " marker at the
# tail of the explanation. Negatives allowed; commas allowed (e.g. "#### 1,200").
_FINAL_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _parse_final_answer(answer_field: str) -> str:
    """Extract the canonical numeric final answer from a GSM8K 'answer' string.

    Strips commas (GSM8K frequently writes "1,200") so ``load_tasks`` stores
    an atomic integer-like string. Returns the empty string if no marker
    found (shouldn't happen on the curated test split but guarded anyway).
    """
    m = _FINAL_ANSWER_RE.search(answer_field or "")
    if not m:
        return ""
    return m.group(1).replace(",", "").strip()


def _download_and_cache() -> Path:
    """Fetch ``openai/gsm8k`` (main/test), write JSONL cache, return path."""
    from datasets import load_dataset  # lazy import; not every env has datasets

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    with open(_CACHE_FILE, "w") as f:
        for row in ds:
            f.write(
                json.dumps(
                    {
                        "question": row["question"],
                        "answer": row["answer"],
                    }
                )
                + "\n"
            )
    return _CACHE_FILE


def _ensure_cached() -> Path:
    """Return cache path, downloading once if missing/empty."""
    if _CACHE_FILE.exists() and _CACHE_FILE.stat().st_size > 0:
        return _CACHE_FILE
    return _download_and_cache()


def load_tasks(n: int = 200) -> list[dict[str, Any]]:
    """Return the first ``n`` normalized GSM8K test tasks.

    Parameters
    ----------
    n : int
        Max number of tasks to return. Default 200 matches the master-plan
        GSM8K primary sample size.

    Returns
    -------
    list[dict]
        Each item: ``{"task_id", "question", "expected_answer"}``.
    """
    cache = _ensure_cached()
    tasks: list[dict[str, Any]] = []
    with open(cache) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            tasks.append(
                {
                    "task_id": f"gsm8k/{idx}",
                    "question": raw["question"],
                    "expected_answer": _parse_final_answer(raw.get("answer", "")),
                }
            )
            if len(tasks) >= n:
                break
    return tasks
