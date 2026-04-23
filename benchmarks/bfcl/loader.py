"""BFCL simple/multiple_function loader.

On first use, downloads the BFCL data file for the requested category from
the public gorilla-llm/berkeley-function-call-leaderboard repository and
caches it under `benchmarks/bfcl/data/<category>.jsonl`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BFCL_CATEGORIES = ("simple", "multiple_function")

_CACHE_DIR = Path("benchmarks/bfcl/data")

# BFCL upstream raw paths. Updated to v4 (repo moved to bfcl_eval/data/).
# "simple" maps to BFCL_v4_simple_python (Python subset; v4 split by language).
# "multiple_function" maps to BFCL_v4_multiple.
_BFCL_URLS: dict[str, str] = {
    "simple": (
        "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
        "berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json"
    ),
    "multiple_function": (
        "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
        "berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multiple.json"
    ),
}


def _ensure_cached(category: str) -> Path:
    """Download the BFCL data file for category if not already cached; return its path."""
    import requests  # lazy so tests can patch this function

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = _CACHE_DIR / f"{category}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        return out

    resp = requests.get(_BFCL_URLS[category], timeout=30)
    resp.raise_for_status()
    # BFCL ships as JSONL (one JSON object per line). Write verbatim.
    out.write_text(resp.text)
    return out


def _extract_question(raw_question: Any) -> str:
    """Normalise BFCL's question field to a plain string.

    v3 and test fixtures use a plain string.
    v4 encodes question as list[list[{role, content}]] — we extract the first
    user message's content.
    """
    if isinstance(raw_question, str):
        return raw_question
    if isinstance(raw_question, list) and raw_question:
        # v4: [[{"role": "user", "content": "..."}], ...]
        first = raw_question[0]
        if isinstance(first, list) and first:
            first = first[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(raw_question)


def _normalize(raw_row: dict[str, Any], category: str) -> dict[str, Any]:
    """Map BFCL's raw JSON object to our internal task shape."""
    return {
        "task_id": raw_row.get("id", raw_row.get("task_id", "unknown")),
        "category": category,
        "question": _extract_question(raw_row.get("question", "")),
        "functions": raw_row.get("function", raw_row.get("functions", [])),
        "ground_truth": raw_row.get("ground_truth"),  # may be None; eval handles it
        "_raw": raw_row,
    }


def load_bfcl_tasks(*, category: str, n: int) -> list[dict[str, Any]]:
    """Return the first n normalized BFCL tasks for the given category."""
    if category not in BFCL_CATEGORIES:
        raise ValueError(
            f"unknown category: {category!r}; must be one of {BFCL_CATEGORIES}"
        )
    data_file = _ensure_cached(category)

    tasks: list[dict[str, Any]] = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            tasks.append(_normalize(raw, category))
            if len(tasks) >= n:
                break
    return tasks
