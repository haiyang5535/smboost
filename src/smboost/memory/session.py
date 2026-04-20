from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path


_LINE_NUMBERS = re.compile(r"line \d+")
_INT_LITERAL = re.compile(r"\b\d+\b")


def _normalize_line(line: str) -> str:
    s = _LINE_NUMBERS.sub("line N", line)
    s = _INT_LITERAL.sub("n", s)
    return s.strip()


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z_]+", text) if len(w) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class FailureRecord:
    task_id: str
    node: str
    attempt: int
    error_class: str
    error_line: str
    traceback_tail: str
    first_assertion: str
    prompt_used: str
    timestamp: float

    @property
    def signature(self) -> tuple[str, str, str]:
        return (self.error_class, _normalize_line(self.error_line), self.first_assertion)


class SessionMemory:
    def __init__(self, log_path: Path | None = None):
        self._records: list[FailureRecord] = []
        self._log_path = log_path
        self._log_fh = open(log_path, "w") if log_path else None
        self.hits = 0

    def record(
        self, *, task_id: str, node: str, attempt: int,
        error_class: str, error_line: str, traceback_tail: str,
        first_assertion: str, prompt_used: str,
    ) -> FailureRecord:
        rec = FailureRecord(
            task_id=task_id, node=node, attempt=attempt,
            error_class=error_class, error_line=error_line,
            traceback_tail=traceback_tail, first_assertion=first_assertion,
            prompt_used=prompt_used, timestamp=time.time(),
        )
        self._records.append(rec)
        if self._log_fh is not None:
            self._log_fh.write(json.dumps(asdict(rec)) + "\n")
            self._log_fh.flush()
        return rec

    def recent_for_task(self, task_id: str, limit: int = 2) -> list[FailureRecord]:
        same = [r for r in self._records if r.task_id == task_id]
        result = list(reversed(same))[:limit]
        if result:
            self.hits += 1
        return result

    def find_similar(
        self, *, task_id: str, prompt: str, error_class: str,
    ) -> FailureRecord | None:
        candidates = [r for r in self._records if r.task_id != task_id]
        sig_matches = [r for r in candidates if r.error_class == error_class]
        if sig_matches:
            self.hits += 1
            return sig_matches[-1]
        prompt_tokens = _tokenize(prompt)
        best: FailureRecord | None = None
        best_score = 0.4
        for r in candidates:
            score = _jaccard(prompt_tokens, _tokenize(r.prompt_used))
            if score > best_score:
                best = r
                best_score = score
        if best is not None:
            self.hits += 1
        return best

    def close(self) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
