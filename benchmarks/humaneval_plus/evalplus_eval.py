"""Real HumanEval+ subset-tolerant scoring via evalplus.

This module wraps `evalplus.evaluate.evaluate(...)` so callers can submit an
arbitrary subset of the 164 HumanEval problems and get correct per-task
`passed_heval` / `passed_heval_plus` results back.

Two upstream issues are worked around here; see
`docs/superpowers/research/2026-04-23-evalplus-he-plus-fix.md` for the full
root-cause analysis.

1. `evalplus.evaluate.evaluate` hard-asserts that the caller submits a
   completion for every problem in the dataset (164 total). We pad the
   submission with `completion=""` rows for every task_id that isn't in the
   caller's subset, then score only the caller's real task_ids on the way
   out.

2. `evalplus.eval.utils.reliability_guard` calls
   `resource.setrlimit(RLIMIT_AS, ...)` inside each worker child process.
   On Python 3.13 (python.org installer) + macOS, that raises
   `ValueError: current limit exceeds maximum limit`, which cascades into
   every worker failing with a TIMEOUT. We monkey-patch
   `reliability_guard` to swallow those ValueError/OSError errors while
   preserving the rest of its hardening. For the patch to propagate into
   the workers, the `multiprocessing` start method must be `fork` (the
   default `spawn` does not inherit parent-side patches), so we force it
   at module import time.
"""
from __future__ import annotations

import json
import multiprocessing as _mp
import os
import tempfile
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Import-time side effects (must run before anything imports evalplus workers)
# ---------------------------------------------------------------------------

def _force_fork_start_method() -> None:
    """Switch multiprocessing to `fork` so worker children inherit our
    monkey-patch of `reliability_guard`. No-op on non-fork-capable platforms
    (Windows) and on already-set start methods that happen to be `fork`.
    """
    try:
        available = _mp.get_all_start_methods()
    except Exception:
        available = []
    if "fork" not in available:
        return  # Windows / locked-down environments: best effort, skip.
    try:
        _mp.set_start_method("fork", force=True)
    except RuntimeError:
        # Already set. Leave it; the running start method may still be fork.
        pass


_force_fork_start_method()


def _ensure_ssl_cert_file() -> None:
    """evalplus downloads HumanEvalPlus.jsonl.gz on first run via urllib.
    The python.org installer ships no system cert bundle; point urllib at
    certifi's bundle so HTTPS works on Python 3.13 + macOS.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except Exception:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


_ensure_ssl_cert_file()


def _patch_reliability_guard() -> None:
    """Wrap `evalplus.eval.utils.reliability_guard` so RLIMIT_AS / RLIMIT_DATA
    failures on macOS 3.13 are silently ignored. Must be applied before
    `evalplus.evaluate.evaluate(...)` forks workers; because we force `fork`
    above, children inherit the patched reference.
    """
    import evalplus.eval.utils as _eu
    import evalplus.eval as _ee

    # Avoid double-patching if this module is reloaded.
    if getattr(_eu.reliability_guard, "__smboost_patched__", False):
        return

    _original_guard = _eu.reliability_guard

    def _patched_guard(maximum_memory_bytes=None):
        import resource as _resource

        # Temporarily shadow setrlimit so RLIMIT_AS / RLIMIT_DATA failures
        # don't abort the rest of the hardening (faulthandler disable,
        # builtins removal, etc). Other rlimit calls inside reliability_guard
        # (e.g. RLIMIT_STACK) are already guarded off for Darwin upstream.
        _orig_setrlimit = _resource.setrlimit

        def _safe_setrlimit(resource_id, limits):
            try:
                return _orig_setrlimit(resource_id, limits)
            except (ValueError, OSError):
                # macOS 3.13 rejects any RLIMIT_AS/DATA change. Harmless: the
                # caller still got their pre-exec setup; we simply don't
                # enforce the VM cap. Accept the weaker sandbox on this
                # platform.
                return None

        _resource.setrlimit = _safe_setrlimit
        try:
            return _original_guard(maximum_memory_bytes=maximum_memory_bytes)
        finally:
            _resource.setrlimit = _orig_setrlimit

    _patched_guard.__smboost_patched__ = True  # type: ignore[attr-defined]
    _eu.reliability_guard = _patched_guard
    # evalplus.eval re-exports via `from .utils import reliability_guard` at
    # `evalplus/eval/__init__.py`, so we also patch the re-export.
    _ee.reliability_guard = _patched_guard


# Patch now so any code path that imports this module before calling
# `evaluate_subset` is already hardened.
_patch_reliability_guard()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class HumanEvalPlusResult:
    pass_at_1_base: float
    pass_at_1_plus: float
    rows: list[dict[str, Any]]  # per-task: task_id, passed_heval, passed_heval_plus


def _evalplus_evaluate(sample_file: str) -> dict[str, list[dict[str, Any]]]:
    """Invoke evalplus and return the parsed eval map.

    evalplus.evaluate.evaluate writes `<sample_file>_eval_results.json` and
    returns None (see evalplus/evaluate.py:158 + 351-353). We read that
    file and hand back only the per-task `eval` dict.

    Return shape:
        { task_id: [ {"base_status": "pass"|"fail",
                      "plus_status": "pass"|"fail",
                      "solution": ..., ...}, ... ] }

    Raises RuntimeError if evalplus finishes but the expected result file is
    missing — that indicates an upstream naming change we need to detect
    loudly rather than silently mis-score.
    """
    # Re-apply the patch defensively; if a test reloaded the module or monkey-
    # patched evalplus back to the original, this makes the evaluate call
    # safe regardless.
    _patch_reliability_guard()

    from evalplus.evaluate import evaluate  # lazy import so tests can patch

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
            f"Looked for {result_path!r} after evaluating samples at "
            f"{sample_file!r}. evalplus version: {version}. "
            f"This usually means an upstream naming change — inspect the "
            f"directory for the real output file and update evalplus_eval.py."
        )

    with open(result_path) as f:
        full = json.load(f)

    if "eval" not in full:
        raise RuntimeError(
            f"evalplus result file {result_path!r} is missing the 'eval' "
            f"key. Upstream shape may have changed; full top-level keys: "
            f"{list(full.keys())}"
        )

    return full["eval"]


def _load_problem_task_ids() -> list[str]:
    """Return the 164 HumanEval+ task_ids. Imported lazily so tests that mock
    `_evalplus_evaluate` don't need a working evalplus dataset cache.
    """
    from evalplus.data import get_human_eval_plus

    return list(get_human_eval_plus().keys())


def evaluate_subset(results: list[dict]) -> HumanEvalPlusResult:
    """Score an arbitrary subset of HumanEval tasks against HumanEval+ tests.

    `results` is a list of `{"task_id": str, "completion": str}` dicts. The
    subset is padded up to the full 164 tasks (with empty completions) so
    that evalplus's 164-task assertion holds; padded entries are then
    ignored when we assemble the return rows.

    Returns a HumanEvalPlusResult with exactly `len(results)` rows in the
    caller's submission order.
    """
    if not results:
        # Still pad and run so the caller's ask ("score an empty subset")
        # returns a valid zero-result. This path is exercised by the
        # test suite as a smoke test.
        problem_ids = _load_problem_task_ids()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for tid in problem_ids:
                json.dump({"task_id": tid, "completion": ""}, f)
                f.write("\n")
            sample_file = f.name
        _evalplus_evaluate(sample_file)
        return HumanEvalPlusResult(pass_at_1_base=0.0, pass_at_1_plus=0.0, rows=[])

    real_task_ids = {r["task_id"] for r in results}
    problem_ids = _load_problem_task_ids()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for r in results:
            json.dump(
                {"task_id": r["task_id"], "completion": r["completion"]}, f
            )
            f.write("\n")
        for tid in problem_ids:
            if tid in real_task_ids:
                continue
            json.dump({"task_id": tid, "completion": ""}, f)
            f.write("\n")
        sample_file = f.name

    eval_map = _evalplus_evaluate(sample_file)

    # Pair each caller entry with the matching completion_id in eval_map.
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
