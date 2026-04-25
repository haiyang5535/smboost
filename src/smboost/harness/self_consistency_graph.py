"""Self-Consistency task graph (C5).

Implements the Prove / "Not All Votes Count" pattern (ArXiv 2410.12608) plus
classic self-consistency (Wang et al. 2022). Instead of a single sequential
attempt, the generator samples ``n_samples`` completions in parallel, runs a
program verifier on each, groups the candidates by ``extracted_answer`` among
those the verifier marked valid, and returns the majority-supported answer.

Published deltas on GSM8K with this pattern:
    0.5B : 48.85% -> 53.83%
    1.0B : 65.66% -> 73.01%
    2.0B : 73.39% -> 79.61%

Protocol
--------
The class obeys the project's :class:`smboost.tasks.base.TaskGraph` protocol
(synchronous, LangGraph-compatible):

* :pyattr:`node_names` -> ``["generate", "verify"]``
  (matches :class:`~smboost.invariants.suite.InvariantSuite.completion` so the
  graph drops into the existing :class:`~smboost.harness.graph.HarnessGraph`
  without surgery).
* :meth:`get_node_fn("generate")` performs N-sample voting internally and
  returns the majority-winning completion string (i.e. the "final answer" string
  a normal generate node would have emitted).
* :meth:`get_node_fn("verify")` returns ``"PASS"`` iff the final voted
  completion is marked valid by the verifier; otherwise ``"FAIL: <detail>"``.

Verifiers
---------
A verifier is a ``Callable[[str, dict], VerifierResult]`` where
``VerifierResult`` is a plain ``dict`` with keys ``valid`` (bool) and
``extracted_answer`` (hashable-ish; used for majority grouping). Three
built-in verifiers are exposed at module level:

* :func:`run_tests_verifier`         - code tasks (HumanEval / LCB)
* :func:`execute_program_verifier`   - GSM8K-style; runs an extracted Python
  program and extracts the last printed number
* :func:`tool_call_valid_verifier`   - BFCL-style structural JSON validation

All-invalid fallback
--------------------
If every sample is marked invalid, we fall back to returning the **first**
sample unchanged (and let the outer harness's shrinkage / retry machinery
handle recovery). This matches the "silent pass-through" contract the other
task graphs use so the HarnessGraph retry loop still gets a signal.

Parallelism note
----------------
We intentionally use a :class:`~concurrent.futures.ThreadPoolExecutor` rather
than :mod:`asyncio`. The existing LLM runtime (``_CompatibleChatOpenAI``) is
synchronous; the ``TaskGraph`` protocol is synchronous; LangGraph executes
nodes synchronously. Threads keep the I/O (HTTP call to llama.cpp) concurrent
without forcing an ``async def`` rewrite across the harness.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from langchain_core.messages import HumanMessage

from smboost.tasks.base import TaskGraph

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from smboost.harness.state import HarnessState


# Public type alias ---------------------------------------------------------
VerifierResult = dict[str, Any]
"""Shape: ``{"valid": bool, "extracted_answer": Any, "trace": str (optional)}``.

``extracted_answer`` must be hashable (str/int/float/tuple) so it can be the
key in the majority Counter. Unhashable types (list/dict) are coerced to a
JSON string for grouping purposes.
"""

Verifier = Callable[[str, dict], VerifierResult]


# ---------------------------------------------------------------------------
# Built-in verifiers
# ---------------------------------------------------------------------------


_FENCED_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_UNCLOSED_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)$", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>\n*", re.DOTALL)


def _strip_to_code(raw: str) -> str:
    """Extract a Python-looking block from a chat completion.

    Mirrors the stripping logic in :mod:`smboost.tasks.completion` (closed
    ``<think>`` blocks, fenced code blocks, unclosed fences). Kept here so
    this module has no cross-import into ``completion`` (verifier helpers
    should be self-contained).
    """
    cleaned = _THINK_BLOCK.sub("", raw)
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    fenced = _FENCED_CODE.search(cleaned)
    if fenced:
        return fenced.group(1)
    unclosed = _UNCLOSED_FENCE.search(cleaned)
    if unclosed:
        return unclosed.group(1)
    return cleaned


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _run_python_subprocess(src: str, *, timeout: int = 8) -> dict:
    """Run ``src`` in a subprocess; return ``{ok, stdout, stderr}``.

    Kept local (doesn't import from :mod:`smboost.tasks.completion`) so this
    module remains ownership-clean per the agent's file-ownership contract.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        tmp = Path(f.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp)],
            input="",
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"TimeoutError: {timeout}s"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    finally:
        tmp.unlink(missing_ok=True)


def run_tests_verifier(completion: str, meta: dict) -> VerifierResult:
    """Verifier for code tasks.

    Requires one of:
      * ``meta["entry_point"]`` + ``meta["test_cases"]`` (functional shape)
      * ``meta["prompt"]`` + ``meta["test"]`` + ``meta["entry_point"]`` (HumanEval shape)

    Returns ``{"valid": bool, "extracted_answer": <normalized_code>}``.
    The ``extracted_answer`` is the cleaned completion source so that
    *syntactically identical* passing solutions are grouped together for the
    majority vote.

    This is a lightweight, self-contained runner — it does NOT substitute for
    :func:`smboost.tasks.completion._verify_grounded`. It only needs to be
    strong enough to distinguish "passes tests" from "doesn't".
    """
    code = _strip_to_code(completion or "")
    if not code.strip():
        return {"valid": False, "extracted_answer": "", "trace": "empty completion"}

    entry_point = meta.get("entry_point", "")
    test_cases = meta.get("test_cases") or []
    he_prompt = meta.get("prompt", "")
    he_test = meta.get("test", "")

    if entry_point and he_prompt and he_test:
        # HumanEval shape
        src = (
            he_prompt
            + code
            + "\n\n"
            + he_test
            + "\n\n"
            + f"check({entry_point})\n"
        )
    elif entry_point and test_cases:
        # Functional shape: assemble `assert fn(args) == expected` lines
        asserts = []
        for tc in test_cases:
            inp = tc.get("input")
            out = tc.get("output")
            if inp is None:
                continue
            # best-effort: if input is a list, splat as args; else a single positional
            if isinstance(inp, list):
                args_src = ", ".join(repr(a) for a in inp)
            else:
                args_src = repr(inp)
            asserts.append(f"assert {entry_point}({args_src}) == {out!r}")
        src = (
            "from typing import *\n"
            "import collections, functools, itertools, math\n\n"
            + code
            + "\n\n"
            + "\n".join(asserts)
            + "\n"
        )
    else:
        return {
            "valid": False,
            "extracted_answer": code.strip(),
            "trace": "run_tests_verifier: meta missing entry_point/test_cases or prompt/test",
        }

    result = _run_python_subprocess(src)
    return {
        "valid": bool(result["ok"]),
        "extracted_answer": code.strip(),
        "trace": result["stderr"][-400:] if not result["ok"] else "",
    }


_GSM8K_FINAL_ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def _coerce_numeric(s: str) -> int | float | None:
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v.is_integer() else v


def execute_program_verifier(completion: str, meta: dict) -> VerifierResult:
    """Verifier for GSM8K-style math tasks.

    Two-tier extraction:
    1. If the completion contains a Python code block, try running it and
       parsing the last number on stdout. This is the Prove pattern from
       ArXiv 2410.12608 — running code as the verifier.
    2. Otherwise, look for the canonical GSM8K final-answer marker
       "#### <number>" in the prose. This works for plain CoT prompts
       ("Let's think step by step. ... #### 42").

    Either path produces a numeric extracted_answer used for majority vote.
    ``meta`` is unused; signature kept uniform across verifiers.

    Returns ``{"valid": bool, "extracted_answer": int|float|None}``.
    """
    _ = meta
    completion = completion or ""
    code = _strip_to_code(completion)

    # Tier 1: program execution (when the model emitted Python code)
    if code.strip():
        result = _run_python_subprocess(code)
        if result["ok"]:
            stdout = result["stdout"].strip()
            if stdout:
                matches = _NUMBER_RE.findall(stdout)
                if matches:
                    answer = _coerce_numeric(matches[-1])
                    if answer is not None:
                        return {"valid": True, "extracted_answer": answer,
                                "trace": "program-exec"}

    # Tier 2: extract "#### <N>" marker from prose (CoT format)
    final = _GSM8K_FINAL_ANSWER_RE.findall(completion)
    if final:
        # Last marker wins (handles few-shot contamination)
        answer = _coerce_numeric(final[-1])
        if answer is not None:
            return {"valid": True, "extracted_answer": answer,
                    "trace": "marker-extracted"}

    return {"valid": False, "extracted_answer": None,
            "trace": "no extractable numeric answer"}


def tool_call_valid_verifier(completion: str, meta: dict) -> VerifierResult:
    """Verifier for BFCL-style emit-only tool calls.

    PASS iff completion parses as a JSON object with:
      * a ``name`` field (string, matching one of ``meta["tools"]`` if present), AND
      * an ``arguments`` field (dict, possibly empty).

    ``extracted_answer`` is ``(name, tuple(sorted(arguments.items())))`` —
    orderless but hashable, so two calls with the same args grouped together.
    """
    if not completion or not completion.strip():
        return {"valid": False, "extracted_answer": None, "trace": "empty completion"}

    # Accept a JSON blob possibly wrapped in a fenced block or chat prose by
    # scanning for the first `{` .. matching `}`. Cheap heuristic: find the
    # first balanced object; if that fails, fall back to whole-string parse.
    text = completion.strip()
    # Strip code fences like ```json\n ... \n```
    text = re.sub(r"^```(?:json)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)

    try:
        obj: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "extracted_answer": None,
            "trace": f"JSONDecodeError: {exc.msg}",
        }

    if not isinstance(obj, dict):
        return {
            "valid": False,
            "extracted_answer": None,
            "trace": f"expected JSON object, got {type(obj).__name__}",
        }

    name = obj.get("name")
    if not isinstance(name, str):
        return {
            "valid": False,
            "extracted_answer": None,
            "trace": "missing or non-string 'name'",
        }

    args = obj.get("arguments")
    if not isinstance(args, dict):
        return {
            "valid": False,
            "extracted_answer": (name, None),
            "trace": "'arguments' must be a dict",
        }

    # Optional signature match against declared tool schemas in meta["tools"].
    # Accept both flat and wrapped shapes (see emit_only_tool_calling).
    tool_names: list[str] = []
    for t in (meta.get("tools") or []):
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            n = t["function"].get("name")
        else:
            n = t.get("name")
        if isinstance(n, str):
            tool_names.append(n)
    if tool_names and name not in tool_names:
        return {
            "valid": False,
            "extracted_answer": (name, _hashable_args(args)),
            "trace": f"function {name!r} not in declared tools {tool_names}",
        }

    return {
        "valid": True,
        "extracted_answer": (name, _hashable_args(args)),
        "trace": "",
    }


def _hashable_args(args: dict) -> tuple:
    """Return a hashable, order-insensitive representation of arguments."""
    try:
        return tuple(sorted(
            (k, json.dumps(v, sort_keys=True, default=str))
            for k, v in args.items()
        ))
    except Exception:  # pragma: no cover - json.dumps with default=str should not fail
        return tuple(sorted(args.items()))


# ---------------------------------------------------------------------------
# Majority-vote helper
# ---------------------------------------------------------------------------


def majority_vote(
    samples: list[str],
    verifier_results: list[VerifierResult],
) -> tuple[str, Any, int, int]:
    """Pick the winning sample by majority vote among verifier-valid results.

    Returns ``(winning_sample, winning_answer, valid_count, vote_count)``.

    * ``valid_count`` - how many samples the verifier marked valid.
    * ``vote_count``  - how many valid samples agree on the winner.

    Tie-break: earliest-first (stable). If zero samples are valid, return
    the first sample unchanged with ``valid_count=0, vote_count=0``.
    """
    if not samples:
        return "", None, 0, 0
    if len(samples) != len(verifier_results):
        raise ValueError(
            f"length mismatch: {len(samples)} samples vs "
            f"{len(verifier_results)} verifier results"
        )

    valid_indices = [i for i, r in enumerate(verifier_results) if r.get("valid")]
    if not valid_indices:
        return samples[0], None, 0, 0

    # Group valid samples by hashable extracted_answer
    def _key(ans: Any) -> Any:
        try:
            hash(ans)
            return ans
        except TypeError:
            try:
                return json.dumps(ans, sort_keys=True, default=str)
            except Exception:
                return repr(ans)

    buckets: dict[Any, list[int]] = {}
    for i in valid_indices:
        k = _key(verifier_results[i].get("extracted_answer"))
        buckets.setdefault(k, []).append(i)

    # Sort buckets by size desc, then earliest-index (stable tie-break)
    best_key = max(
        buckets.keys(),
        key=lambda k: (len(buckets[k]), -min(buckets[k])),
    )
    best_indices = buckets[best_key]
    winning_idx = best_indices[0]

    return (
        samples[winning_idx],
        verifier_results[winning_idx].get("extracted_answer"),
        len(valid_indices),
        len(best_indices),
    )


# ---------------------------------------------------------------------------
# Task graph
# ---------------------------------------------------------------------------


class SelfConsistencyTaskGraph(TaskGraph):
    """C5: parallel-sample + program-verifier + majority-vote task graph.

    Parameters
    ----------
    verifier:
        Callable taking ``(completion, meta)`` and returning a
        :data:`VerifierResult` dict. Required. Choose one of the built-in
        verifiers above or supply your own.
    n_samples:
        Number of parallel samples (default ``5``).
    sampling_temperature:
        Passed as a kwarg to ``llm.invoke(...)`` per sample. The underlying
        langchain ``ChatOpenAI.invoke`` ignores unknown kwargs on older
        versions, so this is a best-effort diversity knob; for guaranteed
        diversity, configure the factory's temperature instead.
    prompt_builder:
        Optional ``Callable[[HarnessState, int], str]`` that, given the state
        and the sample index (0..n-1), returns the prompt for that sample.
        Defaults to a simple passthrough of ``state["task"]``.
    """

    # Node names chosen to match InvariantSuite.completion() so this drops
    # into the existing HarnessGraph + build_condition pipeline for
    # task_graph_kind="completion" without needing a bespoke invariant suite.
    _NODE_NAMES = ["generate", "verify"]

    def __init__(
        self,
        *,
        verifier: Verifier,
        n_samples: int = 5,
        sampling_temperature: float = 0.7,
        prompt_builder: Callable[["HarnessState", int], str] | None = None,
    ):
        if verifier is None:
            raise ValueError("SelfConsistencyTaskGraph requires a verifier")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        self._verifier = verifier
        self._n_samples = int(n_samples)
        self._sampling_temperature = float(sampling_temperature)
        self._prompt_builder = prompt_builder or self._default_prompt_builder
        # Book-keeping for verify node: filled in by generate node each call.
        # The verify node reads this to decide PASS/FAIL without re-running
        # the (potentially expensive) verifier.
        self._last_verifier_valid: bool | None = None
        self._last_trace: str = ""

    # --- TaskGraph protocol -------------------------------------------------

    @property
    def node_names(self) -> list[str]:
        return list(self._NODE_NAMES)

    def get_node_fn(self, node_name: str):
        if node_name == "generate":
            return self._generate_node
        if node_name == "verify":
            return self._verify_node
        raise ValueError(
            f"Unknown node {node_name!r}. Valid names: {self._NODE_NAMES}"
        )

    # --- Nodes --------------------------------------------------------------

    def _generate_node(self, state: "HarnessState", llm: "ChatOpenAI") -> str:
        """Sample N completions in parallel, verify each, return majority winner."""
        samples = self._sample_parallel(state, llm)
        meta = state.get("task_metadata") or {}

        verifier_results: list[VerifierResult] = []
        for s in samples:
            try:
                r = self._verifier(s, meta)
                # Defensive: coerce to dict shape
                if not isinstance(r, dict):
                    r = {"valid": False, "extracted_answer": None, "trace": "verifier returned non-dict"}
            except Exception as exc:  # verifier must not crash the harness
                r = {
                    "valid": False,
                    "extracted_answer": None,
                    "trace": f"verifier raised {type(exc).__name__}: {exc}",
                }
            verifier_results.append(r)

        winning, answer, valid_count, vote_count = majority_vote(samples, verifier_results)

        self._last_verifier_valid = valid_count > 0
        self._last_trace = (
            f"self_consistency: n={len(samples)}, valid={valid_count}, "
            f"vote={vote_count}, answer={answer!r}"
        )
        # Still return the winning sample as the "code" output, so downstream
        # code paths (benchmark scorers, humaneval subprocesses, etc) see the
        # same shape as CompletionTaskGraph's generate output.
        return winning

    def _verify_node(self, state: "HarnessState", _llm: "ChatOpenAI") -> str:
        """PASS iff at least one sample passed the verifier."""
        if self._last_verifier_valid is None:
            # No generate ran — this shouldn't happen under HarnessGraph but
            # is possible in direct unit-test harnessing.
            return "FAIL: verify called before generate"
        if self._last_verifier_valid:
            return f"PASS: {self._last_trace}"
        return f"FAIL: all samples invalid; {self._last_trace}"

    # --- Internals ----------------------------------------------------------

    @staticmethod
    def _default_prompt_builder(state: "HarnessState", _sample_idx: int) -> str:
        return state["task"]

    def _sampling_llm(self, llm: "ChatOpenAI") -> "ChatOpenAI":
        """Build a sibling LLM that hits the same backend with temperature
        ``sampling_temperature`` instead of the cached factory's pinned value.

        The cached factory's temperature is fixed at construction (it's a
        ChatOpenAI model field, not a request param). Without this, every
        sample comes back identical under temperature=0 and majority vote
        provides no signal. We construct a fresh _CompatibleChatOpenAI with
        the same backend params and a different temperature.
        """
        if self._sampling_temperature == getattr(llm, "temperature", None):
            return llm
        try:
            from smboost.llm.runtime import _CompatibleChatOpenAI
            sibling = _CompatibleChatOpenAI(
                model=getattr(llm, "model", "qwen3.5:2b"),
                base_url=getattr(llm, "openai_api_base", None) or
                         getattr(llm, "base_url", None),
                api_key=getattr(llm, "openai_api_key", None) or
                        getattr(llm, "api_key", None),
                temperature=self._sampling_temperature,
                max_tokens=getattr(llm, "max_tokens", None),
            )
            return sibling
        except Exception:
            # Fallback: return the original LLM. Diversity will be limited
            # but C5 will still vote on whatever samples come back.
            return llm

    def _sample_parallel(self, state: "HarnessState", llm: "ChatOpenAI") -> list[str]:
        prompts = [self._prompt_builder(state, i) for i in range(self._n_samples)]
        sampling_llm = self._sampling_llm(llm)

        def _one(idx_prompt: tuple[int, str]) -> str:
            _, prompt = idx_prompt
            try:
                msg = sampling_llm.invoke([HumanMessage(content=prompt)])
            except Exception as exc:
                return f"<<sampling_error: {type(exc).__name__}: {exc}>>"
            content = getattr(msg, "content", "")
            if content is None:
                return ""
            return str(content)

        if self._n_samples == 1:
            # Avoid thread-pool overhead for n=1 (useful for tests / degenerate runs)
            return [_one((0, prompts[0]))]

        # Thread pool keeps HTTP calls to llama.cpp concurrent.
        with ThreadPoolExecutor(max_workers=self._n_samples) as pool:
            return list(pool.map(_one, list(enumerate(prompts))))


__all__ = [
    "SelfConsistencyTaskGraph",
    "Verifier",
    "VerifierResult",
    "execute_program_verifier",
    "majority_vote",
    "run_tests_verifier",
    "tool_call_valid_verifier",
]
