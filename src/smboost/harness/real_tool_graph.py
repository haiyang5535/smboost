"""Real-tool task graph for condition C6.

Pattern inspiration: GAIA small-model+agentic-tools (4B + tools beats 32B raw),
α-UMi role separation, agent distillation.

Shape
-----
Three nodes running in a bounded loop: ``plan → call → verify``.

- **plan**:   prompt the LLM for a JSON plan listing 1..N steps, each step is
              one of ``code`` (execute in PythonSandbox), ``memory_set`` /
              ``memory_get`` (via MemoryStore), or ``answer`` (finalise).
- **call**:   execute each step in order, collecting the observations.
- **verify**: the LLM reviews the observations against ``task_metadata`` and
              either issues ``PASS`` (if the final answer matches the expected
              outcome) or ``FAIL: ...`` which forces a re-plan (up to
              ``n_iterations``).

Key contrasts with C1 (``CompletionTaskGraph`` + ``grounded_verify``)
---------------------------------------------------------------------
- C1 generates code and *internally* checks it. C6 makes the model a *caller*
  of real tools: the planner proposes, the sandbox executes, the verifier
  observes ground-truth stdout / results.
- Verification in C6 uses observed tool outputs, not re-running the generator.

Fallbacks
---------
- ``src.smboost.llm.grammar`` (Agent 1) is optional. If present, we prefer
  its grammar-constrained call helper (``with_json_schema`` or similar); if
  absent, we fall back to free-form ``llm.invoke`` + robust JSON extraction.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from smboost.tasks.base import TaskGraph
from smboost.tools import MemoryStore, PythonSandbox

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from smboost.harness.state import HarnessState


# --- grammar helper (soft dependency on Agent 1) -----------------------------


def _try_import_grammar_helpers():
    """Return ``(json_plan_grammar, apply_grammar_fn)`` if the grammar module
    is available, otherwise ``(None, None)``. Any import failure is swallowed —
    callers fall back to free-form JSON extraction."""
    try:
        from smboost.llm import grammar as _grammar  # type: ignore[attr-defined]
    except Exception:
        return None, None
    grammar_obj = getattr(_grammar, "JSON_PLAN_GRAMMAR", None) or getattr(
        _grammar, "json_plan_grammar", None
    )
    apply_fn = getattr(_grammar, "apply_grammar", None) or getattr(
        _grammar, "bind_grammar", None
    )
    return grammar_obj, apply_fn


# --- JSON extraction ---------------------------------------------------------


_FENCED_JSON = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_json_object(raw: str) -> Any:
    """Parse a JSON object from model output.

    Accepts:
      - a bare JSON string
      - a fenced ```json ... ``` block
      - a JSON object embedded in prose (first ``{...}`` balance-match)
    Raises ``json.JSONDecodeError`` if nothing parseable is found.
    """
    stripped = raw.strip()
    if not stripped:
        raise json.JSONDecodeError("empty output", raw, 0)

    # 1. try whole string
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. try fenced block
    m = _FENCED_JSON.search(stripped)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. try first balanced {...}
    start = stripped.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        # fall through

    raise json.JSONDecodeError("no JSON object found", raw, 0)


# --- helpers on HarnessState -------------------------------------------------


def _last_output(state: "HarnessState", node: str) -> str | None:
    for step in reversed(state["step_outputs"]):
        if step.node == node:
            return step.output
    return None


# --- the graph ---------------------------------------------------------------


_DEFAULT_MAX_ITERATIONS = 3


class RealToolTaskGraph(TaskGraph):
    """C6 task graph: planner → caller → verifier with real tool execution.

    Args:
        sandbox:       PythonSandbox instance. Constructed if ``None``.
        memory:        MemoryStore instance. Constructed if ``None``.
        n_iterations:  Max plan→call→verify passes before giving up. Default 3.
        sandbox_timeout_s: Per-``code``-step timeout passed to the sandbox.

    The graph exposes three node names (``plan``, ``call``, ``verify``) so the
    existing ``HarnessGraph._execute_step`` loop can drive it without needing
    a custom runner. The internal iteration counter is maintained in the
    verify-node via ``state["shrinkage_level"]`` re-use, so we stay compatible
    with the stock harness retry machinery.
    """

    def __init__(
        self,
        sandbox: PythonSandbox | None = None,
        memory: MemoryStore | None = None,
        n_iterations: int = _DEFAULT_MAX_ITERATIONS,
        sandbox_timeout_s: float = 5.0,
    ) -> None:
        self._sandbox = sandbox if sandbox is not None else PythonSandbox()
        self._memory = memory if memory is not None else MemoryStore()
        self._n_iterations = max(1, int(n_iterations))
        self._sandbox_timeout_s = float(sandbox_timeout_s)
        self._grammar, self._apply_grammar = _try_import_grammar_helpers()

    # --- TaskGraph contract -------------------------------------------------

    @property
    def node_names(self) -> list[str]:
        return ["plan", "call", "verify"]

    def get_node_fn(self, node_name: str):
        fns = {
            "plan": self._plan_node,
            "call": self._call_node,
            "verify": self._verify_node,
        }
        if node_name not in fns:
            raise ValueError(
                f"Unknown node {node_name!r}. Valid names: {list(fns)}"
            )
        return fns[node_name]

    # --- planner ------------------------------------------------------------

    def _plan_node(self, state: "HarnessState", llm: "ChatOpenAI") -> str:
        task = state["task"]
        memory_keys = self._memory.list_keys()
        memory_hint = (
            f"Known memory keys: {memory_keys}" if memory_keys else "Memory is empty."
        )

        prompt = (
            "You are a planner for a tool-using agent.\n"
            "Decompose the task into 1-4 ordered steps. Respond with ONE JSON "
            'object: {"steps": [{"action": "<code|memory_set|memory_get|answer>", '
            '"payload": "..."}]}\n\n'
            "Actions:\n"
            "  - code:        payload is Python source. Assign final value to `result`.\n"
            "  - memory_set:  payload is '<key>=<json>'. Stores the value.\n"
            "  - memory_get:  payload is the key. Returns the stored value.\n"
            "  - answer:      payload is the final answer string. Only once, last.\n\n"
            f"{memory_hint}\n\n"
            f"Task: {task}\n\n"
            "JSON only. No prose."
        )

        bound = self._maybe_bind_grammar(llm)
        content = bound.invoke([HumanMessage(content=prompt)]).content or ""
        return content

    def _maybe_bind_grammar(self, llm: "ChatOpenAI") -> "ChatOpenAI":
        """If Agent 1's grammar helper is available, bind it; otherwise return
        the raw llm unchanged."""
        if self._grammar is None or self._apply_grammar is None:
            return llm
        try:
            return self._apply_grammar(llm, self._grammar)  # type: ignore[misc]
        except Exception:
            # Soft dependency — any failure falls back to free-form.
            return llm

    # --- caller -------------------------------------------------------------

    def _call_node(self, state: "HarnessState", _llm: "ChatOpenAI") -> str:
        """Execute the plan from the latest successful ``plan`` step.

        Does NOT call the LLM. Returns a JSON string of observations:
            {"observations": [{"step": {...}, "ok": bool, "output": "..."}],
             "answer": "..." | null}
        """
        plan_raw = _last_output(state, "plan")
        if plan_raw is None:
            return json.dumps({
                "observations": [],
                "answer": None,
                "error": "no plan to execute",
            })

        try:
            plan_obj = _extract_json_object(plan_raw)
        except json.JSONDecodeError as exc:
            return json.dumps({
                "observations": [],
                "answer": None,
                "error": f"plan parse error: {exc.msg}",
            })

        steps = plan_obj.get("steps") if isinstance(plan_obj, dict) else None
        if not isinstance(steps, list):
            return json.dumps({
                "observations": [],
                "answer": None,
                "error": "plan has no 'steps' list",
            })

        observations: list[dict[str, Any]] = []
        final_answer: str | None = None
        for step in steps:
            if not isinstance(step, dict):
                observations.append({
                    "step": step,
                    "ok": False,
                    "output": "",
                    "error": "non-dict step",
                })
                continue
            action = step.get("action")
            payload = step.get("payload", "")
            obs = self._execute_step(action, payload)
            observations.append(obs)
            if action == "answer" and obs.get("ok"):
                final_answer = obs.get("output", "")
                break

        return json.dumps({"observations": observations, "answer": final_answer})

    def _execute_step(self, action: Any, payload: Any) -> dict[str, Any]:
        """Dispatch a single step to the appropriate tool."""
        if action == "code":
            if not isinstance(payload, str):
                return {"step": {"action": action}, "ok": False, "output": "",
                        "error": "code payload must be str"}
            res = self._sandbox.run(payload, timeout_s=self._sandbox_timeout_s)
            # Flatten: prefer explicit `result` channel, else stdout
            observed = res.get("result")
            if observed is None:
                observed = (res.get("stdout") or "").rstrip("\n")
            return {
                "step": {"action": "code", "payload": payload},
                "ok": bool(res.get("ok")),
                "output": observed,
                "stderr": res.get("stderr") or "",
                "error": res.get("error"),
            }

        if action == "memory_set":
            if not isinstance(payload, str) or "=" not in payload:
                return {"step": {"action": action, "payload": payload}, "ok": False,
                        "output": "", "error": "memory_set payload must be 'key=<json>'"}
            key, _, raw = payload.partition("=")
            key = key.strip()
            raw = raw.strip()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw  # tolerant: store as-is
            self._memory.set(key, value)
            return {"step": {"action": "memory_set", "payload": payload},
                    "ok": True, "output": f"stored {key!r}", "error": None}

        if action == "memory_get":
            if not isinstance(payload, str):
                return {"step": {"action": action}, "ok": False, "output": "",
                        "error": "memory_get payload must be a key string"}
            value = self._memory.get(payload)
            return {"step": {"action": "memory_get", "payload": payload},
                    "ok": value is not None,
                    "output": "" if value is None else json.dumps(value, default=str),
                    "error": None if value is not None else "key not found"}

        if action == "answer":
            return {"step": {"action": "answer", "payload": payload},
                    "ok": True, "output": str(payload), "error": None}

        return {"step": {"action": action, "payload": payload}, "ok": False,
                "output": "", "error": f"unknown action: {action!r}"}

    # --- verifier -----------------------------------------------------------

    def _verify_node(self, state: "HarnessState", llm: "ChatOpenAI") -> str:
        """Decide PASS / FAIL based on the observations and task_metadata.

        If ``task_metadata['expected_answer']`` is present we do an exact-match
        check *first* (cheap, deterministic, no LLM round-trip). Only if that
        is absent do we ask the LLM to adjudicate.
        """
        call_raw = _last_output(state, "call")
        if call_raw is None:
            return "FAIL: no call output"

        try:
            call_obj = json.loads(call_raw)
        except json.JSONDecodeError as exc:
            return f"FAIL: call output not JSON ({exc.msg})"

        answer = call_obj.get("answer")
        observations = call_obj.get("observations") or []
        error = call_obj.get("error")

        if error:
            return f"FAIL: {error}"
        if answer is None:
            # no explicit answer; try last observation
            if observations:
                last = observations[-1]
                if not last.get("ok"):
                    return f"FAIL: last step errored: {last.get('error') or last.get('stderr') or ''}"
                answer = last.get("output")
            else:
                return "FAIL: no observations"

        meta = state.get("task_metadata") or {}
        expected = meta.get("expected_answer")
        if expected is not None:
            if str(answer).strip() == str(expected).strip():
                return "PASS"
            return f"FAIL: expected {expected!r}, got {str(answer)[:200]!r}"

        # LLM adjudication fallback. Keep the prompt tiny and deterministic.
        prompt = (
            "You are a verifier. Given the task and the observed answer, respond "
            "with exactly 'PASS' if the answer solves the task, otherwise "
            "respond with 'FAIL: <one-line reason>'. No other text.\n\n"
            f"Task: {state['task']}\n"
            f"Observed answer: {str(answer)[:500]}\n"
        )
        try:
            content = (llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
        except Exception as exc:
            return f"FAIL: verifier llm error: {exc}"
        if content.upper().startswith("PASS"):
            return "PASS"
        if content.upper().startswith("FAIL"):
            return content
        return f"FAIL: verifier produced non-PASS/FAIL: {content[:120]!r}"


__all__ = ["RealToolTaskGraph"]
