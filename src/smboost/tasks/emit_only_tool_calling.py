"""Emit-only tool-calling task graph.

Used for BFCL-style benchmarks where the ground truth is the *function call itself*:
the harness must produce a structurally valid `{"name": ..., "arguments": {...}}`
JSON object, but must NOT execute the tool (BFCL tools are fictitious schemas with
no real implementations — executing them would crash).

Contrast with `ToolCallingTaskGraph`, which binds tools to the LLM and runs them
via `run_tool_loop`. Here we only emit; BFCL's AST-match scorer judges correctness.

Accepts two OpenAI-function-schema dict shapes:
    {"name": "fn", "parameters": {...}}                              (BFCL v4)
    {"type": "function", "function": {"name": "fn", "parameters": {...}}}
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from smboost.tasks.base import TaskGraph

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from smboost.harness.state import HarnessState


def _extract_schema(tool: dict) -> dict:
    """Normalize a tool schema dict to the flat {"name": ..., "parameters": ...} form."""
    if "function" in tool and isinstance(tool["function"], dict):
        return tool["function"]
    return tool


def _tool_names(tools: list[dict]) -> list[str]:
    names: list[str] = []
    for t in tools:
        schema = _extract_schema(t)
        name = schema.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def _last_generate_output(state: HarnessState) -> str | None:
    for step in reversed(state["step_outputs"]):
        if step.node == "generate":
            return step.output
    return None


class EmitOnlyToolCallingTaskGraph(TaskGraph):
    """Three-node task graph (plan/generate/verify) that emits a JSON tool call
    without executing it.

    Verify semantics (critical differentiator from CompletionTaskGraph):
      PASS iff the last `generate` output parses as a JSON object AND has a
      `name` field matching one of the declared schemas AND has an `arguments`
      field that is a dict (possibly empty).

    Argument *values* are NOT checked here — BFCL's ground-truth match is the
    authoritative judge of argument correctness.
    """

    def __init__(self, tools: list[dict]):
        self._tools = tools or []
        self._tool_names = _tool_names(self._tools)

    @property
    def node_names(self) -> list[str]:
        return ["plan", "generate", "verify"]

    def get_node_fn(self, node_name: str):
        _node_fns = {
            "plan": self._plan_node,
            "generate": self._generate_node,
            "verify": self._verify_node,
        }
        if node_name not in _node_fns:
            raise ValueError(
                f"Unknown node {node_name!r}. Valid names: {list(_node_fns)}"
            )
        return _node_fns[node_name]

    # --- plan ---------------------------------------------------------------

    def _plan_node(self, state: HarnessState, llm: ChatOpenAI) -> str:
        level = state["shrinkage_level"]
        task = state["task"]
        names = ", ".join(self._tool_names) or "(none)"
        if level == 0:
            prompt = (
                "Decide which function to call for this request. "
                "State the function name and key arguments in one short line.\n\n"
                f"Functions available: {names}\n"
                f"Request: {task}"
            )
        elif level == 1:
            prompt = (
                f"Pick a function and its args.\n"
                f"Functions: {names}\nRequest: {task}"
            )
        elif level == 2:
            prompt = f"Plan one call. Functions: {names}\nAsk: {task[:200]}"
        else:
            prompt = f"Which call? {task[:120]}"
        return llm.invoke([HumanMessage(content=prompt)]).content or ""

    # --- generate -----------------------------------------------------------

    def _generate_node(self, state: HarnessState, llm: ChatOpenAI) -> str:
        level = state["shrinkage_level"]
        task = state["task"]
        schemas = [_extract_schema(t) for t in self._tools]

        if level == 0:
            schema_blob = json.dumps(schemas, indent=2)
            prompt = (
                "Emit exactly one JSON object for a function call. "
                'Shape: {"name": "<fn>", "arguments": {...}}. '
                "No prose, no markdown fences, no commentary.\n\n"
                f"Available functions:\n{schema_blob}\n\n"
                f"Request: {task}"
            )
        elif level == 1:
            schema_blob = json.dumps(schemas)
            prompt = (
                'Emit one JSON object: {"name":"<fn>","arguments":{...}}. '
                "No prose.\n"
                f"Functions: {schema_blob}\n"
                f"Request: {task}"
            )
        elif level == 2:
            names = ", ".join(self._tool_names) or "(none)"
            prompt = (
                'Output one JSON: {"name":"<fn>","arguments":{...}}.\n'
                f"Functions: {names}\n"
                f"Ask: {task[:160]}"
            )
        else:
            names = ", ".join(self._tool_names) or "(none)"
            prompt = (
                f'JSON call only. Functions: {names}. Ask: {task[:80]}'
            )
        return llm.invoke([HumanMessage(content=prompt)]).content or ""

    # --- verify -------------------------------------------------------------

    def _verify_node(self, state: HarnessState, _llm: ChatOpenAI) -> str:
        """Structural-only verify. Does NOT execute the tool.

        PASS iff:
          1. Last `generate` output parses as a JSON object.
          2. `name` matches one of the declared function schemas.
          3. `arguments` is present and is a dict (possibly empty).
        """
        output = _last_generate_output(state)
        if output is None:
            return "FAIL: no generate output"
        stripped = output.strip()
        if not stripped:
            return "FAIL: empty generate output"

        try:
            obj: Any = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return f"FAIL: JSONDecodeError: {exc.msg}"

        if not isinstance(obj, dict):
            return f"FAIL: expected JSON object, got {type(obj).__name__}"

        name = obj.get("name")
        if not isinstance(name, str):
            return "FAIL: missing or non-string 'name' field"

        if self._tool_names and name not in self._tool_names:
            return (
                f"FAIL: function name {name!r} not declared; "
                f"expected one of {self._tool_names}"
            )

        if "arguments" not in obj:
            return "FAIL: missing 'arguments' field"
        if not isinstance(obj["arguments"], dict):
            return (
                f"FAIL: 'arguments' must be a dict, got "
                f"{type(obj['arguments']).__name__}"
            )

        return "PASS"
