from __future__ import annotations
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from smboost.memory.session import SessionMemory
from smboost.tasks.base import TaskGraph

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from smboost.harness.state import HarnessState

_ACTIVE_MEMORY: ContextVar[SessionMemory | None] = ContextVar("_ACTIVE_MEMORY", default=None)


_THINK_BLOCK = re.compile(r"<think>.*?</think>\n*", re.DOTALL)
_FENCED_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


_UNCLOSED_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)$", re.DOTALL)


def _clean(raw: str) -> str:
    # Strip closed <think>...</think> blocks (normal case)
    cleaned = _THINK_BLOCK.sub("", raw)
    # Strip unclosed <think> block — model was truncated before </think>
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    # Extract from closed ```python...``` fence
    fenced = _FENCED_CODE.search(cleaned)
    if fenced:
        return fenced.group(1)
    # Extract from unclosed fence — model truncated before closing ```
    unclosed = _UNCLOSED_FENCE.search(cleaned)
    if unclosed:
        return unclosed.group(1)
    return cleaned


def _generate_node(state: HarnessState, llm) -> str:
    task = state["task"]
    level = state["shrinkage_level"]
    meta = state.get("task_metadata") or {}
    testtype = meta.get("testtype", "")
    entry_point = meta.get("entry_point", "")

    if testtype == "stdin":
        if level == 0:
            prompt = (
                "Solve the following competitive programming problem. "
                "Write a complete Python program that reads from stdin and writes to stdout. "
                "Output only the Python code, no markdown fences, no explanation.\n\n"
                + task
            )
        elif level == 1:
            prompt = (
                "Solve this problem. Complete Python program, stdin/stdout only. "
                "Code only, no explanation:\n\n" + task
            )
        elif level == 2:
            prompt = "Python stdin/stdout solution:\n\n" + task[:600]
        else:
            prompt = task[:400]
    elif testtype == "functional" and entry_point:
        if level == 0:
            prompt = (
                "Complete the following Python class method. "
                "Output only the full class with the method implemented, no markdown, no explanation.\n\n"
                + task + "\n\n" + entry_point
            )
        elif level == 1:
            prompt = (
                "Implement this Python method. Code only:\n\n"
                + entry_point + "\n\n" + task[:400]
            )
        elif level == 2:
            prompt = "Complete this Python method:\n\n" + entry_point
        else:
            prompt = entry_point
    else:
        # HumanEval-style: task IS the Python code with docstring
        if level == 0:
            prompt = task
        elif level == 1:
            prompt = (
                "Complete the following Python function. Return only the function body "
                "(indented), with no markdown, no explanation.\n\n" + task
            )
        elif level == 2:
            prompt = "Complete this Python code. Code only:\n\n" + task
        else:
            prompt = task[:800]

    mem = _ACTIVE_MEMORY.get()
    if mem is not None:
        task_id = (state.get("task_metadata") or {}).get("task_id", "")
        recent = mem.recent_for_task(task_id, limit=2) if task_id else []
        if recent:
            hints = "\n\n".join(
                f"[{r.error_class}] {r.traceback_tail[-400:]}" for r in recent
            )
            prompt = (
                f"Previous attempts failed with:\n{hints}\nFix the specific issue.\n\n"
                + prompt
            )
        cross = mem.find_similar(
            task_id=task_id,
            prompt=task,
            error_class=recent[0].error_class if recent else "",
        ) if task_id else None
        if cross is not None:
            prompt = (
                f"A previous different task failed similarly with {cross.error_class}. "
                f"Avoid the same pattern.\n\n" + prompt
            )

    # /no_think disables Qwen3's extended thinking mode so the model responds directly
    raw = llm.invoke([HumanMessage(content="/no_think\n\n" + prompt)]).content or ""
    return _clean(raw)


def _entry_point_caller(entry_point: str) -> str:
    """Return a callable expression for use in assertions.

    Handles three formats:
      - LCB class skeleton: 'class Solution:\\n    def count(...):'  → 'Solution().count'
      - Plain function def:  'def double(x):'                         → 'double'
      - Bare name:           'double'                                  → 'double'
    """
    if not entry_point:
        return "solve"
    try:
        # Entry points often end with just whitespace (empty method body).
        # Append `pass` to make the skeleton parseable.
        tree = ast.parse(entry_point.rstrip() + "\n        pass")
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and not child.name.startswith("_"):
                        return f"{class_name}().{child.name}"
                return f"{class_name}()"
            if isinstance(node, ast.FunctionDef):
                return node.name
    except Exception:
        pass
    # bare name with no spaces/newlines
    if "\n" not in entry_point and " " not in entry_point:
        return entry_point.strip()
    return "solve"


def _make_functional_assertions(entry_point: str, test_cases: list) -> str:
    caller = _entry_point_caller(entry_point)
    lines = []
    for tc in test_cases:
        try:
            args = json.loads(tc["input"])
            if not isinstance(args, list):
                args = [args]
            arg_repr = ", ".join(repr(a) for a in args)
        except (json.JSONDecodeError, TypeError):
            arg_repr = repr(tc["input"])
        try:
            expected = json.loads(tc["output"])
        except (json.JSONDecodeError, TypeError):
            expected = tc["output"]
        lines.append(f"assert {caller}({arg_repr}) == {expected!r}")
    return "\n".join(lines)


def _run_subprocess(src: str, stdin_data: str = "", timeout: int = 12) -> dict:
    """Run src in a fresh subprocess. Returns {passed, stderr, traceback, stdout}."""

    def _set_limits():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
            try:
                resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
            except (ValueError, OSError):
                pass
        except ImportError:
            pass

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        tmp = Path(f.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp)],
            input=stdin_data,
            text=True,
            capture_output=True,
            timeout=timeout,
            preexec_fn=_set_limits if sys.platform != "win32" else None,
        )
        tb = proc.stderr[-800:] if proc.stderr else ""
        return {
            "passed": proc.returncode == 0,
            "stderr": proc.stderr,
            "traceback": tb,
            "stdout": proc.stdout,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stderr": "",
            "traceback": f"TimeoutError: exceeded {timeout}s",
            "stdout": "",
        }
    finally:
        tmp.unlink(missing_ok=True)


def _last_generate_output(state: HarnessState) -> str | None:
    return next(
        (s.output for s in reversed(state["step_outputs"]) if s.node == "generate"),
        None,
    )


def _verify_ast_only(state: HarnessState, _llm) -> str:
    completion = _last_generate_output(state)
    if completion is None:
        return "FAIL: no generate output"
    if not completion.strip():
        return "FAIL: empty completion"
    try:
        ast.parse(completion)
    except SyntaxError as exc:
        return f"FAIL: syntax error: {exc.msg}"
    return "PASS"


def _verify_grounded(state: HarnessState, _llm) -> str:
    completion = _last_generate_output(state)
    if completion is None:
        return "FAIL: no generate output"
    meta = state.get("task_metadata") or {}
    testtype = meta.get("testtype")

    if not testtype:
        return _verify_ast_only(state, _llm)

    if testtype == "functional":
        entry_point = meta.get("entry_point", "")
        test_cases = meta.get("test_cases", [])
        if not entry_point or not test_cases:
            return _verify_ast_only(state, _llm)
        assertions = _make_functional_assertions(entry_point, test_cases)
        src = completion + "\n\n" + assertions
        result = _run_subprocess(src)
        # Record failure in session memory
        mem = _ACTIVE_MEMORY.get()
        task_id = (state.get("task_metadata") or {}).get("task_id", "")
        if not result["passed"] and mem is not None and task_id:
            tb_last = result.get("traceback", "")[-800:]
            error_type = "Error"
            for line in reversed(tb_last.splitlines()):
                if "Error" in line or "Exception" in line:
                    error_type = line.strip()
                    break
            first_assertion = next(
                (ln.strip() for ln in tb_last.splitlines() if ln.strip().startswith("assert ")),
                "",
            )
            mem.record(
                task_id=task_id,
                node="verify",
                attempt=len(state["step_outputs"]),
                error_class=error_type,
                error_line=error_type,
                traceback_tail=tb_last,
                first_assertion=first_assertion,
                prompt_used=state["task"],
            )
    elif testtype == "stdin":
        test_cases = meta.get("test_cases", [])
        if not test_cases:
            return _verify_ast_only(state, _llm)
        result = {"passed": True, "traceback": "", "stdout": ""}
        for tc in test_cases:
            result = _run_subprocess(completion, stdin_data=tc.get("input", ""))
            if not result["passed"]:
                break
            if tc.get("output", "").strip() != result["stdout"].strip():
                tb = f"OutputMismatch: expected {tc['output']!r}, got {result['stdout']!r}"
                result = {"passed": False, "traceback": tb}
                break
        # Record failure in session memory
        mem = _ACTIVE_MEMORY.get()
        task_id = (state.get("task_metadata") or {}).get("task_id", "")
        if not result["passed"] and mem is not None and task_id:
            tb_last = result.get("traceback", "")[-800:]
            error_type = "Error"
            for line in reversed(tb_last.splitlines()):
                if "Error" in line or "Exception" in line:
                    error_type = line.strip()
                    break
            first_assertion = next(
                (ln.strip() for ln in tb_last.splitlines() if ln.strip().startswith("assert ")),
                "",
            )
            mem.record(
                task_id=task_id,
                node="verify",
                attempt=len(state["step_outputs"]),
                error_class=error_type,
                error_line=error_type,
                traceback_tail=tb_last,
                first_assertion=first_assertion,
                prompt_used=state["task"],
            )
    else:
        return _verify_ast_only(state, _llm)

    if result["passed"]:
        return "PASS"
    tb = result.get("traceback", "")
    lines = tb.splitlines()
    error_type = "Error"
    for line in reversed(lines):
        if "Error" in line or "Exception" in line:
            error_type = line.strip()
            break
    return f"FAIL: {error_type}\n{tb[-800:]}"


class CompletionTaskGraph(TaskGraph):
    """Code-completion task graph for HumanEval + LiveCodeBench.

    grounded_verify=True  — uses subprocess sandbox against task_metadata test cases.
    grounded_verify=False — legacy ast.parse-only verify (ablation C2, C4).
    """

    def __init__(self, grounded_verify: bool = True):
        self._grounded = grounded_verify

    @property
    def node_names(self) -> list[str]:
        return ["generate", "verify"]

    def get_node_fn(self, node_name: str):
        if node_name == "generate":
            return _generate_node
        if node_name == "verify":
            return _verify_grounded if self._grounded else _verify_ast_only
        raise ValueError(f"Unknown node {node_name!r}. Valid names: {self.node_names}")
