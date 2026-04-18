from __future__ import annotations
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from smboost.tasks.base import TaskGraph
from smboost.tasks._tool_loop import run_tool_loop

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama
    from smboost.harness.state import HarnessState


@tool
def read_file(path: str) -> str:
    """Read a file from disk and return its contents."""
    return Path(path).read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file on disk."""
    Path(path).write_text(content)
    return f"Written to {path}"


@tool
def run_shell(command: str) -> str:
    """Run a shell command and return stdout + stderr. Pipes and shell features are not supported."""
    result = subprocess.run(
        shlex.split(command), shell=False, capture_output=True, text=True, timeout=30
    )
    return result.stdout + result.stderr


_TOOLS = [read_file, write_file, run_shell]
_TOOL_MAP = {t.name: t for t in _TOOLS}


def _plan_node(state: HarnessState, llm: ChatOllama) -> str:
    prompt = (
        f"You are a coding agent. Create a step-by-step plan to complete this task:\n\n"
        f"{state['task']}\n\nRespond with a numbered list of concrete steps."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content or ""


def _execute_node(state: HarnessState, llm: ChatOllama) -> str:
    plan = state["step_outputs"][-1].output if state["step_outputs"] else ""
    prompt = (
        f"Execute this coding task using your tools.\n\n"
        f"Task: {state['task']}\n\nPlan:\n{plan}"
    )
    return run_tool_loop(llm.bind_tools(_TOOLS), [HumanMessage(content=prompt)], _TOOL_MAP)


def _verify_node(state: HarnessState, llm: ChatOllama) -> str:
    last_output = state["step_outputs"][-1].output if state["step_outputs"] else ""
    prompt = (
        f"Verify this coding task was completed correctly.\n\n"
        f"Task: {state['task']}\n\nExecution output:\n{last_output}\n\n"
        f"Respond with PASS or FAIL followed by a brief reason."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content or ""


_NODE_FNS = {"plan": _plan_node, "execute": _execute_node, "verify": _verify_node}


class CodingTaskGraph(TaskGraph):
    @property
    def node_names(self) -> list[str]:
        return ["plan", "execute", "verify"]

    def get_node_fn(self, node_name: str):
        if node_name not in _NODE_FNS:
            raise ValueError(
                f"Unknown node {node_name!r}. Valid names: {list(_NODE_FNS)}"
            )
        return _NODE_FNS[node_name]
