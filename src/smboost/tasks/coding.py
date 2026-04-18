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


_TOOLS_FULL = [read_file, write_file, run_shell]
_TOOLS_REDUCED = [write_file, run_shell]  # shrinkage level 1+: drop read_file
_TOOL_MAP_FULL = {t.name: t for t in _TOOLS_FULL}
_TOOL_MAP_REDUCED = {t.name: t for t in _TOOLS_REDUCED}


def _plan_node(state: HarnessState, llm: ChatOllama) -> str:
    level = state["shrinkage_level"]
    task = state["task"]
    if level == 0:
        prompt = (
            f"You are a coding agent. Create a step-by-step plan to complete this task:\n\n"
            f"{task}\n\nRespond with a numbered list of concrete steps."
        )
    elif level == 1:
        prompt = f"Create a step-by-step plan:\n\n{task}"
    elif level == 2:
        prompt = f"List the steps to complete: {task}"
    else:
        prompt = f"Plan: {task[:300]}"
    return llm.invoke([HumanMessage(content=prompt)]).content or ""


def _execute_node(state: HarnessState, llm: ChatOllama) -> str:
    level = state["shrinkage_level"]
    plan = state["step_outputs"][-1].output if state["step_outputs"] else ""
    task = state["task"]
    tools = _TOOLS_FULL if level == 0 else _TOOLS_REDUCED
    tool_map = _TOOL_MAP_FULL if level == 0 else _TOOL_MAP_REDUCED
    if level == 0:
        prompt = (
            f"Execute this coding task using your tools.\n\n"
            f"Task: {task}\n\nPlan:\n{plan}"
        )
    elif level == 1:
        prompt = f"Execute: {task}\n\nSteps: {plan}"
    elif level == 2:
        prompt = f"Execute this task: {task}"
    else:
        prompt = f"Do: {task[:300]}"
    return run_tool_loop(llm.bind_tools(tools), [HumanMessage(content=prompt)], tool_map)


def _verify_node(state: HarnessState, llm: ChatOllama) -> str:
    level = state["shrinkage_level"]
    last_output = state["step_outputs"][-1].output if state["step_outputs"] else ""
    task = state["task"]
    if level == 0:
        prompt = (
            f"Verify this coding task was completed correctly.\n\n"
            f"Task: {task}\n\nExecution output:\n{last_output}\n\n"
            f"Respond with PASS or FAIL followed by a brief reason."
        )
    elif level == 1:
        prompt = f"Verify task completed:\n\nTask: {task}\nOutput: {last_output}\n\nPASS or FAIL?"
    elif level == 2:
        prompt = f"Did this work? Task: {task[:200]}\nOutput: {last_output[:300]}\nPASS or FAIL."
    else:
        prompt = f"PASS or FAIL: {task[:100]}"
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
