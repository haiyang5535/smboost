from __future__ import annotations
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from smboost.tasks.base import TaskGraph
from smboost.tasks._tool_loop import run_tool_loop

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama
    from smboost.harness.state import HarnessState


class ToolCallingTaskGraph(TaskGraph):
    def __init__(self, tools: list):
        self._tools = tools
        self._tool_map = {t.name: t for t in tools}

    @property
    def node_names(self) -> list[str]:
        return ["plan", "dispatch", "verify"]

    def get_node_fn(self, node_name: str):
        _node_fns = {
            "plan": self._plan_node,
            "dispatch": self._dispatch_node,
            "verify": self._verify_node,
        }
        if node_name not in _node_fns:
            raise ValueError(
                f"Unknown node {node_name!r}. Valid names: {list(_node_fns)}"
            )
        return _node_fns[node_name]

    def _plan_node(self, state: HarnessState, llm: ChatOllama) -> str:
        level = state["shrinkage_level"]
        task = state["task"]
        if level == 0:
            prompt = f"Plan how to complete this task using your available tools:\n\n{task}"
        elif level == 1:
            prompt = f"Create a plan to: {task}"
        elif level == 2:
            prompt = f"Steps to: {task}"
        else:
            prompt = f"Plan: {task[:300]}"
        return llm.invoke([HumanMessage(content=prompt)]).content or ""

    def _dispatch_node(self, state: HarnessState, llm: ChatOllama) -> str:
        level = state["shrinkage_level"]
        plan = state["step_outputs"][-1].output if state["step_outputs"] else ""
        task = state["task"]
        tools = self._tools if level <= 1 else self._tools[:1]  # level 2+: first tool only
        tool_map = {t.name: t for t in tools}
        if level == 0:
            prompt = f"Execute this task using your tools.\n\nTask: {task}\nPlan: {plan}"
        elif level == 1:
            prompt = f"Execute: {task}\nPlan: {plan}"
        elif level == 2:
            prompt = f"Execute: {task}"
        else:
            prompt = f"Do: {task[:300]}"
        return run_tool_loop(llm.bind_tools(tools), [HumanMessage(content=prompt)], tool_map)

    def _verify_node(self, state: HarnessState, llm: ChatOllama) -> str:
        level = state["shrinkage_level"]
        last = state["step_outputs"][-1].output if state["step_outputs"] else ""
        task = state["task"]
        if level == 0:
            prompt = (
                f"Verify the task was completed correctly.\n\n"
                f"Task: {task}\nOutput: {last}\n\nRespond PASS or FAIL."
            )
        elif level == 1:
            prompt = f"Verify: {task}\nOutput: {last}\nPASS or FAIL?"
        elif level == 2:
            prompt = f"Did this work? Task: {task[:200]}\nOutput: {last[:300]}\nPASS or FAIL."
        else:
            prompt = f"PASS or FAIL: {task[:100]}"
        return llm.invoke([HumanMessage(content=prompt)]).content or ""
