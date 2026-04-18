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
        prompt = (
            f"Plan how to complete this task using your available tools:\n\n{state['task']}"
        )
        return llm.invoke([HumanMessage(content=prompt)]).content or ""

    def _dispatch_node(self, state: HarnessState, llm: ChatOllama) -> str:
        plan = state["step_outputs"][-1].output if state["step_outputs"] else ""
        prompt = (
            f"Execute this task using your tools.\n\nTask: {state['task']}\nPlan: {plan}"
        )
        return run_tool_loop(
            llm.bind_tools(self._tools),
            [HumanMessage(content=prompt)],
            self._tool_map,
        )

    def _verify_node(self, state: HarnessState, llm: ChatOllama) -> str:
        last = state["step_outputs"][-1].output if state["step_outputs"] else ""
        prompt = (
            f"Verify the task was completed correctly.\n\n"
            f"Task: {state['task']}\nOutput: {last}\n\nRespond PASS or FAIL."
        )
        return llm.invoke([HumanMessage(content=prompt)]).content or ""
