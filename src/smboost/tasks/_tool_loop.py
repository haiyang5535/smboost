from __future__ import annotations
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def run_tool_loop(llm_with_tools: "BaseChatModel", messages: list, tool_map: dict, max_iters: int = 10) -> str:
    from langchain_core.messages import ToolMessage
    for _ in range(max_iters):
        response = llm_with_tools.invoke(messages)
        messages = messages + [response]
        if not response.tool_calls:
            return response.content or ""
        for tc in response.tool_calls:
            tool = tool_map.get(tc["name"])
            if tool is None:
                tool_result = f"Error: unknown tool '{tc['name']}'"
            else:
                tool_result = tool.invoke(tc["args"])
            messages = messages + [
                ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
            ]
    warnings.warn(f"run_tool_loop hit max_iters={max_iters} without a final response", RuntimeWarning)
    return ""
