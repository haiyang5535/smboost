"""Integration tests — requires `ollama serve` with qwen3.5:2b pulled."""
import pytest
from smboost import HarnessAgent, InvariantSuite
from smboost.harness.result import HarnessResult

@pytest.mark.integration
def test_coding_agent_returns_result():
    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.coding_agent(),
        fallback_chain=["qwen3.5:2b"],
    )
    result = agent.run("Write a Python function that adds two numbers and return the code.")
    assert isinstance(result, HarnessResult)
    assert result.status in ("success", "failed")
    assert isinstance(result.stats.total_latency_s, float)

@pytest.mark.integration
def test_tool_calling_agent_runs():
    from langchain_core.tools import tool
    from smboost.tasks.tool_calling import ToolCallingTaskGraph

    @tool
    def get_current_time() -> str:
        """Return the current UTC time as a string."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.tool_calling(),
        fallback_chain=["qwen3.5:2b"],
        task_graph=ToolCallingTaskGraph(tools=[get_current_time]),
    )
    result = agent.run("What is the current time?")
    assert isinstance(result, HarnessResult)
    assert result.status in ("success", "failed")
