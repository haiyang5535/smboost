from smboost import HarnessAgent, InvariantSuite
from smboost.tasks.completion import CompletionTaskGraph
import inspect


def test_agent_exposes_flags():
    agent = HarnessAgent(
        model="qwen3.5:4b",
        invariants=InvariantSuite.completion(),
        task_graph=CompletionTaskGraph(),
        grounded_verify=False,
        session_memory=False,
        shrinkage_enabled=False,
        scorer_enabled=False,
    )
    assert agent.grounded_verify is False
    assert agent.session_memory is False
    assert agent.shrinkage_enabled is False
    assert agent.scorer_enabled is False


def test_agent_accepts_task_metadata():
    agent = HarnessAgent(
        model="qwen3.5:4b",
        invariants=InvariantSuite.completion(),
        task_graph=CompletionTaskGraph(),
    )
    sig = inspect.signature(agent.run)
    assert "task_metadata" in sig.parameters
