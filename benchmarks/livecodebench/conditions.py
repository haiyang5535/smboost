from smboost.agent import HarnessAgent
from smboost.invariants.suite import InvariantSuite
from smboost.tasks.completion import CompletionTaskGraph


def build_agent(condition: str, model: str, seed: int) -> HarnessAgent:
    """Factory to build an agent configured for a specific ablation condition."""
    fallback_chain = ["qwen3.5:9b"] if "4b" in model.lower() else None

    if condition == "full":
        return HarnessAgent(
            model=model,
            invariants=InvariantSuite.completion(),
            scorer="adaptive",
            fallback_chain=fallback_chain,
            grounded_verify=True,
            session_memory=True,
            shrinkage_enabled=True,
            scorer_enabled=True,
            task_graph=CompletionTaskGraph(grounded_verify=True),
        )
    elif condition == "-grounded_verify":
        return HarnessAgent(
            model=model,
            invariants=InvariantSuite.completion(),
            scorer="adaptive",
            fallback_chain=fallback_chain,
            grounded_verify=False,
            session_memory=True,
            shrinkage_enabled=True,
            scorer_enabled=True,
            task_graph=CompletionTaskGraph(grounded_verify=False),
        )
    elif condition == "-session_memory":
        return HarnessAgent(
            model=model,
            invariants=InvariantSuite.completion(),
            scorer="adaptive",
            fallback_chain=fallback_chain,
            grounded_verify=True,
            session_memory=False,
            shrinkage_enabled=True,
            scorer_enabled=True,
            task_graph=CompletionTaskGraph(grounded_verify=True),
        )
    elif condition == "plain_langgraph_retry":
        return HarnessAgent(
            model=model,
            invariants=InvariantSuite.completion(),
            scorer="off",
            fallback_chain=fallback_chain,
            grounded_verify=False,
            session_memory=False,
            shrinkage_enabled=False,
            scorer_enabled=False,
            task_graph=CompletionTaskGraph(grounded_verify=False),
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")
