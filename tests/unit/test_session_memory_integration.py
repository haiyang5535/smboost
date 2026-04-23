from unittest.mock import MagicMock

from smboost.harness.state import HarnessState, StepOutput
from smboost.memory.session import SessionMemory
from smboost.tasks import completion as comp_mod


def _base_state(task: str, shrinkage: int) -> HarnessState:
    return {
        "task": task,
        "task_metadata": {"task_id": "t_live"},
        "model": "x",
        "fallback_chain": ["x"],
        "step_outputs": [],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 0,
        "shrinkage_level": shrinkage,
        "status": "running",
        "final_output": None,
    }


def test_generate_prompt_includes_previous_failure_when_memory_on(monkeypatch):
    mem = SessionMemory()
    mem.record(task_id="t_live", node="verify", attempt=0,
               error_class="AssertionError",
               error_line="assert double(2) == 4",
               traceback_tail="AssertionError\nassert double(2) == 4",
               first_assertion="assert double(2) == 4",
               prompt_used="prompt")

    comp_mod._ACTIVE_MEMORY.set(mem)
    try:
        captured = {}
        class FakeLLM:
            def invoke(self, messages):
                captured["prompt"] = messages[0].content
                class R:
                    content = "def double(x):\n    return x * 2"
                return R()
        comp_mod._generate_node(_base_state("prompt", shrinkage=1), FakeLLM())
    finally:
        comp_mod._ACTIVE_MEMORY.set(None)

    assert "Previous attempts failed" in captured["prompt"]
    assert "AssertionError" in captured["prompt"]


def test_small_model_stdin_prompt_skips_memory_hints():
    mem = SessionMemory()
    mem.record(task_id="t_live", node="verify", attempt=0,
               error_class="AssertionError",
               error_line="assert f() == 1",
               traceback_tail="AssertionError\nassert f() == 1",
               first_assertion="assert f() == 1",
               prompt_used="prompt")

    token = comp_mod._ACTIVE_MEMORY.set(mem)
    try:
        captured = {}

        class FakeLLM:
            def invoke(self, messages):
                captured["prompt"] = messages[0].content
                class R:
                    content = "import sys\n\ndef solve():\n    pass"
                return R()

        state = _base_state("problem text", shrinkage=2)
        state["model"] = "qwen3.5:2b"
        state["task_metadata"] = {"task_id": "t_live", "testtype": "stdin"}
        comp_mod._generate_node(state, FakeLLM())
    finally:
        comp_mod._ACTIVE_MEMORY.reset(token)

    assert "Previous attempts failed" not in captured["prompt"]
    assert "Avoid the same pattern." not in captured["prompt"]


def test_non_stdin_small_model_still_uses_memory_hints():
    mem = SessionMemory()
    mem.record(task_id="t_live", node="verify", attempt=0,
               error_class="AssertionError",
               error_line="assert f() == 1",
               traceback_tail="AssertionError\nassert f() == 1",
               first_assertion="assert f() == 1",
               prompt_used="prompt")

    token = comp_mod._ACTIVE_MEMORY.set(mem)
    try:
        captured = {}

        class FakeLLM:
            def invoke(self, messages):
                captured["prompt"] = messages[0].content
                class R:
                    content = "def double(x):\n    return x * 2"
                return R()

        state = _base_state("def double(x):\n    ", shrinkage=1)
        state["model"] = "qwen3.5:2b"
        state["task_metadata"] = {"task_id": "t_live", "testtype": "functional"}
        comp_mod._generate_node(state, FakeLLM())
    finally:
        comp_mod._ACTIVE_MEMORY.reset(token)

    assert "Previous attempts failed" in captured["prompt"]
    assert "AssertionError" in captured["prompt"]


def test_generate_prompt_skips_memory_when_off():
    comp_mod._ACTIVE_MEMORY.set(None)
    captured = {}
    class FakeLLM:
        def invoke(self, messages):
            captured["prompt"] = messages[0].content
            class R:
                content = "def double(x):\n    return x * 2"
            return R()
    comp_mod._generate_node(_base_state("prompt", shrinkage=1), FakeLLM())
    assert "Previous attempts failed" not in captured["prompt"]


def test_agent_memory_set_and_cleared(monkeypatch):
    """Verify context var is set during run and cleared after."""
    from smboost import HarnessAgent, InvariantSuite
    from smboost.tasks.completion import CompletionTaskGraph

    agent = HarnessAgent(
        model="qwen3.5:4b",
        invariants=InvariantSuite.completion(),
        task_graph=CompletionTaskGraph(),
        session_memory=True,
    )
    assert agent._memory is not None
    agent_no_mem = HarnessAgent(
        model="qwen3.5:4b",
        invariants=InvariantSuite.completion(),
        task_graph=CompletionTaskGraph(),
        session_memory=False,
    )
    assert agent_no_mem._memory is None
