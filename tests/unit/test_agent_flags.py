from smboost import HarnessAgent, InvariantSuite
from smboost.tasks.completion import CompletionTaskGraph
import inspect
from unittest.mock import MagicMock
import importlib


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


def test_agent_uses_injected_llm_factory():
    calls = []

    class FakeLLM:
        def invoke(self, messages):
            return MagicMock(content="def double(x):\n    return x * 2")

    def factory(model: str):
        calls.append(model)
        return FakeLLM()

    agent = HarnessAgent(
        model="qwen3.5:4b",
        invariants=InvariantSuite.completion(),
        task_graph=CompletionTaskGraph(grounded_verify=False),
        grounded_verify=False,
        scorer_enabled=False,
        llm_factory=factory,
    )

    result = agent.run("def double(x):\n    pass")

    assert result.status == "success"
    assert calls == ["qwen3.5:4b", "qwen3.5:4b"]


def test_probe_style_factory_uses_server_backend(monkeypatch):
    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "server")
    monkeypatch.setenv("SMBOOST_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("SMBOOST_OPENAI_API_KEY", "sk-no-key")

    from smboost.llm.runtime import get_benchmark_llm_factory

    llm = get_benchmark_llm_factory()("qwen3.5:2b")
    assert llm.model_name == "qwen3.5:2b"


def test_c1_probe_helpers_and_infra_exit(monkeypatch, capsys):
    probe = importlib.import_module("scripts.c1_probe")

    assert probe._banner("provider", "qwen3.5:2b", 3) == "BACKEND=provider  MODEL=qwen3.5:2b  TASKS=3"
    assert probe._classify_failure(
        "x" * 121 + "AssertionError: output mismatch"
    ) == "logic_or_output_mismatch"
    assert probe._normalize_backend() == "server"

    class FakeAgent:
        def run(self, prompt, task_metadata):
            raise RuntimeError("boom")

        def close_memory(self):
            pass

    monkeypatch.setattr(probe, "load_livecodebench_tasks", lambda n: [{"task_id": "t1", "testtype": "stdin", "prompt": "p"}])
    monkeypatch.setattr(probe, "CONDITIONS", {"C1": lambda model, seed: FakeAgent()})

    exit_code = probe.main(["qwen3.5:2b", "1"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "BACKEND=server  MODEL=qwen3.5:2b  TASKS=1" in out
    assert "FAILURE_BUCKETS={'other_runtime': 1}" in out
