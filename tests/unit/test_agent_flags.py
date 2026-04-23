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


def test_default_factory_alias_matches_canonical_name():
    """`get_benchmark_llm_factory` is kept as a backward-compatible alias for
    `get_default_llm_factory` so existing import sites don't break."""
    from smboost.llm.runtime import get_benchmark_llm_factory, get_default_llm_factory

    assert get_benchmark_llm_factory is get_default_llm_factory


def test_harness_agent_default_factory_respects_env_base_url(monkeypatch):
    """F7: HarnessAgent(..., llm_factory=None) must NOT hard-code
    `localhost:8000`. It should use SMBOOST_OPENAI_BASE_URL via the shared
    env-var-aware factory. No real connection is made — we inspect the
    LangChain ChatOpenAI config."""
    # Clear the cached factory so env vars take effect for this test.
    from smboost.llm.runtime import _cached_openai_factory
    _cached_openai_factory.cache_clear()

    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "server")
    monkeypatch.setenv("SMBOOST_OPENAI_BASE_URL", "http://example.com:9999/v1")
    monkeypatch.setenv("SMBOOST_OPENAI_API_KEY", "sk-test-key")

    agent = HarnessAgent(
        model="qwen3.5:2b",
        invariants=InvariantSuite.completion(),
        task_graph=CompletionTaskGraph(),
        llm_factory=None,  # the path under test
    )

    # Pull the LLM out of the harness and verify its base_url reflects the env.
    llm = agent._harness._llm_factory("qwen3.5:2b")
    # langchain_openai.ChatOpenAI stores the URL on `openai_api_base` (or
    # `base_url` depending on version). Accept either.
    base_url = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", None)
    assert base_url is not None
    assert "example.com:9999" in str(base_url)


def test_bfcl_make_raw_llm_respects_env_base_url(monkeypatch):
    """F7: BFCL raw path must also honor SMBOOST_OPENAI_BASE_URL rather than
    hitting `localhost:8000` unconditionally."""
    from smboost.llm.runtime import _cached_openai_factory
    _cached_openai_factory.cache_clear()

    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "server")
    monkeypatch.setenv("SMBOOST_OPENAI_BASE_URL", "http://example.com:7777/v1")
    monkeypatch.setenv("SMBOOST_OPENAI_API_KEY", "sk-test-key")

    from benchmarks.bfcl.runner import _make_raw_llm

    llm = _make_raw_llm("qwen3.5:2b")
    base_url = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", None)
    assert base_url is not None
    assert "example.com:7777" in str(base_url)


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
