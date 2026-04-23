import importlib

from benchmarks.livecodebench.conditions import CONDITIONS
from langchain_core.messages import HumanMessage


def test_c1_full():
    agent = CONDITIONS["C1"]("qwen3.5:4b", 0)
    assert agent.grounded_verify is True
    assert agent.session_memory is True
    assert agent.shrinkage_enabled is True
    assert agent.scorer_enabled is True


def test_c2_no_grounded_verify():
    agent = CONDITIONS["C2"]("qwen3.5:4b", 0)
    assert agent.grounded_verify is False
    assert agent.session_memory is True
    assert agent.shrinkage_enabled is True
    assert agent.scorer_enabled is True


def test_c3_no_session_memory():
    agent = CONDITIONS["C3"]("qwen3.5:4b", 0)
    assert agent.grounded_verify is True
    assert agent.session_memory is False
    assert agent.shrinkage_enabled is True
    assert agent.scorer_enabled is True


def test_c4_plain_langgraph_retry():
    agent = CONDITIONS["C4"]("qwen3.5:4b", 0)
    assert agent.grounded_verify is False
    assert agent.session_memory is False
    assert agent.shrinkage_enabled is False
    assert agent.scorer_enabled is False
    assert agent.fallback_chain == ["qwen3.5:4b"]


def test_conditions_keys():
    assert set(CONDITIONS.keys()) == {"C1", "C2", "C3", "C4"}


def test_conditions_support_local_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "local")
    monkeypatch.setenv("SMBOOST_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("SMBOOST_LOCAL_N_CTX", "1234")
    monkeypatch.setenv("SMBOOST_LOCAL_N_GPU_LAYERS", "2")
    monkeypatch.setenv("SMBOOST_LOCAL_OFFLOAD_KQV", "false")
    monkeypatch.setenv("SMBOOST_LOCAL_MAX_TOKENS", "2048")

    import benchmarks.livecodebench.conditions as cond_mod
    cond_mod = importlib.reload(cond_mod)

    agent = cond_mod.CONDITIONS["C1"]("qwen3.5:2b", 0)

    from smboost.llm.local import LlamaCppLocalFactory

    assert isinstance(agent._harness._llm_factory, LlamaCppLocalFactory)
    assert agent._harness._llm_factory._model_dir == tmp_path
    assert agent._harness._llm_factory._n_ctx == 1234
    assert agent._harness._llm_factory._n_gpu_layers == 2
    assert agent._harness._llm_factory._offload_kqv is False
    assert agent._harness._llm_factory._max_tokens == 2048

    monkeypatch.delenv("SMBOOST_LLM_BACKEND")
    monkeypatch.delenv("SMBOOST_MODEL_DIR")
    monkeypatch.delenv("SMBOOST_LOCAL_N_CTX")
    monkeypatch.delenv("SMBOOST_LOCAL_N_GPU_LAYERS")
    monkeypatch.delenv("SMBOOST_LOCAL_OFFLOAD_KQV")
    monkeypatch.delenv("SMBOOST_LOCAL_MAX_TOKENS")
    importlib.reload(cond_mod)


def test_conditions_support_server_backend(monkeypatch):
    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "server")
    monkeypatch.setenv("SMBOOST_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("SMBOOST_OPENAI_API_KEY", "sk-no-key")

    import benchmarks.livecodebench.conditions as cond_mod
    cond_mod = importlib.reload(cond_mod)

    agent = cond_mod.CONDITIONS["C1"]("qwen3.5:2b", 0)

    llm = agent._harness._llm_factory("qwen3.5:2b")
    assert llm.model_name == "qwen3.5:2b"
    assert str(llm.openai_api_base) == "http://127.0.0.1:8000/v1"


def test_server_backend_uses_top_level_max_tokens_for_llama_server(monkeypatch):
    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "server")
    monkeypatch.setenv("SMBOOST_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("SMBOOST_OPENAI_API_KEY", "sk-no-key")
    monkeypatch.setenv("SMBOOST_OPENAI_MAX_TOKENS", "64")

    import benchmarks.livecodebench.conditions as cond_mod
    cond_mod = importlib.reload(cond_mod)

    agent = cond_mod.CONDITIONS["C1"]("qwen3.5:2b", 0)
    llm = agent._harness._llm_factory("qwen3.5:2b")
    payload = llm._get_request_payload([HumanMessage(content="hi")])

    assert payload["max_tokens"] == 64
    assert "max_completion_tokens" not in payload


def test_conditions_support_provider_backend(monkeypatch):
    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "provider")
    monkeypatch.setenv("SMBOOST_OPENAI_BASE_URL", "https://api.together.xyz/v1")
    monkeypatch.setenv("SMBOOST_OPENAI_API_KEY", "test-key")

    import benchmarks.livecodebench.conditions as cond_mod
    cond_mod = importlib.reload(cond_mod)

    agent = cond_mod.CONDITIONS["C1"]("Qwen/Qwen3-1.7B", 0)

    llm = agent._harness._llm_factory("Qwen/Qwen3-1.7B")
    assert llm.model_name == "Qwen/Qwen3-1.7B"
    assert str(llm.openai_api_base) == "https://api.together.xyz/v1"
