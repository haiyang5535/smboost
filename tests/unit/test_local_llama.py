from __future__ import annotations

import sys
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage


def test_local_llama_factory_caches_models_and_converts_messages(monkeypatch):
    created = []
    captured = {}

    class FakeLlama:
        def __init__(self, model_path: str, **kwargs):
            created.append((model_path, kwargs))

        def create_chat_completion(self, *, messages, max_tokens, temperature, stop):
            captured["messages"] = messages
            captured["max_tokens"] = max_tokens
            captured["temperature"] = temperature
            captured["stop"] = stop
            return {"choices": [{"message": {"content": "print(1)"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))

    from smboost.llm.local import LlamaCppLocalFactory

    factory = LlamaCppLocalFactory(model_dir="models")
    llm1 = factory("qwen3.5:2b")
    llm2 = factory("qwen3.5:2b")

    assert llm1 is llm2
    assert len(created) == 1
    assert created[0][0].endswith("models/Qwen3.5-2B-Q4_K_M.gguf")

    resp = llm1.invoke(
        [SystemMessage(content="sys"), HumanMessage(content="hello")],
        stop=["```"],
    )

    assert resp.content == "print(1)"
    assert captured["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert captured["max_tokens"] == 8192
    assert captured["temperature"] == 0.0
    assert captured["stop"] == ["```"]


def test_local_llama_factory_honors_max_tokens(monkeypatch):
    created = {}

    class FakeLlama:
        def __init__(self, model_path: str, **kwargs):
            created["kwargs"] = kwargs

        def create_chat_completion(self, *, messages, max_tokens, temperature, stop):
            created["max_tokens"] = max_tokens
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))

    from smboost.llm.local import LlamaCppLocalFactory

    factory = LlamaCppLocalFactory(
        model_dir="models",
        max_tokens=1234,
        n_gpu_layers=7,
        offload_kqv=False,
    )
    llm = factory("qwen3.5:2b")
    llm.invoke([HumanMessage(content="hello")])

    assert created["kwargs"]["n_gpu_layers"] == 7
    assert created["kwargs"]["offload_kqv"] is False
    assert created["max_tokens"] == 1234


def test_local_llama_invoke_ignores_extra_langchain_kwargs(monkeypatch):
    """F14: callers may pass LangChain kwargs like `config=` or `temperature=`
    (including via the `.bind()` machinery). Silently ignore rather than
    raising `TypeError`."""
    from smboost.llm.local import _LocalLlamaChat

    class FakeClient:
        def create_chat_completion(self, *, messages, max_tokens, temperature, stop):
            return {"choices": [{"message": {"content": "hi"}}]}

    chat = _LocalLlamaChat(FakeClient(), max_tokens=64)
    resp = chat.invoke(
        [HumanMessage(content="hello")],
        stop=None,
        config={"callbacks": []},
        temperature=0.7,
    )
    assert resp.content == "hi"


def test_local_llama_bind_tools_raises_helpful_error():
    """F14: `llm.bind_tools(...)` is used by CodingTaskGraph /
    ToolCallingTaskGraph. `_LocalLlamaChat` isn't a real LangChain
    `BaseChatModel`, so fail fast with a clear message instead of a late
    `TypeError` or `AttributeError`."""
    import pytest

    from smboost.llm.local import _LocalLlamaChat

    chat = _LocalLlamaChat(client=object(), max_tokens=64)

    with pytest.raises(RuntimeError) as exc_info:
        chat.bind_tools([])

    msg = str(exc_info.value)
    assert "SMBOOST_LLM_BACKEND=local" in msg
    assert "tool-calling" in msg
    assert "SMBOOST_LLM_BACKEND=server" in msg


def test_matrix_forces_serial_execution_for_local_backend(monkeypatch):
    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "local")

    import benchmarks.livecodebench.matrix as matrix_mod

    assert matrix_mod._effective_concurrency(ollama_concurrency=3) == 1


def test_matrix_forces_serial_execution_for_mixed_case_local_backend(monkeypatch):
    monkeypatch.setenv("SMBOOST_LLM_BACKEND", "LOCAL")

    import benchmarks.livecodebench.matrix as matrix_mod

    assert matrix_mod._effective_concurrency(ollama_concurrency=3) == 1


def test_matrix_caps_non_local_backend_concurrency(monkeypatch):
    monkeypatch.delenv("SMBOOST_LLM_BACKEND", raising=False)

    import benchmarks.livecodebench.matrix as matrix_mod

    assert matrix_mod._effective_concurrency(ollama_concurrency=5) == 3
