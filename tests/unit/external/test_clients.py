"""Unit tests for benchmarks/external/clients.py.

All HTTP calls are mocked — no live API traffic. We assert on uniform return
shape, cost computation, and the missing-API-key error surface.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.external import clients
from benchmarks.external.clients import (
    ClaudeClient,
    MissingAPIKeyError,
    OpenAIClient,
    OpenRouterClient,
    make_client,
)


def _fake_anthropic_response(text: str, in_tok: int, out_tok: int):
    """Shape: response.content = [TextBlock(type='text', text=...)], .usage."""
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)
    return SimpleNamespace(content=[block], usage=usage)


def _fake_openai_response(text: str, in_tok: int, out_tok: int):
    """Shape: response.choices[0].message.content, .usage.prompt_tokens/completion_tokens."""
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok)
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# Missing-API-key surface
# ---------------------------------------------------------------------------


def test_claude_client_raises_on_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError) as excinfo:
        ClaudeClient()
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_openai_client_raises_on_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError) as excinfo:
        OpenAIClient()
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_openrouter_client_raises_on_missing_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError) as excinfo:
        OpenRouterClient()
    assert "OPENROUTER_API_KEY" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Uniform complete() shape
# ---------------------------------------------------------------------------


def test_claude_client_complete_returns_uniform_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    fake_sdk_client = MagicMock()
    fake_sdk_client.messages.create.return_value = _fake_anthropic_response(
        "hello world", in_tok=100, out_tok=50
    )
    with patch("anthropic.Anthropic", return_value=fake_sdk_client):
        client = ClaudeClient(model="claude-sonnet-4-6")
        out = client.complete("say hi", max_tokens=64, temperature=0.0)

    assert set(out.keys()) == {
        "content", "latency_s", "cost_usd", "input_tokens", "output_tokens",
    }
    assert out["content"] == "hello world"
    assert out["input_tokens"] == 100
    assert out["output_tokens"] == 50
    assert out["cost_usd"] > 0.0
    assert out["latency_s"] >= 0.0


def test_openai_client_complete_returns_uniform_shape(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_sdk_client = MagicMock()
    fake_sdk_client.chat.completions.create.return_value = _fake_openai_response(
        "response text", in_tok=80, out_tok=40
    )
    with patch("openai.OpenAI", return_value=fake_sdk_client):
        client = OpenAIClient(model="gpt-4o")
        out = client.complete("hello", max_tokens=128, temperature=0.0)

    assert out["content"] == "response text"
    assert out["input_tokens"] == 80
    assert out["output_tokens"] == 40
    assert out["cost_usd"] > 0.0


def test_openrouter_client_uses_openrouter_base_url(monkeypatch):
    """OpenRouterClient must point openai.OpenAI at openrouter.ai base_url."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    fake_sdk_client = MagicMock()
    fake_sdk_client.chat.completions.create.return_value = _fake_openai_response(
        "{}", in_tok=10, out_tok=5
    )
    with patch("openai.OpenAI", return_value=fake_sdk_client) as open_factory:
        client = OpenRouterClient()
        out = client.complete("hi")

    # Was the SDK pointed at openrouter?
    kwargs = open_factory.call_args.kwargs
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert client.model == "meta-llama/llama-3-70b-instruct"
    assert out["content"] == "{}"


def test_openrouter_client_respects_explicit_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    with patch("openai.OpenAI", return_value=MagicMock()):
        client = OpenRouterClient(model="llama-3-70b")
    assert client.model == "llama-3-70b"


# ---------------------------------------------------------------------------
# make_client routing
# ---------------------------------------------------------------------------


def test_make_client_routes_claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with patch("anthropic.Anthropic", return_value=MagicMock()):
        c = make_client("claude-sonnet-4-6")
    assert isinstance(c, ClaudeClient)


def test_make_client_routes_gpt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("openai.OpenAI", return_value=MagicMock()):
        c = make_client("gpt-4o")
    assert isinstance(c, OpenAIClient)


def test_make_client_routes_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    with patch("openai.OpenAI", return_value=MagicMock()):
        c = make_client("meta-llama/llama-3-70b-instruct")
    assert isinstance(c, OpenRouterClient)


def test_make_client_routes_llama_short_alias(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    with patch("openai.OpenAI", return_value=MagicMock()):
        c = make_client("llama-3-70b")
    assert isinstance(c, OpenRouterClient)


def test_make_client_rejects_unknown_model():
    with pytest.raises(ValueError) as excinfo:
        make_client("mystery-model-9000")
    assert "Cannot infer provider" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Passthrough of max_tokens / temperature
# ---------------------------------------------------------------------------


def test_claude_client_passes_max_tokens_and_temperature(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    fake_sdk_client = MagicMock()
    fake_sdk_client.messages.create.return_value = _fake_anthropic_response(
        "ok", in_tok=1, out_tok=1
    )
    with patch("anthropic.Anthropic", return_value=fake_sdk_client):
        client = ClaudeClient()
        client.complete("hi", max_tokens=256, temperature=0.3)

    call_kwargs = fake_sdk_client.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 256
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["model"] == "claude-sonnet-4-6"


def test_openai_client_passes_max_tokens_and_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_sdk_client = MagicMock()
    fake_sdk_client.chat.completions.create.return_value = _fake_openai_response(
        "ok", in_tok=1, out_tok=1
    )
    with patch("openai.OpenAI", return_value=fake_sdk_client):
        client = OpenAIClient()
        client.complete("hi", max_tokens=512, temperature=0.5)

    call_kwargs = fake_sdk_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 512
    assert call_kwargs["temperature"] == 0.5
