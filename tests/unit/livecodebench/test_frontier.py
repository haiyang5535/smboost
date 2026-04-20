from unittest.mock import MagicMock, patch

from benchmarks.livecodebench.frontier import OpenAIClient, AnthropicClient
from benchmarks.livecodebench.frontier_pricing import calc_cost


def test_openai_client_reports_usage():
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="hello"))]
    fake_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    with patch("benchmarks.livecodebench.frontier._openai_chat_complete", return_value=fake_resp):
        client = OpenAIClient.__new__(OpenAIClient)
        client.model = "gpt-4o"
        client._client = MagicMock()
        out = client.generate("prompt")
    assert out.text == "hello"
    assert out.input_tokens == 10
    assert out.output_tokens == 5


def test_anthropic_client_reports_usage():
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text="world")]
    fake_resp.usage = MagicMock(input_tokens=3, output_tokens=1)
    with patch("benchmarks.livecodebench.frontier._anthropic_message_create", return_value=fake_resp):
        client = AnthropicClient.__new__(AnthropicClient)
        client.model = "claude-sonnet-4-5"
        client._client = MagicMock()
        out = client.generate("prompt")
    assert out.text == "world"
    assert out.input_tokens == 3
    assert out.output_tokens == 1


def test_pricing_known_model():
    cost = calc_cost("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
    assert cost > 0


def test_pricing_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        calc_cost("totally-fake-model", 1, 1)
