from __future__ import annotations
import os
import time
from dataclasses import dataclass


@dataclass
class FrontierOutput:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


def _openai_chat_complete(client, model: str, prompt: str):
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )


def _anthropic_message_create(client, model: str, prompt: str):
    return client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )


class OpenAIClient:
    def __init__(self, model_name: str, api_key: str | None = None):
        from openai import OpenAI
        self.model = model_name
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def generate(self, prompt: str) -> FrontierOutput:
        start = time.monotonic()
        resp = _openai_chat_complete(self._client, self.model, prompt)
        latency = int((time.monotonic() - start) * 1000)
        text = resp.choices[0].message.content or ""
        return FrontierOutput(
            text=text,
            input_tokens=getattr(resp.usage, "prompt_tokens", 0),
            output_tokens=getattr(resp.usage, "completion_tokens", 0),
            latency_ms=latency,
        )


class AnthropicClient:
    def __init__(self, model_name: str, api_key: str | None = None):
        import anthropic
        self.model = model_name
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def generate(self, prompt: str) -> FrontierOutput:
        start = time.monotonic()
        resp = _anthropic_message_create(self._client, self.model, prompt)
        latency = int((time.monotonic() - start) * 1000)
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return FrontierOutput(
            text=text,
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
            latency_ms=latency,
        )
