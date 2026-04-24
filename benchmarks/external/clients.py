"""Thin uniform wrappers around the Anthropic, OpenAI, and OpenRouter SDKs.

Each client exposes the same signature::

    client.complete(prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> dict

Returning ``{"content": str, "latency_s": float, "cost_usd": float,
"input_tokens": int, "output_tokens": int}``.

These are deliberately minimal — no streaming, no tool use, no retries beyond
what each SDK provides by default. The goal is a measured, deterministic
raw-baseline call path that mirrors how our local small-model `run_baseline`
path invokes the LLM on HumanEval+/BFCL/GSM8K.

Missing-API-key errors are surfaced at ``__init__`` time so a bad run fails
fast instead of silently erroring on the first request.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from .pricing import estimate_cost


@dataclass
class CompletionResult:
    content: str
    latency_s: float
    cost_usd: float
    input_tokens: int
    output_tokens: int

    def asdict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "latency_s": self.latency_s,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class MissingAPIKeyError(RuntimeError):
    """Raised when the env var for a given provider isn't set.

    Using a dedicated subclass makes it trivial to spot the "you forgot to
    export X_API_KEY" case in both unit tests and CLI error handling.
    """


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise MissingAPIKeyError(
            f"{name} is not set. External baselines require real API keys; "
            f"export {name} before running. See .env.example."
        )
    return val


class ClaudeClient:
    """Wraps the Anthropic SDK. Default model: ``claude-sonnet-4-6``.

    We never pass ``thinking`` — Sonnet 4.6 runs non-thinking by default and
    non-thinking is the fair comparison against our local 2B/0.8B models,
    which have no separate reasoning mode.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        import anthropic  # lazy so unit tests can patch without the SDK present

        self.model = model
        self._api_key = api_key or _require_env("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        start = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_s = round(time.monotonic() - start, 3)

        # response.content is a list of blocks; we only care about text.
        text_parts: list[str] = []
        for block in response.content:
            # SDK returns TextBlock objects with .type / .text
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        content = "".join(text_parts)

        in_tok = int(getattr(response.usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(response.usage, "output_tokens", 0) or 0)
        cost = estimate_cost(self.model, in_tok, out_tok)

        return CompletionResult(
            content=content,
            latency_s=latency_s,
            cost_usd=round(cost, 6),
            input_tokens=in_tok,
            output_tokens=out_tok,
        ).asdict()


class OpenAIClient:
    """Wraps the OpenAI SDK. Default model: ``gpt-4o``.

    Uses the standard chat-completions path; the raw baselines are
    single-turn, so we don't need the Responses API.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
    ) -> None:
        import openai  # lazy so unit tests can patch

        self.model = model
        self._api_key = api_key or _require_env("OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=self._api_key)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        start = time.monotonic()
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_s = round(time.monotonic() - start, 3)

        choice = response.choices[0]
        content = (getattr(choice.message, "content", "") or "") if choice else ""

        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        cost = estimate_cost(self.model, in_tok, out_tok)

        return CompletionResult(
            content=content,
            latency_s=latency_s,
            cost_usd=round(cost, 6),
            input_tokens=in_tok,
            output_tokens=out_tok,
        ).asdict()


class OpenRouterClient:
    """Wraps the OpenAI SDK pointed at OpenRouter.

    Default model: ``meta-llama/llama-3-70b-instruct``. Accepts the short
    alias ``llama-3-70b`` too — both are registered in ``pricing.py``.
    OpenRouter's API is OpenAI-compatible; the only differences are
    ``base_url`` and the key env var.
    """

    DEFAULT_MODEL = "meta-llama/llama-3-70b-instruct"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        import openai  # lazy so unit tests can patch

        self.model = model or self.DEFAULT_MODEL
        self._api_key = api_key or _require_env("OPENROUTER_API_KEY")
        self._client = openai.OpenAI(
            api_key=self._api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        start = time.monotonic()
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_s = round(time.monotonic() - start, 3)

        choice = response.choices[0]
        content = (getattr(choice.message, "content", "") or "") if choice else ""

        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        cost = estimate_cost(self.model, in_tok, out_tok)

        return CompletionResult(
            content=content,
            latency_s=latency_s,
            cost_usd=round(cost, 6),
            input_tokens=in_tok,
            output_tokens=out_tok,
        ).asdict()


# Model-string routing. The runner / CLI accepts a single ``--model`` flag;
# we infer the right client from the prefix rather than adding a separate
# ``--provider`` argument the caller would have to remember.
def make_client(model: str) -> ClaudeClient | OpenAIClient | OpenRouterClient:
    """Pick the right client class for a model id.

    - Anything starting with ``claude-`` -> ClaudeClient
    - ``gpt-*`` / ``o1-*`` / ``o3-*`` / ``o4-*`` -> OpenAIClient
    - ``llama-3-70b`` or any slash-containing HF-style id -> OpenRouterClient
    """
    m = model.lower()
    if m.startswith("claude-"):
        return ClaudeClient(model=model)
    if m.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return OpenAIClient(model=model)
    if m == "llama-3-70b" or "/" in m:
        return OpenRouterClient(model=model)
    raise ValueError(
        f"Cannot infer provider for model {model!r}. "
        f"Use a claude-*, gpt-*, or OpenRouter-style id "
        f"(e.g. meta-llama/llama-3-70b-instruct)."
    )
