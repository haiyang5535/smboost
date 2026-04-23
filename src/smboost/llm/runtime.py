from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable

from langchain_openai import ChatOpenAI

from smboost.llm.local import LlamaCppLocalFactory, _LocalLlamaChat


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


class _BenchmarkLlamaCppLocalFactory(LlamaCppLocalFactory):
    def __init__(self, *args, max_tokens: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_tokens = max_tokens

    def __call__(self, model: str):
        if model not in self._cache:
            self._cache[model] = _LocalLlamaChat(
                self._build_client(model),
                max_tokens=self._max_tokens,
            )
        return self._cache[model]


class _CompatibleChatOpenAI(ChatOpenAI):
    """Preserve top-level max_tokens for OpenAI-compatible local servers."""

    @property
    def _default_params(self) -> dict[str, object]:
        params = super()._default_params
        if "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")
        return params

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload


@lru_cache(maxsize=None)
def _cached_local_factory(
    model_dir: str,
    n_ctx: int,
    n_gpu_layers: int,
    offload_kqv: bool,
    max_tokens: int,
    verbose: bool,
) -> Callable[[str], object]:
    return _BenchmarkLlamaCppLocalFactory(
        model_dir=model_dir,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        offload_kqv=offload_kqv,
        verbose=verbose,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=None)
def _cached_openai_factory(
    base_url: str,
    api_key: str,
    max_tokens: int,
) -> Callable[[str], object]:
    def _factory(model: str):
        return _CompatibleChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
        )

    return _factory


def get_benchmark_llm_factory() -> Callable[[str], object]:
    backend = os.environ.get("SMBOOST_LLM_BACKEND", "server").lower()

    if backend == "local":
        return _cached_local_factory(
            os.environ.get("SMBOOST_MODEL_DIR", "models"),
            _env_int("SMBOOST_LOCAL_N_CTX", 8192),
            _env_int("SMBOOST_LOCAL_N_GPU_LAYERS", -1),
            _env_bool("SMBOOST_LOCAL_OFFLOAD_KQV", True),
            _env_int("SMBOOST_LOCAL_MAX_TOKENS", 8192),
            _env_bool("SMBOOST_LOCAL_VERBOSE", False),
        )

    if backend in {"server", "provider"}:
        return _cached_openai_factory(
            os.environ.get("SMBOOST_OPENAI_BASE_URL", "http://localhost:8000/v1"),
            os.environ.get("SMBOOST_OPENAI_API_KEY", "sk-no-key"),
            _env_int("SMBOOST_OPENAI_MAX_TOKENS", 8192),
        )

    raise ValueError(f"Unsupported SMBOOST_LLM_BACKEND: {backend}")
