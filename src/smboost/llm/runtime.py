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
    def __init__(self, *args, max_tokens: int, temperature: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_tokens = max_tokens
        self._temperature = temperature

    def __call__(self, model: str):
        if model not in self._cache:
            self._cache[model] = _LocalLlamaChat(
                self._build_client(model),
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        return self._cache[model]


class _CompatibleChatOpenAI(ChatOpenAI):
    """Preserve top-level max_tokens for OpenAI-compatible local servers.

    Also carries an optional ``grammar`` field through to the request body
    as an OpenAI-API extension.  llama-cpp-python's OpenAI-compatible
    server accepts ``grammar: <GBNF string>`` on ``/v1/chat/completions``
    to constrain decoding.  Set ``grammar`` via the constructor or via
    :meth:`with_grammar` — the plain OpenAI endpoint will simply ignore
    it, so no-grammar behavior is unchanged.

    Warning:
        This class subclasses private LangChain API — ``_default_params`` and
        ``_get_request_payload``.  Tested against ``langchain-openai==1.1.x``.
        Because the API is private, the dependency in ``pyproject.toml`` is
        pinned to ``>=1.0,<2.0``; re-test before loosening the upper bound.
    """

    # LangChain's pydantic config rejects unknown fields on the main
    # schema; it would otherwise warn and stuff our grammar into
    # ``model_kwargs`` (which would then be sent to the server as part
    # of the body anyway — but without our awareness).  We pop
    # ``grammar`` from kwargs before handing them to super().__init__
    # and stash it via ``object.__setattr__`` to bypass pydantic's
    # attribute guard.
    def __init__(self, *args, grammar: str | None = None, **kwargs):
        # Defensive: some callers may route grammar through
        # ``model_kwargs={"grammar": "..."}``.  Normalise both paths.
        if grammar is None:
            mk = kwargs.get("model_kwargs")
            if isinstance(mk, dict) and "grammar" in mk:
                grammar = mk.pop("grammar")
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_grammar", grammar)

    @property
    def grammar(self) -> str | None:
        return getattr(self, "_grammar", None)

    def with_grammar(self, grammar: str | None) -> "_CompatibleChatOpenAI":
        """Return a shallow copy with ``grammar`` replaced.

        LangChain chat models are immutable by convention; this follows
        the same pattern as ``.bind()`` — callers that want to toggle
        grammar per-call should use this helper rather than mutating.
        """
        clone = self.__class__(**_extract_constructor_kwargs(self))
        object.__setattr__(clone, "_grammar", grammar)
        return clone

    @property
    def _default_params(self) -> dict[str, object]:
        params = super()._default_params
        if "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")
        return params

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        # Allow per-call override via kwargs, else fall back to the
        # instance's default grammar.
        grammar = kwargs.pop("grammar", None)
        if grammar is None:
            grammar = self.grammar

        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        if grammar is not None:
            # llama-cpp-python server accepts this as an OpenAI
            # extension.  Real OpenAI ignores unknown fields, so this is
            # safe to leave in on non-local backends.
            payload["grammar"] = grammar
        return payload


def _extract_constructor_kwargs(instance: "_CompatibleChatOpenAI") -> dict[str, object]:
    """Best-effort inverse of ``__init__`` for ``with_grammar`` cloning.

    We pull the public knobs the factory set (``model``, ``base_url``,
    ``api_key``, ``max_tokens``, ``temperature``) off the instance. This
    mirrors how ``_cached_openai_factory`` constructs the object; if
    additional knobs are added there, add them here too.
    """
    return {
        "model": getattr(instance, "model_name", None) or getattr(instance, "model", None),
        "base_url": str(getattr(instance, "openai_api_base", None) or ""),
        "api_key": getattr(instance, "openai_api_key", None),
        "max_tokens": getattr(instance, "max_tokens", None),
        "temperature": getattr(instance, "temperature", None),
    }


@lru_cache(maxsize=None)
def _cached_local_factory(
    model_dir: str,
    n_ctx: int,
    n_gpu_layers: int,
    offload_kqv: bool,
    max_tokens: int,
    verbose: bool,
    temperature: float,
) -> Callable[[str], object]:
    return _BenchmarkLlamaCppLocalFactory(
        model_dir=model_dir,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        offload_kqv=offload_kqv,
        verbose=verbose,
        max_tokens=max_tokens,
        temperature=temperature,
    )


@lru_cache(maxsize=None)
def _cached_openai_factory(
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    grammar: str | None = None,
) -> Callable[[str], object]:
    def _factory(model: str):
        return _CompatibleChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            grammar=grammar,
        )

    return _factory


def get_default_llm_factory(temperature: float = 0.0) -> Callable[[str], object]:
    """Return an LLM factory honoring env-var backend selection.

    ``temperature`` defaults to ``0.0`` so that benchmark evaluation is
    deterministic; callers that want stochasticity can override. Pinning
    temperature here also fixes a noise source in gate runs — two harness
    invocations of the same condition would otherwise flip 1-3 tasks from
    sampling jitter on n=20.
    """
    backend = os.environ.get("SMBOOST_LLM_BACKEND", "server").lower()

    if backend == "local":
        return _cached_local_factory(
            os.environ.get("SMBOOST_MODEL_DIR", "models"),
            _env_int("SMBOOST_LOCAL_N_CTX", 8192),
            _env_int("SMBOOST_LOCAL_N_GPU_LAYERS", -1),
            _env_bool("SMBOOST_LOCAL_OFFLOAD_KQV", True),
            _env_int("SMBOOST_LOCAL_MAX_TOKENS", 8192),
            _env_bool("SMBOOST_LOCAL_VERBOSE", False),
            temperature,
        )

    if backend in {"server", "provider"}:
        return _cached_openai_factory(
            os.environ.get("SMBOOST_OPENAI_BASE_URL", "http://localhost:8000/v1"),
            os.environ.get("SMBOOST_OPENAI_API_KEY", "sk-no-key"),
            _env_int("SMBOOST_OPENAI_MAX_TOKENS", 8192),
            temperature,
        )

    raise ValueError(f"Unsupported SMBOOST_LLM_BACKEND: {backend}")


# Backward-compatible alias. Older callers in the codebase (and external probes)
# referenced `get_benchmark_llm_factory`; the new canonical name is
# `get_default_llm_factory` because the same factory is now also used as the
# default when `HarnessAgent(..., llm_factory=None)`.
get_benchmark_llm_factory = get_default_llm_factory
