from __future__ import annotations

import os
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage


_MODEL_MAP = {
    "qwen3.5:0.8b": "Qwen3.5-0.8B-Q4_K_M.gguf",
    "qwen3.5:2b": "Qwen3.5-2B-Q4_K_M.gguf",
    "qwen3.5:4b": "Qwen3.5-4B-Q4_K_M.gguf",
    "qwen3.5:9b": "Qwen3.5-9B-Q4_K_M.gguf",
}


_LOCAL_BIND_TOOLS_ERROR = (
    "SMBOOST_LLM_BACKEND=local does not support tool-calling; "
    "use SMBOOST_LLM_BACKEND=server"
)


class _LocalLlamaChat:
    def __init__(self, client, *, max_tokens: int = 8192):
        self._client = client
        self._max_tokens = max_tokens

    def invoke(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **_kwargs,
    ):
        # Silently ignore unknown LangChain kwargs (e.g. `config=`, `temperature=`,
        # anything passed by the `.bind()` machinery) so this class stays
        # drop-in compatible with callers that treat it like a LangChain
        # `BaseChatModel`. Stop + max_tokens are the only knobs we actually
        # honor locally.
        payload = [_to_chat_message(msg) for msg in messages]
        resp = self._client.create_chat_completion(
            messages=payload,
            max_tokens=self._max_tokens,
            temperature=0.0,
            stop=stop or [],
        )
        content = resp["choices"][0]["message"]["content"]
        return AIMessage(content=content or "")

    def bind_tools(self, *_args, **_kwargs):
        """Tool binding is not supported on the in-process llama.cpp backend.

        Caller graphs (`CodingTaskGraph`, `ToolCallingTaskGraph`) use
        `llm.bind_tools(...)`, which assumes a LangChain `BaseChatModel`.
        Fail fast with a clear message instead of a late `TypeError`.
        """
        raise RuntimeError(_LOCAL_BIND_TOOLS_ERROR)


class LlamaCppLocalFactory:
    """Lazy, cached in-process llama.cpp chat models for completion probes."""

    def __init__(
        self,
        model_dir: str | Path = "models",
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        offload_kqv: bool = True,
        max_tokens: int = 8192,
        chat_format: str = "qwen",
        verbose: bool = False,
    ):
        self._model_dir = Path(model_dir)
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._offload_kqv = offload_kqv
        self._max_tokens = max_tokens
        self._chat_format = chat_format
        self._verbose = verbose
        self._cache: dict[str, _LocalLlamaChat] = {}

    def __call__(self, model: str):
        if model not in self._cache:
            self._cache[model] = _LocalLlamaChat(
                self._build_client(model),
                max_tokens=self._max_tokens,
            )
        return self._cache[model]

    def _build_client(self, model: str):
        from llama_cpp import Llama

        os.environ.setdefault("GGML_METAL_DEVICES", "")
        model_path = self._resolve_model_path(model)
        return Llama(
            model_path=str(model_path),
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            offload_kqv=self._offload_kqv,
            chat_format=self._chat_format,
            verbose=self._verbose,
        )

    def _resolve_model_path(self, model: str) -> Path:
        if model.endswith(".gguf"):
            return Path(model)
        if model not in _MODEL_MAP:
            raise KeyError(f"Unknown local llama.cpp model alias: {model}")
        return self._model_dir / _MODEL_MAP[model]


def _to_chat_message(msg: BaseMessage) -> dict[str, str]:
    role = "user"
    msg_type = getattr(msg, "type", "")
    if msg_type == "system":
        role = "system"
    elif msg_type == "ai":
        role = "assistant"

    content = msg.content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "".join(text_parts)
    return {"role": role, "content": str(content)}
