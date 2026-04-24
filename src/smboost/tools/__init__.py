"""smboost.tools — runtime tools for the C6 real-tool harness.

Exports:
    PythonSandbox: subprocess-isolated Python exec with timeout + best-effort
                   memory limit. Stateless; one call per `.run(code)`.
    MemoryStore:   in-memory kv store for intermediate planner/caller state.
    Tool:          lightweight protocol any callable tool-like object should
                   satisfy, so the C6 graph can dispatch without isinstance.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from smboost.tools.memory import MemoryStore
from smboost.tools.python_sandbox import PythonSandbox, SandboxResult


@runtime_checkable
class Tool(Protocol):
    """Minimal duck-typed contract any tool may satisfy.

    The C6 harness does not strictly *require* this Protocol — it dispatches
    on action name, not isinstance — but any future plug-in tool that exposes
    a ``name`` and ``__call__`` will be recognisable via ``isinstance(x, Tool)``
    thanks to ``@runtime_checkable``.
    """

    name: str

    def __call__(self, payload: Any) -> dict[str, Any]:
        ...


__all__ = [
    "MemoryStore",
    "PythonSandbox",
    "SandboxResult",
    "Tool",
]
