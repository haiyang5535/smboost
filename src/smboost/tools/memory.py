"""In-memory key-value store for the C6 real-tool harness.

This is a *session-scoped*, *non-persistent* dict wrapper. It exists so that a
planner→caller→verifier loop can stash intermediate results (e.g. "save the
sorted list under the key `sorted_xs` so a later step can read it back") without
embedding everything in the prompt.

Intentionally tiny — no TTL, no size cap, no eviction. One instance per
HarnessAgent session.
"""
from __future__ import annotations

from typing import Any


class MemoryStore:
    """Simple ephemeral kv store.

    Not thread-safe. Not persistent. Reinstantiate per session.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Write ``value`` at ``key``. Overwrites silently."""
        if not isinstance(key, str):
            raise TypeError(f"memory key must be str, got {type(key).__name__}")
        self._store[key] = value

    def get(self, key: str) -> Any | None:
        """Return the value at ``key``, or ``None`` if absent."""
        return self._store.get(key)

    def list_keys(self) -> list[str]:
        """Return the current keys in insertion order."""
        return list(self._store.keys())

    def clear(self) -> None:
        """Drop all entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store


__all__ = ["MemoryStore"]
