"""Unit tests for MemoryStore."""
from __future__ import annotations

import pytest

from smboost.tools import MemoryStore


def test_set_and_get_round_trip():
    m = MemoryStore()
    m.set("foo", 42)
    assert m.get("foo") == 42


def test_get_missing_returns_none():
    m = MemoryStore()
    assert m.get("nope") is None


def test_list_keys_empty_is_empty_list():
    m = MemoryStore()
    assert m.list_keys() == []


def test_list_keys_in_insertion_order():
    m = MemoryStore()
    m.set("a", 1)
    m.set("b", 2)
    m.set("c", 3)
    assert m.list_keys() == ["a", "b", "c"]


def test_set_overwrites_existing_key():
    m = MemoryStore()
    m.set("k", "first")
    m.set("k", "second")
    assert m.get("k") == "second"
    assert m.list_keys() == ["k"]


def test_clear_drops_all_entries():
    m = MemoryStore()
    m.set("a", 1)
    m.set("b", 2)
    m.clear()
    assert m.list_keys() == []
    assert m.get("a") is None


def test_len_tracks_entries():
    m = MemoryStore()
    assert len(m) == 0
    m.set("x", 1)
    assert len(m) == 1
    m.set("y", 2)
    assert len(m) == 2


def test_contains_operator():
    m = MemoryStore()
    m.set("here", 1)
    assert "here" in m
    assert "not_here" not in m


def test_non_string_key_rejected():
    m = MemoryStore()
    with pytest.raises(TypeError):
        m.set(123, "value")  # type: ignore[arg-type]


def test_stores_arbitrary_python_objects():
    m = MemoryStore()
    obj = {"nested": [1, 2, 3], "t": (4, 5)}
    m.set("blob", obj)
    assert m.get("blob") == obj


def test_independent_instances_do_not_share_state():
    a = MemoryStore()
    b = MemoryStore()
    a.set("x", 1)
    assert b.get("x") is None
    assert "x" not in b
