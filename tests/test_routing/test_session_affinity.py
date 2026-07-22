"""Tests for the session affinity store."""

from __future__ import annotations

from llm_router.routing.session_affinity import SessionAffinityStore


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_set_and_get():
    s = SessionAffinityStore(ttl=100.0, now=_Clock())
    assert s.get("a") is None
    s.set("a", "model-x")
    assert s.get("a") == "model-x"


def test_entry_expires_after_ttl():
    clock = _Clock()
    s = SessionAffinityStore(ttl=100.0, now=clock)
    s.set("a", "model-x")
    clock.advance(99.0)
    assert s.get("a") == "model-x"
    clock.advance(2.0)  # now past TTL
    assert s.get("a") is None
    assert len(s) == 0  # expired entry is dropped on read


def test_get_refreshes_lru_but_not_ttl():
    clock = _Clock()
    s = SessionAffinityStore(ttl=100.0, max_entries=2, now=clock)
    s.set("a", "m1")
    s.set("b", "m2")
    s.get("a")  # a is now most-recently-used
    s.set("c", "m3")  # evicts least-recently-used (b)
    assert s.get("a") == "m1"
    assert s.get("b") is None
    assert s.get("c") == "m3"


def test_lru_eviction_when_full():
    s = SessionAffinityStore(ttl=100.0, max_entries=2, now=_Clock())
    s.set("a", "m1")
    s.set("b", "m2")
    s.set("c", "m3")  # over capacity → oldest (a) evicted
    assert len(s) == 2
    assert s.get("a") is None
    assert s.get("b") == "m2"
    assert s.get("c") == "m3"


def test_set_refreshes_ttl():
    clock = _Clock()
    s = SessionAffinityStore(ttl=100.0, now=clock)
    s.set("a", "m1")
    clock.advance(90.0)
    s.set("a", "m1")  # refresh
    clock.advance(90.0)  # 180 total, but only 90 since refresh
    assert s.get("a") == "m1"
