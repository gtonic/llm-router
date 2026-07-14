"""Tests for routing/base.py - RoutingResult, PolicyBase."""

from __future__ import annotations

from abc import ABC

import pytest

from llm_router.routing.base import PolicyBase, RoutingResult


class TestRoutingResult:
    def test_minimal(self):
        r = RoutingResult(model_id="test-model", strategy="test")
        assert r.model_id == "test-model"
        assert r.strategy == "test"
        assert r.policy_matched is None
        assert r.metadata == {}

    def test_full(self):
        r = RoutingResult(
            model_id="gpt-4",
            strategy="policy",
            policy_matched="rule-1",
            metadata={"key": "val"},
        )
        assert r.policy_matched == "rule-1"
        assert r.metadata["key"] == "val"

    def test_equality(self):
        a = RoutingResult(model_id="m", strategy="s")
        b = RoutingResult(model_id="m", strategy="s")
        assert a == b


class TestPolicyBase:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            PolicyBase()

    def test_is_subclass(self):
        assert issubclass(PolicyBase, ABC)

    def test_route_is_abstract(self):
        abstract_methods = getattr(PolicyBase, "__abstractmethods__", set())
        assert "route" in abstract_methods
