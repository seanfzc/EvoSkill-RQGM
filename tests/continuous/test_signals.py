"""Tests for outcome-signal extraction."""

from __future__ import annotations

import pytest

from src.continuous.episode import Outcome, SignalKind
from src.continuous.signals import clamp01, no_signal, signal_from_reward


class TestClamp01:
    @pytest.mark.parametrize(
        "value,expected",
        [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0)],
    )
    def test_clamps(self, value, expected):
        assert clamp01(value) == expected


class TestSignalFromReward:
    def test_full_reward_is_success(self):
        sig = signal_from_reward(1.0)
        assert sig.outcome is Outcome.SUCCESS
        assert sig.value == 1.0
        assert sig.confidence == 1.0
        assert sig.kind is SignalKind.VERIFIER_REWARD

    def test_zero_reward_is_failure(self):
        sig = signal_from_reward(0.0)
        assert sig.outcome is Outcome.FAILURE
        assert sig.value == 0.0

    def test_partial_reward_is_failure_by_default_but_value_preserved(self):
        sig = signal_from_reward(0.5)
        assert sig.outcome is Outcome.FAILURE
        assert sig.value == 0.5

    def test_partial_reward_passes_with_lower_threshold(self):
        sig = signal_from_reward(0.5, success_threshold=0.5)
        assert sig.outcome is Outcome.SUCCESS

    def test_reward_above_one_clamps_value(self):
        sig = signal_from_reward(3.0)
        assert sig.value == 1.0
        assert sig.outcome is Outcome.SUCCESS

    def test_custom_evidence(self):
        sig = signal_from_reward(1.0, evidence="custom")
        assert sig.evidence == "custom"

    def test_default_evidence_mentions_reward(self):
        sig = signal_from_reward(0.0)
        assert "reward" in sig.evidence


class TestNoSignal:
    def test_unknown_zero_confidence(self):
        sig = no_signal()
        assert sig.outcome is Outcome.UNKNOWN
        assert sig.confidence == 0.0
        assert sig.kind is SignalKind.NONE
