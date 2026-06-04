"""Tests for the continuous-evolution episode model."""

from __future__ import annotations

from src.continuous.episode import (
    ActionStep,
    Outcome,
    OutcomeSignal,
    SignalKind,
    TaskEpisode,
    ToolCall,
)


class TestTaskEpisode:
    def test_minimal_episode_defaults(self):
        ep = TaskEpisode(episode_id="e1", source="harbor")
        assert ep.episode_id == "e1"
        assert ep.source == "harbor"
        assert ep.task_text == ""
        assert ep.actions == []
        assert ep.outcome is Outcome.UNKNOWN
        assert ep.signal is None
        assert ep.skills_active == []
        assert ep.cost_usd == 0.0
        assert ep.is_success is False
        assert ep.is_failure is False

    def test_success_and_failure_helpers(self):
        ok = TaskEpisode(episode_id="e", source="s", outcome=Outcome.SUCCESS)
        bad = TaskEpisode(episode_id="e", source="s", outcome=Outcome.FAILURE)
        assert ok.is_success and not ok.is_failure
        assert bad.is_failure and not bad.is_success

    def test_round_trips_through_json(self):
        ep = TaskEpisode(
            episode_id="e1",
            source="harbor",
            task_text="do thing",
            actions=[
                ActionStep(
                    step_id=1,
                    message="m",
                    reasoning="r",
                    tool_calls=[ToolCall(tool_call_id="c1", function_name="bash")],
                )
            ],
            signal=OutcomeSignal(kind=SignalKind.VERIFIER_REWARD, outcome=Outcome.SUCCESS, value=1.0),
            outcome=Outcome.SUCCESS,
        )
        restored = TaskEpisode.model_validate_json(ep.model_dump_json())
        assert restored == ep
        assert restored.actions[0].tool_calls[0].function_name == "bash"
        assert restored.signal.kind is SignalKind.VERIFIER_REWARD


class TestActionStep:
    def test_defaults(self):
        step = ActionStep()
        assert step.source == "agent"
        assert step.message == ""
        assert step.tool_calls == []

    def test_preserves_non_agent_source(self):
        step = ActionStep(source="environment", message="observation")
        assert step.source == "environment"


class TestToolCall:
    def test_observation_optional(self):
        tc = ToolCall(function_name="bash", arguments={"command": "ls"})
        assert tc.observation is None
        assert tc.arguments["command"] == "ls"
