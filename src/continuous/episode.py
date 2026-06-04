"""Core data model for continuous evolution.

Continuous evolution learns from *real agent usage* rather than from a fixed
labeled benchmark. The unit of usage is a `TaskEpisode`: one attempt by some
agent at some task, with whatever step-by-step detail and outcome signal we
could recover from its trace.

Readers in `collector.py` normalize harness-specific trace formats (Harbor's
ATIF `trajectory.json`, raw goose/opencode logs, generic JSONL) into this one
shape, so every downstream stage (clustering, distillation, gating) is
harness-agnostic.

These models are intentionally permissive: traces in the wild are messy and
schema versions drift (e.g. Harbor's ATIF v1.2 vs v1.6), so every field other
than `episode_id` and `source` has a safe default. Validation should never be
the reason a usable episode is dropped.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Outcome(str, Enum):
    """Whether a task attempt succeeded, judged by an `OutcomeSignal`."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class SignalKind(str, Enum):
    """Where an outcome judgement came from.

    `VERIFIER_REWARD` is the only signal produced in Phase 0 — it reads a
    ground-truth reward from a Harbor verifier. The remaining kinds are the
    implicit/surrogate signals planned for the unlabeled production case
    (Phase 5 / Phase 3) and are defined here so the enum is stable across
    phases.
    """

    VERIFIER_REWARD = "verifier_reward"
    TESTS_PASSED = "tests_passed"
    DIFF_COMMITTED = "diff_committed"
    CI_GREEN = "ci_green"
    USER_ACCEPTED = "user_accepted"
    USER_THUMBS = "user_thumbs"
    NO_RETRY = "no_retry"
    SURROGATE_VERIFIED = "surrogate_verified"
    NONE = "none"


class OutcomeSignal(BaseModel):
    """How an episode's outcome was determined, with a confidence.

    `value` is the normalized strength of the signal in [0, 1] (e.g. a raw
    verifier reward), while `outcome` is the discrete judgement derived from it.
    `confidence` lets downstream gating weight a ground-truth verifier (1.0)
    above a noisy implicit signal (e.g. 0.5 for "the agent didn't retry").
    """

    kind: SignalKind = Field(description="Which extractor produced this signal")
    outcome: Outcome = Field(description="Discrete success/failure/unknown judgement")
    value: float = Field(default=0.0, description="Normalized signal strength in [0, 1]")
    confidence: float = Field(default=1.0, description="Trust in this signal in [0, 1]")
    evidence: str = Field(default="", description="Human-readable reason for the judgement")


class ToolCall(BaseModel):
    """A single tool invocation within a step, with its observation if known.

    `observation` is the tool's result, linked back from the step's observation
    block via the matching call id. It may be absent when a trace records the
    call but not (yet) its result.
    """

    tool_call_id: str | None = Field(default=None, description="Provider call id")
    function_name: str | None = Field(default=None, description="Tool/function name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Call arguments")
    observation: str | None = Field(default=None, description="Tool result content")


class ActionStep(BaseModel):
    """One step of an episode: a message, reasoning, and/or tool calls.

    Mirrors a single entry in an ATIF `steps[]` array but is not tied to any
    schema version. `source` is preserved verbatim ("agent", "user", "tool",
    "system", "environment", ...) so we never assume every step is the agent.
    """

    step_id: int | None = Field(default=None, description="Ordinal step id")
    source: str = Field(default="agent", description="Who produced this step")
    message: str = Field(default="", description="Visible message text")
    reasoning: str = Field(default="", description="Hidden/thinking content")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls in this step")


class TaskEpisode(BaseModel):
    """One agent attempt at one task, normalized across harnesses.

    This is the atomic unit continuous evolution consumes. `episode_id` must be
    stable and unique per attempt so the watermark cursor can dedup across ticks.
    """

    episode_id: str = Field(description="Stable unique id for this attempt")
    source: str = Field(description="Reader that produced it: harbor | goose | jsonl | claude_code")

    task_id: str | None = Field(default=None, description="Task identifier, if known")
    task_text: str = Field(default="", description="The task/instruction the agent worked on")
    actions: list[ActionStep] = Field(default_factory=list, description="Step-by-step trace")
    final_output: str = Field(default="", description="Agent's final textual output, best-effort")

    outcome: Outcome = Field(default=Outcome.UNKNOWN, description="Resolved outcome")
    signal: OutcomeSignal | None = Field(default=None, description="How the outcome was judged")

    skills_active: list[str] = Field(
        default_factory=list, description="Names of live skills loaded during the attempt"
    )

    # Provenance / metrics — useful for credit assignment, cost ceilings, and analysis.
    agent_name: str | None = Field(default=None, description="Harness/agent name")
    model_name: str | None = Field(default=None, description="Model identifier")
    prompt_tokens: int = Field(default=0, description="Total prompt tokens")
    completion_tokens: int = Field(default=0, description="Total completion tokens")
    cost_usd: float = Field(default=0.0, description="Total cost in USD if reported")
    num_steps: int = Field(default=0, description="Number of steps in the trace")
    timestamp: str | None = Field(default=None, description="When the attempt ran, if known")
    raw_path: str | None = Field(default=None, description="Source path, for provenance")
    extra: dict[str, Any] = Field(default_factory=dict, description="Reader-specific extras")

    @property
    def is_success(self) -> bool:
        return self.outcome is Outcome.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.outcome is Outcome.FAILURE
