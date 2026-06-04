"""Outcome-signal extraction for continuous evolution.

An `OutcomeSignal` answers "did this episode succeed, and how sure are we?"
without necessarily having ground truth.

Phase 0 ships exactly one extractor — `signal_from_reward` — which reads a
Harbor verifier reward (ground truth, confidence 1.0). The labeled
`arena-tracjectories/` corpus and any live Harbor run both expose such a
reward, so this is enough to bootstrap and test the whole pipeline.

The richer *implicit* extractors for the unlabeled production case
(tests-passed, diff-committed, no-retry, user feedback) are Phase 5. They are
declared as `SignalKind` values in `episode.py` but deliberately not guessed at
here — implementing them against trace formats we cannot yet validate would be
assuming. `SignalExtractor` defines the contract they will implement so adding
them later is additive.
"""

from __future__ import annotations

from typing import Protocol

from .episode import Outcome, OutcomeSignal, SignalKind, TaskEpisode


def clamp01(x: float) -> float:
    """Clamp a value into [0.0, 1.0]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def signal_from_reward(
    reward: float,
    *,
    success_threshold: float = 1.0,
    evidence: str | None = None,
) -> OutcomeSignal:
    """Build a ground-truth `OutcomeSignal` from a verifier reward.

    Harbor verifiers emit a reward in [0, 1]. We treat `reward >= success_threshold`
    as success. The default threshold is strict (1.0) because most Harbor verifiers
    are pass/fail; partial credit (0 < reward < 1) is recorded in `value` but still
    counts as a failure so distillation clusters genuinely-failed attempts. The
    threshold is configurable for graded verifiers.

    Confidence is 1.0: a verifier reward is ground truth, not an estimate.
    """
    value = clamp01(reward)
    outcome = Outcome.SUCCESS if reward >= success_threshold else Outcome.FAILURE
    if evidence is None:
        evidence = f"verifier reward {reward:g} (threshold {success_threshold:g})"
    return OutcomeSignal(
        kind=SignalKind.VERIFIER_REWARD,
        outcome=outcome,
        value=value,
        confidence=1.0,
        evidence=evidence,
    )


def no_signal(evidence: str = "no outcome signal available") -> OutcomeSignal:
    """A null signal: outcome unknown, zero confidence.

    Used when a reader cannot recover any outcome (e.g. a raw log with no
    verifier and no implicit cues). Such episodes are still useful context but
    must not be treated as labeled.
    """
    return OutcomeSignal(
        kind=SignalKind.NONE,
        outcome=Outcome.UNKNOWN,
        value=0.0,
        confidence=0.0,
        evidence=evidence,
    )


class SignalExtractor(Protocol):
    """Contract for implicit/surrogate signal extractors (Phase 3 / Phase 5).

    An extractor inspects an episode (and any side-channel evidence already
    attached to it) and returns an `OutcomeSignal`, or `None` if it has nothing
    to say. Multiple extractors will be combined by the gate in later phases;
    the contract is defined now so that work is purely additive.
    """

    kind: SignalKind

    def extract(self, episode: TaskEpisode) -> OutcomeSignal | None:  # pragma: no cover - protocol
        ...
