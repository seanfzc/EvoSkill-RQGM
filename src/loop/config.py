"""Configuration for the self-improving loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


EvolutionMode = Literal["prompt_only", "skill_only"]
SelectionStrategy = Literal["best", "random", "round_robin"]


@dataclass
class LoopConfig:
    """Configuration parameters for SelfImprovingLoop.

    Attributes:
        max_iterations: Maximum number of improvement iterations.
        frontier_size: Number of top-performing programs to keep.
        no_improvement_limit: Stop early after this many iterations without improvement.
        tolerance: Tolerance for answer matching (0.0 = exact match).
        concurrency: Number of concurrent evaluations.
        evolution_mode: Which dimension to evolve ("prompt_only" or "skill_only").
        selection_strategy: Parent selection from frontier — "best" (greedy, default),
            "random" (uniform random), or "round_robin" (cycle through ranked members).
        reset_feedback: Whether to reset feedback_history.md on fresh loop run.
        cache_enabled: Whether to enable run caching.
        cache_dir: Directory for cache storage.
        cache_store_messages: Whether to store full message history in cache.
    """

    max_iterations: int = 5
    frontier_size: int = 3
    no_improvement_limit: int = 5
    tolerance: float = 0.0
    concurrency: int = 4

    # Evolution mode: which dimension to optimize
    evolution_mode: EvolutionMode = "skill_only"

    # Parent selection strategy: how to pick the next parent from the frontier
    selection_strategy: SelectionStrategy = "best"

    # Multi-sample failure analysis: test this many samples before proposing
    # Helps identify patterns rather than overfitting to single failures
    failure_sample_count: int = 3

    # Category-aware sampling: number of categories to sample per batch
    # (capped by actual number of categories and failure_sample_count)
    categories_per_batch: int = 3

    # Feedback configuration
    reset_feedback: bool = True

    # Continue mode: False = start fresh (reset iteration numbering),
    # True = continue from existing frontier/branch
    continue_mode: bool = False

    # Cache configuration
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".cache/runs"))
    cache_store_messages: bool = False

    # Proposer resilience: adaptive truncation on context limit/timeout
    proposer_max_truncation_level: int = 2  # Max truncation level (0=full, 1=moderate, 2=aggressive)
    proposer_single_failure_fallback: bool = True  # Try single shortest failure if all levels fail
    consecutive_proposer_failures_limit: int = 5  # Stop after N consecutive proposer failures

    # Multi-sample per category: collect N samples per category before proposing
    samples_per_category: int = 2  # Helps identify patterns within categories


# ============================================================================
# RQGM (Red Queen Gödel Machine) — co-evolution extension
# ============================================================================


@dataclass
class RQGMConfig:
    """
    Top-level configuration for the RQGM co-evolution extension to EvoSkill.

    Pass an instance of this to ``SelfImprovingLoop.__init__`` alongside the
    existing ``LoopConfig``.  When ``enabled=False`` (the default) the runner
    skips all epoch logic and behaves identically to unmodified EvoSkill.

    This is deliberately a *separate* dataclass from ``LoopConfig`` so that
    the upstream ``LoopConfig`` can be updated without merge conflicts.

    Attributes
    ----------
    enabled : bool
        Master feature flag.  When False, ``EpochManager`` is never
        instantiated and no epoch boundary checks are performed.
        Set to True only after Phase 3 of the injection plan is merged and
        tested.  Default: False (safe for existing users).
    epoch_config : EpochConfig
        Per-epoch behaviour (size, improvement threshold, mutation params).
        Imported from ``src.loop.epoch``.
    evaluator_epsilon : float
        The ε in RQGM's ε-best-belief promotion criterion
        (BB_ε = I⁻¹_ε(1 + S_gt, 1 + F_gt)).
        Used when computing whether a challenger evaluator statistically
        outperforms the incumbent on the ground-truth anchor.
        Range (0, 1).  Default 0.05 (matches RQGM paper §3.5).
    adversarial_high_score_threshold : float
        Minimum *loose* score for a per-question answer to be considered a
        gaming candidate.  Answers with loose_score below this are ignored
        when building the adversarial pool (they were just wrong, not gaming).
        Range [0, 1].  Default 0.85.
    adversarial_strict_threshold : float
        Maximum *strict* (tolerance=0.0) score for a per-question answer to
        be added to the adversarial pool.  Combined with
        ``adversarial_high_score_threshold`` this selects answers that score
        high under loose criteria but low under strict ones.
        Range [0, 1].  Default 0.4.
    selective_erasure_enabled : bool
        Whether to call ``ProgramManager.invalidate_cross_epoch_scores`` when
        a tolerance transition fires.  Disabling avoids the HIGH-risk frontier
        surgery while still allowing tolerance tightening.
        Default: False (enable explicitly in Phase 6 of the injection plan).
    cache_flush_on_boundary : bool
        Whether to flush the ``RunCache`` at every epoch boundary where
        tolerances change.  Required for correctness (see injection plan
        coupling warning #2).  Default: True.
    max_adversarial_examples_per_proposer : int
        Maximum number of adversarial examples injected into a single proposer
        context.  Guards against context-window overflow.  Default: 5.
    checkpoint_schedule : str
        ``"uniform"`` — epochs are all ``epoch_config.epoch_size`` iterations.
        ``"exponential"`` — epochs grow as rho^k (not yet implemented; raises
        NotImplementedError if selected).
    checkpoint_ratio : float
        Growth ratio ρ for exponential checkpoint schedule.  Ignored when
        ``checkpoint_schedule == "uniform"``.  Must be > 1.  Default: 2.0.
    """

    enabled: bool = False  # SAFE DEFAULT: off until explicitly enabled

    # Imported lazily to avoid circular import at module load time.
    # When constructing: epoch_config=EpochConfig(epoch_size=5, ...)
    epoch_config: object = field(
        default_factory=lambda: _default_epoch_config()
    )

    evaluator_epsilon: float = 0.05
    adversarial_high_score_threshold: float = 0.85
    adversarial_strict_threshold: float = 0.4
    selective_erasure_enabled: bool = False   # HIGH RISK: keep off by default
    cache_flush_on_boundary: bool = True
    max_adversarial_examples_per_proposer: int = 5
    checkpoint_schedule: str = "uniform"
    checkpoint_ratio: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.evaluator_epsilon < 1.0:
            raise ValueError(
                f"evaluator_epsilon must be in (0, 1), got {self.evaluator_epsilon}"
            )
        if not 0.0 <= self.adversarial_strict_threshold <= 1.0:
            raise ValueError(
                f"adversarial_strict_threshold must be in [0, 1], "
                f"got {self.adversarial_strict_threshold}"
            )
        if not 0.0 <= self.adversarial_high_score_threshold <= 1.0:
            raise ValueError(
                f"adversarial_high_score_threshold must be in [0, 1], "
                f"got {self.adversarial_high_score_threshold}"
            )
        if self.adversarial_strict_threshold >= self.adversarial_high_score_threshold:
            raise ValueError(
                "adversarial_strict_threshold must be strictly less than "
                "adversarial_high_score_threshold to detect gaming examples. "
                f"Got strict={self.adversarial_strict_threshold}, "
                f"high={self.adversarial_high_score_threshold}"
            )
        if self.checkpoint_schedule == "exponential":
            raise NotImplementedError(
                "Exponential checkpoint schedule is not yet implemented. "
                "Use checkpoint_schedule='uniform' for now."
            )
        if self.checkpoint_schedule not in ("uniform", "exponential"):
            raise ValueError(
                f"checkpoint_schedule must be 'uniform' or 'exponential', "
                f"got {self.checkpoint_schedule!r}"
            )


def _default_epoch_config() -> object:
    """
    Lazily import and return a default ``EpochConfig`` to avoid circular
    imports at module load time.

    Returns
    -------
    EpochConfig
        With all defaults: epoch_size=5, min_improvement_threshold=0.02, etc.
    """
    from src.loop.epoch import EpochConfig  # noqa: PLC0415
    return EpochConfig()
