"""
src/evaluation/utility_evolution.py
=====================================
RQGM utility-evolution layer for EvoSkill.

Provides the mechanisms that *change* the evaluation criterion at epoch
boundaries — as opposed to ``evaluate.py`` which *runs* a fixed criterion.

Key responsibilities
--------------------
1. ``UtilityEvolution``   — knows how to mutate the active evaluator config
                            (tolerance schedule, adversarial weights).
2. ``evolve_tolerances``  — pure function: given a performance distribution
                            and the current schedule, returns a new schedule.
3. ``adversarial_score``  — augmented per-answer score that penalises answers
                            scoring well under loose tolerances but badly under
                            strict matching (RQGM §5.4 adversarial objective).

Relationship to existing code
------------------------------
  - ``score_answer`` in ``src/evaluation/reward.py:439-444`` is the existing
    per-answer scorer.  ``adversarial_score`` wraps it.
  - ``_score_multi_tolerance`` in ``src/loop/runner.py:28-46`` will be
    extended to accept a ``tolerances`` argument sourced from the active
    ``EpochManager.current_tolerances``.
  - ``UtilityEvolution`` is instantiated by ``EpochManager`` (it is *not*
    instantiated directly by the runner).
  - ``EvolutionResult`` is returned to ``EpochManager.evaluate_epoch_boundary``
    and then propagated via ``EpochTransition`` to the runner.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.loop.epoch import (
        AdversarialExample,
        EpochConfig,
        EpochPerformanceSnapshot,
        UtilityMutationParams,
    )

# Re-use existing per-answer scorer so we don't duplicate logic.
from src.evaluation.reward import score_answer  # type: ignore[import]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Score threshold above which we call an answer "passed" for distribution
#: analysis purposes.
PASS_THRESHOLD: float = 0.5

#: Default adversarial penalty weight.  Final adversarial score =
#: base_score - ADV_PENALTY_WEIGHT * exploitation_signal.
ADV_PENALTY_WEIGHT: float = 0.3

#: Minimum gap between consecutive tolerance levels to preserve meaningful
#: granularity.  Prevents clustering all remaining levels near 0.
MIN_TOLERANCE_GAP: float = 1e-4

#: Number of consecutive epochs without exploitation before decaying the
#: adversarial weight back toward zero.
ADV_WEIGHT_DECAY_EPOCHS: int = 3

#: Maximum adversarial weight (hard cap).
ADV_WEIGHT_MAX: float = 0.8

#: Minimum adversarial weight (floor — never fully disable).
ADV_WEIGHT_MIN: float = 0.05

#: Weight increment per exploitation epoch.
ADV_WEIGHT_INCREMENT: float = 0.1

#: Weight decay per non-exploitation epoch.
ADV_WEIGHT_DECAY: float = 0.05


# ---------------------------------------------------------------------------
# EvolutionResult
# ---------------------------------------------------------------------------

@dataclass
class EvolutionResult:
    """
    Returned by ``UtilityEvolution.apply_mutations``.

    Attributes
    ----------
    new_tolerances : list[float]
        The tolerance schedule to freeze for the next epoch.
    tolerances_changed : bool
        True if ``new_tolerances`` differs from the input schedule.
        When True the runner must flush the ``RunCache`` and trigger
        selective erasure (RQGM §3.4).
    adversarial_weight : float
        The ``adversarial_score`` penalty weight to use next epoch.
        Carried forward even when no other mutation fires.
    mutation_log : list[str]
        Human-readable record of every mutation decision (for
        ``feedback_history.md`` and debugging).
    """

    new_tolerances: list[float]
    tolerances_changed: bool
    adversarial_weight: float = ADV_PENALTY_WEIGHT
    mutation_log: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ScoreDistribution — lightweight stats over a set of per-answer scores
# ---------------------------------------------------------------------------

@dataclass
class ScoreDistribution:
    """
    Summarises a distribution of per-answer scores produced by the base agent
    on a sample of training or validation questions.

    Used by ``evolve_tolerances`` and ``UtilityEvolution`` to decide whether
    the score distribution suggests exploitation.

    Attributes
    ----------
    scores_at_strict : list[float]
        Per-question scores under tolerance 0.0 (exact match).
    scores_at_loose : list[float]
        Per-question scores under the loosest active tolerance.
    questions : list[str]
        The questions corresponding to each score pair (same order).
        Stored so we can construct ``AdversarialExample`` objects for the ones
        that look like gaming.
    """

    scores_at_strict: list[float]
    scores_at_loose: list[float]
    questions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.scores_at_strict) != len(self.scores_at_loose):
            raise ValueError(
                "scores_at_strict and scores_at_loose must have the same length, "
                f"got {len(self.scores_at_strict)} vs {len(self.scores_at_loose)}"
            )

    @property
    def mean_strict(self) -> float:
        """Mean score under exact matching."""
        return statistics.mean(self.scores_at_strict) if self.scores_at_strict else 0.0

    @property
    def mean_loose(self) -> float:
        """Mean score under the loosest active tolerance."""
        return statistics.mean(self.scores_at_loose) if self.scores_at_loose else 0.0

    @property
    def hack_ratio(self) -> float | None:
        """
        mean_strict / mean_loose.  None if mean_loose is 0.

        A value below ``EpochConfig.exploitation_hack_ratio_threshold``
        indicates the agent is likely gaming the loose criterion.
        """
        if self.mean_loose == 0.0:
            return None
        return self.mean_strict / self.mean_loose

    def get_gaming_indices(
        self,
        hack_ratio_threshold: float,
        min_loose_score: float = 0.5,
    ) -> list[int]:
        """
        Return the indices of questions where the agent appears to be gaming.

        A question is flagged if its per-question hack ratio is below the
        threshold *and* the loose score is high enough to actually matter
        (avoids flagging questions the agent just got wrong in both modes).

        Parameters
        ----------
        hack_ratio_threshold : float
            Per-question ratio threshold (mirrors
            ``EpochConfig.exploitation_hack_ratio_threshold``).
        min_loose_score : float
            Ignore questions where loose_score < this value.

        Returns
        -------
        list[int]
            Indices into ``scores_at_strict`` / ``scores_at_loose``.
        """
        gaming: list[int] = []
        for i in range(len(self.scores_at_strict)):
            loose = self.scores_at_loose[i]
            strict = self.scores_at_strict[i]

            # Skip questions the agent just got wrong — not gaming, just wrong.
            if loose < min_loose_score:
                continue

            # Compute per-question hack ratio.
            if loose > 0.0:
                ratio = strict / loose
                if ratio < hack_ratio_threshold:
                    gaming.append(i)

        return gaming


# ---------------------------------------------------------------------------
# evolve_tolerances  (pure function)
# ---------------------------------------------------------------------------

def evolve_tolerances(
    current_tolerances: list[float],
    distribution: ScoreDistribution,
    mutation_params: "UtilityMutationParams",
    hack_ratio_threshold: float,
    epoch_improvement: float,
    min_improvement_threshold: float,
) -> tuple[list[float], list[str]]:
    """
    Derive the tolerance schedule for the next epoch.

    This is a **pure function** — no side effects, no mutation of inputs.
    It can be unit-tested without any EvoSkill infrastructure.

    Parameters
    ----------
    current_tolerances : list[float]
        The tolerance schedule currently in use (ascending order).
    distribution : ScoreDistribution
        Per-answer score distribution from the most recent epoch.
    mutation_params : UtilityMutationParams
        Which mutations are allowed (from ``EpochConfig``).
    hack_ratio_threshold : float
        Threshold below which exploitation is declared.
    epoch_improvement : float
        Signed score improvement across the epoch (end − start).
    min_improvement_threshold : float
        Minimum improvement that constitutes genuine progress.

    Returns
    -------
    new_tolerances : list[float]
        The updated tolerance schedule (sorted ascending, de-duplicated,
        guaranteed to have ≥ ``MIN_TOLERANCE_LEVELS`` entries).
    mutation_log : list[str]
        Human-readable log of every decision taken.

    Algorithm (RQGM §3.4, §3.5)
    ----------------------------
    1. Compute hack_ratio from ``distribution``.
    2. is_exploiting = hack_ratio < hack_ratio_threshold (and ratio is not None).
    3. is_stagnant   = epoch_improvement < min_improvement_threshold.
    4. If is_exploiting and allow_tolerance_tightening:
         Drop the max(current_tolerances) level, up to
         ``max_tolerance_drops_per_epoch`` times, down to
         ``MIN_TOLERANCE_LEVELS`` entries.
    5. If not is_exploiting and not is_stagnant and allow_tolerance_relaxation
       and len(current_tolerances) < len(DEFAULT_TOLERANCES):
         Restore the next-loosest level from ``DEFAULT_TOLERANCES`` that is
         not already present.
    6. Deduplicate and sort.
    7. Validate gaps ≥ MIN_TOLERANCE_GAP.
    8. Return.
    """
    from src.loop.epoch import DEFAULT_TOLERANCES, MIN_TOLERANCE_LEVELS

    mutation_log: list[str] = []
    new_tolerances = list(current_tolerances)

    # --- Step 1-2: compute hack ratio and classify ---------------------------
    hack_ratio = distribution.hack_ratio
    is_exploiting = (
        hack_ratio is not None
        and hack_ratio < hack_ratio_threshold
    )
    is_stagnant = epoch_improvement < min_improvement_threshold

    mutation_log.append(
        f"evolve_tolerances: hack_ratio={hack_ratio:.4f}" if hack_ratio is not None
        else "evolve_tolerances: hack_ratio=None (no loose scores)"
    )
    mutation_log.append(
        f"  is_exploiting={is_exploiting}, is_stagnant={is_stagnant}, "
        f"improvement={epoch_improvement:.4f}"
    )

    # --- Step 4: tighten if exploiting --------------------------------------
    if is_exploiting and mutation_params.allow_tolerance_tightening:
        drops = 0
        max_drops = mutation_params.max_tolerance_drops_per_epoch
        while (
            drops < max_drops
            and len(new_tolerances) > MIN_TOLERANCE_LEVELS
            and len(new_tolerances) > 1
        ):
            removed = new_tolerances.pop()  # remove largest (loosest)
            drops += 1
            mutation_log.append(
                f"  TIGHTEN: dropped tolerance level {removed:.4f} "
                f"(exploitation detected, hack_ratio={hack_ratio:.4f})"
            )

    # --- Step 5: relax if genuinely improving -------------------------------
    elif (
        not is_exploiting
        and not is_stagnant
        and mutation_params.allow_tolerance_relaxation
    ):
        if len(new_tolerances) < len(DEFAULT_TOLERANCES):
            # Find the next tolerance level from DEFAULT that isn't already present.
            for tol in DEFAULT_TOLERANCES:
                if tol not in new_tolerances:
                    new_tolerances.append(tol)
                    new_tolerances.sort()
                    mutation_log.append(
                        f"  RELAX: restored tolerance level {tol:.4f} "
                        f"(genuine improvement, no exploitation)"
                    )
                    break

    # --- Step 6: deduplicate and sort ---------------------------------------
    new_tolerances = sorted(set(new_tolerances))

    # --- Step 7: validate gaps -----------------------------------------------
    for i in range(len(new_tolerances) - 1):
        gap = new_tolerances[i + 1] - new_tolerances[i]
        if gap < MIN_TOLERANCE_GAP:
            mutation_log.append(
                f"  WARNING: tolerance gap {gap:.6f} < MIN_TOLERANCE_GAP "
                f"({MIN_TOLERANCE_GAP}) between {new_tolerances[i]:.4f} and "
                f"{new_tolerances[i + 1]:.4f}"
            )

    # --- Step 8: return -----------------------------------------------------
    if new_tolerances == current_tolerances:
        mutation_log.append("  RESULT: no change to tolerance schedule")

    return new_tolerances, mutation_log


# ---------------------------------------------------------------------------
# adversarial_score  (module-level function)
# ---------------------------------------------------------------------------

def adversarial_score(
    question: str,
    predicted: str,
    ground_truth: str,
    current_tolerances: list[float],
    adversarial_pool: list["AdversarialExample"],
    adversarial_weight: float = ADV_PENALTY_WEIGHT,
) -> float:
    """
    Augmented scorer that penalises answers resembling known gaming patterns.

    This is an *additional* scoring layer on top of the existing
    ``score_answer`` / multi-tolerance pipeline.  It does **not** replace
    them; it is applied as a post-processing penalty and is only active when
    ``trigger_adversarial_injection`` is True in the current epoch's
    ``EpochTransition``.

    Parameters
    ----------
    question : str
        The question being scored.
    predicted : str
        The agent's answer.
    ground_truth : str
        The reference answer.
    current_tolerances : list[float]
        The active tolerance schedule (epoch-local, from
        ``EpochManager.current_tolerances``).
    adversarial_pool : list[AdversarialExample]
        Examples collected during the previous epoch where the agent gamed the
        loose scorer.  Used to compute the exploitation signal.
    adversarial_weight : float
        Weight of the adversarial penalty term (default ``ADV_PENALTY_WEIGHT``).
        Should be tuned to avoid over-penalising legitimate answers.

    Returns
    -------
    float
        Score in [0, 1].  Computed as:
        ``base_score - adversarial_weight * exploitation_signal``
        clipped to [0, 1].

    Internals
    ---------
    base_score
        Mean of ``score_answer(predicted, ground_truth, t)`` across all
        ``t`` in ``current_tolerances`` — identical to the existing
        ``_score_multi_tolerance`` behaviour (runner.py:28-46).
    exploitation_signal
        A float in [0, 1] measuring how much the predicted answer looks like
        a gaming pattern.  Computed per-question: if the answer scores well
        under loose tolerance but poorly under strict tolerance, the signal
        is the gap between them.  This directly measures the gaming condition
        without requiring NLP similarity over the adversarial pool.

    Notes
    -----
    - This function is called by the runner in place of the plain
      ``_score_multi_tolerance`` call *only* when
      ``EpochTransition.trigger_adversarial_injection`` is True.
    - The penalty does **not** affect the frontier qualification score
      (that would cause selective erasure issues).  It is applied only during
      the *training-set failure detection pass* (runner.py:285-303) to bias
      the proposer away from gaming patterns.  See injection plan item #12.
    - If ``adversarial_pool`` is empty the function degrades gracefully to the
      plain multi-tolerance score.
    """
    # --- Base score (multi-tolerance mean) -----------------------------------
    base_score = _compute_base_score(predicted, ground_truth, current_tolerances)

    # --- Early exit if no adversarial context --------------------------------
    if not adversarial_pool:
        return base_score

    # --- Exploitation signal -------------------------------------------------
    exploitation_signal = _compute_exploitation_signal(
        question=question,
        predicted=predicted,
        ground_truth=ground_truth,
        current_tolerances=current_tolerances,
        adversarial_pool=adversarial_pool,
    )

    # --- Final score ---------------------------------------------------------
    penalised = base_score - adversarial_weight * exploitation_signal
    return max(0.0, min(1.0, penalised))


def _compute_base_score(
    predicted: str,
    ground_truth: str,
    tolerances: list[float],
) -> float:
    """
    Mean of ``score_answer`` across all tolerance levels.

    Mirrors ``_score_multi_tolerance`` in runner.py:28-46 but accepts an
    arbitrary tolerance list so it can be called with epoch-local schedules.

    Parameters
    ----------
    predicted : str
    ground_truth : str
    tolerances : list[float]

    Returns
    -------
    float
        Mean score in [0, 1].
    """
    if not tolerances:
        return 0.0
    scores = [score_answer(predicted, ground_truth, t) for t in tolerances]
    return sum(scores) / len(scores)


def _compute_exploitation_signal(
    question: str,
    predicted: str,
    ground_truth: str,
    current_tolerances: list[float],
    adversarial_pool: list["AdversarialExample"],
) -> float:
    """
    Compute how much the predicted answer resembles a gaming pattern.

    Returns a float in [0, 1] where 1.0 = clear gaming, 0.0 = no signal.

    Algorithm (approach b from the docstring)
    ------------------------------------------
    1. Compute per-question strict_score = score_answer(predicted, gt, 0.0).
    2. Compute per-question loose_score  = score_answer(predicted, gt, max(current_tolerances)).
    3. If loose_score > PASS_THRESHOLD and strict_score < PASS_THRESHOLD:
         exploitation_signal = loose_score - strict_score  (in [0, 1])
       Else:
         exploitation_signal = 0.0

    This is cheap (two score_answer calls) and directly tests the gaming
    condition without requiring NLP similarity over the adversarial pool.
    The pool is still accepted as a parameter so future implementations can
    use it for pattern matching.

    Parameters
    ----------
    question : str
    predicted : str
    ground_truth : str
    current_tolerances : list[float]
    adversarial_pool : list[AdversarialExample]

    Returns
    -------
    float
        Exploitation signal in [0, 1].
    """
    if not current_tolerances:
        return 0.0

    # Strict score: exact match (tolerance 0.0).
    strict_score = score_answer(predicted, ground_truth, 0.0)

    # Loose score: loosest active tolerance.
    loose_tol = max(current_tolerances)
    loose_score = score_answer(predicted, ground_truth, loose_tol)

    # Gaming condition: scores well under loose, poorly under strict.
    # This is the signature of an answer that pattern-matches the loose
    # criterion without actually being correct (RQGM §5.4).
    if loose_score >= PASS_THRESHOLD and strict_score < PASS_THRESHOLD:
        # The gap between loose and strict is the exploitation signal.
        # Large gap = strong gaming evidence.
        signal = loose_score - strict_score
        logger.debug(
            "Exploitation signal detected: strict=%.2f, loose=%.2f, signal=%.2f",
            strict_score, loose_score, signal,
        )
        return signal

    return 0.0


# ---------------------------------------------------------------------------
# UtilityEvolution  (class)
# ---------------------------------------------------------------------------

class UtilityEvolution:
    """
    Applies mutations to the evaluator configuration at epoch boundaries.

    Wraps ``evolve_tolerances`` and ``adversarial_score`` into a single
    object that ``EpochManager`` can call without knowing the implementation
    details of each mutation strategy.

    Parameters
    ----------
    epoch_config : EpochConfig
        Provides ``mutation_params``, ``exploitation_hack_ratio_threshold``,
        and ``min_improvement_threshold``.
    initial_adversarial_weight : float
        Starting weight for the adversarial penalty term.

    Attributes
    ----------
    adversarial_weight : float
        Current penalty weight.  May be adjusted by
        ``_update_adversarial_weight`` in future epochs.
    _mutation_history : list[EvolutionResult]
        Log of every mutation decision; useful for debugging and post-run
        analysis.
    _consecutive_no_exploit : int
        Number of consecutive epochs without exploitation detection.
        Used for adversarial weight decay.
    """

    def __init__(
        self,
        epoch_config: "EpochConfig",
        initial_adversarial_weight: float = ADV_PENALTY_WEIGHT,
    ) -> None:
        self.epoch_config = epoch_config
        self.adversarial_weight: float = initial_adversarial_weight
        self._mutation_history: list[EvolutionResult] = []
        self._consecutive_no_exploit: int = 0

    def apply_mutations(
        self,
        current_tolerances: list[float],
        distribution: ScoreDistribution,
        epoch_improvement: float,
        snapshot: "EpochPerformanceSnapshot",
    ) -> EvolutionResult:
        """
        Compute and return the full mutation result for an epoch boundary.

        Called by ``EpochManager._compute_new_tolerances`` (or directly by
        ``evaluate_epoch_boundary`` if wired that way).

        Parameters
        ----------
        current_tolerances : list[float]
            The schedule currently in use.
        distribution : ScoreDistribution
            Per-answer score distribution from the epoch.
        epoch_improvement : float
            Signed score improvement across the epoch (end − start).
        snapshot : EpochPerformanceSnapshot
            The full performance snapshot for the epoch (for logging).

        Returns
        -------
        EvolutionResult
        """
        cfg = self.epoch_config
        mutation_params = cfg.utility_mutation_params

        # --- Tolerance mutation -------------------------------------------
        new_tolerances, tol_log = evolve_tolerances(
            current_tolerances=current_tolerances,
            distribution=distribution,
            mutation_params=mutation_params,
            hack_ratio_threshold=cfg.exploitation_hack_ratio_threshold,
            epoch_improvement=epoch_improvement,
            min_improvement_threshold=cfg.min_improvement_threshold,
        )
        tolerances_changed = new_tolerances != current_tolerances

        # --- Adversarial weight update ------------------------------------
        # Determine if exploitation was detected in this epoch.
        hack_ratio = distribution.hack_ratio
        is_exploiting = (
            hack_ratio is not None
            and hack_ratio < cfg.exploitation_hack_ratio_threshold
        )

        new_weight = self._update_adversarial_weight(
            is_exploiting=is_exploiting,
            tolerances_changed=tolerances_changed,
        )
        self.adversarial_weight = new_weight

        result = EvolutionResult(
            new_tolerances=new_tolerances,
            tolerances_changed=tolerances_changed,
            adversarial_weight=new_weight,
            mutation_log=tol_log,
        )
        self._mutation_history.append(result)
        return result

    def _update_adversarial_weight(
        self,
        is_exploiting: bool,
        tolerances_changed: bool,
    ) -> float:
        """
        Adjust the adversarial penalty weight based on epoch outcomes.

        Strategy (RQGM §5.4 adversarial objective):
          - If exploitation was detected:
              Increase weight (more pressure needed to penalise gaming).
              Reset the no-exploit counter.
          - If no exploitation for ``ADV_WEIGHT_DECAY_EPOCHS`` consecutive epochs:
              Decay weight back toward ``ADV_WEIGHT_MIN`` (agent has overcome
              the gaming pattern).
          - Hard cap at [ADV_WEIGHT_MIN, ADV_WEIGHT_MAX].

        Parameters
        ----------
        is_exploiting : bool
            Whether exploitation was detected in the epoch that just ended.
        tolerances_changed : bool
            Whether the tolerance schedule was mutated this epoch.

        Returns
        -------
        float
            The new adversarial weight to use in the next epoch.
        """
        if is_exploiting:
            # Exploitation detected — increase penalty.
            self._consecutive_no_exploit = 0
            new_weight = min(
                self.adversarial_weight + ADV_WEIGHT_INCREMENT,
                ADV_WEIGHT_MAX,
            )
            logger.info(
                "Adversarial weight increased: %.3f → %.3f (exploitation detected)",
                self.adversarial_weight, new_weight,
            )
            return new_weight

        # No exploitation this epoch.
        self._consecutive_no_exploit += 1

        if self._consecutive_no_exploit >= ADV_WEIGHT_DECAY_EPOCHS:
            # Agent has been clean for several epochs — decay the penalty.
            new_weight = max(
                self.adversarial_weight - ADV_WEIGHT_DECAY,
                ADV_WEIGHT_MIN,
            )
            logger.info(
                "Adversarial weight decayed: %.3f → %.3f "
                "(%d consecutive epochs without exploitation)",
                self.adversarial_weight, new_weight, self._consecutive_no_exploit,
            )
            return new_weight

        # No change — keep current weight.
        return self.adversarial_weight

    def score_with_adversarial(
        self,
        question: str,
        predicted: str,
        ground_truth: str,
        current_tolerances: list[float],
        adversarial_pool: list["AdversarialExample"],
    ) -> float:
        """
        Convenience wrapper around the module-level ``adversarial_score``
        function that uses this instance's ``adversarial_weight``.

        Parameters
        ----------
        question : str
        predicted : str
        ground_truth : str
        current_tolerances : list[float]
        adversarial_pool : list[AdversarialExample]

        Returns
        -------
        float
            Score in [0, 1].
        """
        return adversarial_score(
            question=question,
            predicted=predicted,
            ground_truth=ground_truth,
            current_tolerances=current_tolerances,
            adversarial_pool=adversarial_pool,
            adversarial_weight=self.adversarial_weight,
        )

    @property
    def mutation_count(self) -> int:
        """Number of epoch boundaries where a mutation actually fired."""
        return sum(1 for r in self._mutation_history if r.tolerances_changed)
