"""
src/loop/epoch.py
=================
RQGM (Red Queen Gödel Machine) epoch management for EvoSkill.

Implements controlled utility evolution: the self-improvement loop is divided
into epochs, each with a *frozen* evaluator configuration (tolerance schedule,
adversarial weights).  At epoch boundaries the EpochManager analyses whether
the agent has begun exploiting the current evaluator and, if so, triggers a
utility transition — tightening tolerances, injecting adversarial examples, or
both.

Reference: arXiv 2606.26294, §3 (Controlled Utility Evolution) and §3.5
(Adversarial Objective).

Relationship to existing code
------------------------------
  - ``EpochConfig`` is embedded inside ``RQGMConfig``  (src/loop/config.py).
  - ``EpochManager`` is instantiated once by ``SelfImprovingLoop.__init__``
    (src/loop/runner.py) and queried at the top of each iteration.
  - ``evaluate_epoch_boundary()`` is the public function called by the runner
    at detected boundary points; it returns an ``EpochTransition`` describing
    any actions the runner should take (tolerance update, adversarial flush,
    selective erasure, etc.).
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.loop.config import RQGMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default tolerance schedule (matches existing _score_multi_tolerance in runner.py:28-46).
DEFAULT_TOLERANCES: list[float] = [0.0, 0.001, 0.01, 0.025, 0.05, 0.1]

#: Minimum number of tolerances to keep in the schedule even after aggressive
#: tightening.  Prevents degenerate single-level evaluators.
MIN_TOLERANCE_LEVELS: int = 2


# ---------------------------------------------------------------------------
# EpochConfig
# ---------------------------------------------------------------------------

@dataclass
class EpochConfig:
    """
    Configuration for a single epoch in the RQGM controlled utility evolution.

    An epoch is a contiguous window of loop iterations during which the
    evaluator (tolerance schedule + adversarial weights) is *frozen*.  At the
    end of each epoch the EpochManager may perform a utility transition.

    Attributes
    ----------
    epoch_size : int
        Number of loop iterations per epoch.  At iteration
        ``epoch_start + epoch_size`` the boundary check fires.
        Corresponds to RQGM §3.5 "exponentially-spaced checkpoints" (here
        uniform for simplicity; see ``RQGMConfig.checkpoint_schedule`` for
        exponential variant).
    min_improvement_threshold : float
        If the best-frontier score improvement across the epoch is strictly
        below this value, the epoch is deemed *stagnant* and utility evolution
        is triggered regardless of exploitation detection.
        Range [0, 1].  Default 0.02 (2 pp improvement required per epoch).
    utility_mutation_params : UtilityMutationParams
        Fine-grained control over *what* can change when a utility transition
        fires.  See ``UtilityMutationParams``.
    exploitation_hack_ratio_threshold : float
        Ratio ``score_strict / score_loose`` below which the agent is deemed
        to be gaming the scoring.  For example 0.6 means: if the agent scores
        ≥60% under 10% tolerance but only 36% under exact match, it is likely
        pattern-matching the loose criterion rather than finding correct
        answers.  Reference: RQGM §5.4 (adversarial objective).
    adversarial_pool_min_size : int
        Minimum number of adversarial examples that must accumulate before the
        pool is injected into the proposer context.  Prevents noise from
        near-empty pools early in the run.
    """

    epoch_size: int = 5
    min_improvement_threshold: float = 0.02
    utility_mutation_params: "UtilityMutationParams" = field(
        default_factory=lambda: UtilityMutationParams()
    )
    exploitation_hack_ratio_threshold: float = 0.6
    adversarial_pool_min_size: int = 3

    def __post_init__(self) -> None:
        if self.epoch_size < 1:
            raise ValueError(f"epoch_size must be ≥ 1, got {self.epoch_size}")
        if not 0.0 <= self.min_improvement_threshold <= 1.0:
            raise ValueError(
                f"min_improvement_threshold must be in [0, 1], "
                f"got {self.min_improvement_threshold}"
            )
        if not 0.0 < self.exploitation_hack_ratio_threshold <= 1.0:
            raise ValueError(
                f"exploitation_hack_ratio_threshold must be in (0, 1], "
                f"got {self.exploitation_hack_ratio_threshold}"
            )


@dataclass
class UtilityMutationParams:
    """
    Controls *which* evaluator parameters are allowed to evolve at epoch
    boundaries.  Acts as a feature flag for each mutation class.

    Attributes
    ----------
    allow_tolerance_tightening : bool
        Whether the epoch manager may drop the loosest tolerance level when
        exploitation is detected.  Maps to RQGM utility transition §3.5.
    allow_tolerance_relaxation : bool
        Whether the epoch manager may re-add a tolerance level that was
        previously removed, if the agent shows genuine improvement under
        strict criteria.  Prevents permanent evaluation degradation.
    allow_adversarial_injection : bool
        Whether gaming examples accumulated in ``EpochState.adversarial_pool``
        may be injected into proposer queries (via ``build_proposer_query``).
        Maps to RQGM adversarial objective §5.4.
    allow_failure_threshold_adjustment : bool
        Whether the failure detection threshold (hardcoded 0.8 in
        runner.py:300) may shift in response to distribution changes caused by
        tolerance tightening.  HIGH RISK — enable only after Phase 3 of the
        injection plan is stable.
    max_tolerance_drops_per_epoch : int
        Upper bound on the number of tolerance levels that may be dropped in a
        single epoch boundary.  Prevents too-aggressive tightening from a
        single bad epoch.
    """

    allow_tolerance_tightening: bool = True
    allow_tolerance_relaxation: bool = True
    allow_adversarial_injection: bool = True
    allow_failure_threshold_adjustment: bool = False  # HIGH RISK: off by default
    max_tolerance_drops_per_epoch: int = 1


# ---------------------------------------------------------------------------
# Per-epoch performance snapshot
# ---------------------------------------------------------------------------

@dataclass
class EpochPerformanceSnapshot:
    """
    Captures the performance state of the agent at the *start* and *end* of
    an epoch.  Stored in ``EpochManager.epoch_history`` for trend analysis.

    Attributes
    ----------
    epoch_index : int
        Which epoch this snapshot belongs to (0-indexed).
    start_iteration : int
        Loop iteration number at which this epoch began.
    end_iteration : int
        Loop iteration number at which this epoch ended (exclusive).
    start_best_score : float
        Best frontier score at the start of the epoch (under the epoch's
        tolerance schedule).
    end_best_score : float
        Best frontier score at the end of the epoch.
    score_at_strict_tol : float
        Mean score across val_data under tolerance 0.0 (exact match) at epoch
        end.  Used to compute the hack ratio.
    score_at_loose_tol : float
        Mean score across val_data under the loosest active tolerance at epoch
        end.  Used to compute the hack ratio.
    tolerance_schedule : list[float]
        The tolerance schedule that was *active* during this epoch (frozen
        copy; captures the epoch-local evaluator state).
    adversarial_examples_collected : int
        Number of gaming examples added to the adversarial pool during this
        epoch.
    utility_transition_fired : bool
        Whether a utility transition actually fired at the end of this epoch.
    """

    epoch_index: int
    start_iteration: int
    end_iteration: int
    start_best_score: float
    end_best_score: float
    score_at_strict_tol: float
    score_at_loose_tol: float
    tolerance_schedule: list[float]
    adversarial_examples_collected: int = 0
    utility_transition_fired: bool = False

    @property
    def score_improvement(self) -> float:
        """Signed improvement across the epoch under the epoch's own criterion."""
        return self.end_best_score - self.start_best_score

    @property
    def hack_ratio(self) -> float | None:
        """
        score_strict / score_loose.  Returns None if score_loose is 0 (divide
        by zero guard).  A ratio below ``EpochConfig.exploitation_hack_ratio_threshold``
        indicates the agent is gaming the loose criterion.
        """
        if self.score_at_loose_tol == 0.0:
            return None
        return self.score_at_strict_tol / self.score_at_loose_tol


# ---------------------------------------------------------------------------
# EpochTransition — what the runner should do after a boundary
# ---------------------------------------------------------------------------

class TransitionReason(Enum):
    """Why a utility transition was triggered."""
    EXPLOITATION_DETECTED = auto()   # hack_ratio < threshold
    STAGNATION = auto()              # improvement < min_improvement_threshold
    SCHEDULED = auto()               # every N epochs unconditionally
    NO_TRANSITION = auto()           # boundary checked but no change needed


@dataclass
class EpochTransition:
    """
    Return value of ``evaluate_epoch_boundary()``.  The runner reads this
    struct and applies each action in the order listed in the injection plan.

    Attributes
    ----------
    epoch_index : int
        The epoch index that just *ended* (transition takes effect for epoch
        ``epoch_index + 1``).
    reason : TransitionReason
        Why the transition fired (or why it didn't).
    new_tolerances : list[float]
        The tolerance schedule to use for the *next* epoch.  Unchanged if
        ``reason == NO_TRANSITION``.
    trigger_adversarial_injection : bool
        If True the runner should pass the adversarial pool to
        ``build_proposer_query`` for the remainder of the next epoch.
    trigger_selective_erasure : bool
        If True the runner must call
        ``ProgramManager.invalidate_cross_epoch_scores(epoch_index)`` to
        remove frontier members whose scores were computed under the old
        evaluator.  HIGH RISK — only set when ``new_tolerances`` actually
        changed.
    reset_no_improvement_count : bool
        If True the runner should zero out ``no_improvement_count`` at the
        boundary to avoid premature early stopping caused by selective erasure.
    log_message : str
        Human-readable summary written to ``feedback_history.md``.
    """

    epoch_index: int
    reason: TransitionReason
    new_tolerances: list[float]
    trigger_adversarial_injection: bool = False
    trigger_selective_erasure: bool = False
    reset_no_improvement_count: bool = False
    log_message: str = ""


# ---------------------------------------------------------------------------
# AdversarialExample
# ---------------------------------------------------------------------------

@dataclass
class AdversarialExample:
    """
    A (question, agent_answer, ground_truth) triple where the agent produced
    an answer that scored *well* under loose tolerances but *badly* under
    strict matching — i.e., a potential gaming example.

    Attributes
    ----------
    question : str
        The benchmark question.
    agent_answer : str
        What the base agent returned.
    ground_truth : str
        The reference answer.
    loose_score : float
        Score under the loosest active tolerance.
    strict_score : float
        Score under tolerance 0.0 (exact match).
    iteration : int
        Loop iteration at which this example was collected.
    epoch_index : int
        Epoch during which this example was collected.
    """

    question: str
    agent_answer: str
    ground_truth: str
    loose_score: float
    strict_score: float
    iteration: int
    epoch_index: int

    @property
    def hack_ratio(self) -> float | None:
        if self.loose_score == 0.0:
            return None
        return self.strict_score / self.loose_score


# ---------------------------------------------------------------------------
# EpochManager
# ---------------------------------------------------------------------------

class EpochManager:
    """
    Tracks epoch boundaries across the self-improvement loop and decides when
    to trigger utility evolution.

    The manager is stateful: it accumulates per-iteration scores and
    adversarial examples, producing an ``EpochTransition`` at each boundary.

    Design contract
    ---------------
    The runner calls:
      1. ``record_iteration_result(...)`` once per completed iteration.
      2. ``is_epoch_boundary(iteration)`` to detect boundary points.
      3. ``evaluate_epoch_boundary(...)`` at boundary points to get the
         ``EpochTransition`` and decide on tolerance updates, adversarial
         injection, and selective erasure.
      4. ``advance_epoch(transition)`` to move the manager to the next epoch.

    Serialisation (checkpoint)
    --------------------------
    ``to_checkpoint_dict()`` / ``from_checkpoint_dict()`` round-trip the
    manager state to/from the JSON checkpoint file so ``--continue`` mode
    restores the correct epoch state.

    Parameters
    ----------
    config : EpochConfig
        Epoch behaviour configuration.
    initial_tolerances : list[float]
        The starting tolerance schedule.  Typically ``DEFAULT_TOLERANCES``.
    """

    def __init__(
        self,
        config: EpochConfig,
        initial_tolerances: list[float] | None = None,
    ) -> None:
        self.config = config
        self.current_tolerances: list[float] = (
            list(initial_tolerances) if initial_tolerances else list(DEFAULT_TOLERANCES)
        )
        self.epoch_index: int = 0
        self.epoch_start_iter: int = 0

        # Accumulates per-iteration scores within the current epoch.
        # Each entry: (iteration, best_frontier_score, strict_score, loose_score)
        self._epoch_scores: list[tuple[int, float, float, float]] = []

        # Adversarial examples collected during the current epoch.
        self.adversarial_pool: list[AdversarialExample] = []

        # Historical snapshots; one per completed epoch.
        self.epoch_history: list[EpochPerformanceSnapshot] = []

        logger.debug(
            "EpochManager initialised: epoch_size=%d, initial_tolerances=%s",
            config.epoch_size,
            self.current_tolerances,
        )

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    def is_epoch_boundary(self, iteration: int) -> bool:
        """
        Return True if ``iteration`` marks the last iteration of the current
        epoch (i.e., exactly ``epoch_size`` iterations have elapsed since
        ``epoch_start_iter``).

        Parameters
        ----------
        iteration : int
            0-indexed loop iteration number (including any ``--continue``
            offset applied by the runner).

        Notes
        -----
        The runner should call this *after* the iteration completes, before
        the next parent selection.
        """
        iters_in_epoch = iteration - self.epoch_start_iter + 1
        return iters_in_epoch >= self.config.epoch_size

    # ------------------------------------------------------------------
    # Per-iteration recording
    # ------------------------------------------------------------------

    def record_iteration_result(
        self,
        iteration: int,
        best_frontier_score: float,
        score_at_strict_tol: float,
        score_at_loose_tol: float,
    ) -> None:
        """
        Store performance metrics for a completed iteration.

        Should be called by the runner *after* ``ProgramManager.update_frontier``
        but *before* ``is_epoch_boundary`` check.

        Parameters
        ----------
        iteration : int
            Current loop iteration (0-indexed, with continue offset).
        best_frontier_score : float
            Highest score in the current frontier after this iteration.
        score_at_strict_tol : float
            Val-set mean score under tolerance 0.0 for the best child this
            iteration.  If no child was evaluated (all failures, no mutation),
            pass the previous iteration's value or 0.0.
        score_at_loose_tol : float
            Val-set mean score under the loosest active tolerance for the best
            child this iteration.
        """
        self._epoch_scores.append(
            (iteration, best_frontier_score, score_at_strict_tol, score_at_loose_tol)
        )
        logger.debug(
            "EpochManager recorded iter=%d  frontier=%.4f  strict=%.4f  loose=%.4f",
            iteration, best_frontier_score, score_at_strict_tol, score_at_loose_tol,
        )

    def record_adversarial_example(self, example: AdversarialExample) -> None:
        """
        Add a gaming example to the pool for this epoch.

        Called by the runner when it detects a per-sample hack condition
        (loose_score high, strict_score low) during the training-set pass.

        Parameters
        ----------
        example : AdversarialExample
            The gaming example to store.
        """
        self.adversarial_pool.append(example)
        logger.debug(
            "AdversarialExample added: hack_ratio=%.3f (iter=%d)",
            example.hack_ratio or float("nan"),
            example.iteration,
        )

    # ------------------------------------------------------------------
    # Boundary evaluation (main public API)
    # ------------------------------------------------------------------

    def evaluate_epoch_boundary(
        self,
        current_iteration: int,
    ) -> EpochTransition:
        """
        Analyse the just-completed epoch and decide on a utility transition.

        Called by the runner when ``is_epoch_boundary(iteration)`` returns
        True.  The caller must subsequently call ``advance_epoch(transition)``
        to update manager state.

        Parameters
        ----------
        current_iteration : int
            The iteration that triggered the boundary (i.e., the *last*
            iteration of the epoch that is ending).

        Returns
        -------
        EpochTransition
            Describes what changes (if any) the runner should apply before
            the next epoch starts.
        """
        return evaluate_epoch_boundary(self, current_iteration)

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def advance_epoch(self, transition: EpochTransition) -> None:
        """
        Commit the changes described by ``transition`` and move to the next
        epoch.

        Must be called by the runner *after* ``evaluate_epoch_boundary`` and
        *after* any runner-side actions (selective erasure, cache flush, etc.)
        have been applied.

        Parameters
        ----------
        transition : EpochTransition
            The transition returned by ``evaluate_epoch_boundary``.
        """
        # Build and store the snapshot for the epoch that just ended.
        snapshot = self._build_snapshot(transition)
        self.epoch_history.append(snapshot)

        # Apply tolerance update.
        self.current_tolerances = list(transition.new_tolerances)

        # Reset epoch-local state.
        self.epoch_index += 1
        self.epoch_start_iter = self.epoch_index * self.config.epoch_size

        # Clear per-epoch accumulators.
        self._epoch_scores = []
        # Flush adversarial pool — RQGM §5.4 implies the pool is used for
        # exactly one epoch (the boundary at which it is injected).
        self.adversarial_pool = []

        logger.info(
            "Epoch %d → %d | tolerances: %s | reason: %s",
            transition.epoch_index,
            self.epoch_index,
            self.current_tolerances,
            transition.reason.name,
        )

    # ------------------------------------------------------------------
    # Checkpoint serialisation
    # ------------------------------------------------------------------

    def to_checkpoint_dict(self) -> dict:
        """
        Serialise the manager's mutable state to a JSON-safe dict for
        inclusion in ``.claude/loop_checkpoint.json``.

        Inverse of ``from_checkpoint_dict``.

        Returns
        -------
        dict
            Keys: ``epoch_index``, ``epoch_start_iter``,
            ``current_tolerances``, ``adversarial_pool``,
            ``epoch_scores`` (partial epoch in progress).
        """
        return {
            "epoch_index": self.epoch_index,
            "epoch_start_iter": self.epoch_start_iter,
            "current_tolerances": self.current_tolerances,
            "adversarial_pool": [
                {
                    "question": ex.question,
                    "agent_answer": ex.agent_answer,
                    "ground_truth": ex.ground_truth,
                    "loose_score": ex.loose_score,
                    "strict_score": ex.strict_score,
                    "iteration": ex.iteration,
                    "epoch_index": ex.epoch_index,
                }
                for ex in self.adversarial_pool
            ],
            "epoch_scores": self._epoch_scores,
        }

    @classmethod
    def from_checkpoint_dict(
        cls,
        data: dict,
        config: EpochConfig,
    ) -> "EpochManager":
        """
        Restore an ``EpochManager`` from a checkpoint dict.

        Parameters
        ----------
        data : dict
            Dict previously produced by ``to_checkpoint_dict``.
        config : EpochConfig
            The epoch configuration to use (sourced from ``RQGMConfig`` after
            reload).

        Returns
        -------
        EpochManager
        """
        manager = cls(config=config, initial_tolerances=data["current_tolerances"])
        manager.epoch_index = data["epoch_index"]
        manager.epoch_start_iter = data["epoch_start_iter"]
        manager._epoch_scores = data.get("epoch_scores", [])
        manager.adversarial_pool = [
            AdversarialExample(**ex) for ex in data.get("adversarial_pool", [])
        ]
        return manager

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_epoch_start_score(self) -> float:
        """Return the frontier score at the *start* of the current epoch."""
        if not self._epoch_scores:
            return 0.0
        return self._epoch_scores[0][1]

    def _get_epoch_end_score(self) -> float:
        """Return the frontier score at the *end* of the current epoch."""
        if not self._epoch_scores:
            return 0.0
        return self._epoch_scores[-1][1]

    def _get_mean_hack_ratio(self) -> float | None:
        """
        Compute the mean hack ratio across all iterations in the epoch.

        For each recorded iteration, compute:
            ratio = score_at_strict_tol / score_at_loose_tol
        where score_at_loose_tol > 0.  Returns the mean of all valid ratios.

        Returns None if no iteration had a valid loose score (all zeros).

        The hack ratio measures how much the agent relies on loose tolerance
        for its score.  A low ratio means the agent scores well under loose
        criteria but poorly under strict matching — the signature of reward
        hacking (RQGM §3.5).
        """
        ratios: list[float] = []
        for _iter, _frontier, strict, loose in self._epoch_scores:
            if loose > 0.0:
                ratios.append(strict / loose)

        if not ratios:
            return None

        return statistics.mean(ratios)

    def _compute_new_tolerances(
        self,
        is_exploiting: bool,
        is_stagnant: bool,
    ) -> list[float]:
        """
        Derive the tolerance schedule for the next epoch.

        Parameters
        ----------
        is_exploiting : bool
            Whether exploitation (reward hacking) was detected.
        is_stagnant : bool
            Whether the epoch showed insufficient improvement.

        Returns
        -------
        list[float]
            New tolerance schedule (sorted ascending, de-duplicated).

        Algorithm (RQGM §3.4, §3.5)
        ----------------------------
        If exploitation detected AND tightening allowed:
            Drop the loosest (largest) tolerance level.  This removes the
            criterion the agent was gaming.  Never drop below
            MIN_TOLERANCE_LEVELS entries.  Respect max_tolerance_drops_per_epoch.

        If NOT exploiting AND NOT stagnant AND relaxation allowed:
            If the current schedule is shorter than DEFAULT_TOLERANCES,
            restore the next-loosest level that was previously removed.
            This rewards genuine improvement by relaxing the evaluator.

        Otherwise:
            Keep current tolerances unchanged.
        """
        mutation_params = self.config.utility_mutation_params
        new_tolerances = list(self.current_tolerances)

        if is_exploiting and mutation_params.allow_tolerance_tightening:
            # Drop the loosest tolerance level(s) — the one the agent is gaming.
            drops = 0
            max_drops = mutation_params.max_tolerance_drops_per_epoch
            while (
                drops < max_drops
                and len(new_tolerances) > MIN_TOLERANCE_LEVELS
                and len(new_tolerances) > 1
            ):
                # Remove the largest (loosest) tolerance.
                new_tolerances.pop()
                drops += 1
            if drops > 0:
                logger.info(
                    "Tightened tolerances: dropped %d level(s) → %s",
                    drops, new_tolerances,
                )

        elif not is_exploiting and not is_stagnant and mutation_params.allow_tolerance_relaxation:
            # Agent is genuinely improving — consider restoring a tolerance level.
            if len(new_tolerances) < len(DEFAULT_TOLERANCES):
                # Find the next tolerance level from DEFAULT that isn't already present.
                for tol in DEFAULT_TOLERANCES:
                    if tol not in new_tolerances:
                        new_tolerances.append(tol)
                        new_tolerances.sort()
                        logger.info(
                            "Relaxed tolerances: restored level %.3f → %s",
                            tol, new_tolerances,
                        )
                        break

        return new_tolerances

    def _build_snapshot(self, transition: EpochTransition) -> EpochPerformanceSnapshot:
        """
        Build the ``EpochPerformanceSnapshot`` for the epoch that just ended.

        Parameters
        ----------
        transition : EpochTransition
            The transition that was applied at this boundary (used to set
            ``utility_transition_fired``).

        Returns
        -------
        EpochPerformanceSnapshot
        """
        start_score = self._get_epoch_start_score()
        end_score = self._get_epoch_end_score()

        # Compute mean strict and loose scores across the epoch.
        strict_scores = [s[2] for s in self._epoch_scores]
        loose_scores = [s[3] for s in self._epoch_scores]

        mean_strict = statistics.mean(strict_scores) if strict_scores else 0.0
        mean_loose = statistics.mean(loose_scores) if loose_scores else 0.0

        end_iter = (
            self._epoch_scores[-1][0] + 1
            if self._epoch_scores
            else self.epoch_start_iter + self.config.epoch_size
        )

        return EpochPerformanceSnapshot(
            epoch_index=self.epoch_index,
            start_iteration=self.epoch_start_iter,
            end_iteration=end_iter,
            start_best_score=start_score,
            end_best_score=end_score,
            score_at_strict_tol=mean_strict,
            score_at_loose_tol=mean_loose,
            tolerance_schedule=list(self.current_tolerances),
            adversarial_examples_collected=len(self.adversarial_pool),
            utility_transition_fired=(
                transition.reason != TransitionReason.NO_TRANSITION
            ),
        )


# ---------------------------------------------------------------------------
# Module-level function: evaluate_epoch_boundary
# ---------------------------------------------------------------------------

def evaluate_epoch_boundary(
    manager: EpochManager,
    current_iteration: int,
) -> EpochTransition:
    """
    Analyse the just-completed epoch and decide on a utility transition.

    This is a module-level function (rather than a method) so it can be unit-
    tested in isolation without constructing a full ``EpochManager`` fixture.
    The ``EpochManager.evaluate_epoch_boundary`` method delegates here.

    Parameters
    ----------
    manager : EpochManager
        The epoch manager holding the current epoch's accumulated scores and
        adversarial pool.
    current_iteration : int
        The iteration that triggered the boundary.

    Returns
    -------
    EpochTransition
        Contains the new tolerance schedule and flags for the runner.

    Algorithm (RQGM §3.4 and §3.5)
    --------------------------------
    Step 1  Compute epoch improvement:
              Δ = end_best_score - start_best_score
    Step 2  Compute hack ratio:
              r = mean(score_strict / score_loose) across epoch iterations
    Step 3  Classify:
              is_stagnant   = Δ < config.min_improvement_threshold
              is_exploiting = r is not None and r < config.exploitation_hack_ratio_threshold
    Step 4  If neither → TransitionReason.NO_TRANSITION, keep tolerances.
    Step 5  Compute new_tolerances via manager._compute_new_tolerances.
    Step 6  Set trigger_adversarial_injection if pool large enough and
            allow_adversarial_injection is True.
    Step 7  Set trigger_selective_erasure if new_tolerances differ from
            current (tolerances actually changed → stale scores invalid).
    Step 8  Build and return EpochTransition.
    """
    cfg = manager.config
    mutation_params = cfg.utility_mutation_params

    # --- Step 1: epoch improvement ----------------------------------------
    start_score = manager._get_epoch_start_score()
    end_score = manager._get_epoch_end_score()
    improvement = end_score - start_score

    # --- Step 2: hack ratio --------------------------------------------------
    mean_hack_ratio = manager._get_mean_hack_ratio()

    # --- Step 3: classify ----------------------------------------------------
    is_stagnant = improvement < cfg.min_improvement_threshold
    is_exploiting = (
        mean_hack_ratio is not None
        and mean_hack_ratio < cfg.exploitation_hack_ratio_threshold
    )

    # --- Step 4: early exit if no transition needed --------------------------
    if not is_stagnant and not is_exploiting:
        return EpochTransition(
            epoch_index=manager.epoch_index,
            reason=TransitionReason.NO_TRANSITION,
            new_tolerances=list(manager.current_tolerances),
            trigger_adversarial_injection=False,
            trigger_selective_erasure=False,
            reset_no_improvement_count=False,
            log_message=(
                f"Epoch {manager.epoch_index} boundary: no transition needed. "
                f"Improvement={improvement:.4f}, hack_ratio={mean_hack_ratio}."
            ),
        )

    # --- Step 5: compute new tolerances -------------------------------------
    new_tolerances = manager._compute_new_tolerances(
        is_exploiting=is_exploiting,
        is_stagnant=is_stagnant,
    )

    # --- Step 6: adversarial injection ---------------------------------------
    trigger_adversarial = (
        mutation_params.allow_adversarial_injection
        and len(manager.adversarial_pool) >= cfg.adversarial_pool_min_size
    )

    # --- Step 7: selective erasure ------------------------------------------
    # Only trigger if tolerances actually changed (stale frontier scores
    # must be invalidated, per RQGM §3.4 "selective erasure").
    tolerances_changed = new_tolerances != manager.current_tolerances
    trigger_erasure = tolerances_changed

    # --- Step 8: reason label -----------------------------------------------
    if is_exploiting and is_stagnant:
        reason = TransitionReason.EXPLOITATION_DETECTED  # exploitation is primary
    elif is_exploiting:
        reason = TransitionReason.EXPLOITATION_DETECTED
    else:
        reason = TransitionReason.STAGNATION

    log_msg = (
        f"Epoch {manager.epoch_index} boundary | reason={reason.name} | "
        f"improvement={improvement:.4f} | hack_ratio={mean_hack_ratio} | "
        f"old_tolerances={manager.current_tolerances} | "
        f"new_tolerances={new_tolerances} | "
        f"adversarial_pool_size={len(manager.adversarial_pool)} | "
        f"selective_erasure={trigger_erasure}"
    )
    logger.info(log_msg)

    return EpochTransition(
        epoch_index=manager.epoch_index,
        reason=reason,
        new_tolerances=new_tolerances,
        trigger_adversarial_injection=trigger_adversarial,
        trigger_selective_erasure=trigger_erasure,
        reset_no_improvement_count=trigger_erasure,  # always reset if frontier changed
        log_message=log_msg,
    )
