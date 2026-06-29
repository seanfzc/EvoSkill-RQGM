"""
Test script for RQGM components — no agent harness needed.

Tests:
1. EpochManager boundary detection
2. Tolerance tightening when exploitation detected
3. Tolerance relaxation when genuine improvement
4. Adversarial pool collection
5. Checkpoint serialisation round-trip
6. UtilityEvolution evolve_tolerances pure function
7. adversarial_score penalty computation
"""

import sys
sys.path.insert(0, ".")

from src.loop.epoch import (
    EpochManager,
    EpochConfig,
    EpochTransition,
    TransitionReason,
    AdversarialExample,
    DEFAULT_TOLERANCES,
    MIN_TOLERANCE_LEVELS,
    evaluate_epoch_boundary,
)
from src.evaluation.utility_evolution import (
    UtilityEvolution,
    ScoreDistribution,
    evolve_tolerances,
    adversarial_score,
    ADV_PENALTY_WEIGHT,
)

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

# ============================================================================
# Test 1: EpochManager boundary detection
# ============================================================================
print("\n=== Test 1: EpochManager boundary detection ===")
cfg = EpochConfig(epoch_size=5)
mgr = EpochManager(config=cfg)

check("epoch_index starts at 0", mgr.epoch_index == 0)
check("current_tolerances = DEFAULT_TOLERANCES", mgr.current_tolerances == DEFAULT_TOLERANCES)

# Feed 4 iterations — no boundary yet
for i in range(4):
    mgr.record_iteration_result(i, 0.5, 0.3, 0.6)
    check(f"iteration {i}: is_epoch_boundary=False", not mgr.is_epoch_boundary(i))

# 5th iteration — boundary should fire
mgr.record_iteration_result(4, 0.55, 0.35, 0.65)
check("iteration 4: is_epoch_boundary=True", mgr.is_epoch_boundary(4))

# ============================================================================
# Test 2: Tolerance tightening (exploitation detected)
# ============================================================================
print("\n=== Test 2: Tolerance tightening ===")
cfg2 = EpochConfig(
    epoch_size=3,
    exploitation_hack_ratio_threshold=0.6,
    min_improvement_threshold=0.02,
)
mgr2 = EpochManager(config=cfg2)

# Simulate exploitation: high loose scores, low strict scores
# hack_ratio = 0.3/0.9 = 0.33 < 0.6 threshold
mgr2.record_iteration_result(0, 0.50, 0.30, 0.90)
mgr2.record_iteration_result(1, 0.52, 0.32, 0.88)
mgr2.record_iteration_result(2, 0.51, 0.28, 0.92)

check("iteration 2: is_epoch_boundary=True", mgr2.is_epoch_boundary(2))

transition = mgr2.evaluate_epoch_boundary(2)
check("exploitation detected", transition.reason == TransitionReason.EXPLOITATION_DETECTED)
check("tolerances changed", transition.new_tolerances != mgr2.current_tolerances)
check("tolerances tightened (fewer levels)", len(transition.new_tolerances) < len(mgr2.current_tolerances),
      f"old={mgr2.current_tolerances}, new={transition.new_tolerances}")
check("loosest tolerance dropped", max(transition.new_tolerances) < max(mgr2.current_tolerances),
      f"old_max={max(mgr2.current_tolerances)}, new_max={max(transition.new_tolerances)}")

mgr2.advance_epoch(transition)
check("epoch_index advanced to 1", mgr2.epoch_index == 1)
check("tolerances updated", mgr2.current_tolerances == transition.new_tolerances)

# ============================================================================
# Test 3: No transition when agent is genuinely improving
# ============================================================================
print("\n=== Test 3: No transition when genuinely improving ===")
cfg3 = EpochConfig(
    epoch_size=3,
    exploitation_hack_ratio_threshold=0.6,
    min_improvement_threshold=0.02,
)
mgr3 = EpochManager(config=cfg3)

# Simulate genuine improvement: strict and loose scores both high
mgr3.record_iteration_result(0, 0.50, 0.70, 0.75)
mgr3.record_iteration_result(1, 0.55, 0.72, 0.78)
mgr3.record_iteration_result(2, 0.60, 0.75, 0.80)

transition3 = mgr3.evaluate_epoch_boundary(2)
check("no transition needed", transition3.reason == TransitionReason.NO_TRANSITION)
check("tolerances unchanged", transition3.new_tolerances == mgr3.current_tolerances)

# ============================================================================
# Test 4: Adversarial pool collection
# ============================================================================
print("\n=== Test 4: Adversarial pool collection ===")
cfg4 = EpochConfig(epoch_size=3)
mgr4 = EpochManager(config=cfg4)

# Add gaming examples: high loose, low strict
ex1 = AdversarialExample(
    question="What is 2+2?",
    agent_answer="approximately 4",
    ground_truth="4",
    loose_score=0.9,
    strict_score=0.0,
    iteration=0,
    epoch_index=0,
)
ex2 = AdversarialExample(
    question="What is 5+3?",
    agent_answer="around 8",
    ground_truth="8",
    loose_score=0.85,
    strict_score=0.0,
    iteration=1,
    epoch_index=0,
)
mgr4.record_adversarial_example(ex1)
mgr4.record_adversarial_example(ex2)

check("adversarial pool has 2 examples", len(mgr4.adversarial_pool) == 2)
check("hack_ratio of ex1 is 0.0", ex1.hack_ratio == 0.0)
check("hack_ratio of ex2 is 0.0", ex2.hack_ratio == 0.0)

# ============================================================================
# Test 5: Checkpoint serialisation round-trip
# ============================================================================
print("\n=== Test 5: Checkpoint serialisation ===")
cfg5 = EpochConfig(epoch_size=5)
mgr5 = EpochManager(config=cfg5)
mgr5.record_iteration_result(0, 0.5, 0.3, 0.6)
mgr5.record_iteration_result(1, 0.55, 0.35, 0.65)

checkpoint = mgr5.to_checkpoint_dict()
check("checkpoint has epoch_index", "epoch_index" in checkpoint)
check("checkpoint has current_tolerances", "current_tolerances" in checkpoint)
check("checkpoint has epoch_scores", "epoch_scores" in checkpoint)
check("checkpoint has adversarial_pool", "adversarial_pool" in checkpoint)

restored = EpochManager.from_checkpoint_dict(checkpoint, cfg5)
check("restored epoch_index matches", restored.epoch_index == mgr5.epoch_index)
check("restored tolerances match", restored.current_tolerances == mgr5.current_tolerances)
check("restored epoch_scores match", restored._epoch_scores == mgr5._epoch_scores)

# ============================================================================
# Test 6: evolve_tolerances pure function
# ============================================================================
print("\n=== Test 6: evolve_tolerances pure function ===")
from src.loop.epoch import UtilityMutationParams

# Exploitation scenario: strict=0.3, loose=0.9, hack_ratio=0.33
dist_exploit = ScoreDistribution(
    scores_at_strict=[0.3, 0.32, 0.28],
    scores_at_loose=[0.9, 0.88, 0.92],
)
mutation_params = UtilityMutationParams(allow_tolerance_tightening=True)
new_tols, log = evolve_tolerances(
    current_tolerances=list(DEFAULT_TOLERANCES),
    distribution=dist_exploit,
    mutation_params=mutation_params,
    hack_ratio_threshold=0.6,
    epoch_improvement=0.01,
    min_improvement_threshold=0.02,
)
check("exploitation: tolerances tightened", len(new_tols) < len(DEFAULT_TOLERANCES),
      f"old={len(DEFAULT_TOLERANCES)}, new={len(new_tols)}")
check("exploitation: loosest dropped", max(new_tols) < max(DEFAULT_TOLERANCES),
      f"old_max={max(DEFAULT_TOLERANCES)}, new_max={max(new_tols)}")

# Genuine improvement scenario: strict=0.7, loose=0.75, hack_ratio=0.93
dist_good = ScoreDistribution(
    scores_at_strict=[0.7, 0.72, 0.75],
    scores_at_loose=[0.75, 0.78, 0.80],
)
# Start with tightened tolerances (one level dropped)
tightened = list(DEFAULT_TOLERANCES[:-1])  # drop 0.1
new_tols2, log2 = evolve_tolerances(
    current_tolerances=tightened,
    distribution=dist_good,
    mutation_params=mutation_params,
    hack_ratio_threshold=0.6,
    epoch_improvement=0.05,
    min_improvement_threshold=0.02,
)
check("improvement: tolerances relaxed", len(new_tols2) > len(tightened),
      f"old={len(tightened)}, new={len(new_tols2)}")

# ============================================================================
# Test 7: adversarial_score
# ============================================================================
print("\n=== Test 7: adversarial_score ===")
# Gaming answer: scores well under loose, poorly under strict
score_gaming = adversarial_score(
    question="What is 2+2?",
    predicted="approximately 4",
    ground_truth="4",
    current_tolerances=[0.0, 0.1],
    adversarial_pool=[ex1, ex2],  # non-empty pool triggers penalty
    adversarial_weight=0.3,
)
# Without adversarial pool, should be higher
score_clean = adversarial_score(
    question="What is 2+2?",
    predicted="4",
    ground_truth="4",
    current_tolerances=[0.0, 0.1],
    adversarial_pool=[],  # empty pool = no penalty
    adversarial_weight=0.3,
)
check("gaming answer penalised", score_gaming < 1.0, f"score={score_gaming}")
check("clean answer not penalised", score_clean > score_gaming, f"clean={score_clean}, gaming={score_gaming}")

# ============================================================================
# Test 8: ScoreDistribution.get_gaming_indices
# ============================================================================
print("\n=== Test 8: ScoreDistribution.get_gaming_indices ===")
dist = ScoreDistribution(
    scores_at_strict=[0.0, 0.8, 0.2, 0.9],
    scores_at_loose=[0.9, 0.85, 0.7, 0.95],
    questions=["q1", "q2", "q3", "q4"],
)
gaming = dist.get_gaming_indices(hack_ratio_threshold=0.6, min_loose_score=0.5)
check("q1 flagged as gaming (0.0/0.9=0.0)", 0 in gaming)
check("q2 not flagged (0.8/0.85=0.94 > 0.6)", 1 not in gaming)
check("q3 flagged as gaming (0.2/0.7=0.29)", 2 in gaming)
check("q4 not flagged (0.9/0.95=0.95 > 0.6)", 3 not in gaming)

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
if FAIL == 0:
    print("✅ ALL TESTS PASSED")
else:
    print(f"❌ {FAIL} TEST(S) FAILED")
    sys.exit(1)
