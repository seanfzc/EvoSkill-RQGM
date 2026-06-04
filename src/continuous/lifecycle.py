"""Skill library lifecycle management.

The continuous learner keeps adding skills; without curation the library bloats,
skills conflict, and the agent's context degrades. This module provides the
SkillOpt-style operations that keep the library healthy:

* `find_duplicates` — group near-duplicate skills by similarity (detection).
* `select_skills` — rank skills by relevance to a task (retrieval / top-k).
* `evaluate_deprecation` — strike-and-retire skills that stop earning credit.
* `propose_merge` — synthesize redundant skills into one *candidate* (apply via
  the Phase 3 gate, never a blind live mutation here).
* `archive_skill` / `restore_skill` — reversible filesystem archival.

Design stance (Phase 2 is the safe substrate): detection and proposal only.
Anything that would *change* the live library destructively is either reversible
(archival) or routed through the candidate buffer for the gate to validate.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidates import Candidate, CandidateStore, make_candidate_id
from .harvest import DistillerLike, slugify_skill_name
from .library import Skill, SkillLibrary
from .similarity import SimilarityBackend
from .skill_stats import SkillStats, SkillStatsStore


# ──────────────────────────────────────────────────────────────────────────────
# Dedup detection
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DuplicateGroup:
    """A set of skills judged near-duplicate by the similarity backend."""

    skills: list[Skill]
    max_similarity: float

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.skills]


def find_duplicates(
    skills: list[Skill],
    backend: SimilarityBackend,
    *,
    threshold: float = 0.88,
) -> list[DuplicateGroup]:
    """Group skills whose pairwise similarity meets `threshold`.

    Uses union-find over the similarity graph, so transitively-similar skills
    land in one group. Singletons are dropped — only groups of 2+ are returned.
    """
    n = len(skills)
    if n < 2:
        return []

    matrix = backend.pairwise([s.text for s in skills])

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    max_sim: dict[int, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    result: list[DuplicateGroup] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        within = [matrix[a][b] for a in members for b in members if a < b]
        result.append(
            DuplicateGroup(
                skills=[skills[i] for i in members],
                max_similarity=max(within) if within else 0.0,
            )
        )
    result.sort(key=lambda g: g.max_similarity, reverse=True)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval / selection
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SkillMatch:
    skill: Skill
    score: float


def select_skills(
    skills: list[Skill],
    task: str,
    backend: SimilarityBackend,
    *,
    k: int = 6,
    min_score: float = 0.0,
) -> list[SkillMatch]:
    """Rank skills by relevance to `task`, returning the top-k above `min_score`.

    This is the guard against context bloat: instead of loading every skill, a
    task loads only the most relevant ones. Phase 2 exposes the capability and a
    CLI preview; wiring it into the live agent's skill-loading is a later step
    (it changes agent behavior and must be validated by the gate first).
    """
    if not skills:
        return []
    ranked = backend.rank(task, [s.text for s in skills])
    matches = [SkillMatch(skill=skills[i], score=score) for i, score in ranked if score >= min_score]
    return matches[:k]


# ──────────────────────────────────────────────────────────────────────────────
# Deprecation policy
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DeprecationReport:
    """Outcome of one deprecation-policy window."""

    candidates: list[str] = field(default_factory=list)   # past the strike limit → retire
    struck: list[str] = field(default_factory=list)        # got a strike this window
    recovered: list[str] = field(default_factory=list)     # had strikes, now reset
    unused: list[str] = field(default_factory=list)         # never active → cannot judge


def evaluate_deprecation(
    store: SkillStatsStore,
    library_names: list[str],
    *,
    baseline: float = 0.0,
    strikes_limit: int = 3,
    save: bool = True,
) -> DeprecationReport:
    """Apply one window of the deprecation policy and return what it found.

    For each *active* skill (episodes_active > 0): if its `contribution` is at or
    below `baseline` it earns a strike, otherwise its strikes reset to zero. A
    skill that reaches `strikes_limit` consecutive strikes becomes a deprecation
    candidate. Skills that were never active are reported as `unused` but never
    struck — an unloaded skill has no evidence against it (it may simply be new,
    or retrieval hasn't surfaced it yet).
    """
    report = DeprecationReport()
    for name in library_names:
        if not store.has(name) or store.get(name).episodes_active == 0:
            report.unused.append(name)
            continue
        stats = store.get(name)
        if stats.contribution <= baseline:
            stats.deprecation_strikes += 1
            report.struck.append(name)
        elif stats.deprecation_strikes > 0:
            stats.deprecation_strikes = 0
            report.recovered.append(name)
        store.upsert(stats)
        if stats.deprecation_strikes >= strikes_limit:
            report.candidates.append(name)
    if save:
        store.save()
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Archival (reversible)
# ──────────────────────────────────────────────────────────────────────────────


def archive_skill(skills_dir: str | Path, dir_name: str, archive_dir: str | Path) -> Path:
    """Move a skill directory out of the live library into the archive.

    Reversible: the skill is moved (not deleted) so `restore_skill` can bring it
    back. Returns the archived path.
    """
    skills_dir = Path(skills_dir)
    archive_dir = Path(archive_dir)
    src = skills_dir / dir_name
    if not src.is_dir():
        raise FileNotFoundError(f"no skill directory: {src}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    dst = archive_dir / dir_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    return dst


def restore_skill(archive_dir: str | Path, dir_name: str, skills_dir: str | Path) -> Path:
    """Move an archived skill back into the live library."""
    archive_dir = Path(archive_dir)
    skills_dir = Path(skills_dir)
    src = archive_dir / dir_name
    if not src.is_dir():
        raise FileNotFoundError(f"no archived skill: {src}")
    skills_dir.mkdir(parents=True, exist_ok=True)
    dst = skills_dir / dir_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    return dst


# ──────────────────────────────────────────────────────────────────────────────
# Merge (propose-only)
# ──────────────────────────────────────────────────────────────────────────────


def build_merge_query(skills: list[Skill]) -> str:
    """Render a prompt asking to synthesize several skills into one."""
    lines = [
        f"Merge the following {len(skills)} overlapping skills into ONE coherent, "
        "non-redundant skill. Preserve every distinct rule; drop duplication. "
        "Return a complete SKILL.md (with frontmatter) plus a name.",
        "",
    ]
    for i, s in enumerate(skills, start=1):
        lines.append(f"## Skill {i}: {s.name}")
        lines.append(f"description: {s.description}")
        lines.append(s.body)
        lines.append("")
    return "\n".join(lines)


async def propose_merge(
    skills: list[Skill],
    merger: DistillerLike,
    store: CandidateStore,
    *,
    source: str = "merge",
) -> Candidate | None:
    """Propose a merged skill as a *candidate* (not a live mutation).

    The merged skill lands in the candidate buffer for review/gating; applying it
    (replacing the originals) happens through the Phase 3 gate. Returns the
    Candidate, or None if fewer than 2 skills or the merger produced nothing.
    """
    if len(skills) < 2:
        return None
    query = build_merge_query(skills)
    trace = await merger.run(query)
    output = getattr(trace, "output", None)
    candidate_skill = getattr(output, "candidate_skill", None)
    if not candidate_skill or not str(candidate_skill).strip():
        return None

    source_names = [s.name for s in skills]
    raw_name = getattr(output, "skill_name", "") or f"merged-{source_names[0]}"
    skill_name = slugify_skill_name(str(raw_name))
    candidate = Candidate(
        candidate_id=make_candidate_id(skill_name, source_names),
        skill_name=skill_name,
        skill_markdown=str(candidate_skill),
        target_pattern=str(getattr(output, "target_pattern", "") or "merge of redundant skills"),
        reasoning=str(getattr(output, "reasoning", "") or ""),
        source=source,
        cluster_key="+".join(source_names),
        cluster_size=len(skills),
        episode_ids=[],
        outcome_focus="merge",
        model_name=getattr(trace, "model", None),
        extra={"merged_from": source_names},
    )
    store.save(candidate)
    return candidate
