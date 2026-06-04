"""Per-skill statistics and credit assignment.

Continuous evolution needs to know which skills *earn their place*. As episodes
arrive, `assign_credit` attributes each outcome to the skills that were active
during the attempt, accumulating usage and a credit-weighted contribution score.
The deprecation policy (in `lifecycle.py`) later reads these stats to retire
skills that stop pulling their weight.

Stats live in `.evoskill/continuous/skill_stats.json` rather than the
git-versioned `program.yaml`: credit accrues continuously from deployment
traces, and high-frequency updates should not churn the program's git history.
A snapshot can be folded into `program.yaml` at graduation time (Phase 3).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .episode import TaskEpisode


class SkillStats(BaseModel):
    """Accumulated usage/credit for one skill."""

    name: str
    episodes_active: int = Field(default=0, description="Times this skill was loaded during an attempt")
    success_when_active: int = Field(default=0, description="Successful attempts while active")
    contribution: float = Field(default=0.0, description="Credit-weighted share of wins")
    deprecation_strikes: int = Field(default=0, description="Consecutive windows below baseline")
    last_credited_at: str | None = Field(default=None, description="ISO time of last credit update")

    @property
    def success_rate(self) -> float:
        return self.success_when_active / self.episodes_active if self.episodes_active else 0.0


class SkillStatsStore:
    """Persist a map of skill name → `SkillStats`."""

    DEFAULT_SUBPATH = Path(".evoskill") / "continuous" / "skill_stats.json"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else self.DEFAULT_SUBPATH
        self._stats: dict[str, SkillStats] = {}
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            self._stats = {}
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self._stats = {}
            return
        stats: dict[str, SkillStats] = {}
        if isinstance(data, dict):
            for name, raw in data.items():
                try:
                    stats[name] = SkillStats.model_validate(raw)
                except ValueError:
                    continue
        self._stats = stats

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: s.model_dump() for name, s in self._stats.items()}
        self.path.write_text(json.dumps(payload, indent=2))

    def get(self, name: str) -> SkillStats:
        """Get stats for a skill, creating an empty record if absent."""
        if name not in self._stats:
            self._stats[name] = SkillStats(name=name)
        return self._stats[name]

    def has(self, name: str) -> bool:
        return name in self._stats

    def upsert(self, stats: SkillStats) -> None:
        self._stats[stats.name] = stats

    def all(self) -> list[SkillStats]:
        return list(self._stats.values())

    def remove(self, name: str) -> None:
        self._stats.pop(name, None)


def assign_credit(
    store: SkillStatsStore,
    episode: TaskEpisode,
    *,
    timestamp: str | None = None,
) -> bool:
    """Attribute one episode's outcome to the skills that were active.

    Each active skill gets `episodes_active += 1`. On success, every active
    skill also gets `success_when_active += 1` and an equal share (1/N) of the
    win added to `contribution` — so a skill that is the *sole* active skill in
    a win earns full credit, while one of many shares it.

    Episodes with no `skills_active` (e.g. raw Harbor trajectories that don't
    record loaded skills) are skipped. Returns True if any credit was assigned.
    """
    active = episode.skills_active
    if not active:
        return False
    ts = timestamp or datetime.now().isoformat()
    share = 1.0 / len(active) if episode.is_success else 0.0
    for name in active:
        stats = store.get(name)
        stats.episodes_active += 1
        if episode.is_success:
            stats.success_when_active += 1
            stats.contribution += share
        stats.last_credited_at = ts
        store.upsert(stats)
    return True


def assign_credit_batch(
    store: SkillStatsStore,
    episodes: list[TaskEpisode],
    *,
    timestamp: str | None = None,
    save: bool = True,
) -> int:
    """Assign credit for a batch of episodes. Returns the number credited."""
    credited = sum(assign_credit(store, ep, timestamp=timestamp) for ep in episodes)
    if save:
        store.save()
    return credited
