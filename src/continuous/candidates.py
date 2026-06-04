"""Candidate skill buffer.

Distilled skills do NOT go straight into the live `.claude/skills/` library — they
land here first, as *candidates* for review (and, in later phases, for the
quality gate). This keeps the always-on learner safe: nothing reaches the agent
the user actually runs until it has been gated/approved.

Layout under `.evoskill/continuous/candidates/`:

    <candidate_id>/
        candidate.json   # metadata (pattern, provenance, status, stats)
        SKILL.md         # the human-reviewable distilled skill

`candidate_id` is derived from the skill name + a hash of the source episode ids,
so re-harvesting the same cluster overwrites rather than duplicates.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

CandidateStatus = Literal["pending", "graduated", "rejected"]


class Candidate(BaseModel):
    """A distilled skill awaiting review/gating."""

    candidate_id: str = Field(description="Stable id (skill_name + episode hash)")
    skill_name: str = Field(description="kebab-case skill name")
    skill_markdown: str = Field(description="Full SKILL.md content")
    target_pattern: str = Field(default="", description="Recurring pattern addressed")
    reasoning: str = Field(default="", description="Why it generalizes")

    # Provenance.
    source: str = Field(default="harvest", description="What produced this candidate")
    cluster_key: str = Field(default="", description="Cluster label it was distilled from")
    cluster_size: int = Field(default=0, description="Number of episodes in the cluster")
    episode_ids: list[str] = Field(default_factory=list, description="Source episode ids")
    outcome_focus: str = Field(default="failure", description="failure | success")

    status: CandidateStatus = Field(default="pending", description="Lifecycle status")
    model_name: str | None = Field(default=None, description="Distiller model")
    created_at: str | None = Field(default=None, description="ISO timestamp")
    extra: dict[str, Any] = Field(default_factory=dict)


def make_candidate_id(skill_name: str, episode_ids: list[str]) -> str:
    """Stable id: skill name + short hash of the (order-independent) episode set."""
    digest = hashlib.sha256("|".join(sorted(episode_ids)).encode("utf-8")).hexdigest()[:8]
    safe_name = skill_name.strip().lower() or "skill"
    return f"{safe_name}-{digest}"


class CandidateStore:
    """Read/write the candidate buffer on disk."""

    DEFAULT_SUBPATH = Path(".evoskill") / "continuous" / "candidates"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def dir_for(self, candidate_id: str) -> Path:
        return self.root / candidate_id

    def save(self, candidate: Candidate, *, timestamp: str | None = None) -> Path:
        """Persist a candidate (metadata + SKILL.md). Returns its directory."""
        if candidate.created_at is None:
            candidate = candidate.model_copy(
                update={"created_at": timestamp or datetime.now().isoformat()}
            )
        target = self.dir_for(candidate.candidate_id)
        target.mkdir(parents=True, exist_ok=True)
        (target / "candidate.json").write_text(candidate.model_dump_json(indent=2))
        (target / "SKILL.md").write_text(candidate.skill_markdown)
        return target

    def get(self, candidate_id: str) -> Candidate | None:
        meta = self.dir_for(candidate_id) / "candidate.json"
        if not meta.is_file():
            return None
        try:
            return Candidate.model_validate_json(meta.read_text())
        except (OSError, ValueError):
            return None

    def list(self) -> list[Candidate]:
        """All candidates, newest first (by created_at, then id)."""
        if not self.root.is_dir():
            return []
        out: list[Candidate] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            candidate = self.get(child.name)
            if candidate is not None:
                out.append(candidate)
        out.sort(key=lambda c: (c.created_at or "", c.candidate_id), reverse=True)
        return out

    def set_status(self, candidate_id: str, status: CandidateStatus) -> Candidate | None:
        """Update a candidate's lifecycle status in place."""
        candidate = self.get(candidate_id)
        if candidate is None:
            return None
        candidate = candidate.model_copy(update={"status": status})
        # Preserve created_at (already set).
        target = self.dir_for(candidate_id)
        (target / "candidate.json").write_text(candidate.model_dump_json(indent=2))
        return candidate
