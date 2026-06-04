"""Read the live skill library (`.claude/skills/`).

Lifecycle operations (dedup, retrieval, deprecation) need a structured view of
the installed skills — each skill's name, description, and body — independent of
the git/program machinery in `registry/manager.py`. This module provides that
read-only view by parsing each `SKILL.md`'s YAML frontmatter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# Same frontmatter shape used by the rest of the codebase: a leading
# ---\n...\n--- YAML block.
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


@dataclass
class Skill:
    """One installed skill, parsed from its SKILL.md."""

    name: str
    description: str
    body: str
    path: Path

    @property
    def dir_name(self) -> str:
        return self.path.parent.name

    @property
    def text(self) -> str:
        """Text used for similarity comparisons: name + description.

        The description is the most signal-dense, length-stable part of a skill
        (the body varies wildly), so dedup/retrieval compare on name+description.
        """
        return f"{self.name}: {self.description}".strip().rstrip(":").strip()


def parse_skill_file(path: Path) -> Skill:
    """Parse a SKILL.md into a `Skill`. Tolerant of missing/invalid frontmatter."""
    text = path.read_text()
    name = path.parent.name
    description = ""
    body = text

    match = _FRONTMATTER_RE.match(text)
    if match:
        body = text[match.end():]
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        if isinstance(meta, dict):
            name = str(meta.get("name") or name)
            description = str(meta.get("description") or "")
    return Skill(name=name, description=description, body=body.strip(), path=path)


class SkillLibrary:
    """Read-only view over a `.claude/skills/` directory."""

    def __init__(self, skills_dir: str | Path) -> None:
        self.skills_dir = Path(skills_dir)

    def list(self) -> list[Skill]:
        """All skills (dirs containing a SKILL.md), sorted by name."""
        if not self.skills_dir.is_dir():
            return []
        skills: list[Skill] = []
        for child in sorted(self.skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if skill_file.is_file():
                skills.append(parse_skill_file(skill_file))
        return skills

    def names(self) -> list[str]:
        return [s.name for s in self.list()]

    def get(self, name: str) -> Skill | None:
        """Find a skill by frontmatter name or directory name."""
        for skill in self.list():
            if skill.name == name or skill.dir_name == name:
                return skill
        return None
