from __future__ import annotations

from pathlib import Path
from typing import Any

from src.harness import build_options
from src.schemas import SkillDistillerResponse
from src.agent_profiles.skill_distiller.prompt import SKILL_DISTILLER_SYSTEM_PROMPT


# Read-only toolset by design: the distiller produces a *candidate* skill string
# for the review buffer and must never touch the live `.claude/skills/` library
# (no Write/Edit). Graduation into the library happens later, through the gate.
SKILL_DISTILLER_TOOLS = [
    "Read",
    "Bash",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "BashOutput",
]


def get_skill_distiller_options(
    model: str | None = None,
    project_root: str | Path | None = None,
) -> Any:
    return build_options(
        system=SKILL_DISTILLER_SYSTEM_PROMPT.strip(),
        schema=SkillDistillerResponse.model_json_schema(),
        tools=SKILL_DISTILLER_TOOLS,
        project_root=project_root,
        model=model,
    )


def make_skill_distiller_options(
    *,
    project_root: str | Path | None = None,
    model: str | None = None,
):
    return get_skill_distiller_options(model=model, project_root=project_root)


skill_distiller_options = get_skill_distiller_options()
