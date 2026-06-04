from __future__ import annotations

from pathlib import Path
from typing import Any

from src.harness import build_options
from src.schemas import SurrogateVerifierResponse
from src.agent_profiles.surrogate_verifier.prompt import SURROGATE_VERIFIER_SYSTEM_PROMPT


# Read-only by design: the verifier judges a candidate string; it must never
# modify the skill library. Information isolation (no Write/Edit, fresh session)
# is what keeps its verdict independent of how the skill was produced.
SURROGATE_VERIFIER_TOOLS = [
    "Read",
    "Bash",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "BashOutput",
]


def get_surrogate_verifier_options(
    model: str | None = None,
    project_root: str | Path | None = None,
) -> Any:
    return build_options(
        system=SURROGATE_VERIFIER_SYSTEM_PROMPT.strip(),
        schema=SurrogateVerifierResponse.model_json_schema(),
        tools=SURROGATE_VERIFIER_TOOLS,
        project_root=project_root,
        model=model,
    )


def make_surrogate_verifier_options(
    *,
    project_root: str | Path | None = None,
    model: str | None = None,
):
    return get_surrogate_verifier_options(model=model, project_root=project_root)


surrogate_verifier_options = get_surrogate_verifier_options()
