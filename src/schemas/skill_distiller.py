from pydantic import BaseModel, Field


class SkillDistillerResponse(BaseModel):
    """Response from the skill distiller agent.

    The distiller reads a cluster of *similar* episodes (recurring failures or
    notable successes) and distills one reusable, generalizable skill that would
    help future similar tasks — not a memorized fix for the specific episodes.
    """

    skill_name: str = Field(
        description="kebab-case skill name; also the SKILL.md directory/frontmatter name"
    )
    candidate_skill: str = Field(
        description="The full SKILL.md content, including YAML frontmatter (name, description)"
    )
    target_pattern: str = Field(
        description="The recurring task pattern / failure mode this skill addresses"
    )
    reasoning: str = Field(
        description="Why this generalizes across the cluster rather than memorizing specifics"
    )
