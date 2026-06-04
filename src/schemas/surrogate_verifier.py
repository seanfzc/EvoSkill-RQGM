from pydantic import BaseModel, Field


class SurrogateVerifierResponse(BaseModel):
    """Response from the surrogate verifier agent.

    The verifier judges a *candidate* skill — in an isolated session, with no
    access to how the skill was distilled — by synthesizing test assertions from
    the skill and a set of held-out task descriptions, then deciding whether the
    skill is correct and generalizable (not a memorized one-off).
    """

    score: float = Field(
        description="Confidence in [0,1] that the skill helps and generalizes"
    )
    verdict: bool = Field(
        description="Whether the skill should pass the gate (correct + generalizable)"
    )
    assertions: list[str] = Field(
        default_factory=list,
        description="Test assertions the verifier synthesized to judge the skill",
    )
    reasoning: str = Field(
        default="", description="Why the skill passes or fails, with diagnostics"
    )
