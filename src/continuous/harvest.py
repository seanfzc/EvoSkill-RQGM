"""Offline harvest: turn a window of real episodes into candidate skills.

This is the Phase 1 pipeline — steps 1, 2, 4, 5 of the continuous loop
(COLLECT → SIGNAL → CLUSTER → DISTILL), without the always-on daemon or the
quality gate. Credit assignment (step 3) and gating/graduation (steps 6–8) are
later phases; harvest only *proposes* candidates into the review buffer.

`harvest()` takes its collaborators (readers, distiller, store) as arguments so
it is fully testable without a real LLM or CLI: tests inject a fake distiller.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .candidates import Candidate, CandidateStore, make_candidate_id
from .cluster import EpisodeCluster, cluster_episodes
from .collector import (
    GooseRawReader,
    HarborTrajectoryReader,
    JsonlReader,
    TraceCollector,
    TraceCursor,
    TraceReader,
)
from .episode import Outcome, TaskEpisode

_Logger = Callable[[str], None]
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class DistillerLike(Protocol):
    """Anything with an async `run(query) -> trace` whose `.output` is a
    SkillDistillerResponse (or None on parse failure). `src.harness.Agent`
    satisfies this; tests provide a fake."""

    async def run(self, query: str) -> Any:  # pragma: no cover - protocol
        ...


def build_readers(
    sources: list[str],
    *,
    traces_root: str | None = None,
    jsonl_path: str | None = None,
    success_threshold: float = 1.0,
) -> list[TraceReader]:
    """Construct trace readers for the requested sources, in priority order.

    Unknown source names and sources missing their path are skipped, so a
    partially-configured project still harvests from whatever is available.
    """
    readers: list[TraceReader] = []
    for source in sources:
        if source == "harbor" and traces_root:
            readers.append(HarborTrajectoryReader(traces_root, success_threshold=success_threshold))
        elif source == "goose" and traces_root:
            readers.append(GooseRawReader(traces_root, success_threshold=success_threshold))
        elif source == "jsonl" and jsonl_path:
            readers.append(JsonlReader(jsonl_path, success_threshold=success_threshold))
    return readers


def slugify_skill_name(name: str, *, fallback: str = "distilled-skill") -> str:
    """Normalize a skill name to a safe kebab-case directory/frontmatter name."""
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or fallback


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_distiller_query(
    cluster: EpisodeCluster,
    *,
    max_examples: int = 6,
    task_chars: int = 600,
    output_chars: int = 300,
) -> str:
    """Render a cluster into a prompt for the distiller.

    Summarizes the shared pattern and a handful of example episodes — what was
    asked, which tools the agent used, and its final output — so the distiller
    can find the transferable lesson without seeing every raw trace.
    """
    focus = cluster.outcome_focus.value
    lines: list[str] = [
        f"You are given {cluster.size} episodes that share an outcome of "
        f"'{focus}' and were clustered together by task similarity.",
        f"Common terms across the cluster: {', '.join(cluster.top_terms) or '(none)'}.",
        "",
        "Distill ONE reusable, generalizable skill that would help an agent handle "
        f"this kind of task. Do not memorize specifics from these examples.",
        "",
        "## Example episodes",
    ]
    for i, ep in enumerate(cluster.episodes[:max_examples], start=1):
        tools = sorted(
            {c.function_name for s in ep.actions for c in s.tool_calls if c.function_name}
        )
        lines.append(f"\n### Episode {i} (outcome: {ep.outcome.value})")
        lines.append(f"Task: {_truncate(ep.task_text, task_chars) or '(unknown)'}")
        if tools:
            lines.append(f"Tools used: {', '.join(tools)}")
        if ep.final_output:
            lines.append(f"Final output: {_truncate(ep.final_output, output_chars)}")
    if cluster.size > max_examples:
        lines.append(f"\n(+{cluster.size - max_examples} more similar episodes not shown)")
    return "\n".join(lines)


@dataclass
class HarvestResult:
    """Outcome of one harvest run."""

    episodes_collected: int
    clusters: list[EpisodeCluster]
    candidates: list[Candidate]
    failed_distillations: int = 0
    skipped_clusters: int = field(default=0)
    cost_usd: float = 0.0

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)


async def distill_clusters(
    clusters: list[EpisodeCluster],
    distiller: DistillerLike,
    store: CandidateStore,
    *,
    source: str = "harvest",
    concurrency: int = 4,
    log: _Logger | None = None,
) -> tuple[list[Candidate], int, float]:
    """Distill one candidate per cluster and persist them.

    Shared by the offline `harvest()` and the `watch` daemon's tick so both use
    one distillation path. Returns (candidates, failed_count, total_cost_usd).
    A cluster whose distillation errors or yields no usable skill is counted as
    failed (never aborts the batch).
    """
    emit = log or (lambda _msg: None)
    if not clusters:
        return [], 0, 0.0

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _distill(cluster: EpisodeCluster) -> tuple[Candidate | None, float]:
        async with semaphore:
            query = build_distiller_query(cluster)
            try:
                trace = await distiller.run(query)
            except Exception as exc:  # noqa: BLE001 - one bad cluster shouldn't abort the batch
                emit(f"  [warn] distillation failed for cluster '{cluster.key}': {exc}")
                return None, 0.0
            cost = float(getattr(trace, "total_cost_usd", 0.0) or 0.0)
            candidate = _candidate_from_output(
                getattr(trace, "output", None), cluster,
                source=source, model_name=getattr(trace, "model", None),
            )
            if candidate is None:
                emit(f"  [warn] distiller produced no usable skill for '{cluster.key}'")
            return candidate, cost

    results = await asyncio.gather(*[_distill(c) for c in clusters])

    candidates: list[Candidate] = []
    failed = 0
    total_cost = 0.0
    for candidate, cost in results:
        total_cost += cost
        if candidate is None:
            failed += 1
            continue
        store.save(candidate)
        candidates.append(candidate)
        emit(f"  + candidate '{candidate.skill_name}' "
             f"(from {candidate.cluster_size} episodes) → {candidate.candidate_id}")
    return candidates, failed, total_cost


def _candidate_from_output(
    output: Any,
    cluster: EpisodeCluster,
    *,
    source: str,
    model_name: str | None,
) -> Candidate | None:
    """Build a Candidate from a SkillDistillerResponse-like object, or None."""
    candidate_skill = getattr(output, "candidate_skill", None)
    if not candidate_skill or not str(candidate_skill).strip():
        return None
    raw_name = getattr(output, "skill_name", "") or cluster.key
    skill_name = slugify_skill_name(str(raw_name))
    episode_ids = cluster.episode_ids
    return Candidate(
        candidate_id=make_candidate_id(skill_name, episode_ids),
        skill_name=skill_name,
        skill_markdown=str(candidate_skill),
        target_pattern=str(getattr(output, "target_pattern", "") or ""),
        reasoning=str(getattr(output, "reasoning", "") or ""),
        source=source,
        cluster_key=cluster.key,
        cluster_size=cluster.size,
        episode_ids=episode_ids,
        outcome_focus=cluster.outcome_focus.value,
        model_name=model_name,
    )


async def harvest(
    *,
    readers: list[TraceReader],
    distiller: DistillerLike,
    store: CandidateStore,
    cursor: TraceCursor | None = None,
    window: int = 200,
    min_cluster_size: int = 3,
    similarity_threshold: float = 0.3,
    focus: Outcome = Outcome.FAILURE,
    max_candidates: int | None = None,
    concurrency: int = 4,
    advance_cursor: bool = False,
    source: str = "harvest",
    log: _Logger | None = None,
) -> HarvestResult:
    """Run the offline harvest pipeline and write candidates to the store.

    Args:
        readers: trace readers to collect episodes from.
        distiller: agent that turns a cluster prompt into a candidate skill.
        store: candidate buffer to write into.
        cursor: optional watermark. `advance_cursor=False` (default) means harvest
            does NOT consume the watermark — it is a manual review tool the user
            can re-run; the `watch` daemon is what advances it.
        window: max new episodes to collect.
        min_cluster_size / similarity_threshold: clustering parameters.
        focus: which outcome to mine (FAILURE = capability gaps).
        max_candidates: cap clusters distilled (largest first).
        concurrency: parallel distiller calls.
        source: provenance tag stored on candidates.
        log: optional progress callback.

    Returns:
        HarvestResult with collected/clustered/candidate counts.
    """
    emit = log or (lambda _msg: None)

    collector = TraceCollector(readers, cursor=cursor)
    episodes = collector.collect(advance=advance_cursor, limit=window)
    emit(f"Collected {len(episodes)} episode(s).")

    clusters = cluster_episodes(
        episodes,
        focus=focus,
        min_cluster_size=min_cluster_size,
        similarity_threshold=similarity_threshold,
        max_clusters=max_candidates,
    )
    emit(f"Found {len(clusters)} cluster(s) of '{focus.value}' episodes "
         f"(min size {min_cluster_size}).")

    if not clusters:
        return HarvestResult(episodes_collected=len(episodes), clusters=[], candidates=[])

    candidates, failed, cost = await distill_clusters(
        clusters, distiller, store, source=source, concurrency=concurrency, log=emit,
    )

    return HarvestResult(
        episodes_collected=len(episodes),
        clusters=clusters,
        candidates=candidates,
        failed_distillations=failed,
        cost_usd=cost,
    )
