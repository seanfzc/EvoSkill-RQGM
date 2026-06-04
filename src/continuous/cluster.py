"""Cluster episodes by task similarity to surface recurring patterns.

The harvest pipeline distills a skill from a *cluster* of episodes, not a single
one — a skill earns its place by addressing a pattern that recurs, not a
one-off. This module groups episodes (failures by default) so the distiller sees
"here are 5 attempts that all struggled with the same kind of task."

Implementation is deliberately dependency-free: a small TF-IDF vectorizer over
each episode's task text + tool names, then greedy single-pass agglomerative
clustering by cosine similarity. No numpy/sklearn — the corpus sizes continuous
evolution targets (hundreds–low thousands of episodes per tick) are well within
pure-Python reach, and avoiding a heavy embedding dependency keeps Phase 1
self-contained. (Embedding-based similarity is a Phase 2 concern, for skill
dedup, where the quality/ρ tradeoff is worth a dependency.)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .episode import Outcome, TaskEpisode

# Compact English stopword set — enough to stop common words from dominating the
# TF-IDF space without pulling in a corpus dependency.
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have how i in into is it its of on or
    that the their then there these this to was were what when where which who
    will with you your do does did not but if can could should would may might
    must shall this that's about above after again all also am any because been
    before being below between both during each few more most other some such
    only own same so than too very s t just don now
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, dropping stopwords and very short tokens."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    ]


def _episode_tokens(episode: TaskEpisode) -> list[str]:
    """Bag of tokens describing what an episode was about.

    Combines the task text (what was asked) with the tool/function names the
    agent used (how it approached it) — two episodes that asked similar things
    *and* worked similarly are the strongest signal of a shared pattern.
    """
    tokens = _tokenize(episode.task_text)
    for step in episode.actions:
        for call in step.tool_calls:
            if call.function_name:
                tokens.append(call.function_name.lower())
    return tokens


@dataclass
class EpisodeCluster:
    """A group of similar episodes sharing an outcome focus."""

    episodes: list[TaskEpisode]
    outcome_focus: Outcome
    top_terms: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.episodes)

    @property
    def key(self) -> str:
        """Short human-readable label from the cluster's top terms."""
        return "-".join(self.top_terms[:4]) if self.top_terms else "cluster"

    @property
    def episode_ids(self) -> list[str]:
        return [e.episode_id for e in self.episodes]

    @property
    def task_ids(self) -> list[str]:
        return [e.task_id for e in self.episodes if e.task_id]


def _cosine(a: dict[str, float], b: dict[str, float], a_norm: float, b_norm: float) -> float:
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    # Iterate the smaller vector for the dot product.
    if len(a) > len(b):
        a, b = b, a
    dot = sum(weight * b.get(term, 0.0) for term, weight in a.items())
    return dot / (a_norm * b_norm)


def _norm(vec: dict[str, float]) -> float:
    return math.sqrt(sum(w * w for w in vec.values()))


def cluster_episodes(
    episodes: list[TaskEpisode],
    *,
    focus: Outcome = Outcome.FAILURE,
    min_cluster_size: int = 3,
    similarity_threshold: float = 0.3,
    max_clusters: int | None = None,
) -> list[EpisodeCluster]:
    """Group `focus`-outcome episodes into similarity clusters.

    Args:
        episodes: all collected episodes (mixed outcomes).
        focus: which outcome to cluster (FAILURE surfaces capability gaps;
            SUCCESS surfaces reusable winning patterns).
        min_cluster_size: drop clusters smaller than this — a pattern must recur.
        similarity_threshold: cosine ≥ this joins an episode to a cluster.
        max_clusters: keep only the largest N clusters (None = all).

    Returns:
        Clusters sorted by size (largest first). Empty if nothing qualifies.
    """
    pool = [e for e in episodes if e.outcome is focus]
    if not pool:
        return []

    # ── document frequencies for IDF ──
    docs_tokens = [_episode_tokens(e) for e in pool]
    n_docs = len(pool)
    df: Counter[str] = Counter()
    for tokens in docs_tokens:
        df.update(set(tokens))

    def tfidf(tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        vec: dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
            vec[term] = (count / total) * idf
        return vec

    vectors = [tfidf(tokens) for tokens in docs_tokens]
    norms = [_norm(v) for v in vectors]

    # ── greedy single-pass clustering ──
    # Each cluster keeps a centroid (summed member vectors) updated incrementally.
    cluster_members: list[list[int]] = []
    centroids: list[dict[str, float]] = []
    centroid_norms: list[float] = []

    for i in range(n_docs):
        vec, vnorm = vectors[i], norms[i]
        best_c, best_sim = -1, similarity_threshold
        for c, centroid in enumerate(centroids):
            sim = _cosine(vec, centroid, vnorm, centroid_norms[c])
            if sim >= best_sim:
                best_sim, best_c = sim, c
        if best_c == -1:
            cluster_members.append([i])
            centroids.append(dict(vec))
            centroid_norms.append(vnorm)
        else:
            cluster_members[best_c].append(i)
            centroid = centroids[best_c]
            for term, weight in vec.items():
                centroid[term] = centroid.get(term, 0.0) + weight
            centroid_norms[best_c] = _norm(centroid)

    # ── build, filter, rank ──
    clusters: list[EpisodeCluster] = []
    for members, centroid in zip(cluster_members, centroids):
        if len(members) < min_cluster_size:
            continue
        top_terms = [t for t, _ in sorted(centroid.items(), key=lambda kv: kv[1], reverse=True)[:6]]
        clusters.append(
            EpisodeCluster(
                episodes=[pool[i] for i in members],
                outcome_focus=focus,
                top_terms=top_terms,
            )
        )

    clusters.sort(key=lambda c: c.size, reverse=True)
    if max_clusters is not None:
        clusters = clusters[:max_clusters]
    return clusters
