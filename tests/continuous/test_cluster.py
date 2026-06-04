"""Tests for episode clustering."""

from __future__ import annotations

from src.continuous.cluster import cluster_episodes
from src.continuous.episode import Outcome

from .conftest import make_episode


# Two coherent topics with distinct vocabulary.
def _deficit(i):
    return make_episode(f"def-{i}", "compute the federal budget deficit for fiscal year quarter")


def _receipts(i):
    return make_episode(f"rec-{i}", "extract customs duty receipts revenue from the bulletin")


class TestClusterEpisodes:
    def test_empty_input(self):
        assert cluster_episodes([]) == []

    def test_no_matching_focus(self):
        eps = [make_episode("a", "x y z", outcome=Outcome.SUCCESS)]
        assert cluster_episodes(eps, focus=Outcome.FAILURE) == []

    def test_groups_similar_separates_dissimilar(self):
        eps = [_deficit(i) for i in range(4)] + [_receipts(i) for i in range(4)]
        clusters = cluster_episodes(eps, min_cluster_size=3, similarity_threshold=0.3)
        assert len(clusters) == 2
        sizes = sorted(c.size for c in clusters)
        assert sizes == [4, 4]
        # Each cluster is internally homogeneous.
        for c in clusters:
            prefixes = {e.episode_id.split("-")[0] for e in c.episodes}
            assert len(prefixes) == 1

    def test_min_cluster_size_filters(self):
        eps = [_deficit(i) for i in range(4)] + [_receipts(0), _receipts(1)]
        clusters = cluster_episodes(eps, min_cluster_size=3, similarity_threshold=0.3)
        # only the 4-deficit cluster survives; the 2-receipt one is dropped
        assert len(clusters) == 1
        assert clusters[0].size == 4

    def test_focus_success(self):
        eps = [make_episode(f"s-{i}", "same winning approach reused", outcome=Outcome.SUCCESS)
               for i in range(3)]
        clusters = cluster_episodes(eps, focus=Outcome.SUCCESS, min_cluster_size=3)
        assert len(clusters) == 1
        assert clusters[0].outcome_focus is Outcome.SUCCESS

    def test_max_clusters_caps(self):
        eps = ([_deficit(i) for i in range(4)]
               + [_receipts(i) for i in range(4)]
               + [make_episode(f"t-{i}", "treasury bond maturity schedule listing") for i in range(4)])
        clusters = cluster_episodes(eps, min_cluster_size=3, max_clusters=2)
        assert len(clusters) == 2
        # largest-first ordering preserved
        assert clusters[0].size >= clusters[1].size

    def test_cluster_properties(self):
        eps = [_deficit(i) for i in range(3)]
        c = cluster_episodes(eps, min_cluster_size=3)[0]
        assert c.size == 3
        assert c.episode_ids == ["def-0", "def-1", "def-2"]
        assert isinstance(c.key, str) and c.key
        assert "deficit" in c.top_terms or "budget" in c.top_terms

    def test_high_threshold_yields_singletons_then_filtered(self):
        # Dissimilar episodes + near-1.0 threshold → no merges → all singletons,
        # then filtered out by min_cluster_size.
        eps = [
            make_episode("a", "alpha beta gamma delta"),
            make_episode("b", "epsilon zeta eta theta"),
            make_episode("c", "iota kappa lambda nu"),
            make_episode("d", "omicron pi rho sigma"),
        ]
        clusters = cluster_episodes(eps, min_cluster_size=2, similarity_threshold=0.999)
        assert clusters == []

    def test_tool_names_contribute_to_similarity(self):
        # identical tool usage + overlapping words cluster together
        eps = [
            make_episode("a", "read the report", tools=["grep", "bash"]),
            make_episode("b", "read the report", tools=["grep", "bash"]),
            make_episode("c", "read the report", tools=["grep", "bash"]),
        ]
        clusters = cluster_episodes(eps, min_cluster_size=3, similarity_threshold=0.3)
        assert len(clusters) == 1

    def test_task_ids_property_skips_none(self):
        e1 = make_episode("a", "compute deficit fiscal year")
        e2 = make_episode("b", "compute deficit fiscal year")
        e3 = make_episode("c", "compute deficit fiscal year")
        e1.task_id = "t1"
        clusters = cluster_episodes([e1, e2, e3], min_cluster_size=3)
        assert clusters[0].task_ids == ["t1"]
