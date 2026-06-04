"""Tests for the harvest pipeline and its CLI."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.continuous.candidates import CandidateStore
from src.continuous.cluster import cluster_episodes
from src.continuous.collector import (
    GooseRawReader,
    HarborTrajectoryReader,
    JsonlReader,
)
from src.continuous.episode import Outcome
from src.continuous.harvest import (
    build_distiller_query,
    build_readers,
    harvest,
    slugify_skill_name,
)

from .conftest import make_episode


class FakeDistiller:
    """Stand-in for src.harness.Agent: returns a canned distiller response."""

    def __init__(self, *, name="distilled", skill="---\nname: s\ndescription: d\n---\nbody",
                 raise_on=None, empty=False):
        self.name = name
        self.skill = skill
        self.raise_on = raise_on
        self.empty = empty
        self.calls = 0

    async def run(self, query):
        self.calls += 1
        if self.raise_on and self.raise_on in query:
            raise RuntimeError("boom")
        if self.empty:
            return SimpleNamespace(output=SimpleNamespace(candidate_skill="", skill_name=""), model="m")
        out = SimpleNamespace(
            skill_name=self.name, candidate_skill=self.skill,
            target_pattern="pat", reasoning="why",
        )
        return SimpleNamespace(output=out, model="fake-model")


def _failures(prefix, text, n):
    return [make_episode(f"{prefix}-{i}", text) for i in range(n)]


# ── pure helpers ───────────────────────────────────────────────────────────────


class TestSlugify:
    def test_basic(self):
        assert slugify_skill_name("My Skill!!") == "my-skill"

    def test_empty_falls_back(self):
        assert slugify_skill_name("   ") == "distilled-skill"

    def test_collapses_separators(self):
        assert slugify_skill_name("a__b  c") == "a-b-c"


class TestBuildReaders:
    def test_harbor_and_goose_and_jsonl(self, tmp_path):
        readers = build_readers(
            ["harbor", "goose", "jsonl"],
            traces_root=str(tmp_path), jsonl_path=str(tmp_path / "t.jsonl"),
        )
        kinds = [type(r) for r in readers]
        assert kinds == [HarborTrajectoryReader, GooseRawReader, JsonlReader]

    def test_skips_sources_without_path(self):
        assert build_readers(["harbor", "jsonl"], traces_root=None, jsonl_path=None) == []

    def test_skips_unknown_source(self, tmp_path):
        assert build_readers(["mystery"], traces_root=str(tmp_path)) == []


class TestBuildDistillerQuery:
    def test_includes_size_focus_terms_and_examples(self):
        eps = _failures("d", "compute the federal deficit fiscal quarter", 4)
        cluster = cluster_episodes(eps, min_cluster_size=3)[0]
        q = build_distiller_query(cluster)
        assert "4 episodes" in q
        assert "failure" in q
        assert "Example episodes" in q
        assert "deficit" in q

    def test_truncates_and_caps_examples(self):
        eps = _failures("d", "x " * 1000 + "deficit", 10)
        cluster = cluster_episodes(eps, min_cluster_size=3)[0]
        q = build_distiller_query(cluster, max_examples=2)
        assert "more similar episodes not shown" in q


# ── harvest() ────────────────────────────────────────────────────────────────


class TestHarvest:
    def test_writes_candidates(self, tmp_path):
        store = CandidateStore(tmp_path / "c")
        reader = _StaticReader(_failures("d", "compute the federal deficit fiscal quarter", 5))
        distiller = FakeDistiller()
        result = asyncio.run(harvest(
            readers=[reader], distiller=distiller, store=store,
            min_cluster_size=3, focus=Outcome.FAILURE,
        ))
        assert result.episodes_collected == 5
        assert result.num_candidates == 1
        assert result.failed_distillations == 0
        assert len(store.list()) == 1
        assert distiller.calls == 1

    def test_no_clusters_no_candidates(self, tmp_path):
        store = CandidateStore(tmp_path / "c")
        reader = _StaticReader(_failures("d", "unique solo task", 1))
        result = asyncio.run(harvest(
            readers=[reader], distiller=FakeDistiller(), store=store, min_cluster_size=3,
        ))
        assert result.num_candidates == 0
        assert result.clusters == []

    def test_distiller_exception_counts_as_failed(self, tmp_path):
        store = CandidateStore(tmp_path / "c")
        reader = _StaticReader(_failures("d", "compute the federal deficit fiscal quarter", 4))
        result = asyncio.run(harvest(
            readers=[reader], distiller=FakeDistiller(raise_on="episodes"),
            store=store, min_cluster_size=3,
        ))
        assert result.num_candidates == 0
        assert result.failed_distillations == 1

    def test_empty_skill_counts_as_failed(self, tmp_path):
        store = CandidateStore(tmp_path / "c")
        reader = _StaticReader(_failures("d", "compute the federal deficit fiscal quarter", 4))
        result = asyncio.run(harvest(
            readers=[reader], distiller=FakeDistiller(empty=True), store=store, min_cluster_size=3,
        ))
        assert result.num_candidates == 0
        assert result.failed_distillations == 1

    def test_max_candidates_caps_distillations(self, tmp_path):
        store = CandidateStore(tmp_path / "c")
        eps = (_failures("a", "federal deficit budget fiscal quarter", 4)
               + _failures("b", "customs duty receipts revenue bulletin", 4)
               + _failures("c", "treasury bond maturity schedule listing", 4))
        distiller = FakeDistiller()
        result = asyncio.run(harvest(
            readers=[_StaticReader(eps)], distiller=distiller, store=store,
            min_cluster_size=3, max_candidates=2,
        ))
        assert len(result.clusters) == 2
        assert distiller.calls == 2

    def test_focus_success(self, tmp_path):
        store = CandidateStore(tmp_path / "c")
        eps = [make_episode(f"s-{i}", "reusable winning workflow steps", outcome=Outcome.SUCCESS)
               for i in range(3)]
        result = asyncio.run(harvest(
            readers=[_StaticReader(eps)], distiller=FakeDistiller(), store=store,
            min_cluster_size=3, focus=Outcome.SUCCESS,
        ))
        assert result.num_candidates == 1
        assert store.list()[0].outcome_focus == "success"

    def test_candidate_carries_provenance(self, tmp_path):
        store = CandidateStore(tmp_path / "c")
        reader = _StaticReader(_failures("d", "compute the federal deficit fiscal quarter", 4))
        asyncio.run(harvest(readers=[reader], distiller=FakeDistiller(name="My Skill"),
                            store=store, min_cluster_size=3))
        c = store.list()[0]
        assert c.skill_name == "my-skill"          # slugified
        assert c.cluster_size == 4
        assert len(c.episode_ids) == 4
        assert c.model_name == "fake-model"
        assert c.source == "harvest"


class _StaticReader:
    """A TraceReader that yields a fixed list of episodes."""

    source = "static"

    def __init__(self, episodes):
        self._episodes = episodes

    def read_all(self):
        return iter(self._episodes)
