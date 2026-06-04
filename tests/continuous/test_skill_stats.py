"""Tests for SkillStats, the store, and credit assignment."""

from __future__ import annotations

from src.continuous.episode import Outcome
from src.continuous.skill_stats import (
    SkillStats,
    SkillStatsStore,
    assign_credit,
    assign_credit_batch,
)

from .conftest import make_episode


class TestSkillStats:
    def test_success_rate(self):
        s = SkillStats(name="x", episodes_active=4, success_when_active=1)
        assert s.success_rate == 0.25

    def test_success_rate_zero_episodes(self):
        assert SkillStats(name="x").success_rate == 0.0


class TestSkillStatsStore:
    def test_get_creates_empty(self, tmp_path):
        store = SkillStatsStore(tmp_path / "s.json")
        assert not store.has("x")
        s = store.get("x")
        assert s.name == "x" and s.episodes_active == 0
        assert store.has("x")

    def test_save_load_roundtrip(self, tmp_path):
        store = SkillStatsStore(tmp_path / "sub" / "s.json")
        s = store.get("x")
        s.episodes_active = 3
        store.upsert(s)
        store.save()
        reloaded = SkillStatsStore(tmp_path / "sub" / "s.json")
        assert reloaded.get("x").episodes_active == 3

    def test_corrupt_file_empty(self, tmp_path):
        (tmp_path / "s.json").write_text("{bad")
        assert SkillStatsStore(tmp_path / "s.json").all() == []

    def test_remove(self, tmp_path):
        store = SkillStatsStore(tmp_path / "s.json")
        store.get("x")
        store.remove("x")
        assert not store.has("x")


class TestAssignCredit:
    def test_sole_skill_gets_full_credit_on_success(self, tmp_path):
        store = SkillStatsStore(tmp_path / "s.json")
        ep = make_episode("e", "t", outcome=Outcome.SUCCESS)
        ep.skills_active = ["solo"]
        assert assign_credit(store, ep) is True
        s = store.get("solo")
        assert s.episodes_active == 1
        assert s.success_when_active == 1
        assert s.contribution == 1.0

    def test_credit_shared_among_active_skills(self, tmp_path):
        store = SkillStatsStore(tmp_path / "s.json")
        ep = make_episode("e", "t", outcome=Outcome.SUCCESS)
        ep.skills_active = ["a", "b"]
        assign_credit(store, ep)
        assert store.get("a").contribution == 0.5
        assert store.get("b").contribution == 0.5

    def test_failure_counts_usage_but_no_contribution(self, tmp_path):
        store = SkillStatsStore(tmp_path / "s.json")
        ep = make_episode("e", "t", outcome=Outcome.FAILURE)
        ep.skills_active = ["a"]
        assign_credit(store, ep)
        s = store.get("a")
        assert s.episodes_active == 1
        assert s.success_when_active == 0
        assert s.contribution == 0.0

    def test_no_active_skills_skipped(self, tmp_path):
        store = SkillStatsStore(tmp_path / "s.json")
        ep = make_episode("e", "t", outcome=Outcome.SUCCESS)  # skills_active empty
        assert assign_credit(store, ep) is False
        assert store.all() == []

    def test_batch_counts_and_saves(self, tmp_path):
        store = SkillStatsStore(tmp_path / "s.json")
        eps = [make_episode(f"e{i}", "t", outcome=Outcome.SUCCESS) for i in range(3)]
        for e in eps:
            e.skills_active = ["a"]
        eps.append(make_episode("none", "t"))  # no skills → not credited
        credited = assign_credit_batch(store, eps)
        assert credited == 3
        assert store.get("a").episodes_active == 3
        assert (tmp_path / "s.json").is_file()  # saved
