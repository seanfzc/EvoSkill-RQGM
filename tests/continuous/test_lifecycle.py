"""Tests for skill-library lifecycle operations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.continuous.candidates import CandidateStore
from src.continuous.library import Skill
from src.continuous.lifecycle import (
    archive_skill,
    build_merge_query,
    evaluate_deprecation,
    find_duplicates,
    propose_merge,
    restore_skill,
    select_skills,
)
from src.continuous.similarity import SimilarityBackend
from src.continuous.skill_stats import SkillStatsStore


def _skill(name: str) -> Skill:
    return Skill(name=name, description=f"desc {name}", body="body", path=Path(f"{name}/SKILL.md"))


class MatrixBackend(SimilarityBackend):
    """Similarity driven by an explicit matrix keyed on skill .text."""

    name = "matrix"

    def __init__(self, texts: list[str], matrix: list[list[float]]):
        self.idx = {t: i for i, t in enumerate(texts)}
        self.m = matrix

    def pairwise(self, texts):
        return [[self.m[self.idx[a]][self.idx[b]] for b in texts] for a in texts]

    def rank(self, query, candidates):
        scored = [(i, self.m[self.idx[query]][self.idx[c]]) for i, c in enumerate(candidates)]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored


class TestFindDuplicates:
    def test_groups_above_threshold(self):
        skills = [_skill("a"), _skill("b"), _skill("c")]
        texts = [s.text for s in skills]
        # a~b high, c separate
        m = [[1.0, 0.95, 0.1], [0.95, 1.0, 0.1], [0.1, 0.1, 1.0]]
        groups = find_duplicates(skills, MatrixBackend(texts, m), threshold=0.88)
        assert len(groups) == 1
        assert set(groups[0].names) == {"a", "b"}
        assert groups[0].max_similarity == 0.95

    def test_union_find_transitivity(self):
        skills = [_skill("a"), _skill("b"), _skill("c")]
        texts = [s.text for s in skills]
        # a~b and b~c high, a~c low → all three transitively grouped
        m = [[1.0, 0.9, 0.4], [0.9, 1.0, 0.9], [0.4, 0.9, 1.0]]
        groups = find_duplicates(skills, MatrixBackend(texts, m), threshold=0.85)
        assert len(groups) == 1
        assert set(groups[0].names) == {"a", "b", "c"}

    def test_no_duplicates(self):
        skills = [_skill("a"), _skill("b")]
        texts = [s.text for s in skills]
        m = [[1.0, 0.2], [0.2, 1.0]]
        assert find_duplicates(skills, MatrixBackend(texts, m), threshold=0.88) == []

    def test_fewer_than_two_skills(self):
        assert find_duplicates([_skill("a")], MatrixBackend(["x"], [[1.0]]), threshold=0.5) == []


class TestSelectSkills:
    def test_top_k_ordering(self):
        skills = [_skill("a"), _skill("b"), _skill("c")]
        texts = ["q", *[s.text for s in skills]]
        # q closest to b, then a, then c
        m = [
            [1.0, 0.5, 0.9, 0.2],
            [0.5, 1.0, 0.0, 0.0],
            [0.9, 0.0, 1.0, 0.0],
            [0.2, 0.0, 0.0, 1.0],
        ]
        matches = select_skills(skills, "q", MatrixBackend(texts, m), k=2)
        assert [m.skill.name for m in matches] == ["b", "a"]

    def test_min_score_filters(self):
        skills = [_skill("a"), _skill("b")]
        texts = ["q", _skill("a").text, _skill("b").text]
        m = [[1.0, 0.9, 0.1], [0.9, 1.0, 0.0], [0.1, 0.0, 1.0]]
        matches = select_skills(skills, "q", MatrixBackend(texts, m), k=5, min_score=0.5)
        assert [x.skill.name for x in matches] == ["a"]

    def test_empty_library(self):
        assert select_skills([], "q", MatrixBackend(["q"], [[1.0]])) == []


class TestEvaluateDeprecation:
    def test_strikes_accrue_to_candidate(self, tmp_path):
        store = SkillStatsStore(tmp_path / "s.json")
        s = store.get("dead")
        s.episodes_active = 5  # active but zero contribution
        store.upsert(s)
        rep = None
        for _ in range(3):
            rep = evaluate_deprecation(store, ["dead"], baseline=0.0, strikes_limit=3)
        assert rep.candidates == ["dead"]
        assert store.get("dead").deprecation_strikes == 3

    def test_recovery_resets_strikes(self, tmp_path):
        store = SkillStatsStore(tmp_path / "s.json")
        s = store.get("x")
        s.episodes_active = 2
        s.deprecation_strikes = 2
        store.upsert(s)
        # now it has contribution > baseline
        s2 = store.get("x")
        s2.contribution = 1.0
        store.upsert(s2)
        rep = evaluate_deprecation(store, ["x"], baseline=0.0, strikes_limit=3)
        assert "x" in rep.recovered
        assert store.get("x").deprecation_strikes == 0

    def test_unused_skill_not_struck(self, tmp_path):
        store = SkillStatsStore(tmp_path / "s.json")
        rep = evaluate_deprecation(store, ["never"], baseline=0.0, strikes_limit=3)
        assert rep.unused == ["never"]
        assert rep.struck == []
        assert rep.candidates == []


class TestArchive:
    def _make_skill_dir(self, root, name):
        d = root / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("x")
        return d

    def test_archive_and_restore(self, tmp_path):
        sd = tmp_path / "skills"
        self._make_skill_dir(sd, "foo")
        arch = tmp_path / "deprecated"
        archive_skill(sd, "foo", arch)
        assert (arch / "foo" / "SKILL.md").is_file()
        assert not (sd / "foo").exists()
        restore_skill(arch, "foo", sd)
        assert (sd / "foo" / "SKILL.md").is_file()
        assert not (arch / "foo").exists()

    def test_archive_missing_raises(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            archive_skill(tmp_path / "skills", "nope", tmp_path / "arch")

    def test_archive_overwrites_existing(self, tmp_path):
        sd = tmp_path / "skills"
        arch = tmp_path / "deprecated"
        self._make_skill_dir(sd, "foo")
        self._make_skill_dir(arch, "foo")  # stale archive copy
        archive_skill(sd, "foo", arch)  # should overwrite, not error
        assert (arch / "foo" / "SKILL.md").is_file()


class _FakeMerger:
    def __init__(self, *, empty=False):
        self.empty = empty

    async def run(self, query):
        skill = "" if self.empty else "---\nname: merged\ndescription: m\n---\nrule"
        return SimpleNamespace(
            output=SimpleNamespace(skill_name="Merged Skill", candidate_skill=skill,
                                   target_pattern="", reasoning="r"),
            model="fake",
        )


class TestMerge:
    def test_build_merge_query_lists_skills(self):
        q = build_merge_query([_skill("a"), _skill("b")])
        assert "Merge the following 2" in q
        assert "Skill 1: a" in q and "Skill 2: b" in q

    def test_propose_merge_writes_candidate(self, tmp_path):
        store = CandidateStore(tmp_path / "c")
        cand = asyncio.run(propose_merge([_skill("a"), _skill("b")], _FakeMerger(), store))
        assert cand is not None
        assert cand.skill_name == "merged-skill"
        assert cand.source == "merge"
        assert cand.extra["merged_from"] == ["a", "b"]
        assert len(store.list()) == 1

    def test_propose_merge_needs_two(self, tmp_path):
        store = CandidateStore(tmp_path / "c")
        assert asyncio.run(propose_merge([_skill("a")], _FakeMerger(), store)) is None

    def test_propose_merge_empty_output(self, tmp_path):
        store = CandidateStore(tmp_path / "c")
        cand = asyncio.run(propose_merge([_skill("a"), _skill("b")], _FakeMerger(empty=True), store))
        assert cand is None
        assert store.list() == []
