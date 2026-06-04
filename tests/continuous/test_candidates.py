"""Tests for the candidate buffer."""

from __future__ import annotations

from src.continuous.candidates import Candidate, CandidateStore, make_candidate_id


def _candidate(**kw) -> Candidate:
    base = dict(
        candidate_id="skill-x-abc123",
        skill_name="skill-x",
        skill_markdown="---\nname: skill-x\ndescription: d\n---\nbody",
        target_pattern="pattern",
        episode_ids=["e1", "e2"],
        cluster_size=2,
    )
    base.update(kw)
    return Candidate(**base)


class TestMakeCandidateId:
    def test_stable_and_order_independent(self):
        a = make_candidate_id("my-skill", ["e2", "e1"])
        b = make_candidate_id("my-skill", ["e1", "e2"])
        assert a == b
        assert a.startswith("my-skill-")

    def test_different_episodes_differ(self):
        assert make_candidate_id("s", ["e1"]) != make_candidate_id("s", ["e2"])

    def test_empty_name_falls_back(self):
        assert make_candidate_id("", ["e1"]).startswith("skill-")


class TestCandidateStore:
    def test_save_writes_metadata_and_skill(self, tmp_path):
        store = CandidateStore(tmp_path / "cands")
        target = store.save(_candidate())
        assert (target / "candidate.json").is_file()
        assert (target / "SKILL.md").read_text().startswith("---")

    def test_save_sets_created_at(self, tmp_path):
        store = CandidateStore(tmp_path / "cands")
        store.save(_candidate(), timestamp="2026-01-01T00:00:00")
        got = store.get("skill-x-abc123")
        assert got.created_at == "2026-01-01T00:00:00"

    def test_save_preserves_existing_created_at(self, tmp_path):
        store = CandidateStore(tmp_path / "cands")
        store.save(_candidate(created_at="2020-01-01T00:00:00"), timestamp="2026-01-01T00:00:00")
        assert store.get("skill-x-abc123").created_at == "2020-01-01T00:00:00"

    def test_get_missing_returns_none(self, tmp_path):
        assert CandidateStore(tmp_path).get("nope") is None

    def test_get_corrupt_returns_none(self, tmp_path):
        store = CandidateStore(tmp_path / "cands")
        d = store.dir_for("broken")
        d.mkdir(parents=True)
        (d / "candidate.json").write_text("{bad json")
        assert store.get("broken") is None

    def test_list_newest_first(self, tmp_path):
        store = CandidateStore(tmp_path / "cands")
        store.save(_candidate(candidate_id="a", skill_name="a"), timestamp="2026-01-01T00:00:00")
        store.save(_candidate(candidate_id="b", skill_name="b"), timestamp="2026-02-01T00:00:00")
        ids = [c.candidate_id for c in store.list()]
        assert ids == ["b", "a"]

    def test_list_ignores_non_dirs_and_missing_root(self, tmp_path):
        assert CandidateStore(tmp_path / "absent").list() == []
        store = CandidateStore(tmp_path / "cands")
        store.root.mkdir(parents=True)
        (store.root / "stray.txt").write_text("noise")
        store.save(_candidate())
        assert len(store.list()) == 1

    def test_set_status(self, tmp_path):
        store = CandidateStore(tmp_path / "cands")
        store.save(_candidate())
        updated = store.set_status("skill-x-abc123", "graduated")
        assert updated.status == "graduated"
        assert store.get("skill-x-abc123").status == "graduated"

    def test_set_status_missing(self, tmp_path):
        assert CandidateStore(tmp_path).set_status("nope", "rejected") is None
