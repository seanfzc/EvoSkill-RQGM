"""Tests for trace collection: ATIF parsing, readers, cursor, collector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.continuous.collector import (
    GooseRawReader,
    HarborTrajectoryReader,
    JsonlReader,
    TraceCollector,
    TraceCursor,
    TraceReadError,
    parse_atif_trajectory,
    read_reward_file,
)
from src.continuous.episode import Outcome, SignalKind

from .conftest import make_trial


# ── ATIF parser ────────────────────────────────────────────────────────────────


class TestParseAtif:
    def test_v12_common_fields(self, atif_v12):
        ep = parse_atif_trajectory(atif_v12, episode_id="e1")
        assert ep.episode_id == "e1"
        assert ep.agent_name == "goose"
        assert ep.model_name == "openrouter/foo"
        assert ep.prompt_tokens == 98662
        assert ep.completion_tokens == 32888
        assert ep.num_steps == 2
        assert ep.cost_usd == 0.0  # v1.2 has no total_cost_usd
        assert ep.timestamp is None  # v1.2 has no per-step timestamp
        assert ep.extra["schema_version"] == "ATIF-v1.2"

    def test_v12_links_tool_call_to_observation(self, atif_v12):
        ep = parse_atif_trajectory(atif_v12, episode_id="e1")
        first = ep.actions[0]
        assert first.tool_calls[0].function_name == "analyze"
        assert first.tool_calls[0].observation == "0 files found"

    def test_task_text_from_first_reasoning(self, atif_v12):
        ep = parse_atif_trajectory(atif_v12, episode_id="e1")
        assert ep.task_text.startswith("I need to find the average")

    def test_final_output_skips_tool_call_placeholder(self, atif_v12):
        # last step message is "[tool call]" → falls back to its reasoning
        ep = parse_atif_trajectory(atif_v12, episode_id="e1")
        assert ep.final_output == "The answer is 12345."

    def test_v16_timestamp_and_cost(self, atif_v16):
        ep = parse_atif_trajectory(atif_v16, episode_id="e2")
        assert ep.agent_name == "opencode"
        assert ep.cost_usd == 0.154
        assert ep.timestamp == "2026-05-06T13:41:00Z"  # last step with a timestamp
        assert ep.extra["schema_version"] == "ATIF-v1.6"

    def test_v16_multiple_tool_calls_each_linked(self, atif_v16):
        ep = parse_atif_trajectory(atif_v16, episode_id="e2")
        tcs = ep.actions[0].tool_calls
        assert [t.function_name for t in tcs] == ["glob", "bash"]
        assert tcs[0].observation == "/app/hand1.json"
        assert tcs[1].observation == "total 20"

    def test_handles_empty_and_garbage(self):
        ep = parse_atif_trajectory({}, episode_id="e")
        assert ep.actions == []
        assert ep.num_steps == 0
        # Non-list steps / non-dict agent must not crash.
        ep2 = parse_atif_trajectory({"steps": "nope", "agent": "nope"}, episode_id="e")
        assert ep2.actions == []
        assert ep2.agent_name is None

    def test_num_steps_falls_back_to_len(self):
        payload = {"steps": [{"source": "agent", "message": "a"}], "final_metrics": {}}
        ep = parse_atif_trajectory(payload, episode_id="e")
        assert ep.num_steps == 1

    def test_non_numeric_metrics_coerce_to_zero(self):
        payload = {
            "steps": [{"source": "agent", "message": "a"}],
            "final_metrics": {
                "total_prompt_tokens": "oops",
                "total_cost_usd": None,
            },
        }
        ep = parse_atif_trajectory(payload, episode_id="e")
        assert ep.prompt_tokens == 0
        assert ep.cost_usd == 0.0

    def test_task_text_falls_back_to_message_when_no_reasoning(self):
        payload = {"steps": [{"source": "agent", "message": "Just do it", "reasoning_content": ""}]}
        ep = parse_atif_trajectory(payload, episode_id="e")
        assert ep.task_text == "Just do it"

    def test_task_text_empty_when_only_placeholder(self):
        payload = {"steps": [{"source": "agent", "message": "[tool call]"}]}
        ep = parse_atif_trajectory(payload, episode_id="e")
        assert ep.task_text == ""

    def test_tool_calls_tolerate_bad_shapes(self):
        payload = {
            "steps": [{
                "source": "agent",
                "tool_calls": ["not-a-dict", {"function_name": "f", "arguments": "scalar"}],
                "observation": "not-a-dict",
            }]
        }
        ep = parse_atif_trajectory(payload, episode_id="e")
        # the string entry is dropped; the dict entry survives with wrapped args
        assert len(ep.actions[0].tool_calls) == 1
        assert ep.actions[0].tool_calls[0].arguments == {"value": "scalar"}

    def test_non_agent_steps_excluded_from_final_output(self):
        payload = {"steps": [
            {"source": "agent", "message": "hello"},
            {"source": "environment", "message": "env noise"},
        ]}
        ep = parse_atif_trajectory(payload, episode_id="e")
        assert ep.final_output == "hello"


# ── read_reward_file ─────────────────────────────────────────────────────────


class TestReadRewardFile:
    def test_reads_value(self, tmp_path):
        (tmp_path / "verifier").mkdir()
        (tmp_path / "verifier" / "reward.txt").write_text("0.75\n")
        assert read_reward_file(tmp_path) == 0.75

    def test_missing_returns_none(self, tmp_path):
        assert read_reward_file(tmp_path) is None

    def test_empty_returns_none(self, tmp_path):
        (tmp_path / "verifier").mkdir()
        (tmp_path / "verifier" / "reward.txt").write_text("   ")
        assert read_reward_file(tmp_path) is None

    def test_unparseable_returns_none(self, tmp_path):
        (tmp_path / "verifier").mkdir()
        (tmp_path / "verifier" / "reward.txt").write_text("not-a-number")
        assert read_reward_file(tmp_path) is None


# ── HarborTrajectoryReader ─────────────────────────────────────────────────────


class TestHarborTrajectoryReader:
    def test_discovers_and_labels_all_trials(self, jobs_root):
        eps = list(HarborTrajectoryReader(jobs_root).read_all())
        assert len(eps) == 3
        by_id = {e.task_id: e for e in eps}
        assert by_id["bench/task-pass"].outcome is Outcome.SUCCESS
        assert by_id["bench/task-fail"].outcome is Outcome.FAILURE
        # partial (0.5) is a failure under the default strict threshold
        assert by_id["bench/task-partial"].outcome is Outcome.FAILURE
        assert by_id["bench/task-partial"].signal.value == 0.5

    def test_episode_id_is_trial_dir_name(self, jobs_root):
        eps = {e.episode_id for e in HarborTrajectoryReader(jobs_root).read_all()}
        assert "task-pass__AAA" in eps

    def test_reward_signal_is_verifier_kind(self, jobs_root):
        ep = next(HarborTrajectoryReader(jobs_root).read_all())
        assert ep.signal.kind is SignalKind.VERIFIER_REWARD
        assert ep.signal.confidence == 1.0

    def test_no_reward_yields_unknown(self, tmp_path):
        root = tmp_path / "jobs"
        make_trial(root, "t__X", reward=None, task_name="b/t")
        ep = next(HarborTrajectoryReader(root).read_all())
        assert ep.outcome is Outcome.UNKNOWN
        assert ep.signal.kind is SignalKind.NONE

    def test_custom_success_threshold(self, jobs_root):
        eps = {e.task_id: e for e in HarborTrajectoryReader(jobs_root, success_threshold=0.5).read_all()}
        assert eps["bench/task-partial"].outcome is Outcome.SUCCESS

    def test_missing_root_yields_nothing(self, tmp_path):
        assert list(HarborTrajectoryReader(tmp_path / "nope").read_all()) == []

    def test_malformed_trajectory_is_skipped(self, tmp_path):
        root = tmp_path / "jobs"
        make_trial(root, "good__A", reward="1", task_name="b/good")
        bad = root / "job-1" / "bad__B" / "agent"
        bad.mkdir(parents=True)
        (bad / "trajectory.json").write_text("{not valid json")
        eps = list(HarborTrajectoryReader(root).read_all())
        assert {e.episode_id for e in eps} == {"good__A"}

    def test_parse_trial_raises_on_garbage(self, tmp_path):
        agent = tmp_path / "t" / "agent"
        agent.mkdir(parents=True)
        (agent / "trajectory.json").write_text("[]")  # JSON, but not an object
        with pytest.raises(TraceReadError):
            HarborTrajectoryReader(tmp_path).parse_trial(tmp_path / "t")

    def test_task_id_from_result_task_id_path(self, tmp_path):
        trial = make_trial(tmp_path / "jobs", "t__X", reward="1", task_name=None)
        (trial / "result.json").write_text(
            json.dumps({"task_id": {"path": "/data/datasets/officeqa/officeqa-uid0132"}})
        )
        ep = HarborTrajectoryReader(tmp_path / "jobs").parse_trial(trial)
        assert ep.task_id == "officeqa-uid0132"

    def test_task_id_none_when_no_result(self, tmp_path):
        trial = make_trial(tmp_path / "jobs", "t__X", reward="1", task_name=None)
        ep = HarborTrajectoryReader(tmp_path / "jobs").parse_trial(trial)
        assert ep.task_id is None

    def test_corrupt_result_json_tolerated(self, tmp_path):
        trial = make_trial(tmp_path / "jobs", "t__X", reward="1", task_name=None)
        (trial / "result.json").write_text("{bad json")
        ep = HarborTrajectoryReader(tmp_path / "jobs").parse_trial(trial)
        assert ep.task_id is None  # falls back gracefully


# ── GooseRawReader ─────────────────────────────────────────────────────────────


class TestGooseRawReader:
    def test_reconstructs_messages(self, tmp_path):
        root = tmp_path / "jobs"
        make_trial(root, "t__X", reward="1", task_name="b/t", goose_log=True)
        eps = list(GooseRawReader(root).read_all())
        assert len(eps) == 1
        ep = eps[0]
        assert ep.source == "goose"
        assert ep.outcome is Outcome.SUCCESS  # reward picked up from sibling verifier
        # m1 accumulated thinking "Let me think." and text "Answer: 42"
        first = ep.actions[0]
        assert first.reasoning == "Let me think."
        assert first.message == "Answer: 42"
        assert ep.final_output == "Answer: 42"
        # m2 carried a tool request
        assert ep.actions[1].tool_calls[0].function_name == "bash"

    def test_skips_malformed_lines(self, tmp_path):
        root = tmp_path / "jobs"
        trial = make_trial(root, "t__X", reward=None, goose_log=True)
        log = trial / "agent" / "goose.txt"
        log.write_text(log.read_text() + "\nnot json\n{}\n")
        eps = list(GooseRawReader(root).read_all())
        assert len(eps) == 1  # garbage lines ignored, episode still built

    def test_empty_log_skipped(self, tmp_path):
        root = tmp_path / "jobs"
        trial = make_trial(root, "t__X", reward=None, goose_log=True)
        (trial / "agent" / "goose.txt").write_text("Loading recipe\n")  # no messages
        assert list(GooseRawReader(root).read_all()) == []

    def test_missing_root_yields_nothing(self, tmp_path):
        assert list(GooseRawReader(tmp_path / "nope").read_all()) == []


# ── JsonlReader ────────────────────────────────────────────────────────────────


class TestJsonlReader:
    def _write(self, tmp_path, records) -> Path:
        p = tmp_path / "traces.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records))
        return p

    def test_reward_record(self, tmp_path):
        p = self._write(tmp_path, [{"task": "do x", "output": "y", "reward": 1.0, "episode_id": "z"}])
        ep = next(JsonlReader(p).read_all())
        assert ep.episode_id == "z"
        assert ep.task_text == "do x"
        assert ep.final_output == "y"
        assert ep.outcome is Outcome.SUCCESS
        assert ep.signal.kind is SignalKind.VERIFIER_REWARD

    def test_explicit_outcome_without_reward(self, tmp_path):
        p = self._write(tmp_path, [{"task": "t", "outcome": "success"}])
        ep = next(JsonlReader(p).read_all())
        assert ep.outcome is Outcome.SUCCESS
        assert ep.signal.confidence == 0.5  # implicit, not ground truth

    def test_no_signal_when_neither(self, tmp_path):
        p = self._write(tmp_path, [{"task": "t"}])
        ep = next(JsonlReader(p).read_all())
        assert ep.outcome is Outcome.UNKNOWN

    @pytest.mark.parametrize(
        "raw,expected",
        [("fail", Outcome.FAILURE), ("passed", Outcome.SUCCESS),
         ("0", Outcome.FAILURE), ("weird", Outcome.UNKNOWN)],
    )
    def test_explicit_outcome_strings(self, tmp_path, raw, expected):
        p = self._write(tmp_path, [{"task": "t", "outcome": raw}])
        ep = next(JsonlReader(p).read_all())
        assert ep.outcome is expected

    def test_parses_steps_and_skills(self, tmp_path):
        p = self._write(tmp_path, [{
            "task": "t",
            "skills_active": ["s1", "s2"],
            "steps": [{"source": "agent", "message": "m", "reasoning": "r"}],
        }])
        ep = next(JsonlReader(p).read_all())
        assert ep.skills_active == ["s1", "s2"]
        assert ep.actions[0].reasoning == "r"
        assert ep.num_steps == 1

    def test_missing_task_skipped(self, tmp_path):
        p = self._write(tmp_path, [{"output": "no task"}, {"task": "ok"}])
        eps = list(JsonlReader(p).read_all())
        assert len(eps) == 1
        assert eps[0].task_text == "ok"

    def test_blank_and_nonjson_lines_skipped(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('\n  \nnot json\n{"task": "ok"}\n[1,2,3]\n')
        eps = list(JsonlReader(p).read_all())
        assert len(eps) == 1

    def test_derived_episode_id_is_stable(self, tmp_path):
        rec = [{"task": "same"}]
        p1 = self._write(tmp_path, rec)
        a = next(JsonlReader(p1).read_all())
        b = next(JsonlReader(p1).read_all())
        assert a.episode_id == b.episode_id  # deterministic from path+index+task

    def test_missing_file_yields_nothing(self, tmp_path):
        assert list(JsonlReader(tmp_path / "nope.jsonl").read_all()) == []

    def test_custom_source_tag(self, tmp_path):
        p = self._write(tmp_path, [{"task": "t"}])
        ep = next(JsonlReader(p, source="hook").read_all())
        assert ep.source == "hook"


# ── TraceCursor ────────────────────────────────────────────────────────────────


class TestTraceCursor:
    def test_starts_empty(self, tmp_path):
        cur = TraceCursor(tmp_path / "cursor.json")
        assert len(cur) == 0
        assert not cur.seen("x")

    def test_mark_save_reload(self, tmp_path):
        path = tmp_path / "sub" / "cursor.json"
        cur = TraceCursor(path)
        cur.mark("a")
        cur.mark("b")
        cur.save()
        assert path.is_file()
        reloaded = TraceCursor(path)
        assert reloaded.seen("a") and reloaded.seen("b")
        assert len(reloaded) == 2

    def test_corrupt_file_treated_as_empty(self, tmp_path):
        path = tmp_path / "cursor.json"
        path.write_text("{not json")
        cur = TraceCursor(path)
        assert len(cur) == 0

    def test_non_dict_payload_treated_as_empty(self, tmp_path):
        path = tmp_path / "cursor.json"
        path.write_text("[1,2,3]")
        assert len(TraceCursor(path)) == 0


# ── TraceCollector ─────────────────────────────────────────────────────────────


class TestTraceCollector:
    def test_collects_all_with_no_cursor(self, jobs_root):
        col = TraceCollector([HarborTrajectoryReader(jobs_root)])
        assert len(col.collect()) == 3

    def test_reader_priority_dedup(self, tmp_path):
        # One trial with BOTH an ATIF trajectory and a goose log → same episode_id.
        root = tmp_path / "jobs"
        make_trial(root, "t__X", reward="1", task_name="b/t", goose_log=True)
        col = TraceCollector([HarborTrajectoryReader(root), GooseRawReader(root)])
        eps = col.collect()
        assert len(eps) == 1
        assert eps[0].source == "harbor"  # earlier reader wins

    def test_cursor_dedup_across_batches(self, jobs_root, tmp_path):
        cur = TraceCursor(tmp_path / "cursor.json")
        col = TraceCollector([HarborTrajectoryReader(jobs_root)], cursor=cur)
        assert len(col.collect()) == 3
        assert len(col.collect()) == 0  # all watermarked
        assert len(cur) == 3

    def test_advance_false_does_not_persist(self, jobs_root, tmp_path):
        path = tmp_path / "cursor.json"
        cur = TraceCursor(path)
        col = TraceCollector([HarborTrajectoryReader(jobs_root)], cursor=cur)
        col.collect(advance=False)
        assert len(cur) == 0
        assert not path.is_file()

    def test_limit_caps_new_episodes(self, jobs_root):
        col = TraceCollector([HarborTrajectoryReader(jobs_root)])
        assert len(col.collect(limit=2)) == 2

    def test_limit_across_readers(self, tmp_path):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        make_trial(root_a, "a1__X", reward="1", task_name="x/a1")
        make_trial(root_b, "b1__Y", reward="1", task_name="x/b1", goose_log=True)
        col = TraceCollector([HarborTrajectoryReader(root_a), GooseRawReader(root_b)])
        assert len(col.collect(limit=1)) == 1
