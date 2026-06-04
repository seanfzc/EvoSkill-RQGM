"""Fixtures for continuous-evolution tests.

Builders here mirror the *real* Harbor/ATIF trial layout confirmed against the
`arena-tracjectories/` corpus, so tests stay hermetic (no dependency on any
external path) while exercising the exact shapes the readers see in the wild.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _atif_v12_payload(*, with_tools: bool = True) -> dict:
    """An ATIF v1.2 trajectory (goose-style): no per-step timestamp/cost."""
    steps = [
        {
            "step_id": 1,
            "source": "agent",
            "message": "I'll find the answer.",
            "reasoning_content": "I need to find the average federal deficit for Q1 1994-1996.",
            "tool_calls": [
                {
                    "tool_call_id": "call_a",
                    "function_name": "analyze",
                    "arguments": {"path": "/app/resources", "max_depth": 2},
                }
            ]
            if with_tools
            else [],
            "observation": {
                "results": [{"source_call_id": "call_a", "content": "0 files found"}]
            }
            if with_tools
            else {},
        },
        {
            "step_id": 2,
            "source": "agent",
            "message": "[tool call]",
            "reasoning_content": "The answer is 12345.",
        },
    ]
    return {
        "schema_version": "ATIF-v1.2",
        "session_id": "sess-v12",
        "agent": {"name": "goose", "version": "1.36.0", "model_name": "openrouter/foo"},
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": 98662,
            "total_completion_tokens": 32888,
            "total_steps": 2,
            "extra": {"total_tokens": 131550},
        },
    }


def _atif_v16_payload() -> dict:
    """An ATIF v1.6 trajectory (opencode-style): per-step timestamp + total_cost_usd."""
    return {
        "schema_version": "ATIF-v1.6",
        "session_id": "ses_v16",
        "agent": {"name": "opencode", "version": "1.14.39", "model_name": "openrouter/bar"},
        "steps": [
            {
                "step_id": 1,
                "timestamp": "2026-05-06T13:40:00Z",
                "source": "agent",
                "model_name": "openrouter/bar",
                "message": "Working on it.",
                "reasoning_content": "Inspect the workspace first.",
                "tool_calls": [
                    {"tool_call_id": "c1", "function_name": "glob", "arguments": {"pattern": "**/*.json"}},
                    {"tool_call_id": "c2", "function_name": "bash", "arguments": {"command": "ls -la"}},
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "c1", "content": "/app/hand1.json"},
                        {"source_call_id": "c2", "content": "total 20"},
                    ]
                },
                "metrics": {"prompt_tokens": 100},
            },
            {
                "step_id": 2,
                "timestamp": "2026-05-06T13:41:00Z",
                "source": "agent",
                "message": "Done: result is 42.",
                "reasoning_content": "",
            },
        ],
        "final_metrics": {
            "total_prompt_tokens": 1562256,
            "total_completion_tokens": 18242,
            "total_cached_tokens": 1490683,
            "total_cost_usd": 0.154,
            "total_steps": 2,
        },
    }


def make_trial(
    jobs_root: Path,
    trial_name: str,
    *,
    payload: dict | None = None,
    reward: str | None = "1",
    task_name: str | None = None,
    goose_log: bool = False,
) -> Path:
    """Create a Harbor-style trial dir with the standard layout.

        <jobs_root>/<job>/<trial_name>/
            agent/trajectory.json   (+ optional goose.txt)
            verifier/reward.txt     (optional)
            result.json             (optional)
    """
    trial = jobs_root / "job-1" / trial_name
    agent_dir = trial / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    payload = payload if payload is not None else _atif_v12_payload()
    (agent_dir / "trajectory.json").write_text(json.dumps(payload))

    if reward is not None:
        verifier = trial / "verifier"
        verifier.mkdir(parents=True, exist_ok=True)
        (verifier / "reward.txt").write_text(reward)

    if task_name is not None:
        (trial / "result.json").write_text(json.dumps({"task_name": task_name}))

    if goose_log:
        lines = [
            {"type": "message", "message": {"id": "m1", "role": "assistant",
             "content": [{"type": "thinking", "thinking": "Let me "}]}},
            {"type": "message", "message": {"id": "m1", "role": "assistant",
             "content": [{"type": "thinking", "thinking": "think."}]}},
            {"type": "message", "message": {"id": "m1", "role": "assistant",
             "content": [{"type": "text", "text": "Answer: 42"}]}},
            {"type": "message", "message": {"id": "m2", "role": "assistant",
             "content": [{"type": "toolRequest", "id": "t1",
                          "toolCall": {"name": "bash"}}]}},
        ]
        (agent_dir / "goose.txt").write_text(
            "Loading recipe: harbor-task\n" + "\n".join(json.dumps(x) for x in lines)
        )

    return trial


def make_episode(
    episode_id: str,
    task_text: str,
    *,
    outcome=None,
    tools: list[str] | None = None,
    source: str = "harbor",
    final_output: str = "",
):
    """Build a TaskEpisode for clustering/harvest tests."""
    from src.continuous.episode import ActionStep, Outcome, TaskEpisode, ToolCall

    outcome = outcome if outcome is not None else Outcome.FAILURE
    actions = []
    if tools:
        actions = [ActionStep(
            step_id=1, source="agent",
            tool_calls=[ToolCall(function_name=t) for t in tools],
        )]
    return TaskEpisode(
        episode_id=episode_id,
        source=source,
        task_text=task_text,
        actions=actions,
        final_output=final_output,
        outcome=outcome,
    )


@pytest.fixture
def atif_v12():
    return _atif_v12_payload()


@pytest.fixture
def atif_v16():
    return _atif_v16_payload()


@pytest.fixture
def jobs_root(tmp_path):
    """A jobs root with three labeled trials: pass, fail, and partial."""
    root = tmp_path / "harbor_jobs"
    make_trial(root, "task-pass__AAA", reward="1", task_name="bench/task-pass")
    make_trial(root, "task-fail__BBB", reward="0", task_name="bench/task-fail")
    make_trial(root, "task-partial__CCC", reward="0.5", task_name="bench/task-partial")
    return root
