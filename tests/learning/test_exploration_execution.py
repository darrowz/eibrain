from __future__ import annotations

import json

from eibrain.learning.execution import execute_exploration_plan, write_outcomes


def test_execute_plan_marks_user_dependent_tasks_without_synthetic_memory() -> None:
    plan = {"tasks": [{"task_id": "t1", "task_type": "brain.respond", "reason": "need_memory"}]}

    outcomes = execute_exploration_plan(plan)

    assert outcomes[0]["status"] == "needs_real_user_evidence"
    assert outcomes[0]["metadata"]["synthetic_memory_written"] is False


def test_execute_skill_replay_reads_replay_results(tmp_path) -> None:
    replay = tmp_path / "replay-results.json"
    replay.write_text(json.dumps({"replay_results": [{"skill_id": "x", "passed": True}]}), encoding="utf-8")
    plan = {"tasks": [{"task_id": "t1", "task_type": "skill.replay", "reason": "candidate"}]}

    outcomes = execute_exploration_plan(plan, replay_results_path=replay)

    assert outcomes[0]["status"] == "success"
    assert outcomes[0]["next_action"] == "allow_canary_gate"
    assert outcomes[0]["evidence"][0]["passed_count"] == 1


def test_write_outcomes_writes_jsonl(tmp_path) -> None:
    target = tmp_path / "outcomes.jsonl"
    write_outcomes(target, [{"task_id": "t1", "status": "armed"}])

    assert json.loads(target.read_text(encoding="utf-8"))["task_id"] == "t1"
