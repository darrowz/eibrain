from __future__ import annotations

from eibrain.learning.exploration import plan_exploration_tasks


def test_plans_memory_exploration_when_meaningful_traces_are_missing() -> None:
    tasks = plan_exploration_tasks(experiences=[], registry_assets=[], max_tasks=3)

    assert tasks
    assert tasks[0]["reason"] == "meaningful_skill_trace_count_below_replay_minimum"
    assert "meaningful_event_v1" in tasks[0]["success_criteria"][0]


def test_skips_memory_seed_when_enough_meaningful_traces_exist() -> None:
    trace = {
        "meta": {"report_type": "skill_trace"},
        "content": {"meta": {"write_policy_version": "meaningful_event_v1", "trace_reason": "explicit_remember"}},
    }

    tasks = plan_exploration_tasks(experiences=[trace, trace], registry_assets=[], max_tasks=5)

    assert all(task["reason"] != "meaningful_skill_trace_count_below_replay_minimum" for task in tasks)


def test_plans_replay_when_candidates_exist() -> None:
    tasks = plan_exploration_tasks(registry_assets=[{"skill_id": "skill.x", "status": "candidate"}], max_tasks=5)

    assert any(task["reason"] == "candidate_skills_waiting_for_replay" for task in tasks)
