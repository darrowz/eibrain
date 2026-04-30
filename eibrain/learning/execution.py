from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExplorationOutcome:
    task_id: str
    task_type: str
    status: str
    reason: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    next_action: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def execute_exploration_plan(
    plan: dict[str, Any] | list[dict[str, Any]],
    *,
    replay_results_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    max_tasks: int = 5,
) -> list[dict[str, Any]]:
    tasks = _tasks(plan)[: max(0, max_tasks)]
    outcomes = [execute_task(task, replay_results_path=replay_results_path, registry_path=registry_path) for task in tasks]
    return [outcome.to_dict() for outcome in outcomes]


def execute_task(
    task: dict[str, Any],
    *,
    replay_results_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> ExplorationOutcome:
    task_type = str(task.get("task_type") or "")
    task_id = str(task.get("task_id") or "")
    reason = str(task.get("reason") or "")
    if task_type == "skill.replay":
        return _execute_skill_replay(task_id, task_type, reason, replay_results_path=replay_results_path)
    if task_type in {"brain.respond", "world_knowledge.seed"}:
        return ExplorationOutcome(
            task_id=task_id,
            task_type=task_type,
            status="needs_real_user_evidence",
            reason=reason,
            next_action="wait_for_user_approved_meaningful_event",
            metadata={"write_policy_version": "meaningful_event_v1", "synthetic_memory_written": False},
        )
    if task_type == "vision.semantic_change":
        return ExplorationOutcome(
            task_id=task_id,
            task_type=task_type,
            status="armed",
            reason=reason,
            next_action="observe_next_semantic_visual_change_with_dedupe",
            metadata={"write_policy_version": "meaningful_event_v1", "synthetic_memory_written": False},
        )
    return ExplorationOutcome(
        task_id=task_id,
        task_type=task_type,
        status="unsupported_task_type",
        reason=reason,
        next_action="planner_review",
    )


def write_outcomes(path: str | Path, outcomes: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in outcomes), encoding="utf-8")


def _execute_skill_replay(task_id: str, task_type: str, reason: str, *, replay_results_path: str | Path | None) -> ExplorationOutcome:
    if replay_results_path is None:
        return ExplorationOutcome(task_id, task_type, "blocked_missing_replay_path", reason, next_action="run_eitraining_loop")
    path = Path(replay_results_path)
    if not path.exists() or path.stat().st_size == 0:
        return ExplorationOutcome(task_id, task_type, "blocked_missing_replay_results", reason, next_action="run_eitraining_loop")
    replay_items = _read_items(path)
    passed = [item for item in replay_items if isinstance(item, dict) and item.get("passed") is True]
    failed = [item for item in replay_items if isinstance(item, dict) and item.get("passed") is False]
    status = "success" if passed and not failed else "blocked_replay_not_passing"
    return ExplorationOutcome(
        task_id=task_id,
        task_type=task_type,
        status=status,
        reason=reason,
        evidence=[{"path": str(path), "passed_count": len(passed), "failed_count": len(failed), "total_count": len(replay_items)}],
        next_action="allow_canary_gate" if status == "success" else "collect_more_replay_evidence",
        metadata={"synthetic_memory_written": False},
    )


def _tasks(plan: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(plan, list):
        return [item for item in plan if isinstance(item, dict)]
    if isinstance(plan, dict):
        value = plan.get("tasks")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _read_items(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        for key in ("replay_results", "items", "results"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []
