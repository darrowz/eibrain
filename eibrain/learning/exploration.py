from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExplorationTask:
    task_id: str
    task_type: str
    prompt: str
    reason: str
    success_criteria: list[str]
    expected_skill_ids: list[str] = field(default_factory=list)
    priority: int = 50
    source: str = "eibrain.active_exploration"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_exploration_tasks(
    *,
    experiences: list[dict[str, Any]] | None = None,
    registry_assets: list[dict[str, Any]] | None = None,
    max_tasks: int = 5,
) -> list[dict[str, Any]]:
    """Plan active exploration tasks without writing production memory."""

    experiences = experiences or []
    registry_assets = registry_assets or []
    meaningful_count = sum(1 for item in experiences if _is_meaningful_trace(item))
    active_skills = [item for item in registry_assets if str(item.get("status") or "") == "active"]
    candidate_skills = [item for item in registry_assets if str(item.get("status") or "") == "candidate"]

    tasks: list[ExplorationTask] = []
    if meaningful_count < 2:
        tasks.append(
            _task(
                task_type="brain.respond",
                prompt="Collect two real explicit-memory-request turns from user-approved interactions; do not create synthetic memories.",
                reason="meaningful_skill_trace_count_below_replay_minimum",
                success_criteria=[
                    "trace meta.write_policy_version == meaningful_event_v1",
                    "trace meta.trace_reason is present",
                    "outcome is planned/success and feedback is not negative",
                    "raw greeting or casual chat is not recorded",
                ],
                expected_skill_ids=["reply.default"],
                priority=95,
                metadata={"meaningful_count": meaningful_count, "target_minimum": 2},
            )
        )
    if not active_skills and candidate_skills:
        tasks.append(
            _task(
                task_type="skill.replay",
                prompt="Run replay validation for current candidate skills before any promotion.",
                reason="candidate_skills_waiting_for_replay",
                success_criteria=[
                    "replay_results.json exists",
                    "sample_count >= 2 for each promoted candidate",
                    "regression_count == 0 for promoted candidates",
                ],
                priority=85,
                metadata={"candidate_count": len(candidate_skills)},
            )
        )
    if not active_skills and not candidate_skills:
        tasks.append(
            _task(
                task_type="world_knowledge.seed",
                prompt="Identify one user-approved reusable preference, workflow rule, or visual state change to seed world knowledge.",
                reason="no_active_or_candidate_skills",
                success_criteria=[
                    "source event is user-approved or observed semantic visual change",
                    "knowledge includes applicability boundary",
                    "knowledge links back to trace or outcome evidence",
                ],
                priority=75,
                metadata={"active_count": 0, "candidate_count": 0},
            )
        )
    tasks.append(
        _task(
            task_type="vision.semantic_change",
            prompt="Watch for a new semantic visual scene or object-feature change; write only if dedupe and confidence gates pass.",
            reason="maintain_visual_world_knowledge_stream",
            success_criteria=[
                "duplicate frame is skipped",
                "new scene/object/state change gets trace_reason visual_new_scene or visual_state_change",
                "low confidence observations are skipped",
            ],
            expected_skill_ids=["orient.default"],
            priority=40,
        )
    )
    return [task.to_dict() for task in sorted(tasks, key=lambda item: (-item.priority, item.task_id))[: max(max_tasks, 0)]]


def _task(
    *,
    task_type: str,
    prompt: str,
    reason: str,
    success_criteria: list[str],
    expected_skill_ids: list[str] | None = None,
    priority: int,
    metadata: dict[str, Any] | None = None,
) -> ExplorationTask:
    digest = hashlib.sha256(f"{task_type}:{reason}:{prompt}".encode("utf-8")).hexdigest()[:12]
    return ExplorationTask(
        task_id=f"explore:{digest}",
        task_type=task_type,
        prompt=prompt,
        reason=reason,
        success_criteria=success_criteria,
        expected_skill_ids=list(expected_skill_ids or []),
        priority=priority,
        metadata=dict(metadata or {}),
    )


def _is_meaningful_trace(item: dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    payload_meta = content.get("meta") if isinstance(content.get("meta"), dict) else {}
    report_type = meta.get("report_type") or provenance.get("report_type") or item.get("report_type")
    return report_type == "skill_trace" and bool(
        meta.get("write_policy_version") == "meaningful_event_v1"
        or meta.get("trace_reason")
        or payload_meta.get("write_policy_version") == "meaningful_event_v1"
        or payload_meta.get("trace_reason")
    )


def dumps_plan(tasks: list[dict[str, Any]]) -> str:
    return json.dumps({"ok": True, "tasks": tasks, "task_count": len(tasks)}, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
