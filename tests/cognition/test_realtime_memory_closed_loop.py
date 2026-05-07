from __future__ import annotations

from types import SimpleNamespace

from eibrain.cognition.realtime.memory import MemoryOrchestrator
from eibrain.memory.adapters.eimemory_rpc import FakeEIMemoryRPCAdapter
from eibrain.memory.contracts import MemoryResult


def _turn() -> SimpleNamespace:
    return SimpleNamespace(
        round_id="round-closed-loop-1",
        cancellation_token="tok-closed-loop-1",
        memory_candidates=[],
        memory_traces=[],
    )


def test_visual_world_writeback_is_inert_and_commits_with_world_observation_writer() -> None:
    class MemorySpy:
        last_writeback_status = {"status": "idle"}

        def __init__(self) -> None:
            self.world_observations: list[dict[str, object]] = []
            self.episodes: list[dict[str, object]] = []
            self.memory_traces: list[dict[str, object]] = []

        def remember_world_observation(self, **kwargs: object) -> None:
            self.world_observations.append(dict(kwargs))
            self.last_writeback_status = {
                "status": "ok",
                "source": "eibrain.visual_world",
                "memory_type": "world_observation",
                "record_id": "world-1",
            }

        def remember_episode(self, **kwargs: object) -> None:
            self.episodes.append(dict(kwargs))

        def record_memory_trace(
            self,
            payload: dict[str, object],
            *,
            session_id: str | None = None,
            actor_id: str | None = None,
        ) -> dict[str, object]:
            self.memory_traces.append(
                {
                    "payload": dict(payload),
                    "session_id": session_id,
                    "actor_id": actor_id,
                }
            )
            return {"ok": True, "result": {"record_id": "trace-1"}}

    turn = _turn()
    memory = MemorySpy()
    candidate = MemoryOrchestrator(memory_service=memory).build_visual_world_writeback(
        turn,
        scene={
            "summary": "person near desk",
            "objects": [{"label": "person", "confidence": 0.92}],
            "relations": [{"subject": "person", "relation": "near", "object": "desk"}],
        },
        evidence=[{"type": "frame", "id": "frame-123"}],
        links=[{"rel": "scene", "href": "scene-123"}],
        source_event_id="frame-123",
        trace_id="vision:frame-123",
        frame_ref="/tmp/frame-123.jpg",
        priority="realtime",
        reason="closed_loop_visual_grounding",
    )

    assert candidate["kind"] == "writeback_proposal"
    assert candidate["external_call"] is False
    assert candidate["requires_commit"] is True
    assert turn.memory_candidates == [candidate]
    assert candidate["summary"] == "person near desk"
    assert candidate["metadata"]["memory_type"] == "world_observation"
    assert candidate["metadata"]["source"] == "eibrain.visual_world"
    assert candidate["metadata"]["modality"] == "vision"
    assert candidate["metadata"]["organ"] == "eye"
    assert candidate["metadata"]["content"]["objects"] == [{"label": "person", "confidence": 0.92}]
    assert candidate["metadata"]["content"]["frame_ref"] == "/tmp/frame-123.jpg"
    assert candidate["metadata"]["meta"]["trace_id"] == "vision:frame-123"
    assert candidate["metadata"]["meta"]["source_event_id"] == "frame-123"
    assert candidate["metadata"]["tags"] == ["world_observation", "vision", "eye", "person"]

    trace = MemoryOrchestrator(memory_service=memory).commit_candidates(
        turn,
        session_id="vision:desk",
        actor_id="user-1",
    )

    assert len(memory.world_observations) == 1
    assert memory.episodes == []
    assert memory.world_observations[0]["session_id"] == "vision:desk"
    assert memory.world_observations[0]["summary"] == "person near desk"
    assert memory.world_observations[0]["content"]["scene"]["summary"] == "person near desk"
    assert memory.world_observations[0]["meta"]["source_event_id"] == "frame-123"
    assert memory.world_observations[0]["evidence"] == [{"type": "frame", "id": "frame-123"}]
    assert memory.world_observations[0]["links"] == [{"rel": "scene", "href": "scene-123"}]
    assert trace["writeback"]["count"] == 1
    assert trace["writeback"]["items"][0]["source"] == "eibrain.visual_world"
    assert trace["writeback"]["items"][0]["memory_type"] == "world_observation"
    assert trace["writeback"]["items"][0]["record_id"] == "world-1"
    assert trace["write"]["proposed"][0]["summary"] == "person near desk"
    assert trace["write"]["committed"][0]["record_id"] == "world-1"
    assert trace["memory_trace_summary"]["write_committed"] == 1
    assert trace["trace_record"]["status"] == "ok"
    assert trace["trace_record"]["record_id"] == "trace-1"
    assert memory.memory_traces[0]["session_id"] == "vision:desk"
    assert memory.memory_traces[0]["actor_id"] == "user-1"
    assert memory.memory_traces[0]["payload"]["writeback"]["items"][0]["record_id"] == "world-1"


def test_memory_trace_records_empty_and_error_diagnostics() -> None:
    class EmptyTraceMemory:
        last_writeback_status = {"status": "idle"}

        def remember_episode(self, **kwargs: object) -> None:
            self.last_writeback_status = {"status": "ok", "source": kwargs.get("source"), "memory_type": kwargs.get("memory_type")}

        def record_memory_trace(self, payload: dict[str, object], **kwargs: object) -> dict[str, object]:
            return {}

    class RaisingTraceMemory(EmptyTraceMemory):
        def record_memory_trace(self, payload: dict[str, object], **kwargs: object) -> dict[str, object]:
            raise RuntimeError("trace down")

    for memory, expected_status in ((EmptyTraceMemory(), "skipped"), (RaisingTraceMemory(), "error")):
        turn = _turn()
        MemoryOrchestrator(memory_service=memory).build_writeback_proposal(
            turn,
            query="trace branch",
            reason="trace_branch_test",
        )

        trace = MemoryOrchestrator(memory_service=memory).commit_candidates(turn, session_id="s1")

        assert trace["trace_record"]["status"] == expected_status


def test_visual_world_writeback_falls_back_to_episode_writer_when_world_writer_missing() -> None:
    class EpisodeOnlyMemorySpy:
        last_writeback_status = {"status": "idle"}

        def __init__(self) -> None:
            self.episodes: list[dict[str, object]] = []

        def remember_episode(self, **kwargs: object) -> None:
            self.episodes.append(dict(kwargs))
            self.last_writeback_status = {
                "status": "ok",
                "source": kwargs.get("source"),
                "memory_type": kwargs.get("memory_type"),
                "record_id": "episode-world-1",
            }

    turn = _turn()
    memory = EpisodeOnlyMemorySpy()
    MemoryOrchestrator(memory_service=memory).build_visual_world_writeback(
        turn,
        observation={"summary": "cup on table", "objects": [{"label": "cup"}]},
        source_event_id="frame-456",
    )

    trace = MemoryOrchestrator(memory_service=memory).commit_candidates(turn, session_id="vision:desk")

    assert memory.episodes[0]["memory_type"] == "world_observation"
    assert memory.episodes[0]["source"] == "eibrain.visual_world"
    assert memory.episodes[0]["modality"] == "vision"
    assert memory.episodes[0]["organ"] == "eye"
    assert trace["writeback"]["items"][0]["record_id"] == "episode-world-1"


def test_action_outcome_writeback_builds_closed_loop_candidate_metadata() -> None:
    turn = _turn()
    candidate = MemoryOrchestrator().build_action_outcome_writeback(
        turn,
        action="MoveHeadAction",
        organ="neck",
        success=False,
        status="failed",
        outcome={"error": "oscillation", "latency_ms": 280},
        evidence=[{"type": "head_feedback", "message": "tracking overshot target"}],
        links=[{"rel": "frame", "href": "frame-789"}],
        source_event_id="head-outcome-1",
        trace_id="trace-neck-1",
        suggested_adjustment="increase yaw deadband before moving",
    )

    assert candidate["kind"] == "writeback_proposal"
    assert candidate["external_call"] is False
    assert candidate["summary"] == "MoveHeadAction on neck failed"
    assert candidate["metadata"]["memory_type"] == "action_outcome"
    assert candidate["metadata"]["source"] == "eibrain.outcome_feedback"
    assert candidate["metadata"]["modality"] == "multimodal_action"
    assert candidate["metadata"]["organ"] == "neck"
    assert candidate["metadata"]["outcome"]["success"] is False
    assert candidate["metadata"]["outcome"]["status"] == "failed"
    assert candidate["metadata"]["content"]["action"] == "MoveHeadAction"
    assert candidate["metadata"]["content"]["suggested_adjustment"] == "increase yaw deadband before moving"
    assert candidate["metadata"]["meta"]["trace_id"] == "trace-neck-1"
    assert candidate["metadata"]["meta"]["source_event_id"] == "head-outcome-1"
    assert candidate["metadata"]["evidence"] == [{"type": "head_feedback", "message": "tracking overshot target"}]
    assert candidate["metadata"]["links"] == [{"rel": "frame", "href": "frame-789"}]
    assert "action_outcome" in candidate["metadata"]["tags"]


def test_memory_orchestrator_records_recalled_items_used_in_reply() -> None:
    turn = _turn()
    memory = FakeEIMemoryRPCAdapter(
        recall_result=MemoryResult(
            summary="Prefer concise spoken replies.",
            relevant_memories=["Reply Style: Prefer concise spoken replies."],
            recall_diagnostics={
                "selected_count": 1,
                "selected_records": [
                    {
                        "record_id": "mem_reply_1",
                        "title": "Reply Style",
                        "source": "eibrain.preference",
                        "memory_type": "preference",
                    }
                ],
            },
        )
    )
    orchestrator = MemoryOrchestrator(memory_service=memory)
    orchestrator.build_recall_request(
        turn,
        query="用户回复风格偏好",
        reason="prefetch_context_for_reply",
        metadata={"task_type": "brain.respond"},
    )

    orchestrator.commit_candidates(turn, session_id="voice-session", actor_id="user-1")
    trace = orchestrator.record_reply_memory_usage(
        turn,
        reply_text="我记得你喜欢更简短一点的回复。",
        used_items=[{"record_id": "mem_reply_1"}],
        session_id="voice-session",
        actor_id="user-1",
    )

    recalled = [item for item in turn.memory_candidates if item.get("kind") == "recall"]
    assert recalled[0]["record_id"] == "mem_reply_1"
    assert recalled[0]["used_in_reply"] is True
    assert trace["reply"]["reply_text"] == "我记得你喜欢更简短一点的回复。"
    assert trace["reply"]["used_recall_items"] == [
        {
            "record_id": "mem_reply_1",
            "text": "Reply Style: Prefer concise spoken replies.",
            "title": "Reply Style",
            "memory_type": "preference",
            "memory_source": "eibrain.preference",
        }
    ]
    assert trace["memory_trace_summary"]["reply_used"] == 1
    assert trace["memory_trace_summary"]["used_memory_ids"] == ["mem_reply_1"]
    assert memory.memory_traces[-1]["payload"]["reply"]["used_recall_items"][0]["record_id"] == "mem_reply_1"


def test_memory_trace_records_closed_loop_lifecycle_policy_conflict_and_reply_filters() -> None:
    turn = _turn()
    memory = FakeEIMemoryRPCAdapter(
        recall_result=MemoryResult(
            summary="Preference and blocked persona memory.",
            relevant_memories=[
                "Reply Style: Prefer concise spoken replies.",
                "Blocked Persona: unrelated persona note.",
            ],
            recall_diagnostics={
                "selected_count": 2,
                "selected_records": [
                    {
                        "record_id": "mem_allowed",
                        "title": "Reply Style",
                        "source": "eibrain.preference",
                        "memory_type": "preference",
                        "policy_decision": {"decision": "allow", "reason": "reply_context"},
                    },
                    {
                        "record_id": "mem_filtered",
                        "title": "Blocked Persona",
                        "source": "eibrain.persona",
                        "memory_type": "persona",
                        "policy_decision": {"decision": "filter", "reason": "persona_policy"},
                    },
                ],
                "recall_filters": {
                    "filtered_records": [
                        {
                            "record_id": "mem_filtered",
                            "reason": "persona_policy",
                            "source": "eibrain.persona",
                        }
                    ]
                },
            },
        )
    )
    orchestrator = MemoryOrchestrator(memory_service=memory)
    orchestrator.build_recall_request(
        turn,
        query="用户回复风格偏好",
        reason="prefetch_context_for_reply",
        metadata={"task_type": "brain.respond"},
    )
    orchestrator.build_writeback_proposal(
        turn,
        query="用户要求回答短一点",
        reason="explicit_memory_request",
        summary="用户偏好更短的回复。",
        metadata={
            "memory_type": "preference",
            "source": "eibrain.preference",
            "policy_decision": {"decision": "commit", "reason": "explicit_user_request"},
            "conflict_resolution": {
                "status": "resolved",
                "strategy": "merge",
                "conflict_record_ids": ["pref_old"],
            },
            "meta": {
                "idempotency_key": "memtrace:s1:evt-1",
                "source_event_id": "evt-1",
                "conflict": {"strategy": "merge", "conflict_record_ids": ["pref_old"]},
            },
        },
    )

    trace = orchestrator.commit_candidates(turn, session_id="s1", actor_id="user-1")
    trace = orchestrator.record_reply_memory_usage(
        turn,
        reply_text="我会尽量短一点。",
        used_items=[{"record_id": "mem_allowed"}],
        filtered_items=[{"record_id": "mem_filtered", "reason": "persona_policy"}],
        session_id="s1",
        actor_id="user-1",
    )

    assert [event["stage"] for event in trace["lifecycle"]] == [
        "prefetch",
        "candidates",
        "policy_decision",
        "prefetch_result",
        "policy_decision",
        "conflict_resolution",
        "write_committed",
        "recall_used",
    ]
    assert trace["policy_decision"]["write"][0]["decision"] == "commit"
    filtered_policy_rows = [
        item
        for item in trace["policy_decision"]["recall"]
        if item["record_id"] == "mem_filtered" and item["decision"] == "filter"
    ]
    assert len(filtered_policy_rows) == 1
    assert trace["conflict_resolution"]["write"][0]["strategy"] == "merge"
    assert trace["reply_context"]["used"][0]["record_id"] == "mem_allowed"
    assert trace["reply_context"]["filtered"][0]["record_id"] == "mem_filtered"
    assert trace["reply_context"]["filtered"][0]["reason"] == "persona_policy"


def test_reply_memory_usage_does_not_mark_all_recall_items_used_without_explicit_ids() -> None:
    turn = _turn()
    turn.memory_candidates = [
        {
            "kind": "recall",
            "record_id": "mem_unknown_1",
            "text": "A potentially useful memory",
            "memory_type": "preference",
        },
        {
            "kind": "recall",
            "record_id": "mem_unknown_2",
            "text": "Another potentially useful memory",
            "memory_type": "fact",
        },
    ]

    trace = MemoryOrchestrator().record_reply_memory_usage(turn, reply_text="收到。")

    assert trace["reply"]["used_recall_items"] == []
    assert trace["reply"]["used_count"] == 0
    assert [item["reply_context_status"] for item in turn.memory_candidates] == ["available", "available"]


def test_reply_memory_usage_auto_records_policy_filtered_recall_items() -> None:
    turn = _turn()
    turn.memory_candidates = [
        {
            "kind": "recall",
            "record_id": "mem_allowed",
            "text": "A usable preference memory",
            "memory_type": "preference",
            "policy_decision": {"decision": "allow", "reason": "reply_context"},
        },
        {
            "kind": "recall",
            "record_id": "mem_filtered",
            "text": "A persona-drifting memory",
            "memory_type": "persona",
            "policy_decision": {"decision": "filter", "reason": "persona_policy"},
        },
    ]

    trace = MemoryOrchestrator().record_reply_memory_usage(turn, reply_text="收到。")

    assert trace["reply_context"]["used"] == []
    assert trace["reply_context"]["filtered"] == [
        {
            "record_id": "mem_filtered",
            "text": "A persona-drifting memory",
            "title": "",
            "memory_type": "persona",
            "memory_source": "",
            "reason": "persona_policy",
        }
    ]
    assert turn.memory_candidates[0]["reply_context_status"] == "available"
    assert turn.memory_candidates[1]["reply_context_status"] == "filtered"


def test_memory_policy_filters_visual_frame_writeback_before_rpc_commit() -> None:
    turn = _turn()
    memory = FakeEIMemoryRPCAdapter()
    orchestrator = MemoryOrchestrator(memory_service=memory)
    orchestrator.build_writeback_proposal(
        turn,
        query="raw frame",
        reason="visual_noise",
        summary="Transient low confidence visual frame.",
        metadata={
            "memory_type": "working_event",
            "event_type": "visual_frame",
            "modality": "vision",
            "source": "eibrain.visual_frame",
            "confidence": 0.2,
        },
    )

    trace = orchestrator.commit_candidates(turn, session_id="s1", actor_id="user-1")

    assert trace["writeback"]["items"][0]["status"] == "skipped"
    assert trace["writeback"]["items"][0]["reason"] == "memory_policy_rejected"
    assert trace["policy_decision"]["write"][0]["decision"] == "reject"
    assert memory.memory_traces[-1]["payload"]["writeback"]["items"][0]["status"] == "skipped"


def test_memory_policy_deferred_writeback_stays_retryable() -> None:
    turn = _turn()
    memory = FakeEIMemoryRPCAdapter()
    orchestrator = MemoryOrchestrator(memory_service=memory)
    candidate = orchestrator.build_writeback_proposal(
        turn,
        query="记住我以后回复长一点",
        reason="explicit_memory_request",
        summary="用户偏好更长的回复。",
        metadata={
            "memory_type": "preference",
            "source": "eibrain.preference",
            "subject": "user-1",
            "key": "response.length",
            "value": "long",
            "existing_memories": [
                {
                    "id": "pref-short",
                    "subject": "user-1",
                    "key": "response.length",
                    "value": "short",
                    "summary": "用户偏好短回复。",
                }
            ],
        },
    )

    trace = orchestrator.commit_candidates(turn, session_id="s1", actor_id="user-1")

    assert trace["writeback"]["items"][0]["status"] == "skipped"
    assert trace["writeback"]["items"][0]["reason"] == "memory_policy_deferred"
    assert trace["policy_decision"]["write"][0]["decision"] == "defer"
    assert candidate["committed"] is False
    assert candidate["commit_status"] == "deferred"


def test_memory_trace_preserves_recall_and_write_source_event_ids() -> None:
    turn = _turn()
    memory = FakeEIMemoryRPCAdapter(
        recall_result=MemoryResult(
            summary="Recall result",
            relevant_memories=["Known preference."],
            recall_diagnostics={
                "selected_count": 1,
                "selected_records": [{"record_id": "mem-trace-1", "memory_type": "preference"}],
            },
        )
    )
    orchestrator = MemoryOrchestrator(memory_service=memory)
    orchestrator.build_recall_request(
        turn,
        query="trace recall",
        reason="prefetch_context_for_reply",
        metadata={"trace_id": "trace-recall-1", "source_event_id": "evt-recall-1"},
    )
    orchestrator.build_writeback_proposal(
        turn,
        query="trace write",
        reason="explicit_memory_request",
        summary="User likes concise answers.",
        metadata={
            "memory_type": "preference",
            "source": "eibrain.preference",
            "key": "response.length",
            "value": "short",
            "meta": {"trace_id": "trace-write-1", "source_event_id": "evt-write-1"},
        },
    )

    trace = orchestrator.commit_candidates(turn, session_id="s1", actor_id="user-1")

    assert trace["prefetch"]["requested"][0]["trace_id"] == "trace-recall-1"
    assert trace["prefetch"]["requested"][0]["source_event_id"] == "evt-recall-1"
    assert trace["recall"]["items"][0]["trace_id"] == "trace-recall-1"
    assert trace["recall"]["items"][0]["source_event_id"] == "evt-recall-1"
    assert trace["write"]["proposed"][0]["trace_id"] == "trace-write-1"
    assert trace["write"]["proposed"][0]["source_event_id"] == "evt-write-1"
    assert trace["writeback"]["items"][0]["trace_id"] == "trace-write-1"
    assert trace["writeback"]["items"][0]["source_event_id"] == "evt-write-1"
