from __future__ import annotations

from types import SimpleNamespace

from eibrain.cognition.realtime.memory import MemoryOrchestrator


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
