from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eibrain.cognition.realtime.events import asr_partial, prosody, vision
from eibrain.cognition.realtime.fast import FastThinkEngine
from eibrain.cognition.realtime.memory import MemoryOrchestrator


def _duck_turn() -> SimpleNamespace:
    return SimpleNamespace(
        round_id="round-fast-1",
        cancellation_token="tok-fast-1",
        asr_partial=[],
        emotion_state={},
        fast_hypotheses=[],
        intent_hypotheses=[],
        memory_candidates=[],
        memory_traces=[],
    )


def test_fast_think_uses_partial_vision_and_emotion_for_non_stable_hints() -> None:
    turn = _duck_turn()
    result = FastThinkEngine().process(
        turn,
        asr_partial_text="能不能打开灯，我有点看不清",
        vision_hint={"objects": ["lamp"], "attention": "present", "scene": "dim_room"},
        emotion_hint={"label": "uncertain", "arousal": 0.72},
        deadline_ms=900,
    )
    payload = result.to_dict()

    assert payload["round_id"] == "round-fast-1"
    assert payload["cancellation_token"] == "tok-fast-1"
    assert payload["stable"] is False
    assert payload["deadline_ms"] <= 500
    assert "decision" not in payload
    assert payload["micro_feedback"]["risk"] == "low"
    assert payload["micro_feedback"]["stable"] is False
    assert payload["micro_feedback"]["text"] in FastThinkEngine.SAFE_MICRO_FEEDBACK
    assert {item["source"] for item in payload["hypotheses"]} >= {
        "asr_partial",
        "vision_hint",
        "emotion_hint",
    }
    assert all(item["stable"] is False for item in payload["hypotheses"])
    assert any(item["intent"] == "possible_action_request" for item in payload["intent_hints"])
    assert all(item["stable"] is False for item in payload["intent_hints"])
    assert turn.fast_hypotheses == payload["hypotheses"]
    assert turn.intent_hypotheses == payload["intent_hints"]
    assert json.loads(json.dumps(payload, ensure_ascii=False))["round_id"] == "round-fast-1"


def test_fast_think_accepts_task1_observation_dicts_without_committing_facts() -> None:
    turn = _duck_turn()
    observations = [
        asr_partial(round_id=turn.round_id, cancellation_token=turn.cancellation_token, text="明天会下雨吗"),
        vision(round_id=turn.round_id, cancellation_token=turn.cancellation_token, hints={"face_expression": "curious"}),
        prosody(round_id=turn.round_id, cancellation_token=turn.cancellation_token, hints={"stress": 0.66}),
    ]

    payload = FastThinkEngine().process(turn, observations=observations).to_dict()
    rendered = json.dumps(payload, ensure_ascii=False)

    assert payload["stable"] is False
    assert all(item.get("stable") is False for item in payload["hypotheses"])
    assert all(item.get("stable") is False for item in payload["intent_hints"])
    assert payload["micro_feedback"]["kind"] == "companionship"
    assert payload["micro_feedback"]["text"] in FastThinkEngine.SAFE_MICRO_FEEDBACK
    for forbidden in ("我确定", "事实是", "最终决定", "一定会", "已经为你", "我会打开"):
        assert forbidden not in rendered


def test_fast_think_rejects_stable_or_decision_shaped_hypotheses() -> None:
    turn = _duck_turn()
    engine = FastThinkEngine()

    with pytest.raises(ValueError, match="non-stable"):
        engine.sanitize_hypothesis({"intent": "unsafe", "stable": True})

    with pytest.raises(ValueError, match="final decisions"):
        engine.sanitize_hypothesis({"intent": "unsafe", "decision": "turn_on_light"})

    safe = engine.sanitize_hypothesis({"intent": "listen", "confidence": 0.41})
    assert safe["stable"] is False
    assert FastThinkEngine().process(turn, asr_partial_text="嗯").stable is False


def test_memory_orchestrator_builds_recall_request_without_external_calls() -> None:
    class ExternalServiceSpy:
        called = False

        def retrieve_context(self, *_args: object, **_kwargs: object) -> None:
            self.called = True
            raise AssertionError("MemoryOrchestrator must not call external services")

    turn = _duck_turn()
    spy = ExternalServiceSpy()
    request = MemoryOrchestrator(memory_service=spy).build_recall_request(
        turn,
        query="用户提到有点看不清，可能需要灯光偏好",
        channels=["voice", "vision"],
        priority="realtime",
        reason="prefetch_context_for_fast_lane",
    )

    assert spy.called is False
    assert request["kind"] == "recall_request"
    assert request["round_id"] == "round-fast-1"
    assert request["cancellation_token"] == "tok-fast-1"
    assert request["query"] == "用户提到有点看不清，可能需要灯光偏好"
    assert request["channels"] == ["voice", "vision"]
    assert request["priority"] == "realtime"
    assert request["reason"] == "prefetch_context_for_fast_lane"
    assert request["external_call"] is False
    assert turn.memory_candidates == [request]
    assert json.loads(json.dumps(request, ensure_ascii=False))["kind"] == "recall_request"


def test_memory_orchestrator_builds_writeback_proposal_as_inert_candidate() -> None:
    turn = _duck_turn()
    proposal = MemoryOrchestrator().build_writeback_proposal(
        turn,
        query="用户请求记住以后回答短一点",
        channels=("voice",),
        priority="normal",
        reason="user_explicit_memory_candidate",
        summary="用户偏好更短的语音回复。",
        metadata={"source_event_id": "evt-user-1"},
    )

    assert proposal["kind"] == "writeback_proposal"
    assert proposal["round_id"] == "round-fast-1"
    assert proposal["query"] == "用户请求记住以后回答短一点"
    assert proposal["channels"] == ["voice"]
    assert proposal["priority"] == "normal"
    assert proposal["reason"] == "user_explicit_memory_candidate"
    assert proposal["summary"] == "用户偏好更短的语音回复。"
    assert proposal["metadata"]["source_event_id"] == "evt-user-1"
    assert proposal["external_call"] is False
    assert proposal["requires_commit"] is True
    assert turn.memory_candidates == [proposal]


def test_memory_orchestrator_commits_recall_and_writeback_candidates_with_trace() -> None:
    from eibrain.memory.contracts import MemoryResult

    class MemorySpy:
        last_writeback_status = {"status": "idle"}

        def __init__(self) -> None:
            self.queries = []
            self.episodes = []

        def retrieve_context(self, query):
            self.queries.append(query)
            return MemoryResult(
                summary="Prefer concise spoken replies.",
                relevant_memories=["Reply Style: Prefer concise spoken replies."],
                recall_diagnostics={
                    "selected_count": 1,
                    "selected_records": [{"record_id": "mem_1", "title": "Reply Style"}],
                    "source_composition": {"eibrain.preference": 1},
                },
            )

        def remember_episode(self, **kwargs):
            self.episodes.append(dict(kwargs))
            self.last_writeback_status = {
                "status": "ok",
                "source": kwargs.get("source"),
                "memory_type": kwargs.get("memory_type"),
            }

    turn = _duck_turn()
    memory = MemorySpy()
    orchestrator = MemoryOrchestrator(memory_service=memory)
    orchestrator.build_recall_request(
        turn,
        query="用户提到简短回复偏好",
        channels=["voice"],
        priority="realtime",
        reason="prefetch_context_for_fast_lane",
        metadata={"task_type": "brain.respond"},
    )
    orchestrator.build_writeback_proposal(
        turn,
        query="用户明确要求记住偏好",
        channels=["voice"],
        priority="normal",
        reason="user_explicit_memory_candidate",
        summary="用户偏好更短的语音回复。",
        metadata={
            "memory_type": "semantic_candidate",
            "source": "eibrain.semantic_candidate",
            "modality": "audio_text",
            "organ": "ear",
            "tags": ["semantic_candidate", "voice"],
            "outcome": {"success": True, "status": "planned"},
            "content": {"event_type": "dialogue"},
            "meta": {"trace_id": "trace-memory-1", "identity_memory": False},
        },
    )

    trace = orchestrator.commit_candidates(turn, session_id="s1", actor_id="user-1")

    assert trace["schema"] == "eibrain.memory.closed_loop_trace.v1"
    assert trace["round_id"] == "round-fast-1"
    assert trace["external_call"] is True
    assert trace["recall"]["count"] == 1
    assert trace["recall"]["items"][0]["summary"] == "Prefer concise spoken replies."
    assert trace["recall"]["items"][0]["selected_count"] == 1
    assert trace["writeback"]["count"] == 1
    assert trace["writeback"]["items"][0]["status"] == "ok"
    assert trace["errors"] == []
    assert memory.queries[0].query == "用户提到简短回复偏好"
    assert memory.queries[0].session_id == "s1"
    assert memory.queries[0].actor_id == "user-1"
    assert memory.queries[0].task_context["task_type"] == "brain.respond"
    assert memory.queries[0].task_context["reason"] == "prefetch_context_for_fast_lane"
    assert memory.episodes[0]["summary"] == "用户偏好更短的语音回复。"
    assert memory.episodes[0]["memory_type"] == "semantic_candidate"
    assert memory.episodes[0]["source"] == "eibrain.semantic_candidate"
    assert memory.episodes[0]["modality"] == "audio_text"
    assert memory.episodes[0]["organ"] == "ear"
    assert memory.episodes[0]["tags"] == ["semantic_candidate", "voice"]
    assert turn.memory_traces == [trace]


def test_memory_orchestrator_skips_non_committable_writeback_candidates() -> None:
    class MemorySpy:
        def __init__(self) -> None:
            self.episodes = []

        def remember_episode(self, **kwargs):
            self.episodes.append(dict(kwargs))

    turn = _duck_turn()
    memory = MemorySpy()
    proposal = MemoryOrchestrator(memory_service=memory).build_writeback_proposal(
        turn,
        query="短期工作记忆，不应持久化",
        reason="working_memory_only",
        summary="这只是本轮短期上下文。",
        requires_commit=False,
    )

    trace = MemoryOrchestrator(memory_service=memory).commit_candidates(turn, session_id="s1")

    assert proposal["requires_commit"] is False
    assert memory.episodes == []
    assert trace["writeback"]["items"] == [
        {
            "candidate_index": 0,
            "kind": "writeback_proposal",
            "status": "skipped",
            "reason": "requires_commit_false",
            "summary": "这只是本轮短期上下文。",
        }
    ]
