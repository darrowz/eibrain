from __future__ import annotations

import json

import pytest

from eibrain.cognition.realtime import (
    ProactiveActivityManager,
    RealtimeCognitiveScheduler,
    SlowReasoner,
)
from eibrain.cognition.realtime.memory import MemoryOrchestrator
from eibrain.memory.contracts import MemoryResult


def _clock():
    values = iter(
        [
            3000.0,
            3000.05,
            3000.1,
            3000.15,
            3000.2,
            3000.25,
            3000.3,
            3000.35,
            3000.4,
            3000.45,
            3000.5,
        ]
    )

    def clock() -> float:
        return next(values)

    return clock


def test_slow_reasoner_builds_stable_decision_from_context_without_llm() -> None:
    reasoner = SlowReasoner()

    decision = reasoner.decide(
        round_id="round-42",
        cancellation_token="token-42",
        final_text="明天提醒我给妈妈打电话",
        fast_hypotheses=[{"intent": "create_reminder", "confidence": 0.82}],
        memory_candidates=[
            {"id": "mem-1", "text": "妈妈通常晚上八点后方便接电话", "score": 0.91}
        ],
        persona_context={"name": "Eiri", "style": "gentle", "language": "zh-CN"},
        emotion_context={"mood": "tired", "arousal": "low"},
    )

    assert decision["round_id"] == "round-42"
    assert decision["cancellation_token"] == "token-42"
    assert decision["stable"] is True
    assert decision["decision"] == "create_reminder"
    assert decision["final_text"] == "明天提醒我给妈妈打电话"
    assert decision["memory_refs"] == [
        {"id": "mem-1", "text": "妈妈通常晚上八点后方便接电话", "score": 0.91}
    ]
    assert decision["persona"]["style"] == "gentle"
    assert decision["emotion"]["mood"] == "tired"
    assert decision["speech_segments"][0]["stable"] is True
    assert decision["speech_segments"][0]["source"] == "slow_reasoner"
    assert json.loads(json.dumps(decision, ensure_ascii=False))["stable"] is True


def test_slow_reasoner_final_question_overrides_partial_action_hint() -> None:
    reasoner = SlowReasoner()

    decision = reasoner.decide(
        round_id="round-question",
        cancellation_token="token-question",
        final_text="为什么灯打不开？",
        fast_hypotheses=[{"intent": "possible_action_request", "confidence": 0.88}],
    )

    assert decision["decision"] == "answer_question"
    assert decision["action_plan"] == []
    assert "会按你的指令" not in decision["speech_text"]


def test_scheduler_decide_cannot_override_committed_final_asr_with_external_text() -> None:
    scheduler = RealtimeCognitiveScheduler(clock=_clock())

    scheduler.observe_partial("能不能打开灯")
    scheduler.observe_final("为什么灯打不开？")
    result = scheduler.decide(final_text="请帮我打开灯")

    assert result["decision"]["final_text"] == "为什么灯打不开？"
    assert result["decision"]["decision"] == "answer_question"
    assert result["action_plan"] == []


def test_proactive_activity_manager_chooses_low_disturbance_channel() -> None:
    manager = ProactiveActivityManager()

    silent = manager.propose(
        idle_seconds=10,
        emotion_context={"mood": "neutral"},
        memory_candidates=[],
        execution_result=None,
    )
    visual = manager.propose(
        idle_seconds=90,
        emotion_context={"mood": "neutral"},
        memory_candidates=[{"id": "mem-2", "text": "稍后可以整理桌面", "importance": 0.4}],
        execution_result={"ok": False, "reason": "device_busy"},
    )
    spoken = manager.propose(
        idle_seconds=180,
        emotion_context={"mood": "sad"},
        memory_candidates=[{"id": "mem-3", "text": "喝水提醒", "importance": 0.9}],
        execution_result=None,
    )

    assert silent["channel"] == "silent"
    assert silent["should_emit"] is False
    assert visual["channel"] == "visual_only"
    assert visual["disturbance"] == "low"
    assert visual["should_emit"] is True
    assert spoken["channel"] == "speak"
    assert spoken["disturbance"] == "low"
    assert spoken["requires_user_attention"] is False


def test_scheduler_facade_observes_prefetches_decides_and_snapshots() -> None:
    scheduler = RealtimeCognitiveScheduler(clock=_clock())

    partial = scheduler.observe_partial(
        "帮我记一下",
        persona_context={"name": "Eiri", "style": "gentle", "language": "zh-CN"},
        emotion_context={"mood": "focused"},
    )
    scheduler.observe_final(
        "帮我记一下明天给妈妈打电话",
        memory_candidates=[{"id": "mem-4", "text": "妈妈晚上八点后方便接电话", "score": 0.9}],
    )
    result = scheduler.decide()
    snapshot = scheduler.snapshot()

    assert partial["fast"]["stable"] is False
    assert partial["memory_prefetch"][0]["source"] == "scheduler_prefetch"
    assert result["can_speak"] is True
    assert result["decision"]["stable"] is True
    assert result["decision"]["decision"] in {"create_reminder", "remember"}
    assert result["speech_segments"][0]["stable"] is True
    assert snapshot["active"] is True
    assert snapshot["current"]["stable_decisions"] == [result["decision"]]
    assert snapshot["current"]["memory_candidates"][0]["id"] == "mem-4"


def test_scheduler_returns_arbiter_checked_structured_action_segments() -> None:
    scheduler = RealtimeCognitiveScheduler(clock=_clock())

    scheduler.observe_partial("请打开灯")
    scheduler.observe_final("请打开灯")
    result = scheduler.decide()

    assert result["can_speak"] is True
    assert result["decision"]["action_segments"] == result["action_plan"]
    assert result["action_plan"]
    assert {
        "capabilityId",
        "startOffsetMs",
        "durationMs",
        "style",
    }.issubset(result["action_plan"][0])


def test_scheduler_fast_lane_can_refresh_after_slow_decision_without_replacing_decision() -> None:
    scheduler = RealtimeCognitiveScheduler(clock=_clock())

    scheduler.observe_partial("帮我记一下")
    scheduler.observe_final("帮我记一下明天给妈妈打电话")
    result = scheduler.decide()
    refreshed = scheduler.observe_partial("我还想补一句")
    snapshot = scheduler.snapshot()

    assert refreshed["fast"]["microfeedback"]
    assert snapshot["current"]["stable_decisions"] == [result["decision"]]
    assert snapshot["scheduler"]["fast_hypothesis_count"] >= 2


def test_scheduler_interrupt_blocks_old_token_from_committing_stable_decision() -> None:
    scheduler = RealtimeCognitiveScheduler(clock=_clock())
    partial = scheduler.observe_partial("请打开客厅灯")
    scheduler.observe_final("请打开客厅灯")

    old_round_id = partial["round_id"]
    old_token = partial["cancellation_token"]
    interrupt = scheduler.interrupt(reason="barge_in")

    with pytest.raises(RuntimeError, match="not current|cancelled"):
        scheduler.decide(round_id=old_round_id, cancellation_token=old_token)

    snapshot = scheduler.snapshot()
    assert interrupt["mark_interrupted"]["round_id"] == old_round_id
    assert interrupt["start_new_round"]["round_id"] != old_round_id
    assert snapshot["history"][old_round_id]["cancellation"]["cancelled"] is True
    assert snapshot["history"][old_round_id]["stable_decisions"] == []


def test_scheduler_decide_returns_operator_summary_and_trace() -> None:
    scheduler = RealtimeCognitiveScheduler(clock=_clock())

    scheduler.observe_partial("帮我记一下")
    scheduler.observe_final("帮我记一下明天给妈妈打电话")
    result = scheduler.decide()

    assert result["summary"] == {
        "round_id": result["round_id"],
        "cancellation_token": result["cancellation_token"],
        "lane": "speaking",
        "decision": result["decision"]["decision"],
        "can_speak": True,
        "speech_count": len(result["speech_segments"]),
        "action_count": len(result["action_plan"]),
    }
    assert result["trace"] == {
        "round_id": result["round_id"],
        "cancellation_token": result["cancellation_token"],
        "fast_hypothesis_count": 1,
        "stable_decision_count": 1,
        "speaking_state": "approved",
        "source": "realtime_cognitive_scheduler",
    }


def test_scheduler_interrupt_cancels_old_action_plan_and_returns_trace() -> None:
    scheduler = RealtimeCognitiveScheduler(clock=_clock())

    scheduler.observe_partial("请打开灯")
    scheduler.observe_final("请打开灯")
    decided = scheduler.decide()
    interrupt = scheduler.interrupt(reason="barge_in")
    old_turn = scheduler.snapshot()["history"][decided["round_id"]]

    assert old_turn["action_plan"]
    assert old_turn["action_plan"][0]["status"] == "cancelled"
    assert interrupt["summary"] == {
        "old_round_id": decided["round_id"],
        "new_round_id": interrupt["start_new_round"]["round_id"],
        "reason": "barge_in",
        "cancelled": True,
    }
    assert interrupt["trace"] == {
        "round_id": interrupt["start_new_round"]["round_id"],
        "cancellation_token": interrupt["start_new_round"]["cancellation_token"],
        "interrupted_round_id": decided["round_id"],
        "source": "realtime_cognitive_scheduler",
    }


def test_scheduler_explicitly_commits_memory_candidates_and_exposes_trace() -> None:
    class MemorySpy:
        last_writeback_status = {"status": "idle"}

        def __init__(self) -> None:
            self.queries = []
            self.episodes = []

        def retrieve_context(self, query):
            self.queries.append(query)
            return MemoryResult(
                summary="User prefers concise replies.",
                relevant_memories=["Preference: concise replies"],
                recall_diagnostics={"selected_count": 1},
            )

        def remember_episode(self, **kwargs):
            self.episodes.append(dict(kwargs))
            self.last_writeback_status = {"status": "ok", "source": kwargs.get("source")}

    memory = MemorySpy()
    scheduler = RealtimeCognitiveScheduler(
        clock=_clock(),
        memory_orchestrator=MemoryOrchestrator(memory_service=memory),
    )
    scheduler.observe_partial("帮我记一下")
    turn = scheduler.current_turn()
    assert turn is not None
    scheduler.memory_orchestrator.build_recall_request(
        turn,
        query="用户偏好简短回复",
        reason="prefetch_context_for_fast_lane",
    )
    scheduler.memory_orchestrator.build_writeback_proposal(
        turn,
        query="用户要求记住简短回复偏好",
        reason="user_explicit_memory_candidate",
        summary="用户偏好简短回复。",
        metadata={"memory_type": "semantic_candidate", "source": "eibrain.semantic_candidate"},
    )

    trace = scheduler.commit_memory_candidates(session_id="s1", actor_id="user-1")
    snapshot = scheduler.snapshot()

    assert trace["recall"]["count"] == 1
    assert trace["writeback"]["count"] == 1
    assert trace["errors"] == []
    assert memory.queries[0].query == "用户偏好简短回复"
    assert memory.episodes[0]["summary"] == "用户偏好简短回复。"
    assert snapshot["current"]["memory_traces"] == [trace]
    assert snapshot["scheduler"]["memory_trace_count"] == 1


def test_scheduler_marks_each_memory_candidate_commit_status_independently() -> None:
    class PartialMemory:
        def __init__(self) -> None:
            self.queries = []

        def retrieve_context(self, query):
            self.queries.append(query)
            return MemoryResult(
                summary="用户偏好被成功召回。",
                relevant_memories=["Preference: 用户偏好短回答"],
                recall_diagnostics={"selected_count": 1},
            )

    scheduler = RealtimeCognitiveScheduler(
        clock=_clock(),
        memory_orchestrator=MemoryOrchestrator(memory_service=PartialMemory()),
    )
    scheduler.observe_partial("准备记忆闭环")
    turn = scheduler.current_turn()
    assert turn is not None
    writeback = scheduler.memory_orchestrator.build_writeback_proposal(
        turn,
        query="需要写回但服务缺少 writer",
        reason="missing_writer_regression",
        summary="这条写回会失败。",
    )
    recall = scheduler.memory_orchestrator.build_recall_request(
        turn,
        query="用户偏好短回答",
        reason="recall_after_writeback_failure",
    )

    trace = scheduler.commit_memory_candidates(session_id="s1", actor_id="user-1")

    assert trace["errors"][0]["error"] == "remember_episode_missing"
    assert writeback.get("committed") is None
    assert recall["committed"] is True
    assert recall["commit_status"] == "ok"
    assert recall["resolved_count"] == 1


def test_scheduler_prefetch_uses_eimemory_recall_before_slow_decision() -> None:
    class MemorySpy:
        def __init__(self) -> None:
            self.queries = []
            self.memory_traces = []

        def retrieve_context(self, query):
            self.queries.append(query)
            return MemoryResult(
                summary="妈妈通常晚上八点后方便接电话。",
                relevant_memories=["Family: 妈妈通常晚上八点后方便接电话。"],
                recall_diagnostics={
                    "selected_count": 1,
                    "selected_records": [
                        {
                            "record_id": "mem_mom_1",
                            "title": "Family contact window",
                            "source": "eibrain.preference",
                            "score": 0.93,
                        }
                    ],
                    "source_composition": {"eibrain.preference": 1},
                },
            )

        def record_memory_trace(self, payload, *, session_id=None, actor_id=None):
            self.memory_traces.append(
                {
                    "payload": dict(payload),
                    "session_id": session_id,
                    "actor_id": actor_id,
                }
            )
            return {"ok": True, "result": {"record_id": "trace_recall_1"}}

    memory = MemorySpy()
    scheduler = RealtimeCognitiveScheduler(
        clock=_clock(),
        memory_orchestrator=MemoryOrchestrator(memory_service=memory),
    )

    partial = scheduler.observe_partial(
        "上次妈妈什么时候方便接电话",
        session_id="voice-session",
        actor_id="darrow",
    )
    scheduler.observe_final("上次妈妈什么时候方便接电话")
    result = scheduler.decide(session_id="voice-session", actor_id="darrow")
    snapshot = scheduler.snapshot()

    assert memory.queries[0].query == "上次妈妈什么时候方便接电话"
    assert memory.queries[0].session_id == "voice-session"
    assert memory.queries[0].actor_id == "darrow"
    assert memory.queries[0].task_context["phase"] == "fast_prefetch"
    assert partial["memory_prefetch"][0]["source"] == "eimemory_recall"
    assert partial["memory_prefetch"][0]["record_id"] == "mem_mom_1"
    assert "八点后" in partial["memory_prefetch"][0]["text"]
    assert result["decision"]["memory_refs"][0]["id"] == "mem_mom_1"
    assert snapshot["current"]["memory_traces"][0]["recall"]["count"] == 1
    assert snapshot["current"]["memory_traces"][0]["trace_record"]["record_id"] == "trace_recall_1"
    assert memory.memory_traces[0]["session_id"] == "voice-session"


def test_scheduler_decide_auto_commits_explicit_memory_writeback() -> None:
    class MemorySpy:
        last_writeback_status = {"status": "idle"}

        def __init__(self) -> None:
            self.episodes = []
            self.memory_traces = []

        def remember_episode(self, **kwargs):
            self.episodes.append(dict(kwargs))
            self.last_writeback_status = {
                "status": "ok",
                "source": kwargs.get("source"),
                "memory_type": kwargs.get("memory_type"),
                "modality": kwargs.get("modality"),
                "organ": kwargs.get("organ"),
                "record_id": "dialogue_mem_1",
            }

        def record_memory_trace(self, payload, *, session_id=None, actor_id=None):
            self.memory_traces.append(
                {
                    "payload": dict(payload),
                    "session_id": session_id,
                    "actor_id": actor_id,
                }
            )
            return {"ok": True, "result": {"record_id": "trace_writeback_1"}}

    memory = MemorySpy()
    scheduler = RealtimeCognitiveScheduler(
        clock=_clock(),
        memory_orchestrator=MemoryOrchestrator(memory_service=memory),
    )

    scheduler.observe_final("记住我喜欢短回答")
    result = scheduler.decide(session_id="voice-session", actor_id="darrow")
    snapshot = scheduler.snapshot()

    assert memory.episodes
    payload = memory.episodes[0]
    assert payload["session_id"] == "voice-session"
    assert payload["actor_id"] == "darrow"
    assert payload["summary"].startswith("user:记住我喜欢短回答 | reply:")
    assert payload["memory_type"] == "conversation"
    assert payload["source"] == "eibrain.audio_dialogue"
    assert payload["modality"] == "audio_text"
    assert payload["organ"] == "ear"
    assert payload["outcome"]["status"] == "planned"
    assert payload["content"]["event_type"] == "dialogue_turn"
    assert payload["meta"]["round_id"] == result["round_id"]
    assert result["memory_trace"]["writeback"]["items"][0]["record_id"] == "dialogue_mem_1"
    assert snapshot["scheduler"]["memory_trace_count"] == 1
    assert snapshot["current"]["memory_traces"][0]["trace_record"]["record_id"] == "trace_writeback_1"


def test_scheduler_memory_writeback_failure_does_not_block_decision() -> None:
    class FailingMemory:
        last_writeback_status = {"status": "idle"}

        def remember_episode(self, **kwargs):
            raise RuntimeError("eimemory down")

        def record_memory_trace(self, payload, *, session_id=None, actor_id=None):
            return {"ok": True, "result": {"record_id": "trace_error_1"}}

    scheduler = RealtimeCognitiveScheduler(
        clock=_clock(),
        memory_orchestrator=MemoryOrchestrator(memory_service=FailingMemory()),
    )

    scheduler.observe_final("记住我喜欢短回答")
    result = scheduler.decide(session_id="voice-session", actor_id="darrow")

    assert result["decision"]["stable"] is True
    assert result["memory_trace"]["writeback"]["items"][0]["status"] == "error"
    assert result["memory_trace"]["errors"][0]["error"] == "RuntimeError: eimemory down"
