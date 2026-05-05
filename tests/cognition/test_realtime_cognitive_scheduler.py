from __future__ import annotations

import json

import pytest

from eibrain.cognition.realtime import (
    ProactiveActivityManager,
    RealtimeCognitiveScheduler,
    SlowReasoner,
)


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
