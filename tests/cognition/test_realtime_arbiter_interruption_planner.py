from __future__ import annotations

import json

from eibrain.cognition.realtime import RealtimeTurnManager
from eibrain.cognition.realtime.arbiter import ResponseArbiter
from eibrain.cognition.realtime.interruption import InterruptionController
from eibrain.cognition.realtime.planner import SpeechActionPlanner


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
        ]
    )

    def clock() -> float:
        return next(values)

    return clock


def test_arbiter_only_allows_current_stable_speech_action_plans() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    turn = manager.start_round()
    manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="打开客厅灯",
    )
    turn.tool_candidates.append(
        {
            "capabilityId": "light.living_room.toggle",
            "durationMs": 700,
            "style": "gentle",
        }
    )

    plan = SpeechActionPlanner().plan(
        turn,
        speech_text="好的，我会打开客厅灯。",
        emotion="focused",
    )

    assert ResponseArbiter().allow_speaking(manager, turn, plan) is True
    assert json.loads(json.dumps(plan, ensure_ascii=False))["round_id"] == turn.round_id


def test_arbiter_rejects_unstable_hypotheses_and_stale_cancelled_round_outputs() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    arbiter = ResponseArbiter()
    old_turn = manager.start_round()
    hypothesis = manager.write_fast_hypothesis(
        round_id=old_turn.round_id,
        cancellation_token=old_turn.cancellation_token,
        hypothesis={"intent": "turn_on_light", "speech_segments": [{"text": "我来看看"}]},
        source="partial_asr",
    )
    stable_plan = {
        "round_id": old_turn.round_id,
        "cancellation_token": old_turn.cancellation_token,
        "stable": True,
        "speech_segments": [{"text": "旧轮次结果", "emotion": "warm", "startOffsetMs": 0, "stable": True}],
        "action_segments": [],
    }

    assert arbiter.allow_speaking(manager, old_turn, hypothesis) is False

    manager.interrupt(reason="barge_in")

    assert old_turn.cancellation is not None
    assert old_turn.cancellation.cancelled is True
    assert arbiter.allow_speaking(manager, old_turn, stable_plan) is False


def test_arbiter_validates_canonical_action_plan_segments() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    turn = manager.start_round()
    manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="请打开灯",
    )
    plan = {
        "round_id": turn.round_id,
        "cancellation_token": turn.cancellation_token,
        "stable": True,
        "speech_segments": [{"text": "好的。", "startOffsetMs": 0, "stable": True}],
        "action_plan": [{"type": "action_oriented", "status": "ready"}],
    }

    assert ResponseArbiter().allow_speaking(manager, turn, plan) is False

    plan["action_plan"] = [
        {
            "capabilityId": "light.toggle",
            "startOffsetMs": 120,
            "durationMs": 0,
            "style": "default",
            "status": "ready",
        }
    ]
    assert ResponseArbiter().allow_speaking(manager, turn, plan) is True


def test_arbiter_rejects_non_mapping_or_negative_action_plan_segments() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    turn = manager.start_round()
    manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="请打开灯",
    )
    base_plan = {
        "round_id": turn.round_id,
        "cancellation_token": turn.cancellation_token,
        "stable": True,
        "speech_segments": [{"text": "好的。", "startOffsetMs": 0, "stable": True}],
    }

    mixed_plan = {
        **base_plan,
        "action_plan": [
            "bad-segment",
            {"capabilityId": "light.toggle", "startOffsetMs": 120, "durationMs": 0, "style": "default"},
        ],
    }
    negative_plan = {
        **base_plan,
        "action_plan": [
            {"capabilityId": "light.toggle", "startOffsetMs": -1, "durationMs": -1, "style": "default"},
        ],
    }

    arbiter = ResponseArbiter()
    assert arbiter.allow_speaking(manager, turn, mixed_plan) is False
    assert arbiter.allow_speaking(manager, turn, negative_plan) is False


def test_interruption_controller_emits_complete_cancellation_chain() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    old_turn = manager.start_round()
    old_turn.action_plan.append({"capabilityId": "light.living_room.toggle", "status": "ready"})

    summary = InterruptionController().interrupt_and_start_new_round(
        manager,
        reason="barge_in",
    )

    assert summary["cancellation_chain"] == [
        "stop_tts",
        "cancel_generation",
        "cancel_memory_prefetch",
        "cancel_action_plan",
        "mark_interrupted",
        "start_new_round",
    ]
    assert summary["stop_tts"] is True
    assert summary["cancel_generation"] is True
    assert summary["cancel_memory_prefetch"] is True
    assert summary["cancel_action_plan"] is True
    assert summary["mark_interrupted"]["round_id"] == old_turn.round_id
    assert summary["mark_interrupted"]["state"] == "interrupted"
    assert summary["start_new_round"]["round_id"] != old_turn.round_id
    assert old_turn.cancellation is not None
    assert old_turn.cancellation.cancelled is True


def test_speech_action_planner_outputs_structured_segments_with_offsets() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    turn = manager.start_round()
    manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="请打开客厅灯",
    )
    turn.tool_candidates.append(
        {
            "capabilityId": "light.living_room.toggle",
            "startOffsetMs": 180,
            "durationMs": 650,
            "style": "gentle",
            "payload": {"state": "on"},
        }
    )

    plan = SpeechActionPlanner().plan(
        turn,
        speech_text="好的，我会打开客厅灯。",
        emotion="calm",
    )

    assert plan["speech_segments"] == [
        {
            "text": "好的，我会打开客厅灯。",
            "emotion": "calm",
            "startOffsetMs": 0,
            "stable": True,
            "source": "slow_reasoner",
        }
    ]
    assert plan["action_segments"] == [
        {
            "capabilityId": "light.living_room.toggle",
            "startOffsetMs": 180,
            "durationMs": 650,
            "style": "gentle",
            "payload": {"state": "on"},
            "status": "ready",
        }
    ]
    assert turn.speech_plan is plan
    assert turn.action_plan == plan["action_segments"]


def test_speech_action_planner_uses_fallback_speech_when_action_is_unavailable() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    turn = manager.start_round()
    manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="打开客厅灯",
    )
    turn.tool_candidates.append(
        {
            "capabilityId": "light.living_room.toggle",
            "available": False,
            "fallbackText": "我现在不能直接控制客厅灯，但可以告诉你开关位置。",
            "durationMs": 650,
            "style": "gentle",
        }
    )

    plan = SpeechActionPlanner().plan(
        turn,
        speech_text="好的，我会打开客厅灯。",
        emotion="apologetic",
    )

    assert plan["stable"] is True
    assert plan["speech_segments"][0]["text"] == "我现在不能直接控制客厅灯，但可以告诉你开关位置。"
    assert plan["speech_segments"][0]["emotion"] == "apologetic"
    assert plan["speech_segments"][0]["source"] == "action_unavailable_fallback"
    assert plan["action_segments"][0]["status"] == "unavailable"


def test_speech_action_planner_cancels_ready_actions_when_fallback_is_active() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    turn = manager.start_round()
    manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="请打开客厅灯",
    )
    turn.tool_candidates.append(
        {
            "capabilityId": "light.living_room.toggle",
            "startOffsetMs": 180,
            "durationMs": 650,
            "style": "gentle",
        }
    )

    plan = SpeechActionPlanner().plan(
        turn,
        speech_text="好的，我会打开客厅灯。",
        action_results=[{"ok": False, "id": "light.living_room.toggle", "reason": "device_busy"}],
    )

    assert plan["speech_segments"][0]["source"] == "action_fallback"
    assert all(segment["status"] != "ready" for segment in plan["action_segments"])
    assert any(segment["status"] == "retry" for segment in plan["action_segments"])
    assert plan["speech_action_alignment"]["consistent"] is True
    assert plan["speech_action_alignment"]["ready_action_count"] == 0
    assert plan["speech_action_alignment"]["fallback_active"] is True


def test_speech_action_planner_flags_semantic_action_speech_mismatch() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    turn = manager.start_round()
    manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="请打开客厅灯",
    )
    turn.tool_candidates.append(
        {
            "capabilityId": "light.living_room.toggle",
            "startOffsetMs": 180,
            "durationMs": 650,
            "style": "gentle",
        }
    )

    plan = SpeechActionPlanner().plan(turn, speech_text="我会播放一段音乐。")

    assert plan["speech_action_alignment"]["semantic_checked"] is True
    assert plan["speech_action_alignment"]["consistent"] is False
