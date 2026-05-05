from __future__ import annotations

import json

import pytest

from eibrain.cognition.realtime import (
    FastThinkEngine,
    InterruptionController,
    RealtimeTurnManager,
    ResponseArbiter,
    SpeechActionPlanner,
)
from eibrain.cognition.realtime.turn import TurnBlackboard


def _fixed_clock() -> callable:
    values = iter([1000.0, 1000.05, 1000.1, 1000.15, 1000.2, 1000.25, 1000.3, 1000.35, 1000.4, 1000.45])

    def clock() -> float:
        return next(values)

    return clock


def _incrementing_clock() -> callable:
    current = 1000.0

    def clock() -> float:
        nonlocal current
        current += 0.05
        return current

    return clock


def test_turn_blackboard_json_payload() -> None:
    turn = TurnBlackboard(
        round_id="round-1",
        cancellation_token="tok-1",
        state="active",
        asr_partial=["ni", "hao"],
        asr_final="ni hao",
        intent_hypotheses=[{"intent": "greet", "confidence": 0.7}],
        emotion_state={"valence": 0.4},
        memory_candidates=[{"key": "last_turn"}],
        tool_candidates=[{"name": "tool"}],
        persona_state={"tone": "neutral"},
        safety_state={"risk": "low"},
        speech_plan={"language": "zh-CN"},
        action_plan=[{"type": "none"}],
        stable_speech_segments=[{"text": "ni hao", "stable": True}],
        created_at_ts=1.0,
        updated_at_ts=1.1,
    )

    payload = turn.to_dict()
    dumped = json.dumps(payload, ensure_ascii=False)

    assert payload["round_id"] == "round-1"
    assert payload["cancellation_token"] == "tok-1"
    assert payload["state"] == "active"
    assert payload["speech_plan"]["language"] == "zh-CN"
    assert json.loads(dumped)["round_id"] == "round-1"


def test_realtime_turn_manager_start_round_and_token_isolation() -> None:
    manager = RealtimeTurnManager(clock=_fixed_clock())

    first = manager.start_round()
    second = manager.start_round()

    assert first.round_id != second.round_id
    assert first.cancellation_token != second.cancellation_token
    assert first.cancellation is not None
    assert first.cancellation.cancelled is True
    assert first.cancellation.reason == "superseded"
    assert manager.current_turn() == second
    assert manager.is_current(round_id=second.round_id, cancellation_token=second.cancellation_token)
    assert not manager.is_current(round_id=first.round_id, cancellation_token=first.cancellation_token)


def test_realtime_turn_manager_ten_round_rollover_cancels_superseded_tokens() -> None:
    manager = RealtimeTurnManager(clock=_incrementing_clock())
    turns = [manager.start_round(reason=f"audit-{index}") for index in range(10)]
    current = turns[-1]
    stale_outputs_emitted = 0

    for stale in turns[:-1]:
        stable_plan = {
            "round_id": stale.round_id,
            "cancellation_token": stale.cancellation_token,
            "stable": True,
            "speech_segments": [{"text": "stale", "startOffsetMs": 0, "stable": True}],
            "action_plan": [],
        }
        allowed = ResponseArbiter().allow_speaking(manager, stale, stable_plan)
        stale_outputs_emitted += int(allowed)
        assert stale.cancellation is not None
        assert stale.cancellation.cancelled is True
        assert stale.cancellation.reason == "superseded"

    assert manager.current_turn() == current
    assert stale_outputs_emitted == 0


def test_realtime_fast_think_produces_partial_microfeedback_under_500ms() -> None:
    manager = RealtimeTurnManager(clock=_fixed_clock())
    turn = manager.start_round()
    manager.observe_partial(round_id=turn.round_id, cancellation_token=turn.cancellation_token, asr_text="我想问你天气怎么样？")

    engine = FastThinkEngine()
    result = engine.process_partial(manager.current_turn() or TurnBlackboard(round_id="", cancellation_token=""), "我想问你天气怎么样？")

    assert result.round_id == turn.round_id
    assert result.cancellation_token == turn.cancellation_token
    assert result.deadline_ms <= 500
    assert result.microfeedback
    assert len(result.intent_hypotheses) >= 1
    assert result.stable is False


def test_package_fast_think_engine_uses_realtime_fast_lane_contract() -> None:
    engine = FastThinkEngine()
    assert hasattr(engine, "sanitize_hypothesis")

    turn = RealtimeTurnManager(clock=_fixed_clock()).start_round()
    result = engine.process_partial(turn, "能不能帮我看看灯？")
    payload = result.to_dict()

    assert payload["stable"] is False
    assert payload["micro_feedback"]["text"]
    assert any(item["intent"] == "possible_action_request" for item in payload["intent_hints"])


def test_response_arbiter_blocks_non_stable_or_cancelled_round() -> None:
    manager = RealtimeTurnManager(clock=_fixed_clock())
    turn = manager.start_round()
    manager.finalize_asr(round_id=turn.round_id, cancellation_token=turn.cancellation_token, asr_text="我在家")
    arbiter = ResponseArbiter()

    stable_plan = {
        "round_id": turn.round_id,
        "cancellation_token": turn.cancellation_token,
        "stable": True,
        "speech_segments": [{"text": "ok", "startOffsetMs": 0, "stable": True}],
    }
    assert arbiter.allow_speaking(manager, turn, stable_plan) is True

    hypothesis_plan = {
        "round_id": turn.round_id,
        "cancellation_token": turn.cancellation_token,
        "stable": False,
        "hypothesis": True,
        "speech_segments": [{"text": "maybe", "startOffsetMs": 0, "stable": False}],
    }
    assert arbiter.allow_speaking(manager, turn, hypothesis_plan) is False

    new_turn = manager.interrupt(reason="user interrupts")
    assert turn.state == "interrupted"
    # verify old turn is invalidated after interruption path
    assert manager.current_turn() == new_turn
    assert manager.current_turn() is not turn
    assert manager.is_current(round_id=turn.round_id, cancellation_token=turn.cancellation_token) is False
    assert arbiter.allow_speaking(manager, turn, stable_plan) is False


def test_interrupt_marks_old_round_and_starts_new_round() -> None:
    manager = RealtimeTurnManager(clock=_fixed_clock())
    old = manager.start_round()
    manager.observe_partial(round_id=old.round_id, cancellation_token=old.cancellation_token, asr_text="早")
    new = manager.interrupt(reason="user interrupt")
    controller = InterruptionController()
    summary = controller.summarize(old_turn=old, new_turn=new, reason="user interrupt")

    assert old.state in {"cancelled", "interrupted"}
    assert old.interrupted_at_ts is not None
    assert new.state == "active"
    assert manager.current_turn() == new
    assert new.cancellation_token != old.cancellation_token
    assert summary["stop_tts"] is True
    assert summary["cancel_generation"] is True
    assert summary["cancel_actions"] is True
    assert summary["mark_interrupted"]["round_id"] == old.round_id
    assert summary["start_new_round"]["round_id"] == new.round_id


def test_speech_action_planner_structured_output_and_failure_fallback() -> None:
    manager = RealtimeTurnManager(clock=_fixed_clock())
    turn = manager.start_round()
    manager.finalize_asr(round_id=turn.round_id, cancellation_token=turn.cancellation_token, asr_text="请打开灯")
    turn.tool_candidates = [
        {"type": "move_head_action", "name": "look_up", "startOffsetMs": 50},
        {"type": "play_animation", "name": "nod", "startOffsetMs": 220},
    ]

    planner = SpeechActionPlanner()
    plan_ok = planner.plan(turn)
    assert plan_ok["round_id"] == turn.round_id
    assert plan_ok["cancellation_token"] == turn.cancellation_token
    assert plan_ok["language"] == "zh-CN"
    assert plan_ok["stable"] is True
    assert plan_ok["speech"] == plan_ok["speech_segments"]
    assert plan_ok["actions"] == plan_ok["action_plan"]
    assert plan_ok["speech_segments"][0]["startOffsetMs"] == 0
    assert plan_ok["speech_segments"][0]["text"] != turn.asr_final
    assert plan_ok["speech_segments"][0]["source"] == "safe_ack"
    assert all("startOffsetMs" in item for item in plan_ok["action_plan"])
    assert all("capabilityId" in item for item in plan_ok["action_plan"])

    plan_failed = planner.plan(
        turn,
        action_results=[{"ok": False, "id": "a1", "reason": "动作失败"}],
    )
    assert plan_failed["stable"] is True
    assert any(segment.get("source") == "action_fallback" for segment in plan_failed["speech_segments"])
    assert any(item["status"] == "retry" for item in plan_failed["action_plan"])
    assert all(item["status"] != "ready" for item in plan_failed["action_plan"])


def test_reject_if_cancelled_guard_refuses_stale_updates() -> None:
    manager = RealtimeTurnManager(clock=_fixed_clock())
    first = manager.start_round()
    second = manager.interrupt(reason="user interrupt")

    with pytest.raises(RuntimeError, match="not current"):
        manager.observe_partial(
            round_id=first.round_id,
            cancellation_token=first.cancellation_token,
            asr_text="late",
        )
    assert manager.current_turn() == second
