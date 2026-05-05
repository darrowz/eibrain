from __future__ import annotations

import json

import pytest

from eibrain.cognition.realtime import CancellationToken, RealtimeTurnManager, ResponseArbiter


def _clock():
    values = iter([2000.0, 2000.05, 2000.1, 2000.15, 2000.2, 2000.25, 2000.3, 2000.35])

    def clock() -> float:
        return next(values)

    return clock


def test_status_payload_exposes_cancellable_turn_state() -> None:
    manager = RealtimeTurnManager(clock=_clock())

    turn = manager.start_round(reason="voice_activity")
    manager.observe_partial(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="打开",
    )
    manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="打开客厅灯",
    )

    payload = manager.status_payload()

    assert isinstance(turn.cancellation, CancellationToken)
    assert turn.cancellation.round_id == turn.round_id
    assert turn.cancellation.token_id == turn.cancellation_token
    assert turn.cancellation.cancelled is False
    assert payload["active"] is True
    assert payload["current_round_id"] == turn.round_id
    assert payload["current"]["asr_partial"] == ["打开"]
    assert payload["current"]["asr_final"] == "打开客厅灯"
    assert payload["current"]["cancellation"]["cancelled"] is False
    assert json.loads(json.dumps(payload, ensure_ascii=False))["current_round_id"] == turn.round_id


def test_fast_lane_records_only_hypotheses_and_stable_decisions_are_explicit() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    turn = manager.start_round()

    hypothesis = manager.write_fast_hypothesis(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        hypothesis={"intent": "turn_on_light", "confidence": 0.62},
        source="partial_asr",
    )

    assert hypothesis["stable"] is False
    assert hypothesis["source"] == "partial_asr"
    assert turn.fast_hypotheses == [hypothesis]
    assert turn.stable_decisions == []

    with pytest.raises(ValueError, match="fast lane"):
        manager.write_fast_hypothesis(
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
            hypothesis={"intent": "turn_on_light", "stable": True},
        )

    with pytest.raises(ValueError, match="stable"):
        manager.commit_stable_decision(
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
            decision={"speech_segments": [{"text": "好的", "stable": True}]},
        )

    decision = manager.commit_stable_decision(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        decision={
            "stable": True,
            "decision": "respond",
            "speech_segments": [{"text": "好的，我来处理。", "stable": True}],
        },
    )

    assert decision["stable"] is True
    assert decision["round_id"] == turn.round_id
    assert decision["cancellation_token"] == turn.cancellation_token
    assert turn.stable_decisions == [decision]


def test_interrupt_cancels_old_token_and_blocks_stale_outputs() -> None:
    manager = RealtimeTurnManager(clock=_clock())
    arbiter = ResponseArbiter()
    old_turn = manager.start_round()
    stale_decision = manager.commit_stable_decision(
        round_id=old_turn.round_id,
        cancellation_token=old_turn.cancellation_token,
        decision={
            "stable": True,
            "speech_segments": [{"text": "旧轮次结果", "stable": True}],
        },
    )

    new_turn = manager.interrupt(reason="barge_in")

    assert old_turn.cancellation.cancelled is True
    assert old_turn.cancellation.reason == "barge_in"
    assert manager.is_current(
        round_id=old_turn.round_id,
        cancellation_token=old_turn.cancellation_token,
    ) is False
    assert manager.is_current(
        round_id=new_turn.round_id,
        cancellation_token=new_turn.cancellation_token,
    ) is True
    assert arbiter.allow_speaking(manager, old_turn, stale_decision) is False

    with pytest.raises(RuntimeError, match="not current"):
        manager.write_fast_hypothesis(
            round_id=old_turn.round_id,
            cancellation_token=old_turn.cancellation_token,
            hypothesis={"intent": "late_partial"},
        )

    status = manager.status_payload()
    assert status["history"][old_turn.round_id]["state"] == "interrupted"
    assert status["history"][old_turn.round_id]["cancellation"]["cancelled"] is True
    assert status["current_round_id"] == new_turn.round_id
