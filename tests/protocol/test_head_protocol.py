from __future__ import annotations

import json


def _round_trip(model):
    payload = json.loads(json.dumps(model.to_dict()))
    return type(model).from_dict(payload), payload


def test_audio_turn_round_trips_as_head_observation() -> None:
    from eibrain.protocol.envelopes import Envelope
    from eibrain.protocol.head import AudioTurn

    turn = AudioTurn(
        ts=2.0,
        source="eihead.honjia.ear",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_100,
        trace_id="trace-audio",
        session_id="session-1",
        device_id="mic.u4k",
        text="你好鸿途",
        language="zh",
        is_final=True,
        audio_level=0.62,
    )

    restored, payload = _round_trip(turn)
    envelope = Envelope.wrap(channel="observations", payload=restored)

    assert restored.to_dict() == payload
    assert restored.kind == "audio_turn"
    assert restored.observation_type == "audio_turn"
    assert envelope.payload["trace_id"] == "trace-audio"
    assert envelope.payload["target"] == "eibrain.honxin"


def test_vision_observation_carries_detections() -> None:
    from eibrain.protocol.head import VisionObservation

    observation = VisionObservation(
        ts=3.0,
        source="eihead.honjia.eye",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_200,
        trace_id="trace-vision",
        frame_id="frame-42",
        width=1280,
        height=720,
        detections=[
            {
                "label": "person",
                "score": 0.91,
                "bbox": [0.3, 0.2, 0.4, 0.5],
            }
        ],
        tracked_target={"label": "person", "center_x": 0.5},
    )

    restored, payload = _round_trip(observation)

    assert restored.to_dict() == payload
    assert restored.kind == "vision_observation"
    assert restored.detections[0]["score"] == 0.91
    assert restored.tracked_target["center_x"] == 0.5


def test_device_status_head_action_and_outcome_round_trip() -> None:
    from eibrain.protocol.capabilities import HeadHealth
    from eibrain.protocol.head import DeviceStatus, ExecutionOutcome, HeadAction

    status = DeviceStatus(
        ts=4.0,
        source="eihead.honjia",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_300,
        trace_id="trace-device",
        device_id="neck.pan",
        device_kind="pan_servo",
        health=HeadHealth(status="ok", metrics={"angle": 90}),
        metrics={"angle": 90, "jitter": 0.01},
    )
    action = HeadAction(
        ts=5.0,
        source="eibrain.honxin",
        target="eihead.honjia",
        timestamp_ms=1_714_800_000_400,
        trace_id="trace-action",
        action_id="action-1",
        action_type="move_head",
        device_id="neck.pan",
        params={"target_angle": 92, "reason": "center_target"},
    )
    outcome = ExecutionOutcome(
        ts=6.0,
        source="eihead.honjia",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_500,
        trace_id="trace-action",
        action_id="action-1",
        action_type="move_head",
        device_id="neck.pan",
        latency_ms=23.5,
        details={"final_angle": 92},
    )

    restored_status, status_payload = _round_trip(status)
    restored_action, action_payload = _round_trip(action)
    restored_outcome, outcome_payload = _round_trip(outcome)

    assert restored_status.to_dict() == status_payload
    assert restored_action.to_dict() == action_payload
    assert restored_outcome.to_dict() == outcome_payload
    assert restored_status.health.metrics["angle"] == 90
    assert restored_action.params["target_angle"] == 92
    assert restored_outcome.details["final_angle"] == 92


def test_user_feedback_round_trips_without_safety_gating() -> None:
    from eibrain.protocol.head import UserFeedback

    feedback = UserFeedback(
        ts=7.0,
        source="eihead.honjia",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_600,
        trace_id="trace-feedback",
        feedback_type="interaction",
        value="positive",
        score=0.8,
        text="response was faster",
        related_trace_id="trace-action",
    )

    restored, payload = _round_trip(feedback)

    assert restored.to_dict() == payload
    assert restored.kind == "user_feedback"
    assert "permission" not in restored.to_dict()
