from __future__ import annotations

import json


def _round_trip_event(event):
    from eiprotocol import EventEnvelope, validate_event

    payload = json.loads(event.to_json())
    restored = EventEnvelope.from_dict(payload)

    assert restored.to_dict() == payload
    assert validate_event(restored) == []
    return restored, payload


def test_bridge_capability_manifest_to_eiprotocol_envelope() -> None:
    from eibrain.protocol import to_eiprotocol_event
    from eibrain.protocol.capabilities import (
        CapabilityManifest,
        HeadBackend,
        HeadDevice,
        HeadHealth,
        HeadLimit,
    )
    from eiprotocol import EventEnvelope

    manifest = CapabilityManifest(
        ts=1.0,
        source="eihead.honjia",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_000,
        trace_id="trace-cap",
        node_id="honjia",
        devices=[
            HeadDevice(
                device_id="camera.main",
                kind="camera",
                name="U4K",
                path="/dev/video0",
                capabilities=["frame_capture"],
                limits=[HeadLimit(name="fps", min_value=1, max_value=30, unit="hz")],
                health=HeadHealth(status="ok"),
            )
        ],
        backends=[
            HeadBackend(
                backend_id="vision.hailo",
                kind="vision",
                provider="hailo",
                model="face_detection.hef",
                capabilities=["detection"],
            )
        ],
        capabilities=["audio_turn", "vision_observation"],
        health=HeadHealth(status="ok", message="ready"),
        metadata={"site": "lab"},
    )

    event = to_eiprotocol_event(
        manifest,
        event_id="evt_cap_bridge",
        request_id="req_cap_bridge",
        sequence=1,
        time="2026-05-04T10:31:00+08:00",
    )
    restored, payload = _round_trip_event(event)

    assert isinstance(restored, EventEnvelope)
    assert payload["name"] == "ei.capability.manifest.report"
    assert payload["type"] == "capability"
    assert payload["source"]["domain"] == "eihead"
    assert payload["source"]["instanceId"] == "honjia"
    assert payload["target"]["domain"] == "eibrain"
    assert payload["content"]["manifestId"] == "honjia"
    assert payload["content"]["manifestVersion"] == "head.v1"
    assert payload["content"]["capabilities"][0]["capabilityId"] == "camera.main"
    assert payload["content"]["capabilities"][0]["limits"]["fps"]["max_value"] == 30
    assert payload["content"]["backends"][0]["capabilityId"] == "vision.hailo"
    assert payload["content"]["metadata"]["legacyCapabilities"] == ["audio_turn", "vision_observation"]


def test_bridge_audio_turn_uses_final_flag_for_event_name() -> None:
    from eibrain.protocol.eiprotocol_bridge import audio_turn_to_eiprotocol_event
    from eibrain.protocol.head import AudioTurn

    turn = AudioTurn(
        ts=2.0,
        source="eihead.honjia.ear",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_100,
        trace_id="trace-audio",
        session_id="session-1",
        device_id="mic.u4k",
        text="still listening",
        language="en-US",
        is_final=False,
        start_ms=120,
        end_ms=480,
        audio_level=0.62,
        payload={"vad": "active"},
    )

    event = audio_turn_to_eiprotocol_event(
        turn,
        event_id="evt_asr_partial",
        request_id="req_voice",
        sequence=2,
        time="2026-05-04T10:32:00+08:00",
    )
    _, payload = _round_trip_event(event)

    assert payload["name"] == "ei.dialogue.asr.partial"
    assert payload["type"] == "dialogue"
    assert payload["roundId"] == "session-1"
    assert payload["source"]["deviceId"] == "mic.u4k"
    assert payload["content"]["text"] == "still listening"
    assert payload["content"]["final"] is False
    assert payload["content"]["audioLevel"] == 0.62
    assert payload["content"]["metadata"]["legacyPayload"] == {"vad": "active"}


def test_bridge_vision_observation_preserves_detections() -> None:
    from eibrain.protocol.eiprotocol_bridge import vision_observation_to_eiprotocol_event
    from eibrain.protocol.head import VisionObservation

    detections = [
        {
            "label": "person",
            "score": 0.91,
            "bbox": [0.3, 0.2, 0.4, 0.5],
            "track_id": "track-1",
        }
    ]
    observation = VisionObservation(
        ts=3.0,
        source="eihead.honjia.eye",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_200,
        trace_id="trace-vision",
        session_id="session-1",
        frame_id="frame-42",
        image_url="file:///frames/frame-42.jpg",
        width=1280,
        height=720,
        detections=detections,
        tracked_target={"label": "person", "center_x": 0.5},
    )

    event = vision_observation_to_eiprotocol_event(
        observation,
        event_id="evt_vision",
        request_id="req_vision",
        sequence=3,
        time="2026-05-04T10:32:00.100+08:00",
    )
    _, payload = _round_trip_event(event)

    assert payload["name"] == "ei.observation.vision.frame"
    assert payload["type"] == "observation"
    assert payload["content"]["frameId"] == "frame-42"
    assert payload["content"]["imageUrl"] == "file:///frames/frame-42.jpg"
    assert payload["content"]["detections"] == detections
    assert payload["content"]["trackedTarget"] == {"label": "person", "center_x": 0.5}


def test_bridge_head_action_derives_missing_idempotency_key_from_action_id() -> None:
    from eibrain.protocol.eiprotocol_bridge import head_action_to_eiprotocol_event
    from eibrain.protocol.head import HeadAction

    action = HeadAction(
        ts=5.0,
        source="eibrain.honxin",
        target="eihead.honjia",
        timestamp_ms=1_714_800_000_400,
        trace_id="trace-action",
        session_id="session-1",
        action_id="action-1",
        action_type="move_head",
        device_id="neck.pan",
        params={"target_angle": 92, "reason": "center_target"},
    )

    event = head_action_to_eiprotocol_event(action)
    repeated = head_action_to_eiprotocol_event(action)
    _, payload = _round_trip_event(event)

    assert event.to_dict() == repeated.to_dict()
    assert payload["id"] == "evt_head_action_action-1"
    assert payload["requestId"] == "trace-action"
    assert payload["sequence"] == 0
    assert payload["name"] == "ei.action.request"
    assert payload["type"] == "action"
    assert payload["roundId"] == "session-1"
    assert payload["content"]["actionId"] == "action-1"
    assert payload["content"]["target"] == "neck.pan"
    assert payload["content"]["params"] == {"target_angle": 92, "reason": "center_target"}
    assert payload["content"]["idempotencyKey"] == "action-1"


def test_bridge_execution_outcome_to_eiprotocol_envelope() -> None:
    from eibrain.protocol.eiprotocol_bridge import execution_outcome_to_eiprotocol_event
    from eibrain.protocol.head import ExecutionOutcome

    outcome = ExecutionOutcome(
        ts=6.0,
        source="eihead.honjia",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_500,
        trace_id="trace-action",
        session_id="session-1",
        action_id="action-1",
        action_type="move_head",
        device_id="neck.pan",
        status="completed",
        success=True,
        latency_ms=23.5,
        details={"final_angle": 92},
    )

    event = execution_outcome_to_eiprotocol_event(
        outcome,
        event_id="evt_outcome",
        request_id="req_action",
        sequence=4,
        time="2026-05-04T10:33:01+08:00",
    )
    _, payload = _round_trip_event(event)

    assert payload["name"] == "ei.outcome.execution"
    assert payload["type"] == "outcome"
    assert payload["roundId"] == "session-1"
    assert payload["content"]["outcomeId"] == "outcome-action-1"
    assert payload["content"]["actionId"] == "action-1"
    assert payload["content"]["success"] is True
    assert payload["content"]["details"] == {"final_angle": 92}
