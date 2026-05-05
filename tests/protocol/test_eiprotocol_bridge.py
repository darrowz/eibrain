from __future__ import annotations

import json
from typing import Any


def _round_trip_event(event):
    from eiprotocol import EventEnvelope, validate_event

    payload = json.loads(event.to_json())
    restored = EventEnvelope.from_dict(payload)

    assert restored.to_dict() == payload
    assert validate_event(restored) == []
    return restored, payload


def _assert_strict_round_trip_and_route(event: Any, expected_route: str) -> dict[str, Any]:
    from eiprotocol import classify_event
    from eiprotocol.codec import dumps_event, loads_event
    from eiprotocol.validation import validate_event_strict

    payload = event.to_dict()

    assert validate_event_strict(payload, known_event_required=True) == []
    assert loads_event(dumps_event(event)).to_dict() == payload
    route = classify_event(event)
    assert route["status"] == "routed"
    assert route["route"] == expected_route
    return payload


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


def test_wave3_bridge_converts_internal_head_status_payload_to_typed_event() -> None:
    from eibrain.protocol.eiprotocol_bridge import head_status_report_to_eiprotocol_event

    event = head_status_report_to_eiprotocol_event(
        {
            "node_id": "honjia",
            "overall_status": "degraded",
            "captured_at_ts": 1_714_800_090.0,
            "capabilities": {
                "camera.front": {"status": "online", "metrics": {"fps": 29}},
                "mouth.tts": {"status": "degraded", "message": "queue backing up"},
            },
            "summary": {"online": 1, "degraded": 1, "offline": 0, "total": 2},
            "trace_id": "trace-head-status",
        },
        target="eibrain.honxin",
        event_id="evt_head_status_bridge",
        request_id="req_head_status_bridge",
        sequence=7,
        time="2026-05-04T10:31:30.000+08:00",
    )

    payload = _assert_strict_round_trip_and_route(event, "head_status_report")
    assert payload["source"]["domain"] == "eihead"
    assert payload["source"]["instanceId"] == "honjia"
    assert payload["source"]["deviceId"] == "honjia"
    assert payload["target"]["domain"] == "eibrain"
    assert payload["content"]["status"] == "degraded"
    assert payload["content"]["components"]["mouth.tts"]["status"] == "degraded"
    assert payload["content"]["reportedAt"] == "2024-05-04T05:21:30.000Z"


def test_wave3_bridge_converts_internal_dialogue_payloads_to_typed_events() -> None:
    from eibrain.protocol.eiprotocol_bridge import (
        dialogue_fast_hypothesis_to_eiprotocol_event,
        dialogue_stable_decision_to_eiprotocol_event,
    )

    fast = dialogue_fast_hypothesis_to_eiprotocol_event(
        {
            "hypothesis_id": "hyp-1",
            "text": "The user is asking about snacks.",
            "confidence": 0.69,
            "basis_event_id": "evt_asr_partial",
            "latency_ms": 88,
            "session_id": "ses-1",
            "round_id": "rnd-1",
            "trace_id": "trace-dialogue",
        },
        target="eihead.honjia",
        event_id="evt_fast_bridge",
        request_id="req_fast_bridge",
        sequence=8,
        time="2026-05-04T10:32:00.420+08:00",
    )
    stable = dialogue_stable_decision_to_eiprotocol_event(
        {
            "decision_id": "decision-1",
            "decision": "respond",
            "confidence": 0.94,
            "text": "Fuzhou fish balls are a classic choice.",
            "actions": [{"type": "speak", "text": "Fuzhou fish balls are a classic choice."}],
            "stable_since_ms": 310,
            "session_id": "ses-1",
            "round_id": "rnd-1",
            "trace_id": "trace-dialogue",
        },
        target="eihead.honjia",
        event_id="evt_stable_bridge",
        request_id="req_stable_bridge",
        sequence=9,
        time="2026-05-04T10:32:01.700+08:00",
    )

    fast_payload = _assert_strict_round_trip_and_route(fast, "dialogue_fast_hypothesis")
    stable_payload = _assert_strict_round_trip_and_route(stable, "dialogue_decision_stable")
    assert fast_payload["source"]["domain"] == "eibrain"
    assert fast_payload["target"]["domain"] == "eihead"
    assert fast_payload["roundId"] == "rnd-1"
    assert fast_payload["content"]["hypothesisId"] == "hyp-1"
    assert fast_payload["content"]["basisEventId"] == "evt_asr_partial"
    assert stable_payload["roundId"] == "rnd-1"
    assert stable_payload["content"]["decisionId"] == "decision-1"
    assert stable_payload["content"]["actions"] == [
        {"type": "speak", "text": "Fuzhou fish balls are a classic choice."}
    ]


def test_bridge_converts_realtime_vision_payload_to_head_to_brain_frame_event() -> None:
    from eibrain.protocol.eiprotocol_bridge import realtime_vision_payload_to_eiprotocol_event

    event = realtime_vision_payload_to_eiprotocol_event(
        {
            "kind": "realtime_vision_frame",
            "source": "eihead.honjia.eye",
            "target": "eibrain.honxin",
            "device_id": "camera.front",
            "trace_id": "trace-vision-1",
            "session_id": "session-vision-1",
            "frame_id": "frame-bridge-1",
            "width": 1280,
            "height": 720,
            "backend": "hailo",
            "detections": [
                {
                    "label": "person",
                    "score": 0.91,
                    "bbox": {"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6},
                    "track_id": "track-person-1",
                }
            ],
            "boxes": [{"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6}],
            "scores": [0.91],
            "tracked_target": {"label": "person", "track_id": "track-person-1"},
            "latency_ms": {"capture": 8.0, "detect": 15.0, "publish": 3.5},
            "metadata": {"scene": "lab"},
        },
        event_id="evt_vision_bridge",
        request_id="req_vision_bridge",
        sequence=5,
        time="2026-05-05T09:40:00.000+08:00",
    )

    payload = _assert_strict_round_trip_and_route(event, "realtime_vision_frame")
    assert payload["name"] == "ei.observation.vision.frame"
    assert payload["type"] == "observation"
    assert payload["source"]["domain"] == "eihead"
    assert payload["source"]["instanceId"] == "honjia"
    assert payload["source"]["deviceId"] == "camera.front"
    assert payload["target"]["domain"] == "eibrain"
    assert payload["content"]["frameId"] == "frame-bridge-1"
    assert payload["content"]["trackedTarget"] == {"label": "person", "track_id": "track-person-1"}
    assert payload["content"]["boxes"] == [{"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6}]
    assert payload["content"]["scores"] == [0.91]
    assert payload["content"]["latencyMs"] == {"capture": 8.0, "detect": 15.0, "publish": 3.5}
    assert payload["content"]["metadata"] == {"scene": "lab"}


def test_payload_to_eiprotocol_event_routes_realtime_vision_payload_kind() -> None:
    from eibrain.protocol.eiprotocol_bridge import payload_to_eiprotocol_event

    event = payload_to_eiprotocol_event(
        {
            "type": "ei.observation.vision.frame",
            "source": "eihead.honjia.eye",
            "target": "eibrain.honxin",
            "frameId": "frame-routed-1",
            "boxes": [[0.1, 0.2, 0.3, 0.4]],
            "scores": [0.88],
            "metadata": {"origin": "realtime_detector"},
        },
        event_id="evt_vision_payload",
        request_id="req_vision_payload",
        sequence=6,
        time="2026-05-05T09:40:00.050+08:00",
    )

    payload = _assert_strict_round_trip_and_route(event, "realtime_vision_frame")
    assert payload["content"]["frameId"] == "frame-routed-1"
    assert payload["content"]["boxes"] == [[0.1, 0.2, 0.3, 0.4]]
    assert payload["content"]["scores"] == [0.88]


def test_payload_to_eiprotocol_event_routes_v011_generic_payload_kinds() -> None:
    from eibrain.protocol.eiprotocol_bridge import payload_to_eiprotocol_event

    cases = [
        (
            "ei.observation.vision.scene",
            "ei.observation.vision.scene",
            "observation",
            "vision_scene",
            {
                "scene_id": "scene-1",
                "summary": "A person is standing near the desk.",
                "objects": [{"label": "person", "score": 0.94}],
                "metadata": {"lighting": "indoor"},
            },
            {"sceneId": "scene-1", "summary": "A person is standing near the desk."},
        ),
        (
            "vision_scene",
            "ei.observation.vision.scene",
            "observation",
            "vision_scene",
            {
                "sceneId": "scene-2",
                "summary": "The desk is clear.",
                "objects": [{"label": "desk", "score": 0.87}],
            },
            {"sceneId": "scene-2", "summary": "The desk is clear."},
        ),
        (
            "ei.observation.vision.event",
            "ei.observation.vision.event",
            "observation",
            "vision_event",
            {
                "vision_event_id": "vision-event-1",
                "event_type": "object_entered",
                "subject": {"label": "person", "track_id": "track-1"},
                "metadata": {"zone": "left"},
            },
            {"eventId": "vision-event-1", "eventType": "object_entered"},
        ),
        (
            "vision_event",
            "ei.observation.vision.event",
            "observation",
            "vision_event",
            {
                "eventId": "vision-event-2",
                "eventType": "object_left",
                "subject": {"label": "person", "track_id": "track-2"},
            },
            {"eventId": "vision-event-2", "eventType": "object_left"},
        ),
        (
            "ei.memory.policy.report",
            "ei.memory.policy.report",
            "memory",
            "memory_policy_report",
            {
                "report_id": "memory-policy-1",
                "scope": {"operation": "write"},
                "decision": "allow",
                "reason": "salient user preference",
                "writes": [{"memory_id": "mem-1", "status": "proposed"}],
            },
            {"policyId": "memory-policy-1", "decision": "allow"},
        ),
        (
            "memory_policy_report",
            "ei.memory.policy.report",
            "memory",
            "memory_policy_report",
            {
                "reportId": "memory-policy-2",
                "scope": {"operation": "writeback"},
                "decision": "deny",
                "reason": "low confidence",
                "metadata": {"policy": "salient_or_user_requested"},
            },
            {"policyId": "memory-policy-2", "decision": "deny"},
        ),
    ]

    for kind, expected_name, expected_type, expected_route, extra_payload, expected_content in cases:
        event = payload_to_eiprotocol_event(
            {
                "kind": kind,
                "source": "eibrain.honxin",
                "target": "eimemory.default",
                "session_id": "session-v011",
                "round_id": "round-v011",
                "trace_id": f"trace-{kind}",
                **extra_payload,
            },
            event_id=f"evt_{kind.replace('.', '_')}",
            request_id=f"req_{kind.replace('.', '_')}",
            sequence=11,
            time="2026-05-05T10:00:00.000+08:00",
        )

        payload = _assert_strict_round_trip_and_route(event, expected_route)
        assert payload["name"] == expected_name
        assert payload["type"] == expected_type
        assert payload["source"]["domain"] == "eibrain"
        assert payload["target"]["domain"] == "eimemory"
        assert payload["roundId"] == ("round-v011" if expected_type == "memory" else "")
        for key, value in expected_content.items():
            assert payload["content"][key] == value


def test_v011_vision_bridge_defaults_to_head_to_brain_direction() -> None:
    from eibrain.protocol.eiprotocol_bridge import payload_to_eiprotocol_event

    event = payload_to_eiprotocol_event(
        {
            "kind": "vision_scene",
            "sceneId": "scene-default-direction",
            "observedAt": "2026-05-05T10:01:00.000+08:00",
        },
        event_id="evt_vision_scene_default_direction",
        request_id="req_vision_scene_default_direction",
        sequence=12,
        time="2026-05-05T10:01:00.100+08:00",
    )

    payload = _assert_strict_round_trip_and_route(event, "vision_scene")
    assert payload["source"]["domain"] == "eihead"
    assert payload["source"]["instanceId"] == "honjia"
    assert payload["target"]["domain"] == "eibrain"
    assert payload["target"]["instanceId"] == "honxin"
