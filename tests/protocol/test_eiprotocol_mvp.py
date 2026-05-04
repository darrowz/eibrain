from __future__ import annotations

import json


def _round_trip_event(event):
    from eiprotocol import EventEnvelope

    payload = json.loads(event.to_json())
    restored = EventEnvelope.from_dict(payload)
    return restored, payload


def test_envelope_preserves_round_trace_policy_and_extensions() -> None:
    from eiprotocol import EventEnvelope, PolicyState, SourceRef, TargetRef, validate_event

    event = EventEnvelope(
        event_id="evt_001",
        event_type="dialogue",
        name="ei.dialogue.agent.delta",
        time="2026-05-04T10:30:20.123+08:00",
        sequence=7,
        request_id="req_001",
        session_id="ses_user_bot",
        round_id="rnd_001",
        correlation_id="evt_root",
        causation_id="evt_asr_final",
        trace_id="trc_001",
        source=SourceRef(domain="eibrain", instance_id="honxin", bot_id="bot_hongtu", uid="darrow"),
        target=TargetRef(domain="eihead", instance_id="honjia"),
        priority="realtime",
        ttl_ms=3000,
        mode={"conversationState": "thinking", "interactionMode": "free"},
        content={"delta": "hello"},
        policy=PolicyState(decision="not_required", risk_level="L0"),
        extensions={"joyinside": {"finishReason": ""}},
    )

    restored, payload = _round_trip_event(event)

    assert payload["specVersion"] == "eiprotocol/0.1"
    assert payload["roundId"] == "rnd_001"
    assert payload["traceId"] == "trc_001"
    assert payload["source"]["instanceId"] == "honxin"
    assert payload["policy"]["decision"] == "not_required"
    assert restored.to_dict() == payload
    assert validate_event(restored) == []


def test_capability_manifest_models_honjia_head_devices() -> None:
    from eiprotocol import (
        Capability,
        CapabilityManifest,
        DeviceStatus,
        EventEnvelope,
        SourceRef,
        validate_event,
    )

    manifest = CapabilityManifest(
        manifest_id="cap_honjia_001",
        manifest_version="0.1.0",
        device={"deviceId": "honjia", "model": "raspberry-pi-5-hailo"},
        runtime={"os": "linux", "lan": "tailscale", "timezone": "Asia/Shanghai"},
        capabilities=[
            Capability(capability_id="camera.front", kind="camera", device_path="/dev/video0", status="degraded"),
            Capability(capability_id="accelerator.hailo", kind="vision_accelerator", device_path="/dev/hailo0"),
            Capability(capability_id="microphone.u4k", kind="audio_input", device_path="plughw:CARD=U4K,DEV=0"),
            Capability(capability_id="speaker.default", kind="audio_output"),
            Capability(
                capability_id="neck.pan",
                kind="actuator",
                actions=["move_head"],
                status="online",
                limits={"axis": "pan", "minAngle": 0, "maxAngle": 180, "tiltSupported": False},
            ),
        ],
        backends=[
            Capability(capability_id="asr.sherpa", kind="asr", model="sherpa-onnx"),
            Capability(capability_id="tts.minimax", kind="tts", provider="minimax"),
            Capability(capability_id="vision.hailo", kind="vision", model="face_detection.hef"),
        ],
        health=DeviceStatus(status="degraded", message="camera pending onsite verification"),
    )
    event = manifest.to_event(
        event_id="evt_cap",
        request_id="req_cap",
        sequence=1,
        source=SourceRef(domain="eihead", instance_id="honjia", device_id="honjia"),
        time="2026-05-04T10:31:00+08:00",
    )

    restored, payload = _round_trip_event(event)

    assert isinstance(restored, EventEnvelope)
    assert payload["name"] == "ei.capability.manifest.report"
    assert payload["content"]["capabilities"][4]["limits"]["tiltSupported"] is False
    assert payload["content"]["backends"][1]["provider"] == "minimax"
    assert validate_event(restored) == []


def test_audio_and_realtime_vision_observations_use_round_ids() -> None:
    from eiprotocol import AudioTurn, Detection, RealtimeVisionObservation, SourceRef, validate_event

    source = SourceRef(domain="eihead", instance_id="honjia", device_id="honjia")
    audio = AudioTurn(
        text="hello hongtu",
        language="en-US",
        final=True,
        confidence=0.83,
        start_ms=120,
        end_ms=1460,
        asr_backend="sherpa-onnx",
        timings_ms={"vadEndpoint": 80, "asrFinal": 620},
    ).to_event(
        event_id="evt_asr",
        request_id="req_voice",
        session_id="ses_1",
        round_id="rnd_1",
        sequence=2,
        source=source,
        time="2026-05-04T10:32:00+08:00",
    )
    vision = RealtimeVisionObservation(
        frame_id="frame_42",
        width=1280,
        height=720,
        frame_age_ms=42.5,
        backend="hailo",
        detections=[Detection(label="person", score=0.91, bbox=[0.2, 0.1, 0.4, 0.6])],
        latency_ms={"capture": 8.0, "detect": 15.0},
    ).to_event(
        event_id="evt_vision",
        request_id="req_vision",
        session_id="ses_1",
        round_id="rnd_1",
        sequence=3,
        source=source,
        time="2026-05-04T10:32:00.100+08:00",
    )

    _, audio_payload = _round_trip_event(audio)
    _, vision_payload = _round_trip_event(vision)

    assert audio_payload["name"] == "ei.dialogue.asr.final"
    assert audio_payload["roundId"] == "rnd_1"
    assert audio_payload["content"]["timingsMs"]["asrFinal"] == 620
    assert vision_payload["name"] == "ei.observation.vision.frame"
    assert vision_payload["content"]["detections"][0]["bbox"] == [0.2, 0.1, 0.4, 0.6]
    assert validate_event(audio) == []
    assert validate_event(vision) == []


def test_head_action_requires_idempotency_and_outcomes_feedback_round_trip() -> None:
    from eiprotocol import ExecutionOutcome, HeadAction, SourceRef, UserFeedback, validate_event

    brain = SourceRef(domain="eibrain", instance_id="honxin")
    head = SourceRef(domain="eihead", instance_id="honjia")
    action = HeadAction(
        action_id="act_speak_001",
        action_type="speak",
        target="speaker.default",
        params={"text": "I am listening."},
        risk_level="L1",
        idempotency_key="act_speak_001_once",
    ).to_event(
        event_id="evt_action",
        request_id="req_action",
        session_id="ses_1",
        round_id="rnd_1",
        sequence=4,
        source=brain,
        time="2026-05-04T10:33:00+08:00",
    )
    invalid_action = HeadAction(
        action_id="act_missing_key",
        action_type="move_head",
        target="neck.pan",
        params={"targetAngle": 90},
        risk_level="L2",
    ).to_event(
        event_id="evt_invalid",
        request_id="req_action",
        session_id="ses_1",
        round_id="rnd_1",
        sequence=5,
        source=brain,
        time="2026-05-04T10:33:00+08:00",
    )
    outcome = ExecutionOutcome(
        outcome_id="out_001",
        action_id="act_speak_001",
        action_type="speak",
        success=True,
        status="completed",
        latency_ms=180.5,
        did_what=["queued speech", "played audio"],
    ).to_event(
        event_id="evt_outcome",
        request_id="req_action",
        session_id="ses_1",
        round_id="rnd_1",
        sequence=6,
        source=head,
        time="2026-05-04T10:33:01+08:00",
    )
    feedback = UserFeedback(
        feedback_id="fb_001",
        satisfied=True,
        rating=5,
        text="faster now",
        next_time_change="keep answers short",
    ).to_event(
        event_id="evt_feedback",
        request_id="req_action",
        session_id="ses_1",
        round_id="rnd_1",
        sequence=7,
        source=SourceRef(domain="user", uid="darrow"),
        time="2026-05-04T10:33:05+08:00",
    )

    _, action_payload = _round_trip_event(action)
    _, outcome_payload = _round_trip_event(outcome)
    _, feedback_payload = _round_trip_event(feedback)

    assert action_payload["name"] == "ei.action.request"
    assert action_payload["content"]["idempotencyKey"] == "act_speak_001_once"
    assert validate_event(action) == []
    assert validate_event(invalid_action) == ["content.idempotencyKey is required for side-effecting action events"]
    assert outcome_payload["name"] == "ei.outcome.execution"
    assert outcome_payload["content"]["didWhat"] == ["queued speech", "played audio"]
    assert feedback_payload["name"] == "ei.outcome.user.feedback"
    assert feedback_payload["content"]["nextTimeChange"] == "keep answers short"
