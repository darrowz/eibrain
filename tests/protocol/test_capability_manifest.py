from __future__ import annotations

import json


def test_capability_manifest_round_trips_through_json() -> None:
    from eibrain.protocol.capabilities import (
        CapabilityManifest,
        HeadBackend,
        HeadDevice,
        HeadHealth,
        HeadLimit,
    )
    from eibrain.protocol.envelopes import Envelope

    manifest = CapabilityManifest(
        ts=1.0,
        source="eihead.honjia",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_000,
        trace_id="trace-001",
        node_id="honjia",
        devices=[
            HeadDevice(
                device_id="camera.main",
                kind="camera",
                name="U4K",
                path="/dev/video0",
                capabilities=["frame_capture", "video_stream"],
                limits=[HeadLimit(name="fps", min_value=1, max_value=30, unit="hz")],
                health=HeadHealth(status="ok", metrics={"fps": 15}),
            )
        ],
        backends=[
            HeadBackend(
                backend_id="vision.hailo",
                kind="vision",
                provider="hailo",
                model="face_detection.hef",
                capabilities=["detection"],
                health=HeadHealth(status="ok"),
            )
        ],
        capabilities=["audio_turn", "vision_observation", "move_head"],
        health=HeadHealth(status="ok", message="ready"),
    )

    encoded = json.dumps(manifest.to_dict())
    decoded = json.loads(encoded)
    restored = CapabilityManifest.from_dict(decoded)
    envelope = Envelope.wrap(channel="capabilities", payload=restored)

    assert restored.to_dict() == decoded
    assert envelope.payload["kind"] == "capability_manifest"
    assert envelope.payload["trace_id"] == "trace-001"
    assert envelope.payload["devices"][0]["limits"][0]["max_value"] == 30
    assert envelope.payload["backends"][0]["capabilities"] == ["detection"]


def test_head_health_accepts_missing_payload() -> None:
    from eibrain.protocol.capabilities import HeadHealth

    assert HeadHealth.from_dict(None).status == "unknown"
