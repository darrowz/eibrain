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


def test_eiprotocol_capability_manifest_round_trips_grouped_modalities_and_health() -> None:
    from eiprotocol.models import Capability, CapabilityManifest, DeviceStatus

    manifest = CapabilityManifest(
        manifest_id="cap-honjia-v011",
        manifest_version="0.1.1",
        device={"nodeId": "honjia", "nodeRole": "eihead"},
        runtime={"protocolVersion": "head.v1", "runtimeId": "eihead.runtime"},
        transports={"websocket": {"path": "/events"}},
        modalities={
            "audio": {
                "available": True,
                "microphones": ["mic.main"],
                "speakers": ["speaker.main"],
                "asr": ["asr.sherpa"],
                "tts": ["tts.minimax"],
            },
            "vision": {"available": True, "cameras": ["camera.main"], "backends": ["vision.hailo"]},
            "actuation": {"available": True, "neck": ["neck.pan"]},
            "embedding": {"available": True, "backends": ["embedding.bge"]},
        },
        capabilities=[
            Capability(capability_id="camera.main", kind="camera", device_path="/dev/video0", status="online"),
            Capability(capability_id="mic.main", kind="audio_input", device_path="plughw:U4K,0", status="online"),
            Capability(capability_id="speaker.main", kind="audio_output", status="online"),
            Capability(
                capability_id="neck.pan",
                kind="actuator",
                actions=["move_head"],
                status="online",
                limits={"axis": "pan", "minAngle": 0, "maxAngle": 180},
            ),
        ],
        backends=[
            Capability(capability_id="asr.sherpa", kind="asr", provider="sherpa-onnx", status="online"),
            Capability(capability_id="tts.minimax", kind="tts", provider="minimax", status="online"),
            Capability(capability_id="vision.hailo", kind="vision", provider="hailo", status="online"),
            Capability(capability_id="embedding.bge", kind="embedding", provider="bge", status="online"),
        ],
        health=DeviceStatus(status="ok", message="ready", checked_at_ms=1_777_854_660_000, metrics={"cpuTempC": 42.5}),
        metadata={"fixture": "grouped capability manifest"},
    )

    restored = CapabilityManifest.from_content(manifest.to_content())

    assert restored == manifest
    assert restored.modalities["audio"]["microphones"] == ["mic.main"]
    assert restored.modalities["audio"]["tts"] == ["tts.minimax"]
    assert restored.modalities["vision"]["cameras"] == ["camera.main"]
    assert restored.modalities["actuation"]["neck"] == ["neck.pan"]
    assert restored.modalities["embedding"]["backends"] == ["embedding.bge"]
    assert restored.health.metrics == {"cpuTempC": 42.5}
