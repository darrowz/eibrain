from __future__ import annotations

import pytest

from eibrain.infra.head_registry import HeadRegistry


class ManifestObject:
    def to_dict(self):
        return {
            "node_id": "honjia",
            "health": {"status": "ok"},
            "capabilities": ["audio_turn", "move_head"],
            "devices": [
                {
                    "device_id": "camera.main",
                    "kind": "camera",
                    "path": "/dev/video0",
                    "enabled": True,
                    "health": {"status": "ok"},
                }
            ],
            "backends": [
                {
                    "backend_id": "vision.hailo",
                    "kind": "vision_backend",
                    "provider": "hailo",
                    "model": "face_detection.hef",
                    "health": {"status": "ok"},
                }
            ],
        }


def test_update_manifest_indexes_devices_backends_and_declared_capabilities() -> None:
    registry = HeadRegistry(time_fn=lambda: 123.5)

    node = registry.update_manifest(ManifestObject())

    assert node["node_id"] == "honjia"
    assert node["last_seen_ts"] == 123.5
    assert node["overall_status"] == "ok"
    assert registry.get_capability("audio_turn") == {
        "id": "audio_turn",
        "name": "audio_turn",
        "kind": "audio_turn",
        "category": "capability",
        "source": "manifest",
        "online": True,
    }
    assert registry.get_capability("camera.main")["path"] == "/dev/video0"
    assert registry.get_capability("camera")["id"] == "camera.main"
    assert registry.get_capability("vision.hailo")["model"] == "face_detection.hef"
    assert registry.summary()["online_count"] == 1


def test_update_status_merges_capability_metrics_and_errors() -> None:
    registry = HeadRegistry(time_fn=lambda: 10.0)
    registry.update_manifest(
        {
            "node_id": "honjia",
            "devices": [{"device_id": "camera.main", "kind": "camera", "health": {"status": "ok"}}],
        }
    )

    node = registry.update_status(
        {
            "node_id": "honjia",
            "overall_status": "degraded",
            "capabilities": {"camera.main": {"status": "degraded", "latency_ms": 42.0}},
            "errors": ["camera fps low"],
        },
        observed_at=11.0,
    )

    capability = registry.get_capability("camera.main")
    assert node["overall_status"] == "degraded"
    assert node["last_seen_ts"] == 11.0
    assert capability["latency_ms"] == 42.0
    assert capability["status"] == "degraded"
    assert node["errors"][0]["message"] == "camera fps low"
    assert registry.summary()["error_count"] == 1
    assert registry.summary()["degraded_count"] == 1


def test_status_without_overall_status_preserves_previous_health() -> None:
    registry = HeadRegistry()
    registry.update_manifest({"node_id": "honjia", "health": {"status": "ok"}})

    node = registry.update_status({"node_id": "honjia", "capabilities": {"ear": {"audio_level": 0.12}}})

    assert node["overall_status"] == "ok"
    assert registry.get_capability("ear")["audio_level"] == 0.12


def test_update_from_client_uses_head_client_like_fake_without_network() -> None:
    calls: list[str] = []

    class FakeClient:
        def get_capabilities(self):
            calls.append("capabilities")
            return {
                "manifest": {
                    "node_id": "honjia",
                    "health": {"status": "online"},
                    "capabilities": {"tts": {"provider": "minimax", "model": "speech-02"}},
                }
            }

        def get_status(self):
            calls.append("status")
            return {
                "overall_status": "online",
                "capabilities": {"tts": {"latency_ms": 280.0, "status": "online"}},
            }

    registry = HeadRegistry(time_fn=lambda: 20.0)
    node = registry.update_from_client(FakeClient())

    assert calls == ["capabilities", "status"]
    assert node["node_id"] == "honjia"
    assert node["overall_status"] == "online"
    assert registry.get_capability("tts")["provider"] == "minimax"
    assert registry.get_capability("tts")["latency_ms"] == 280.0


def test_update_from_client_records_errors_without_raising() -> None:
    class FakeClientError(RuntimeError):
        def to_dict(self):
            return {"kind": "network_error", "message": "eihead unreachable"}

    class FakeClient:
        def get_capabilities(self):
            raise FakeClientError("eihead unreachable")

        def get_status(self):
            return {"node_id": "honjia", "overall_status": "degraded"}

    registry = HeadRegistry(time_fn=lambda: 30.0)
    node = registry.update_from_client(FakeClient())

    assert node["overall_status"] == "degraded"
    assert node["errors"] == [
        {
            "kind": "network_error",
            "message": "eihead unreachable",
            "operation": "get_capabilities",
            "source": "head_client",
            "ts": 30.0,
        }
    ]


def test_supports_multiple_nodes_and_defaults_to_honjia() -> None:
    registry = HeadRegistry()
    registry.update_manifest({"node_id": "honjia", "health": {"status": "ok"}})
    registry.update_manifest({"node_id": "bench-head", "health": {"status": "offline"}})

    assert registry.get_node()["node_id"] == "honjia"
    assert registry.get_node("bench-head")["overall_status"] == "offline"
    assert registry.summary()["node_count"] == 2
    assert set(registry.to_dict()["nodes"]) == {"honjia", "bench-head"}


def test_rejects_non_mapping_payloads() -> None:
    registry = HeadRegistry()

    with pytest.raises(TypeError):
        registry.update_manifest(["not", "a", "mapping"])

    with pytest.raises(TypeError):
        registry.update_status(object())
