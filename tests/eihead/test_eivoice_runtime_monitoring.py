from __future__ import annotations

from contextlib import contextmanager
import json
import threading
from typing import Any, Iterator
from urllib import request

from eihead.monitoring.eivoice_runtime import build_eivoice_runtime_panel
from eihead.monitoring.web import create_server


def test_empty_status_waits_for_runtime_state() -> None:
    panel = build_eivoice_runtime_panel({})

    assert panel["state"] == "waiting"
    assert panel["conversationState"] == "unknown"
    assert panel["health"] == "waiting"
    assert panel["warnings"] == ["runtime state is missing"]
    assert panel["droppedTotal"] == 0


def test_normal_four_queues_are_healthy_with_fill_ratios() -> None:
    panel = build_eivoice_runtime_panel(
        {
            "state": "running",
            "conversation_state": "Idle",
            "queues": {
                "opus_encode_queue": {"depth": 1, "capacity": 4, "policy": "drop_oldest"},
                "ws_send_queue": {"depth": 2, "capacity": 4, "policy": "drop_newest"},
                "opus_decode_queue": {"depth": 0, "capacity": 5},
                "audio_playback_queue": {"depth": 3, "capacity": 6},
            },
            "audio_frontend": {
                "aec": {"enabled": True, "available": True},
                "ns": {"enabled": True},
                "vad": {"enabled": True},
                "loopback": {"enabled": True},
            },
            "wakeword": {"enabled": True, "state": "armed"},
        }
    )

    assert panel["health"] == "healthy"
    assert panel["state"] == "running"
    assert panel["conversationState"] == "Idle"
    assert panel["queueSummary"] == {
        "count": 4,
        "totalDepth": 6,
        "totalCapacity": 19,
        "maxFillRatio": 0.5,
    }
    assert panel["queues"]["opus_encode_queue"]["fillRatio"] == 0.25
    assert panel["queues"]["ws_send_queue"]["fillRatio"] == 0.5
    assert panel["queues"]["opus_decode_queue"]["fillRatio"] == 0.0
    assert panel["queues"]["audio_playback_queue"]["fillRatio"] == 0.5
    assert panel["droppedTotal"] == 0
    assert panel["audioFrontend"]["aec"] == {"enabled": True, "available": True}
    assert panel["wakeword"] == {"enabled": True, "state": "armed"}
    assert panel["warnings"] == []


def test_drop_oldest_and_drop_newest_counts_degrade_health() -> None:
    panel = build_eivoice_runtime_panel(
        {
            "state": "running",
            "queues": {
                "opus_encode_queue": {
                    "depth": 4,
                    "capacity": 4,
                    "dropped_oldest": 2,
                    "dropped_newest": 1,
                }
            },
        }
    )

    queue = panel["queues"]["opus_encode_queue"]
    assert panel["health"] == "degraded"
    assert panel["droppedTotal"] == 3
    assert queue["droppedOldest"] == 2
    assert queue["droppedNewest"] == 1
    assert "queue drops detected: 3" in panel["warnings"]


def test_acoustic_frontend_aec_unavailable_degrades_health() -> None:
    panel = build_eivoice_runtime_panel(
        {
            "state": "running",
            "acousticFrontend": {
                "aec": {"enabled": True, "available": False},
                "ns": {"enabled": True, "available": True},
                "vad": {"enabled": True, "available": True},
                "loopback": {"enabled": False},
            },
        }
    )

    assert panel["health"] == "degraded"
    assert panel["audioFrontend"]["aec"] == {"enabled": True, "available": False}
    assert "AEC unavailable" in panel["warnings"]


def test_audio_frontend_readiness_boolean_false_degrades_health() -> None:
    from eihead.devices.audio import evaluate_audio_frontend_readiness

    readiness = evaluate_audio_frontend_readiness(
        capture_device="hw:U4K",
        loopback_device=None,
        supports_aec=False,
        supports_ns=False,
        supports_vad=True,
    )

    panel = build_eivoice_runtime_panel(
        {
            "state": "running",
            "audio_frontend": readiness.to_dict(),
        }
    )

    assert panel["health"] == "degraded"
    assert panel["audioFrontend"]["aec"] == {"enabled": False}
    assert "AEC unavailable" in panel["warnings"]


def test_runtime_status_without_audio_frontend_is_not_reported_healthy() -> None:
    from eihead.eivoice_runtime import EiVoiceRuntimeCore

    runtime = EiVoiceRuntimeCore()
    runtime.state_machine.wake_detected()
    runtime.wakeword_buffer.append(_audio_frame())

    panel = build_eivoice_runtime_panel(runtime.status())

    assert panel["state"] == "conversation"
    assert panel["conversationState"] == "conversation"
    assert panel["wakeword"]["depth"] == 1
    assert panel["health"] == "degraded"
    assert "audio frontend readiness is missing" in panel["warnings"]


def test_transport_status_is_exposed_and_degrades_on_reconnect_errors() -> None:
    panel = build_eivoice_runtime_panel(
        {
            "state": "conversation",
            "audio_frontend": {
                "aec": {"enabled": True, "available": True},
                "ns": {"enabled": True, "available": True},
                "vad": {"enabled": True, "available": True},
                "loopback": {"enabled": True, "available": True},
            },
            "transport": {
                "transport": "fake_websocket",
                "state": "reconnect_wait",
                "heartbeat": {"awaiting_pong": True, "timed_out": True, "latency_ms": 1250},
                "reconnect": {"attempt": 2, "backoff_s": 4.0, "ready": False, "reason": "heartbeat_timeout"},
                "last_error": {"kind": "TimeoutError", "message": "pong timeout", "context": "heartbeat"},
            },
        }
    )

    assert panel["transport"]["name"] == "fake_websocket"
    assert panel["transport"]["state"] == "reconnect_wait"
    assert panel["transport"]["heartbeat"]["timed_out"] is True
    assert panel["transport"]["reconnect"]["attempt"] == 2
    assert panel["health"] == "degraded"
    assert "transport reconnect_wait" in panel["warnings"]
    assert "transport error: TimeoutError heartbeat" in panel["warnings"]


class RuntimePanelApp:
    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "ok",
            "runtime": "eihead",
            "node_id": "honjia-test",
            "eivoice_runtime": {
                "state": "running",
                "conversation_state": "Conversation",
                "queues": {
                    "opus_encode_queue": {"depth": 1, "capacity": 2},
                    "ws_send_queue": {"depth": 0, "capacity": 2},
                    "opus_decode_queue": {"depth": 0, "capacity": 2},
                    "audio_playback_queue": {"depth": 0, "capacity": 2},
                },
                "transport": {
                    "transport": "fake_websocket",
                    "state": "connected",
                    "heartbeat": {"latency_ms": 24},
                    "reconnect": {"attempt": 0},
                },
            },
        }

    def capabilities(self) -> dict[str, Any]:
        return {"capabilities": {}}


@contextmanager
def running_server(app: Any) -> Iterator[str]:
    server = create_server(app, host="127.0.0.1", port=0, clock=lambda: 123.0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def read_text(url: str) -> str:
    with request.urlopen(url, timeout=2.0) as response:
        return response.read().decode("utf-8")


def read_json(url: str) -> dict[str, Any]:
    return json.loads(read_text(url))


def _audio_frame():
    from eihead.eivoice_runtime import AudioFrame

    return AudioFrame(pcm=b"x")


def test_web_exposes_eivoice_runtime_panel() -> None:
    with running_server(RuntimePanelApp()) as base_url:
        body = read_text(f"{base_url}/")
        payload = read_json(f"{base_url}/api/eivoice/runtime")

    assert "EIVoice Runtime" in body
    assert "Transport state" in body
    assert "fake_websocket / connected" in body
    assert "eivoiceRuntime" in body
    assert payload["eivoiceRuntime"]["state"] == "running"
    assert payload["eivoiceRuntime"]["conversationState"] == "Conversation"
    assert payload["eivoiceRuntime"]["transport"]["name"] == "fake_websocket"
