from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import threading
from typing import Any, Iterator
from urllib import request
from urllib.error import HTTPError

import pytest

from eihead.monitoring import build_realtime_vision_payload
from eihead.monitoring.web import create_handler, create_server
from eihead.protocol import RealtimeVisionObservation, VisionObservation


class FakeMonitorApp:
    def __init__(self, *, health_payload: dict[str, Any] | None = None, fail_status: bool = False) -> None:
        self.health_payload = health_payload
        self.fail_status = fail_status

    def status(self) -> dict[str, Any]:
        if self.fail_status:
            raise RuntimeError("status boom")
        return {
            "ok": True,
            "status": "ok",
            "runtime": "eihead",
            "node_id": "honjia-test",
            "overall_status": "online",
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "schema": "eihead.status_snapshot.v1",
            "node_id": "honjia-test",
            "summary": {"online": 2, "degraded": 0, "offline": 0, "total": 2},
            "capabilities": {
                "camera": {"status": "online"},
                "neck": {"status": "online", "limits": {"yaw_deg": [0, 180], "tilt_deg": None}},
            },
        }

    def health(self) -> dict[str, Any]:
        return self.health_payload or {"ok": True, "status": "ok", "node_id": "honjia-test"}


def _realtime_observation() -> RealtimeVisionObservation:
    return RealtimeVisionObservation(
        ts=12.0,
        source="eihead.honjia.eye.realtime",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_001_000,
        trace_id="trace-eye-live",
        stream_id="front-main",
        camera_id="front",
        status="tracking",
        frame_id="frame-live-1",
        width=1280,
        height=720,
        fps=30.0,
        latency_ms=38.5,
        payload={"simulated": True},
        detections=[{"label": "person", "score": 0.95}],
        tracked_target={"label": "person", "center_x": 0.51},
        stream={"transport": "simulated", "connected": True},
        health={"dropped_frames": 0},
    )


class RealtimeVisionMethodApp(FakeMonitorApp):
    def vision_realtime(self) -> RealtimeVisionObservation:
        return _realtime_observation()


class EmptyRealtimeVisionMethodApp(FakeMonitorApp):
    def vision_realtime(self) -> None:
        return None


class PlaceholderRealtimeVisionMethodApp(FakeMonitorApp):
    def vision_realtime(self) -> dict[str, object]:
        return {
            "schema": "eihead.eye.realtime_status.v1",
            "mode": "realtime_stream",
            "status": "not_wired",
            "not_wired": True,
            "placeholder": True,
            "message": "detector not wired",
        }


class CompatStaticVisionMethodApp(FakeMonitorApp):
    def vision_realtime(self) -> VisionObservation:
        return VisionObservation(
            ts=13.0,
            source="eihead.honjia.eye.compat",
            frame_id="still-1",
            detections=[{"label": "person", "score": 0.8}],
        )


class RecentMethodApp(FakeMonitorApp):
    def recent_actions(self) -> list[dict[str, Any]]:
        return [
            {"action_id": "a1", "type": "speak", "status": "accepted"},
            {"action_id": "a2", "type": "move_head", "status": "skipped"},
        ]


@dataclass(slots=True)
class ActionRecord:
    action_id: str
    action_type: str
    status: str


class ActionLogAttributeApp(FakeMonitorApp):
    action_log = [ActionRecord("a3", "capture_frame", "accepted")]


class FallbackHealthApp(FakeMonitorApp):
    health = None


@contextmanager
def running_server(app: Any, **kwargs: Any) -> Iterator[tuple[str, object, threading.Thread]]:
    server = create_server(app, host="127.0.0.1", port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", server, thread
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_root_returns_simple_html_with_not_wired_recent_actions() -> None:
    app = FakeMonitorApp()

    with running_server(app, clock=lambda: 123.0) as (base_url, _server, _thread):
        status_code, headers, body = read_text(f"{base_url}/")

    assert status_code == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "eihead native monitor" in body
    assert "honjia-test" in body
    assert "/api/status" in body
    assert "/api/capabilities" in body
    assert "/api/actions/recent" in body
    assert "not_wired" in body


def test_health_status_and_capabilities_return_json_objects() -> None:
    app = FakeMonitorApp()

    with running_server(app) as (base_url, _server, _thread):
        health_code, health_headers, health_payload = read_json(f"{base_url}/health")
        status_code, _, status_payload = read_json(f"{base_url}/api/status")
        capabilities_code, _, capabilities_payload = read_json(f"{base_url}/api/capabilities?source=test")

    assert health_code == 200
    assert health_headers["Content-Type"].startswith("application/json")
    assert health_payload == {"ok": True, "status": "ok", "node_id": "honjia-test"}
    assert status_code == 200
    assert status_payload["runtime"] == "eihead"
    assert capabilities_code == 200
    assert capabilities_payload["capabilities"]["neck"]["limits"]["tilt_deg"] is None


def test_realtime_vision_helper_standardizes_observation_payload() -> None:
    payload = build_realtime_vision_payload(
        _realtime_observation(),
        timestamp=321.0,
        source="vision_realtime",
    )

    assert payload["schema"] == "eihead.monitor.vision_realtime.v1"
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["source"] == "vision_realtime"
    assert payload["channel"] == "eye.realtime"
    assert payload["aliases"] == ["vision.realtime"]
    assert payload["primary_mode"] == "realtime"
    assert payload["compat_static"] == {"mode": "compat/static", "primary": False}
    assert payload["captured_at_ts"] == 321.0
    assert payload["observation"]["kind"] == "realtime_vision_observation"
    assert payload["observation"]["mode"] == "realtime"
    assert payload["observation"]["primary_mode"] is True


def test_realtime_vision_api_and_html_render_wired_payload() -> None:
    app = RealtimeVisionMethodApp()

    with running_server(app, clock=lambda: 654.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")
        alias_code, _, alias_payload = read_json(f"{base_url}/api/eye/realtime")
        html_code, html_headers, body = read_text(f"{base_url}/")

    assert status_code == 200
    assert payload["schema"] == "eihead.monitor.vision_realtime.v1"
    assert payload["status"] == "wired"
    assert payload["channel"] == "eye.realtime"
    assert payload["aliases"] == ["vision.realtime"]
    assert payload["primary_mode"] == "realtime"
    assert payload["compat_static"]["mode"] == "compat/static"
    assert payload["observation"]["stream_id"] == "front-main"
    assert alias_code == 200
    assert alias_payload["observation"] == payload["observation"]
    assert html_code == 200
    assert html_headers["Content-Type"].startswith("text/html")
    assert "Realtime Vision" in body
    assert "/api/vision/realtime" in body
    assert "eye.realtime" in body


def test_realtime_vision_api_reports_not_wired_without_runtime_hook() -> None:
    app = FakeMonitorApp()

    with running_server(app, clock=lambda: 987.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["schema"] == "eihead.monitor.vision_realtime.v1"
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["source"] is None
    assert payload["captured_at_ts"] == 987.0
    assert payload["primary_mode"] == "realtime"
    assert payload["compat_static"] == {"mode": "compat/static", "primary": False}
    assert payload["observation"] is None
    assert "eye.realtime" in payload["message"]


def test_realtime_vision_api_reports_not_wired_when_runtime_hook_has_no_payload() -> None:
    app = EmptyRealtimeVisionMethodApp()

    with running_server(app, clock=lambda: 988.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["source"] == "vision_realtime"
    assert payload["observation"] is None


def test_realtime_vision_api_does_not_promote_placeholder_payload_to_wired() -> None:
    app = PlaceholderRealtimeVisionMethodApp()

    with running_server(app, clock=lambda: 989.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["source"] == "vision_realtime"
    assert payload["observation"]["status"] == "not_wired"
    assert payload["observation"]["placeholder"] is True
    assert "not ready" in payload["message"]


def test_realtime_vision_api_does_not_promote_static_vision_observation_to_wired() -> None:
    app = CompatStaticVisionMethodApp()

    with running_server(app, clock=lambda: 990.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "compat_static"
    assert payload["wired"] is False
    assert payload["source"] == "vision_realtime"
    assert payload["observation"]["kind"] == "vision_observation"
    assert payload["observation"]["mode"] == "compat/static"
    assert "compat/static" in payload["message"]


def test_recent_actions_uses_runtime_recent_actions_method() -> None:
    app = RecentMethodApp()

    with running_server(app, clock=lambda: 456.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/actions/recent")

    assert status_code == 200
    assert payload["schema"] == "eihead.monitor.recent_actions.v1"
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["source"] == "recent_actions"
    assert payload["captured_at_ts"] == 456.0
    assert payload["count"] == 2
    assert payload["actions"][0]["action_id"] == "a1"


def test_recent_actions_alias_and_action_log_attribute_are_supported() -> None:
    app = ActionLogAttributeApp()

    with running_server(app) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/recent-actions")

    assert status_code == 200
    assert payload["status"] == "wired"
    assert payload["source"] == "action_log"
    assert payload["count"] == 1
    assert payload["actions"] == [
        {"action_id": "a3", "action_type": "capture_frame", "status": "accepted"}
    ]


def test_recent_actions_reports_not_wired_when_runtime_exposes_no_log() -> None:
    app = FakeMonitorApp()

    with running_server(app, clock=lambda: 789.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/actions/recent")

    assert status_code == 200
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["source"] is None
    assert payload["captured_at_ts"] == 789.0
    assert payload["actions"] == []
    assert "does not expose" in payload["message"]


def test_health_falls_back_to_status_and_can_return_503() -> None:
    app = FallbackHealthApp(health_payload=None)

    with running_server(app, clock=lambda: 111.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/health")

    assert status_code == 200
    assert payload["source"] == "status"
    assert payload["checked_at_ts"] == 111.0

    unhealthy_app = FakeMonitorApp(health_payload={"ok": False, "status": "offline", "node_id": "honjia-test"})
    with running_server(unhealthy_app) as (base_url, _server, _thread):
        error_code, error_payload = read_error_json(f"{base_url}/health")

    assert error_code == 503
    assert error_payload["ok"] is False
    assert error_payload["status"] == "offline"


def test_unknown_path_wrong_method_and_runtime_errors_are_json() -> None:
    with running_server(FakeMonitorApp()) as (base_url, _server, _thread):
        not_found_code, not_found_payload = read_error_json(f"{base_url}/missing")
        method_code, method_payload = read_error_json(f"{base_url}/api/status", method="POST")

    assert not_found_code == 404
    assert not_found_payload["error"]["code"] == "not_found"
    assert method_code == 405
    assert method_payload["error"]["code"] == "method_not_allowed"

    with running_server(FakeMonitorApp(fail_status=True)) as (base_url, _server, _thread):
        runtime_code, runtime_payload = read_error_json(f"{base_url}/api/status")

    assert runtime_code == 500
    assert runtime_payload["error"]["code"] == "internal_error"
    assert runtime_payload["error"]["details"]["exception"] == "RuntimeError"


def test_server_wrapper_shutdown_stops_serve_forever_cleanly() -> None:
    server = create_server(FakeMonitorApp(), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    thread.start()
    read_json(f"http://{server.server_address[0]}:{server.server_address[1]}/health")
    server.shutdown()
    thread.join(timeout=2.0)
    server.server_close()

    assert not thread.is_alive()


def test_handler_factory_validates_monitor_contract() -> None:
    with pytest.raises(TypeError, match="missing required callables"):
        create_handler(object())


def read_json(url: str) -> tuple[int, Any, Any]:
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=2.0) as response:
        return response.status, response.headers, json.loads(response.read().decode("utf-8"))


def read_text(url: str) -> tuple[int, Any, str]:
    req = request.Request(url, headers={"Accept": "text/html"})
    with request.urlopen(req, timeout=2.0) as response:
        return response.status, response.headers, response.read().decode("utf-8")


def read_error_json(url: str, *, method: str = "GET") -> tuple[int, dict[str, Any]]:
    req = request.Request(url, method=method, headers={"Accept": "application/json"})
    with pytest.raises(HTTPError) as exc:
        request.urlopen(req, timeout=2.0)
    return exc.value.code, json.loads(exc.value.read().decode("utf-8"))
