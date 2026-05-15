from __future__ import annotations

from contextlib import contextmanager
import json
import threading
from typing import Any, Iterator
from urllib import request
from urllib.error import HTTPError

import pytest

from eibrain.infra.head_events import post_head_action_event
from eibrain.infra.head_client import HeadClient
from eihead.protocol import ActionExecuted, MoveHeadAction
from eihead.runtime.app import HeadRuntimeApp
from eihead.runtime.http_api import create_handler, create_server


class FakeHeadApp:
    def __init__(self, *, health_payload: dict[str, Any] | None = None, fail_status: bool = False) -> None:
        self.health_payload = health_payload
        self.fail_status = fail_status
        self.actions: list[tuple[dict[str, Any], str | None]] = []

    def status(self) -> dict[str, Any]:
        if self.fail_status:
            raise RuntimeError("status boom")
        return {
            "ok": True,
            "status": "ok",
            "runtime": "eihead",
            "node_id": "honjia-test",
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "schema": "eihead.capability_manifest.v1",
            "node_id": "honjia-test",
            "capabilities": {
                "camera": {"status": "online"},
                "neck": {"status": "online", "limits": {"axis": ["yaw"]}},
            },
        }

    def health(self) -> dict[str, Any]:
        return self.health_payload or {"ok": True, "status": "ok", "node_id": "honjia-test"}

    def handle_action(self, action: dict[str, Any], *, trace_id: str | None = None) -> dict[str, Any]:
        self.actions.append((dict(action), trace_id))
        return {
            "ok": True,
            "accepted": True,
            "trace_id": trace_id,
            "action_type": action.get("type"),
        }


class FallbackHealthApp(FakeHeadApp):
    health = None


class DegradedStatusFallbackApp(FallbackHealthApp):
    def status(self) -> dict[str, Any]:
        return {
            "status": "degraded",
            "runtime": "eihead",
            "node_id": "honjia-test",
            "checks": {"native_provider_boundaries": "degraded"},
        }


class EventHeadApp(FakeHeadApp):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[dict[str, Any], str | None]] = []

    def handle_event(self, event: dict[str, Any], *, trace_id: str | None = None) -> dict[str, Any]:
        self.events.append((dict(event), trace_id))
        return {
            "ok": True,
            "accepted": True,
            "trace_id": trace_id,
            "event_type": event.get("type"),
        }


class RecordingBodyRuntime:
    def __init__(self) -> None:
        self.dispatched: list[object] = []

    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-test"}

    def dispatch_actions(self, actions: list[object]) -> list[object]:
        self.dispatched.extend(actions)
        action = actions[0]
        if isinstance(action, MoveHeadAction):
            return [
                ActionExecuted(
                    ts=action.ts,
                    source="neck.motor",
                    status="ok",
                    action_kind=action.kind,
                    details={"target_angle": action.target_angle},
                )
            ]
        return []


class FailingSnapshotBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        raise RuntimeError("snapshot boom")


class SnapshotRealtimeBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "organs": {
                "eye": {
                    "realtime_vision": {
                        "kind": "realtime_vision_observation",
                        "mode": "realtime_stream",
                        "primary_mode": True,
                        "stream_id": "front-main",
                        "status": "tracking",
                        "frame_id": "live-1",
                    }
                }
            },
        }


def wired_native_providers() -> dict[str, dict[str, str]]:
    return {
        "eye": {"status": "wired"},
        "ear": {"status": "wired"},
        "mouth": {"status": "wired"},
        "neck": {"status": "wired"},
    }


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


def test_status_capabilities_and_health_return_json_objects() -> None:
    app = FakeHeadApp()

    with running_server(app) as (base_url, _server, _thread):
        status_code, headers, status_payload = read_json(f"{base_url}/status?source=test")
        status_json_code, _, status_json_payload = read_json(f"{base_url}/status.json")
        api_status_code, _, api_status_payload = read_json(f"{base_url}/api/status")
        _, _, capabilities_payload = read_json(f"{base_url}/capabilities")
        health_code, _, health_payload = read_json(f"{base_url}/health")

    assert status_code == 200
    assert status_json_code == 200
    assert api_status_code == 200
    assert headers["Content-Type"].startswith("application/json")
    assert status_payload["runtime"] == "eihead"
    assert status_json_payload["runtime"] == "eihead"
    assert api_status_payload["runtime"] == "eihead"
    assert capabilities_payload["capabilities"]["camera"]["status"] == "online"
    assert health_code == 200
    assert health_payload == {"ok": True, "status": "ok", "node_id": "honjia-test"}


def test_head_client_posts_actions_to_api_contract() -> None:
    app = FakeHeadApp()

    with running_server(app) as (base_url, _server, _thread):
        client = HeadClient(base_url, trace_id="trace-123")
        payload = client.speak("你好鸿途", voice="minimax")

    assert payload == {
        "ok": True,
        "accepted": True,
        "trace_id": "trace-123",
        "action_type": "speak",
    }
    assert app.actions == [
        (
            {"type": "speak", "text": "你好鸿途", "voice": "minimax"},
            "trace-123",
        )
    ]


def test_post_events_calls_runtime_event_handler() -> None:
    app = EventHeadApp()

    with running_server(app) as (base_url, _server, _thread):
        status_code, _, payload = read_json(
            f"{base_url}/events",
            method="POST",
            json_body={"event": {"type": "gesture.detected", "name": "wave"}, "trace_id": "trace-event-1"},
        )

    assert status_code == 200
    assert payload == {
        "ok": True,
        "accepted": True,
        "trace_id": "trace-event-1",
        "event_type": "gesture.detected",
    }
    assert app.events == [
        (
            {"type": "gesture.detected", "name": "wave"},
            "trace-event-1",
        )
    ]


def test_head_client_post_event_routes_eiprotocol_action_through_runtime_app() -> None:
    body_runtime = RecordingBodyRuntime()
    app = HeadRuntimeApp(body_runtime=body_runtime)
    head_action_request = {
        "specVersion": "eiprotocol/0.1",
        "id": "evt_head_action_001",
        "type": "action",
        "name": "ei.action.request",
        "time": "2026-05-04T10:32:01.700+08:00",
        "sequence": 5,
        "requestId": "req_action_001",
        "sessionId": "ses_honjia_001",
        "roundId": "rnd_voice_001",
        "causationId": "evt_vision_frame_001",
        "traceId": "trc_voice_001",
        "source": {
            "domain": "eibrain",
            "instanceId": "honxin",
            "botId": "bot_hongtu",
            "metadata": {},
        },
        "priority": "high",
        "ttlMs": 3000,
        "mode": {
            "conversationState": "responding",
            "interactionMode": "free",
        },
        "content": {
            "actionId": "act_move_head_001",
            "actionType": "move_head",
            "target": "neck.pan",
            "params": {
                "targetAngle": 92,
                "durationMs": 240,
                "reason": "center tracked person",
            },
            "riskLevel": "L1",
            "timeline": [
                {"atMs": 0, "operation": "move"},
                {"atMs": 240, "operation": "hold"},
            ],
            "requiresPolicy": False,
            "metadata": {"planner": "active_attention"},
            "idempotencyKey": "act_move_head_001_once",
        },
        "policy": {
            "decision": "not_required",
            "riskLevel": "L1",
            "decisionId": "",
            "requiredAck": False,
            "reason": "",
            "expiresAt": "",
            "extensions": {},
        },
        "extensions": {},
        "target": {
            "domain": "eihead",
            "instanceId": "honjia",
            "metadata": {},
        },
    }

    with running_server(app) as (base_url, _server, _thread):
        client = HeadClient(base_url, trace_id="trace-runtime-event")
        payload = post_head_action_event(client, head_action_request)

    assert payload["ok"] is True
    assert payload["accepted"] is True
    assert payload["processed"] is True
    assert payload["status"] == "accepted"
    assert payload["trace_id"] == "trc_voice_001"
    assert payload["action_outcome"]["action_type"] == "move_head"
    assert isinstance(body_runtime.dispatched[0], MoveHeadAction)
    assert body_runtime.dispatched[0].target_angle == 92
    assert body_runtime.dispatched[0].target_name == "neck.pan"


def test_post_events_returns_503_when_handler_is_absent() -> None:
    app = FakeHeadApp()

    with running_server(app) as (base_url, _server, _thread):
        status_code, payload = read_error_json(
            f"{base_url}/events",
            method="POST",
            json_body={"event": {"type": "wakeword.detected"}, "trace_id": "trace-event-2"},
        )

    assert status_code == 503
    assert payload["ok"] is False
    assert payload["error"]["code"] == "event_handler_not_wired"
    assert payload["error"]["details"] == {
        "accepted": False,
        "status": "not_wired",
        "reason": "runtime_app_handle_event_unavailable",
        "trace_id": "trace-event-2",
    }


def test_post_actions_rejects_invalid_request_body_with_json_error() -> None:
    app = FakeHeadApp()

    with running_server(app) as (base_url, _server, _thread):
        status_code, payload = read_error_json(
            f"{base_url}/actions",
            method="POST",
            data=b'["not", "an", "object"]',
        )

    assert status_code == 400
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_json_object"
    assert payload["error"]["status_code"] == 400
    assert app.actions == []


def test_post_actions_rejects_missing_action_with_json_error() -> None:
    app = FakeHeadApp()

    with running_server(app) as (base_url, _server, _thread):
        status_code, payload = read_error_json(f"{base_url}/actions", method="POST", json_body={"trace_id": "x"})

    assert status_code == 400
    assert payload["error"]["code"] == "invalid_action"
    assert app.actions == []


@pytest.mark.parametrize(
    ("body", "error_code"),
    [
        ({"trace_id": "x"}, "invalid_event"),
        ({"event": ["not", "an", "object"]}, "invalid_event"),
        ({"event": {"type": "wakeword.detected"}, "trace_id": 123}, "invalid_trace_id"),
    ],
)
def test_post_events_rejects_invalid_event_request_with_json_error(
    body: dict[str, Any],
    error_code: str,
) -> None:
    app = EventHeadApp()

    with running_server(app) as (base_url, _server, _thread):
        status_code, payload = read_error_json(f"{base_url}/events", method="POST", json_body=body)

    assert status_code == 400
    assert payload["ok"] is False
    assert payload["error"]["code"] == error_code
    assert app.events == []


def test_unknown_path_and_wrong_method_return_json_errors() -> None:
    app = FakeHeadApp()

    with running_server(app) as (base_url, _server, _thread):
        not_found_status, not_found_payload = read_error_json(f"{base_url}/missing")
        method_status, method_payload = read_error_json(f"{base_url}/actions")

    assert not_found_status == 404
    assert not_found_payload["error"]["code"] == "not_found"
    assert method_status == 405
    assert method_payload["error"]["code"] == "method_not_allowed"


def test_runtime_exception_returns_internal_error_json() -> None:
    app = FakeHeadApp(fail_status=True)

    with running_server(app) as (base_url, _server, _thread):
        status_code, payload = read_error_json(f"{base_url}/status")

    assert status_code == 500
    assert payload["error"]["code"] == "internal_error"
    assert payload["error"]["details"]["exception"] == "RuntimeError"


def test_health_can_return_service_unavailable_json() -> None:
    app = FakeHeadApp(health_payload={"ok": False, "status": "offline", "node_id": "honjia-test"})

    with running_server(app) as (base_url, _server, _thread):
        status_code, payload = read_error_json(f"{base_url}/health")

    assert status_code == 503
    assert payload["ok"] is False
    assert payload["status"] == "offline"
    assert payload["node_id"] == "honjia-test"


def test_health_treats_degraded_payload_as_service_unavailable_without_ok_flag() -> None:
    app = FakeHeadApp(health_payload={"status": "degraded", "node_id": "honjia-test"})

    with running_server(app) as (base_url, _server, _thread):
        status_code, payload = read_error_json(f"{base_url}/health")

    assert status_code == 503
    assert payload["status"] == "degraded"
    assert payload["node_id"] == "honjia-test"


def test_health_fallback_treats_degraded_status_as_service_unavailable() -> None:
    app = DegradedStatusFallbackApp()

    with running_server(app, clock=lambda: 456.0) as (base_url, _server, _thread):
        status_code, payload = read_error_json(f"{base_url}/health")

    assert status_code == 503
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["source"] == "status"
    assert payload["checked_at_ts"] == 456.0


def test_health_reports_blocked_when_runtime_body_snapshot_fails() -> None:
    app = HeadRuntimeApp(
        body_runtime=FailingSnapshotBodyRuntime(),
        config_path="config/test.yaml",
        delegate_name="eihead.native",
        native_providers=wired_native_providers(),
        neck_servo_adapter=object(),
    )

    with running_server(app) as (base_url, _server, _thread):
        status_code, payload = read_error_json(f"{base_url}/health")

    assert status_code == 503
    assert payload["ok"] is False
    assert payload["status"] == "blocked"
    assert payload["checks"]["body_runtime_snapshot"] == "blocked"


def test_health_falls_back_to_status_when_health_method_is_absent() -> None:
    app = FallbackHealthApp()

    with running_server(app, clock=lambda: 123.456) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/health")

    assert status_code == 200
    assert payload["ok"] is True
    assert payload["source"] == "status"
    assert payload["checked_at_ts"] == 123.456
    assert payload["node_id"] == "honjia-test"


def test_realtime_vision_api_uses_runtime_snapshot_eye_payload() -> None:
    app = HeadRuntimeApp(body_runtime=SnapshotRealtimeBodyRuntime(), config_path="config/test.yaml")

    with running_server(app, clock=lambda: 789.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["schema"] == "eihead.monitor.vision_realtime.v1"
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["source"] == "vision_realtime"
    assert payload["observation"]["stream_id"] == "front-main"
    assert payload["frame_id"] == "live-1"


def test_realtime_vision_api_reports_not_wired_without_realtime_payload() -> None:
    app = HeadRuntimeApp(body_runtime=RecordingBodyRuntime(), config_path="config/test.yaml")

    with running_server(app, clock=lambda: 790.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["not_wired"] is True
    assert payload["observation"] is None


def test_server_wrapper_shutdown_stops_serve_forever_cleanly() -> None:
    app = FakeHeadApp()
    server = create_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    thread.start()
    read_json(f"http://{server.server_address[0]}:{server.server_address[1]}/health")
    server.shutdown()
    thread.join(timeout=2.0)
    server.server_close()

    assert not thread.is_alive()


def test_handler_factory_validates_runtime_contract() -> None:
    with pytest.raises(TypeError, match="missing required callables"):
        create_handler(object())


def read_json(url: str, *, method: str = "GET", json_body: dict[str, Any] | None = None) -> tuple[int, Any, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    with request.urlopen(req, timeout=2.0) as response:
        return response.status, response.headers, json.loads(response.read().decode("utf-8"))


def read_error_json(
    url: str,
    *,
    method: str = "GET",
    json_body: dict[str, Any] | None = None,
    data: bytes | None = None,
) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif data is not None:
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, method=method, headers=headers)
    with pytest.raises(HTTPError) as exc:
        request.urlopen(req, timeout=2.0)
    return exc.value.code, json.loads(exc.value.read().decode("utf-8"))
