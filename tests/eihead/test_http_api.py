from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
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


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "eiprotocol"


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


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


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
        _, _, capabilities_payload = read_json(f"{base_url}/capabilities")
        health_code, _, health_payload = read_json(f"{base_url}/health")

    assert status_code == 200
    assert headers["Content-Type"].startswith("application/json")
    assert status_payload["runtime"] == "eihead"
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

    with running_server(app) as (base_url, _server, _thread):
        client = HeadClient(base_url, trace_id="trace-runtime-event")
        payload = post_head_action_event(client, load_fixture("head_action_request.json"))

    assert payload["ok"] is True
    assert payload["accepted"] is True
    assert payload["processed"] is True
    assert payload["status"] == "accepted"
    assert payload["trace_id"] == "trc_voice_001"
    assert payload["action_outcome"]["action_type"] == "move_head"
    assert isinstance(body_runtime.dispatched[0], MoveHeadAction)
    assert body_runtime.dispatched[0].target_angle == 92
    assert body_runtime.dispatched[0].target_name == "neck.pan"


def test_post_events_returns_not_wired_success_when_handler_is_absent() -> None:
    app = FakeHeadApp()

    with running_server(app) as (base_url, _server, _thread):
        status_code, _, payload = read_json(
            f"{base_url}/events",
            method="POST",
            json_body={"event": {"type": "wakeword.detected"}, "trace_id": "trace-event-2"},
        )

    assert status_code == 200
    assert payload == {
        "ok": True,
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


def test_health_falls_back_to_status_when_health_method_is_absent() -> None:
    app = FallbackHealthApp()

    with running_server(app, clock=lambda: 123.456) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/health")

    assert status_code == 200
    assert payload["ok"] is True
    assert payload["source"] == "status"
    assert payload["checked_at_ts"] == 123.456
    assert payload["node_id"] == "honjia-test"


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
