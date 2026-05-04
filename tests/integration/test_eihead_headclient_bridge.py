from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from eibrain.infra.head_client import HeadClient


JsonObject = dict[str, Any]


class FakeHeadRuntimeApp:
    """Small no-hardware runtime used only for the eihead bridge contract."""

    def __init__(self) -> None:
        self.status_payload: JsonObject = {
            "ok": True,
            "node_id": "honjia",
            "runtime": "fake-eihead",
            "contract_test": True,
            "hardware": "not_touched",
        }
        self.capabilities_payload: JsonObject = {
            "node_id": "honjia",
            "capabilities": {
                "microphone": {"online": True, "device": "fake-u4k"},
                "speaker": {"online": True, "provider": "fake-tts"},
                "camera": {"online": True, "device": "fake-video0"},
                "neck": {"online": True, "axes": ["yaw"]},
            },
            "contract_test": True,
        }
        self.requests: list[JsonObject] = []
        self.action_requests: list[JsonObject] = []

    def record_request(self, *, method: str, path: str, trace_header: str | None) -> None:
        self.requests.append(
            {
                "method": method,
                "path": path,
                "trace_header": trace_header,
            }
        )

    def accept_action(self, body: JsonObject, *, trace_header: str | None) -> JsonObject:
        action = body.get("action")
        action_type = action.get("type") if isinstance(action, dict) else None
        self.action_requests.append({"body": body, "trace_header": trace_header})
        return {
            "accepted": True,
            "action_id": f"fake-action-{len(self.action_requests)}",
            "action_type": action_type,
            "contract_test": True,
        }


def _make_handler(app: FakeHeadRuntimeApp) -> type[BaseHTTPRequestHandler]:
    class FakeEiheadApiHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            trace_header = self.headers.get("X-Trace-Id")
            app.record_request(method="GET", path=path, trace_header=trace_header)

            if path == "/status":
                self._write_json(200, app.status_payload)
                return
            if path == "/capabilities":
                self._write_json(200, app.capabilities_payload)
                return
            self._write_json(404, {"ok": False, "error": "unknown endpoint"})

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            trace_header = self.headers.get("X-Trace-Id")
            app.record_request(method="POST", path=path, trace_header=trace_header)

            if path != "/actions":
                self._write_json(404, {"ok": False, "error": "unknown endpoint"})
                return

            length = int(self.headers.get("Content-Length", "0") or 0)
            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            self._write_json(200, app.accept_action(body, trace_header=trace_header))

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write_json(self, status: int, payload: JsonObject) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return FakeEiheadApiHandler


@contextmanager
def run_fake_eihead_http(app: FakeHeadRuntimeApp | None = None) -> Iterator[tuple[str, FakeHeadRuntimeApp]]:
    """Run a loopback-only fake server on an ephemeral port.

    This is intentionally not a live honjia runtime: it never opens /dev nodes,
    never binds a fixed port, and never reaches outside the local test process.
    """

    runtime = app or FakeHeadRuntimeApp()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(runtime))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}", runtime
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_headclient_gets_status_and_capabilities_from_fake_eihead_http_contract() -> None:
    with run_fake_eihead_http() as (base_url, app):
        client = HeadClient(base_url, timeout=1.0, trace_id="bridge-contract-trace")

        status = client.get_status()
        capabilities = client.get_capabilities()

    assert status == app.status_payload
    assert capabilities == app.capabilities_payload
    assert app.requests == [
        {"method": "GET", "path": "/status", "trace_header": "bridge-contract-trace"},
        {"method": "GET", "path": "/capabilities", "trace_header": "bridge-contract-trace"},
    ]


def test_headclient_action_helpers_post_expected_actions_body_contract() -> None:
    with run_fake_eihead_http() as (base_url, app):
        client = HeadClient(base_url, timeout=1.0, trace_id="action-contract-trace")

        speak_result = client.speak("hello hongtu", voice="minimax", emotion="warm")
        move_result = client.move_head(92, speed=0.35)
        frame_result = client.capture_frame(camera_id="u4k", format="jpeg")

    assert speak_result == {
        "accepted": True,
        "action_id": "fake-action-1",
        "action_type": "speak",
        "contract_test": True,
    }
    assert move_result["action_type"] == "move_head"
    assert frame_result["action_type"] == "capture_frame"
    assert [request["body"] for request in app.action_requests] == [
        {
            "action": {
                "type": "speak",
                "text": "hello hongtu",
                "voice": "minimax",
                "emotion": "warm",
            },
            "trace_id": "action-contract-trace",
        },
        {
            "action": {
                "type": "move_head",
                "axis": "yaw",
                "angle": 92,
                "speed": 0.35,
            },
            "trace_id": "action-contract-trace",
        },
        {
            "action": {
                "type": "capture_frame",
                "camera_id": "u4k",
                "format": "jpeg",
            },
            "trace_id": "action-contract-trace",
        },
    ]
    assert {request["path"] for request in app.requests} == {"/actions"}
    assert {request["trace_header"] for request in app.action_requests} == {"action-contract-trace"}
    for request in app.action_requests:
        action = request["body"]["action"]
        assert "permission" not in action
        assert "safety" not in action


def test_headclient_per_call_trace_id_overrides_default_trace_in_header_and_body() -> None:
    with run_fake_eihead_http() as (base_url, app):
        client = HeadClient(base_url, timeout=1.0, trace_id="default-trace")

        result = client.capture_frame(trace_id="turn-trace-001", camera_id="diagnostic")

    assert result["accepted"] is True
    assert app.requests == [
        {"method": "POST", "path": "/actions", "trace_header": "turn-trace-001"}
    ]
    assert app.action_requests == [
        {
            "body": {
                "action": {
                    "type": "capture_frame",
                    "camera_id": "diagnostic",
                },
                "trace_id": "turn-trace-001",
            },
            "trace_header": "turn-trace-001",
        }
    ]
