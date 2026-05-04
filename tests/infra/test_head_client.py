from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from eibrain.infra.head_client import HeadClient, HeadClientError


class FakeResponse:
    def __init__(self, body: dict[str, Any] | list[Any] | str | bytes, *, status: int = 200) -> None:
        self.status = status
        if isinstance(body, bytes):
            self._body = body
        elif isinstance(body, str):
            self._body = body.encode("utf-8")
        else:
            self._body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status


def test_get_status_sends_trace_header_and_uses_base_url(monkeypatch) -> None:
    calls: list[tuple[object, float]] = []

    def fake_urlopen(req: object, timeout: float) -> FakeResponse:
        calls.append((req, timeout))
        return FakeResponse({"ok": True, "node_id": "honjia"})

    monkeypatch.setattr("eibrain.infra.head_client.request.urlopen", fake_urlopen)

    client = HeadClient("http://honjia:18081/api/", timeout=1.25, trace_id="trace-001")
    payload = client.get_status()

    req, timeout = calls[0]
    assert payload == {"ok": True, "node_id": "honjia"}
    assert req.full_url == "http://honjia:18081/api/status"
    assert req.get_method() == "GET"
    assert req.get_header("X-trace-id") == "trace-001"
    assert timeout == 1.25


def test_get_capabilities_returns_json_object(monkeypatch) -> None:
    monkeypatch.setattr(
        "eibrain.infra.head_client.request.urlopen",
        lambda req, timeout: FakeResponse({"capabilities": {"vision": ["capture_frame"]}}),
    )

    client = HeadClient("http://honjia:18081")

    assert client.get_capabilities() == {"capabilities": {"vision": ["capture_frame"]}}


def test_post_action_wraps_action_and_trace_id(monkeypatch) -> None:
    calls: list[object] = []

    def fake_urlopen(req: object, timeout: float) -> FakeResponse:
        calls.append(req)
        return FakeResponse({"accepted": True, "action_id": "act-1"})

    monkeypatch.setattr("eibrain.infra.head_client.request.urlopen", fake_urlopen)

    client = HeadClient("http://honjia:18081", trace_id="trace-002")
    payload = client.speak("你好鸿途", voice="minimax", emotion=None)

    req = calls[0]
    body = json.loads(req.data.decode("utf-8"))
    assert payload == {"accepted": True, "action_id": "act-1"}
    assert req.full_url == "http://honjia:18081/actions"
    assert req.get_method() == "POST"
    assert req.get_header("Content-type") == "application/json"
    assert body == {
        "action": {
            "type": "speak",
            "text": "你好鸿途",
            "voice": "minimax",
        },
        "trace_id": "trace-002",
    }


def test_post_event_wraps_mapping_event_and_trace_id(monkeypatch) -> None:
    calls: list[object] = []

    def fake_urlopen(req: object, timeout: float) -> FakeResponse:
        calls.append(req)
        return FakeResponse({"accepted": True, "event_id": "evt-1"})

    monkeypatch.setattr("eibrain.infra.head_client.request.urlopen", fake_urlopen)

    client = HeadClient("http://honjia:18081", trace_id="trace-evt")
    payload = client.post_event({"type": "speech.started", "source": "mouth"})

    req = calls[0]
    body = json.loads(req.data.decode("utf-8"))
    assert payload == {"accepted": True, "event_id": "evt-1"}
    assert req.full_url == "http://honjia:18081/events"
    assert req.get_method() == "POST"
    assert req.get_header("Content-type") == "application/json"
    assert req.get_header("X-trace-id") == "trace-evt"
    assert body == {
        "event": {"type": "speech.started", "source": "mouth"},
        "trace_id": "trace-evt",
    }


def test_post_event_accepts_to_dict_event_envelope(monkeypatch) -> None:
    bodies: list[dict[str, Any]] = []

    class EventEnvelope:
        def to_dict(self) -> dict[str, Any]:
            return {"type": "vision.frame", "payload": {"frame_id": "frame-1"}}

    def fake_urlopen(req: object, timeout: float) -> FakeResponse:
        bodies.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse({"ok": True})

    monkeypatch.setattr("eibrain.infra.head_client.request.urlopen", fake_urlopen)

    client = HeadClient("http://honjia:18081")

    assert client.post_event(EventEnvelope()) == {"ok": True}
    assert bodies == [
        {"event": {"type": "vision.frame", "payload": {"frame_id": "frame-1"}}},
    ]


def test_post_event_rejects_non_mapping_event_inputs() -> None:
    class InvalidEnvelope:
        def to_dict(self) -> str:
            return "not a mapping"

    client = HeadClient("http://honjia:18081")

    with pytest.raises(TypeError, match="event must be a mapping"):
        client.post_event(["not", "a", "mapping"])

    with pytest.raises(TypeError, match="event.to_dict\\(\\) must return a mapping"):
        client.post_event(InvalidEnvelope())


def test_action_helpers_build_expected_payloads(monkeypatch) -> None:
    bodies: list[dict[str, Any]] = []

    def fake_urlopen(req: object, timeout: float) -> FakeResponse:
        bodies.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse({"ok": True})

    monkeypatch.setattr("eibrain.infra.head_client.request.urlopen", fake_urlopen)

    client = HeadClient("http://honjia:18081")
    client.move_head(92, speed=0.35)
    client.stop_speech(reason="barge_in")
    client.capture_frame(camera_id="u4k", format="jpeg")

    assert bodies == [
        {"action": {"type": "move_head", "axis": "yaw", "angle": 92, "speed": 0.35}},
        {"action": {"type": "stop_speech", "reason": "barge_in"}},
        {"action": {"type": "capture_frame", "camera_id": "u4k", "format": "jpeg"}},
    ]


def test_timeout_raises_structured_error(monkeypatch) -> None:
    def fake_urlopen(req: object, timeout: float) -> FakeResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr("eibrain.infra.head_client.request.urlopen", fake_urlopen)

    client = HeadClient("http://honjia:18081", trace_id="trace-timeout")

    with pytest.raises(HeadClientError) as exc:
        client.get_status()

    assert exc.value.kind == "timeout"
    assert exc.value.to_dict()["trace_id"] == "trace-timeout"
    assert "timed out" in str(exc.value)


def test_urlerror_timeout_raises_timeout_kind(monkeypatch) -> None:
    def fake_urlopen(req: object, timeout: float) -> FakeResponse:
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr("eibrain.infra.head_client.request.urlopen", fake_urlopen)

    client = HeadClient("http://honjia:18081")

    with pytest.raises(HeadClientError) as exc:
        client.get_capabilities()

    assert exc.value.kind == "timeout"


def test_http_error_raises_structured_error(monkeypatch) -> None:
    def fake_urlopen(req: object, timeout: float) -> FakeResponse:
        raise HTTPError(
            req.full_url,
            503,
            "Service Unavailable",
            hdrs={},
            fp=io.BytesIO(b'{"error":"head offline"}'),
        )

    monkeypatch.setattr("eibrain.infra.head_client.request.urlopen", fake_urlopen)

    client = HeadClient("http://honjia:18081")

    with pytest.raises(HeadClientError) as exc:
        client.get_status()

    assert exc.value.kind == "http_error"
    assert exc.value.status_code == 503
    assert exc.value.response_body == '{"error":"head offline"}'


def test_invalid_json_raises_structured_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "eibrain.infra.head_client.request.urlopen",
        lambda req, timeout: FakeResponse("<html>not json</html>"),
    )

    client = HeadClient("http://honjia:18081")

    with pytest.raises(HeadClientError) as exc:
        client.capture_frame()

    assert exc.value.kind == "invalid_json"
    assert exc.value.response_body == "<html>not json</html>"


def test_non_object_json_raises_structured_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "eibrain.infra.head_client.request.urlopen",
        lambda req, timeout: FakeResponse(["not", "object"]),
    )

    client = HeadClient("http://honjia:18081")

    with pytest.raises(HeadClientError) as exc:
        client.get_status()

    assert exc.value.kind == "invalid_json"
