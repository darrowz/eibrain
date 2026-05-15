from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eibrain.infra.head_events import event_idempotency_key, post_head_action_event


FIXTURE_DIR = (Path(__file__).resolve().parents[3] / "eiprotocol" / "tests" / "fixtures" / "eiprotocol").resolve()


class FakeHeadClient:
    def __init__(self) -> None:
        self.events: list[tuple[dict[str, Any], str | None]] = []

    def post_event(self, event: dict[str, Any], *, trace_id: str | None = None) -> dict[str, Any]:
        self.events.append((dict(event), trace_id))
        return {"accepted": True, "event_id": event.get("id")}


class EventFixture:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _action_event(**overrides: Any) -> dict[str, Any]:
    event = {
        "specVersion": "eiprotocol/0.1",
        "id": "evt_action_1",
        "type": "action",
        "name": "ei.action.request",
        "traceId": "trace-from-event",
        "content": {
            "actionId": "act-1",
            "actionType": "speak",
            "target": "speaker.default",
            "params": {"text": "hello"},
            "idempotencyKey": "act-1-once",
        },
    }
    event.update(overrides)
    return event


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_event_idempotency_key_reads_eiprotocol_content_key() -> None:
    event = EventFixture(_action_event())

    assert event_idempotency_key(event) == "act-1-once"


def test_event_idempotency_key_falls_back_to_action_id_for_action_events() -> None:
    event = _action_event(content={"actionId": "act-2", "params": {}})

    assert event_idempotency_key(event) == "act-2"


def test_post_head_action_event_posts_json_mapping_and_preserves_event_trace_id() -> None:
    client = FakeHeadClient()
    event = EventFixture(_fixture("head_action_request.json"))

    response = post_head_action_event(client, event)

    assert response == {"accepted": True, "event_id": "evt_head_action_001"}
    assert client.events == [(_fixture("head_action_request.json"), "trc_voice_001")]


def test_post_head_action_event_allows_explicit_trace_id_override() -> None:
    client = FakeHeadClient()

    post_head_action_event(client, _fixture("head_action_request.json"), trace_id="trace-explicit")

    assert client.events == [(_fixture("head_action_request.json"), "trace-explicit")]


def test_post_head_action_event_rejects_incomplete_action_request_envelope() -> None:
    client = FakeHeadClient()

    with pytest.raises(ValueError, match="valid ei.action.request"):
        post_head_action_event(client, _action_event())

    assert client.events == []


def test_post_head_action_event_rejects_non_action_events() -> None:
    client = FakeHeadClient()
    event = {
        "specVersion": "eiprotocol/0.1",
        "id": "evt_audio_1",
        "type": "dialogue",
        "name": "ei.dialogue.asr.final",
        "content": {"text": "hello"},
    }

    with pytest.raises(ValueError, match="expected valid ei.action.request"):
        post_head_action_event(client, event)

    assert client.events == []


def test_post_head_action_event_rejects_incomplete_non_action_events() -> None:
    client = FakeHeadClient()

    with pytest.raises(ValueError, match="expected valid ei.action.request"):
        post_head_action_event(client, {"id": "evt_incomplete"})

    assert client.events == []
