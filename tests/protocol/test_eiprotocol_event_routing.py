from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eiprotocol import EventEnvelope
from eiprotocol.event_routing import classify_event


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "eiprotocol"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


class EventLike:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


def test_classify_event_is_exported_from_package() -> None:
    import eiprotocol

    assert eiprotocol.classify_event is classify_event


def test_classifies_supported_event_fixture_routes() -> None:
    expected_routes = {
        "capability_manifest.json": "capability_manifest",
        "asr_partial.json": "asr_partial",
        "asr_final.json": "asr_final",
        "realtime_vision_frame.json": "realtime_vision_frame",
        "execution_outcome.json": "execution_outcome",
        "user_feedback.json": "user_feedback",
    }

    for fixture_name, expected_route in expected_routes.items():
        payload = _fixture(fixture_name)

        route = classify_event(payload)

        assert route["status"] == "routed"
        assert route["route"] == expected_route
        assert route["eventName"] == payload["name"]
        assert route["eventType"] == payload["type"]


def test_extracts_action_request_fields_for_eihead_runtime() -> None:
    payload = _fixture("head_action_request.json")

    route = classify_event(EventEnvelope.from_dict(payload))

    assert route["status"] == "routed"
    assert route["route"] == "action_request"
    assert route["actionId"] == "act_move_head_001"
    assert route["actionType"] == "move_head"
    assert route["target"] == "neck.pan"
    assert route["params"] == {"targetAngle": 92, "durationMs": 240, "reason": "center tracked person"}
    assert route["riskLevel"] == "L1"
    assert route["idempotencyKey"] == "act_move_head_001_once"


def test_accepts_event_like_object_with_to_dict() -> None:
    payload = _fixture("asr_final.json")

    route = classify_event(EventLike(payload))

    assert route["status"] == "routed"
    assert route["route"] == "asr_final"
    assert route["eventName"] == "ei.dialogue.asr.final"


def test_unknown_valid_event_is_not_processed_not_invalid() -> None:
    payload = _fixture("asr_final.json")
    payload["name"] = "ei.dialogue.agent.delta"
    payload["content"] = {"delta": "hello"}

    route = classify_event(payload)

    assert route == {
        "status": "not_processed",
        "reason": "unsupported_event_name",
        "eventName": "ei.dialogue.agent.delta",
        "eventType": "dialogue",
    }


def test_invalid_event_reports_validation_errors() -> None:
    payload = _fixture("asr_final.json")
    del payload["source"]

    route = classify_event(payload)

    assert route["status"] == "invalid"
    assert route["reason"] == "invalid_event"
    assert route["eventName"] == "ei.dialogue.asr.final"
    assert "source is required" in route["errors"]
