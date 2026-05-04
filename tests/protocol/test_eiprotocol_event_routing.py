from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eiprotocol import EventEnvelope
from eiprotocol.catalog import get_event_definition, list_event_names
from eiprotocol.event_routing import classify_event


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "eiprotocol"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _event(name: str, event_type: str, content: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = _fixture("asr_final.json")
    payload["id"] = f"evt_{name.replace('ei.', '').replace('.', '_')}"
    payload["type"] = event_type
    payload["name"] = name
    if content is not None:
        payload["content"] = dict(content)
    else:
        payload["content"] = _minimal_content(name)
    return payload


def _minimal_content(name: str) -> dict[str, Any]:
    definition = get_event_definition(name)
    if definition is None:
        return {"value": True}

    values: dict[str, Any] = {}
    for field_name in definition.required_content_fields:
        values[field_name] = _minimal_value(field_name)
    return values or {"value": True}


def _minimal_value(field_name: str) -> Any:
    if field_name in {"capabilities", "results"}:
        return []
    if field_name in {"proposal", "target"}:
        return {}
    if field_name in {"final", "success", "committed"}:
        return True
    if field_name in {"chunkIndex", "limit", "progress", "resultCount"}:
        return 0
    if field_name == "riskLevel":
        return "L1"
    if field_name == "decision":
        return "not_required"
    if field_name == "status":
        return "ok"
    if field_name == "sentAt":
        return "2026-05-04T10:30:20.123+08:00"
    return f"{field_name}_test"


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


def test_classifies_v01_mvp_routes_with_metadata() -> None:
    cases = [
        ("ei.control.hello", "control", "control_hello", "control", False, False, False),
        ("ei.control.ping", "control", "control_ping", "control", False, False, True),
        ("ei.control.pong", "control", "control_pong", "control", False, False, True),
        ("ei.control.resume", "control", "control_resume", "control", False, False, False),
        ("ei.control.ack", "control", "control_ack", "control", False, False, True),
        ("ei.control.error", "control", "control_error", "control", False, False, True),
        ("ei.error.event", "error", "error_event", "error", False, False, False),
        ("ei.observation.audio.chunk", "observation", "audio_chunk", "observation", False, False, True),
        ("ei.dialogue.agent.delta", "dialogue", "agent_delta", "dialogue", False, True, True),
        ("ei.dialogue.agent.final", "dialogue", "agent_final", "dialogue", False, True, False),
        ("ei.dialogue.tts.delta", "dialogue", "tts_delta", "dialogue", False, True, True),
        ("ei.dialogue.tts.final", "dialogue", "tts_final", "dialogue", False, True, False),
        ("ei.dialogue.interrupt.requested", "dialogue", "interrupt_requested", "dialogue", False, True, True),
        ("ei.action.progress", "action", "action_progress", "action", False, True, True),
        ("ei.action.complete", "action", "action_complete", "action", False, True, False),
        ("ei.policy.decision", "policy", "policy_decision", "policy", False, True, False),
        ("ei.memory.recall.request", "memory", "memory_recall_request", "memory", False, True, False),
        ("ei.memory.recall.result", "memory", "memory_recall_result", "memory", False, True, False),
        ("ei.memory.write.proposed", "memory", "memory_write_proposed", "memory", False, True, False),
        ("ei.memory.write.committed", "memory", "memory_write_committed", "memory", True, True, False),
        ("ei.training.signal", "training", "training_signal", "training", False, True, False),
    ]

    for name, event_type, expected_route, plane, side_effecting, round_scoped, realtime in cases:
        route = classify_event(_event(name, event_type))

        assert route["status"] == "routed"
        assert route["route"] == expected_route
        assert route["eventName"] == name
        assert route["eventType"] == event_type
        assert route["plane"] == plane
        assert route["sideEffecting"] is side_effecting
        assert route["roundScoped"] is round_scoped
        assert route["realtime"] is realtime
        assert route["knownEvent"] is True


def test_existing_routes_include_metadata() -> None:
    route = classify_event(_fixture("asr_final.json"))

    assert route["status"] == "routed"
    assert route["route"] == "asr_final"
    assert route["plane"] == "dialogue"
    assert route["sideEffecting"] is False
    assert route["roundScoped"] is True
    assert route["realtime"] is False
    assert route["knownEvent"] is True


def test_route_metadata_matches_catalog_definitions() -> None:
    for event_name in list_event_names():
        definition = get_event_definition(event_name)
        assert definition is not None
        route = classify_event(_event(event_name, definition.event_type))

        assert route["status"] == "routed"
        assert route["eventType"] == definition.event_type
        assert route["plane"] == definition.plane
        assert route["sideEffecting"] is definition.side_effecting
        assert route["roundScoped"] is definition.round_scoped
        assert route["realtime"] is definition.realtime
        assert route["knownEvent"] is True


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


def test_extracts_action_dispatch_fields_for_eihead_runtime() -> None:
    payload = _fixture("head_action_request.json")
    payload["name"] = "ei.action.dispatch"

    route = classify_event(payload)

    assert route["status"] == "routed"
    assert route["route"] == "action_dispatch"
    assert route["actionId"] == "act_move_head_001"
    assert route["actionType"] == "move_head"
    assert route["target"] == "neck.pan"
    assert route["params"] == {"targetAngle": 92, "durationMs": 240, "reason": "center tracked person"}
    assert route["riskLevel"] == "L1"
    assert route["idempotencyKey"] == "act_move_head_001_once"
    assert route["sideEffecting"] is True


def test_extracts_action_emergency_stop_fields_with_policy_risk_fallback() -> None:
    payload = _fixture("head_action_request.json")
    payload["name"] = "ei.action.emergency.stop"
    payload["content"] = {
        "actionId": "act_stop_001",
        "actionType": "emergency_stop",
        "target": "all",
        "reason": "operator_requested",
        "params": {"reason": "operator_requested"},
        "idempotencyKey": "act_stop_001_once",
    }
    payload["policy"]["riskLevel"] = "L4"

    route = classify_event(payload)

    assert route["status"] == "routed"
    assert route["route"] == "action_emergency_stop"
    assert route["actionId"] == "act_stop_001"
    assert route["actionType"] == "emergency_stop"
    assert route["target"] == "all"
    assert route["params"] == {"reason": "operator_requested"}
    assert route["riskLevel"] == "L4"
    assert route["idempotencyKey"] == "act_stop_001_once"
    assert route["sideEffecting"] is True


def test_accepts_event_like_object_with_to_dict() -> None:
    payload = _fixture("asr_final.json")

    route = classify_event(EventLike(payload))

    assert route["status"] == "routed"
    assert route["route"] == "asr_final"
    assert route["eventName"] == "ei.dialogue.asr.final"


def test_unknown_valid_event_is_not_processed_not_invalid() -> None:
    payload = _fixture("asr_final.json")
    payload["name"] = "ei.dialogue.future.unsupported"
    payload["content"] = {"value": "hello"}

    route = classify_event(payload)

    assert route == {
        "status": "not_processed",
        "reason": "unsupported_event_name",
        "eventName": "ei.dialogue.future.unsupported",
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


def test_known_event_missing_catalog_content_fields_is_invalid() -> None:
    payload = _fixture("control_ping.json")
    del payload["content"]["sentAt"]

    route = classify_event(payload)

    assert route["status"] == "invalid"
    assert route["reason"] == "invalid_event"
    assert any("missing_content_field at content.sentAt" in error for error in route["errors"])
