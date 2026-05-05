from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eiprotocol.catalog import require_event_definition
from eiprotocol.codec import EventDecodeError, dumps_event, loads_event
from eiprotocol.event_routing import classify_event
from eiprotocol.validation import ValidationIssue, validate_event_strict


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "eiprotocol"

WAVE2_EVENT_CASES = {
    "fast_hypothesis.json": {
        "name": "ei.dialogue.fast_hypothesis",
        "event_type": "dialogue",
        "plane": "dialogue",
        "direction": "brain_to_head",
        "route": "dialogue_fast_hypothesis",
        "realtime": True,
        "round_scoped": True,
        "required_fields": ("hypothesisId", "text", "confidence"),
    },
    "dialogue_decision_stable.json": {
        "name": "ei.dialogue.decision.stable",
        "event_type": "dialogue",
        "plane": "dialogue",
        "direction": "brain_to_head",
        "route": "dialogue_decision_stable",
        "realtime": True,
        "round_scoped": True,
        "required_fields": ("decisionId", "decision", "confidence"),
    },
    "head_status_report.json": {
        "name": "ei.observation.head.status.report",
        "event_type": "observation",
        "plane": "observation",
        "direction": "head_to_brain",
        "route": "head_status_report",
        "realtime": True,
        "round_scoped": False,
        "required_fields": ("status", "components", "reportedAt"),
    },
}


def _fixture(fixture_name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))


def _issue_by_path(issues: list[ValidationIssue]) -> dict[str, ValidationIssue]:
    return {issue.path: issue for issue in issues}


@pytest.mark.parametrize("fixture_name", sorted(WAVE2_EVENT_CASES))
def test_wave2_realtime_dialogue_and_head_events_have_catalog_definitions(fixture_name: str) -> None:
    expected = WAVE2_EVENT_CASES[fixture_name]

    definition = require_event_definition(expected["name"])

    assert definition.event_type == expected["event_type"]
    assert definition.plane == expected["plane"]
    assert definition.direction == expected["direction"]
    assert definition.realtime is expected["realtime"]
    assert definition.round_scoped is expected["round_scoped"]
    assert definition.side_effecting is False
    assert definition.required_content_fields == expected["required_fields"]


@pytest.mark.parametrize("fixture_name", sorted(WAVE2_EVENT_CASES))
def test_wave2_realtime_dialogue_and_head_fixtures_validate_and_codec_round_trip(fixture_name: str) -> None:
    payload = _fixture(fixture_name)

    assert validate_event_strict(payload, known_event_required=True) == []
    assert loads_event(dumps_event(payload)).to_dict() == payload


@pytest.mark.parametrize("fixture_name", sorted(WAVE2_EVENT_CASES))
def test_wave2_realtime_dialogue_and_head_fixtures_route_with_metadata(fixture_name: str) -> None:
    expected = WAVE2_EVENT_CASES[fixture_name]
    payload = _fixture(fixture_name)

    route = classify_event(payload)

    assert route["status"] == "routed"
    assert route["route"] == expected["route"]
    assert route["eventName"] == expected["name"]
    assert route["eventType"] == expected["event_type"]
    assert route["plane"] == expected["plane"]
    assert route["sideEffecting"] is False
    assert route["roundScoped"] is expected["round_scoped"]
    assert route["realtime"] is expected["realtime"]
    assert route["knownEvent"] is True


@pytest.mark.parametrize("fixture_name", sorted(WAVE2_EVENT_CASES))
def test_wave2_required_content_fields_are_enforced_by_strict_validation_and_codec(fixture_name: str) -> None:
    payload = _fixture(fixture_name)
    required_field = WAVE2_EVENT_CASES[fixture_name]["required_fields"][0]
    del payload["content"][required_field]

    issues = validate_event_strict(payload, known_event_required=True)

    by_path = _issue_by_path(issues)
    assert by_path[f"content.{required_field}"].code == "missing_content_field"
    with pytest.raises(EventDecodeError) as exc_info:
        loads_event(json.dumps(payload))

    assert exc_info.value.kind == "invalid_event"
    assert any(f"content.{required_field}" in error for error in exc_info.value.details["errors"])
