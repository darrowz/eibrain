from __future__ import annotations

import json
from pathlib import Path

import pytest

from eiprotocol.models import EventEnvelope
from eiprotocol.validation import ValidationIssue, assert_event_valid, validate_event_strict


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "eiprotocol"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _issue_by_path(issues: list[ValidationIssue]) -> dict[str, ValidationIssue]:
    return {issue.path: issue for issue in issues}


def test_valid_fixture_has_no_strict_issues() -> None:
    payload = _fixture("head_action_request.json")

    assert validate_event_strict(payload, known_event_required=True) == []
    assert_event_valid(payload)


def test_bad_enums_report_structured_issues() -> None:
    payload = _fixture("head_action_request.json")
    payload["type"] = "robot"
    payload["priority"] = "urgent"
    payload["source"]["domain"] = "mars"
    payload["target"]["domain"] = "venus"
    payload["policy"]["decision"] = "maybe"
    payload["policy"]["riskLevel"] = "L99"

    issues = validate_event_strict(payload)
    by_path = _issue_by_path(issues)

    assert by_path["type"].code == "invalid_type"
    assert by_path["priority"].code == "invalid_priority"
    assert by_path["source.domain"].code == "invalid_domain"
    assert by_path["target.domain"].code == "invalid_domain"
    assert by_path["policy.decision"].code == "invalid_policy_decision"
    assert by_path["policy.riskLevel"].code == "invalid_policy_risk_level"
    assert by_path["type"].severity == "error"
    assert by_path["type"].to_dict() == {
        "path": "type",
        "code": "invalid_type",
        "message": "type must be one of action, capability, control, dialogue, error, memory, observation, outcome, policy, training",
        "severity": "error",
    }


def test_bad_timestamp_and_numeric_fields_report_strict_issues() -> None:
    payload = _fixture("capability_manifest.json")
    payload["time"] = "not-a-timestamp"
    payload["sequence"] = 0
    payload["ttlMs"] = -1

    issues = validate_event_strict(payload)
    by_path = _issue_by_path(issues)

    assert by_path["time"].code == "invalid_datetime"
    assert by_path["sequence"].code == "invalid_sequence"
    assert by_path["ttlMs"].code == "invalid_ttl_ms"

    payload["time"] = "2026-05-04"
    payload["sequence"] = 1
    payload["ttlMs"] = 0
    assert _issue_by_path(validate_event_strict(payload))["time"].code == "invalid_datetime"

    payload["time"] = "2026-05-04T10:00:00"
    assert _issue_by_path(validate_event_strict(payload))["time"].code == "invalid_datetime"

    payload["time"] = "2026-05-04T02:31:00Z"

    assert validate_event_strict(payload) == []


def test_policy_decision_and_risk_level_are_optional_but_strict_when_present() -> None:
    payload = _fixture("capability_manifest.json")
    payload["policy"] = {"decision": "confirm"}

    assert validate_event_strict(payload) == []

    payload["policy"]["decision"] = "approved"
    payload["policy"]["riskLevel"] = "L99"

    issues = validate_event_strict(payload)
    by_path = _issue_by_path(issues)

    assert by_path["policy.decision"].code == "invalid_policy_decision"
    assert by_path["policy.riskLevel"].code == "invalid_policy_risk_level"


def test_unknown_event_is_only_an_issue_when_required() -> None:
    payload = _fixture("capability_manifest.json")
    payload["name"] = "ei.future.experimental"

    assert validate_event_strict(payload) == []

    issues = validate_event_strict(payload, known_event_required=True)

    assert len(issues) == 1
    assert issues[0].path == "name"
    assert issues[0].code == "unknown_event"


def test_known_event_name_must_match_type_and_required_content_fields() -> None:
    payload = _fixture("control_ping.json")
    payload["type"] = "dialogue"
    del payload["content"]["pingId"]

    issues = validate_event_strict(payload, known_event_required=True)
    by_path = _issue_by_path(issues)

    assert by_path["type"].code == "event_type_mismatch"
    assert by_path["content.pingId"].code == "missing_content_field"


def test_capability_manifest_grouped_modalities_backends_and_health_remain_optional_for_compatibility() -> None:
    payload = _fixture("capability_manifest.json")
    del payload["content"]["modalities"]
    del payload["content"]["backends"]
    del payload["content"]["health"]

    assert validate_event_strict(payload, known_event_required=True) == []


def test_catalog_round_scoped_events_require_round_id_even_for_policy_events() -> None:
    payload = _fixture("policy_decision.json")
    payload["roundId"] = ""

    issues = validate_event_strict(payload, known_event_required=True)

    assert _issue_by_path(issues)["roundId"].code == "required"


def test_to_dict_input_is_accepted() -> None:
    envelope = EventEnvelope.from_dict(_fixture("capability_manifest.json"))

    assert validate_event_strict(envelope, known_event_required=True) == []


def test_assert_event_valid_includes_code_and_path() -> None:
    payload = _fixture("head_action_request.json")
    payload["priority"] = "urgent"

    with pytest.raises(ValueError) as exc_info:
        assert_event_valid(payload)

    message = str(exc_info.value)
    assert "invalid_priority" in message
    assert "priority" in message
