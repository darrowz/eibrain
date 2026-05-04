from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eiprotocol import EventEnvelope, validate_event
from eiprotocol.catalog import get_event_definition
from eiprotocol.validation import validate_event_strict


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "eiprotocol"

GOLDEN_FIXTURES = {
    "action_complete.json",
    "action_dispatch.json",
    "action_progress.json",
    "agent_delta.json",
    "agent_final.json",
    "audio_chunk.json",
    "control_ack.json",
    "control_ping.json",
    "emergency_stop.json",
    "error_event.json",
    "interrupt_requested.json",
    "memory_recall_request.json",
    "memory_recall_result.json",
    "memory_write_committed.json",
    "memory_write_proposed.json",
    "policy_decision.json",
    "training_signal.json",
    "tts_delta.json",
    "tts_final.json",
}

REQUIRED_ENVELOPE_FIELDS = {
    "specVersion",
    "id",
    "type",
    "name",
    "time",
    "sequence",
    "requestId",
    "sessionId",
    "roundId",
    "traceId",
    "source",
    "priority",
    "content",
    "policy",
}

TURN_SCOPED_TYPES = {"dialogue", "action", "memory", "outcome", "training"}


def _load_fixture(fixture_name: str) -> dict[str, Any]:
    fixture_path = FIXTURE_DIR / fixture_name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_golden_fixture_set_exists() -> None:
    actual = {path.name for path in FIXTURE_DIR.glob("*.json")}

    assert GOLDEN_FIXTURES <= actual


@pytest.mark.parametrize("fixture_name", sorted(GOLDEN_FIXTURES))
def test_golden_fixture_validates_and_round_trips(fixture_name: str) -> None:
    payload = _load_fixture(fixture_name)

    assert isinstance(payload, dict)
    assert REQUIRED_ENVELOPE_FIELDS <= payload.keys()
    assert payload["sessionId"]
    assert payload["traceId"]
    if payload["type"] in TURN_SCOPED_TYPES:
        assert payload["roundId"]

    assert validate_event(payload) == []
    assert validate_event_strict(payload, known_event_required=True) == []

    envelope = EventEnvelope.from_dict(payload)
    assert envelope.to_dict() == payload

    round_trip = EventEnvelope.from_json(envelope.to_json())
    assert round_trip.to_dict() == payload


@pytest.mark.parametrize("fixture_name", sorted(GOLDEN_FIXTURES))
def test_golden_fixture_content_contract(fixture_name: str) -> None:
    payload = _load_fixture(fixture_name)
    content = payload["content"]
    definition = get_event_definition(payload["name"])
    assert definition is not None
    required_fields = set(definition.required_content_fields)

    assert isinstance(content, dict)
    assert required_fields <= content.keys()

    if fixture_name == "audio_chunk.json":
        assert content["chunkIndex"] == 0
        assert content["audioBase64"]
    elif fixture_name == "agent_delta.json":
        assert content["delta"]
    elif fixture_name == "action_dispatch.json":
        assert content["actionId"]
        assert content["idempotencyKey"]
    elif fixture_name.startswith("memory_recall"):
        assert content["query"]
        if fixture_name == "memory_recall_result.json":
            assert content["results"]
    elif fixture_name.startswith("memory_write"):
        assert content["writeId"]
