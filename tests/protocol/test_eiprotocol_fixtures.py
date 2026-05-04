from __future__ import annotations

import json
from pathlib import Path

import pytest

from eiprotocol import EventEnvelope, validate_event
from eiprotocol.catalog import list_event_names
from eiprotocol.validation import validate_event_strict


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "eiprotocol"
EXPECTED_FIXTURES = {
    "action_complete.json",
    "action_dispatch.json",
    "action_progress.json",
    "agent_delta.json",
    "agent_final.json",
    "asr_final.json",
    "asr_partial.json",
    "audio_chunk.json",
    "capability_manifest.json",
    "control_ack.json",
    "control_error.json",
    "control_hello.json",
    "control_ping.json",
    "control_pong.json",
    "control_resume.json",
    "emergency_stop.json",
    "error_event.json",
    "execution_outcome.json",
    "head_action_request.json",
    "interrupt_requested.json",
    "memory_recall_request.json",
    "memory_recall_result.json",
    "memory_write_committed.json",
    "memory_write_proposed.json",
    "policy_decision.json",
    "realtime_vision_frame.json",
    "training_signal.json",
    "tts_delta.json",
    "tts_final.json",
    "user_feedback.json",
}


def test_eiprotocol_fixture_set_is_complete() -> None:
    actual = {path.name for path in FIXTURE_DIR.glob("*.json")}

    assert actual == EXPECTED_FIXTURES


def test_eiprotocol_fixtures_cover_every_catalog_event_name() -> None:
    event_names = set()
    for fixture_path in FIXTURE_DIR.glob("*.json"):
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        event_names.add(payload["name"])

    assert event_names == set(list_event_names())


@pytest.mark.parametrize("fixture_name", sorted(EXPECTED_FIXTURES))
def test_eiprotocol_fixture_validates_and_round_trips(fixture_name: str) -> None:
    fixture_path = FIXTURE_DIR / fixture_name
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
    assert validate_event(payload) == []
    assert validate_event_strict(payload, known_event_required=True) == []
    assert EventEnvelope.from_dict(payload).to_dict() == payload
