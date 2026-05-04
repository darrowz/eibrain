from __future__ import annotations

import json
from pathlib import Path

import pytest

from eiprotocol import EventEnvelope, validate_event


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "eiprotocol"
EXPECTED_FIXTURES = {
    "asr_final.json",
    "asr_partial.json",
    "capability_manifest.json",
    "execution_outcome.json",
    "head_action_request.json",
    "realtime_vision_frame.json",
    "user_feedback.json",
}


def test_eiprotocol_fixture_set_is_complete() -> None:
    actual = {path.name for path in FIXTURE_DIR.glob("*.json")}

    assert actual == EXPECTED_FIXTURES


@pytest.mark.parametrize("fixture_name", sorted(EXPECTED_FIXTURES))
def test_eiprotocol_fixture_validates_and_round_trips(fixture_name: str) -> None:
    fixture_path = FIXTURE_DIR / fixture_name
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
    assert validate_event(payload) == []
    assert EventEnvelope.from_dict(payload).to_dict() == payload
