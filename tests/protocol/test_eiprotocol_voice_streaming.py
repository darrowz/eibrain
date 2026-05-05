from __future__ import annotations

from typing import Any, Callable

import pytest

from eiprotocol import builders
from eiprotocol.builders import build_asr_event
from eiprotocol.catalog import require_event_definition
from eiprotocol.codec import dumps_event, loads_event
from eiprotocol.event_routing import classify_event
from eiprotocol.models import EventEnvelope
from eiprotocol.validation import ValidationIssue, validate_event_strict


VOICE_STREAMING_EVENTS = {
    "ei.voice.audio.frame": ("observation", "head_to_brain", "voice_audio_frame"),
    "ei.voice.asr.partial": ("dialogue", "head_to_brain", "voice_asr_partial"),
    "ei.voice.asr.final": ("dialogue", "head_to_brain", "voice_asr_final"),
    "ei.voice.tts.sentence_start": ("dialogue", "brain_to_head", "voice_tts_sentence_start"),
    "ei.voice.tts.chunk": ("dialogue", "brain_to_head", "voice_tts_chunk"),
    "ei.voice.playback.started": ("dialogue", "head_to_brain", "voice_playback_started"),
    "ei.voice.playback.stopped": ("dialogue", "head_to_brain", "voice_playback_stopped"),
    "ei.voice.barge_in.detected": ("dialogue", "head_to_brain", "voice_barge_in_detected"),
    "ei.voice.session.heartbeat": ("control", "bidirectional", "voice_session_heartbeat"),
}

VOICE_REQUIRED_FIELDS = {
    "ei.voice.audio.frame": ("streamId", "chunkIndex", "audioBase64"),
    "ei.voice.asr.partial": ("text", "final"),
    "ei.voice.asr.final": ("text", "final"),
    "ei.voice.tts.sentence_start": ("text",),
    "ei.voice.tts.chunk": ("streamId", "chunkIndex", "audioBase64"),
    "ei.voice.playback.started": ("state",),
    "ei.voice.playback.stopped": ("state", "reason"),
    "ei.voice.barge_in.detected": ("reason",),
    "ei.voice.session.heartbeat": ("state", "health"),
}

HEAD_SOURCE = {"domain": "eihead", "instanceId": "head-1"}
BRAIN_SOURCE = {"domain": "eibrain", "instanceId": "brain-1"}


def _builder(name: str) -> Callable[..., EventEnvelope]:
    builder = getattr(builders, name, None)
    assert callable(builder), f"missing builder: {name}"
    return builder


def _issue_by_path(issues: list[ValidationIssue]) -> dict[str, ValidationIssue]:
    return {issue.path: issue for issue in issues}


def _voice_builder_cases() -> list[tuple[str, Callable[[], EventEnvelope]]]:
    common = {"session_id": "session-1", "round_id": "round-1", "trace_id": "trace-1"}
    return [
        (
            "ei.voice.audio.frame",
            lambda: _builder("build_voice_audio_frame_event")(
                source=HEAD_SOURCE,
                stream_id="mic-stream",
                chunk_index=2,
                audio_base64="UklGRg==",
                latency_ms=9,
                **common,
            ),
        ),
        (
            "ei.voice.asr.partial",
            lambda: _builder("build_voice_asr_event")(
                source=HEAD_SOURCE,
                text="ni hao",
                final=False,
                latency_ms=31.5,
                **common,
            ),
        ),
        (
            "ei.voice.asr.final",
            lambda: _builder("build_voice_asr_event")(
                source=HEAD_SOURCE,
                text="ni hao hong tu",
                final=True,
                latency_ms=85,
                **common,
            ),
        ),
        (
            "ei.voice.tts.sentence_start",
            lambda: _builder("build_voice_tts_sentence_start_event")(
                source=BRAIN_SOURCE,
                text="你好，我在。",
                latency_ms=120,
                **common,
            ),
        ),
        (
            "ei.voice.tts.chunk",
            lambda: _builder("build_voice_tts_chunk_event")(
                source=BRAIN_SOURCE,
                stream_id="tts-stream",
                chunk_index=3,
                audio_base64="AAEC",
                latency_ms=145,
                **common,
            ),
        ),
        (
            "ei.voice.playback.started",
            lambda: _builder("build_voice_playback_state_event")(
                source=HEAD_SOURCE,
                started=True,
                latency_ms=12,
                **common,
            ),
        ),
        (
            "ei.voice.playback.stopped",
            lambda: _builder("build_voice_playback_state_event")(
                source=HEAD_SOURCE,
                started=False,
                reason="completed",
                latency_ms=240,
                **common,
            ),
        ),
        (
            "ei.voice.barge_in.detected",
            lambda: _builder("build_voice_barge_in_detected_event")(
                source=HEAD_SOURCE,
                reason="near_field_speech",
                latency_ms=18,
                **common,
            ),
        ),
        (
            "ei.voice.session.heartbeat",
            lambda: _builder("build_voice_session_heartbeat_event")(
                source=HEAD_SOURCE,
                session_id="session-1",
                trace_id="trace-1",
                state="capturing",
                health={"capture": "ok", "playback": "idle"},
                latency_ms=5,
            ),
        ),
    ]


@pytest.mark.parametrize("event_name", sorted(VOICE_STREAMING_EVENTS))
def test_voice_streaming_events_have_catalog_definitions(event_name: str) -> None:
    expected_type, expected_direction, _ = VOICE_STREAMING_EVENTS[event_name]

    definition = require_event_definition(event_name)

    assert definition.event_type == expected_type
    assert definition.plane == expected_type
    assert definition.direction == expected_direction
    assert definition.realtime is True
    assert definition.round_scoped is (expected_type == "dialogue")
    assert definition.side_effecting is False
    assert definition.required_content_fields == VOICE_REQUIRED_FIELDS[event_name]


@pytest.mark.parametrize("event_name,builder_case", _voice_builder_cases())
def test_voice_streaming_builders_emit_strictly_valid_round_trippable_events(
    event_name: str,
    builder_case: Callable[[], EventEnvelope],
) -> None:
    event = builder_case()
    payload = event.to_dict()

    assert isinstance(event, EventEnvelope)
    assert payload["name"] == event_name
    assert payload["sessionId"] == "session-1"
    assert payload["traceId"] == "trace-1"
    assert validate_event_strict(event, known_event_required=True) == []
    assert loads_event(dumps_event(event)).to_dict() == payload


@pytest.mark.parametrize("event_name,builder_case", _voice_builder_cases())
def test_voice_streaming_builder_payloads_preserve_streaming_fields(
    event_name: str,
    builder_case: Callable[[], EventEnvelope],
) -> None:
    payload = builder_case().to_dict()
    content = payload["content"]

    if event_name in {"ei.voice.audio.frame", "ei.voice.tts.chunk"}:
        assert content["streamId"]
        assert content["chunkIndex"] >= 0
        assert content["audioBase64"]
    if event_name in {"ei.voice.asr.partial", "ei.voice.asr.final", "ei.voice.tts.sentence_start"}:
        assert content["text"]
    if "latencyMs" in content:
        assert content["latencyMs"] >= 0
    if event_name in {"ei.voice.playback.stopped", "ei.voice.barge_in.detected"}:
        assert content["reason"]


@pytest.mark.parametrize("event_name,builder_case", _voice_builder_cases())
def test_voice_streaming_events_route_with_expected_route_names(
    event_name: str,
    builder_case: Callable[[], EventEnvelope],
) -> None:
    expected_type, _, expected_route = VOICE_STREAMING_EVENTS[event_name]

    route = classify_event(builder_case())

    assert route["status"] == "routed"
    assert route["route"] == expected_route
    assert route["eventName"] == event_name
    assert route["eventType"] == expected_type
    assert route["knownEvent"] is True


@pytest.mark.parametrize("event_name,builder_case", _voice_builder_cases())
def test_voice_streaming_required_content_fields_are_enforced(
    event_name: str,
    builder_case: Callable[[], EventEnvelope],
) -> None:
    payload = builder_case().to_dict()
    required_field = VOICE_REQUIRED_FIELDS[event_name][0]
    del payload["content"][required_field]

    issues = validate_event_strict(payload, known_event_required=True)

    by_path = _issue_by_path(issues)
    assert by_path[f"content.{required_field}"].code == "missing_content_field"


def test_existing_dialogue_asr_events_remain_compatible() -> None:
    event = build_asr_event(
        source=HEAD_SOURCE,
        text="legacy partial",
        final=False,
        session_id="session-1",
        round_id="round-1",
    )

    route = classify_event(event)

    assert event.name == "ei.dialogue.asr.partial"
    assert route["status"] == "routed"
    assert route["route"] == "asr_partial"
