from __future__ import annotations

import json

import pytest

from eiprotocol.catalog import (
    EventDefinition,
    get_event_definition,
    is_known_event,
    list_event_names,
    require_event_definition,
)


CORE_EVENT_NAMES = {
    "ei.control.hello",
    "ei.control.ping",
    "ei.control.pong",
    "ei.control.resume",
    "ei.control.ack",
    "ei.control.error",
    "ei.capability.manifest.report",
    "ei.observation.audio.chunk",
    "ei.observation.vision.frame",
    "ei.observation.head.status.report",
    "ei.dialogue.asr.partial",
    "ei.dialogue.asr.final",
    "ei.dialogue.fast_hypothesis",
    "ei.dialogue.decision.stable",
    "ei.dialogue.agent.delta",
    "ei.dialogue.agent.final",
    "ei.dialogue.tts.delta",
    "ei.dialogue.tts.final",
    "ei.dialogue.interrupt.requested",
    "ei.action.request",
    "ei.action.dispatch",
    "ei.action.progress",
    "ei.action.complete",
    "ei.action.emergency.stop",
    "ei.policy.decision",
    "ei.memory.recall.request",
    "ei.memory.recall.result",
    "ei.memory.write.proposed",
    "ei.memory.write.committed",
    "ei.outcome.execution",
    "ei.outcome.user.feedback",
    "ei.training.signal",
}


def test_catalog_lists_core_v01_events_and_serializes_definitions() -> None:
    names = list_event_names()

    assert CORE_EVENT_NAMES <= set(names)
    for name in CORE_EVENT_NAMES:
        definition = get_event_definition(name)

        assert isinstance(definition, EventDefinition)
        assert is_known_event(name) is True
        payload = definition.to_dict()
        assert payload == {
            "name": definition.name,
            "event_type": definition.event_type,
            "plane": definition.plane,
            "direction": definition.direction,
            "realtime": definition.realtime,
            "round_scoped": definition.round_scoped,
            "side_effecting": definition.side_effecting,
            "required_content_fields": list(definition.required_content_fields),
            "description": definition.description,
        }
        json.dumps(payload)


def test_list_event_names_filters_by_event_type_and_plane() -> None:
    assert list_event_names(event_type="action") == [
        "ei.action.request",
        "ei.action.dispatch",
        "ei.action.progress",
        "ei.action.complete",
        "ei.action.emergency.stop",
    ]
    assert list_event_names(plane="dialogue") == [
        "ei.dialogue.asr.partial",
        "ei.dialogue.asr.final",
        "ei.voice.asr.partial",
        "ei.voice.asr.final",
        "ei.dialogue.fast_hypothesis",
        "ei.dialogue.decision.stable",
        "ei.dialogue.speech_action.plan",
        "ei.dialogue.cancellation.applied",
        "ei.dialogue.agent.delta",
        "ei.dialogue.agent.final",
        "ei.dialogue.tts.delta",
        "ei.dialogue.tts.final",
        "ei.voice.tts.sentence_start",
        "ei.voice.tts.chunk",
        "ei.voice.playback.started",
        "ei.voice.playback.stopped",
        "ei.voice.barge_in.detected",
        "ei.dialogue.interrupt.requested",
    ]
    assert list_event_names(event_type="memory", plane="memory") == [
        "ei.memory.recall.request",
        "ei.memory.prefetch.requested",
        "ei.memory.recall.result",
        "ei.memory.write.proposed",
        "ei.memory.write.committed",
    ]
    assert list_event_names(event_type="memory", plane="actuation") == []


def test_round_scoped_and_side_effecting_flags_are_queryable() -> None:
    assert require_event_definition("ei.control.hello").round_scoped is False
    assert require_event_definition("ei.dialogue.asr.final").round_scoped is True
    assert require_event_definition("ei.memory.recall.request").round_scoped is True

    assert require_event_definition("ei.action.request").side_effecting is True
    assert require_event_definition("ei.action.dispatch").side_effecting is True
    assert require_event_definition("ei.action.progress").side_effecting is False
    assert require_event_definition("ei.action.emergency.stop").side_effecting is True
    assert require_event_definition("ei.memory.write.proposed").side_effecting is False
    assert require_event_definition("ei.memory.write.committed").side_effecting is True


def test_required_content_fields_and_routing_metadata_are_documented() -> None:
    action = require_event_definition("ei.action.request")
    assert action.required_content_fields == ("actionId", "actionType", "target", "idempotencyKey")
    assert action.event_type == "action"
    assert action.plane == "action"
    assert action.direction == "brain_to_head"

    asr = require_event_definition("ei.dialogue.asr.partial")
    assert asr.required_content_fields == ("text", "final")
    assert asr.realtime is True
    assert asr.direction == "head_to_brain"

    fast_hypothesis = require_event_definition("ei.dialogue.fast_hypothesis")
    assert fast_hypothesis.required_content_fields == ("hypothesisId", "text", "confidence")
    assert fast_hypothesis.realtime is True
    assert fast_hypothesis.direction == "brain_to_head"

    stable_decision = require_event_definition("ei.dialogue.decision.stable")
    assert stable_decision.required_content_fields == ("decisionId", "decision", "confidence")
    assert stable_decision.round_scoped is True
    assert stable_decision.direction == "brain_to_head"

    manifest = require_event_definition("ei.capability.manifest.report")
    assert manifest.required_content_fields == ("manifestId", "manifestVersion", "capabilities")
    assert manifest.round_scoped is False

    audio = require_event_definition("ei.observation.audio.chunk")
    assert audio.required_content_fields == ("streamId", "chunkIndex", "audioBase64")
    assert audio.realtime is True

    head_status = require_event_definition("ei.observation.head.status.report")
    assert head_status.required_content_fields == ("status", "components", "reportedAt")
    assert head_status.round_scoped is False
    assert head_status.direction == "head_to_brain"


def test_unknown_lookup_is_optional_until_required() -> None:
    assert get_event_definition("ei.unknown.future") is None
    assert is_known_event("ei.unknown.future") is False
    assert list_event_names(event_type="unknown") == []

    with pytest.raises(KeyError) as exc_info:
        require_event_definition("ei.unknown.future")

    assert "ei.unknown.future" in str(exc_info.value)
