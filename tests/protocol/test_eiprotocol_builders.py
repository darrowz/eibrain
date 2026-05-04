from __future__ import annotations

from datetime import datetime, timezone


def _ids(*suffixes: str):
    from eiprotocol.builders import EventIdFactory

    values = iter(suffixes)
    return EventIdFactory(clock=lambda: "2026-05-04T10:30:20.123+08:00", id_factory=lambda: next(values))


def test_event_id_factory_generates_protocol_prefixes_and_clock_time() -> None:
    from eiprotocol.builders import EventIdFactory

    fixed_time = datetime(2026, 5, 4, 10, 30, 20, 123000, tzinfo=timezone.utc)
    ids = EventIdFactory(clock=lambda: fixed_time, id_factory=iter(["a", "b", "c", "d"]).__next__)

    assert ids.event_id() == "evt_a"
    assert ids.request_id() == "req_b"
    assert ids.round_id() == "rnd_c"
    assert ids.trace_id() == "trc_d"
    assert ids.time() == "2026-05-04T10:30:20.123+00:00"


def test_build_event_defaults_required_envelope_fields_and_accepts_dict_refs() -> None:
    from eiprotocol import EventEnvelope, validate_event
    from eiprotocol.builders import build_event

    event = build_event(
        ids=_ids("event", "request"),
        name="ei.control.ping",
        source={"domain": "eibrain", "instanceId": "honxin"},
        target={"domain": "eihead", "instanceId": "honjia"},
        content={"pingId": "ping_001", "sentAt": "2026-05-04T10:30:20.123+08:00"},
    )

    assert isinstance(event, EventEnvelope)
    assert event.event_id == "evt_event"
    assert event.request_id == "req_request"
    assert event.spec_version == "eiprotocol/0.1"
    assert event.event_type == "control"
    assert event.sequence == 1
    assert event.priority == "normal"
    assert event.policy.decision == "not_required"
    assert event.source.instance_id == "honxin"
    assert event.target is not None
    assert event.target.instance_id == "honjia"
    assert validate_event(event) == []


def test_build_event_rejects_known_events_missing_catalog_content_fields() -> None:
    import pytest

    from eiprotocol.builders import build_event

    with pytest.raises(ValueError) as exc_info:
        build_event(
            ids=_ids("event", "request"),
            name="ei.control.ping",
            source={"domain": "eibrain", "instanceId": "honxin"},
            content={"nonce": "legacy"},
        )

    message = str(exc_info.value)
    assert "missing_content_field at content.pingId" in message
    assert "missing_content_field at content.sentAt" in message


def test_build_event_generates_round_ids_for_round_scoped_events() -> None:
    from eiprotocol import SourceRef, validate_event
    from eiprotocol.builders import build_event

    event = build_event(
        ids=_ids("event", "request", "round"),
        name="ei.dialogue.agent.delta",
        source=SourceRef(domain="eibrain", instance_id="honxin"),
        content={"delta": "hello"},
    )

    assert event.event_type == "dialogue"
    assert event.round_id == "rnd_round"
    assert validate_event(event) == []


def test_action_request_builder_sets_required_content_and_defaults_round_scoped() -> None:
    from eiprotocol import validate_event
    from eiprotocol.builders import build_action_request_event

    event = build_action_request_event(
        ids=_ids("event", "request", "round"),
        source={"domain": "eibrain", "instanceId": "honxin"},
        action_id="act_speak_001",
        action_type="speak",
        target="speaker.default",
        params={"text": "I am listening."},
        risk_level="L1",
    )

    assert event.name == "ei.action.request"
    assert event.event_type == "action"
    assert event.round_id == "rnd_round"
    assert event.priority == "high"
    assert event.policy.decision == "not_required"
    assert event.policy.risk_level == "L1"
    assert event.content == {
        "actionId": "act_speak_001",
        "actionType": "speak",
        "target": "speaker.default",
        "params": {"text": "I am listening."},
        "riskLevel": "L1",
        "idempotencyKey": "act_speak_001",
    }
    assert validate_event(event) == []


def test_asr_builder_sets_final_or_partial_name_and_realtime_priority() -> None:
    from eiprotocol import validate_event
    from eiprotocol.builders import build_asr_event

    final = build_asr_event(
        ids=_ids("final-event", "final-request", "final-round"),
        source={"domain": "eihead", "instanceId": "honjia"},
        text="hello hongtu",
        final=True,
        confidence=0.83,
    )
    partial = build_asr_event(
        ids=_ids("partial-event", "partial-request", "partial-round"),
        source={"domain": "eihead", "instanceId": "honjia"},
        text="hello",
        final=False,
    )

    assert final.name == "ei.dialogue.asr.final"
    assert final.priority == "realtime"
    assert final.content["final"] is True
    assert partial.name == "ei.dialogue.asr.partial"
    assert partial.priority == "realtime"
    assert partial.content["final"] is False
    assert validate_event(final) == []
    assert validate_event(partial) == []


def test_vision_frame_and_execution_outcome_builders_validate() -> None:
    from eiprotocol import validate_event
    from eiprotocol.builders import build_execution_outcome_event, build_vision_frame_event

    vision = build_vision_frame_event(
        ids=_ids("vision-event", "vision-request"),
        source={"domain": "eihead", "instanceId": "honjia"},
        frame_id="frame_42",
        width=1280,
        height=720,
        detections=[{"label": "person", "score": 0.91, "bbox": [0.2, 0.1, 0.4, 0.6]}],
    )
    outcome = build_execution_outcome_event(
        ids=_ids("outcome-event", "outcome-request", "outcome-round"),
        source={"domain": "eihead", "instanceId": "honjia"},
        outcome_id="out_001",
        action_id="act_speak_001",
        action_type="speak",
        did_what=["queued speech", "played audio"],
    )

    assert vision.name == "ei.observation.vision.frame"
    assert vision.priority == "realtime"
    assert vision.content["detections"][0]["bbox"] == [0.2, 0.1, 0.4, 0.6]
    assert outcome.name == "ei.outcome.execution"
    assert outcome.round_id == "rnd_outcome-round"
    assert outcome.content["didWhat"] == ["queued speech", "played audio"]
    assert validate_event(vision) == []
    assert validate_event(outcome) == []
