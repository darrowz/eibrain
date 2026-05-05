from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import eiprotocol
from eiprotocol import (
    EventEnvelope,
    MemoryPolicyReport,
    SourceRef,
    TargetRef,
    VisionEventObservation,
    VisionSceneObservation,
    build_memory_policy_report_event,
    build_vision_event_event,
    build_vision_scene_event,
    dumps_event,
    loads_event,
)
from eiprotocol.catalog import require_event_definition
from eiprotocol.event_routing import classify_event
from eiprotocol.validation import validate_event_strict


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "eiprotocol"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _source(domain: str = "eihead") -> SourceRef:
    return SourceRef(domain=domain, instance_id="honjia")


def _target(domain: str) -> TargetRef:
    return TargetRef(domain=domain, instance_id="honxin")


def test_v011_gap_events_are_registered_with_protocol_metadata() -> None:
    cases = [
        (
            "ei.observation.vision.scene",
            "observation",
            "observation",
            "head_to_brain",
            True,
            False,
            ("sceneId", "observedAt"),
        ),
        (
            "ei.observation.vision.event",
            "observation",
            "observation",
            "head_to_brain",
            True,
            False,
            ("eventId", "eventType", "observedAt"),
        ),
        (
            "ei.memory.policy.report",
            "memory",
            "policy",
            "brain_internal",
            False,
            True,
            ("policyId", "scope", "decision"),
        ),
    ]

    for name, event_type, plane, direction, realtime, round_scoped, required_fields in cases:
        definition = require_event_definition(name)

        assert definition.event_type == event_type
        assert definition.plane == plane
        assert definition.direction == direction
        assert definition.realtime is realtime
        assert definition.round_scoped is round_scoped
        assert definition.side_effecting is False
        assert set(required_fields) <= set(definition.required_content_fields)
        assert json.loads(json.dumps(definition.to_dict())) == definition.to_dict()


def test_v011_dataclasses_are_json_friendly_content_contracts() -> None:
    scene = VisionSceneObservation(
        scene_id="scene_front_001",
        observed_at="2026-05-05T09:42:00.100+08:00",
        summary="person at desk",
        objects=[{"label": "person", "confidence": 0.91}],
        relationships=[{"subject": "person", "relation": "near", "object": "desk"}],
        environment={"lighting": "indoor"},
        metadata={"camera": "front"},
    )
    vision_event = VisionEventObservation(
        event_id="vevt_person_entered_001",
        event_type="person_entered",
        observed_at="2026-05-05T09:42:01.000+08:00",
        scene_id="scene_front_001",
        subject={"trackId": "track_person_1"},
        confidence=0.88,
        metadata={"source": "vision"},
    )
    memory_policy = MemoryPolicyReport(
        policy_id="mpol_recall_privacy_001",
        scope={"roundId": "rnd_voice_001", "memoryTypes": ["preference"]},
        decision="allow",
        reason="round context allows recall",
        metadata={"policyVersion": "2026-05-05"},
    )

    assert VisionSceneObservation.from_content(scene.to_content()) == scene
    assert VisionEventObservation.from_content(vision_event.to_content()) == vision_event
    assert MemoryPolicyReport.from_content(memory_policy.to_content()) == memory_policy
    json.dumps(scene.to_content())
    json.dumps(vision_event.to_content())
    json.dumps(memory_policy.to_content())


def test_v011_builders_create_strictly_valid_routable_events() -> None:
    scene_event = build_vision_scene_event(
        source=_source(),
        target=_target("eibrain"),
        scene_id="scene_front_001",
        observed_at="2026-05-05T09:42:00.100+08:00",
        objects=[{"label": "person"}],
        event_id="evt_vision_scene_001",
        request_id="req_vision_scene_001",
        time="2026-05-05T09:42:00.110+08:00",
    )
    vision_event = build_vision_event_event(
        source=_source(),
        target=_target("eibrain"),
        event_id="vevt_person_entered_001",
        event_type="person_entered",
        observed_at="2026-05-05T09:42:01.000+08:00",
        scene_id="scene_front_001",
        protocol_event_id="evt_vision_event_001",
        request_id="req_vision_event_001",
        time="2026-05-05T09:42:01.010+08:00",
    )
    policy_event = build_memory_policy_report_event(
        source=_source("eibrain"),
        target=_target("eimemory"),
        policy_id="mpol_recall_privacy_001",
        scope={"roundId": "rnd_voice_001", "memoryTypes": ["preference"]},
        decision="allow",
        event_id="evt_memory_policy_report_001",
        request_id="req_memory_policy_report_001",
        round_id="rnd_voice_001",
        time="2026-05-05T09:42:02.000+08:00",
    )

    assert scene_event.round_id == ""
    assert vision_event.round_id == ""
    assert policy_event.round_id == "rnd_voice_001"

    expected_routes = [
        (scene_event, "vision_scene"),
        (vision_event, "vision_event"),
        (policy_event, "memory_policy_report"),
    ]
    for event, expected_route in expected_routes:
        assert validate_event_strict(event, known_event_required=True) == []
        assert loads_event(dumps_event(event)).to_dict() == event.to_dict()
        assert classify_event(event)["route"] == expected_route


@pytest.mark.parametrize(
    ("fixture_name", "route_name"),
    [
        ("vision_scene.json", "vision_scene"),
        ("vision_event.json", "vision_event"),
        ("memory_policy.json", "memory_policy_report"),
    ],
)
def test_v011_fixtures_are_strictly_valid_and_roundtrip(fixture_name: str, route_name: str) -> None:
    payload = _fixture(fixture_name)

    restored = loads_event(dumps_event(payload))

    assert isinstance(restored, EventEnvelope)
    assert restored.to_dict() == payload
    assert validate_event_strict(payload, known_event_required=True) == []
    assert classify_event(payload)["route"] == route_name


@pytest.mark.parametrize(
    ("fixture_name", "missing_path"),
    [
        ("vision_scene.json", "content.observedAt"),
        ("vision_event.json", "content.eventType"),
        ("memory_policy.json", "content.decision"),
    ],
)
def test_v011_required_content_fields_are_strict(fixture_name: str, missing_path: str) -> None:
    payload = _fixture(fixture_name)
    _, field_name = missing_path.split(".")
    del payload["content"][field_name]

    issues = validate_event_strict(payload, known_event_required=True)

    assert any(issue.path == missing_path and issue.code == "missing_content_field" for issue in issues)


def test_v011_exports_are_available_from_package_root() -> None:
    assert eiprotocol.VisionSceneObservation is VisionSceneObservation
    assert eiprotocol.VisionEventObservation is VisionEventObservation
    assert eiprotocol.MemoryPolicyReport is MemoryPolicyReport
    assert eiprotocol.build_vision_scene_event is build_vision_scene_event
    assert eiprotocol.build_vision_event_event is build_vision_event_event
    assert eiprotocol.build_memory_policy_report_event is build_memory_policy_report_event
