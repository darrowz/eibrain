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
        clip_labels=[{"label": "person at desk", "score": 0.84}],
        semantic_labels=[{"label": "workspace", "confidence": 0.79}],
        depth={"unit": "m", "median": 1.8},
        distance={"nearestObjectId": "track_person_1", "meters": 1.7},
        scene_graph={
            "nodes": [{"id": "track_person_1", "label": "person"}],
            "edges": [{"subjectId": "track_person_1", "relation": "near", "objectId": "desk_1"}],
        },
        scene_graph_provenance={"sourceFrameIds": ["frame_000042"], "builder": "scene_graph_v2"},
        metadata={"camera": "front"},
    )
    vision_event = VisionEventObservation(
        event_id="vevt_person_entered_001",
        event_type="person_entered",
        observed_at="2026-05-05T09:42:01.000+08:00",
        scene_id="scene_front_001",
        subject={"trackId": "track_person_1"},
        confidence=0.88,
        pose={"standing": 0.82},
        clip_labels=[{"label": "person entering room", "score": 0.86}],
        semantic_labels=[{"label": "arrival", "confidence": 0.81}],
        depth={"unit": "m", "subjectMedian": 1.9},
        distance={"fromCameraM": 1.9, "fromZone": "desk"},
        scene_graph_provenance={"sceneId": "scene_front_001", "edgeIds": ["edge_person_near_desk"]},
        metadata={"source": "vision"},
    )
    memory_policy = MemoryPolicyReport(
        policy_id="mpol_recall_privacy_001",
        scope={"roundId": "rnd_voice_001", "memoryTypes": ["preference"]},
        decision="allow",
        reason="round context allows recall",
        filters=[{"field": "memoryType", "op": "in", "values": ["preference"]}],
        conflict_resolution={"strategy": "prefer_recent_confirmed", "winnerMemoryId": "mem_pref_001"},
        persona_consistency_signals=[{"signal": "tone_match", "score": 0.92}],
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
        clip_labels=[{"label": "person at desk", "score": 0.84}],
        semantic_labels=[{"label": "workspace", "confidence": 0.79}],
        depth={"unit": "m", "median": 1.8},
        distance={"nearestObjectId": "track_person_1", "meters": 1.7},
        scene_graph={"nodes": [{"id": "track_person_1", "label": "person"}]},
        scene_graph_provenance={"sourceFrameIds": ["frame_000042"], "builder": "scene_graph_v2"},
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
        pose={"standing": 0.82},
        clip_labels=[{"label": "person entering room", "score": 0.86}],
        semantic_labels=[{"label": "arrival", "confidence": 0.81}],
        depth={"unit": "m", "subjectMedian": 1.9},
        distance={"fromCameraM": 1.9, "fromZone": "desk"},
        scene_graph_provenance={"sceneId": "scene_front_001", "edgeIds": ["edge_person_near_desk"]},
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
        filters=[{"field": "memoryType", "op": "in", "values": ["preference"]}],
        conflict_resolution={"strategy": "prefer_recent_confirmed", "winnerMemoryId": "mem_pref_001"},
        persona_consistency_signals=[{"signal": "tone_match", "score": 0.92}],
        event_id="evt_memory_policy_report_001",
        request_id="req_memory_policy_report_001",
        round_id="rnd_voice_001",
        time="2026-05-05T09:42:02.000+08:00",
    )

    assert scene_event.round_id == ""
    assert vision_event.round_id == ""
    assert policy_event.round_id == "rnd_voice_001"
    assert scene_event.content["sceneGraphProvenance"]["builder"] == "scene_graph_v2"
    assert vision_event.content["clipLabels"][0]["label"] == "person entering room"
    assert policy_event.content["conflictResolution"]["strategy"] == "prefer_recent_confirmed"

    expected_routes = [
        (scene_event, "vision_scene"),
        (vision_event, "vision_event"),
        (policy_event, "memory_policy_report"),
    ]
    for event, expected_route in expected_routes:
        assert validate_event_strict(event, known_event_required=True) == []
        assert loads_event(dumps_event(event)).to_dict() == event.to_dict()
        assert classify_event(event)["route"] == expected_route


def test_v011_memory_policy_report_preserves_writes_across_model_builder_and_catalog() -> None:
    writes = [{"memoryId": "mem_pref_001", "status": "proposed", "operation": "upsert"}]
    report = MemoryPolicyReport(
        policy_id="mpol_write_001",
        scope={"roundId": "rnd_voice_001"},
        decision="allow",
        reason="user confirmed preference",
        writes=writes,
    )

    content = report.to_content()
    restored = MemoryPolicyReport.from_content(content)
    event = build_memory_policy_report_event(
        source=_source("eibrain"),
        target=_target("eimemory"),
        policy_id="mpol_write_001",
        scope={"roundId": "rnd_voice_001"},
        decision="allow",
        writes=writes,
        event_id="evt_memory_policy_writes_001",
        request_id="req_memory_policy_writes_001",
        round_id="rnd_voice_001",
        time="2026-05-05T09:42:02.100+08:00",
    )
    definition = require_event_definition("ei.memory.policy.report")

    assert content["writes"] == writes
    assert restored == report
    assert event.content["writes"] == writes
    assert "writes" in definition.optional_content_fields
    assert validate_event_strict(event, known_event_required=True) == []
    assert loads_event(dumps_event(event)).content["writes"] == writes


def test_v011_vision_observation_to_event_matches_non_round_scoped_catalog_semantics() -> None:
    scene = VisionSceneObservation(
        scene_id="scene_direct_001",
        observed_at="2026-05-05T09:42:00.100+08:00",
    )
    vision_event = VisionEventObservation(
        event_id="vevt_direct_001",
        event_type="object_entered",
        observed_at="2026-05-05T09:42:01.000+08:00",
        scene_id="scene_direct_001",
    )

    scene_envelope = scene.to_event(
        event_id="evt_scene_direct_001",
        request_id="req_scene_direct_001",
        sequence=1,
        source=_source(),
        target=_target("eibrain"),
        time="2026-05-05T09:42:00.110+08:00",
        session_id="session_vision_001",
    )
    event_envelope = vision_event.to_event(
        event_id="evt_vision_direct_001",
        request_id="req_vision_direct_001",
        sequence=2,
        source=_source(),
        target=_target("eibrain"),
        time="2026-05-05T09:42:01.010+08:00",
        session_id="session_vision_001",
    )

    assert require_event_definition(scene_envelope.name).round_scoped is False
    assert require_event_definition(event_envelope.name).round_scoped is False
    assert scene_envelope.session_id == "session_vision_001"
    assert event_envelope.session_id == "session_vision_001"
    assert scene_envelope.round_id == ""
    assert event_envelope.round_id == ""
    assert validate_event_strict(scene_envelope, known_event_required=True) == []
    assert validate_event_strict(event_envelope, known_event_required=True) == []


def test_v011_models_accept_legacy_snake_case_optional_fields() -> None:
    scene = VisionSceneObservation.from_content(
        {
            "scene_id": "scene_snake_001",
            "observed_at": "2026-05-05T09:42:00.100+08:00",
            "summary": "person at desk",
            "objects": [{"label": "person"}],
            "relationships": [{"subject": "person", "relation": "near", "object": "desk"}],
            "clip_labels": [{"label": "person at desk", "score": 0.84}],
            "semantic_labels": [{"label": "workspace", "confidence": 0.79}],
            "depth": {"median": 1.8},
            "distance": {"meters": 1.7},
            "scene_graph": {"nodes": [{"id": "track_person_1"}]},
            "scene_graph_provenance": {"builder": "scene_graph_v2"},
        }
    )
    event = VisionEventObservation.from_content(
        {
            "event_id": "vevt_snake_001",
            "event_type": "looking_at_device",
            "observed_at": "2026-05-05T09:42:01.000+08:00",
            "scene_id": "scene_snake_001",
            "subject": {"trackId": "track_person_1"},
            "pose": {"lookingAtDevice": True},
            "clip_labels": [{"label": "person looking at device", "score": 0.86}],
            "semantic_labels": [{"label": "attention", "confidence": 0.81}],
            "depth": {"subjectMedian": 1.9},
            "distance": {"fromCameraM": 1.9},
            "scene_graph_provenance": {"edgeIds": ["edge_person_near_screen"]},
        }
    )
    policy = MemoryPolicyReport.from_content(
        {
            "policy_id": "mpol_snake_001",
            "scope": {"roundId": "rnd_voice_001"},
            "decision": "defer",
            "conflict_resolution": {"strategy": "ask_user"},
            "persona_consistency_signals": [{"signal": "tone_match", "score": 0.92}],
        }
    )

    assert scene.to_content()["sceneGraph"]["nodes"][0]["id"] == "track_person_1"
    assert scene.to_content()["sceneGraphProvenance"]["builder"] == "scene_graph_v2"
    assert event.to_content()["sceneGraphProvenance"]["edgeIds"] == ["edge_person_near_screen"]
    assert event.to_content()["clipLabels"][0]["label"] == "person looking at device"
    assert policy.to_content()["conflictResolution"]["strategy"] == "ask_user"
    assert policy.to_content()["personaConsistencySignals"][0]["signal"] == "tone_match"


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


def test_v011_accepts_realtime_vision_simulator_observation_content() -> None:
    scene_content = {
        "sceneId": "scene_rt_001",
        "observedAt": "2026-05-05T10:15:00.000+08:00",
        "summary": "Observed person, cup; realtime events: appeared, attention",
        "objects": [
            {
                "trackId": "person-001",
                "label": "person",
                "confidence": 0.93,
                "bbox": {"x_min": 0.12, "y_min": 0.15, "x_max": 0.32, "y_max": 0.70},
                "region": "left_middle",
            },
            {
                "trackId": "cup-001",
                "label": "cup",
                "confidence": 0.81,
                "bbox": {"x_min": 0.62, "y_min": 0.45, "x_max": 0.72, "y_max": 0.58},
                "region": "right_middle",
            },
        ],
        "relationships": [{"subjectId": "person-001", "relation": "left_of", "objectId": "cup-001"}],
        "environment": {"source": "realtime_vision_simulator"},
        "metadata": {"realtime": True, "simulator": "software", "trackCount": 2},
    }
    event_content = {
        "eventId": "scene_rt_001:appeared:person-001",
        "eventType": "appeared",
        "observedAt": "2026-05-05T10:15:00.000+08:00",
        "sceneId": "scene_rt_001",
        "subject": {"trackId": "person-001", "label": "person"},
        "confidence": 0.93,
        "details": {"fromRegion": "", "toRegion": "left_middle", "distance": 0.0},
        "metadata": {"frameId": "frame-rt-001"},
    }

    scene_event = build_vision_scene_event(
        source=_source(),
        target=_target("eibrain"),
        scene=scene_content,
        event_id="evt_realtime_vision_scene_001",
        request_id="req_realtime_vision_scene_001",
        time="2026-05-05T10:15:00.010+08:00",
    )
    vision_event = build_vision_event_event(
        source=_source(),
        target=_target("eibrain"),
        event_id=event_content["eventId"],
        event_type=event_content["eventType"],
        observed_at=event_content["observedAt"],
        scene_id=event_content["sceneId"],
        subject=event_content["subject"],
        confidence=event_content["confidence"],
        details=event_content["details"],
        metadata=event_content["metadata"],
        protocol_event_id="evt_realtime_vision_event_001",
        request_id="req_realtime_vision_event_001",
        time="2026-05-05T10:15:00.020+08:00",
    )

    assert scene_content["metadata"]["realtime"] is True
    assert scene_event.content["objects"][0]["trackId"]
    assert validate_event_strict(scene_event, known_event_required=True) == []
    assert validate_event_strict(vision_event, known_event_required=True) == []
    assert loads_event(dumps_event(scene_event)).to_dict() == scene_event.to_dict()
    assert classify_event(scene_event)["route"] == "vision_scene"
    assert classify_event(vision_event)["route"] == "vision_event"


def test_v011_vision_scene_and_event_builders_preserve_realtime_tracking_fields() -> None:
    scene_event = build_vision_scene_event(
        source=_source(),
        target=_target("eibrain"),
        scene={
            "sceneId": "scene-rich-001",
            "observedAt": "2026-05-06T10:00:00.000+08:00",
            "summary": "person near cup",
            "objects": [{"trackId": "person-1", "label": "person"}],
            "attention": {"trackId": "person-1"},
            "stableTarget": {"trackId": "person-1", "stableFrames": 12},
            "eventSummary": "hand_near_object",
            "trackingDiagnostics": {"activeTracks": 2, "stabilityRatio": 0.96},
            "temporal": {"windowMs": 1500},
            "events": [{"eventId": "scene-rich-001:hand_near_object:person-1:cup-1"}],
            "clipLabels": ["person at desk"],
            "semanticLabels": ["attention"],
        },
        event_id="evt_scene_rich",
        request_id="req_scene_rich",
    )
    vision_event = build_vision_event_event(
        source=_source(),
        target=_target("eibrain"),
        event_id="vevt-rich-001",
        event_type="hand_near_object",
        observed_at="2026-05-06T10:00:00.010+08:00",
        event={
            "eventId": "vevt-rich-001",
            "eventType": "hand_near_object",
            "observedAt": "2026-05-06T10:00:00.010+08:00",
            "sceneId": "scene-rich-001",
            "subject": {"trackId": "person-1"},
            "trackingDiagnostics": {"targetStable": True},
            "clipLabels": ["hand near cup"],
            "semanticLabels": ["interaction"],
        },
        protocol_event_id="evt_vision_rich",
        request_id="req_vision_rich",
    )

    scene_content = scene_event.content
    event_content = vision_event.content
    assert scene_content["stableTarget"]["trackId"] == "person-1"
    assert scene_content["trackingDiagnostics"]["activeTracks"] == 2
    assert scene_content["clipLabels"] == [{"label": "person at desk"}]
    assert scene_content["semanticLabels"] == [{"label": "attention"}]
    assert scene_content["events"][0]["eventId"].endswith(":cup-1")
    assert event_content["trackingDiagnostics"]["targetStable"] is True
    assert event_content["clipLabels"] == [{"label": "hand near cup"}]
    assert event_content["semanticLabels"] == [{"label": "interaction"}]
    assert loads_event(dumps_event(scene_event)).to_dict() == scene_event.to_dict()
    assert loads_event(dumps_event(vision_event)).to_dict() == vision_event.to_dict()
