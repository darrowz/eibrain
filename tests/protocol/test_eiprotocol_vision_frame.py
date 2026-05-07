from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "eiprotocol"


def _assert_strict_round_trip_and_route(event: Any) -> dict[str, Any]:
    from eiprotocol.codec import dumps_event, loads_event
    from eiprotocol.event_routing import classify_event
    from eiprotocol.validation import validate_event_strict

    payload = event.to_dict()

    assert validate_event_strict(payload, known_event_required=True) == []
    assert loads_event(dumps_event(event)).to_dict() == payload
    route = classify_event(event)
    assert route["status"] == "routed"
    assert route["route"] == "realtime_vision_frame"
    return payload


def test_realtime_vision_observation_typed_roundtrip_stabilizes_list_and_dict_bbox() -> None:
    from eiprotocol import Detection, RealtimeVisionObservation

    observation = RealtimeVisionObservation(
        frame_id="frame-typed-1",
        width=1280,
        height=720,
        frame_age_ms=24.5,
        backend="hailo",
        detections=[
            Detection(
                label="person",
                score=0.91,
                bbox={"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6},
                track_id="track-person-1",
                pose={"keypoints": [{"name": "nose", "x": 0.41, "y": 0.22, "score": 0.88}]},
                clip_labels=[{"label": "person looking at camera", "score": 0.83}],
                semantic_labels=[{"label": "human", "confidence": 0.94}],
                depth={"unit": "m", "median": 1.75},
                distance={"fromCameraM": 1.75},
                tracking_diagnostics={"ageFrames": 12, "lostFrames": 0},
            ),
            Detection(
                label="cat",
                score=0.73,
                bbox=[0.6, 0.3, 0.2, 0.2],
                track_id="track-cat-1",
            ),
        ],
        boxes=[
            {"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6},
            [0.6, 0.3, 0.2, 0.2],
        ],
        scores=[0.91, 0.73],
        tracked_target={"label": "person", "trackId": "track-person-1"},
        latency_ms={"capture": 8.0, "detect": 15.0},
        tracking_diagnostics={"tracker": "bytetrack", "activeTracks": 2},
        pose={"camera": {"roll": 0.0, "pitch": -2.5, "yaw": 1.0}},
        clip_labels=[{"label": "person and cat indoors", "score": 0.82}],
        semantic_labels=[{"label": "home_scene", "confidence": 0.8}],
        depth={"backend": "stereo", "unit": "m", "min": 0.8, "max": 4.2},
        distance={"trackedTargetM": 1.75, "nearestObjectM": 1.2},
        metadata={"camera": "front"},
    )

    content = observation.to_content()
    restored = RealtimeVisionObservation.from_content(content)

    assert content["detections"][0]["bbox"] == {"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6}
    assert content["detections"][0]["pose"]["keypoints"][0]["name"] == "nose"
    assert content["trackingDiagnostics"]["tracker"] == "bytetrack"
    assert content["clipLabels"][0]["label"] == "person and cat indoors"
    assert content["depth"]["backend"] == "stereo"
    assert content["detections"][1]["bbox"] == [0.6, 0.3, 0.2, 0.2]
    assert restored.to_content() == content


def test_realtime_vision_builder_emits_optional_diagnostics_and_distance_contracts() -> None:
    from eiprotocol import build_vision_frame_event

    event = build_vision_frame_event(
        source={"domain": "eihead", "instanceId": "honjia"},
        target={"domain": "eibrain", "instanceId": "honxin"},
        frame_id="frame-builder-1",
        detections=[
            {
                "label": "person",
                "score": 0.91,
                "bbox": {"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6},
                "trackId": "track-person-1",
                "pose": {"keypoints": [{"name": "left_eye", "x": 0.38, "y": 0.2, "score": 0.9}]},
                "clipLabels": [{"label": "person facing camera", "score": 0.84}],
                "semanticLabels": [{"label": "human", "confidence": 0.94}],
                "depth": {"unit": "m", "median": 1.7},
                "distance": {"fromCameraM": 1.7},
                "trackingDiagnostics": {"ageFrames": 7, "lostFrames": 0},
            }
        ],
        tracking_diagnostics={"tracker": "bytetrack", "activeTracks": 1},
        pose={"camera": {"roll": 0.0, "pitch": -1.0, "yaw": 0.5}},
        clip_labels=[{"label": "person at desk", "score": 0.81}],
        semantic_labels=[{"label": "workspace", "confidence": 0.79}],
        depth={"backend": "mono_depth", "unit": "m", "median": 2.0},
        distance={"trackedTargetM": 1.7},
        event_id="evt_frame_builder_001",
        request_id="req_frame_builder_001",
        time="2026-05-05T09:42:00.000+08:00",
    )

    payload = _assert_strict_round_trip_and_route(event)

    assert payload["content"]["trackingDiagnostics"]["activeTracks"] == 1
    assert payload["content"]["detections"][0]["distance"]["fromCameraM"] == 1.7
    assert payload["content"]["pose"]["camera"]["pitch"] == -1.0


def test_realtime_vision_frame_accepts_confidence_only_detector_payloads() -> None:
    from eiprotocol import Detection, build_vision_frame_event

    builder_event = build_vision_frame_event(
        source={"domain": "eihead", "instanceId": "honjia"},
        target={"domain": "eibrain", "instanceId": "honxin"},
        frame_id="frame-confidence-builder",
        detections=[{"label": "person", "confidence": 0.91, "bbox": [0.62, 0.35, 0.12, 0.18]}],
        event_id="evt_confidence_builder",
        request_id="req_confidence_builder",
    )

    assert Detection.from_dict({"label": "person", "confidence": 0.93, "bbox": [0, 0, 1, 1]}).score == 0.93
    assert builder_event.content["detections"][0]["score"] == 0.91


def test_realtime_vision_observation_accepts_legacy_snake_case_optional_fields() -> None:
    from eiprotocol import Detection, RealtimeVisionObservation

    detection = Detection.from_dict(
        {
            "label": "person",
            "score": 0.91,
            "bbox": {"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6},
            "track_id": "track-person-1",
            "clip_labels": [{"label": "person facing camera", "score": 0.84}],
            "semantic_labels": [{"label": "human", "confidence": 0.94}],
            "tracking_diagnostics": {"ageFrames": 7},
            "depth": {"median": 1.7},
            "distance": {"fromCameraM": 1.7},
        }
    )
    observation = RealtimeVisionObservation.from_content(
        {
            "frame_id": "frame-snake-1",
            "detections": [detection.to_dict()],
            "tracking_diagnostics": {"activeTracks": 1},
            "clip_labels": [{"label": "person at desk", "score": 0.81}],
            "semantic_labels": [{"label": "workspace", "confidence": 0.79}],
            "depth": {"median": 2.0},
            "distance": {"trackedTargetM": 1.7},
        }
    )

    content = observation.to_content()

    assert content["frameId"] == "frame-snake-1"
    assert content["detections"][0]["trackId"] == "track-person-1"
    assert content["detections"][0]["clipLabels"][0]["label"] == "person facing camera"
    assert content["trackingDiagnostics"]["activeTracks"] == 1
    assert content["clipLabels"][0]["label"] == "person at desk"
    assert content["distance"]["trackedTargetM"] == 1.7


def test_realtime_vision_fixture_is_strict_valid_and_typed_roundtrips() -> None:
    from eiprotocol import RealtimeVisionObservation, require_event_definition
    from eiprotocol.codec import dumps_event, loads_event
    from eiprotocol.event_routing import classify_event
    from eiprotocol.validation import validate_event_strict

    payload = json.loads((FIXTURE_DIR / "realtime_vision_frame.json").read_text(encoding="utf-8"))

    definition = require_event_definition("ei.observation.vision.frame")
    assert definition.direction == "head_to_brain"
    assert validate_event_strict(payload, known_event_required=True) == []
    assert loads_event(dumps_event(payload)).to_dict() == payload
    assert classify_event(payload)["route"] == "realtime_vision_frame"
    assert RealtimeVisionObservation.from_content(payload["content"]).to_content() == payload["content"]

