from __future__ import annotations

from eibrain.cognition.vision_realtime import (
    RealtimeVisionSimulator,
    to_eiprotocol_event_contents,
    to_eiprotocol_scene_content,
)


def _det(label: str, bbox: tuple[float, float, float, float], confidence: float = 0.9) -> dict[str, object]:
    x_min, y_min, x_max, y_max = bbox
    return {
        "label": label,
        "confidence": confidence,
        "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
    }


def test_realtime_simulator_keeps_track_ids_stable_and_emits_motion_events() -> None:
    simulator = RealtimeVisionSimulator(move_threshold=0.08)

    first = simulator.update(
        frame_id="frame-001",
        observed_at="2026-05-05T10:00:00.000+08:00",
        detections=[_det("cup", (0.10, 0.20, 0.20, 0.35))],
    )
    second = simulator.update(
        frame_id="frame-002",
        observed_at="2026-05-05T10:00:00.100+08:00",
        detections=[_det("cup", (0.32, 0.22, 0.42, 0.37))],
    )

    first_track = first["sceneSnapshot"]["objects"][0]["trackId"]
    second_track = second["sceneSnapshot"]["objects"][0]["trackId"]
    appeared = [event for event in first["events"] if event["eventType"] == "appeared"]

    assert first_track == second_track
    assert appeared == [
        {
            "eventId": f"{first['sceneSnapshot']['sceneId']}:appeared:{first_track}",
            "eventType": "appeared",
            "observedAt": "2026-05-05T10:00:00.000+08:00",
            "sceneId": first["sceneSnapshot"]["sceneId"],
            "subject": {"trackId": first_track, "label": "cup"},
            "confidence": 0.9,
            "details": {"fromRegion": "", "toRegion": "left_middle", "distance": 0.0, "temporalState": "appeared"},
            "metadata": {"frameId": "frame-001"},
        }
    ]
    assert second["events"][0]["eventType"] == "moved"
    assert second["events"][0]["subject"]["trackId"] == first_track
    assert second["events"][0]["details"]["fromRegion"] == "left_middle"
    assert second["events"][0]["details"]["toRegion"] == "center_middle"
    assert second["sceneSnapshot"]["objects"][0]["temporalState"] == "moving"


def test_realtime_simulator_emits_attention_snapshot_relationships_and_disappearance() -> None:
    simulator = RealtimeVisionSimulator(max_missing_frames=0)
    simulator.update(
        frame_id="frame-001",
        observed_at="2026-05-05T10:00:00.000+08:00",
        detections=[
            _det("cup", (0.58, 0.45, 0.68, 0.58), 0.82),
            _det("person", (0.20, 0.30, 0.45, 0.92), 0.94),
        ],
    )

    snapshot = simulator.update(
        frame_id="frame-002",
        observed_at="2026-05-05T10:00:00.200+08:00",
        detections=[_det("person", (0.22, 0.31, 0.47, 0.93), 0.95)],
    )

    event_types = [event["eventType"] for event in snapshot["events"]]
    scene = snapshot["sceneSnapshot"]
    relations = {
        (item["subjectLabel"], item["relation"], item["objectLabel"])
        for item in scene["relationships"]
    }

    assert "disappeared" in event_types
    assert scene["attention"]["label"] == "person"
    assert scene["attention"]["trackId"].startswith("person-")
    assert scene["stableTarget"]["trackId"] == scene["attention"]["trackId"]
    assert ("person", "left_of", "cup") in relations or scene["relationships"] == []
    assert "person" in snapshot["sceneGraphSummary"]
    assert scene["metadata"]["frameId"] == "frame-002"
    assert scene["metadata"]["stableTargetTrackId"] == scene["attention"]["trackId"]


def test_realtime_simulator_maps_snapshots_to_eiprotocol_friendly_content() -> None:
    simulator = RealtimeVisionSimulator()

    snapshot = simulator.update(
        frame_id="frame-001",
        observed_at="2026-05-05T10:00:00.000+08:00",
        detections=[
            _det("person", (0.10, 0.10, 0.30, 0.55), 0.91),
            _det("book", (0.66, 0.58, 0.82, 0.72), 0.86),
        ],
    )

    scene_content = to_eiprotocol_scene_content(snapshot)
    event_contents = to_eiprotocol_event_contents(snapshot)

    assert scene_content["sceneId"] == snapshot["sceneSnapshot"]["sceneId"]
    assert scene_content["observedAt"] == "2026-05-05T10:00:00.000+08:00"
    assert scene_content["summary"] == snapshot["sceneGraphSummary"]
    assert {item["trackId"] for item in scene_content["objects"]} == {
        item["trackId"] for item in snapshot["sceneSnapshot"]["objects"]
    }
    assert scene_content["metadata"]["realtime"] is True
    assert event_contents
    assert event_contents[0]["sceneId"] == scene_content["sceneId"]
    assert event_contents[0]["subject"]["trackId"]


def test_realtime_simulator_accepts_protocol_bbox_and_exports_multimodal_fields() -> None:
    simulator = RealtimeVisionSimulator()

    snapshot = simulator.update(
        frame_id="frame-protocol",
        observed_at="2026-05-05T10:00:00.000+08:00",
        detections=[
            {
                "label": "person",
                "confidence": 0.91,
                "bbox": {"x": 0.10, "y": 0.10, "w": 0.20, "h": 0.45},
                "clipLabels": [{"label": "person at desk", "score": 0.84}],
                "semanticLabels": [{"label": "workspace", "confidence": 0.79}],
                "depth": {"median": 0.72, "unit": "m"},
                "distance": {"fromCameraM": 0.72},
                "pose": {"lookingAtDevice": True},
            }
        ],
    )

    scene_content = to_eiprotocol_scene_content(snapshot)
    event_content = to_eiprotocol_event_contents(snapshot)[0]

    assert snapshot["sceneSnapshot"]["objects"][0]["bbox"] == {
        "x_min": 0.1,
        "y_min": 0.1,
        "x_max": 0.3,
        "y_max": 0.55,
    }
    assert scene_content["clipLabels"][0]["label"] == "person at desk"
    assert scene_content["depth"]["median"] == 0.72
    assert scene_content["distance"]["fromCameraM"] == 0.72
    assert event_content["pose"]["lookingAtDevice"] is True


def test_realtime_simulator_accepts_protocol_list_bbox_and_tracking_diagnostics() -> None:
    simulator = RealtimeVisionSimulator()

    snapshot = simulator.update(
        frame_id="frame-list-bbox",
        observed_at="2026-05-05T10:00:00.000+08:00",
        detections=[
            {
                "label": "person",
                "score": 0.91,
                "bbox": [0.62, 0.35, 0.12, 0.18],
                "trackingDiagnostics": {
                    "trackIdSwitchCount": 0,
                    "targetStabilityRatio": 1.0,
                },
            }
        ],
    )

    scene_content = to_eiprotocol_scene_content(snapshot)
    event_content = to_eiprotocol_event_contents(snapshot)[0]

    assert snapshot["sceneSnapshot"]["objects"][0]["bbox"] == {
        "x_min": 0.62,
        "y_min": 0.35,
        "x_max": 0.74,
        "y_max": 0.53,
    }
    assert snapshot["sceneSnapshot"]["objects"][0]["trackingDiagnostics"]["trackIdSwitchCount"] == 0
    assert scene_content["trackingDiagnostics"]["targetStabilityRatio"] == 1.0
    assert event_content["trackingDiagnostics"]["trackIdSwitchCount"] == 0


def test_realtime_simulator_honors_explicit_normalized_list_xyxy_bbox() -> None:
    simulator = RealtimeVisionSimulator()

    snapshot = simulator.update(
        frame_id="frame-list-xyxy",
        observed_at="2026-05-05T10:00:00.000+08:00",
        detections=[
            {
                "label": "person",
                "score": 0.91,
                "bbox": [0.62, 0.35, 0.74, 0.53],
                "bboxFormat": "xyxy",
            }
        ],
    )

    assert snapshot["sceneSnapshot"]["objects"][0]["bbox"] == {
        "x_min": 0.62,
        "y_min": 0.35,
        "x_max": 0.74,
        "y_max": 0.53,
    }
    assert snapshot["sceneSnapshot"]["objects"][0]["center"] == {"x": 0.68, "y": 0.44}


def test_realtime_simulator_replays_detection_frames_without_raw_frame_dump() -> None:
    simulator = RealtimeVisionSimulator(move_threshold=0.05, max_missing_frames=0)

    snapshots = simulator.replay(
        [
            {
                "frame_id": "frame-001",
                "observed_at": "2026-05-05T10:00:00.000+08:00",
                "detections": [_det("cup", (0.10, 0.20, 0.20, 0.35))],
            },
            {
                "frame_id": "frame-002",
                "observed_at": "2026-05-05T10:00:00.100+08:00",
                "detections": [_det("cup", (0.30, 0.20, 0.40, 0.35))],
            },
            {
                "frame_id": "frame-003",
                "observed_at": "2026-05-05T10:00:00.200+08:00",
                "detections": [],
            },
        ]
    )

    assert len(snapshots) == 3
    assert snapshots[0]["sceneSnapshot"]["objects"][0]["trackId"] == snapshots[1]["sceneSnapshot"]["objects"][0]["trackId"]
    assert [event["eventType"] for event in snapshots[1]["events"]] == ["moved"]
    assert [event["eventType"] for event in snapshots[2]["events"]] == ["disappeared"]
    assert "detections" not in snapshots[1]["sceneSnapshot"]


def test_realtime_simulator_keeps_track_through_jitter_and_single_missing_frame() -> None:
    simulator = RealtimeVisionSimulator(move_threshold=0.08, max_missing_frames=1)

    first = simulator.update(
        frame_id="frame-001",
        observed_at="2026-05-05T10:00:00.000+08:00",
        detections=[_det("person", (0.40, 0.20, 0.60, 0.80), 0.91)],
    )
    missing = simulator.update(
        frame_id="frame-002",
        observed_at="2026-05-05T10:00:00.100+08:00",
        detections=[],
    )
    recovered = simulator.update(
        frame_id="frame-003",
        observed_at="2026-05-05T10:00:00.200+08:00",
        detections=[_det("person", (0.405, 0.205, 0.605, 0.805), 0.9)],
    )

    track_id = first["sceneSnapshot"]["objects"][0]["trackId"]

    assert missing["events"] == []
    assert recovered["sceneSnapshot"]["objects"][0]["trackId"] == track_id
    assert recovered["sceneSnapshot"]["objects"][0]["temporalState"] == "stationary"
    assert recovered["sceneSnapshot"]["stableTarget"]["trackId"] == track_id
    assert [event["eventType"] for event in recovered["events"]] == []


def test_realtime_simulator_uses_hysteresis_to_avoid_attention_switch_from_score_jitter() -> None:
    simulator = RealtimeVisionSimulator(attention_switch_margin=0.25)

    first = simulator.update(
        frame_id="frame-001",
        observed_at="2026-05-05T10:00:00.000+08:00",
        detections=[
            _det("person", (0.10, 0.20, 0.30, 0.80), 0.93),
            _det("person", (0.65, 0.20, 0.85, 0.80), 0.90),
        ],
    )
    jitter = simulator.update(
        frame_id="frame-002",
        observed_at="2026-05-05T10:00:00.100+08:00",
        detections=[
            _det("person", (0.105, 0.20, 0.305, 0.80), 0.89),
            _det("person", (0.65, 0.20, 0.85, 0.80), 0.92),
        ],
    )

    assert jitter["sceneSnapshot"]["attention"]["trackId"] == first["sceneSnapshot"]["attention"]["trackId"]
    assert jitter["sceneSnapshot"]["stableTarget"]["trackId"] == first["sceneSnapshot"]["stableTarget"]["trackId"]
    assert "attention" not in [event["eventType"] for event in jitter["events"]]


def test_realtime_simulator_requires_persistent_candidate_before_target_swap() -> None:
    simulator = RealtimeVisionSimulator(
        attention_switch_margin=0.10,
        attention_switch_cooldown_frames=2,
    )

    first = simulator.update(
        frame_id="frame-001",
        observed_at="2026-05-05T10:00:00.000+08:00",
        detections=[
            _det("person", (0.10, 0.20, 0.32, 0.82), 0.94),
            _det("person", (0.66, 0.24, 0.84, 0.76), 0.86),
        ],
    )
    first_target = first["sceneSnapshot"]["stableTarget"]["trackId"]

    candidate_spike = simulator.update(
        frame_id="frame-002",
        observed_at="2026-05-05T10:00:00.100+08:00",
        detections=[
            _det("person", (0.10, 0.20, 0.32, 0.82), 0.90),
            _det("person", (0.62, 0.18, 0.88, 0.84), 0.96),
        ],
    )
    persistent_candidate = simulator.update(
        frame_id="frame-003",
        observed_at="2026-05-05T10:00:00.200+08:00",
        detections=[
            _det("person", (0.10, 0.20, 0.32, 0.82), 0.90),
            _det("person", (0.62, 0.18, 0.88, 0.84), 0.97),
        ],
    )

    assert candidate_spike["sceneSnapshot"]["stableTarget"]["trackId"] == first_target
    assert persistent_candidate["sceneSnapshot"]["stableTarget"]["trackId"] != first_target
    assert [event["eventType"] for event in candidate_spike["events"] if event["eventType"] == "attention_changed"] == []
    assert [event["eventType"] for event in persistent_candidate["events"] if event["eventType"] == "attention_changed"] == [
        "attention_changed"
    ]
