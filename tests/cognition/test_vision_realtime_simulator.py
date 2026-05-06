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
