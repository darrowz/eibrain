from __future__ import annotations

import pytest

from eihead.eye import RealtimeVisionSceneBridge


def _observation(
    *,
    frame_id: str,
    observed_at: str,
    detections: list[dict[str, object]],
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "realtime_vision_observation",
        "mode": "realtime_stream",
        "status": "ok",
        "stream_ready": True,
        "not_wired": False,
        "placeholder": False,
        "stale": False,
        "compatibility_mode": False,
        "frame_id": frame_id,
        "observed_at": observed_at,
        "detections": detections,
        "frame_payload": b"not-for-protocol",
        "image": "raw-image-data",
    }
    payload.update(overrides)
    return payload


def _det(label: str, bbox: tuple[float, float, float, float], confidence: float) -> dict[str, object]:
    x_min, y_min, x_max, y_max = bbox
    return {
        "label": label,
        "confidence": confidence,
        "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
    }


def test_scene_bridge_emits_protocol_scene_and_events_without_raw_frames() -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        _observation(
            frame_id="frame-001",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[_det("person", (0.10, 0.20, 0.30, 0.70), 0.91)],
        )
    )

    scene = result["scene_snapshot"]

    assert result["live"] is True
    assert result["latest_scene_id"] == scene["sceneId"]
    assert result["sceneGraphSummary"] == scene["summary"]
    assert result["object_count"] == 1
    assert result["track_count"] == 1
    assert scene["observedAt"] == "2026-05-05T10:00:00.000+08:00"
    assert scene["metadata"]["frameId"] == "frame-001"
    assert scene["objects"][0]["label"] == "person"
    assert scene["objects"][0]["confidence"] == 0.91
    assert scene["objects"][0]["bbox"] == {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.7}
    assert result["event_contents"]
    assert result["events"] == result["event_contents"]
    assert result["event_contents"][0]["sceneId"] == result["latest_scene_id"]
    assert "frame_payload" not in str(result)
    assert "raw-image-data" not in str(result)


def test_scene_bridge_keeps_simulator_track_ids_stable_across_frames() -> None:
    bridge = RealtimeVisionSceneBridge(move_threshold=0.08)

    first = bridge.update(
        _observation(
            frame_id="frame-001",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[_det("cup", (0.10, 0.20, 0.20, 0.35), 0.87)],
        )
    )
    second = bridge.update(
        _observation(
            frame_id="frame-002",
            observed_at="2026-05-05T10:00:00.100+08:00",
            detections=[_det("cup", (0.32, 0.22, 0.42, 0.37), 0.88)],
        )
    )

    first_track = first["scene_snapshot"]["objects"][0]["trackId"]
    second_track = second["scene_snapshot"]["objects"][0]["trackId"]

    assert first_track == second_track
    assert second["scene_snapshot"]["objects"][0]["label"] == "cup"
    assert second["scene_snapshot"]["objects"][0]["confidence"] == 0.88
    assert second["scene_snapshot"]["metadata"]["frameId"] == "frame-002"
    assert second["event_contents"][0]["eventType"] == "moved"
    assert second["event_contents"][0]["subject"]["trackId"] == first_track


def test_scene_bridge_accepts_latest_status_dicts_from_realtime_eye_service() -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        {
            "mode": "realtime_stream",
            "status": "ok",
            "stream_ready": True,
            "not_wired": False,
            "placeholder": False,
            "stale": False,
            "last_frame_id": "frame-003",
            "last_frame_captured_at_ts": 1777975200.125,
            "detections": [_det("book", (0.66, 0.58, 0.82, 0.72), 0.86)],
            "payload": "must-not-leak",
        }
    )

    assert result["live"] is True
    assert result["frame_id"] == "frame-003"
    assert result["scene_snapshot"]["observedAt"] == "2026-05-05T10:00:00.125+00:00"
    assert result["scene_snapshot"]["objects"][0]["label"] == "book"
    assert "must-not-leak" not in str(result)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"not_wired": True, "status": "not_wired", "stream_ready": False}, "not_wired"),
        ({"placeholder": True, "backend": "placeholder"}, "placeholder"),
        ({"mode": "compat_static_frame", "status": "compat_static", "compatibility_mode": True}, "compat_static"),
        ({"status": "static"}, "static"),
        ({"stale": True, "status": "stale"}, "stale"),
    ],
)
def test_scene_bridge_marks_non_live_observations_without_durable_events(
    overrides: dict[str, object],
    reason: str,
) -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        _observation(
            frame_id="non-live-frame",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[_det("person", (0.10, 0.20, 0.30, 0.70), 0.91)],
            **overrides,
        )
    )

    assert result["live"] is False
    assert result["reason"] == reason
    assert result["latest_scene_id"] == ""
    assert result["scene_snapshot"]["objects"] == []
    assert result["event_contents"] == []
    assert result["events"] == []
    assert result["object_count"] == 0
    assert result["track_count"] == 0
