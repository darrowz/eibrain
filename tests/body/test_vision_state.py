from __future__ import annotations

import json


def _assert_json_safe(value) -> None:
    json.dumps(value, ensure_ascii=False)


def test_vision_state_writer_writes_atomic_json_and_reader_marks_fresh(tmp_path) -> None:
    from eibrain.body.vision_state import VisionStateReader, VisionStateWriter

    state_path = tmp_path / "state.json"
    writer = VisionStateWriter(state_path)
    writer.write({"status": "ok", "updated_at_ts": 100.0, "detections": []})

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    snapshot = VisionStateReader(state_path, stale_after_s=3.0).read(now_ts=101.0)

    assert payload["status"] == "ok"
    assert snapshot.payload["detections"] == []
    assert snapshot.age_s == 1.0
    assert snapshot.stale is False


def test_build_vision_state_normalizes_detection_contract() -> None:
    from eibrain.body.vision_state import build_vision_state

    state = build_vision_state(
        detections=[
            {
                "label": "face",
                "confidence": 0.9,
                "class_id": "1",
                "bbox": {"xmin": -0.1, "ymin": 0.2, "xmax": 1.2, "ymax": 0.8},
            }
        ],
        frame_path="/tmp/latest.jpg",
        frame_captured_at_ts=100.0,
    )

    assert state["detection_count"] == 1
    assert state["top_detection"]["label"] == "face"
    assert state["top_detection"]["class_id"] == 1
    assert state["top_detection"]["bbox"]["x_min"] == 0.0
    assert state["top_detection"]["bbox"]["x_max"] == 1.0
    assert state["scene_summary"] == "1 face"


def test_build_vision_state_keeps_legacy_fields_with_schema_v2_defaults() -> None:
    from eibrain.body.vision_state import build_vision_state

    state = build_vision_state(
        detections=[
            {
                "label": "person",
                "score": 0.8,
                "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                "track_id": "track-1",
            }
        ],
        frame_path="/tmp/latest.jpg",
        frame_captured_at_ts=100.0,
    )

    assert state["schema_version"] == 2
    assert state["status"] == "ok"
    assert state["backend"] == "hailo8"
    assert state["detections"] == state["objects"]
    assert state["detection_count"] == 1
    assert state["top_detection"]["track_id"] == "track-1"
    assert state["scene_labels"] == ["person"]
    assert state["scene_summary"] == "1 person"
    assert state["scene"] == {
        "summary": "1 person",
        "labels": ["person"],
        "place_hint": None,
    }
    assert state["spatial"] == {"relations": []}
    assert state["events"] == []
    assert state["pipeline"] == {}
    _assert_json_safe(state)


def test_normalize_detection_preserves_json_safe_extension_fields() -> None:
    from eibrain.body.vision_state import normalize_detection

    normalized = normalize_detection(
        {
            "label": "chair",
            "score": 0.7,
            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9},
            "class_id": 2,
            "track_id": "abc",
            "category": "furniture",
            "source": "detector",
            "model_id": "yolo-v8",
            "attributes": {"color": "red", "confidence": 0.6, "tags": ["seat", None]},
            "spatial": {"center": [0.45, 0.55], "depth_m": 1.2},
            "stable_id": "chair-1",
            "region": {"name": "left", "priority": 1},
            "canonical_label": "chair",
            "unsafe": object(),
        }
    )

    assert normalized is not None
    assert normalized["label"] == "chair"
    assert normalized["score"] == 0.7
    assert normalized["class_id"] == 2
    assert normalized["track_id"] == "abc"
    assert normalized["category"] == "furniture"
    assert normalized["source"] == "detector"
    assert normalized["model_id"] == "yolo-v8"
    assert normalized["attributes"] == {"color": "red", "confidence": 0.6, "tags": ["seat", None]}
    assert normalized["spatial"] == {"center": [0.45, 0.55], "depth_m": 1.2}
    assert normalized["stable_id"] == "chair-1"
    assert normalized["region"] == {"name": "left", "priority": 1}
    assert normalized["canonical_label"] == "chair"
    assert "unsafe" not in normalized
    _assert_json_safe(normalized)


def test_build_vision_state_accepts_v2_metadata_and_keeps_it_json_safe() -> None:
    from eibrain.body.vision_state import build_vision_state

    state = build_vision_state(
        detections=[],
        frame_path=None,
        schema_version=3,
        pipeline={"name": "demo", "steps": ["detect"], "unsafe": object()},
        objects=[{"label": "manual", "score": 1.0, "bbox": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}}],
        scene={"summary": "manual scene", "labels": ["manual"], "extra": object()},
        spatial={"relations": [{"subject": "manual", "relation": "left_of", "object": "door"}], "unsafe": object()},
        events=[{"type": "appeared", "object": "manual", "unsafe": object()}],
        details={"safe": True, "unsafe": object()},
        frame_id="frame-1",
        camera_id="camera-1",
        scene_id="scene-1",
        place_hint="lab",
    )

    assert state["schema_version"] == 3
    assert state["objects"][0]["label"] == "manual"
    assert state["scene"] == {"summary": "manual scene", "labels": ["manual"], "place_hint": "lab"}
    assert state["spatial"] == {"relations": [{"subject": "manual", "relation": "left_of", "object": "door"}]}
    assert state["events"] == [{"type": "appeared", "object": "manual"}]
    assert state["pipeline"] == {"name": "demo", "steps": ["detect"]}
    assert state["details"] == {"safe": True}
    assert state["frame_id"] == "frame-1"
    assert state["camera_id"] == "camera-1"
    assert state["scene_id"] == "scene-1"
    _assert_json_safe(state)
