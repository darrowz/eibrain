from __future__ import annotations

import json


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
