from __future__ import annotations

from eihead.monitoring.realtime_vision import build_realtime_vision_payload


def test_monitor_payload_surfaces_native_eye_status_reasons_and_readiness() -> None:
    payload = build_realtime_vision_payload(
        {
            "schema": "eihead.eye.realtime_status.v1",
            "mode": "realtime_stream",
            "status": "degraded",
            "backend": "gstreamer_hailo",
            "last_frame_id": "frame-7",
            "last_frame_age": 0.44,
            "stream_ready": False,
            "stale": False,
            "degraded": True,
            "status_reason": "detection_reader_failed",
            "degraded_reason": "realtime detection reader failed: RuntimeError: parser down",
            "readiness": {"ready": False, "reason": "detection_reader_failed"},
            "detections": [
                {
                    "label": "person",
                    "score": 0.92,
                    "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                }
            ],
            "detection_boxes": [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
            ],
            "detection_scores": [0.92],
            "placeholder": False,
            "not_wired": False,
            "compatibility_mode": False,
        },
        timestamp=123.0,
        source="eye_realtime",
    )

    assert payload["status"] == "degraded"
    assert payload["wired"] is True
    assert payload["stream_ready"] is False
    assert payload["stale"] is False
    assert payload["degraded"] is True
    assert payload["status_reason"] == "detection_reader_failed"
    assert payload["degraded_reason"] == "realtime detection reader failed: RuntimeError: parser down"
    assert payload["readiness"] == {"ready": False, "reason": "detection_reader_failed"}
    assert payload["boxes"] == [
        {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
    ]
    assert payload["scores"] == [0.92]
    assert payload["diagnostic"]["stream_ready"] is False
    assert payload["diagnostic"]["degraded"] is True
    assert payload["diagnostic"]["status_reason"] == "detection_reader_failed"
