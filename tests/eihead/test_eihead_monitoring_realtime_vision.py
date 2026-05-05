from __future__ import annotations

from eihead.monitoring import build_realtime_vision_payload


def test_realtime_vision_payload_exposes_visual_overlay_for_box_diagnostics() -> None:
    payload = build_realtime_vision_payload(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "tracking",
            "frame_id": "frame-overlay-1",
            "width": 640,
            "height": 480,
            "fps": 30.0,
            "detections": [
                {
                    "label": "person",
                    "score": 0.91,
                    "bbox": {"x_min": 160, "y_min": 120, "x_max": 320, "y_max": 360},
                },
                {
                    "label": "dog",
                    "confidence": 0.6,
                    "bbox": {"x_min": 0.6, "y_min": 0.25, "x_max": 0.8, "y_max": 0.75},
                },
                {
                    "label": "face",
                    "score": 0.5,
                    "bbox": {"x": 320, "y": 120, "w": 160, "h": 120},
                },
            ],
        },
        timestamp=1000.0,
        source="eye_realtime",
    )

    assert payload["overlay"] == payload["visual_diagnostic"]
    assert payload["overlay"]["frame"] == {
        "width": 640,
        "height": 480,
        "frame_id": "frame-overlay-1",
        "image_available": False,
        "image_message": "no live frame image yet",
    }
    assert payload["overlay"]["stream_ready"] is True
    assert payload["overlay"]["normalized_boxes"] == [
        {
            "label": "person",
            "score": 0.91,
            "score_label": "person 0.91",
            "x_min": 0.25,
            "y_min": 0.25,
            "x_max": 0.5,
            "y_max": 0.75,
        },
        {
            "label": "dog",
            "score": 0.6,
            "score_label": "dog 0.60",
            "x_min": 0.6,
            "y_min": 0.25,
            "x_max": 0.8,
            "y_max": 0.75,
        },
        {
            "label": "face",
            "score": 0.5,
            "score_label": "face 0.50",
            "x_min": 0.5,
            "y_min": 0.25,
            "x_max": 0.75,
            "y_max": 0.5,
        },
    ]
    assert payload["overlay"]["score_labels"] == ["person 0.91", "dog 0.60", "face 0.50"]
    assert payload["overlay"]["top_target"] == {
        "label": "person",
        "score": 0.91,
        "score_label": "person 0.91",
        "center": {"x": 0.375, "y": 0.5},
        "error": {"x": -0.125, "y": 0.0},
    }
