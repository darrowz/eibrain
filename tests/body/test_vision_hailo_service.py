from __future__ import annotations


def test_single_frame_detector_publishes_vision_state(tmp_path, monkeypatch) -> None:
    from apps.body_runtime.vision_hailo_service import SingleFrameHailoDetector

    def _capture_frame(**kwargs):
        kwargs["output_path"].write_bytes(b"frame")
        return {"status": "ok", "details": {"device": kwargs["device"]}}

    def _infer(**_kwargs):
        return {
            "status": "ok",
            "details": {
                "detections": [
                    {
                        "label": "person",
                        "score": 0.8,
                        "bbox": {"x_min": 0.2, "y_min": 0.1, "x_max": 0.7, "y_max": 0.9},
                    }
                ]
            },
        }

    monkeypatch.setattr("apps.body_runtime.vision_hailo_service.capture_frame", _capture_frame)
    monkeypatch.setattr("apps.body_runtime.vision_hailo_service.run_hailo_frame_inference", _infer)

    detector = SingleFrameHailoDetector(
        camera_device="/dev/video0",
        frame_path=tmp_path / "latest.jpg",
        hef_path="/tmp/model.hef",
        labels=["person", "face"],
        score_threshold=0.3,
    )
    state = detector.detect_once()

    assert state["status"] == "ok"
    assert state["backend"] == "hailort_single_frame"
    assert state["detection_count"] == 1
    assert state["top_detection"]["label"] == "person"
