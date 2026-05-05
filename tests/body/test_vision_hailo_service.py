from __future__ import annotations


def test_detector_from_config_preserves_legacy_single_model_config() -> None:
    from apps.body_runtime.vision_hailo_service import SingleFrameHailoDetector
    from apps.body_runtime.vision_hailo_service import detector_from_config
    from eibrain.infra.config import BodyConfig
    from eibrain.infra.config import DriverConfig
    from eibrain.infra.config import EIBrainConfig
    from eibrain.infra.config import OrganConfig
    from eibrain.infra.config import SubfunctionConfig

    config = EIBrainConfig(
        body=BodyConfig(
            organs={
                "eye": OrganConfig(
                    subfunctions={
                        "camera": SubfunctionConfig(driver=DriverConfig(extra={"device": "/dev/video-test"})),
                        "detection": SubfunctionConfig(
                            driver=DriverConfig(
                                extra={
                                    "hef_path": "/tmp/personface.hef",
                                    "labels": ["person", "face"],
                                    "score_threshold": 0.42,
                                }
                            )
                        ),
                    }
                )
            }
        )
    )

    detector = detector_from_config(config, backend="single_frame")

    assert isinstance(detector, SingleFrameHailoDetector)
    assert detector.camera_device == "/dev/video-test"
    assert detector.hef_path == "/tmp/personface.hef"
    assert detector.labels == ["person", "face"]
    assert detector.score_threshold == 0.42


def test_detector_from_config_selects_highest_priority_enabled_model() -> None:
    from apps.body_runtime.vision_hailo_service import GStreamerHailoDetector
    from apps.body_runtime.vision_hailo_service import detector_from_config
    from eibrain.infra.config import BodyConfig
    from eibrain.infra.config import DriverConfig
    from eibrain.infra.config import EIBrainConfig
    from eibrain.infra.config import OrganConfig
    from eibrain.infra.config import SubfunctionConfig

    config = EIBrainConfig(
        body=BodyConfig(
            organs={
                "eye": OrganConfig(
                    subfunctions={
                        "camera": SubfunctionConfig(driver=DriverConfig(extra={})),
                        "detection": SubfunctionConfig(
                            driver=DriverConfig(
                                extra={
                                    "models": [
                                        {
                                            "id": "personface",
                                            "enabled": True,
                                            "hef_path": "/tmp/personface.hef",
                                            "postprocess_config_path": "/tmp/personface.json",
                                            "priority": 50,
                                        },
                                        {
                                            "id": "objects",
                                            "enabled": False,
                                            "hef_path": "/tmp/objects-disabled.hef",
                                            "postprocess_config_path": "/tmp/objects-disabled.json",
                                            "priority": 100,
                                        },
                                        {
                                            "id": "objects-enabled",
                                            "enabled": True,
                                            "hef_path": "/tmp/objects.hef",
                                            "postprocess_config_path": "/tmp/objects.json",
                                            "postprocess_function": "objects_filter",
                                            "score_threshold": 0.25,
                                            "priority": 75,
                                        },
                                    ]
                                }
                            )
                        ),
                    }
                )
            }
        )
    )

    detector = detector_from_config(config, backend="gstreamer")

    assert isinstance(detector, GStreamerHailoDetector)
    assert detector.hef_path == "/tmp/objects.hef"
    assert detector.postprocess_config_path == "/tmp/objects.json"
    assert detector.postprocess_function == "objects_filter"
    assert detector.score_threshold == 0.25


def test_detector_from_config_expands_coco80_label_set() -> None:
    from apps.body_runtime.vision_hailo_service import SingleFrameHailoDetector
    from apps.body_runtime.vision_hailo_service import detector_from_config
    from eibrain.infra.config import BodyConfig
    from eibrain.infra.config import DriverConfig
    from eibrain.infra.config import EIBrainConfig
    from eibrain.infra.config import OrganConfig
    from eibrain.infra.config import SubfunctionConfig

    config = EIBrainConfig(
        body=BodyConfig(
            organs={
                "eye": OrganConfig(
                    subfunctions={
                        "detection": SubfunctionConfig(
                            driver=DriverConfig(
                                extra={
                                    "models": [
                                        {
                                            "id": "objects",
                                            "enabled": True,
                                            "hef_path": "/tmp/objects.hef",
                                            "label_set": "coco80",
                                            "priority": 10,
                                        }
                                    ]
                                }
                            )
                        ),
                    }
                )
            }
        )
    )

    detector = detector_from_config(config, backend="single_frame")

    assert isinstance(detector, SingleFrameHailoDetector)
    assert len(detector.labels) == 80
    assert detector.labels[:3] == ["person", "bicycle", "car"]
    assert detector.labels[-1] == "toothbrush"


def test_model_label_set_overrides_legacy_top_level_labels() -> None:
    from apps.body_runtime.vision_hailo_service import SingleFrameHailoDetector
    from apps.body_runtime.vision_hailo_service import detector_from_config
    from eibrain.infra.config import BodyConfig
    from eibrain.infra.config import DriverConfig
    from eibrain.infra.config import EIBrainConfig
    from eibrain.infra.config import OrganConfig
    from eibrain.infra.config import SubfunctionConfig

    config = EIBrainConfig(
        body=BodyConfig(
            organs={
                "eye": OrganConfig(
                    subfunctions={
                        "detection": SubfunctionConfig(
                            driver=DriverConfig(
                                extra={
                                    "labels": ["person", "face"],
                                    "models": [
                                        {
                                            "id": "objects",
                                            "enabled": True,
                                            "hef_path": "/tmp/objects.hef",
                                            "label_set": "coco80",
                                            "priority": 10,
                                        }
                                    ],
                                }
                            )
                        ),
                    }
                )
            }
        )
    )

    detector = detector_from_config(config, backend="single_frame")

    assert isinstance(detector, SingleFrameHailoDetector)
    assert len(detector.labels) == 80
    assert detector.labels[41] == "cup"


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
        model_id="personface",
        labels=["person", "face"],
        score_threshold=0.3,
    )
    state = detector.detect_once()

    assert state["status"] == "ok"
    assert state["backend"] == "hailort_single_frame"
    assert state["detection_count"] == 1
    assert state["top_detection"]["label"] == "person"


def test_gstreamer_detector_omits_empty_postprocess_config_path(tmp_path) -> None:
    from apps.body_runtime.vision_hailo_service import GStreamerHailoDetector

    detector = GStreamerHailoDetector(
        camera_device="/dev/video0",
        frame_path=tmp_path / "latest.jpg",
        hef_path="/tmp/yolov8s.hef",
        model_id="yolov8s_h8",
        postprocess_so_path="/tmp/libyolo.so",
        postprocess_config_path="",
        postprocess_function="yolov8s",
        labels=["person", "cup"],
        score_threshold=0.35,
    )

    pipeline = detector._pipeline_text()

    assert "function-name=yolov8s" in pipeline
    assert "config-path=" not in pipeline


def test_vision_service_writes_sleeping_state_without_detection(tmp_path) -> None:
    from apps.body_runtime.engagement_state import EngagementStateReader
    from apps.body_runtime.engagement_state import EngagementStateWriter
    from apps.body_runtime.vision_hailo_service import VisionHailoService
    from eibrain.body.vision_state import VisionStateWriter

    class _Detector:
        def __init__(self) -> None:
            self.detect_calls = 0
            self.stop_calls = 0

        def detect_once(self):
            self.detect_calls += 1
            return {"status": "ok", "detections": []}

        def stop(self) -> None:
            self.stop_calls += 1

    engagement_path = tmp_path / "engagement.json"
    state_path = tmp_path / "state.json"
    EngagementStateWriter(engagement_path).write(conversation_active=False, phase="idle")
    detector = _Detector()
    service = VisionHailoService(
        detector=detector,
        writer=VisionStateWriter(state_path),
        interval_s=0.01,
        sleeping_interval_s=0.01,
        engagement_reader=EngagementStateReader(engagement_path),
    )
    service._running = True
    # Exercise the sleep branch by running the loop in a short helper thread.
    import threading
    import time

    thread = threading.Thread(target=service.run_forever, daemon=True)
    thread.start()
    time.sleep(0.03)
    service.stop()
    thread.join(timeout=1)

    import json

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert detector.detect_calls == 0
    assert detector.stop_calls >= 1
    assert state["status"] == "sleeping"
    assert state["backend"] == "vision_sleep_gate"


def test_vision_service_enriches_detection_frames_with_realtime_scene_events(tmp_path) -> None:
    from apps.body_runtime.vision_hailo_service import VisionHailoService
    from eibrain.body.vision_state import VisionStateWriter

    class _ReplayDetector:
        def __init__(self) -> None:
            self.frames = [
                {
                    "status": "ok",
                    "backend": "fake_hailo",
                    "frame_id": "frame-001",
                    "frame_captured_at_ts": 100.0,
                    "detections": [
                        {
                            "label": "cup",
                            "score": 0.9,
                            "bbox": {"x_min": 0.10, "y_min": 0.20, "x_max": 0.20, "y_max": 0.35},
                        }
                    ],
                    "details": {"latency_ms": 12.5},
                },
                {
                    "status": "ok",
                    "backend": "fake_hailo",
                    "frame_id": "frame-002",
                    "frame_captured_at_ts": 100.1,
                    "detections": [
                        {
                            "label": "cup",
                            "score": 0.91,
                            "bbox": {"x_min": 0.35, "y_min": 0.20, "x_max": 0.45, "y_max": 0.35},
                        }
                    ],
                    "details": {"latency_ms": 14.0},
                },
            ]
            self.index = 0

        def detect_once(self):
            frame = self.frames[self.index]
            self.index += 1
            return frame

    state_path = tmp_path / "state.json"
    detector = _ReplayDetector()
    service = VisionHailoService(
        detector=detector,
        writer=VisionStateWriter(state_path),
        interval_s=0.01,
        clock=lambda: 100.2,
    )

    first = service.process_once()
    second = service.process_once()

    assert first["track_count"] == 1
    assert second["track_count"] == 1
    assert first["scene"]["objects"][0]["trackId"] == second["scene"]["objects"][0]["trackId"]
    assert second["event_count"] >= 1
    assert any(event["eventType"] == "moved" for event in second["events"])
    assert second["fps"] > 0
    assert second["latency"] == {"ms": 14.0}
    assert second["freshness"]["source"] == "fake_hailo"
    assert second["freshness"]["age_s"] == 0.1
    assert second["source"] == {"backend": "fake_hailo", "mode": "realtime_simulated"}
    assert second["last_detection_summary"] == "Observed cup; realtime events: attention, moved"
    assert "detections" not in second["scene"]
