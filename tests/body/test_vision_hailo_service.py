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
