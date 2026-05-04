from __future__ import annotations

import sys

from eihead.eye import adapters

GStreamerHailoDetector = adapters.GStreamerHailoDetector
GStreamerHailoFrameSource = adapters.GStreamerHailoFrameSource
GStreamerHailoRealtimeAdapter = adapters.GStreamerHailoRealtimeAdapter
GStreamerHailoRealtimeConfig = adapters.GStreamerHailoRealtimeConfig
RealtimeVisionFrame = adapters.RealtimeVisionFrame


def test_adapter_classes_are_exported_from_eihead_eye_package() -> None:
    from eihead.eye import GStreamerHailoRealtimeAdapter as ExportedAdapter
    from eihead.eye import GStreamerHailoRealtimeConfig as ExportedConfig

    assert ExportedAdapter is GStreamerHailoRealtimeAdapter
    assert ExportedConfig is GStreamerHailoRealtimeConfig


def test_gstreamer_hailo_config_exposes_realtime_pipeline_fields() -> None:
    config = GStreamerHailoRealtimeConfig(
        camera_device="/dev/video2",
        hailo_device="/dev/hailo1",
        width=1280,
        height=720,
        framerate=60,
        hef_path="/opt/models/face.hef",
    )

    fields = config.pipeline_fields()
    pipeline = config.build_pipeline_description()

    assert config.mode == "realtime_stream"
    assert fields["camera_device"] == "/dev/video2"
    assert fields["hailo_device"] == "/dev/hailo1"
    assert fields["caps"] == "video/x-raw,width=1280,height=720,framerate=60/1"
    assert fields["source"].startswith("v4l2src")
    assert fields["inference"].startswith("hailonet")
    assert "/opt/models/face.hef" in fields["inference"]
    assert "/dev/video2" in pipeline
    assert "/dev/hailo1" in pipeline
    assert "filesrc" not in pipeline
    assert "compat_static_frame" not in pipeline


def test_device_paths_are_configured_without_touching_hardware() -> None:
    default_config = GStreamerHailoRealtimeConfig()
    custom_config = GStreamerHailoRealtimeConfig(camera_device="/dev/video9", hailo_device="/dev/hailo9")

    assert default_config.device_paths == ("/dev/video0", "/dev/hailo0")
    assert custom_config.device_paths == ("/dev/video9", "/dev/hailo9")


def test_missing_hardware_reports_not_wired_instead_of_fake_ok() -> None:
    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: False,
        gst_available=lambda: True,
    )

    status = adapter.status()

    assert status.status == "not_wired"
    assert status.backend == "gstreamer_hailo"
    assert status.placeholder is True
    assert status.not_wired is True
    assert status.frame_count == 0
    assert status.detection_count == 0
    assert "missing realtime devices" in status.message
    assert "/dev/video0" in status.message
    assert "/dev/hailo0" in status.message


def test_missing_gstreamer_backend_reports_not_wired_without_importing_gst() -> None:
    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: False,
    )

    status = adapter.status()

    assert status.status == "not_wired"
    assert status.not_wired is True
    assert "GStreamer backend is not installed" in status.message


def test_default_gstreamer_probe_requires_importable_gst_even_when_gi_spec_exists(monkeypatch) -> None:
    monkeypatch.setattr(adapters.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "gi", None)

    assert adapters._default_gst_available() is False


def test_ready_adapter_without_frame_reports_waiting_not_ok() -> None:
    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        frame_reader=lambda: None,
        detection_reader=lambda _frame: [],
        clock=lambda: 100.0,
    )

    status = adapter.poll()

    assert status.status == "waiting_for_frame"
    assert status.not_wired is False
    assert status.placeholder is False
    assert status.frame_count == 0
    assert "no realtime frame available" in status.message


def test_frame_reader_exception_reports_not_wired_without_raising() -> None:
    def failing_frame_reader():
        raise RuntimeError("camera appsink failed")

    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        frame_reader=failing_frame_reader,
        detection_reader=lambda _frame: [],
    )

    status = adapter.poll()

    assert status.status == "not_wired"
    assert status.not_wired is True
    assert "frame reader failed" in status.message
    assert "RuntimeError" in status.message


def test_detection_reader_exception_reports_not_wired_without_raising() -> None:
    def failing_detection_reader(_frame):
        raise RuntimeError("hailo parser failed")

    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        frame_reader=lambda: {"frame_id": "cam-1", "timestamp": 42.0},
        detection_reader=failing_detection_reader,
    )

    status = adapter.poll()

    assert status.status == "not_wired"
    assert status.not_wired is True
    assert "detection reader failed" in status.message
    assert "RuntimeError" in status.message


def test_ready_hardware_without_frame_reader_reports_not_wired() -> None:
    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        detection_reader=lambda _frame: [],
    )

    status = adapter.status()

    assert status.status == "not_wired"
    assert status.not_wired is True
    assert "frame reader is not wired" in status.message


def test_ready_hardware_without_detection_reader_reports_not_wired() -> None:
    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        frame_reader=lambda: {"frame_id": "cam-1", "timestamp": 42.0},
    )

    status = adapter.poll()

    assert status.status == "not_wired"
    assert status.not_wired is True
    assert "detection reader is not wired" in status.message


def test_frame_source_primary_path_is_realtime_not_static_image() -> None:
    source = GStreamerHailoFrameSource(
        GStreamerHailoRealtimeConfig(camera_device="/dev/video3", hailo_device="/dev/hailo3"),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        frame_reader=lambda: {"frame_id": "cam-1", "timestamp": 42.0, "payload": b"raw", "width": 800, "height": 600},
    )

    frame = source.next_frame()

    assert frame is not None
    assert frame.mode == "realtime_stream"
    assert frame.source == "gstreamer_hailo"
    assert frame.frame_id == "cam-1"
    assert frame.width == 800
    assert frame.height == 600
    assert frame.payload == b"raw"
    assert frame.metadata["camera_device"] == "/dev/video3"
    assert not hasattr(source.config, "static_image_path")


def test_detector_normalizes_hailo_results_for_realtime_status_consumers() -> None:
    frame = RealtimeVisionFrame(
        frame_id="cam-2",
        timestamp=50.0,
        width=200,
        height=400,
        source="gstreamer_hailo",
    )
    detector = GStreamerHailoDetector(
        GStreamerHailoRealtimeConfig(model_id="face-yolo"),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        detection_reader=lambda _frame: [
            {
                "label": "face",
                "score": "0.925",
                "box": {"xmin": 10, "ymin": 20, "xmax": 110, "ymax": 220},
                "class_id": "7",
                "track_id": 12,
                "attributes": {"source_tensor": "hailo0"},
            }
        ],
    )

    detection = detector.detect(frame)[0]
    payload = detection.to_dict()

    assert payload["label"] == "face"
    assert payload["confidence"] == 0.925
    assert payload["score"] == 0.925
    assert payload["bbox"] == {"x_min": 0.05, "y_min": 0.05, "x_max": 0.55, "y_max": 0.55}
    assert payload["class_id"] == 7
    assert payload["track_id"] == 12
    assert payload["source"] == "gstreamer_hailo"
    assert payload["model_id"] == "face-yolo"
    assert payload["attributes"] == {"source_tensor": "hailo0"}
