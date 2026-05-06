from __future__ import annotations

from pathlib import Path
import json
import time


def test_eye_organ_builds_live_detection_and_identity_details(tmp_path, monkeypatch) -> None:
    from eibrain.body.organs.eye.organ import EyeOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    def _capture_frame(*, device: str, output_path: str | Path, runner=None, **_kwargs):
        Path(output_path).write_bytes(b"frame-data")
        return {
            "status": "ok",
            "details": {
                "device": device,
                "output_path": str(output_path),
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            },
        }

    def _infer(*, image_path: str | Path, hef_path: str, labels=None, score_threshold: float = 0.3):
        return {
            "status": "ok",
            "details": {
                "image_path": str(image_path),
                "hef_path": hef_path,
                "detections": [
                    {
                        "label": "face",
                        "score": 0.91,
                        "bbox": {
                            "x_min": 0.2,
                            "y_min": 0.1,
                            "x_max": 0.6,
                            "y_max": 0.7,
                        },
                    },
                    {
                        "label": "person",
                        "score": 0.72,
                        "bbox": {
                            "x_min": 0.05,
                            "y_min": 0.02,
                            "x_max": 0.95,
                            "y_max": 0.98,
                        },
                    },
                ],
            },
        }

    monkeypatch.setattr("eibrain.body.organs.eye.organ.capture_frame", _capture_frame)
    monkeypatch.setattr("eibrain.body.organs.eye.organ.run_hailo_frame_inference", _infer)

    config = OrganConfig(
        enabled=True,
        subfunctions={
            "camera": SubfunctionConfig(
                driver=DriverConfig(kind="command", command=["python"], extra={"device": "/dev/video0"}),
            ),
            "detection": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={
                        "device": "/dev/hailo0",
                        "hef_path": "/tmp/model.hef",
                        "labels": ["person", "face"],
                        "refresh_interval_s": 0.0,
                    },
                ),
            ),
            "identity": SubfunctionConfig(
                driver=DriverConfig(kind="command", command=["python"], extra={"device": "/dev/hailo0"}),
            ),
        },
    )

    organ = EyeOrgan(config=config)
    heartbeat = organ.heartbeat()

    assert heartbeat.health == "healthy"
    assert heartbeat.subfunctions["camera"].health == "healthy"
    assert heartbeat.subfunctions["detection"].health == "healthy"
    assert heartbeat.subfunctions["detection"].details["detection_count"] == 2
    assert heartbeat.subfunctions["detection"].details["top_detection"]["label"] == "face"
    assert heartbeat.subfunctions["identity"].health == "healthy"
    assert heartbeat.subfunctions["identity"].details["status"] == "observing_unknown_face"
    assert heartbeat.subfunctions["identity"].details["face_candidate_count"] == 1
    assert heartbeat.subfunctions["identity"].details["identity_candidates"][0]["identity"] == "unknown"
    assert "error" not in heartbeat.subfunctions["identity"].details
    assert organ.latest_frame_path is not None


def test_eye_organ_keeps_identity_healthy_when_no_face_candidates(tmp_path, monkeypatch) -> None:
    from eibrain.body.organs.eye.organ import EyeOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    def _capture_frame(*, device: str, output_path: str | Path, runner=None, **_kwargs):
        Path(output_path).write_bytes(b"frame-data")
        return {"status": "ok", "details": {"device": device}}

    def _infer(*, image_path: str | Path, hef_path: str, labels=None, score_threshold: float = 0.3):
        return {
            "status": "ok",
            "details": {
                "detections": [
                    {
                        "label": "person",
                        "score": 0.78,
                        "bbox": {"x_min": 0.05, "y_min": 0.05, "x_max": 0.9, "y_max": 0.95},
                    }
                ]
            },
        }

    monkeypatch.setattr("eibrain.body.organs.eye.organ.capture_frame", _capture_frame)
    monkeypatch.setattr("eibrain.body.organs.eye.organ.run_hailo_frame_inference", _infer)

    config = OrganConfig(
        enabled=True,
        subfunctions={
            "camera": SubfunctionConfig(
                driver=DriverConfig(kind="command", command=["python"], extra={"device": "/dev/video0"}),
            ),
            "detection": SubfunctionConfig(
                driver=DriverConfig(kind="command", command=["python"], extra={"refresh_interval_s": 0.0}),
            ),
            "identity": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"])),
        },
    )

    heartbeat = EyeOrgan(config=config).heartbeat()

    assert heartbeat.health == "healthy"
    assert heartbeat.subfunctions["detection"].health == "healthy"
    assert heartbeat.subfunctions["identity"].health == "healthy"
    assert heartbeat.subfunctions["identity"].details["status"] == "no_face_candidates"
    assert heartbeat.subfunctions["identity"].details["face_candidate_count"] == 0
    assert "error" not in heartbeat.subfunctions["identity"].details


def test_eye_organ_uses_cached_heartbeat_until_refresh_interval_expires(monkeypatch) -> None:
    from eibrain.body.organs.eye.organ import EyeOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    capture_calls: list[int] = []

    def _capture_frame(*, device: str, output_path: str | Path, runner=None, **_kwargs):
        capture_calls.append(1)
        Path(output_path).write_bytes(b"frame-data")
        return {"status": "ok", "details": {"device": device}}

    def _infer(*, image_path: str | Path, hef_path: str, labels=None, score_threshold: float = 0.3):
        return {"status": "ok", "details": {"detections": []}}

    monkeypatch.setattr("eibrain.body.organs.eye.organ.capture_frame", _capture_frame)
    monkeypatch.setattr("eibrain.body.organs.eye.organ.run_hailo_frame_inference", _infer)

    config = OrganConfig(
        enabled=True,
        subfunctions={
            "camera": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"], extra={"device": "/dev/video0"})),
            "detection": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"], extra={"refresh_interval_s": 999.0})),
            "identity": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"])),
        },
    )

    organ = EyeOrgan(config=config)
    organ.heartbeat()
    organ.heartbeat()

    assert len(capture_calls) == 1


def test_eye_organ_reads_vision_service_state_without_capturing(tmp_path, monkeypatch) -> None:
    from eibrain.body.organs.eye.organ import EyeOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    def _should_not_capture(**_kwargs):
        raise AssertionError("vision_state mode must not capture frames synchronously")

    monkeypatch.setattr("eibrain.body.organs.eye.organ.capture_frame", _should_not_capture)
    frame_path = tmp_path / "latest.jpg"
    state_path = tmp_path / "state.json"
    frame_path.write_bytes(b"frame")
    state_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "backend": "gstreamer_hailo",
                "updated_at_ts": time.time(),
                "frame_path": str(frame_path),
                "frame_captured_at_ts": 123.0,
                "detections": [
                    {
                        "label": "face",
                        "score": 0.95,
                        "bbox": {"x_min": 0.4, "y_min": 0.1, "x_max": 0.6, "y_max": 0.5},
                    }
                ],
                "top_detection": {
                    "label": "face",
                    "score": 0.95,
                    "bbox": {"x_min": 0.4, "y_min": 0.1, "x_max": 0.6, "y_max": 0.5},
                },
                "fps": 9.8,
                "pipeline": {"target_fps": 10.0, "interval_s": 0.1},
                "telemetry": {"loop_elapsed_ms": 12.4, "configured_interval_s": 0.1},
                "scene_summary": "1 face",
            }
        ),
        encoding="utf-8",
    )
    config = OrganConfig(
        enabled=True,
        subfunctions={
            "camera": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={"provider": "vision_state", "state_path": str(state_path), "frame_path": str(frame_path)},
                ),
            ),
            "detection": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={"provider": "vision_state", "state_path": str(state_path), "frame_path": str(frame_path)},
                ),
            ),
            "identity": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"])),
        },
    )

    organ = EyeOrgan(config=config)
    heartbeat = organ.heartbeat()

    assert heartbeat.health == "healthy"
    assert heartbeat.subfunctions["camera"].details["driver"] == "vision_state"
    assert heartbeat.subfunctions["detection"].details["detection_count"] == 1
    assert heartbeat.subfunctions["detection"].details["top_detection"]["label"] == "face"
    assert heartbeat.subfunctions["detection"].details["fps"] == 9.8
    assert heartbeat.subfunctions["detection"].details["pipeline"]["target_fps"] == 10.0
    assert heartbeat.subfunctions["detection"].details["telemetry"]["loop_elapsed_ms"] == 12.4
    assert heartbeat.subfunctions["identity"].details["face_candidate_count"] == 1
    assert organ.latest_frame_path == str(frame_path)


def test_eye_organ_treats_sleeping_vision_state_as_healthy_standby(tmp_path) -> None:
    from eibrain.body.organs.eye.organ import EyeOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "sleeping",
                "backend": "vision_sleep_gate",
                "updated_at_ts": time.time(),
                "frame_path": None,
                "detections": [],
                "pipeline": {"mode": "sleeping"},
            }
        ),
        encoding="utf-8",
    )
    config = OrganConfig(
        enabled=True,
        subfunctions={
            "camera": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={"provider": "vision_state", "state_path": str(state_path), "stale_after_s": 3.0},
                ),
            ),
            "detection": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={"provider": "vision_state", "state_path": str(state_path), "stale_after_s": 3.0},
                ),
            ),
            "identity": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"])),
        },
    )

    heartbeat = EyeOrgan(config=config).heartbeat()

    assert heartbeat.health == "healthy"
    assert heartbeat.subfunctions["camera"].health == "healthy"
    assert heartbeat.subfunctions["detection"].details["status"] == "sleeping"
    assert "error" not in heartbeat.subfunctions["detection"].details


def test_eye_organ_passive_heartbeat_reads_vision_state(tmp_path) -> None:
    from eibrain.body.organs.eye.organ import EyeOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "updated_at_ts": time.time(),
                "frame_path": str(tmp_path / "latest.jpg"),
                "detections": [],
            }
        ),
        encoding="utf-8",
    )
    config = OrganConfig(
        enabled=True,
        subfunctions={
            "camera": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={"provider": "vision_state", "state_path": str(state_path)},
                ),
            ),
            "detection": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={"provider": "vision_state", "state_path": str(state_path)},
                ),
            ),
            "identity": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"])),
        },
    )

    heartbeat = EyeOrgan(config=config).passive_heartbeat()

    assert heartbeat.subfunctions["camera"].details["driver"] == "vision_state"
    assert heartbeat.subfunctions["detection"].details["status"] == "live"


def test_eye_organ_degrades_when_vision_service_state_is_stale(tmp_path) -> None:
    from eibrain.body.organs.eye.organ import EyeOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "updated_at_ts": time.time() - 60,
                "detections": [],
                "frame_path": str(tmp_path / "latest.jpg"),
            }
        ),
        encoding="utf-8",
    )
    config = OrganConfig(
        enabled=True,
        subfunctions={
            "camera": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={"provider": "vision_state", "state_path": str(state_path), "stale_after_s": 1.0},
                ),
            ),
            "detection": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={"provider": "vision_state", "state_path": str(state_path), "stale_after_s": 1.0},
                ),
            ),
            "identity": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"])),
        },
    )

    heartbeat = EyeOrgan(config=config).heartbeat()

    assert heartbeat.health == "degraded"
    assert heartbeat.subfunctions["camera"].health == "degraded"
    assert heartbeat.subfunctions["detection"].details["error"] == "vision_state_stale"
