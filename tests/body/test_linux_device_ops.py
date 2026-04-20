from __future__ import annotations

import json
from pathlib import Path


def test_probe_sherpa_model_dir_reports_required_files(tmp_path) -> None:
    from eibrain.body.runtime_linux import probe_sherpa_model_dir

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"):
        (model_dir / name).write_text("x", encoding="utf-8")

    result = probe_sherpa_model_dir(str(model_dir))

    assert result["status"] == "healthy"
    assert result["details"]["model_dir"] == str(model_dir)


def test_probe_binary_device_reports_missing_device() -> None:
    from eibrain.body.runtime_linux import probe_binary_device

    result = probe_binary_device(
        binary_name="v4l2-ctl",
        device_path="/dev/does-not-exist",
        label="camera",
    )

    assert result["status"] in {"degraded", "unavailable"}
    assert result["details"]["label"] == "camera"


def test_speak_text_uses_espeak_and_aplay_commands(tmp_path) -> None:
    from eibrain.body.runtime_linux import speak_text

    calls: list[list[str]] = []

    def _runner(command: list[str], **kwargs):
        calls.append(command)

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    result = speak_text(
        text="hello",
        output_device="plughw:2,0",
        runner=_runner,
        temp_dir=tmp_path,
    )

    assert result["status"] == "ok"
    assert calls[0][0] == "espeak"
    assert calls[1][:3] == ["aplay", "-D", "plughw:2,0"]


def test_speak_text_uses_minimax_t2a_and_aplay(tmp_path) -> None:
    from eibrain.body.runtime_linux import speak_text

    calls: list[list[str]] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": {"audio": "52494646", "status": 2},
                    "extra_info": {
                        "audio_size": 4,
                        "audio_length": 100,
                        "audio_sample_rate": 32000,
                        "audio_format": "wav",
                        "audio_channel": 1,
                    },
                    "trace_id": "trace-1",
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
            ).encode("utf-8")

    def _runner(command: list[str], **kwargs):
        calls.append(command)

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    result = speak_text(
        text="hello honjia",
        output_device="plughw:2,0",
        backend="minimax",
        api_key="secret",
        runner=_runner,
        urlopen=lambda req, timeout=0: _Response(),
        temp_dir=tmp_path,
    )

    assert result["status"] == "ok"
    assert result["details"]["backend"] == "minimax"
    assert result["details"]["trace_id"] == "trace-1"
    assert calls[0][:3] == ["aplay", "-D", "plughw:2,0"]


def test_probe_tts_playback_requires_minimax_api_key(monkeypatch) -> None:
    from eibrain.body.runtime_linux import probe_tts_playback

    monkeypatch.setattr("eibrain.body.runtime_linux.shutil.which", lambda name: "/usr/bin/aplay")

    result = probe_tts_playback(
        output_device="plughw:2,0",
        backend="minimax",
        api_key="",
    )

    assert result["status"] in {"degraded", "unavailable"}
    assert result["details"]["reason"] == "missing_minimax_api_key"


def test_move_gimbal_uses_injected_driver() -> None:
    from eibrain.body.runtime_linux import move_gimbal

    calls: list[tuple[int, int]] = []

    class _Driver:
        def ctrl_servo(self, angle: int, servo_id: int | None = None):
            calls.append((servo_id or 1, angle))
            return [255, 43, servo_id or 1, angle]

    result = move_gimbal(
        target_name="speaker",
        servo_id=1,
        home_angle=90,
        driver=_Driver(),
    )

    assert result["status"] == "ok"
    assert calls == [(1, 90)]


def test_raspbot_driver_uses_simple_servo_payload() -> None:
    from eibrain.body.raspbot_driver import RaspbotDriver

    driver = RaspbotDriver(mock=True, servo_id=1)

    payload = driver.ctrl_servo(90)

    assert payload == [1, 90]


def test_raspbot_driver_clamps_servo_2_to_110_degrees() -> None:
    from eibrain.body.raspbot_driver import RaspbotDriver

    driver = RaspbotDriver(mock=True, servo_id=2)

    payload = driver.ctrl_servo(180)

    assert payload == [2, 110]


def test_run_hailo_detection_reports_uvc_camera_gap() -> None:
    from eibrain.body.runtime_linux import run_hailo_detection

    def _runner(command: list[str], **kwargs):
        class _Completed:
            returncode = 1
            stdout = "\n".join(
                [
                    "Adding camera '/base/axi/pcie.../uvcvideo'",
                    "ERROR: *** no cameras available ***",
                ]
            )
            stderr = ""

        return _Completed()

    result = run_hailo_detection(
        post_process_file="/usr/share/rpi-camera-assets/hailo_yolov5_personface.json",
        runner=_runner,
    )

    assert result["status"] == "degraded"
    assert result["details"]["reason"] == "uvc_camera_not_usable_by_rpicam"


def test_run_hailo_detection_marks_timeout_as_started() -> None:
    from eibrain.body.runtime_linux import run_hailo_detection

    def _runner(command: list[str], **kwargs):
        class _Completed:
            returncode = 124
            stdout = ""
            stderr = ""

        return _Completed()

    result = run_hailo_detection(
        post_process_file="/usr/share/rpi-camera-assets/hailo_yolov5_personface.json",
        runner=_runner,
    )

    assert result["status"] == "ok"
    assert result["details"]["reason"] == "timed_out_after_start"


def test_parse_hailo_nms_output_extracts_sorted_detections() -> None:
    from eibrain.body.runtime_linux import parse_hailo_nms_output

    raw_output = [
        [
            [[0.1, 0.2, 0.5, 0.6, 0.55]],
            [[0.2, 0.3, 0.7, 0.8, 0.91]],
        ]
    ]

    detections = parse_hailo_nms_output(
        raw_output,
        class_labels=["person", "face"],
        score_threshold=0.5,
    )

    assert [item["label"] for item in detections] == ["face", "person"]
    assert detections[0]["bbox"]["x_min"] == 0.3
    assert detections[0]["score"] == 0.91
