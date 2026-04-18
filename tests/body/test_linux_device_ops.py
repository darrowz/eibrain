from __future__ import annotations

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
