from __future__ import annotations

from pathlib import Path
import shutil


def _make_tmp_dir(name: str) -> Path:
    path = Path.cwd() / ".tmp-test-artifacts" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_capture_frame_uses_ffmpeg_command() -> None:
    from eibrain.body.runtime_linux import capture_frame

    calls: list[list[str]] = []

    def _runner(command: list[str], **kwargs):
        calls.append(command)

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        output = Path(command[-1])
        output.write_bytes(b"jpeg")
        return _Completed()

    tmp_path = _make_tmp_dir("capture-frame")
    try:
        output_path = tmp_path / "frame.jpg"
        result = capture_frame(device="/dev/video0", output_path=output_path, runner=_runner)

        assert result["status"] == "ok"
        assert calls[0][0] == "ffmpeg"
        assert result["details"]["output_path"] == str(output_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_compare_frame_hashes_reports_difference() -> None:
    from eibrain.body.runtime_linux import compare_frame_hashes

    tmp_path = _make_tmp_dir("compare-frame")
    try:
        left = tmp_path / "left.jpg"
        right = tmp_path / "right.jpg"
        left.write_bytes(b"left-frame")
        right.write_bytes(b"right-frame")

        result = compare_frame_hashes(left, right)

        assert result["status"] == "changed"
        assert result["details"]["same_hash"] is False
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
