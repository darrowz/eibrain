from __future__ import annotations

from pathlib import Path
import shutil


def _make_tmp_dir(name: str) -> Path:
    path = Path.cwd() / ".tmp-test-artifacts" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_run_gimbal_frame_check_collects_changed_frames() -> None:
    from eibrain.verification.body_checks import run_gimbal_frame_check

    moves: list[int] = []

    def _move(angle: int) -> dict[str, object]:
        moves.append(angle)
        return {"status": "ok", "details": {"angle": angle}}

    def _capture(angle: int, frame_path: Path) -> dict[str, object]:
        frame_path.write_bytes(f"frame-{angle}".encode("utf-8"))
        return {"status": "ok", "details": {"output_path": str(frame_path)}}

    def _compare(left: Path, right: Path) -> dict[str, object]:
        return {"status": "changed", "details": {"left": str(left), "right": str(right), "same_hash": False}}

    tmp_path = _make_tmp_dir("verification-gimbal")
    try:
        result = run_gimbal_frame_check(
            angles=[40, 90, 140],
            output_dir=tmp_path,
            move_fn=_move,
            capture_fn=_capture,
            compare_fn=_compare,
        )

        assert moves == [40, 90, 140]
        assert result["status"] == "ok"
        assert len(result["captures"]) == 3
        assert len(result["comparisons"]) == 2
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_vision_frame_check_collects_summaries() -> None:
    from eibrain.verification.body_checks import run_vision_frame_check

    tmp_path = _make_tmp_dir("verification-vision")
    try:
        frame_a = tmp_path / "a.jpg"
        frame_b = tmp_path / "b.jpg"
        frame_a.write_bytes(b"a")
        frame_b.write_bytes(b"b")

        result = run_vision_frame_check(
            image_paths=[frame_a, frame_b],
            describe_fn=lambda image_path: {"summary": f"seen:{Path(image_path).name}"},
        )

        assert result["status"] == "ok"
        assert result["frames"][0]["summary"] == "seen:a.jpg"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_ear_stream_check_returns_transcript() -> None:
    from eibrain.verification.body_checks import run_ear_stream_check

    result = run_ear_stream_check(
        chunk_count=3,
        transcribe_fn=lambda chunk_count: {"text": f"heard-{chunk_count}"},
    )

    assert result["status"] == "ok"
    assert result["transcript"]["text"] == "heard-3"
