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
        assert result["movement_verified"] is True
        assert result["issues"] == []
        assert len(result["captures"]) == 3
        assert len(result["comparisons"]) == 2
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_gimbal_frame_check_marks_unchanged_frames_degraded() -> None:
    from eibrain.verification.body_checks import run_gimbal_frame_check

    tmp_path = _make_tmp_dir("verification-gimbal-degraded")
    try:
        result = run_gimbal_frame_check(
            angles=[40, 90],
            output_dir=tmp_path,
            move_fn=lambda angle: {"status": "ok", "details": {"angle": angle}},
            capture_fn=lambda angle, frame_path: {"status": "ok", "details": {"output_path": str(frame_path)}},
            compare_fn=lambda left, right: {"status": "unchanged", "details": {"same_hash": True}},
        )

        assert result["status"] == "degraded"
        assert result["movement_verified"] is False
        assert result["issues"] == ["camera frames did not change after gimbal movement"]
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
        assert result["recognized_frame_count"] == 2
        assert result["frames"][0]["summary"] == "seen:a.jpg"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_vision_frame_check_marks_blank_frames_degraded() -> None:
    from eibrain.verification.body_checks import run_vision_frame_check

    result = run_vision_frame_check(
        image_paths=["a.jpg", "b.jpg"],
        describe_fn=lambda image_path: {"summary": "", "primary_subject": "", "confidence": 0.0},
    )

    assert result["status"] == "degraded"
    assert result["recognized_frame_count"] == 0
    assert result["issues"] == ["vision recognizer did not return identifiable content for one or more frames"]


def test_run_ear_stream_check_returns_transcript() -> None:
    from eibrain.verification.body_checks import run_ear_stream_check

    result = run_ear_stream_check(
        chunk_count=3,
        transcribe_fn=lambda chunk_count: {"text": f"heard-{chunk_count}"},
    )

    assert result["status"] == "ok"
    assert result["transcript"]["text"] == "heard-3"


def test_run_hailo_camera_check_returns_detection() -> None:
    from eibrain.verification.body_checks import run_hailo_camera_check

    result = run_hailo_camera_check(
        detect_fn=lambda: {"status": "ok", "details": {"reason": "timed_out_after_start"}},
    )

    assert result["status"] == "ok"
    assert result["issues"] == []


def test_run_hailo_camera_check_surfaces_reason() -> None:
    from eibrain.verification.body_checks import run_hailo_camera_check

    result = run_hailo_camera_check(
        detect_fn=lambda: {"status": "degraded", "details": {"reason": "uvc_camera_not_usable_by_rpicam"}},
    )

    assert result["status"] == "degraded"
    assert result["issues"] == ["uvc_camera_not_usable_by_rpicam"]


def test_run_hailo_frame_check_collects_capture_and_inference() -> None:
    from eibrain.verification.body_checks import run_hailo_frame_check

    result = run_hailo_frame_check(
        capture_fn=lambda: {"status": "ok", "details": {"output_path": "frame.jpg"}},
        infer_fn=lambda: {"status": "ok", "details": {"detection_count": 2}},
    )

    assert result["status"] == "ok"
    assert result["issues"] == []


def test_run_hailo_frame_check_marks_zero_detections_degraded() -> None:
    from eibrain.verification.body_checks import run_hailo_frame_check

    result = run_hailo_frame_check(
        capture_fn=lambda: {"status": "ok", "details": {"output_path": "frame.jpg"}},
        infer_fn=lambda: {"status": "ok", "details": {"detection_count": 0}},
    )

    assert result["status"] == "degraded"
    assert result["issues"] == ["no_detections_found"]
