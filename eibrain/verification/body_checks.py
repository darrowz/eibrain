"""Hardware verification workflows for honjia/honxin."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def run_gimbal_frame_check(
    *,
    angles: list[int],
    output_dir: str | Path,
    move_fn: Callable[[int], dict[str, object]],
    capture_fn: Callable[[int, Path], dict[str, object]],
    compare_fn: Callable[[Path, Path], dict[str, object]],
) -> dict[str, object]:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    captures: list[dict[str, object]] = []
    frame_paths: list[Path] = []
    for angle in angles:
        move_result = move_fn(angle)
        frame_path = artifact_dir / f"frame-angle-{angle}.jpg"
        capture_result = capture_fn(angle, frame_path)
        frame_paths.append(frame_path)
        captures.append(
            {
                "angle": angle,
                "move": move_result,
                "capture": capture_result,
                "frame_path": str(frame_path),
            }
        )

    comparisons: list[dict[str, object]] = []
    for left, right in zip(frame_paths, frame_paths[1:]):
        comparisons.append(compare_fn(left, right))

    issues: list[str] = []
    has_error = any(item["move"].get("status") != "ok" or item["capture"].get("status") != "ok" for item in captures)
    if has_error:
        issues.append("gimbal move or frame capture failed")
    unchanged_pairs = [
        comparison
        for comparison in comparisons
        if comparison.get("status") != "changed"
    ]
    if unchanged_pairs:
        issues.append("camera frames did not change after gimbal movement")
    status = "ok"
    if has_error:
        status = "error"
    elif unchanged_pairs:
        status = "degraded"
    return {
        "status": status,
        "movement_verified": not has_error and not unchanged_pairs,
        "issues": issues,
        "captures": captures,
        "comparisons": comparisons,
    }


def run_vision_frame_check(
    *,
    image_paths: list[str | Path],
    describe_fn: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for image_path in image_paths:
        summary = describe_fn(str(image_path))
        frames.append(
            {
                "image_path": str(image_path),
                **summary,
            }
        )
    weak_frames = [
        frame
        for frame in frames
        if not str(frame.get("summary", "")).strip()
        and not str(frame.get("primary_subject", "")).strip()
    ]
    issues: list[str] = []
    if weak_frames:
        issues.append("vision recognizer did not return identifiable content for one or more frames")
    return {
        "status": "degraded" if weak_frames else "ok",
        "recognized_frame_count": len(frames) - len(weak_frames),
        "issues": issues,
        "frames": frames,
    }


def run_ear_stream_check(
    *,
    chunk_count: int,
    transcribe_fn: Callable[[int], dict[str, object]],
) -> dict[str, object]:
    transcript = transcribe_fn(chunk_count)
    return {
        "status": "ok" if transcript.get("text") else "degraded",
        "chunk_count": chunk_count,
        "transcript": transcript,
    }


def run_hailo_camera_check(
    *,
    detect_fn: Callable[[], dict[str, object]],
) -> dict[str, object]:
    detection = detect_fn()
    status = str(detection.get("status", "error"))
    details = detection.get("details", {})
    reason = details.get("reason") if isinstance(details, dict) else ""
    issues: list[str] = []
    if status != "ok":
        issues.append(str(reason or "hailo camera detection failed"))
    return {
        "status": status,
        "issues": issues,
        "detection": detection,
    }
