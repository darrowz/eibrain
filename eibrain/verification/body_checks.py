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

    has_error = any(item["move"].get("status") != "ok" or item["capture"].get("status") != "ok" for item in captures)
    return {
        "status": "error" if has_error else "ok",
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
    return {"status": "ok", "frames": frames}


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
