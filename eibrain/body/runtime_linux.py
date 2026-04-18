"""Linux helpers for honjia device probing and local actuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def probe_sherpa_model_dir(model_dir: str) -> dict[str, object]:
    path = Path(model_dir).expanduser()
    required = ("tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx")
    missing = [name for name in required if not (path / name).exists()]
    return {
        "status": "healthy" if not missing else "degraded",
        "details": {
            "driver": "sherpa_onnx",
            "model_dir": str(path),
            "missing_files": missing,
        },
    }


def probe_binary_device(*, binary_name: str, device_path: str, label: str) -> dict[str, object]:
    binary = shutil.which(binary_name)
    exists = Path(device_path).exists()
    status = "healthy" if binary and exists else "degraded"
    if not binary and not exists:
        status = "unavailable"
    return {
        "status": status,
        "details": {
            "label": label,
            "binary": binary or "",
            "device": device_path,
            "device_exists": exists,
        },
    }


def map_target_x_to_angle(*, target_x: float, pan_min: int, pan_max: int) -> int:
    clipped = min(max(target_x, 0.0), 1.0)
    return int(round(pan_min + (pan_max - pan_min) * clipped))


def speak_text(
    *,
    text: str,
    output_device: str,
    runner=subprocess.run,
    temp_dir: str | Path | None = None,
) -> dict[str, object]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as handle:
        wav_path = Path(handle.name)
    try:
        synth = runner(
            ["espeak", "-w", str(wav_path), "--", text],
            capture_output=True,
            text=True,
            check=False,
        )
        if synth.returncode != 0:
            return {"status": "error", "details": {"stderr": synth.stderr.strip(), "stdout": synth.stdout.strip()}}
        playback = runner(
            ["aplay", "-D", output_device, str(wav_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "status": "ok" if playback.returncode == 0 else "error",
            "details": {
                "output_device": output_device,
                "returncode": playback.returncode,
                "stdout": playback.stdout.strip(),
                "stderr": playback.stderr.strip(),
            },
        }
    finally:
        wav_path.unlink(missing_ok=True)


def move_gimbal(
    *,
    target_name: str,
    servo_id: int,
    home_angle: int = 90,
    target_x: float | None = None,
    pan_min: int = 40,
    pan_max: int = 140,
    driver=None,
) -> dict[str, object]:
    if driver is None:
        raise RuntimeError("gimbal driver is required")
    angle = home_angle if target_x is None else map_target_x_to_angle(target_x=target_x, pan_min=pan_min, pan_max=pan_max)
    payload = driver.ctrl_servo(angle, servo_id=servo_id)
    return {
        "status": "ok",
        "details": {
            "target_name": target_name,
            "servo_id": servo_id,
            "angle": angle,
            "payload": payload,
        },
    }


def capture_frame(
    *,
    device: str,
    output_path: str | Path,
    runner=subprocess.run,
) -> dict[str, object]:
    frame_path = Path(output_path)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "v4l2",
        "-i",
        device,
        "-frames:v",
        "1",
        "-y",
        str(frame_path),
    ]
    completed = runner(command, capture_output=True, text=True, check=False)
    return {
        "status": "ok" if completed.returncode == 0 and frame_path.exists() else "error",
        "details": {
            "device": device,
            "output_path": str(frame_path),
            "returncode": completed.returncode,
            "stdout": (completed.stdout or "").strip(),
            "stderr": (completed.stderr or "").strip(),
        },
    }


def compare_frame_hashes(left_path: str | Path, right_path: str | Path) -> dict[str, object]:
    left = Path(left_path)
    right = Path(right_path)
    left_hash = hashlib.sha256(left.read_bytes()).hexdigest()
    right_hash = hashlib.sha256(right.read_bytes()).hexdigest()
    same_hash = left_hash == right_hash
    return {
        "status": "unchanged" if same_hash else "changed",
        "details": {
            "left_path": str(left),
            "right_path": str(right),
            "left_hash": left_hash,
            "right_hash": right_hash,
            "same_hash": same_hash,
            "left_size": left.stat().st_size,
            "right_size": right.stat().st_size,
        },
    }
