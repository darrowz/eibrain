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


def run_hailo_detection(
    *,
    post_process_file: str,
    timeout_s: int = 8,
    runner=subprocess.run,
) -> dict[str, object]:
    command = [
        "timeout",
        f"{timeout_s}s",
        "rpicam-hello",
        "--nopreview",
        "--post-process-file",
        post_process_file,
        "--verbose",
        "2",
    ]
    completed = runner(command, capture_output=True, text=True, check=False)
    combined_output = "\n".join(
        part.strip()
        for part in (completed.stdout or "", completed.stderr or "")
        if part.strip()
    )
    lowered = combined_output.lower()
    status = "ok" if completed.returncode == 0 else "error"
    reason = ""
    if "adding camera" in lowered and "no cameras available" in lowered:
        status = "degraded"
        reason = "uvc_camera_not_usable_by_rpicam"
    elif "no cameras available" in lowered:
        status = "degraded"
        reason = "rpicam_no_cameras_available"
    elif completed.returncode == 124:
        status = "ok"
        reason = "timed_out_after_start"
    return {
        "status": status,
        "details": {
            "post_process_file": post_process_file,
            "returncode": completed.returncode,
            "reason": reason,
            "stdout": (completed.stdout or "").strip(),
            "stderr": (completed.stderr or "").strip(),
            "combined_output": combined_output,
        },
    }


def parse_hailo_nms_output(
    raw_output,
    *,
    class_labels: list[str] | None = None,
    score_threshold: float = 0.0,
) -> list[dict[str, object]]:
    labels = class_labels or []
    detections: list[dict[str, object]] = []
    for batch_index, batch in enumerate(raw_output or []):
        if not isinstance(batch, list):
            continue
        for class_id, class_detections in enumerate(batch):
            if class_detections is None:
                continue
            for row in class_detections:
                values = row.tolist() if hasattr(row, "tolist") else list(row)
                if len(values) < 5:
                    continue
                y_min, x_min, y_max, x_max, score = (float(value) for value in values[:5])
                if score < score_threshold:
                    continue
                detections.append(
                    {
                        "batch_index": batch_index,
                        "class_id": class_id,
                        "label": labels[class_id] if class_id < len(labels) else f"class_{class_id}",
                        "score": round(score, 6),
                        "bbox": {
                            "x_min": round(x_min, 6),
                            "y_min": round(y_min, 6),
                            "x_max": round(x_max, 6),
                            "y_max": round(y_max, 6),
                        },
                    }
                )
    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    return detections


def run_hailo_frame_inference(
    *,
    image_path: str | Path,
    hef_path: str,
    labels: list[str] | None = None,
    score_threshold: float = 0.3,
) -> dict[str, object]:
    try:
        import numpy as np  # type: ignore
        from hailo_platform import (  # type: ignore
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            HEF,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )
    except Exception as exc:
        return _run_hailo_frame_inference_with_system_python(
            image_path=image_path,
            hef_path=hef_path,
            labels=labels,
            score_threshold=score_threshold,
            import_error=str(exc),
        )

    frame_path = Path(image_path)
    if not frame_path.exists():
        return {
            "status": "error",
            "details": {
                "reason": "image_load_failed",
                "image_path": str(frame_path),
                "hef_path": hef_path,
            },
        }

    hef = HEF(hef_path)
    input_info = hef.get_input_vstream_infos()[0]
    height, width = tuple(input_info.shape[:2])
    decode = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(frame_path),
            "-vf",
            f"scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    expected_size = width * height * 3
    if decode.returncode != 0 or len(decode.stdout) != expected_size:
        return {
            "status": "error",
            "details": {
                "reason": "image_decode_failed",
                "image_path": str(frame_path),
                "hef_path": hef_path,
                "returncode": decode.returncode,
                "stderr": decode.stderr.decode("utf-8", "replace").strip(),
                "stdout_size": len(decode.stdout),
                "expected_size": expected_size,
            },
        }
    resized = np.frombuffer(decode.stdout, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32)

    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        input_params = InputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )
        output_params = OutputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                result = infer_pipeline.infer({input_info.name: np.expand_dims(resized, axis=0)})

    output_name, raw_output = next(iter(result.items()))
    detections = parse_hailo_nms_output(
        raw_output,
        class_labels=labels or ["person", "face"],
        score_threshold=score_threshold,
    )
    return {
        "status": "ok",
        "details": {
            "image_path": str(frame_path),
            "hef_path": hef_path,
            "input_name": input_info.name,
            "output_name": output_name,
            "model_shape": [int(height), int(width), int(input_info.shape[2])],
            "detection_count": len(detections),
            "detections": detections,
        },
    }


def _run_hailo_frame_inference_with_system_python(
    *,
    image_path: str | Path,
    hef_path: str,
    labels: list[str] | None,
    score_threshold: float,
    import_error: str,
) -> dict[str, object]:
    payload = {
        "image_path": str(image_path),
        "hef_path": hef_path,
        "labels": labels or ["person", "face"],
        "score_threshold": score_threshold,
    }
    script = r"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from hailo_platform import (
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    HEF,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


def parse_hailo_nms_output(raw_output, class_labels, score_threshold):
    detections = []
    for batch_index, batch in enumerate(raw_output or []):
        if not isinstance(batch, list):
            continue
        for class_id, class_detections in enumerate(batch):
            if class_detections is None:
                continue
            for row in class_detections:
                values = row.tolist() if hasattr(row, "tolist") else list(row)
                if len(values) < 5:
                    continue
                y_min, x_min, y_max, x_max, score = (float(value) for value in values[:5])
                if score < score_threshold:
                    continue
                detections.append(
                    {
                        "batch_index": batch_index,
                        "class_id": class_id,
                        "label": class_labels[class_id] if class_id < len(class_labels) else f"class_{class_id}",
                        "score": round(score, 6),
                        "bbox": {
                            "x_min": round(x_min, 6),
                            "y_min": round(y_min, 6),
                            "x_max": round(x_max, 6),
                            "y_max": round(y_max, 6),
                        },
                    }
                )
    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    return detections


payload = json.loads(sys.stdin.read())
frame_path = Path(payload["image_path"])
hef_path = payload["hef_path"]
labels = payload["labels"]
score_threshold = float(payload["score_threshold"])

hef = HEF(hef_path)
input_info = hef.get_input_vstream_infos()[0]
height, width = tuple(input_info.shape[:2])
decode = subprocess.run(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(frame_path),
        "-vf",
        f"scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ],
    capture_output=True,
    check=False,
)
expected_size = width * height * 3
if decode.returncode != 0 or len(decode.stdout) != expected_size:
    print(
        json.dumps(
            {
                "status": "error",
                "details": {
                    "reason": "image_decode_failed",
                    "image_path": str(frame_path),
                    "hef_path": hef_path,
                    "returncode": decode.returncode,
                    "stderr": decode.stderr.decode("utf-8", "replace").strip(),
                    "stdout_size": len(decode.stdout),
                    "expected_size": expected_size,
                },
            }
        )
    )
    raise SystemExit(0)
resized = np.frombuffer(decode.stdout, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32)
with VDevice() as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    input_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            result = infer_pipeline.infer({input_info.name: np.expand_dims(resized, axis=0)})
output_name, raw_output = next(iter(result.items()))
print(
    json.dumps(
        {
            "status": "ok",
            "details": {
                "image_path": str(frame_path),
                "hef_path": hef_path,
                "input_name": input_info.name,
                "output_name": output_name,
                "model_shape": [int(height), int(width), int(input_info.shape[2])],
                "detection_count": len(parse_hailo_nms_output(raw_output, labels, score_threshold)),
                "detections": parse_hailo_nms_output(raw_output, labels, score_threshold),
            },
        }
    )
)
"""
    completed = subprocess.run(
        ["/usr/bin/python3", "-c", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        try:
            return json.loads(completed.stdout)
        except Exception:
            pass
    return {
        "status": "error",
        "details": {
            "reason": "hailo_runtime_unavailable",
            "error": import_error,
            "fallback_returncode": completed.returncode,
            "fallback_stdout": completed.stdout.strip(),
            "fallback_stderr": completed.stderr.strip(),
            "image_path": str(image_path),
            "hef_path": hef_path,
        },
    }
