"""Command-driver entrypoints for honjia local device operations."""

from __future__ import annotations

import argparse
import json
import sys

from eibrain.body.raspbot_driver import RaspbotDriver
from eibrain.body.runtime_linux import capture_frame
from eibrain.body.runtime_linux import compare_frame_hashes
from eibrain.body.runtime_linux import move_gimbal
from eibrain.body.runtime_linux import probe_binary_device
from eibrain.body.runtime_linux import run_hailo_frame_inference
from eibrain.body.runtime_linux import run_hailo_detection
from eibrain.body.runtime_linux import probe_sherpa_model_dir
from eibrain.body.runtime_linux import speak_text


def main() -> None:
    parser = argparse.ArgumentParser(description="honjia local device operations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    binary_probe = subparsers.add_parser("probe-binary-device")
    binary_probe.add_argument("--binary", required=True)
    binary_probe.add_argument("--device", required=True)
    binary_probe.add_argument("--label", required=True)

    sherpa_probe = subparsers.add_parser("probe-sherpa-model")
    sherpa_probe.add_argument("--model-dir", required=True)

    speaker_probe = subparsers.add_parser("probe-speaker")
    speaker_probe.add_argument("--output-device", required=True)

    speak = subparsers.add_parser("speak")
    speak.add_argument("--output-device", required=True)

    gimbal = subparsers.add_parser("move-gimbal")
    gimbal.add_argument("--servo-id", type=int, default=1)
    gimbal.add_argument("--home-angle", type=int, default=90)

    capture = subparsers.add_parser("capture-frame")
    capture.add_argument("--device", required=True)
    capture.add_argument("--output-path", required=True)

    compare = subparsers.add_parser("compare-frames")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)

    hailo = subparsers.add_parser("hailo-camera-detect")
    hailo.add_argument(
        "--post-process-file",
        default="/usr/share/rpi-camera-assets/hailo_yolov5_personface.json",
    )
    hailo.add_argument("--timeout-s", type=int, default=8)

    hailo_frame = subparsers.add_parser("hailo-frame-infer")
    hailo_frame.add_argument("--image-path", required=True)
    hailo_frame.add_argument("--hef-path", default="/usr/share/hailo-models/yolov5s_personface_h8l.hef")
    hailo_frame.add_argument("--score-threshold", type=float, default=0.3)

    args = parser.parse_args()
    if args.command == "probe-binary-device":
        result = probe_binary_device(binary_name=args.binary, device_path=args.device, label=args.label)
    elif args.command == "probe-sherpa-model":
        result = probe_sherpa_model_dir(args.model_dir)
    elif args.command == "probe-speaker":
        result = probe_binary_device(binary_name="aplay", device_path="/dev/snd", label=f"speaker:{args.output_device}")
    elif args.command == "speak":
        payload = json.loads(sys.stdin.read() or "{}")
        result = speak_text(
            text=str(payload.get("payload", {}).get("text", "")),
            output_device=args.output_device,
        )
    elif args.command == "move-gimbal":
        payload = json.loads(sys.stdin.read() or "{}")
        try:
            body_payload = payload.get("payload", {})
            driver = RaspbotDriver(bus=1, addr=0x2B, servo_id=args.servo_id, enabled=True, mock=False)
            result = move_gimbal(
                target_name=str(body_payload.get("target_name", "")),
                servo_id=args.servo_id,
                home_angle=args.home_angle,
                target_x=body_payload.get("target_x"),
                pan_min=int(body_payload.get("pan_min", 40)),
                pan_max=int(body_payload.get("pan_max", 140)),
                driver=driver,
            )
        except Exception as exc:  # pragma: no cover - only on honjia
            result = {"status": "error", "details": {"error": str(exc), "driver": "raspbot"}}
    elif args.command == "capture-frame":
        result = capture_frame(device=args.device, output_path=args.output_path)
    elif args.command == "compare-frames":
        result = compare_frame_hashes(args.left, args.right)
    elif args.command == "hailo-camera-detect":
        result = run_hailo_detection(
            post_process_file=args.post_process_file,
            timeout_s=args.timeout_s,
        )
    elif args.command == "hailo-frame-infer":
        result = run_hailo_frame_inference(
            image_path=args.image_path,
            hef_path=args.hef_path,
            score_threshold=args.score_threshold,
        )
    else:  # pragma: no cover - argparse enforces
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
