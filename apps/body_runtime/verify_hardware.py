"""Hardware verification CLI for honjia/honxin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apps.cognitive_runtime.app import CognitiveRuntimeApp
from apps.body_runtime.app import BodyRuntimeApp
from eibrain.body.raspbot_driver import RaspbotDriver
from eibrain.body.runtime_linux import capture_frame
from eibrain.body.runtime_linux import compare_frame_hashes
from eibrain.body.runtime_linux import move_gimbal
from eibrain.verification import run_ear_stream_check, run_gimbal_frame_check, run_vision_frame_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify honjia hardware and vision chains")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gimbal = subparsers.add_parser("gimbal-frame-check")
    gimbal.add_argument("--device", required=True)
    gimbal.add_argument("--output-dir", required=True)
    gimbal.add_argument("--angles", nargs="+", type=int, default=[40, 90, 140])
    gimbal.add_argument("--servo-id", type=int, default=1)

    ear = subparsers.add_parser("ear-stream-check")
    ear.add_argument("--config", default="config/eibrain.yaml")
    ear.add_argument("--chunk-count", type=int, default=3)
    ear.add_argument("--session-id", default="verify-ear")
    ear.add_argument("--actor-id", default="verify-user")

    vision = subparsers.add_parser("vision-frame-check")
    vision.add_argument("--config", default="config/eibrain.yaml")
    vision.add_argument("--images", nargs="+", required=True)

    args = parser.parse_args()
    if args.command == "gimbal-frame-check":
        driver = RaspbotDriver(bus=1, addr=0x2B, servo_id=args.servo_id, enabled=True, mock=False)
        result = run_gimbal_frame_check(
            angles=list(args.angles),
            output_dir=args.output_dir,
            move_fn=lambda angle: move_gimbal(
                target_name=f"angle-{angle}",
                servo_id=args.servo_id,
                home_angle=angle,
                driver=driver,
            ),
            capture_fn=lambda angle, frame_path: capture_frame(device=args.device, output_path=frame_path),
            compare_fn=compare_frame_hashes,
        )
    elif args.command == "ear-stream-check":
        runtime = BodyRuntimeApp.from_config_path(args.config)
        result = run_ear_stream_check(
            chunk_count=args.chunk_count,
            transcribe_fn=lambda chunk_count: runtime.transcribe_audio_window(
                chunk_count=chunk_count,
                session_id=args.session_id,
                actor_id=args.actor_id,
            ).to_dict(),
        )
    else:
        runtime = CognitiveRuntimeApp.from_config_path(args.config)
        result = run_vision_frame_check(
            image_paths=list(args.images),
            describe_fn=lambda image_path: _describe_frame(runtime, image_path),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _describe_frame(runtime: CognitiveRuntimeApp, image_path: str) -> dict[str, object]:
    understanding = runtime.describe_visual_frame(image_url=image_path)
    if understanding is None:
        return {"summary": "", "primary_subject": "", "confidence": 0.0}
    return {
        "summary": understanding.summary,
        "primary_subject": understanding.primary_subject,
        "confidence": understanding.confidence,
    }


if __name__ == "__main__":
    main()
