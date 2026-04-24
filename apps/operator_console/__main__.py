"""CLI entrypoint for honjia monitoring web."""

from __future__ import annotations

import argparse

from apps.body_runtime.app import BodyRuntimeApp
from apps.body_runtime.visual_tracking_loop import VisualTrackingLoop
from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop
from apps.cognitive_runtime.app import CognitiveRuntimeApp
from eibrain.infra.config import load_config

from .web import MonitoringWebServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Start eibrain honjia monitoring web")
    parser.add_argument("--config", default="config/eibrain.yaml")
    parser.add_argument("--disable-voice-dialogue-loop", action="store_true")
    parser.add_argument("--disable-visual-tracking-loop", action="store_true")
    parser.add_argument("--voice-chunk-count", type=int, default=2)
    parser.add_argument("--visual-tracking-interval", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.config)
    runtime = BodyRuntimeApp(config=config)
    cognitive_runtime = CognitiveRuntimeApp(config=config)
    voice_loop = None
    if not args.disable_voice_dialogue_loop:
        voice_loop = VoiceDialogueLoop(
            body_runtime=runtime,
            cognitive_runtime=cognitive_runtime,
            chunk_count=args.voice_chunk_count,
        )
        voice_loop.start()
    visual_loop = None
    if not args.disable_visual_tracking_loop:
        visual_loop = VisualTrackingLoop(
            body_runtime=runtime,
            interval_s=args.visual_tracking_interval,
        )
        visual_loop.start()
    server = MonitoringWebServer(
        runtime=runtime,
        cognitive_runtime=cognitive_runtime,
        host=config.monitoring.host,
        port=config.monitoring.port,
    )
    server.start()
    try:
        print(f"monitoring web listening on http://{config.monitoring.host}:{server.port}")
        server._thread.join()  # type: ignore[union-attr]
    except KeyboardInterrupt:
        pass
    finally:
        if visual_loop is not None:
            visual_loop.stop()
        if voice_loop is not None:
            voice_loop.stop()
        server.stop()


if __name__ == "__main__":
    main()
