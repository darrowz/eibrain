"""CLI entrypoint for honjia monitoring web."""

from __future__ import annotations

import argparse

from apps.body_runtime.app import BodyRuntimeApp
from apps.body_runtime.engagement_state import DEFAULT_ENGAGEMENT_STATE_PATH
from apps.body_runtime.engagement_state import EngagementStateReader
from apps.body_runtime.engagement_state import EngagementStateWriter
from apps.body_runtime.visual_tracking_loop import VisualTrackingLoop
from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop
from apps.cognitive_runtime.app import CognitiveRuntimeApp
from eibrain.body.realtime_audio import ArecordRawChunkSource
from eibrain.body.realtime_audio import PcmRingBuffer
from eibrain.body.realtime_audio import RealtimeAudioCaptureWorker
from eibrain.body.realtime_audio import RealtimeWakeAudioPipeline
from eibrain.body.realtime_audio import RealtimeWakeDetector
from eibrain.infra.config import load_config

from .web import MonitoringWebServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Start eibrain honjia monitoring web")
    parser.add_argument("--config", default="config/eibrain.yaml")
    parser.add_argument("--disable-voice-dialogue-loop", action="store_true")
    parser.add_argument("--disable-visual-tracking-loop", action="store_true")
    parser.add_argument("--voice-chunk-count", type=int, default=2)
    parser.add_argument("--visual-tracking-interval", type=float, default=0.5)
    parser.add_argument("--visual-tracking-source", choices=("active", "state"), default="active")
    parser.add_argument("--engagement-state-path", default=str(DEFAULT_ENGAGEMENT_STATE_PATH))
    parser.add_argument("--security-vision-always-on", action="store_true")
    parser.add_argument("--wake-word", default=r"\u9e3f\u9014")
    parser.add_argument("--sleep-word", default=r"\u7ed3\u675f\u5bf9\u8bdd")
    parser.add_argument("--enable-realtime-wake-audio", action="store_true")
    parser.add_argument("--realtime-wake-buffer-ms", type=int, default=6000)
    parser.add_argument("--realtime-wake-lookback-ms", type=int, default=2400)
    parser.add_argument("--realtime-wake-min-buffer-ms", type=int, default=480)
    parser.add_argument("--realtime-wake-frame-ms", type=int, default=120)
    parser.add_argument("--realtime-wake-poll-interval", type=float, default=0.25)
    args = parser.parse_args()

    config = load_config(args.config)
    runtime = BodyRuntimeApp(config=config)
    cognitive_runtime = CognitiveRuntimeApp(config=config)
    engagement_writer = EngagementStateWriter(args.engagement_state_path)
    engagement_reader = EngagementStateReader(
        args.engagement_state_path,
        security_mode=args.security_vision_always_on,
    )
    voice_loop = None
    if not args.disable_voice_dialogue_loop:
        realtime_wake_source = None
        wake_word = _decode_cli_text(args.wake_word)
        if args.enable_realtime_wake_audio:
            try:
                realtime_wake_source = _build_realtime_wake_source(runtime, config, args, wake_word=wake_word)
            except Exception as exc:
                runtime.update_voice_dialogue_state(
                    realtime_audio={"enabled": True, "running": False, "last_error": str(exc)}
                )
        voice_loop = VoiceDialogueLoop(
            body_runtime=runtime,
            cognitive_runtime=cognitive_runtime,
            chunk_count=args.voice_chunk_count,
            wake_word=wake_word,
            sleep_word=_decode_cli_text(args.sleep_word),
            engagement_writer=engagement_writer,
            realtime_wake_source=realtime_wake_source,
        )
        voice_loop.start()
    visual_loop = None
    if not args.disable_visual_tracking_loop:
        visual_loop = VisualTrackingLoop(
            body_runtime=runtime,
            interval_s=args.visual_tracking_interval,
            source=args.visual_tracking_source,
            engagement_reader=engagement_reader,
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


def _decode_cli_text(value: str) -> str:
    if "\\u" not in value and "\\U" not in value:
        return value
    try:
        return value.encode("ascii").decode("unicode_escape")
    except UnicodeError:
        return value


def _build_realtime_wake_source(runtime, config, args, *, wake_word: str):
    ear_cfg = config.body.organs.get("ear")
    if ear_cfg is None:
        raise RuntimeError("ear organ not configured")
    capture_cfg = ear_cfg.subfunctions.get("capture")
    asr_cfg = ear_cfg.subfunctions.get("asr")
    if capture_cfg is None or asr_cfg is None:
        raise RuntimeError("ear capture/asr configuration is incomplete")
    sample_rate = int(capture_cfg.driver.extra.get("sample_rate", 16000))
    channels = int(capture_cfg.driver.extra.get("channels", 1))
    frame_ms = max(20, int(args.realtime_wake_frame_ms))
    ring = PcmRingBuffer(
        max_duration_ms=max(frame_ms, int(args.realtime_wake_buffer_ms)),
        sample_rate=sample_rate,
        channels=channels,
    )
    chunk_source = ArecordRawChunkSource(
        device=str(capture_cfg.driver.extra.get("device", "default")),
        sample_rate=sample_rate,
        channels=channels,
        frame_ms=frame_ms,
    )
    capture_worker = RealtimeAudioCaptureWorker(
        ring_buffer=ring,
        chunk_source=chunk_source,
        chunk_duration_ms=frame_ms,
    )
    recognizer = runtime._make_recognizer(asr_cfg)
    detector = RealtimeWakeDetector(
        ring_buffer=ring,
        recognizer=recognizer,
        wake_words=(wake_word,),
        transcript_replacements=asr_cfg.driver.extra.get("transcript_replacements", {}),
        lookback_ms=int(args.realtime_wake_lookback_ms),
        min_buffer_ms=int(args.realtime_wake_min_buffer_ms),
        poll_interval_s=float(args.realtime_wake_poll_interval),
    )
    return RealtimeWakeAudioPipeline(
        ring_buffer=ring,
        capture_worker=capture_worker,
        wake_detector=detector,
    )


if __name__ == "__main__":
    main()
