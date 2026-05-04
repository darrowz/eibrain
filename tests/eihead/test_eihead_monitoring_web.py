from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import threading
from typing import Any, Iterator
from urllib import request
from urllib.error import HTTPError

import pytest

from eihead.monitoring import build_realtime_vision_payload
from eihead.monitoring import build_voice_diagnostics_from_app as exported_build_voice_diagnostics_from_app
from eihead.monitoring.voice import build_voice_diagnostics_from_app
from eihead.monitoring.web import create_handler, create_server
from eihead.mouth import MouthTtsConfig, build_mouth_status
from eihead.protocol import RealtimeVisionObservation, VisionObservation


class FakeMonitorApp:
    def __init__(self, *, health_payload: dict[str, Any] | None = None, fail_status: bool = False) -> None:
        self.health_payload = health_payload
        self.fail_status = fail_status

    def status(self) -> dict[str, Any]:
        if self.fail_status:
            raise RuntimeError("status boom")
        return {
            "ok": True,
            "status": "ok",
            "runtime": "eihead",
            "node_id": "honjia-test",
            "overall_status": "online",
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "schema": "eihead.status_snapshot.v1",
            "node_id": "honjia-test",
            "summary": {"online": 2, "degraded": 0, "offline": 0, "total": 2},
            "capabilities": {
                "camera": {"status": "online"},
                "neck": {"status": "online", "limits": {"yaw_deg": [0, 180], "tilt_deg": None}},
            },
        }

    def health(self) -> dict[str, Any]:
        return self.health_payload or {"ok": True, "status": "ok", "node_id": "honjia-test"}


def _realtime_observation() -> RealtimeVisionObservation:
    return RealtimeVisionObservation(
        ts=12.0,
        source="eihead.honjia.eye.realtime",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_001_000,
        trace_id="trace-eye-live",
        stream_id="front-main",
        camera_id="front",
        status="tracking",
        frame_id="frame-live-1",
        width=1280,
        height=720,
        fps=30.0,
        latency_ms=38.5,
        payload={"simulated": True},
        detections=[
            {
                "label": "person",
                "score": 0.95,
                "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.8},
            },
            {
                "label": "cat",
                "confidence": 0.72,
                "bbox": {"x_min": 0.5, "y_min": 0.3, "x_max": 0.7, "y_max": 0.6},
            },
        ],
        tracked_target={"label": "person", "center_x": 0.51},
        stream={"backend": "hailo8l", "transport": "simulated", "connected": True},
        health={"dropped_frames": 0, "last_frame_age": 0.12},
    )


class RealtimeVisionMethodApp(FakeMonitorApp):
    def vision_realtime(self) -> RealtimeVisionObservation:
        return _realtime_observation()


class EmptyRealtimeVisionMethodApp(FakeMonitorApp):
    def vision_realtime(self) -> None:
        return None


class PlaceholderRealtimeVisionMethodApp(FakeMonitorApp):
    def vision_realtime(self) -> dict[str, object]:
        return {
            "schema": "eihead.eye.realtime_status.v1",
            "mode": "realtime_stream",
            "status": "not_wired",
            "not_wired": True,
            "placeholder": True,
            "message": "detector not wired",
        }


class MultiHookRealtimeVisionApp(FakeMonitorApp):
    eye_realtime = {
        "schema": "eihead.eye.realtime_status.v1",
        "mode": "realtime_stream",
        "status": "not_wired",
        "not_wired": True,
        "placeholder": True,
    }

    def vision_realtime(self) -> RealtimeVisionObservation:
        return _realtime_observation()


class PipelineStatusDictApp(FakeMonitorApp):
    def eye_realtime(self) -> dict[str, object]:
        return {
            "schema": "eihead.eye.realtime_status.v1",
            "mode": "realtime_stream",
            "status": "ok",
            "backend": "gstreamer_hailo",
            "frame_count": 7,
            "detection_count": 3,
            "fps": 27.5,
            "last_frame_id": "frame-42",
            "last_frame_age": 0.333,
            "last_frame_captured_at_ts": 1233.667,
            "top_detection": {
                "label": "person",
                "score": 0.91,
                "bbox": {"x_min": 0.3, "y_min": 0.1, "x_max": 0.6, "y_max": 0.9},
            },
            "detections": [
                {
                    "label": "person",
                    "score": 0.91,
                    "bbox": {"x_min": 0.3, "y_min": 0.1, "x_max": 0.6, "y_max": 0.9},
                },
                {
                    "label": "dog",
                    "confidence": 0.66,
                    "bbox": {"x_min": 0.62, "y_min": 0.24, "x_max": 0.86, "y_max": 0.74},
                },
            ],
            "source": "eihead.eye.realtime",
            "placeholder": False,
            "not_wired": False,
            "compatibility_mode": False,
            "message": "realtime frame processed",
        }


ADAPTER_PIPELINE_STATUS = {
    "schema": "eihead.eye.realtime_status.v1",
    "mode": "realtime_stream",
    "status": "tracking",
    "backend": "gstreamer_hailo",
    "frame_count": 9,
    "detection_count": 2,
    "fps": 26.5,
    "last_frame_id": "frame-adapter-9",
    "last_frame_age": 0.245,
    "last_frame_captured_at_ts": 1233.755,
    "top_detection": {
        "label": "person",
        "score": 0.93,
        "bbox": {"x_min": 0.22, "y_min": 0.11, "x_max": 0.56, "y_max": 0.88},
    },
    "detections": [
        {
            "label": "person",
            "score": 0.93,
            "bbox": {"x_min": 0.22, "y_min": 0.11, "x_max": 0.56, "y_max": 0.88},
        },
        {
            "label": "dog",
            "confidence": 0.61,
            "bbox": {"x_min": 0.61, "y_min": 0.26, "x_max": 0.84, "y_max": 0.72},
        },
    ],
    "pipeline": {"source": "v4l2src", "sink": "appsink", "transport": "gstreamer"},
    "devices": {"camera": "/dev/video0", "hailo": "/dev/hailo0"},
    "readiness_message": "reader/parser wired",
    "parse_error_count": "2",
    "parse_errors": [{"index": 3, "exception": "AttributeError", "message": "bbox missing"}],
    "source": "eihead.eye.adapters",
    "placeholder": "false",
    "not_wired": "false",
    "compatibility_mode": "false",
    "message": "realtime adapter status",
}


class AdapterPayload:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = dict(payload)

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)


class LatestStatusAdapterApp(FakeMonitorApp):
    class Adapter:
        latest_status = AdapterPayload(ADAPTER_PIPELINE_STATUS)

    eye_realtime = Adapter()


class StatusMethodAdapterApp(FakeMonitorApp):
    class Adapter:
        def status(self) -> AdapterPayload:
            return AdapterPayload(ADAPTER_PIPELINE_STATUS)

    eye_realtime = Adapter()


class PollMethodAdapterApp(FakeMonitorApp):
    class Adapter:
        def poll(self) -> AdapterPayload:
            return AdapterPayload(ADAPTER_PIPELINE_STATUS)

    eye_realtime = Adapter()


class ToDictAdapterApp(FakeMonitorApp):
    class Adapter:
        def to_dict(self) -> dict[str, object]:
            return dict(ADAPTER_PIPELINE_STATUS)

    eye_realtime = Adapter()


class NotWiredAdapterApp(FakeMonitorApp):
    class Adapter:
        latest_status = AdapterPayload(
            {
                **ADAPTER_PIPELINE_STATUS,
                "status": "not_wired",
                "placeholder": "true",
                "not_wired": "true",
                "readiness_message": "reader missing",
            }
        )

    eye_realtime = Adapter()


class CompatStaticAdapterApp(FakeMonitorApp):
    class Adapter:
        latest_status = AdapterPayload(
            {
                **ADAPTER_PIPELINE_STATUS,
                "mode": "compat/static",
                "compatibility_mode": "true",
            }
        )

    eye_realtime = Adapter()


class CompatStaticVisionMethodApp(FakeMonitorApp):
    def vision_realtime(self) -> VisionObservation:
        return VisionObservation(
            ts=13.0,
            source="eihead.honjia.eye.compat",
            frame_id="still-1",
            detections=[
                {
                    "label": "person",
                    "score": 0.8,
                    "bbox": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.8, "y_max": 0.8},
                }
            ],
        )


class RecentMethodApp(FakeMonitorApp):
    def recent_actions(self) -> list[dict[str, Any]]:
        return [
            {"action_id": "a1", "type": "speak", "status": "accepted"},
            {"action_id": "a2", "type": "move_head", "status": "skipped"},
        ]


@dataclass(slots=True)
class ActionRecord:
    action_id: str
    action_type: str
    status: str


class ActionLogAttributeApp(FakeMonitorApp):
    action_log = [ActionRecord("a3", "capture_frame", "accepted")]


class FallbackHealthApp(FakeMonitorApp):
    health = None


class SnapshotVoiceApp(FakeMonitorApp):
    def snapshot(self) -> dict[str, object]:
        return {
            "runtime": "eihead",
            "node_id": "honjia-test",
            "voice_dialogue": {
                "enabled": True,
                "running": True,
                "phase": "speaking",
                "last_status": "completed",
                "last_transcript": "你好 honjia",
                "last_reply": "你好",
                "last_stage_latency_ms": {"capture": 85.0, "llm": 415.0, "tts": 240.0},
                "last_bottleneck_stage": "llm",
                "last_bottleneck_ms": 415.0,
                "last_completed_turn": {"transcript": "你好 honjia", "reply": "你好"},
            },
            "organs": {
                "ear": {
                    "organ": "ear",
                    "health": "healthy",
                    "subfunctions": {
                        "capture": {"health": "healthy", "details": {"device": "default"}},
                        "asr": {"health": "healthy", "details": {"provider": "faster_whisper"}},
                    },
                },
                "mouth": {
                    "organ": "mouth",
                    "health": "healthy",
                    "subfunctions": {
                        "tts_playback": {
                            "health": "healthy",
                            "details": {
                                "backend": "minimax",
                                "model": "speech-2.8-hd",
                                "voice_id": "female-shaonv",
                                "text_preview": "你好 honjia",
                            },
                        }
                    },
                },
            },
        }


class WaitingEarVoiceApp(FakeMonitorApp):
    def voice_status(self) -> dict[str, object]:
        return {
            "ear": {
                "status": "waiting_for_data",
                "live_probe_skipped": True,
                "readiness_message": "ear waiting for data",
            },
            "mouth": {
                "status": "noop",
                "backend": "noop",
                "readiness_message": "mouth noop backend",
            },
            "dialogue": {
                "phase": "idle",
                "last_status": "waiting_for_voice",
            },
        }


class NativeVoiceRealtimeApp(FakeMonitorApp):
    def voice_realtime(self) -> dict[str, object]:
        return {
            "schema": "eihead.monitor.voice_realtime.v1",
            "status": "wired",
            "ear": {
                "status": "ok",
                "provider": "faster_whisper",
                "readiness_message": "ear wired",
            },
            "mouth": {
                "status": "ok",
                "backend": "minimax",
                "model": "speech-2.8-hd",
                "voice_id": "female-shaonv",
                "text_preview": "你好 honjia",
            },
            "dialogue": {
                "phase": "speaking",
                "last_status": "completed",
                "last_completed_turn": {"transcript": "你好 honjia", "reply": "你好"},
            },
            "last_stage_latency_ms": {"capture": 70.0, "llm": 390.0, "tts": 250.0},
            "last_bottleneck_stage": "llm",
            "last_bottleneck_ms": 390.0,
            "readiness_message": "voice loop wired",
        }


class RoundSchedulerInterruptVoiceApp(FakeMonitorApp):
    def voice_realtime(self) -> dict[str, object]:
        return {
            "schema": "eihead.monitor.voice_realtime.v1",
            "status": "running",
            "ear": {
                "status": "ok",
                "provider": "faster_whisper",
            },
            "mouth": {
                "status": "ok",
                "backend": "minimax",
                "model": "speech-2.8-hd",
            },
            "dialogue": {
                "phase": "listening",
                "last_status": "interrupted",
                "current_round_id": "round-42",
                "scheduler_state": {
                    "state": "stale",
                    "active_round_id": "round-41",
                    "stale": True,
                    "last_tick_age_ms": 2500.0,
                },
                "last_stage_latency_ms": {"capture": 30.0, "scheduler": 11.0, "interrupt": 4.0},
            },
            "current_cancellation_token": {
                "token_id": "cancel-42",
                "cancelled": True,
            },
            "interrupt_count": 2,
            "interrupted_round_count": 1,
            "last_interrupt": {
                "round_id": "round-42",
                "reason": "barge_in",
                "stale": True,
            },
            "microfeedback": {
                "last": "too_slow",
                "score": -1,
            },
        }


class HistoricInterruptVoiceApp(FakeMonitorApp):
    def voice_realtime(self) -> dict[str, object]:
        return {
            "schema": "eihead.monitor.voice_realtime.v1",
            "status": "running",
            "ear": {"status": "ok", "provider": "faster_whisper"},
            "mouth": {"status": "ok", "backend": "minimax", "model": "speech-2.8-hd"},
            "dialogue": {
                "phase": "speaking",
                "last_status": "completed",
                "current_round_id": "round-43",
                "current_cancellation_token": "cancel-43",
                "scheduler_state": {"state": "active", "round_id": "round-43"},
                "interrupt_active": False,
                "interrupted_round_count": 2,
                "last_interrupt": {"round_id": "round-41", "reason": "barge_in"},
                "last_completed_turn": {"transcript": "你好", "reply": "你好"},
                "last_stage_latency_ms": {"listen_asr": 100.0, "think": 200.0, "speak": 300.0, "total": 600.0},
            },
        }


class CleanRoundVoiceApp(FakeMonitorApp):
    def voice_realtime(self) -> dict[str, object]:
        return {
            "schema": "eihead.monitor.voice_realtime.v1",
            "status": "running",
            "ear": {"status": "ok", "provider": "faster_whisper"},
            "mouth": {"status": "ok", "backend": "minimax", "model": "speech-2.8-hd"},
            "dialogue": {
                "phase": "listening",
                "last_status": "listening",
                "current_round_id": "round-44",
                "current_cancellation_token": "cancel-44",
                "scheduler_state": {"state": "active", "round_id": "round-44"},
                "interrupt_active": False,
                "interrupted_round_count": 0,
            },
        }


class ClosedLoopRealtimeVoiceApp(FakeMonitorApp):
    def voice_realtime(self) -> dict[str, object]:
        return {
            "schema": "eihead.monitor.voice_realtime.v1",
            "status": "running",
            "ear": {"status": "ok", "provider": "faster_whisper"},
            "mouth": {"status": "ok", "backend": "minimax", "model": "speech-2.8-hd"},
            "dialogue": {
                "phase": "speaking_stream",
                "last_status": "speaking_stream",
                "current_round_id": "round-live-1",
                "current_cancellation_token": "tok-live-1",
                "scheduler_state": {"state": "active", "round_id": "round-live-1"},
                "interrupt_active": False,
                "interrupted_round_count": 0,
            },
            "realtime_session": {
                "session_id": "session-live-1",
                "actor_id": "darrow",
                "round_id": "round-live-1",
                "roundId": "round-live-1",
                "cancellation_token": "tok-live-1",
                "cancellationToken": "tok-live-1",
                "phase": "speaking_stream",
                "status": "reply_delta",
                "complete": False,
                "closed_loop": False,
                "closed_loop_state": {
                    "final_asr": True,
                    "first_reply_delta": True,
                    "first_speech": True,
                    "complete": False,
                },
                "transcript_final": "你好鸿途",
                "reply_text": "我在。",
                "latency_ms": {
                    "final_asr": 800.0,
                    "first_reply_token": 1100.0,
                    "first_speech": 1300.0,
                    "final_asr_to_first_reply_token": 300.0,
                    "first_reply_token_to_first_speech": 200.0,
                },
                "event_count": 4,
                "events": [
                    {"event_type": "listening_started", "lane": "listening", "round_id": "round-live-1"},
                    {"event_type": "asr_final", "lane": "listening", "transcript": "你好鸿途"},
                    {
                        "event_type": "reply_delta",
                        "lane": "slow_thinking",
                        "reply_delta": "我在。",
                        "round_id": "round-live-1",
                    },
                    {"event_type": "tts_started", "lane": "speaking", "round_id": "round-live-1"},
                ],
                "cancellation_chain": [
                    {
                        "target": "generation",
                        "event_type": "generation_cancelled",
                        "reason": "new_round",
                        "round_id": "round-old",
                        "cancellation_token": "tok-old",
                    }
                ],
            },
        }


class HistoricalClosedLoopOnlyVoiceApp(FakeMonitorApp):
    def voice_realtime(self) -> dict[str, object]:
        return {
            "schema": "eihead.monitor.voice_realtime.v1",
            "status": "completed",
            "realtime_session": {
                "session_id": "session-old",
                "round_id": "round-old",
                "cancellation_token": "tok-old",
                "phase": "completed",
                "status": "completed",
                "complete": True,
                "closed_loop": True,
                "closed_loop_state": {
                    "final_asr": True,
                    "first_reply_delta": True,
                    "first_speech": True,
                    "complete": True,
                },
                "latency_ms": {"first_reply_token": 1100.0, "first_speech": 1300.0, "total": 2000.0},
                "event_count": 6,
                "events": [{"event_type": "complete", "lane": "complete", "reply_delta": "旧回复"}],
            },
        }


class FallbackMouthVoiceApp(FakeMonitorApp):
    def voice_status(self) -> dict[str, object]:
        return {
            "ear": {
                "status": "ok",
                "provider": "faster_whisper",
                "readiness_message": "ear wired",
            },
            "mouth": build_mouth_status(
                config=MouthTtsConfig(provider="espeak", model="espeak-ng", voice_id="zh"),
                status="completed",
                details={"text_preview": "fallback done", "text_char_count": 13},
            ).to_dict(),
            "dialogue": {
                "phase": "speaking",
                "last_status": "completed",
                "last_completed_turn": {"transcript": "你好", "reply": "你好"},
            },
        }


@contextmanager
def running_server(app: Any, **kwargs: Any) -> Iterator[tuple[str, object, threading.Thread]]:
    server = create_server(app, host="127.0.0.1", port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", server, thread
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_root_returns_simple_html_with_not_wired_recent_actions() -> None:
    app = FakeMonitorApp()

    with running_server(app, clock=lambda: 123.0) as (base_url, _server, _thread):
        status_code, headers, body = read_text(f"{base_url}/")

    assert status_code == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "eihead native monitor" in body
    assert "honjia-test" in body
    assert "/api/status" in body
    assert "/api/capabilities" in body
    assert "/api/actions/recent" in body
    assert "not_wired" in body


def test_health_status_and_capabilities_return_json_objects() -> None:
    app = FakeMonitorApp()

    with running_server(app) as (base_url, _server, _thread):
        health_code, health_headers, health_payload = read_json(f"{base_url}/health")
        status_code, _, status_payload = read_json(f"{base_url}/api/status")
        capabilities_code, _, capabilities_payload = read_json(f"{base_url}/api/capabilities?source=test")

    assert health_code == 200
    assert health_headers["Content-Type"].startswith("application/json")
    assert health_payload == {"ok": True, "status": "ok", "node_id": "honjia-test"}
    assert status_code == 200
    assert status_payload["runtime"] == "eihead"
    assert capabilities_code == 200
    assert capabilities_payload["capabilities"]["neck"]["limits"]["tilt_deg"] is None


def test_voice_helper_uses_body_snapshot_and_exposes_latency_bottleneck_and_last_turn() -> None:
    payload = build_voice_diagnostics_from_app(SnapshotVoiceApp(), timestamp=432.1)

    assert payload["schema"] == "eihead.monitor.voice_realtime.v1"
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["source"] == "snapshot"
    assert payload["captured_at_ts"] == 432.1
    assert payload["dialogue"]["phase"] == "speaking"
    assert payload["ear"]["provider"] == "faster_whisper"
    assert payload["mouth"]["backend"] == "minimax"
    assert payload["latency"]["stage_latency_ms"]["llm"] == 415.0
    assert payload["bottleneck"] == {"stage": "llm", "latency_ms": 415.0}
    assert payload["last_turn"] == {"transcript": "你好 honjia", "reply": "你好"}


def test_voice_helper_marks_waiting_ear_and_noop_mouth_as_truthful_not_fake_ok() -> None:
    payload = build_voice_diagnostics_from_app(WaitingEarVoiceApp(), timestamp=433.0)

    assert payload["status"] == "degraded"
    assert payload["wired"] is False
    assert payload["not_wired"] is False
    assert payload["ear"]["status"] == "waiting_for_data"
    assert payload["ear"]["state"] == "degraded"
    assert payload["mouth"]["backend"] == "noop"
    assert payload["mouth"]["state"] == "not_wired"
    assert "waiting" in payload["readiness_message"]


def test_voice_helper_reports_not_wired_without_runtime_hook() -> None:
    payload = build_voice_diagnostics_from_app(FakeMonitorApp(), timestamp=434.0)

    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["not_wired"] is True
    assert payload["source"] is None
    assert "not wired" in payload["readiness_message"] or "not_wired" in payload["readiness_message"]


def test_realtime_vision_helper_standardizes_observation_payload() -> None:
    payload = build_realtime_vision_payload(
        _realtime_observation(),
        timestamp=321.0,
        source="vision_realtime",
    )

    assert payload["schema"] == "eihead.monitor.vision_realtime.v1"
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["source"] == "vision_realtime"
    assert payload["channel"] == "eye.realtime"
    assert payload["aliases"] == ["vision.realtime"]
    assert payload["primary_mode"] == "realtime"
    assert payload["compat_static"] == {"mode": "compat/static", "primary": False}
    assert payload["captured_at_ts"] == 321.0
    assert payload["observation"]["kind"] == "realtime_vision_observation"
    assert payload["observation"]["mode"] == "realtime"
    assert payload["observation"]["primary_mode"] is True
    assert payload["frame_id"] == "frame-live-1"
    assert payload["fps"] == 30.0
    assert payload["backend"] == "hailo8l"
    assert payload["last_frame_age"] == 0.12
    assert payload["boxes"] == [
        {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.8},
        {"x_min": 0.5, "y_min": 0.3, "x_max": 0.7, "y_max": 0.6},
    ]
    assert payload["scores"] == [0.95, 0.72]
    assert payload["top_detection"]["label"] == "person"
    assert payload["diagnostic"]["status"] == "wired"
    assert payload["diagnostic"]["wired"] is True
    assert payload["diagnostic"]["not_wired"] is False
    assert payload["diagnostic"]["compat_static"] is False
    assert payload["diagnostic"]["stale"] is False
    assert payload["diagnostic"]["detection_count"] == 2
    assert payload["diagnostic"]["detection_count_raw"] == 2
    assert payload["diagnostic"]["score_threshold"] == 0.0
    assert payload["diagnostic"]["detection_score_threshold"] == 0.0
    assert payload["diagnostic"]["top_k"] is None
    assert payload["diagnostic"]["frame_interval_ms"] is None
    assert payload["diagnostic"]["jitter_guard"] is None
    assert payload["diagnostic"]["hooks_used"] is None
    assert payload["score_threshold"] == 0.0
    assert payload["top_k"] is None
    assert payload["frame_interval_ms"] is None
    assert payload["jitter_guard"] is None
    assert payload["hooks_used"] is None


def test_realtime_vision_api_and_html_render_wired_payload() -> None:
    app = RealtimeVisionMethodApp()

    with running_server(app, clock=lambda: 654.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")
        alias_code, _, alias_payload = read_json(f"{base_url}/api/eye/realtime")
        html_code, html_headers, body = read_text(f"{base_url}/")

    assert status_code == 200
    assert payload["schema"] == "eihead.monitor.vision_realtime.v1"
    assert payload["status"] == "wired"
    assert payload["channel"] == "eye.realtime"
    assert payload["aliases"] == ["vision.realtime"]
    assert payload["primary_mode"] == "realtime"
    assert payload["compat_static"]["mode"] == "compat/static"
    assert payload["observation"]["stream_id"] == "front-main"
    assert payload["boxes"][0]["x_min"] == 0.1
    assert payload["scores"] == [0.95, 0.72]
    assert payload["diagnostic"]["top_detection"]["score"] == 0.95
    assert payload["diagnostic"]["detection_count_raw"] == 2
    assert payload["score_threshold"] == 0.0
    assert payload["top_k"] is None
    assert payload["frame_interval_ms"] is None
    assert payload["jitter_guard"] is None
    assert payload["hooks_used"] is None
    assert alias_code == 200
    assert alias_payload["observation"] == payload["observation"]
    assert alias_payload["diagnostic"] == payload["diagnostic"]
    assert html_code == 200
    assert html_headers["Content-Type"].startswith("text/html")
    assert "Realtime Vision" in body
    assert "Realtime Vision Diagnostic" in body
    assert "Top detection" in body
    assert "Frame age" in body
    assert "Backend" in body
    assert "boxes" in body
    assert "scores" in body
    assert "Frame interval" in body
    assert "Jitter guard" in body
    assert "Top K" in body
    assert "Score threshold" in body
    assert "Hooks used" in body
    assert "Pipeline" in body
    assert "Devices" in body
    assert "Readiness" in body
    assert "Parse errors" in body
    assert "/api/vision/realtime" in body
    assert "eye.realtime" in body


def test_voice_api_aliases_and_html_render_voice_diagnostics() -> None:
    app = NativeVoiceRealtimeApp()

    with running_server(app, clock=lambda: 654.5) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/voice/realtime")
        alias_code, _, alias_payload = read_json(f"{base_url}/api/audio/realtime")
        html_code, html_headers, body = read_text(f"{base_url}/")

    assert status_code == 200
    assert alias_code == 200
    assert payload == alias_payload
    assert payload["status"] == "wired"
    assert payload["mouth"]["backend"] == "minimax"
    assert payload["mouth"]["model"] == "speech-2.8-hd"
    assert payload["latency"]["stage_latency_ms"]["tts"] == 250.0
    assert payload["bottleneck"] == {"stage": "llm", "latency_ms": 390.0}
    assert payload["last_turn"]["transcript"] == "你好 honjia"
    assert html_code == 200
    assert html_headers["Content-Type"].startswith("text/html")
    assert "Voice Diagnostics" in body
    assert "/api/voice/realtime" in body
    assert "MiniMax" in body or "minimax" in body


def test_voice_helper_exposes_round_scheduler_interrupt_and_microfeedback_state() -> None:
    payload = build_voice_diagnostics_from_app(RoundSchedulerInterruptVoiceApp(), timestamp=656.0)

    assert payload["status"] == "degraded"
    assert payload["wired"] is False
    assert payload["round"]["current_round_id"] == "round-42"
    assert payload["round"]["current_cancellation_token"] == {"token_id": "cancel-42", "cancelled": True}
    assert payload["scheduler"]["state"] == "stale"
    assert payload["scheduler"]["active_round_id"] == "round-41"
    assert payload["scheduler"]["stale"] is True
    assert payload["interruption"]["interrupted"] is True
    assert payload["interruption"]["interrupt_count"] == 2
    assert payload["interruption"]["interrupted_round_count"] == 1
    assert payload["interruption"]["last_interrupt"]["reason"] == "barge_in"
    assert payload["interruption"]["stale"] is True
    assert payload["microfeedback"] == {"last": "too_slow", "score": -1}
    assert payload["latency"]["stage_latency_ms"]["scheduler"] == 11.0
    assert "interrupted" in payload["readiness_message"]
    assert "stale" in payload["readiness_message"]


def test_voice_api_and_html_render_round_scheduler_interrupt_cards() -> None:
    app = RoundSchedulerInterruptVoiceApp()

    with running_server(app, clock=lambda: 656.5) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/voice/realtime")
        alias_code, _, alias_payload = read_json(f"{base_url}/api/audio/realtime")
        html_code, html_headers, body = read_text(f"{base_url}/")

    assert status_code == 200
    assert alias_code == 200
    assert payload == alias_payload
    assert payload["round"]["current_round_id"] == "round-42"
    assert payload["scheduler"]["state"] == "stale"
    assert payload["interruption"]["interrupted"] is True
    assert html_code == 200
    assert html_headers["Content-Type"].startswith("text/html")
    assert "Round" in body
    assert "round-42" in body
    assert "Scheduler" in body
    assert "stale" in body
    assert "Interrupts" in body
    assert "barge_in" in body
    assert "Microfeedback" in body
    assert "too_slow" in body


def test_voice_helper_keeps_historic_interrupt_visible_without_degrading_current_round() -> None:
    payload = build_voice_diagnostics_from_app(HistoricInterruptVoiceApp(), timestamp=656.8)

    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["interruption"]["state"] == "history"
    assert payload["interruption"]["active"] is False
    assert payload["interruption"]["interrupted_round_count"] == 2
    assert payload["interruption"]["last_interrupt"]["reason"] == "barge_in"
    assert payload["latency"]["total_ms"] == 600.0


def test_voice_helper_shows_clean_interrupt_state_when_no_interrupts_seen() -> None:
    payload = build_voice_diagnostics_from_app(CleanRoundVoiceApp(), timestamp=656.9)

    assert payload["status"] == "wired"
    assert payload["interruption"]["state"] == "clear"
    assert payload["interruption"]["component_state"] == "wired"
    assert payload["interruption"]["interrupted"] is False
    assert payload["interruption"]["interrupted_round_count"] == 0


def test_voice_helper_exposes_closed_loop_realtime_session_fields() -> None:
    payload = build_voice_diagnostics_from_app(ClosedLoopRealtimeVoiceApp(), timestamp=657.0)

    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["realtime_session"]["session_id"] == "session-live-1"
    assert payload["realtime_session"]["round_id"] == "round-live-1"
    assert payload["realtime_session"]["latency_ms"]["first_reply_token"] == 1100.0
    assert payload["realtime_events"][-2]["reply_delta"] == "我在。"
    assert payload["event_count"] == 4
    assert payload["closed_loop_state"] == {
        "final_asr": True,
        "first_reply_delta": True,
        "first_speech": True,
        "complete": False,
    }
    assert payload["last_reply_delta"] == "我在。"
    assert payload["latency"]["stage_latency_ms"]["first_reply_token"] == 1100.0
    assert payload["latency"]["stage_latency_ms"]["first_speech"] == 1300.0
    assert payload["latency"]["stage_latency_ms"]["final_asr_to_first_reply_token"] == 300.0
    assert payload["cancellation_chain"] == [
        {
            "target": "generation",
            "event_type": "generation_cancelled",
            "reason": "new_round",
            "round_id": "round-old",
            "cancellation_token": "tok-old",
        }
    ]


def test_voice_api_and_html_render_closed_loop_realtime_fields() -> None:
    app = ClosedLoopRealtimeVoiceApp()

    with running_server(app, clock=lambda: 657.5) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/voice/realtime")
        html_code, html_headers, body = read_text(f"{base_url}/")

    assert status_code == 200
    assert payload["realtime_session"]["session_id"] == "session-live-1"
    assert payload["event_count"] == 4
    assert payload["last_reply_delta"] == "我在。"
    assert payload["latency"]["stage_latency_ms"]["first_speech"] == 1300.0
    assert html_code == 200
    assert html_headers["Content-Type"].startswith("text/html")
    assert "Closed loop" in body
    assert "first_reply_delta" in body
    assert "Realtime events" in body
    assert "4" in body
    assert "Last reply delta" in body
    assert "我在。" in body
    assert "First reply token" in body
    assert "1100.0ms" in body
    assert "First speech" in body
    assert "1300.0ms" in body
    assert "Cancellation chain" in body
    assert "generation" in body


def test_voice_helper_keeps_historical_closed_loop_session_not_wired_without_live_components() -> None:
    payload = build_voice_diagnostics_from_app(HistoricalClosedLoopOnlyVoiceApp(), timestamp=657.8)

    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["not_wired"] is True
    assert payload["realtime_session"]["session_id"] == "session-old"
    assert payload["event_count"] == 6
    assert payload["closed_loop_state"]["complete"] is True


def test_voice_helper_does_not_promote_fallback_mouth_to_wired() -> None:
    assert exported_build_voice_diagnostics_from_app is build_voice_diagnostics_from_app

    payload = build_voice_diagnostics_from_app(FallbackMouthVoiceApp(), timestamp=655.5)

    assert payload["status"] == "degraded"
    assert payload["wired"] is False
    assert payload["ear"]["state"] == "wired"
    assert payload["mouth"]["backend"] == "espeak"
    assert payload["mouth"]["health"] == "degraded"
    assert payload["mouth"]["state"] == "degraded"
    assert payload["mouth"]["tts_playback"] is None


def test_realtime_vision_api_prefers_later_live_hook_over_earlier_placeholder() -> None:
    app = MultiHookRealtimeVisionApp()

    with running_server(app, clock=lambda: 655.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["source"] == "vision_realtime"
    assert payload["observation"]["stream_id"] == "front-main"


def test_realtime_vision_api_normalizes_pipeline_status_dict() -> None:
    app = PipelineStatusDictApp()

    with running_server(app, clock=lambda: 1234.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/eye/realtime")

    assert status_code == 200
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["source"] == "eye_realtime"
    assert payload["frame_id"] == "frame-42"
    assert payload["fps"] == 27.5
    assert payload["backend"] == "gstreamer_hailo"
    assert payload["last_frame_age"] == 0.333
    assert payload["boxes"] == [
        {"x_min": 0.3, "y_min": 0.1, "x_max": 0.6, "y_max": 0.9},
        {"x_min": 0.62, "y_min": 0.24, "x_max": 0.86, "y_max": 0.74},
    ]
    assert payload["scores"] == [0.91, 0.66]
    assert payload["score_threshold"] == 0.0
    assert payload["top_k"] is None
    assert payload["frame_interval_ms"] is None
    assert payload["jitter_guard"] is None
    assert payload["hooks_used"] is None
    assert payload["top_detection"]["label"] == "person"
    assert payload["diagnostic"]["status"] == "wired"
    assert payload["diagnostic"]["pipeline_status"] == "ok"
    assert payload["diagnostic"]["wired"] is True
    assert payload["diagnostic"]["not_wired"] is False
    assert payload["diagnostic"]["placeholder"] is False
    assert payload["diagnostic"]["compat_static"] is False
    assert payload["diagnostic"]["stale"] is False
    assert payload["diagnostic"]["backend"] == "gstreamer_hailo"
    assert payload["diagnostic"]["frame_id"] == "frame-42"
    assert payload["diagnostic"]["fps"] == 27.5
    assert payload["diagnostic"]["last_frame_age"] == 0.333
    assert payload["diagnostic"]["last_frame_age_s"] == 0.333
    assert payload["diagnostic"]["detection_count"] == 2
    assert payload["diagnostic"]["detection_count_raw"] == 2
    assert payload["diagnostic"]["detection_score_threshold"] == 0.0
    assert payload["diagnostic"]["score_threshold"] == 0.0
    assert payload["diagnostic"]["top_k"] is None
    assert payload["diagnostic"]["frame_interval_ms"] is None
    assert payload["diagnostic"]["jitter_guard"] is None
    assert payload["diagnostic"]["hooks_used"] is None


@pytest.mark.parametrize(
    "app_cls",
    [
        LatestStatusAdapterApp,
        StatusMethodAdapterApp,
        PollMethodAdapterApp,
        ToDictAdapterApp,
    ],
)
def test_realtime_vision_api_accepts_live_adapter_payload_forms(app_cls: type[FakeMonitorApp]) -> None:
    app = app_cls()

    with running_server(app, clock=lambda: 1234.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/eye/realtime")

    assert status_code == 200
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["backend"] == "gstreamer_hailo"
    assert payload["frame_id"] == "frame-adapter-9"
    assert payload["pipeline"] == {"source": "v4l2src", "sink": "appsink", "transport": "gstreamer"}
    assert payload["devices"] == {"camera": "/dev/video0", "hailo": "/dev/hailo0"}
    assert payload["readiness_message"] == "reader/parser wired"
    assert payload["parse_error_count"] == 2
    assert payload["parse_errors"] == [{"index": 3, "exception": "AttributeError", "message": "bbox missing"}]
    assert payload["diagnostic"]["pipeline"] == payload["pipeline"]
    assert payload["diagnostic"]["devices"] == payload["devices"]
    assert payload["diagnostic"]["readiness_message"] == "reader/parser wired"
    assert payload["diagnostic"]["parse_error_count"] == 2
    assert payload["diagnostic"]["parse_errors"] == payload["parse_errors"]


def test_realtime_vision_payload_filters_detections_by_threshold_and_top_k() -> None:
    payload = build_realtime_vision_payload(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "status": "ok",
            "frame_id": "filtered-1",
            "score_threshold": 0.7,
            "top_k": 2,
            "frame_interval_ms": 33,
            "jitter_guard": True,
            "hooks_used": ["model-a", "tracker-b"],
            "fps": 30.0,
            "detections": [
                {"label": "person", "score": 0.95, "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2}},
                {"label": "cat", "confidence": 0.72, "bbox": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.3, "y_max": 0.3}},
                {"label": "dog", "score": 0.68, "bbox": {"x_min": 0.3, "y_min": 0.3, "x_max": 0.4, "y_max": 0.4}},
            ],
        },
        timestamp=777.0,
        source="eye_realtime",
    )

    assert payload["status"] == "wired"
    assert payload["score_threshold"] == 0.7
    assert payload["top_k"] == 2
    assert payload["diagnostic"]["detection_count"] == 2
    assert payload["diagnostic"]["detection_count_raw"] == 3
    assert payload["scores"] == [0.95, 0.72]
    assert payload["boxes"] == [
        {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2},
        {"x_min": 0.2, "y_min": 0.2, "x_max": 0.3, "y_max": 0.3},
    ]
    assert payload["frame_interval_ms"] == 33.0
    assert payload["jitter_guard"] is True
    assert payload["hooks_used"] == ["model-a", "tracker-b"]
    assert payload["diagnostic"]["detection_count"] == 2
    assert payload["diagnostic"]["detection_count_raw"] == 3
    assert payload["diagnostic"]["top_detection"]["label"] == "person"


def test_realtime_vision_api_reports_not_wired_without_runtime_hook() -> None:
    app = FakeMonitorApp()

    with running_server(app, clock=lambda: 987.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["schema"] == "eihead.monitor.vision_realtime.v1"
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["source"] is None
    assert payload["captured_at_ts"] == 987.0
    assert payload["primary_mode"] == "realtime"
    assert payload["compat_static"] == {"mode": "compat/static", "primary": False}
    assert payload["observation"] is None
    assert payload["not_wired"] is True
    assert payload["boxes"] == []
    assert payload["scores"] == []
    assert payload["diagnostic"]["not_wired"] is True
    assert "eye.realtime" in payload["message"]


def test_realtime_vision_api_reports_not_wired_when_runtime_hook_has_no_payload() -> None:
    app = EmptyRealtimeVisionMethodApp()

    with running_server(app, clock=lambda: 988.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["source"] == "vision_realtime"
    assert payload["observation"] is None


def test_realtime_vision_api_does_not_promote_placeholder_payload_to_wired() -> None:
    app = PlaceholderRealtimeVisionMethodApp()

    with running_server(app, clock=lambda: 989.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["source"] == "vision_realtime"
    assert payload["observation"]["status"] == "not_wired"
    assert payload["observation"]["placeholder"] is True
    assert payload["not_wired"] is True
    assert payload["diagnostic"]["placeholder"] is True
    assert payload["diagnostic"]["not_wired"] is True
    assert payload["diagnostic"]["compat_static"] is False
    assert "not ready" in payload["message"]


def test_realtime_vision_api_parses_string_false_flags_as_false() -> None:
    payload = build_realtime_vision_payload(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "ok",
            "not_wired": "false",
            "placeholder": "false",
            "compatibility_mode": "false",
            "detections": [{"label": "person", "score": 0.8}],
        },
        timestamp=991.0,
        source="eye_realtime",
    )

    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["not_wired"] is False
    assert payload["placeholder"] is False
    assert payload["compat_static_active"] is False


def test_realtime_vision_api_keeps_live_adapter_status_wired_when_flags_are_false_strings() -> None:
    payload = build_realtime_vision_payload(
        ADAPTER_PIPELINE_STATUS,
        timestamp=991.5,
        source="eye_realtime",
    )

    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["not_wired"] is False
    assert payload["placeholder"] is False
    assert payload["compat_static_active"] is False


def test_realtime_vision_api_does_not_promote_not_wired_adapter_status() -> None:
    app = NotWiredAdapterApp()

    with running_server(app, clock=lambda: 1234.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/eye/realtime")

    assert status_code == 200
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["readiness_message"] == "reader missing"
    assert payload["diagnostic"]["not_wired"] is True
    assert payload["diagnostic"]["placeholder"] is True


def test_realtime_vision_api_rejects_compat_static_adapter_status() -> None:
    app = CompatStaticAdapterApp()

    with running_server(app, clock=lambda: 1234.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/eye/realtime")

    assert status_code == 200
    assert payload["status"] == "compat_static"
    assert payload["wired"] is False
    assert payload["compat_static_active"] is True
    assert payload["diagnostic"]["compat_static"] is True


def test_realtime_vision_api_does_not_promote_static_vision_observation_to_wired() -> None:
    app = CompatStaticVisionMethodApp()

    with running_server(app, clock=lambda: 990.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "compat_static"
    assert payload["wired"] is False
    assert payload["source"] == "vision_realtime"
    assert payload["observation"]["kind"] == "vision_observation"
    assert payload["observation"]["mode"] == "compat/static"
    assert payload["compat_static_active"] is True
    assert payload["diagnostic"]["compat_static"] is True
    assert payload["diagnostic"]["not_wired"] is False
    assert payload["boxes"] == [{"x_min": 0.2, "y_min": 0.2, "x_max": 0.8, "y_max": 0.8}]
    assert payload["scores"] == [0.8]
    assert "compat/static" in payload["message"]


def test_recent_actions_uses_runtime_recent_actions_method() -> None:
    app = RecentMethodApp()

    with running_server(app, clock=lambda: 456.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/actions/recent")

    assert status_code == 200
    assert payload["schema"] == "eihead.monitor.recent_actions.v1"
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["source"] == "recent_actions"
    assert payload["captured_at_ts"] == 456.0
    assert payload["count"] == 2
    assert payload["actions"][0]["action_id"] == "a1"


def test_recent_actions_alias_and_action_log_attribute_are_supported() -> None:
    app = ActionLogAttributeApp()

    with running_server(app) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/recent-actions")

    assert status_code == 200
    assert payload["status"] == "wired"
    assert payload["source"] == "action_log"
    assert payload["count"] == 1
    assert payload["actions"] == [
        {"action_id": "a3", "action_type": "capture_frame", "status": "accepted"}
    ]


def test_recent_actions_reports_not_wired_when_runtime_exposes_no_log() -> None:
    app = FakeMonitorApp()

    with running_server(app, clock=lambda: 789.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/api/actions/recent")

    assert status_code == 200
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["source"] is None
    assert payload["captured_at_ts"] == 789.0
    assert payload["actions"] == []
    assert "does not expose" in payload["message"]


def test_health_falls_back_to_status_and_can_return_503() -> None:
    app = FallbackHealthApp(health_payload=None)

    with running_server(app, clock=lambda: 111.0) as (base_url, _server, _thread):
        status_code, _, payload = read_json(f"{base_url}/health")

    assert status_code == 200
    assert payload["source"] == "status"
    assert payload["checked_at_ts"] == 111.0

    unhealthy_app = FakeMonitorApp(health_payload={"ok": False, "status": "offline", "node_id": "honjia-test"})
    with running_server(unhealthy_app) as (base_url, _server, _thread):
        error_code, error_payload = read_error_json(f"{base_url}/health")

    assert error_code == 503
    assert error_payload["ok"] is False
    assert error_payload["status"] == "offline"


def test_unknown_path_wrong_method_and_runtime_errors_are_json() -> None:
    with running_server(FakeMonitorApp()) as (base_url, _server, _thread):
        not_found_code, not_found_payload = read_error_json(f"{base_url}/missing")
        method_code, method_payload = read_error_json(f"{base_url}/api/status", method="POST")

    assert not_found_code == 404
    assert not_found_payload["error"]["code"] == "not_found"
    assert method_code == 405
    assert method_payload["error"]["code"] == "method_not_allowed"

    with running_server(FakeMonitorApp(fail_status=True)) as (base_url, _server, _thread):
        runtime_code, runtime_payload = read_error_json(f"{base_url}/api/status")

    assert runtime_code == 500
    assert runtime_payload["error"]["code"] == "internal_error"
    assert runtime_payload["error"]["details"]["exception"] == "RuntimeError"


def test_server_wrapper_shutdown_stops_serve_forever_cleanly() -> None:
    server = create_server(FakeMonitorApp(), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    thread.start()
    read_json(f"http://{server.server_address[0]}:{server.server_address[1]}/health")
    server.shutdown()
    thread.join(timeout=2.0)
    server.server_close()

    assert not thread.is_alive()


def test_handler_factory_validates_monitor_contract() -> None:
    with pytest.raises(TypeError, match="missing required callables"):
        create_handler(object())


def read_json(url: str) -> tuple[int, Any, Any]:
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=2.0) as response:
        return response.status, response.headers, json.loads(response.read().decode("utf-8"))


def read_text(url: str) -> tuple[int, Any, str]:
    req = request.Request(url, headers={"Accept": "text/html"})
    with request.urlopen(req, timeout=2.0) as response:
        return response.status, response.headers, response.read().decode("utf-8")


def read_error_json(url: str, *, method: str = "GET") -> tuple[int, dict[str, Any]]:
    req = request.Request(url, method=method, headers={"Accept": "application/json"})
    with pytest.raises(HTTPError) as exc:
        request.urlopen(req, timeout=2.0)
    return exc.value.code, json.loads(exc.value.read().decode("utf-8"))
