from __future__ import annotations

from array import array
import time

from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth
from eibrain.protocol.actions import StopSpeechAction
from eibrain.protocol.outcomes import ActionExecuted


def test_voice_realtime_default_runtime_does_not_report_wired() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()

    payload = runtime.voice_realtime()

    assert payload["schema"] == "eihead.monitor.voice_realtime.v1"
    assert payload["status"] in {"not_wired", "degraded", "unknown"}
    assert payload["status"] != "wired"
    assert payload["not_wired"] is True
    assert "not wired" in payload["readiness_message"].lower()
    assert payload["current_round_id"] is None
    assert payload["current_cancellation_token"] is None
    assert payload["scheduler_state"]["state"] == "unknown"
    assert payload["interruption"]["state"] == "unknown"
    assert payload["last_interrupt"] is None
    assert payload["cancellation_chain"] == []
    assert payload["lanes"]["component_state"] == "unknown"
    assert payload["fast_think"]["state"] == "unknown"
    assert payload["slow_reasoner"]["state"] == "unknown"
    assert payload["arbiter"]["state"] == "unknown"
    assert payload["speech_action_plan"]["state"] == "not_wired"
    assert payload["proactive_activity"]["state"] == "not_wired"


def test_voice_realtime_exports_native_payload_consumed_by_eihead_monitoring() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eihead.monitoring.voice import build_voice_diagnostics_from_app

    runtime = BodyRuntimeApp()
    runtime.organs = [_HealthyEar(), _HealthyMouth()]
    runtime.update_voice_dialogue_state(
        enabled=True,
        running=True,
        phase="speaking",
        last_status="completed",
        last_transcript="hello honjia",
        last_reply="hello",
        last_stage_latency_ms={"capture": 70.0, "llm": 390.0, "tts": 250.0, "total": 710.0, "overhead": 0.0},
        last_bottleneck_stage="llm",
        last_bottleneck_ms=390.0,
        last_completed_turn={"transcript": "hello honjia", "reply": "hello"},
    )

    payload = runtime.voice_realtime()
    consumed = build_voice_diagnostics_from_app(runtime, timestamp=123.0)

    assert payload["schema"] == "eihead.monitor.voice_realtime.v1"
    assert payload["status"] == "wired"
    assert payload["ear"]["provider"] == "faster_whisper"
    assert payload["mouth"]["backend"] == "minimax"
    assert payload["latency"]["stage_latency_ms"]["llm"] == 390.0
    assert payload["latency"]["total_ms"] == 710.0
    assert payload["bottleneck"] == {"stage": "llm", "latency_ms": 390.0}
    assert payload["last_turn"] == {"transcript": "hello honjia", "reply": "hello"}
    assert consumed["source"] == "voice_realtime"
    assert consumed["status"] == "wired"
    assert consumed["latency"]["stage_latency_ms"]["llm"] == 390.0
    assert consumed["latency"]["total_ms"] == 710.0
    assert consumed["last_turn"] == {"transcript": "hello honjia", "reply": "hello"}


def test_voice_realtime_exposes_voice_chain_benchmark_in_dialogue_payload() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eihead.monitoring.voice import build_voice_diagnostics_from_app

    benchmark = {
        "turnCount": 2,
        "roundLeakCount": 0,
        "roundLeakRate": 0.0,
        "metrics": {
            "asrFinalMs": {"count": 2, "avg": 120.0, "p95": 140.0, "threshold": 800.0, "pass": True},
            "firstAudioMs": {"count": 2, "avg": 420.0, "p95": 450.0, "threshold": 2000.0, "pass": True},
            "interruptStopMs": {"count": 2, "avg": 0.0, "p95": 0.0, "threshold": 300.0, "pass": True},
        },
        "bottleneck": {"field": "firstAudioMs", "label": "first_audio", "p95": 450.0, "threshold": 2000.0, "ratio": 0.225},
    }
    runtime = BodyRuntimeApp()
    runtime.update_voice_dialogue_state(
        enabled=True,
        running=True,
        phase="idle",
        last_status="reply_ready",
        last_completed_turn={"transcript": "hello", "reply": "hello"},
        last_stage_latency_ms={"listen_asr": 100.0, "think": 200.0, "speak": 50.0, "total": 350.0},
        voice_chain_benchmark=benchmark,
    )

    payload = runtime.voice_realtime()
    consumed = build_voice_diagnostics_from_app(runtime, timestamp=126.0)

    assert payload["dialogue"]["voice_chain_benchmark"] == benchmark
    assert payload["voice_dialogue"]["voice_chain_benchmark"] == benchmark
    assert consumed["observation"]["dialogue"]["voice_chain_benchmark"] == benchmark


def test_voice_realtime_projects_round_scheduler_interruption_and_cancellation_chain() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eihead.monitoring.voice import build_voice_diagnostics_from_app

    runtime = BodyRuntimeApp()
    runtime.organs = [_HealthyEar(), _HealthyMouth()]
    runtime.update_voice_dialogue_state(
        enabled=True,
        running=True,
        phase="barge_in",
        last_status="interrupted",
        current_round_id="round-42",
        current_cancellation_token="cancel-42",
        scheduler_state={
            "state": "stale",
            "round_id": "round-42",
            "active_round_id": "round-43",
            "cancellation_token": "cancel-42",
            "stale": True,
        },
        interrupt_active=True,
        interrupted_round_count=1,
        last_interrupt={
            "round_id": "round-42",
            "reason": "user_barge_in",
            "stale": True,
        },
        realtime_session={
            "round_id": "round-42",
            "roundId": "round-42",
            "cancellation_token": "cancel-42",
            "cancellationToken": "cancel-42",
            "phase": "barge_in",
            "status": "interrupted",
            "interrupted": True,
            "complete": False,
            "latency_ms": {"first_speech": 1300.0},
            "cancellation_chain": [
                {
                    "target": "generation",
                    "event_type": "generation_cancelled",
                    "reason": "user_barge_in",
                    "round_id": "round-42",
                    "cancellation_token": "cancel-42",
                }
            ],
        },
    )

    payload = runtime.voice_realtime()
    consumed = build_voice_diagnostics_from_app(runtime, timestamp=124.0)

    assert payload["status"] == "degraded"
    assert payload["wired"] is False
    assert payload["current_round_id"] == "round-42"
    assert payload["current_cancellation_token"] == "cancel-42"
    assert payload["scheduler_state"]["state"] == "stale"
    assert payload["interruption"]["interrupted"] is True
    assert payload["interruption"]["last_interrupt"]["reason"] == "user_barge_in"
    assert payload["last_interrupt"]["round_id"] == "round-42"
    assert payload["cancellation_chain"][0]["event_type"] == "generation_cancelled"
    assert payload["realtime_session"]["round_id"] == "round-42"
    assert payload["round"]["state"] == "interrupted"
    assert payload["latency"]["first_speech_within_2s"] is True
    assert payload["latency"]["first_speech_ms"] == 1300.0
    assert consumed["round"]["lifecycle"] == "interrupted"
    assert consumed["round"]["state"] == "interrupted"
    assert consumed["scheduler"]["state"] == "stale"
    assert consumed["interruption"]["last_interrupt"]["reason"] == "user_barge_in"
    assert consumed["cancellation_chain"][0]["cancellation_token"] == "cancel-42"


def test_voice_realtime_summarizes_realtime_cognition_scheduler_snapshot() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eihead.monitoring.voice import build_voice_diagnostics_from_app

    runtime = BodyRuntimeApp()
    runtime.update_voice_dialogue_state(
        enabled=True,
        running=True,
        phase="thinking",
        last_status="active",
        current_round_id="round-rt-1",
        current_cancellation_token="cancel-rt-1",
        scheduler_state={
            "state": "active",
            "lanes": {
                "fast_think": {"state": "ready", "latency_ms": 96.0, "hypothesis_count": 2},
                "slow_reasoner": {"state": "thinking", "latency_ms": 340.0},
                "arbiter": {"state": "approved", "last_decision": "speak"},
            },
            "speech_action_plan": {
                "plan_id": "plan-rt-1",
                "stable": True,
                "speech_segments": [{"text": "I will reply softly.", "stable": True}],
                "action_segments": [{"capabilityId": "neck.pan"}],
            },
            "proactive_activity": {
                "proposal_id": "activity-rt-1",
                "channel": "visual_only",
                "reason": "memory_nudge",
                "should_emit": True,
            },
        },
    )

    payload = runtime.voice_realtime()
    consumed = build_voice_diagnostics_from_app(runtime, timestamp=125.0)

    assert payload["lanes"]["fast_think"]["state"] == "ready"
    assert payload["lanes"]["slow_reasoner"]["state"] == "thinking"
    assert payload["lanes"]["arbiter"]["state"] == "approved"
    assert payload["fast_think"]["summary"] == "ready (96.0ms)"
    assert payload["speech_action_plan"]["summary"] == "plan-rt-1: 1 speech, 1 action"
    assert payload["proactive_activity"]["summary"] == "activity-rt-1: visual_only / emit"
    assert consumed["lanes"]["fast_think"]["state"] == "ready"
    assert consumed["speech_action_plan"]["plan_id"] == "plan-rt-1"
    assert consumed["proactive_activity"]["proposal_id"] == "activity-rt-1"


def test_request_voice_interrupt_clears_busy_dispatches_stop_and_records_state() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    mouth = _InterruptibleMouth()
    runtime = BodyRuntimeApp()
    runtime.organs = [mouth]
    runtime._speech_busy_until = time.time() + 60.0
    runtime.update_voice_dialogue_state(enabled=True, running=True, phase="speaking", last_status="speaking")

    summary = runtime.request_voice_interrupt(reason="user_barge_in")

    assert summary["status"] == "ok"
    assert summary["busy_cleared"] is True
    assert summary["busy_retained"] is False
    assert runtime._speech_busy_until == 0.0
    assert mouth.actions and isinstance(mouth.actions[-1], StopSpeechAction)
    assert mouth.actions[-1].reason == "user_barge_in"
    assert runtime.voice_dialogue_state["last_status"] == "interrupted"
    assert runtime.voice_dialogue_state["interrupt"]["reason"] == "user_barge_in"
    assert runtime.voice_dialogue_state["interrupt"]["status"] == "ok"
    assert any(event["kind"] == "voice_interrupt_requested" for event in runtime.recent_events())


def test_request_voice_interrupt_without_mouth_is_truthful_degraded() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    runtime.organs = []
    runtime._speech_busy_until = time.time() + 60.0

    summary = runtime.request_voice_interrupt(reason="user_barge_in")

    assert summary["status"] == "degraded"
    assert summary["mouth_available"] is False
    assert summary["busy_cleared"] is False
    assert summary["busy_retained"] is True
    assert runtime.is_speaking() is True
    assert runtime.voice_dialogue_state["interrupt"]["status"] == "degraded"
    assert runtime.recent_events()[-1]["kind"] == "voice_interrupt_requested"
    assert runtime.recent_events()[-1]["status"] == "degraded"


def test_request_voice_interrupt_failure_keeps_busy_gate_truthful() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    runtime.organs = [_FailingInterruptMouth()]
    runtime._speech_busy_until = time.time() + 60.0

    summary = runtime.request_voice_interrupt(reason="user_barge_in")

    assert summary["status"] == "degraded"
    assert summary["busy_cleared"] is False
    assert summary["busy_retained"] is True
    assert runtime.is_speaking() is True
    assert runtime.voice_dialogue_state["last_status"] == "interrupt_degraded"


def test_probe_barge_in_detects_short_voice_window_and_records_event() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    runtime.ear_processor = _ProbeEarProcessor([_pcm_chunk(12000)])

    result = runtime.probe_barge_in(session_id="session-1", actor_id="user-1")

    assert result["detected"] is True
    assert result["status"] == "detected"
    assert result["reason"] == "voice_activity_above_threshold"
    assert result["rms_level"] > 0.015
    assert result["capture_elapsed_ms"] >= 0.0
    event = runtime.recent_events()[-1]
    assert event["kind"] == "voice_barge_in_probe"
    assert event["source"] == "body_runtime.voice_barge_in_probe"
    assert event["status"] == "detected"
    assert event["session_id"] == "session-1"
    assert event["details"]["actor_id"] == "user-1"
    assert event["details"]["detected"] is True


def test_probe_barge_in_uses_low_latency_voice_window_budget() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    runtime.ear_processor = _ProbeEarProcessor([_pcm_chunk(12000)])

    runtime.probe_barge_in(session_id="session-1", actor_id="user-1")

    capture = runtime.ear_processor.capture
    assert capture.last_max_duration_s <= 0.25


def test_probe_barge_in_ignores_low_rms_window() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    runtime.ear_processor = _ProbeEarProcessor([_pcm_chunk(1)])

    result = runtime.probe_barge_in(session_id="session-1", actor_id="user-1")

    assert result["detected"] is False
    assert result["status"] == "clear"
    assert result["reason"] == "below_threshold"
    assert result["rms_level"] < 0.015
    assert runtime.recent_events()[-1]["kind"] == "voice_barge_in_probe"
    assert runtime.recent_events()[-1]["details"]["detected"] is False


def test_probe_barge_in_returns_false_without_audio_chunks() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    runtime.ear_processor = _ProbeEarProcessor([])

    result = runtime.probe_barge_in(session_id="session-1", actor_id="user-1")

    assert result["detected"] is False
    assert result["status"] == "no_audio"
    assert result["reason"] == "no_audio_captured"
    assert result["rms_level"] == 0.0
    assert result["dbfs"] == -120.0
    assert runtime.recent_events()[-1]["status"] == "no_audio"


def test_probe_barge_in_reports_not_wired_when_ear_processor_cannot_be_built() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()

    result = runtime.probe_barge_in(session_id="session-1", actor_id="user-1")

    assert result["detected"] is False
    assert result["status"] == "not_wired"
    assert "ear organ not configured" in result["reason"]
    assert runtime.recent_events()[-1]["kind"] == "voice_barge_in_probe"
    assert runtime.recent_events()[-1]["status"] == "not_wired"


def test_voice_realtime_stopped_dialogue_with_history_is_not_fake_wired() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    runtime.organs = [_HealthyEar(), _HealthyMouth()]
    runtime.update_voice_dialogue_state(
        enabled=True,
        running=True,
        phase="speaking",
        last_status="completed",
        last_completed_turn={"transcript": "hello", "reply": "hello"},
        last_stage_latency_ms={"listen_asr": 100.0, "think": 200.0, "speak": 100.0, "total": 400.0},
    )
    runtime.update_voice_dialogue_state(running=False, phase="stopped", last_status="stopped")

    payload = runtime.voice_realtime()

    assert payload["status"] == "degraded"
    assert payload["wired"] is False
    assert payload["dialogue"]["state"] == "not_wired"
    assert payload["dialogue"]["status"] == "stopped"
    assert "historical" in payload["dialogue"]["readiness_message"]


def test_ear_processor_event_reports_processor_latency_fields() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.protocol.observations import AudioTranscriptFinal

    class _EarProcessor:
        capture = object()
        recognizer = object()
        last_capture_elapsed_ms = 12.5
        last_decode_elapsed_ms = 34.5
        last_transcribe_elapsed_ms = 56.5

        def transcribe_window(self, *, chunk_count: int, session_id: str, actor_id: str) -> AudioTranscriptFinal:
            return AudioTranscriptFinal(
                ts=1.0,
                source="ear.asr",
                text=f"heard {chunk_count}",
                session_id=session_id,
                actor_id=actor_id,
            )

    runtime = BodyRuntimeApp()
    runtime.ear_processor = _EarProcessor()

    runtime.transcribe_audio_window(chunk_count=3, session_id="session-1", actor_id="user-1")

    details = runtime.recent_events()[-1]["details"]
    assert details["capture_elapsed_ms"] == 12.5
    assert details["asr_decode_elapsed_ms"] == 34.5
    assert details["asr_elapsed_ms"] == 56.5


def _pcm_chunk(level: int, *, sample_count: int = 160) -> bytes:
    return array("h", [level] * sample_count).tobytes()


class _ProbeEarProcessor:
    def __init__(self, chunks: list[bytes]) -> None:
        self.capture = _ProbeCapture(chunks)
        self.recognizer = object()


class _ProbeCapture:
    sample_rate = 16000
    channels = 1

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.last_chunks: list[bytes] = []

    def read_voice_window(self, max_duration_s: int) -> list[bytes]:
        self.last_max_duration_s = max_duration_s
        self.last_chunks = list(self.chunks)
        return list(self.chunks)


class _HealthyEar:
    name = "ear"

    def supports_action(self, action) -> bool:
        return False

    def heartbeat(self) -> OrganHealth:
        return OrganHealth(
            organ="ear",
            health="healthy",
            subfunctions={
                "capture": SubfunctionHealth(
                    name="capture",
                    health="healthy",
                    details={"device": "default", "status": "ready"},
                ),
                "asr": SubfunctionHealth(
                    name="asr",
                    health="healthy",
                    details={"provider": "faster_whisper", "status": "ready"},
                ),
            },
        )


class _HealthyMouth:
    name = "mouth"

    def supports_action(self, action) -> bool:
        return False

    def heartbeat(self) -> OrganHealth:
        return OrganHealth(
            organ="mouth",
            health="healthy",
            subfunctions={
                "tts_playback": SubfunctionHealth(
                    name="tts_playback",
                    health="healthy",
                    details={
                        "backend": "minimax",
                        "model": "speech-2.8-hd",
                        "voice_id": "female-shaonv",
                        "status": "ready",
                    },
                )
            },
        )


class _InterruptibleMouth(_HealthyMouth):
    def __init__(self) -> None:
        self.actions: list[object] = []

    def supports_action(self, action) -> bool:
        return isinstance(action, StopSpeechAction)

    def handle_action(self, action) -> ActionExecuted:
        self.actions.append(action)
        return ActionExecuted(
            ts=action.ts,
            source="mouth.tts_playback",
            status="ok",
            session_id=action.session_id,
            actor_id=action.actor_id,
            action_kind=action.kind,
            details={"stopped": True},
        )


class _FailingInterruptMouth(_HealthyMouth):
    def supports_action(self, action) -> bool:
        return isinstance(action, StopSpeechAction)

    def handle_action(self, action) -> ActionExecuted:
        return ActionExecuted(
            ts=action.ts,
            source="mouth.tts_playback",
            status="error",
            session_id=action.session_id,
            actor_id=action.actor_id,
            action_kind=action.kind,
            details={"error": "stop failed"},
        )
