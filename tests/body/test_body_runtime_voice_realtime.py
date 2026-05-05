from __future__ import annotations

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

    assert payload["current_round_id"] == "round-42"
    assert payload["current_cancellation_token"] == "cancel-42"
    assert payload["scheduler_state"]["state"] == "stale"
    assert payload["interruption"]["interrupted"] is True
    assert payload["interruption"]["last_interrupt"]["reason"] == "user_barge_in"
    assert payload["last_interrupt"]["round_id"] == "round-42"
    assert payload["cancellation_chain"][0]["event_type"] == "generation_cancelled"
    assert payload["realtime_session"]["round_id"] == "round-42"
    assert consumed["round"]["lifecycle"] == "interrupted"
    assert consumed["scheduler"]["state"] == "stale"
    assert consumed["interruption"]["last_interrupt"]["reason"] == "user_barge_in"
    assert consumed["cancellation_chain"][0]["cancellation_token"] == "cancel-42"


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
