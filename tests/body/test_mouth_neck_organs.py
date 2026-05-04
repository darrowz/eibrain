from __future__ import annotations

from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig
from eibrain.protocol.actions import MoveHeadAction, PlaySpeechAction, StopSpeechAction


def test_mouth_organ_exposes_tts_plan_and_playback_runtime_state() -> None:
    from eibrain.body.organs.mouth.organ import MouthOrgan

    mouth = MouthOrgan(
        config=OrganConfig(
            enabled=True,
            subfunctions={
                "tts_plan": SubfunctionConfig(
                    driver=DriverConfig(
                        kind="command",
                        extra={"backend": "minimax", "model": "speech-2.8-hd", "voice_id": "female-shaonv"},
                    )
                ),
                "tts_playback": SubfunctionConfig(
                    driver=DriverConfig(
                        kind="command",
                        extra={
                            "backend": "minimax",
                            "model": "speech-2.8-hd",
                            "voice_id": "female-shaonv",
                            "output_device": "plughw:2,0",
                        },
                    )
                ),
            },
        )
    )

    class _Driver:
        def __init__(self, status: str = "healthy") -> None:
            self.status = status

        def heartbeat(self):
            from eibrain.body.drivers.base import DriverResult

            return DriverResult(status=self.status, details={"driver": "command", "status": self.status})

        def invoke(self, operation: str, payload: dict[str, object]):
            from eibrain.body.drivers.base import DriverResult

            return DriverResult(
                status="playing",
                details={
                    "driver": "command",
                    "operation": operation,
                    "busy": True,
                    "provider": "minimax",
                    "model": "speech-2.8-hd",
                    "voice": "female-shaonv",
                    "reason": "stream_started",
                },
            )

    mouth.drivers["tts_plan"] = _Driver()
    mouth.drivers["tts_playback"] = _Driver()
    mouth.handle_action(
        PlaySpeechAction(ts=12.0, source="test", text="你好 honjia", session_id="s1", actor_id="user-1")
    )

    heartbeat = mouth.heartbeat()

    assert heartbeat.subfunctions["tts_plan"].health == "healthy"
    assert heartbeat.subfunctions["tts_plan"].details["status"] == "planned"
    assert heartbeat.subfunctions["tts_plan"].details["session_id"] == "s1"
    assert heartbeat.subfunctions["tts_plan"].details["actor_id"] == "user-1"
    assert heartbeat.subfunctions["tts_plan"].details["busy"] is False
    assert heartbeat.subfunctions["tts_playback"].details["backend"] == "minimax"
    assert heartbeat.subfunctions["tts_playback"].details["provider"] == "minimax"
    assert heartbeat.subfunctions["tts_playback"].details["model"] == "speech-2.8-hd"
    assert heartbeat.subfunctions["tts_playback"].details["voice"] == "female-shaonv"
    assert heartbeat.subfunctions["tts_playback"].details["session_id"] == "s1"
    assert heartbeat.subfunctions["tts_playback"].details["actor_id"] == "user-1"
    assert heartbeat.subfunctions["tts_playback"].details["status"] == "playing"
    assert heartbeat.subfunctions["tts_playback"].details["busy"] is True
    assert heartbeat.subfunctions["tts_playback"].details["text_preview"] == "你好 honjia"
    assert heartbeat.subfunctions["tts_playback"].details["text_char_count"] == len("你好 honjia")
    assert "started_at" in heartbeat.subfunctions["tts_playback"].details
    assert "finished_at" in heartbeat.subfunctions["tts_playback"].details
    assert "elapsed_ms" in heartbeat.subfunctions["tts_playback"].details


def test_mouth_stop_speech_marks_playback_stopped_with_reason() -> None:
    from eibrain.body.organs.mouth.organ import MouthOrgan

    mouth = MouthOrgan(
        config=OrganConfig(
            enabled=True,
            subfunctions={
                "tts_plan": SubfunctionConfig(driver=DriverConfig(kind="command")),
                "tts_playback": SubfunctionConfig(
                    driver=DriverConfig(
                        kind="command",
                        extra={"backend": "minimax", "model": "speech-2.8-hd", "voice_id": "female-shaonv"},
                    )
                ),
            },
        )
    )

    class _Driver:
        def heartbeat(self):
            from eibrain.body.drivers.base import DriverResult

            return DriverResult(status="healthy", details={"driver": "command", "status": "healthy"})

        def invoke(self, operation: str, payload: dict[str, object]):
            from eibrain.body.drivers.base import DriverResult

            if operation == "stop_speech":
                return DriverResult(
                    status="ok",
                    details={"driver": "command", "status": "stopped", "busy": False, "reason": "barge_in"},
                )
            return DriverResult(status="playing", details={"driver": "command", "status": "playing", "busy": True})

    mouth.drivers["tts_plan"] = _Driver()
    mouth.drivers["tts_playback"] = _Driver()

    mouth.handle_action(PlaySpeechAction(ts=12.0, source="test", text="hello", session_id="s1", actor_id="user-1"))
    outcome = mouth.handle_action(
        StopSpeechAction(ts=13.0, source="test", session_id="s1", actor_id="user-1", target_id="mouth")
    )
    heartbeat = mouth.heartbeat()

    assert outcome.details["status"] == "stopped"
    assert outcome.details["busy"] is False
    assert outcome.details["reason"] == "barge_in"
    assert heartbeat.subfunctions["tts_playback"].details["status"] == "stopped"
    assert heartbeat.subfunctions["tts_playback"].details["busy"] is False
    assert heartbeat.subfunctions["tts_playback"].details["reason"] == "barge_in"


def test_mouth_stop_speech_uses_action_reason_when_driver_does_not_echo_it() -> None:
    from eibrain.body.organs.mouth.organ import MouthOrgan

    mouth = MouthOrgan(
        config=OrganConfig(
            enabled=True,
            subfunctions={
                "tts_plan": SubfunctionConfig(driver=DriverConfig(kind="command")),
                "tts_playback": SubfunctionConfig(driver=DriverConfig(kind="command", extra={"backend": "minimax"})),
            },
        )
    )

    class _Driver:
        def heartbeat(self):
            from eibrain.body.drivers.base import DriverResult

            return DriverResult(status="healthy", details={"driver": "command", "status": "healthy"})

        def invoke(self, operation: str, payload: dict[str, object]):
            from eibrain.body.drivers.base import DriverResult

            if operation == "stop_speech":
                return DriverResult(status="ok", details={"driver": "command", "status": "stopped", "busy": False})
            return DriverResult(status="playing", details={"driver": "command", "status": "playing", "busy": True})

    mouth.drivers["tts_plan"] = _Driver()
    mouth.drivers["tts_playback"] = _Driver()

    mouth.handle_action(PlaySpeechAction(ts=12.0, source="test", text="hello", session_id="s1", actor_id="user-1"))
    outcome = mouth.handle_action(
        StopSpeechAction(
            ts=13.0,
            source="test",
            session_id="s1",
            actor_id="user-1",
            reason="user_barge_in",
            details={"reason": "user_barge_in"},
        )
    )

    assert outcome.details["status"] == "stopped"
    assert outcome.details["reason"] == "user_barge_in"


def test_mouth_noop_heartbeat_remains_truthful_not_wired_for_monitoring() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.organs.mouth.organ import MouthOrgan
    from eihead.monitoring.voice import build_voice_diagnostics_from_app

    mouth = MouthOrgan()
    heartbeat = mouth.heartbeat()

    assert heartbeat.health == "unavailable"
    assert heartbeat.subfunctions["tts_playback"].details["driver"] == "noop"
    assert heartbeat.subfunctions["tts_playback"].details["not_wired"] is True
    assert "backend" not in heartbeat.subfunctions["tts_playback"].details

    runtime = BodyRuntimeApp()
    runtime.organs = [mouth]
    payload = build_voice_diagnostics_from_app(runtime, timestamp=1.0)

    assert payload["status"] == "not_wired"
    assert payload["mouth"]["state"] == "not_wired"
    assert payload["mouth"]["backend"] == ""


def test_mouth_stop_speech_failure_reports_truthful_busy_and_error() -> None:
    from eibrain.body.organs.mouth.organ import MouthOrgan

    mouth = MouthOrgan(
        config=OrganConfig(
            enabled=True,
            subfunctions={
                "tts_plan": SubfunctionConfig(driver=DriverConfig(kind="command")),
                "tts_playback": SubfunctionConfig(driver=DriverConfig(kind="command")),
            },
        )
    )

    class _Driver:
        def heartbeat(self):
            from eibrain.body.drivers.base import DriverResult

            return DriverResult(status="healthy", details={"driver": "command", "status": "healthy"})

        def invoke(self, operation: str, payload: dict[str, object]):
            from eibrain.body.drivers.base import DriverResult

            if operation == "stop_speech":
                return DriverResult(
                    status="error",
                    details={
                        "driver": "command",
                        "status": "error",
                        "busy": True,
                        "reason": "driver_timeout",
                        "last_error": "timeout waiting for tts stop",
                    },
                )
            return DriverResult(status="playing", details={"driver": "command", "status": "playing", "busy": True})

    mouth.drivers["tts_plan"] = _Driver()
    mouth.drivers["tts_playback"] = _Driver()

    mouth.handle_action(PlaySpeechAction(ts=12.0, source="test", text="hello", session_id="s1", actor_id="user-1"))
    outcome = mouth.handle_action(StopSpeechAction(ts=13.0, source="test", session_id="s1", actor_id="user-1"))
    heartbeat = mouth.heartbeat()

    assert outcome.details["status"] == "stop_failed"
    assert outcome.details["busy"] is True
    assert outcome.details["reason"] == "driver_timeout"
    assert outcome.details["last_error"] == "timeout waiting for tts stop"
    assert heartbeat.subfunctions["tts_playback"].details["status"] == "stop_failed"
    assert heartbeat.subfunctions["tts_playback"].details["busy"] is True
    assert heartbeat.subfunctions["tts_playback"].details["reason"] == "driver_timeout"
    assert heartbeat.subfunctions["tts_playback"].details["last_error"] == "timeout waiting for tts stop"


def test_neck_organ_exposes_tracking_runtime_state() -> None:
    from eibrain.body.organs.neck.organ import NeckOrgan

    neck = NeckOrgan(
        config=OrganConfig(
            enabled=True,
            subfunctions={
                "motor": SubfunctionConfig(driver=DriverConfig(kind="command", extra={"pan_min": 40, "pan_max": 140})),
                "tracking": SubfunctionConfig(driver=DriverConfig(kind="command")),
            },
        )
    )

    class _Driver:
        def heartbeat(self):
            from eibrain.body.drivers.base import DriverResult

            return DriverResult(status="healthy", details={"driver": "command", "status": "healthy"})

        def invoke(self, operation: str, payload: dict[str, object]):
            from eibrain.body.drivers.base import DriverResult

            return DriverResult(status="ok", details={"driver": "command", **payload, "angle": payload.get("target_angle", 90)})

    neck.drivers["motor"] = _Driver()
    neck.drivers["tracking"] = _Driver()
    neck.handle_action(
        MoveHeadAction(
            ts=21.0,
            source="eye.tracking",
            session_id="s2",
            actor_id="user-1",
            target_name="speaker",
            target_x=0.6,
        )
    )

    heartbeat = neck.heartbeat()

    assert heartbeat.subfunctions["tracking"].health == "healthy"
    assert heartbeat.subfunctions["tracking"].details["status"] == "tracking_target"
    assert heartbeat.subfunctions["tracking"].details["target_name"] == "speaker"
