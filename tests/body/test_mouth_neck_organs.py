from __future__ import annotations

from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig
from eibrain.protocol.actions import MoveHeadAction, PlaySpeechAction


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

            return DriverResult(status="ok", details={"driver": "command", "operation": operation})

    mouth.drivers["tts_plan"] = _Driver()
    mouth.drivers["tts_playback"] = _Driver()
    mouth.handle_action(
        PlaySpeechAction(ts=12.0, source="test", text="你好 honjia", session_id="s1", actor_id="user-1")
    )

    heartbeat = mouth.heartbeat()

    assert heartbeat.subfunctions["tts_plan"].health == "healthy"
    assert heartbeat.subfunctions["tts_plan"].details["status"] == "planned"
    assert heartbeat.subfunctions["tts_playback"].details["backend"] == "minimax"
    assert heartbeat.subfunctions["tts_playback"].details["model"] == "speech-2.8-hd"
    assert heartbeat.subfunctions["tts_playback"].details["text_preview"] == "你好 honjia"


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
