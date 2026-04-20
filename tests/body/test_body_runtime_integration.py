from __future__ import annotations


def test_body_runtime_reports_capabilities_from_registered_organs() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    snapshot = runtime.snapshot()

    assert snapshot["organ_count"] == 4
    assert snapshot["degradation_mode"] == "mute_companion"
    assert snapshot["capabilities"]["can_hear_voice"] is False
    assert snapshot["capabilities"]["can_speak"] is False


def test_body_runtime_can_transcribe_audio_window() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    class _Capture:
        sample_rate = 16000
        channels = 1

        def read_chunks(self, chunk_count: int):
            return [b"a"] * chunk_count

    class _Recognizer:
        def transcribe(self, pcm_chunks, *, sample_rate: int, channels: int) -> str:
            return "streamed text"

    runtime = BodyRuntimeApp()
    runtime.ear_processor = runtime._build_ear_processor(capture=_Capture(), recognizer=_Recognizer())

    observation = runtime.transcribe_audio_window(
        chunk_count=2,
        session_id="session-1",
        actor_id="user-1",
    )

    assert observation.text == "streamed text"


def test_body_runtime_transcribes_from_ear_organ_heartbeat() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    runtime = BodyRuntimeApp()

    class _Ear:
        name = "ear"
        _chunk_count = 1
        _cached_heartbeat = None

        def heartbeat(self):
            return OrganHealth(
                organ="ear",
                health="healthy",
                subfunctions={
                    "asr": SubfunctionHealth(
                        name="asr",
                        health="healthy",
                        details={"transcript": "你好 honjia", "speech_window_summary": "heard speech"},
                    )
                },
            )

    runtime.organs = [_Ear()]

    observation = runtime.transcribe_audio_window(
        chunk_count=3,
        session_id="session-2",
        actor_id="user-2",
    )

    assert observation.text == "你好 honjia"
    assert runtime.recent_events()[-1]["status"] == "ok"


def test_body_runtime_maps_visual_target_to_move_head_action() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    action = runtime.plan_visual_tracking_action(
        target_name="person",
        target_x=0.75,
        session_id="session-1",
        actor_id="user-1",
    )

    assert action.target_x == 0.75


def test_body_runtime_can_dispatch_visual_tracking_to_neck() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    runtime = BodyRuntimeApp()

    class _Eye:
        name = "eye"

        def supports_action(self, action) -> bool:
            return False

        def heartbeat(self):
            return OrganHealth(
                organ="eye",
                health="healthy",
                subfunctions={
                    "detection": SubfunctionHealth(
                        name="detection",
                        health="healthy",
                        details={
                            "detections": [
                                {
                                    "label": "person",
                                    "score": 0.6,
                                    "bbox": {"x_min": 0.0, "x_max": 0.4},
                                },
                                {
                                    "label": "face",
                                    "score": 0.8,
                                    "bbox": {"x_min": 0.6, "x_max": 0.8},
                                },
                            ]
                        },
                    )
                },
            )

    class _Neck:
        name = "neck"

        def supports_action(self, action) -> bool:
            return True

        def handle_action(self, action):
            from eibrain.protocol.outcomes import ActionExecuted

            return ActionExecuted(
                ts=action.ts,
                source="neck.motor",
                status="ok",
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                action_kind=action.kind,
                details={"target_x": action.target_x, "target_name": action.target_name},
            )

    runtime.organs = [_Eye(), _Neck()]

    outcome = runtime.track_visual_target_once(session_id="track-1", actor_id="vision-1")

    assert outcome is not None
    assert outcome.details["target_name"] == "face"
    assert outcome.details["target_x"] == 0.7
