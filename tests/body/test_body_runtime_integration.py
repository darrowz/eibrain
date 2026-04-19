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
