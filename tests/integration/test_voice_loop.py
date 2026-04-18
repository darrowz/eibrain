from __future__ import annotations


def test_minimal_voice_loop_returns_play_speech_action() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    body = BodyRuntimeApp()
    cognitive = CognitiveRuntimeApp()

    observation = body.simulate_transcript(
        text="你好，eibrain",
        session_id="session-1",
        actor_id="user-1",
    )
    actions = cognitive.handle_observation(observation)

    assert len(actions) == 1
    assert actions[0].kind == "play_speech_action"
    assert "你好" in actions[0].text
