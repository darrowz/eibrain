from __future__ import annotations


def test_voice_dialogue_loop_processes_one_turn() -> None:
    import time

    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop
    from eibrain.protocol.actions import PlaySpeechAction
    from eibrain.protocol.observations import AudioTranscriptFinal

    class _Body:
        def __init__(self) -> None:
            self.voice_dialogue_state = {"turn_count": 0}
            self.updates = []
            self.dispatched = []
            self.calls = 0

        def update_voice_dialogue_state(self, **updates):
            self.voice_dialogue_state.update(updates)
            self.updates.append(updates)

        def is_speaking(self) -> bool:
            return False

        def transcribe_audio_window(self, *, chunk_count: int, session_id: str, actor_id: str):
            self.calls += 1
            return AudioTranscriptFinal(
                ts=1.0,
                source="ear.asr",
                text="你好",
                session_id=session_id,
                actor_id=actor_id,
            )

        def dispatch_actions(self, actions):
            self.dispatched.extend(actions)
            return [type("Outcome", (), {"status": "ok"})()]

    class _Cognition:
        def handle_observation(self, observation):
            return [
                PlaySpeechAction(
                    ts=1.0,
                    source="test",
                    session_id=observation.session_id,
                    actor_id=observation.actor_id,
                    text="你好，我在。",
                )
            ]

    body = _Body()
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=_Cognition(),
        idle_interval_s=0.01,
        empty_interval_s=0.01,
    )

    loop.start()
    deadline = time.time() + 1
    while not body.dispatched and time.time() < deadline:
        time.sleep(0.01)
    loop.stop()

    assert body.dispatched[0].text == "你好，我在。"
    assert body.voice_dialogue_state["turn_count"] >= 1
    assert body.voice_dialogue_state["last_transcript"] == "你好"
    assert body.voice_dialogue_state["last_reply"] == "你好，我在。"
