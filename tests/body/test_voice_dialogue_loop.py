from __future__ import annotations

import time

from eibrain.protocol.actions import PlaySpeechAction
from eibrain.protocol.observations import AudioTranscriptFinal


class _Body:
    def __init__(self, transcripts: list[str]) -> None:
        self.voice_dialogue_state = {"turn_count": 0}
        self.updates = []
        self.dispatched = []
        self.calls = 0
        self._transcripts = transcripts

    def update_voice_dialogue_state(self, **updates):
        self.voice_dialogue_state.update(updates)
        self.updates.append(updates)

    def is_speaking(self) -> bool:
        return False

    def transcribe_audio_window(self, *, chunk_count: int, session_id: str, actor_id: str):
        self.calls += 1
        text = self._transcripts[min(self.calls - 1, len(self._transcripts) - 1)]
        return AudioTranscriptFinal(
            ts=1.0,
            source="ear.asr",
            text=text,
            session_id=session_id,
            actor_id=actor_id,
        )

    def dispatch_actions(self, actions):
        self.dispatched.extend(actions)
        return [type("Outcome", (), {"status": "ok"})()]


class _Cognition:
    def __init__(self, reply: str = "你好，我在。") -> None:
        self.reply = reply
        self.observations = []

    def handle_observation(self, observation):
        self.observations.append(observation)
        return [
            PlaySpeechAction(
                ts=1.0,
                source="test",
                session_id=observation.session_id,
                actor_id=observation.actor_id,
                text=self.reply,
            )
        ]


def _start_loop(body: _Body, cognition: _Cognition, **kwargs):
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop

    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=cognition,
        idle_interval_s=0.01,
        empty_interval_s=0.01,
        **kwargs,
    )
    loop.start()
    return loop


def _wait_until(predicate, timeout_s: float = 1.0) -> None:
    deadline = time.time() + timeout_s
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)


def _first_update(body: _Body, *, last_status: str):
    return next(update for update in body.updates if update.get("last_status") == last_status)


def test_voice_dialogue_loop_ignores_plain_text_while_dormant() -> None:
    body = _Body(["你好"])
    cognition = _Cognition()
    loop = _start_loop(body, cognition)

    _wait_until(lambda: body.calls >= 1)
    loop.stop()

    assert cognition.observations == []
    assert body.dispatched == []
    ignored_update = _first_update(body, last_status="waiting_for_wake_word")
    assert ignored_update["conversation_active"] is False
    assert ignored_update["phase"] == "idle"
    assert ignored_update["last_transcript"] == "你好"


def test_voice_dialogue_loop_wakes_on_wake_word_only_without_llm() -> None:
    body = _Body(["鸿途"])
    cognition = _Cognition()
    loop = _start_loop(body, cognition)

    _wait_until(lambda: bool(body.dispatched))
    loop.stop()

    assert cognition.observations == []
    assert body.dispatched[0].text == "我在。"
    assert body.voice_dialogue_state["conversation_active"] is True
    assert body.voice_dialogue_state["wake_word"] == "\u9e3f\u9014"
    assert body.voice_dialogue_state["sleep_word"] == "\u7ed3\u675f\u5bf9\u8bdd"
    wake_update = _first_update(body, last_status="wake_acknowledged")
    assert wake_update["conversation_active"] is True
    assert wake_update["last_reply"] == "我在。"


def test_voice_dialogue_loop_writes_engagement_state_on_wake_and_sleep(tmp_path) -> None:
    import json

    from apps.body_runtime.engagement_state import EngagementStateWriter

    body = _Body(["鸿途", "结束对话"])
    cognition = _Cognition()
    writer = EngagementStateWriter(tmp_path / "engagement.json")
    loop = _start_loop(body, cognition, engagement_writer=writer)

    _wait_until(lambda: any(update.get("last_status") == "sleep_acknowledged" for update in body.updates), timeout_s=2.0)
    loop.stop()

    state = json.loads((tmp_path / "engagement.json").read_text(encoding="utf-8"))
    assert state["conversation_active"] is False
    assert state["phase"] == "stopped"


def test_voice_dialogue_loop_strips_wake_word_before_cognition() -> None:
    body = _Body(["鸿途，介绍下你自己"])
    cognition = _Cognition(reply="我是 eibrain。")
    loop = _start_loop(body, cognition)

    _wait_until(lambda: bool(cognition.observations))
    loop.stop()

    assert cognition.observations[0].text == "介绍下你自己"
    assert body.dispatched[0].text == "我是 eibrain。"
    assert body.voice_dialogue_state["conversation_active"] is True


def test_voice_dialogue_loop_sleeps_on_sleep_word_without_llm() -> None:
    body = _Body(["结束对话"])
    cognition = _Cognition()
    loop = _start_loop(body, cognition, initial_conversation_active=True)

    _wait_until(lambda: bool(body.dispatched))
    loop.stop()

    assert cognition.observations == []
    assert body.dispatched[0].text == "好的，先休息。"
    assert body.voice_dialogue_state["conversation_active"] is False
    sleep_update = _first_update(body, last_status="sleep_acknowledged")
    assert sleep_update["conversation_active"] is False
    assert sleep_update["last_reply"] == "好的，先休息。"


def test_voice_dialogue_loop_processes_one_turn() -> None:
    body = _Body(["你好"])
    cognition = _Cognition()
    loop = _start_loop(body, cognition, initial_conversation_active=True)

    _wait_until(lambda: bool(body.dispatched))
    loop.stop()

    assert body.dispatched[0].text == "你好，我在。"
    assert cognition.observations[0].text == "你好"
    assert body.voice_dialogue_state["conversation_active"] is True
    assert body.voice_dialogue_state["turn_count"] >= 1
    assert body.voice_dialogue_state["last_transcript"] == "你好"
    assert body.voice_dialogue_state["last_reply"] == "你好，我在。"
    assert body.voice_dialogue_state["last_latency_s"]["total"] >= 0
    assert body.voice_dialogue_state["last_latency_s"]["listen_asr"] >= 0
