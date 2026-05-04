from __future__ import annotations

import threading
import time

from eibrain.protocol.actions import PlaySpeechAction, StopSpeechAction
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


class _BlockingCognition:
    def __init__(self, *, ready: threading.Event, release: threading.Event) -> None:
        self.ready = ready
        self.release = release
        self.observations = []

    def handle_observation(self, observation):
        self.observations.append(observation)
        self.ready.set()
        assert self.release.wait(timeout=1.0)
        return [
            PlaySpeechAction(
                ts=1.0,
                source="test",
                session_id=observation.session_id,
                actor_id=observation.actor_id,
                text="旧 round 回复",
            )
        ]


class _NoStopSpeechBody(_Body):
    def __init__(self, transcripts: list[str]) -> None:
        super().__init__(transcripts)
        self.stop_attempts = 0

    def dispatch_actions(self, actions):
        if any(isinstance(action, StopSpeechAction) for action in actions):
            self.stop_attempts += 1
            raise RuntimeError("stop speech is not supported by this fake")
        return super().dispatch_actions(actions)


class _InterruptOnAckPublishBody(_Body):
    def __init__(self, transcripts: list[str]) -> None:
        super().__init__(transcripts)
        self.loop = None
        self.interrupted = False

    def update_voice_dialogue_state(self, **updates):
        super().update_voice_dialogue_state(**updates)
        if (
            self.loop is not None
            and not self.interrupted
            and updates.get("last_status") in {"wake_acknowledged", "sleep_acknowledged"}
            and updates.get("last_reply")
        ):
            self.interrupted = True
            self.loop.request_interrupt(reason="test_barge_in_before_ack_dispatch")


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
    assert wake_update["last_reply"] != wake_update["last_transcript"]
    assert wake_update["round_id"].startswith("round-")
    assert wake_update["cancellation_token"]
    assert wake_update["current_round_id"] == wake_update["round_id"]
    assert wake_update["current_cancellation_token"] == wake_update["cancellation_token"]


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
    assert sleep_update["last_reply"] != sleep_update["last_transcript"]
    assert sleep_update["round_id"].startswith("round-")
    assert sleep_update["cancellation_token"]
    assert sleep_update["current_round_id"] == sleep_update["round_id"]
    assert sleep_update["current_cancellation_token"] == sleep_update["cancellation_token"]


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
    assert body.voice_dialogue_state["last_stage_latency_ms"]["total"] >= 0
    assert body.voice_dialogue_state["last_stage_latency_ms"]["listen_asr"] >= 0
    assert body.voice_dialogue_state["last_bottleneck_stage"]
    assert body.voice_dialogue_state["last_bottleneck_ms"] >= 0
    assert body.voice_dialogue_state["last_completed_turn"]["stage_latency_ms"]["total"] >= 0


def test_voice_dialogue_loop_attaches_round_metadata_to_turn_lifecycle() -> None:
    body = _Body(["你好"])
    cognition = _Cognition()
    loop = _start_loop(body, cognition, initial_conversation_active=True)

    _wait_until(lambda: any(update.get("last_status") == "reply_ready" for update in body.updates))
    loop.stop()

    transcribed_update = _first_update(body, last_status="transcribed")
    thinking_update = _first_update(body, last_status="thinking")
    reply_update = _first_update(body, last_status="reply_ready")

    for update in (transcribed_update, thinking_update, reply_update):
        assert update["round_id"].startswith("round-")
        assert update["cancellation_token"]
        assert update["current_round_id"] == update["round_id"]
        assert update["current_cancellation_token"] == update["cancellation_token"]
        assert update["scheduler_state"]["round_id"] == update["round_id"]
        assert update["scheduler_state"]["cancellation_token"] == update["cancellation_token"]

    assert transcribed_update["scheduler_state"]["asr_final"] == "你好"
    assert transcribed_update["microfeedback"]["deadline_ms"] <= 500
    assert transcribed_update["microfeedback"]["text"]
    assert transcribed_update["last_transcript"] == "你好"
    assert reply_update["last_completed_turn"]["round_id"] == reply_update["round_id"]
    assert reply_update["last_completed_turn"]["cancellation_token"] == reply_update["cancellation_token"]


def test_voice_dialogue_loop_request_interrupt_marks_round_and_tolerates_missing_stop_support() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop
    from eibrain.cognition.realtime.turn import RealtimeTurnManager

    body = _NoStopSpeechBody([""])
    cognition = _Cognition()
    turn_manager = RealtimeTurnManager()
    old_turn = turn_manager.start_round(reason="test")
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=cognition,
        realtime_turn_manager=turn_manager,
        idle_interval_s=0.01,
        empty_interval_s=0.01,
    )

    loop.request_interrupt(reason="user_barge_in")

    interrupt_update = _first_update(body, last_status="interrupted")
    assert body.stop_attempts == 1
    assert interrupt_update["interrupted_round_count"] == 1
    assert interrupt_update["interruption"]["reason"] == "user_barge_in"
    assert interrupt_update["interruption"]["mark_interrupted"]["round_id"] == old_turn.round_id
    assert interrupt_update["current_round_id"] != old_turn.round_id
    assert interrupt_update["current_cancellation_token"] != old_turn.cancellation_token
    assert interrupt_update["stop_speech_status"] == "unsupported"


def test_voice_dialogue_loop_does_not_dispatch_actions_from_stale_round() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop

    ready = threading.Event()
    release = threading.Event()
    body = _Body(["你好", ""])
    cognition = _BlockingCognition(ready=ready, release=release)
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=cognition,
        initial_conversation_active=True,
        idle_interval_s=0.01,
        empty_interval_s=0.01,
    )
    loop.start()

    assert ready.wait(timeout=1.0)
    loop.request_interrupt(reason="user_barge_in")
    release.set()
    _wait_until(lambda: any(update.get("last_status") == "stale_round_blocked" for update in body.updates))
    loop.stop()

    assert not any(
        isinstance(action, PlaySpeechAction) and action.text == "旧 round 回复"
        for action in body.dispatched
    )
    stale_update = _first_update(body, last_status="stale_round_blocked")
    assert stale_update["stale_round"]["round_id"] != stale_update["current_round_id"]


def test_voice_dialogue_loop_blocks_stale_wake_ack_after_interrupt() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop

    body = _InterruptOnAckPublishBody(["鸿途", ""])
    cognition = _Cognition()
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=cognition,
        idle_interval_s=0.01,
        empty_interval_s=0.01,
    )
    body.loop = loop
    loop.start()

    _wait_until(lambda: any(update.get("last_status") == "stale_round_blocked" for update in body.updates))
    loop.stop()

    assert not any(
        isinstance(action, PlaySpeechAction) and action.text == "我在。"
        for action in body.dispatched
    )
    stale_update = _first_update(body, last_status="stale_round_blocked")
    assert stale_update["stale_round"]["reason"] == "wake_ack_round_not_current"
    assert stale_update["last_reply"] == ""


def test_voice_dialogue_loop_blocks_stale_sleep_ack_after_interrupt() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop

    body = _InterruptOnAckPublishBody(["结束对话", ""])
    cognition = _Cognition()
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=cognition,
        initial_conversation_active=True,
        idle_interval_s=0.01,
        empty_interval_s=0.01,
    )
    body.loop = loop
    loop.start()

    _wait_until(lambda: any(update.get("last_status") == "stale_round_blocked" for update in body.updates))
    loop.stop()

    assert not any(
        isinstance(action, PlaySpeechAction) and action.text == "好的，先休息。"
        for action in body.dispatched
    )
    stale_update = _first_update(body, last_status="stale_round_blocked")
    assert stale_update["stale_round"]["reason"] == "sleep_ack_round_not_current"
    assert stale_update["last_reply"] == ""


def test_voice_dialogue_loop_clears_microfeedback_on_next_round() -> None:
    body = _Body(["你好", ""])
    cognition = _Cognition()
    loop = _start_loop(body, cognition, initial_conversation_active=True)

    _wait_until(lambda: any(update.get("last_status") == "reply_ready" for update in body.updates))
    _wait_until(lambda: any(update.get("last_status") == "no_transcript" for update in body.updates))
    loop.stop()

    reply_update = _first_update(body, last_status="reply_ready")
    no_transcript_update = _first_update(body, last_status="no_transcript")
    assert reply_update["microfeedback"]["text"]
    assert no_transcript_update["microfeedback"] == {}


def test_voice_dialogue_loop_records_latency_for_empty_transcript() -> None:
    body = _Body([""])
    cognition = _Cognition()
    loop = _start_loop(body, cognition)

    _wait_until(lambda: body.calls >= 1)
    loop.stop()

    update = _first_update(body, last_status="no_transcript")
    assert update["last_stage_latency_ms"]["listen_asr"] >= 0
    assert update["last_bottleneck_stage"] == "listen_asr"
