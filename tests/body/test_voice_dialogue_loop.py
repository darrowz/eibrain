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


class _StreamingCognition:
    def __init__(self, deltas: list[str] | None = None, reply: str = "你好，我在。") -> None:
        self.deltas = deltas or ["你好", "，我在。"]
        self.reply = reply
        self.observations = []
        self.stream_kwargs = []
        self.batch_called = False

    def stream_observation(self, observation, **kwargs):
        self.observations.append(observation)
        self.stream_kwargs.append(kwargs)
        for delta in self.deltas:
            yield {"type": "reply_delta", "text": delta}
        yield {
            "type": "actions",
            "actions": [
                PlaySpeechAction(
                    ts=1.0,
                    source="test",
                    session_id=observation.session_id,
                    actor_id=observation.actor_id,
                    text=self.reply,
                )
            ],
        }

    def handle_observation(self, observation):
        self.batch_called = True
        raise AssertionError("streaming cognition should not use batch fallback")


class _JsonStreamingCognition:
    def __init__(self) -> None:
        self.observations = []
        self.stream_kwargs = []
        self.batch_called = False

    def handle_observation_stream(self, observation, *, round_id: str = "", cancellation_token: str = ""):
        self.observations.append(observation)
        self.stream_kwargs.append({"round_id": round_id, "cancellation_token": cancellation_token})
        yield {"type": "status", "status": "started", "round_id": round_id, "cancellation_token": cancellation_token}
        yield {"type": "reply_delta", "delta": "JSON 回复", "round_id": round_id, "cancellation_token": cancellation_token}
        yield {
            "type": "actions_final",
            "status": "planned",
            "reply_text": "JSON 回复",
            "round_id": round_id,
            "cancellation_token": cancellation_token,
            "actions": [
                {
                    "kind": "play_speech_action",
                    "ts": 1.0,
                    "source": "json-test",
                    "session_id": observation.session_id,
                    "actor_id": observation.actor_id,
                    "text": "JSON 回复",
                }
            ],
        }

    def handle_observation(self, observation):
        self.batch_called = True
        raise AssertionError("JSON streaming cognition should not use batch fallback")


class _BlockingStreamingCognition:
    def __init__(self, *, ready: threading.Event, release: threading.Event) -> None:
        self.ready = ready
        self.release = release
        self.observations = []

    def stream_observation(self, observation, **kwargs):
        self.observations.append(observation)
        yield {"type": "reply_delta", "text": "先想一下"}
        self.ready.set()
        assert self.release.wait(timeout=1.0)
        yield {"type": "reply_delta", "text": "旧 round delta"}
        yield {
            "type": "actions",
            "actions": [
                PlaySpeechAction(
                    ts=1.0,
                    source="test",
                    session_id=observation.session_id,
                    actor_id=observation.actor_id,
                    text="旧 round 回复",
                )
            ],
        }


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


class _PlaybackBargeInBody(_Body):
    def __init__(self, transcripts: list[str]) -> None:
        super().__init__(transcripts)
        self.speaking = True
        self.probe_calls = []

    def is_speaking(self) -> bool:
        return self.speaking

    def probe_barge_in(self, *, session_id: str, actor_id: str):
        self.probe_calls.append({"session_id": session_id, "actor_id": actor_id})
        return {"detected": True, "reason": "playback_vad"}

    def dispatch_actions(self, actions):
        self.dispatched.extend(actions)
        if any(isinstance(action, StopSpeechAction) for action in actions):
            self.speaking = False
        return [type("Outcome", (), {"status": "ok"})()]


class _StopAckButStillSpeakingBody(_Body):
    def __init__(self, transcripts: list[str]) -> None:
        super().__init__(transcripts)
        self.speaking = True

    def is_speaking(self) -> bool:
        return self.speaking

    def dispatch_actions(self, actions):
        self.dispatched.extend(actions)
        return [type("Outcome", (), {"status": "ok"})()]


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


def test_voice_dialogue_loop_updates_voice_chain_benchmark_after_completed_turns() -> None:
    body = _Body(["你好", "继续说"])
    cognition = _Cognition(reply="你好，我在。")
    loop = _start_loop(body, cognition, initial_conversation_active=True)

    _wait_until(lambda: int(body.voice_dialogue_state.get("turn_count", 0) or 0) >= 2, timeout_s=2.0)
    loop.stop()

    benchmark = body.voice_dialogue_state["voice_chain_benchmark"]
    assert benchmark["turnCount"] == body.voice_dialogue_state["turn_count"]
    assert benchmark["roundLeakCount"] == 0
    assert "asrFinalMs" in benchmark["metrics"]
    assert "firstAudioMs" in benchmark["metrics"]
    assert "interruptStopMs" not in benchmark["metrics"]
    assert benchmark["bottleneck"]["field"] is not None
    for trace in benchmark["recentTraces"]:
        assert {"asrFinalMs", "firstAudioMs", "roundLeak"} <= set(trace)
        assert "interruptStopMs" not in trace


def test_voice_dialogue_loop_replaces_duplicate_voice_chain_trace_for_same_round_status() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop
    from eibrain.cognition.realtime.turn import RealtimeTurnManager

    body = _Body([])
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=_Cognition(),
        realtime_turn_manager=RealtimeTurnManager(),
        session_id="session-123",
        actor_id="actor-456",
        idle_interval_s=0.01,
        empty_interval_s=0.01,
    )
    turn = loop._start_round(reason="wake_ack")

    loop._publish_state(
        phase="idle",
        last_status="wake_acknowledged",
        turn=turn,
        last_stage_latency_ms={"listen_asr": 10.0, "think": 20.0, "total": 30.0},
    )
    loop._publish_state(
        phase="idle",
        last_status="wake_acknowledged",
        turn=turn,
        last_stage_latency_ms={"listen_asr": 10.0, "think": 25.0, "total": 35.0},
    )

    benchmark = body.voice_dialogue_state["voice_chain_benchmark"]
    assert benchmark["turnCount"] == 1
    assert len(benchmark["recentTraces"]) == 1
    assert benchmark["recentTraces"][0]["firstAudioMs"] == 35.0


def test_voice_dialogue_loop_uses_streaming_facade_for_closed_loop_diagnostics() -> None:
    body = _Body(["你好"])
    cognition = _StreamingCognition(deltas=["你", "好"], reply="你好")
    loop = _start_loop(body, cognition, initial_conversation_active=True)

    _wait_until(lambda: any(update.get("last_status") == "reply_ready" for update in body.updates))
    loop.stop()

    assert cognition.observations[0].text == "你好"
    assert cognition.stream_kwargs[0]["round_id"].startswith("round-")
    assert cognition.stream_kwargs[0]["cancellation_token"]
    assert cognition.batch_called is False
    assert body.dispatched[0].text == "你好"

    delta_updates = [update for update in body.updates if update.get("last_status") == "reply_delta"]
    assert [update["last_reply_delta"] for update in delta_updates[:2]] == ["你", "好"]
    final_state = body.voice_dialogue_state
    assert final_state["last_reply_delta"] == "好"
    assert final_state["closed_loop_state"]["listening"] is True
    assert final_state["closed_loop_state"]["final_asr"] is True
    assert final_state["closed_loop_state"]["reply_delta"] is True
    assert final_state["closed_loop_state"]["speaking"] is True
    assert final_state["closed_loop_state"]["complete"] is True
    assert final_state["realtime_session"]["transcript_final"] == "你好"
    assert final_state["realtime_session"]["reply_text"] == "你好"
    event_types = [event["event_type"] for event in final_state["realtime_events"]]
    assert "listening_started" in event_types
    assert "asr_final" in event_types
    assert "agent_think" in event_types
    assert "tts_started" in event_types
    assert "complete" in event_types


def test_voice_dialogue_loop_voice_chain_benchmark_exposes_live_stage_and_streaming_signals() -> None:
    body = _Body(["你好"])
    cognition = _StreamingCognition(deltas=["你", "好"], reply="你好")
    loop = _start_loop(body, cognition, initial_conversation_active=True)

    _wait_until(lambda: any(update.get("last_status") == "reply_ready" for update in body.updates))
    loop.stop()

    benchmark = body.voice_dialogue_state["voice_chain_benchmark"]
    trace = benchmark["recentTraces"][-1]
    round_payload = benchmark["rounds"][-1]
    assert trace["stageLatencyMs"]["listen_asr"] >= 0
    assert trace["stageLatencyMs"]["think"] >= 0
    assert trace["stageLatencyMs"]["speak"] >= 0
    assert trace["streaming"] == {"asrPartial": True, "llmDelta": True, "ttsChunk": True}
    assert round_payload["stageLatencyMs"]["think"] == trace["stageLatencyMs"]["think"]
    assert benchmark["stageLatencyMetrics"]["think"]["count"] >= 1
    assert benchmark["streaming"]["ready"] is True
    assert benchmark["roundLeak"]["free"] is True
    assert benchmark["interruptStop"]["requiredCount"] == 0


def test_voice_dialogue_loop_preserves_round_token_for_json_stream_facade() -> None:
    body = _Body(["你好"])
    cognition = _JsonStreamingCognition()
    loop = _start_loop(body, cognition, initial_conversation_active=True)

    _wait_until(lambda: any(update.get("last_status") == "reply_ready" for update in body.updates))
    loop.stop()

    assert cognition.batch_called is False
    assert cognition.stream_kwargs[0]["round_id"].startswith("round-")
    assert cognition.stream_kwargs[0]["cancellation_token"]
    assert body.dispatched
    assert isinstance(body.dispatched[0], PlaySpeechAction)
    assert body.dispatched[0].text == "JSON 回复"
    assert body.voice_dialogue_state["last_reply_delta"] == "JSON 回复"


def test_voice_dialogue_loop_publishes_closed_loop_snapshot_for_batch_fallback() -> None:
    body = _Body(["你好"])
    cognition = _Cognition(reply="批量回复")
    loop = _start_loop(body, cognition, initial_conversation_active=True)

    _wait_until(lambda: any(update.get("last_status") == "reply_ready" for update in body.updates))
    loop.stop()

    final_state = body.voice_dialogue_state
    assert final_state["last_reply"] == "批量回复"
    assert final_state["last_reply_delta"] == "批量回复"
    assert final_state["closed_loop_state"]["final_asr"] is True
    assert final_state["closed_loop_state"]["reply_delta"] is True
    assert final_state["closed_loop_state"]["speaking"] is True
    assert final_state["closed_loop_state"]["complete"] is True
    assert final_state["realtime_session"]["transcript_final"] == "你好"
    assert final_state["realtime_events"]
    assert final_state["realtime_latency_ms"]["total"] >= 0


def test_voice_dialogue_loop_blocks_stale_streaming_deltas_and_actions_after_interrupt() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop

    ready = threading.Event()
    release = threading.Event()
    body = _Body(["你好", ""])
    cognition = _BlockingStreamingCognition(ready=ready, release=release)
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

    assert any(update.get("last_reply_delta") == "先想一下" for update in body.updates)
    assert not any(update.get("last_reply_delta") == "旧 round delta" for update in body.updates)
    assert not any(
        isinstance(action, PlaySpeechAction) and action.text == "旧 round 回复"
        for action in body.dispatched
    )
    interrupted_updates = [update for update in body.updates if update.get("last_status") == "interrupted"]
    assert interrupted_updates
    assert interrupted_updates[-1]["closed_loop_state"]["interrupted"] is True
    stale_update = _first_update(body, last_status="stale_round_blocked")
    assert stale_update["stale_round"]["reason"] == "streaming_round_not_current"


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
    assert 0 <= transcribed_update["microfeedback"]["elapsed_ms"] <= transcribed_update["microfeedback"]["deadline_ms"]
    assert transcribed_update["microfeedback"]["within_deadline"] is True
    assert transcribed_update["microfeedback"]["text"]
    assert transcribed_update["last_transcript"] == "你好"
    assert reply_update["last_completed_turn"]["round_id"] == reply_update["round_id"]
    assert reply_update["last_completed_turn"]["cancellation_token"] == reply_update["cancellation_token"]


def test_voice_dialogue_loop_detects_barge_in_while_speaking() -> None:
    body = _PlaybackBargeInBody([""])
    cognition = _Cognition()
    loop = _start_loop(
        body,
        cognition,
        initial_conversation_active=True,
        session_id="session-123",
        actor_id="actor-456",
    )

    _wait_until(lambda: any(isinstance(action, StopSpeechAction) for action in body.dispatched))
    loop.stop()

    assert body.probe_calls == [{"session_id": "session-123", "actor_id": "actor-456"}]
    assert any(isinstance(action, StopSpeechAction) for action in body.dispatched)
    interrupt_update = _first_update(body, last_status="interrupted")
    assert interrupt_update["interruption"]["reason"] == "playback_vad"
    assert not any(update.get("last_status") == "playback_active" for update in body.updates)


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
    assert 0 <= interrupt_update["stop_dispatch_elapsed_ms"] <= 300
    assert interrupt_update["interrupt_to_tts_stop_ms"] is None
    assert interrupt_update["tts_stop_confirmed"] is False
    assert interrupt_update["interruption"]["tts_stop_within_300ms"] is False
    benchmark = body.voice_dialogue_state["voice_chain_benchmark"]
    assert benchmark["turnCount"] == 1
    assert "interruptStopMs" not in benchmark["metrics"]
    assert "interruptStopMs" not in benchmark["recentTraces"][-1]
    assert benchmark["recentTraces"][-1]["roundLeak"] is False


def test_voice_dialogue_loop_request_interrupt_counts_confirmed_tts_stop_latency() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop
    from eibrain.cognition.realtime.turn import RealtimeTurnManager

    body = _PlaybackBargeInBody([""])
    cognition = _Cognition()
    turn_manager = RealtimeTurnManager()
    turn_manager.start_round(reason="test")
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=cognition,
        realtime_turn_manager=turn_manager,
        idle_interval_s=0.01,
        empty_interval_s=0.01,
    )

    loop.request_interrupt(reason="user_barge_in")

    interrupt_update = _first_update(body, last_status="interrupted")
    benchmark = body.voice_dialogue_state["voice_chain_benchmark"]
    assert interrupt_update["tts_stop_confirmed"] is True
    assert benchmark["metrics"]["interruptStopMs"]["count"] == 1
    assert benchmark["recentTraces"][-1]["interruptStopMs"] == interrupt_update["interrupt_to_tts_stop_ms"]


def test_voice_dialogue_loop_request_interrupt_does_not_count_stop_dispatch_without_playback_confirmation() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop
    from eibrain.cognition.realtime.turn import RealtimeTurnManager

    body = _StopAckButStillSpeakingBody([""])
    cognition = _Cognition()
    turn_manager = RealtimeTurnManager()
    turn_manager.start_round(reason="test")
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=cognition,
        realtime_turn_manager=turn_manager,
        idle_interval_s=0.01,
        empty_interval_s=0.01,
    )

    loop.request_interrupt(reason="user_barge_in")

    interrupt_update = _first_update(body, last_status="interrupted")
    benchmark = body.voice_dialogue_state["voice_chain_benchmark"]
    assert interrupt_update["stop_speech_status"] == "ok"
    assert interrupt_update["tts_stop_confirmed"] is False
    assert interrupt_update["interrupt_to_tts_stop_ms"] is None
    assert "interruptStopMs" not in benchmark["metrics"]
    assert "interruptStopMs" not in benchmark["recentTraces"][-1]
    assert benchmark["interruptStop"]["requiredCount"] == 1
    assert benchmark["interruptStop"]["confirmedCount"] == 0
    assert benchmark["interruptStop"]["ready"] is False


def test_voice_dialogue_loop_request_interrupt_surfaces_stop_failure() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop
    from eibrain.cognition.realtime.turn import RealtimeTurnManager

    class _StopFailedBody(_Body):
        def dispatch_actions(self, actions):
            if any(isinstance(action, StopSpeechAction) for action in actions):
                details = {"last_error": "device busy", "busy": True}
                return [type("Outcome", (), {"status": "stop_failed", "details": details})()]
            return super().dispatch_actions(actions)

    body = _StopFailedBody([""])
    cognition = _Cognition()
    turn_manager = RealtimeTurnManager()
    turn_manager.start_round(reason="test")
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=cognition,
        realtime_turn_manager=turn_manager,
        idle_interval_s=0.01,
        empty_interval_s=0.01,
    )

    loop.request_interrupt(reason="user_barge_in")

    interrupt_update = _first_update(body, last_status="interrupted")
    assert interrupt_update["stop_speech_status"] == "stop_failed"
    assert interrupt_update["stop_speech_error"] == "device busy"
    assert 0 <= interrupt_update["stop_dispatch_elapsed_ms"] <= 300
    assert interrupt_update["interrupt_to_tts_stop_ms"] is None
    assert interrupt_update["tts_stop_confirmed"] is False


def test_voice_dialogue_loop_reports_degraded_when_reply_dispatch_fails() -> None:
    class _DegradedSpeechBody(_Body):
        def dispatch_actions(self, actions):
            self.dispatched.extend(actions)
            if any(isinstance(action, PlaySpeechAction) for action in actions):
                return [type("Outcome", (), {"status": "error", "details": {"error": "speaker failed"}})()]
            return super().dispatch_actions(actions)

    body = _DegradedSpeechBody(["你好"])
    cognition = _Cognition(reply="我在。")
    loop = _start_loop(body, cognition, initial_conversation_active=True)

    _wait_until(lambda: any(update.get("last_status") == "reply_degraded" for update in body.updates))
    loop.stop()

    degraded_update = _first_update(body, last_status="reply_degraded")
    assert degraded_update["last_error"] == "speech_dispatch_degraded"
    assert degraded_update["last_completed_turn"]["status"] == "degraded"
    assert not any(update.get("last_status") == "reply_ready" for update in body.updates)


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
    benchmark = body.voice_dialogue_state["voice_chain_benchmark"]
    stale_traces = [trace for trace in benchmark["recentTraces"] if trace.get("status") == "stale_round_blocked"]
    assert benchmark["roundLeakCount"] == 1
    assert stale_traces
    assert stale_traces[-1]["roundLeak"] is True


def test_voice_dialogue_loop_gates_real_actions_not_synthetic_plan() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop
    from eibrain.cognition.realtime import RealtimeTurnManager

    class _CapturingArbiter:
        def __init__(self) -> None:
            self.plans = []

        def allow_speaking(self, manager, turn, plan):
            self.plans.append(plan)
            return True

    body = _Body([""])
    turn_manager = RealtimeTurnManager()
    turn = turn_manager.start_round(reason="test")
    turn_manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="介绍下你自己",
    )
    arbiter = _CapturingArbiter()
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=_Cognition(),
        realtime_turn_manager=turn_manager,
        response_arbiter=arbiter,
    )
    action = PlaySpeechAction(
        ts=1.0,
        source="test",
        session_id="s1",
        actor_id="user-1",
        text="我是鸿途。",
    )

    assert loop._actions_allowed_for_turn(turn, [action], "我是鸿途。") is True
    assert arbiter.plans
    assert arbiter.plans[-1]["actions"][0]["capabilityId"] == "speech.play"
    assert arbiter.plans[-1]["actions"][0]["payload"]["text"] == "我是鸿途。"


def test_voice_dialogue_loop_reports_blocked_arbiter_verdict_in_scheduler_state() -> None:
    from apps.body_runtime.voice_dialogue_loop import VoiceDialogueLoop
    from eibrain.cognition.realtime import RealtimeTurnManager

    class _DenyingArbiter:
        def allow_speaking(self, manager, turn, plan):
            return False

    body = _Body([""])
    turn_manager = RealtimeTurnManager()
    turn = turn_manager.start_round(reason="test")
    turn_manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="介绍下你自己",
    )
    loop = VoiceDialogueLoop(
        body_runtime=body,
        cognitive_runtime=_Cognition(),
        realtime_turn_manager=turn_manager,
        response_arbiter=_DenyingArbiter(),
    )
    action = PlaySpeechAction(
        ts=1.0,
        source="test",
        session_id="s1",
        actor_id="user-1",
        text="我是鸿途。",
    )

    assert loop._actions_allowed_for_turn(turn, [action], "我是鸿途。") is False
    scheduler_state = loop._scheduler_state_payload(turn)
    assert scheduler_state["arbiter"]["state"] == "blocked"
    assert scheduler_state["arbiter"]["can_speak"] is False


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
