"""Continuous honjia voice dialogue loop."""

from __future__ import annotations

import re
import threading
import time
from typing import TYPE_CHECKING, Iterable

from apps.body_runtime.app import BodyRuntimeApp
from eibrain.cognition.realtime import (
    FastThinkEngine,
    InterruptionController,
    RealtimeTurnManager,
    ResponseArbiter,
    SpeechActionPlanner,
    TurnBlackboard,
)
from eibrain.protocol.actions import PlaySpeechAction, StopSpeechAction
from eibrain.protocol.observations import AudioTranscriptFinal

try:  # Task A may not be present on every branch yet.
    from eibrain.body.realtime_voice import RealtimeVoiceSession
except Exception:  # pragma: no cover - compatibility shim
    RealtimeVoiceSession = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp


class _FallbackRealtimeVoiceSession:
    """Small local snapshot model used when Task A's session object is absent."""

    def __init__(
        self,
        *,
        session_id: str,
        actor_id: str,
        round_id: str,
        cancellation_token: str,
    ) -> None:
        self.session_id = session_id
        self.actor_id = actor_id
        self.round_id = round_id
        self.cancellation_token = cancellation_token
        self.phase = "idle"
        self.status = "idle"
        self.transcript_final = ""
        self.reply_text = ""
        self.interrupted = False
        self.interrupt_reason = ""
        self.started_at_s: float | None = None
        self.final_asr_at_s: float | None = None
        self.first_reply_at_s: float | None = None
        self.first_speech_at_s: float | None = None
        self.completed_at_s: float | None = None
        self.events: list[dict[str, object]] = []

    def start_listening(self, **_: object) -> None:
        self.started_at_s = time.perf_counter()
        self.phase = "listening"
        self.status = "waiting_for_audio"
        self._record("listening", "waiting_for_audio", lane="listening", event_type="listening_started")

    def finalize_transcript(self, text: str, **_: object) -> None:
        self.final_asr_at_s = time.perf_counter()
        self.transcript_final = text.strip()
        self.phase = "thinking_stream"
        self.status = "final_transcript"
        self._record(
            "thinking_stream",
            "final_transcript",
            transcript=self.transcript_final,
            lane="listening",
            event_type="asr_final",
        )

    def update_microfeedback(self, text: str, **_: object) -> None:
        self._record(
            self.phase,
            "microfeedback",
            detail=text.strip(),
            lane="fast_think",
            event_type="microfeedback",
            payload={"text": text.strip()},
        )

    def append_reply_delta(self, delta: str, **_: object) -> None:
        if self.first_reply_at_s is None:
            self.first_reply_at_s = time.perf_counter()
        self.reply_text += delta
        self.phase = "thinking_stream"
        self.status = "reply_delta"
        self._record(
            "thinking_stream",
            "reply_delta",
            reply_delta=delta,
            lane="slow_thinking",
            event_type="agent_think",
        )

    def start_speaking(self, **_: object) -> None:
        if self.first_speech_at_s is None:
            self.first_speech_at_s = time.perf_counter()
        self.phase = "speaking_stream"
        self.status = "speech_started"
        self._record("speaking_stream", "speech_started", lane="speaking", event_type="tts_started")

    def complete(self, *, status: str = "ok", **_: object) -> None:
        self.completed_at_s = time.perf_counter()
        self.phase = "completed"
        self.status = status
        self._record("completed", status, lane="complete", event_type="complete")

    def fail(self, error: str, **_: object) -> None:
        self.completed_at_s = time.perf_counter()
        self.phase = "error"
        self.status = "error"
        self._record("error", "error", detail=error, lane="complete", event_type="error")

    def interrupt(self, *, reason: str = "user_barge_in", **_: object) -> None:
        self.interrupted = True
        self.interrupt_reason = reason
        self.phase = "barge_in"
        self.status = "interrupted"
        self._record(
            "barge_in",
            "interrupted",
            detail=reason,
            lane="interrupt",
            event_type="interrupt",
            payload={"reason": reason},
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "actor_id": self.actor_id,
            "round_id": self.round_id,
            "roundId": self.round_id,
            "cancellation_token": self.cancellation_token,
            "cancellationToken": self.cancellation_token,
            "phase": self.phase,
            "status": self.status,
            "transcript_final": self.transcript_final,
            "reply_text": self.reply_text,
            "interrupted": self.interrupted,
            "interrupt_reason": self.interrupt_reason,
            "latency_ms": self.latency_ms(),
            "event_count": len(self.events),
            "events": [dict(event) for event in self.events],
        }

    def latency_ms(self) -> dict[str, float]:
        started = self.started_at_s
        if started is None:
            return {}
        result: dict[str, float] = {}
        if self.final_asr_at_s is not None:
            result["final_asr"] = self._elapsed_ms(started, self.final_asr_at_s)
        if self.first_reply_at_s is not None:
            result["first_reply_token"] = self._elapsed_ms(started, self.first_reply_at_s)
        if self.first_speech_at_s is not None:
            result["first_speech"] = self._elapsed_ms(started, self.first_speech_at_s)
        if self.completed_at_s is not None:
            result["total"] = self._elapsed_ms(started, self.completed_at_s)
        return result

    def _record(
        self,
        phase: str,
        status: str,
        *,
        transcript: str = "",
        reply_delta: str = "",
        detail: str = "",
        lane: str,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        self.events.append(
            {
                "phase": phase,
                "status": status,
                "lane": lane,
                "event_type": event_type,
                "transcript": transcript,
                "reply_delta": reply_delta,
                "detail": detail,
                "at_s": time.perf_counter(),
                "round_id": self.round_id,
                "roundId": self.round_id,
                "cancellation_token": self.cancellation_token,
                "cancellationToken": self.cancellation_token,
                "payload": dict(payload or {}),
            }
        )

    @staticmethod
    def _elapsed_ms(start_s: float, end_s: float) -> float:
        return round(max(0.0, end_s - start_s) * 1000, 2)


class VoiceDialogueLoop:
    def __init__(
        self,
        *,
        body_runtime: BodyRuntimeApp,
        cognitive_runtime: CognitiveRuntimeApp,
        chunk_count: int = 2,
        max_chunk_count: int = 4,
        min_chunk_count: int = 1,
        idle_interval_s: float = 0.5,
        empty_interval_s: float = 0.25,
        session_id: str = "voice-dialogue-loop",
        actor_id: str = "darrow",
        wake_word: str = "\u9e3f\u9014",
        sleep_word: str = "\u7ed3\u675f\u5bf9\u8bdd",
        initial_conversation_active: bool = False,
        engagement_writer: object | None = None,
        waking_phrase: str = "\u6211\u5728\u3002",
        sleeping_phrase: str = "\u597d\u7684\uff0c\u5148\u4f11\u606f\u3002",
        realtime_turn_manager: RealtimeTurnManager | None = None,
        fast_think_engine: FastThinkEngine | None = None,
        response_arbiter: ResponseArbiter | None = None,
        interruption_controller: InterruptionController | None = None,
        speech_action_planner: SpeechActionPlanner | None = None,
    ) -> None:
        self.body_runtime = body_runtime
        self.cognitive_runtime = cognitive_runtime
        self.wake_word = wake_word
        self.sleep_word = sleep_word
        self.waking_phrase = waking_phrase
        self.sleeping_phrase = sleeping_phrase
        self.conversation_active = bool(initial_conversation_active)
        self.engagement_writer = engagement_writer
        self.chunk_count = max(1, int(chunk_count))
        self.max_chunk_count = max(1, int(max_chunk_count))
        self.min_chunk_count = max(1, int(min_chunk_count))
        if self.min_chunk_count > self.max_chunk_count:
            self.min_chunk_count = self.max_chunk_count
        if self.chunk_count > self.max_chunk_count:
            self.chunk_count = self.max_chunk_count
        if self.chunk_count < self.min_chunk_count:
            self.chunk_count = self.min_chunk_count
        self._rolling_chunk_count = self.chunk_count
        self.idle_interval_s = idle_interval_s
        self.empty_interval_s = empty_interval_s
        self.session_id = session_id
        self.actor_id = actor_id
        self._last_engagement_state = self.conversation_active
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.realtime_turn_manager = realtime_turn_manager or RealtimeTurnManager()
        self.fast_think_engine = fast_think_engine or FastThinkEngine()
        self.response_arbiter = response_arbiter or ResponseArbiter()
        self.interruption_controller = interruption_controller or InterruptionController()
        self.speech_action_planner = speech_action_planner or SpeechActionPlanner()
        self._turn_lock = threading.RLock()
        self._interrupted_round_count = 0
        self._last_microfeedback: dict[str, object] | None = None
        self._realtime_session: object | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        payload = {
            "enabled": True,
            "running": True,
            "phase": "starting",
            "last_status": "starting",
            "last_error": "",
            "wake_word": self.wake_word,
            "sleep_word": self.sleep_word,
            "conversation_active": self.conversation_active,
            "last_reply": "",
        }
        payload.update(self._round_state_payload())
        self.body_runtime.update_voice_dialogue_state(**payload)
        self._publish_engagement_state(phase="running" if self.conversation_active else "sleeping")
        self._thread = threading.Thread(target=self._run, name="voice-dialogue-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None
        payload = {
            "running": False,
            "phase": "stopped",
            "last_status": "stopped",
            "conversation_active": self.conversation_active,
            "wake_word": self.wake_word,
            "sleep_word": self.sleep_word,
        }
        payload.update(self._round_state_payload())
        self.body_runtime.update_voice_dialogue_state(**payload)
        self._publish_engagement_state(
            phase="stopped",
            conversation_active=self.conversation_active,
            reason="loop_stopped",
        )

    def _publish_engagement_state(
        self,
        *,
        phase: str,
        conversation_active: bool | None = None,
        reason: str = "",
    ) -> None:
        if self.engagement_writer is None:
            return
        next_conversation_active = self.conversation_active if conversation_active is None else conversation_active
        if next_conversation_active == self._last_engagement_state and phase != "stopped":
            return
        try:
            self.engagement_writer.write(
                conversation_active=next_conversation_active,
                phase=phase,
                reason=reason,
                security_mode=False,
            )
            self._last_engagement_state = next_conversation_active
        except Exception:
            self.body_runtime.update_voice_dialogue_state(
                last_status="engagement_writer_error",
                phase="error",
                last_error="engagement write failed",
            )

    def request_interrupt(self, *, reason: str = "user_barge_in") -> dict[str, object]:
        with self._turn_lock:
            old_turn = self.realtime_turn_manager.current_turn()
            old_realtime_session = self._realtime_session
            interruption = self.interruption_controller.interrupt_and_start_new_round(
                self.realtime_turn_manager,
                reason=reason,
            )
            new_turn = self.realtime_turn_manager.current_turn()
            if old_turn is not None and old_turn.state == "interrupted":
                self._interrupted_round_count += 1
            self._last_microfeedback = None

        self._call_realtime(old_realtime_session, "interrupt", reason=reason, turn=old_turn)
        stop_status, stop_error = self._dispatch_stop_speech()
        self._publish_state(
            phase="listening" if self.conversation_active else "idle",
            last_status="interrupted",
            turn=new_turn,
            realtime_voice_session=old_realtime_session,
            interruption=interruption,
            stop_speech_status=stop_status,
            stop_speech_error=stop_error,
            interrupt_active=True,
            last_interrupt=interruption,
            last_error="",
        )
        return interruption

    def _start_round(self, *, reason: str) -> TurnBlackboard:
        with self._turn_lock:
            self._last_microfeedback = None
            return self.realtime_turn_manager.start_round(reason=reason)

    def _finalize_asr(
        self,
        turn: TurnBlackboard,
        transcript: str,
    ) -> TurnBlackboard:
        with self._turn_lock:
            finalized = self.realtime_turn_manager.finalize_asr(
                round_id=turn.round_id,
                cancellation_token=turn.cancellation_token,
                asr_text=transcript,
            )
            if transcript.strip():
                fast_result = self.fast_think_engine.process_partial(finalized, transcript)
                self._last_microfeedback = {
                    "text": fast_result.microfeedback,
                    "deadline_ms": fast_result.deadline_ms,
                    "source": fast_result.source,
                    "stable": fast_result.stable,
                }
            return finalized

    def _is_current_turn(self, turn: TurnBlackboard) -> bool:
        with self._turn_lock:
            return self.realtime_turn_manager.is_current(
                round_id=turn.round_id,
                cancellation_token=turn.cancellation_token,
            )

    def _start_realtime_session(self, turn: TurnBlackboard) -> object:
        session_cls = RealtimeVoiceSession or _FallbackRealtimeVoiceSession
        session = session_cls(
            session_id=self.session_id,
            actor_id=self.actor_id,
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
        )
        self._call_realtime(session, "start_listening", turn=turn)
        with self._turn_lock:
            self._realtime_session = session
        return session

    def _realtime_session_for_turn(self, turn: TurnBlackboard | None = None) -> object | None:
        with self._turn_lock:
            session = self._realtime_session
        if session is None or turn is None:
            return session
        snapshot = self._realtime_snapshot(session)
        if snapshot.get("round_id") != turn.round_id:
            return None
        return session

    def _call_realtime(
        self,
        session: object | None,
        method_name: str,
        *args: object,
        turn: TurnBlackboard | None = None,
        **kwargs: object,
    ) -> None:
        if session is None:
            return
        method = getattr(session, method_name, None)
        if not callable(method):
            return
        if turn is not None:
            kwargs.setdefault("round_id", turn.round_id)
            kwargs.setdefault("cancellation_token", turn.cancellation_token)
        try:
            method(*args, **kwargs)
        except TypeError:
            try:
                method(*args)
            except (RuntimeError, ValueError, TypeError):
                return
        except (RuntimeError, ValueError):
            return

    def _realtime_snapshot(self, session: object | None) -> dict[str, object]:
        if session is None:
            return {}
        snapshot = getattr(session, "snapshot", None)
        if callable(snapshot):
            try:
                value = snapshot()
                if isinstance(value, dict):
                    return value
            except (RuntimeError, ValueError, TypeError):
                return {}
        return {}

    def _realtime_updates(
        self,
        session: object | None,
        *,
        last_reply_delta: str | None = None,
    ) -> dict[str, object]:
        snapshot = self._realtime_snapshot(session)
        if not snapshot:
            return {}
        events = list(snapshot.get("events", []) or [])
        if last_reply_delta is None:
            for event in reversed(events):
                if isinstance(event, dict) and event.get("reply_delta"):
                    last_reply_delta = str(event.get("reply_delta") or "")
                    break
        latency_ms = dict(snapshot.get("latency_ms", {}) or {})
        return {
            "realtime_session": snapshot,
            "realtime_events": events,
            "last_reply_delta": last_reply_delta or "",
            "closed_loop_state": self._closed_loop_state(snapshot, events),
            "realtime_latency_ms": latency_ms,
        }

    @staticmethod
    def _closed_loop_state(
        snapshot: dict[str, object],
        events: list[object],
    ) -> dict[str, bool]:
        event_types = {
            str(event.get("event_type", ""))
            for event in events
            if isinstance(event, dict)
        }
        phase = str(snapshot.get("phase", ""))
        return {
            "listening": "listening_started" in event_types or phase == "listening",
            "final_asr": "asr_final" in event_types or bool(snapshot.get("transcript_final")),
            "reply_delta": "agent_think" in event_types or bool(snapshot.get("reply_text")),
            "speaking": "tts_started" in event_types or bool(snapshot.get("first_speech_at_s")),
            "complete": "complete" in event_types or phase == "completed" or bool(snapshot.get("complete")),
            "error": "error" in event_types or phase == "error",
            "interrupted": "interrupt" in event_types or phase == "barge_in" or bool(snapshot.get("interrupted")),
        }

    def _round_state_payload(self, turn: TurnBlackboard | None = None) -> dict[str, object]:
        with self._turn_lock:
            current_turn = self.realtime_turn_manager.current_turn()
            selected_turn = turn or current_turn
            current_round_id = current_turn.round_id if current_turn is not None else ""
            current_cancellation_token = (
                current_turn.cancellation_token if current_turn is not None else ""
            )
            scheduler_state = (
                selected_turn.to_dict()
                if selected_turn is not None
                else {"state": "idle", "interrupted_round_count": self._interrupted_round_count}
            )
            interrupted_round_count = self._interrupted_round_count

        payload: dict[str, object] = {
            "current_round_id": current_round_id,
            "current_cancellation_token": current_cancellation_token,
            "scheduler_state": scheduler_state,
            "interrupted_round_count": interrupted_round_count,
            "microfeedback": dict(self._last_microfeedback or {}),
        }
        if selected_turn is not None:
            payload["round_id"] = selected_turn.round_id
            payload["cancellation_token"] = selected_turn.cancellation_token
        return payload

    def _publish_stale_round(
        self,
        turn: TurnBlackboard,
        *,
        reason: str,
        last_transcript: str = "",
        last_reply: str = "",
        last_error: str = "",
        realtime_voice_session: object | None = None,
    ) -> None:
        self._publish_state(
            phase="idle",
            last_status="stale_round_blocked",
            turn=turn,
            realtime_voice_session=realtime_voice_session,
            last_transcript=last_transcript,
            last_reply=last_reply,
            last_error=last_error,
            stale_round={
                "round_id": turn.round_id,
                "cancellation_token": turn.cancellation_token,
                "state": turn.state,
                "reason": reason,
            },
        )

    def _dispatch_stop_speech(self) -> tuple[str, str]:
        action = StopSpeechAction(
            ts=time.time(),
            source="voice_dialogue_loop",
            session_id=self.session_id,
            actor_id=self.actor_id,
            reason="voice_loop_interrupt",
            details={"reason": "voice_loop_interrupt"},
        )
        try:
            outcomes = self.body_runtime.dispatch_actions([action])
        except Exception as exc:
            return "unsupported", str(exc)
        if outcomes:
            statuses = [str(getattr(outcome, "status", "") or "") for outcome in outcomes]
            ok_statuses = {"ok", "healthy", "completed", "stopped"}
            if all(status in ok_statuses for status in statuses):
                return "ok", ""
            first_status = next((status for status in statuses if status), "failed")
            details = getattr(outcomes[0], "details", None)
            error = ""
            if isinstance(details, dict):
                error = str(details.get("last_error") or details.get("error") or details.get("reason") or "")
            return first_status, error
        return "not_supported", ""

    def _actions_allowed_for_turn(
        self,
        turn: TurnBlackboard,
        actions: list[object],
        reply: str,
    ) -> bool:
        if not self._is_current_turn(turn):
            return False
        if not actions or not reply:
            return True
        plan = self.speech_action_planner.plan(turn, speech_text=reply)
        with self._turn_lock:
            return self.response_arbiter.allow_speaking(
                self.realtime_turn_manager,
                turn,
                plan,
            )

    @staticmethod
    def _strip_trigger(text: str, trigger: str) -> tuple[str, bool]:
        value = text.strip()
        if not value or not trigger:
            return value, False
        if value == trigger:
            return "", True
        pattern = re.compile(r"^\s*" + re.escape(trigger) + r"[，,、\\s:.!！?？:：]*")
        match = pattern.match(value)
        if match is None:
            return value, False
        remainder = value[match.end() :].strip()
        return remainder, True

    def _publish_state(
        self,
        *,
        phase: str,
        last_status: str,
        turn: TurnBlackboard | None = None,
        realtime_voice_session: object | None = None,
        **updates: object,
    ) -> None:
        payload = {
            "phase": phase,
            "last_status": last_status,
            "running": True,
            "enabled": True,
            "wake_word": self.wake_word,
            "sleep_word": self.sleep_word,
            "conversation_active": self.conversation_active,
        }
        payload.update(self._round_state_payload(turn))
        session = realtime_voice_session or self._realtime_session_for_turn(turn)
        payload.update(self._realtime_updates(session))
        if last_status not in {"interrupted", "stale_round_blocked"}:
            payload.setdefault("interrupt_active", False)
        payload.update(updates)
        payload.setdefault("last_reply", self.body_runtime.voice_dialogue_state.get("last_reply", ""))
        self.body_runtime.update_voice_dialogue_state(**payload)

    def _dispatch_ack_reply(self, text: str, turn: TurnBlackboard) -> bool:
        action = PlaySpeechAction(
            ts=time.time(),
            source="voice_dialogue_loop",
            session_id=self.session_id,
            actor_id=self.actor_id,
            text=text,
        )
        if not self._actions_allowed_for_turn(turn, [action], text):
            return False
        if not self._is_current_turn(turn):
            return False
        self.body_runtime.dispatch_actions([action])
        return True

    def _publish_stale_ack(
        self,
        turn: TurnBlackboard,
        *,
        reason: str,
        last_transcript: str,
    ) -> None:
        self._publish_stale_round(
            turn,
            reason=reason,
            last_transcript=last_transcript,
            last_reply="",
        )

    def _replace_transcript(
        self,
        observation: AudioTranscriptFinal,
        text: str,
    ) -> AudioTranscriptFinal:
        if observation.text == text:
            return observation
        return AudioTranscriptFinal(
            ts=observation.ts,
            source=observation.source,
            text=text,
            language=getattr(observation, "language", "und"),
            session_id=observation.session_id,
            actor_id=observation.actor_id,
            target_id=observation.target_id,
        )

    def _streaming_facade(self):
        for name in (
            "stream_observation",
            "handle_observation_stream",
            "stream_handle_observation",
            "stream_response",
        ):
            facade = getattr(self.cognitive_runtime, name, None)
            if callable(facade):
                return facade
        return None

    def _open_cognitive_stream(
        self,
        facade,
        observation: AudioTranscriptFinal,
        turn: TurnBlackboard,
        session: object,
    ) -> Iterable[object]:
        try:
            return facade(
                observation,
                round_id=turn.round_id,
                cancellation_token=turn.cancellation_token,
                realtime_session=session,
            )
        except TypeError:
            try:
                return facade(
                    observation,
                    round_id=turn.round_id,
                    cancellation_token=turn.cancellation_token,
                )
            except TypeError:
                return facade(observation)

    def _run_cognition_turn(
        self,
        observation: AudioTranscriptFinal,
        turn: TurnBlackboard,
        session: object,
    ) -> tuple[list[object], str, float, str, bool]:
        think_started = time.perf_counter()
        facade = self._streaming_facade()
        if facade is None:
            actions = list(self.cognitive_runtime.handle_observation(observation) or [])
            reply = self._reply_from_actions(actions)
            think_s = time.perf_counter() - think_started
            if not self._is_current_turn(turn):
                return actions, reply, think_s, "round_not_current_or_unstable", False
            return actions, reply, think_s, "", False

        actions: list[object] = []
        reply_parts: list[str] = []
        emitted_reply_delta = False
        for item in self._open_cognitive_stream(facade, observation, turn, session):
            if not self._is_current_turn(turn):
                return actions, "".join(reply_parts), time.perf_counter() - think_started, "streaming_round_not_current", emitted_reply_delta
            delta = self._reply_delta_from_stream_item(item)
            if delta:
                reply_parts.append(delta)
                emitted_reply_delta = True
                self._call_realtime(session, "append_reply_delta", delta, turn=turn)
                self._publish_state(
                    phase="thinking",
                    last_status="reply_delta",
                    turn=turn,
                    realtime_voice_session=session,
                    conversation_active=True,
                    last_transcript=observation.text,
                    last_reply="".join(reply_parts),
                    last_reply_delta=delta,
                    last_error="",
                )
            item_actions = self._actions_from_stream_item(item, observation)
            if item_actions:
                actions.extend(item_actions)

        reply = self._reply_from_actions(actions) or "".join(reply_parts)
        if not actions and reply:
            actions = [
                PlaySpeechAction(
                    ts=time.time(),
                    source="voice_dialogue_loop",
                    session_id=observation.session_id,
                    actor_id=observation.actor_id,
                    text=reply,
                )
            ]
        return actions, reply, time.perf_counter() - think_started, "", emitted_reply_delta

    @staticmethod
    def _reply_from_actions(actions: list[object]) -> str:
        for action in actions:
            if getattr(action, "kind", "") == "play_speech_action":
                return str(getattr(action, "text", "") or "")
        return ""

    @staticmethod
    def _reply_delta_from_stream_item(item: object) -> str:
        if isinstance(item, dict):
            item_type = str(item.get("type") or item.get("kind") or item.get("event") or item.get("event_type") or "")
            if item_type in {"reply_delta", "delta", "agent_think", "thinking_delta"}:
                return str(item.get("delta") or item.get("text") or item.get("reply_delta") or "")
            return ""
        item_type = str(
            getattr(item, "type", "")
            or getattr(item, "kind", "")
            or getattr(item, "event", "")
            or getattr(item, "event_type", "")
        )
        if item_type in {"reply_delta", "delta", "agent_think", "thinking_delta"}:
            return str(
                getattr(item, "delta", "")
                or getattr(item, "text", "")
                or getattr(item, "reply_delta", "")
                or ""
            )
        return ""

    @staticmethod
    def _actions_from_stream_item(item: object, observation: AudioTranscriptFinal | None = None) -> list[object]:
        if isinstance(item, list):
            return VoiceDialogueLoop._coerce_stream_actions(item, observation)
        if isinstance(item, dict):
            actions = item.get("actions")
            if isinstance(actions, list):
                return VoiceDialogueLoop._coerce_stream_actions(actions, observation)
            action = item.get("action")
            return VoiceDialogueLoop._coerce_stream_actions([action], observation) if action is not None else []
        if getattr(item, "kind", "") == "play_speech_action":
            return [item]
        actions = getattr(item, "actions", None)
        if isinstance(actions, list):
            return VoiceDialogueLoop._coerce_stream_actions(actions, observation)
        return []

    @staticmethod
    def _coerce_stream_actions(actions: list[object], observation: AudioTranscriptFinal | None) -> list[object]:
        coerced: list[object] = []
        for action in actions:
            if isinstance(action, dict):
                payload = action
                kind = str(payload.get("kind") or payload.get("type") or "")
                if kind != "play_speech_action":
                    continue
                text = str(payload.get("text") or payload.get("reply_text") or "")
                if not text:
                    continue
                coerced.append(
                    PlaySpeechAction(
                        ts=float(payload.get("ts") or time.time()),
                        source=str(payload.get("source") or "voice_dialogue_loop"),
                        session_id=str(payload.get("session_id") or getattr(observation, "session_id", "") or ""),
                        actor_id=str(payload.get("actor_id") or getattr(observation, "actor_id", "") or ""),
                        target_id=str(payload.get("target_id") or getattr(observation, "target_id", "") or ""),
                        text=text,
                    )
                )
                continue
            coerced.append(action)
        return coerced

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.body_runtime.is_speaking():
                    self._publish_state(
                        phase="speaking",
                        last_status="playback_active",
                    )
                    self._sleep(self.idle_interval_s)
                    continue
                turn_started = time.perf_counter()
                turn = self._start_round(reason="listening")
                realtime_session = self._start_realtime_session(turn)
                self._publish_state(
                    phase="listening",
                    last_status="listening",
                    turn=turn,
                    realtime_voice_session=realtime_session,
                )
                listen_started = time.perf_counter()
                chunk_count = max(
                    self.min_chunk_count,
                    min(self.max_chunk_count, self._rolling_chunk_count),
                )
                observation = self.body_runtime.transcribe_audio_window(
                    chunk_count=chunk_count,
                    session_id=self.session_id,
                    actor_id=self.actor_id,
                )
                listen_asr_s = time.perf_counter() - listen_started
                transcript = observation.text.strip()
                try:
                    turn = self._finalize_asr(turn, transcript)
                except RuntimeError as exc:
                    self._call_realtime(realtime_session, "fail", "finalize_asr_rejected", turn=turn)
                    self._publish_stale_round(
                        turn,
                        reason="finalize_asr_rejected",
                        last_transcript=transcript,
                        last_error=str(exc),
                        realtime_voice_session=realtime_session,
                    )
                    self._sleep(self.empty_interval_s)
                    continue
                self._call_realtime(realtime_session, "finalize_transcript", transcript, turn=turn)
                if self._last_microfeedback:
                    self._call_realtime(
                        realtime_session,
                        "update_microfeedback",
                        str(self._last_microfeedback.get("text") or ""),
                        turn=turn,
                    )

                if not self.conversation_active:
                    wake_transcript, woke = self._strip_trigger(transcript, self.wake_word)
                    if not woke:
                        if not transcript:
                            status = "no_transcript"
                            self._rolling_chunk_count = max(self.min_chunk_count, chunk_count - 1)
                        else:
                            status = "waiting_for_wake_word"
                            self._rolling_chunk_count = max(self.min_chunk_count, chunk_count - 1)
                        stage_latency_ms = self._stage_latency_ms(
                            listen_asr_s=listen_asr_s,
                            total_s=time.perf_counter() - turn_started,
                        )
                        self._call_realtime(realtime_session, "complete", status=status, turn=turn)
                        self._publish_state(
                            phase="idle",
                            last_status=status,
                            turn=turn,
                            realtime_voice_session=realtime_session,
                            conversation_active=False,
                            last_transcript=transcript,
                            last_latency_s={
                                "listen_asr": round(listen_asr_s, 2),
                                "total": round(time.perf_counter() - turn_started, 2),
                            },
                            last_stage_latency_ms=stage_latency_ms,
                            last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                            last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                        )
                        self._publish_engagement_state(
                            phase="sleeping",
                            conversation_active=False,
                            reason="waiting_for_wake_word",
                        )
                        self._sleep(self.empty_interval_s)
                        continue

                    self.conversation_active = True
                    transcript = wake_transcript
                    observation = self._replace_transcript(observation, transcript)
                    self._publish_engagement_state(
                        phase="running",
                        conversation_active=True,
                        reason="wake_word_detected",
                    )
                    if not transcript:
                        self._publish_state(
                            phase="idle",
                            last_status="wake_acknowledged",
                            turn=turn,
                            conversation_active=True,
                            last_transcript=self.wake_word,
                            last_reply=self.waking_phrase,
                        )
                        think_started = time.perf_counter()
                        ack_dispatched = self._dispatch_ack_reply(self.waking_phrase, turn)
                        if not ack_dispatched:
                            self._publish_stale_ack(
                                turn,
                                reason="wake_ack_round_not_current",
                                last_transcript=self.wake_word,
                            )
                            self._sleep(self.empty_interval_s)
                            continue
                        self._call_realtime(realtime_session, "append_reply_delta", self.waking_phrase, turn=turn)
                        self._call_realtime(realtime_session, "start_speaking", turn=turn)
                        think_s = time.perf_counter() - think_started
                        total_s = time.perf_counter() - turn_started
                        stage_latency_ms = self._stage_latency_ms(
                            listen_asr_s=listen_asr_s,
                            think_s=think_s,
                            total_s=total_s,
                        )
                        self._call_realtime(realtime_session, "complete", status="wake_acknowledged", turn=turn)
                        self._publish_state(
                            phase="idle",
                            last_status="wake_acknowledged",
                            turn=turn,
                            realtime_voice_session=realtime_session,
                            conversation_active=True,
                            last_transcript=self.wake_word,
                            last_reply=self.waking_phrase,
                            last_reply_delta=self.waking_phrase,
                            last_latency_s={
                                "listen_asr": round(listen_asr_s, 2),
                                "think": round(think_s, 2),
                                "speak": 0.0,
                                "total": round(total_s, 2),
                            },
                            last_stage_latency_ms=stage_latency_ms,
                            last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                            last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                            last_error="",
                        )
                        self._sleep(self.empty_interval_s)
                        continue

                if not transcript:
                    stage_latency_ms = self._stage_latency_ms(
                        listen_asr_s=listen_asr_s,
                        total_s=time.perf_counter() - turn_started,
                    )
                    self._rolling_chunk_count = max(self.min_chunk_count, chunk_count - 1)
                    self._call_realtime(realtime_session, "complete", status="no_transcript", turn=turn)
                    self._publish_state(
                        phase="idle",
                        last_status="no_transcript",
                        turn=turn,
                        realtime_voice_session=realtime_session,
                        last_transcript="",
                        last_error="",
                        last_latency_s={
                            "listen_asr": round(listen_asr_s, 2),
                            "total": round(time.perf_counter() - turn_started, 2),
                        },
                        last_stage_latency_ms=stage_latency_ms,
                        last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                        last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                    )
                    self._sleep(self.empty_interval_s)
                    continue

                transcript_for_cognitive, requested_sleep = self._strip_trigger(
                    transcript,
                    self.sleep_word,
                )
                if requested_sleep:
                    transcript = transcript_for_cognitive
                if requested_sleep and not transcript:
                    think_started = time.perf_counter()
                    self.conversation_active = False
                    self._publish_engagement_state(
                        phase="stopped",
                        conversation_active=False,
                        reason="sleep_word_detected",
                    )
                    self._publish_state(
                        phase="idle",
                        last_status="sleep_acknowledged",
                        turn=turn,
                        conversation_active=False,
                        last_transcript=self.sleep_word,
                        last_reply=self.sleeping_phrase,
                    )
                    ack_dispatched = self._dispatch_ack_reply(self.sleeping_phrase, turn)
                    if not ack_dispatched:
                        self._publish_stale_ack(
                            turn,
                            reason="sleep_ack_round_not_current",
                            last_transcript=self.sleep_word,
                        )
                        self._sleep(self.empty_interval_s)
                        continue
                    self._call_realtime(realtime_session, "append_reply_delta", self.sleeping_phrase, turn=turn)
                    self._call_realtime(realtime_session, "start_speaking", turn=turn)
                    think_s = time.perf_counter() - think_started
                    self._call_realtime(realtime_session, "complete", status="sleep_acknowledged", turn=turn)
                    self._publish_state(
                        phase="idle",
                        last_status="sleep_acknowledged",
                        turn=turn,
                        realtime_voice_session=realtime_session,
                        conversation_active=False,
                        last_transcript=self.sleep_word,
                        last_reply=self.sleeping_phrase,
                        last_reply_delta=self.sleeping_phrase,
                        last_error="",
                        last_latency_s={
                            "listen_asr": round(listen_asr_s, 2),
                            "think": round(think_s, 2),
                            "speak": 0.0,
                            "total": round(time.perf_counter() - turn_started, 2),
                        },
                    )
                    self._sleep(self.empty_interval_s)
                    continue

                if len(transcript_for_cognitive) <= 1:
                    self._rolling_chunk_count = max(self.min_chunk_count, chunk_count - 1)
                    self._call_realtime(realtime_session, "complete", status="short_transcript_ignored", turn=turn)
                    self._publish_state(
                        phase="idle",
                        last_status="short_transcript_ignored",
                        turn=turn,
                        realtime_voice_session=realtime_session,
                        last_transcript=transcript_for_cognitive,
                    )
                    self._sleep(self.empty_interval_s)
                    continue

                observation = self._replace_transcript(observation, transcript_for_cognitive)
                self._publish_state(
                    phase="thinking",
                    last_status="transcribed",
                    turn=turn,
                    conversation_active=True,
                    last_transcript=observation.text,
                    last_error="",
                )
                self._publish_state(
                    phase="thinking",
                    last_status="thinking",
                    turn=turn,
                    conversation_active=True,
                    last_transcript=observation.text,
                    last_error="",
                )
                actions, reply, think_s, stale_reason, emitted_reply_delta = self._run_cognition_turn(
                    observation,
                    turn,
                    realtime_session,
                )
                if stale_reason:
                    self._publish_stale_round(
                        turn,
                        reason=stale_reason,
                        last_transcript=observation.text,
                        realtime_voice_session=realtime_session,
                    )
                    self._sleep(self.empty_interval_s)
                    continue
                if not self._actions_allowed_for_turn(turn, actions, reply):
                    self._publish_stale_round(
                        turn,
                        reason="round_not_current_or_unstable",
                        last_transcript=observation.text,
                        realtime_voice_session=realtime_session,
                    )
                    self._sleep(self.empty_interval_s)
                    continue
                if reply and not emitted_reply_delta:
                    self._call_realtime(realtime_session, "append_reply_delta", reply, turn=turn)
                    self._publish_state(
                        phase="thinking",
                        last_status="reply_delta",
                        turn=turn,
                        realtime_voice_session=realtime_session,
                        conversation_active=True,
                        last_transcript=observation.text,
                        last_reply=reply,
                        last_reply_delta=reply,
                        last_error="",
                    )
                speak_started = time.perf_counter()
                if actions:
                    self._call_realtime(realtime_session, "start_speaking", turn=turn)
                    self._publish_state(
                        phase="speaking",
                        last_status="speaking_dispatch",
                        turn=turn,
                        realtime_voice_session=realtime_session,
                        conversation_active=True,
                        last_transcript=observation.text,
                        last_reply=reply,
                        last_error="",
                    )
                    if not self._is_current_turn(turn):
                        self._publish_stale_round(
                            turn,
                            reason="speaking_round_not_current",
                            last_transcript=observation.text,
                            realtime_voice_session=realtime_session,
                        )
                        self._sleep(self.empty_interval_s)
                        continue
                    outcomes = self.body_runtime.dispatch_actions(actions)
                    all_ok = bool(outcomes) and all(
                        getattr(outcome, "status", "") == "ok" for outcome in outcomes
                    )
                    status = "ok" if all_ok else "degraded"
                    self._rolling_chunk_count = min(self.max_chunk_count, chunk_count + 1)
                else:
                    outcomes = []
                    status = "no_reply"
                speak_s = time.perf_counter() - speak_started
                self._call_realtime(
                    realtime_session,
                    "complete",
                    status="reply_ready" if reply else status,
                    turn=turn,
                )
                turn_count = int(self.body_runtime.voice_dialogue_state.get("turn_count", 0) or 0) + 1
                total_s = time.perf_counter() - turn_started
                stage_latency_ms = self._stage_latency_ms(
                    listen_asr_s=listen_asr_s,
                    think_s=think_s,
                    speak_s=speak_s,
                    total_s=total_s,
                )
                self._publish_state(
                    phase="idle",
                    last_status="reply_ready" if reply else "no_reply",
                    turn=turn,
                    realtime_voice_session=realtime_session,
                    conversation_active=True,
                    last_transcript=observation.text,
                    last_reply=reply,
                    turn_count=turn_count,
                    last_error="",
                    last_latency_s={
                        "listen_asr": round(listen_asr_s, 2),
                        "think": round(think_s, 2),
                        "speak": round(speak_s, 2),
                        "total": round(total_s, 2),
                    },
                    last_stage_latency_ms=stage_latency_ms,
                    last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                    last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                    last_completed_turn={
                        "round_id": turn.round_id,
                        "cancellation_token": turn.cancellation_token,
                        "turn_count": turn_count,
                        "transcript": observation.text,
                        "reply": reply,
                        "status": status,
                        "latency_s": {
                            "listen_asr": round(listen_asr_s, 2),
                            "think": round(think_s, 2),
                            "speak": round(speak_s, 2),
                            "total": round(total_s, 2),
                        },
                        "stage_latency_ms": stage_latency_ms,
                        "bottleneck_stage": self._bottleneck_stage(stage_latency_ms),
                        "bottleneck_ms": self._bottleneck_ms(stage_latency_ms),
                        "completed_at_ts": time.time(),
                    },
                )
                self._sleep(self.idle_interval_s)
            except Exception as exc:  # pragma: no cover - runtime boundary
                session = self._realtime_session_for_turn()
                self._call_realtime(session, "fail", str(exc))
                payload = {
                    "phase": "error",
                    "last_status": "error",
                    "last_error": str(exc),
                }
                payload.update(self._realtime_updates(session))
                self.body_runtime.update_voice_dialogue_state(**payload)
                self._sleep(max(1.5, self.empty_interval_s))

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(max(0.0, seconds))

    @staticmethod
    def _stage_latency_ms(
        *,
        listen_asr_s: float,
        total_s: float,
        think_s: float = 0.0,
        speak_s: float = 0.0,
    ) -> dict[str, float]:
        stages = {
            "listen_asr": round(max(0.0, listen_asr_s) * 1000, 2),
            "think": round(max(0.0, think_s) * 1000, 2),
            "speak": round(max(0.0, speak_s) * 1000, 2),
            "total": round(max(0.0, total_s) * 1000, 2),
        }
        stages["overhead"] = round(
            max(0.0, stages["total"] - stages["listen_asr"] - stages["think"] - stages["speak"]),
            2,
        )
        return stages

    @classmethod
    def _bottleneck_stage(cls, stage_latency_ms: dict[str, float]) -> str:
        candidates = {
            key: value
            for key, value in stage_latency_ms.items()
            if key not in {"total", "overhead"}
        }
        if not candidates:
            return ""
        return max(candidates, key=candidates.get)

    @classmethod
    def _bottleneck_ms(cls, stage_latency_ms: dict[str, float]) -> float | None:
        stage = cls._bottleneck_stage(stage_latency_ms)
        if not stage:
            return None
        return float(stage_latency_ms.get(stage, 0.0))
