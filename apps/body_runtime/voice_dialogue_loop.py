"""Continuous honjia voice dialogue loop."""

from __future__ import annotations

import threading
import time

from apps.cognitive_runtime.app import CognitiveRuntimeApp
from apps.body_runtime.app import BodyRuntimeApp
from apps.body_runtime.engagement_state import EngagementStateWriter
from eibrain.protocol.actions import PlaySpeechAction


WAKE_WORD = "\u9e3f\u9014"
SLEEP_WORD = "\u7ed3\u675f\u5bf9\u8bdd"
WAKE_ACK_REPLY = "\u6211\u5728\u3002"
SLEEP_ACK_REPLY = "\u597d\u7684\uff0c\u5148\u4f11\u606f\u3002"
WAKE_STRIP_CHARS = " \t\r\n\uff0c,\u3002.!\uff01?\uff1f:\uff1a;\uff1b\u3001"



class VoiceDialogueLoop:
    def __init__(
        self,
        *,
        body_runtime: BodyRuntimeApp,
        cognitive_runtime: CognitiveRuntimeApp,
        chunk_count: int = 3,
        idle_interval_s: float = 0.5,
        empty_interval_s: float = 0.25,
        session_id: str = "voice-dialogue-loop",
        actor_id: str = "darrow",
        initial_conversation_active: bool = False,
        wake_word: str = WAKE_WORD,
        sleep_word: str = SLEEP_WORD,
        wake_ack_reply: str = WAKE_ACK_REPLY,
        sleep_ack_reply: str = SLEEP_ACK_REPLY,
        engagement_writer: EngagementStateWriter | None = None,
    ) -> None:
        self.body_runtime = body_runtime
        self.cognitive_runtime = cognitive_runtime
        self.chunk_count = chunk_count
        self.idle_interval_s = idle_interval_s
        self.empty_interval_s = empty_interval_s
        self.session_id = session_id
        self.actor_id = actor_id
        self.conversation_active = initial_conversation_active
        self.wake_word = wake_word
        self.sleep_word = sleep_word
        self.wake_ack_reply = wake_ack_reply
        self.sleep_ack_reply = sleep_ack_reply
        self.engagement_writer = engagement_writer
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.body_runtime.update_voice_dialogue_state(
            enabled=True,
            running=True,
            phase="starting",
            last_status="starting",
            conversation_active=self.conversation_active,
            wake_word=self.wake_word,
            sleep_word=self.sleep_word,
            last_error="",
        )
        self._write_engagement_state(phase="starting", reason="voice_loop_start")
        self._thread = threading.Thread(target=self._run, name="voice-dialogue-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None
        self.body_runtime.update_voice_dialogue_state(
            running=False,
            phase="stopped",
            last_status="stopped",
            conversation_active=self.conversation_active,
            wake_word=self.wake_word,
            sleep_word=self.sleep_word,
        )
        self._write_engagement_state(phase="stopped", reason="voice_loop_stop")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.body_runtime.is_speaking():
                    self.body_runtime.update_voice_dialogue_state(
                        phase="speaking",
                        last_status="playback_active",
                        conversation_active=self.conversation_active,
                    )
                    self._sleep(self.idle_interval_s)
                    continue
                turn_started = time.perf_counter()
                self.body_runtime.update_voice_dialogue_state(
                    phase="listening",
                    last_status="listening",
                    conversation_active=self.conversation_active,
                )
                listen_started = time.perf_counter()
                observation = self.body_runtime.transcribe_audio_window(
                    chunk_count=self.chunk_count,
                    session_id=self.session_id,
                    actor_id=self.actor_id,
                )
                listen_asr_s = time.perf_counter() - listen_started
                transcript = observation.text.strip()
                if len(transcript) == 1:
                    total_s = time.perf_counter() - turn_started
                    stage_latency_ms = self._stage_latency_ms(listen_asr_s=listen_asr_s, total_s=total_s)
                    self.body_runtime.update_voice_dialogue_state(
                        phase="idle",
                        last_status="short_transcript_ignored",
                        conversation_active=self.conversation_active,
                        last_transcript=transcript,
                        last_latency_s={
                            "listen_asr": round(listen_asr_s, 2),
                            "total": round(total_s, 2),
                        },
                        last_stage_latency_ms=stage_latency_ms,
                        last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                        last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                    )
                    self._sleep(self.empty_interval_s)
                    continue
                if not transcript:
                    total_s = time.perf_counter() - turn_started
                    stage_latency_ms = self._stage_latency_ms(listen_asr_s=listen_asr_s, total_s=total_s)
                    self.body_runtime.update_voice_dialogue_state(
                        phase="idle",
                        last_status="no_transcript",
                        conversation_active=self.conversation_active,
                        last_transcript="",
                        last_latency_s={
                            "listen_asr": round(listen_asr_s, 2),
                            "total": round(total_s, 2),
                        },
                        last_stage_latency_ms=stage_latency_ms,
                        last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                        last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                    )
                    self._sleep(self.empty_interval_s)
                    continue
                if self.conversation_active and self.sleep_word in transcript:
                    self.conversation_active = False
                    self._write_engagement_state(phase="idle", reason="sleep_word")
                    self._dispatch_short_reply(
                        reply=self.sleep_ack_reply,
                        phase="idle",
                        status="sleep_acknowledged",
                        transcript=transcript,
                        listen_asr_s=listen_asr_s,
                        turn_started=turn_started,
                    )
                    self._sleep(self.idle_interval_s)
                    continue
                if not self.conversation_active:
                    if self.wake_word not in transcript:
                        total_s = time.perf_counter() - turn_started
                        stage_latency_ms = self._stage_latency_ms(listen_asr_s=listen_asr_s, total_s=total_s)
                        self.body_runtime.update_voice_dialogue_state(
                            phase="idle",
                            last_status="waiting_for_wake_word",
                            conversation_active=False,
                            last_transcript=transcript,
                            last_latency_s={
                                "listen_asr": round(listen_asr_s, 2),
                                "total": round(total_s, 2),
                            },
                            last_stage_latency_ms=stage_latency_ms,
                            last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                            last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                        )
                        self._sleep(self.empty_interval_s)
                        continue
                    self.conversation_active = True
                    self._write_engagement_state(phase="idle", reason="wake_word")
                    transcript_after_wake = self._strip_wake_word(transcript)
                    if not transcript_after_wake:
                        self._dispatch_short_reply(
                            reply=self.wake_ack_reply,
                            phase="idle",
                            status="wake_acknowledged",
                            transcript=transcript,
                            listen_asr_s=listen_asr_s,
                            turn_started=turn_started,
                        )
                        self._sleep(self.idle_interval_s)
                        continue
                    observation = self._with_transcript(observation, transcript_after_wake)
                    transcript = transcript_after_wake
                self.body_runtime.update_voice_dialogue_state(
                    phase="thinking",
                    last_status="transcribed",
                    conversation_active=self.conversation_active,
                    last_transcript=transcript,
                    last_latency_s={"listen_asr": round(listen_asr_s, 2)},
                )
                think_started = time.perf_counter()
                actions = self.cognitive_runtime.handle_observation(observation)
                think_s = time.perf_counter() - think_started
                reply = next((str(getattr(action, "text", "") or "") for action in actions if getattr(action, "kind", "") == "play_speech_action"), "")
                self.body_runtime.update_voice_dialogue_state(
                    phase="speaking",
                    last_status="reply_ready" if reply else "no_reply",
                    conversation_active=self.conversation_active,
                    last_reply=reply,
                    last_latency_s={
                        "listen_asr": round(listen_asr_s, 2),
                        "think": round(think_s, 2),
                    },
                )
                speak_started = time.perf_counter()
                outcomes = self.body_runtime.dispatch_actions(actions)
                speak_s = time.perf_counter() - speak_started
                status = "ok" if outcomes and all(getattr(outcome, "status", "") == "ok" for outcome in outcomes) else "degraded"
                turn_count = int(self.body_runtime.voice_dialogue_state.get("turn_count", 0) or 0) + 1
                total_s = time.perf_counter() - turn_started
                stage_latency_ms = self._stage_latency_ms(
                    listen_asr_s=listen_asr_s,
                    think_s=think_s,
                    speak_s=speak_s,
                    total_s=total_s,
                )
                self.body_runtime.update_voice_dialogue_state(
                    phase="idle",
                    last_status=status,
                    conversation_active=self.conversation_active,
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
                        "turn_count": turn_count,
                        "transcript": transcript,
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
            except Exception as exc:  # pragma: no cover - hardware loop resilience
                self.body_runtime.update_voice_dialogue_state(
                    phase="error",
                    last_status="error",
                    conversation_active=self.conversation_active,
                    last_error=str(exc),
                )
                self._sleep(max(2.0, self.empty_interval_s))

    def _strip_wake_word(self, transcript: str) -> str:
        return transcript.replace(self.wake_word, "", 1).lstrip(WAKE_STRIP_CHARS).strip()

    def _with_transcript(self, observation, transcript: str):
        return type(observation)(
            ts=observation.ts,
            source=observation.source,
            session_id=observation.session_id,
            actor_id=observation.actor_id,
            target_id=observation.target_id,
            text=transcript,
            language=getattr(observation, "language", "und"),
        )

    def _dispatch_short_reply(
        self,
        *,
        reply: str,
        phase: str,
        status: str,
        transcript: str,
        listen_asr_s: float,
        turn_started: float,
    ) -> None:
        action = PlaySpeechAction(
            ts=time.time(),
            source="body.voice_dialogue_loop",
            session_id=self.session_id,
            actor_id=self.actor_id,
            text=reply,
        )
        speak_started = time.perf_counter()
        outcomes = self.body_runtime.dispatch_actions([action])
        speak_s = time.perf_counter() - speak_started
        dispatch_status = "ok" if outcomes and all(getattr(outcome, "status", "") == "ok" for outcome in outcomes) else "degraded"
        total_s = time.perf_counter() - turn_started
        stage_latency_ms = self._stage_latency_ms(
            listen_asr_s=listen_asr_s,
            speak_s=speak_s,
            total_s=total_s,
        )
        self.body_runtime.update_voice_dialogue_state(
            phase=phase,
            last_status=status if dispatch_status == "ok" else dispatch_status,
            conversation_active=self.conversation_active,
            wake_word=self.wake_word,
            sleep_word=self.sleep_word,
            last_transcript=transcript,
            last_reply=reply,
            last_error="",
            last_latency_s={
                "listen_asr": round(listen_asr_s, 2),
                "speak": round(speak_s, 2),
                "total": round(total_s, 2),
            },
            last_stage_latency_ms=stage_latency_ms,
            last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
            last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
        )

    def _write_engagement_state(self, *, phase: str, reason: str) -> None:
        if self.engagement_writer is None:
            return
        try:
            self.engagement_writer.write(
                conversation_active=self.conversation_active,
                phase=phase,
                reason=reason,
            )
        except OSError:
            return

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(max(0.0, seconds))

    @staticmethod
    def _stage_latency_ms(
        *,
        listen_asr_s: float,
        total_s: float,
        think_s: float | None = None,
        speak_s: float | None = None,
    ) -> dict[str, float]:
        stage_latency_ms: dict[str, float] = {
            "listen_asr": round(max(0.0, listen_asr_s) * 1000, 2),
            "total": round(max(0.0, total_s) * 1000, 2),
        }
        accounted_s = listen_asr_s
        if think_s is not None:
            stage_latency_ms["think"] = round(max(0.0, think_s) * 1000, 2)
            accounted_s += think_s
        if speak_s is not None:
            stage_latency_ms["speak"] = round(max(0.0, speak_s) * 1000, 2)
            accounted_s += speak_s
        overhead_s = total_s - accounted_s
        if overhead_s > 0.01:
            stage_latency_ms["overhead"] = round(overhead_s * 1000, 2)
        return stage_latency_ms

    @classmethod
    def _bottleneck_stage(cls, stage_latency_ms: dict[str, float]) -> str:
        candidates = {key: value for key, value in stage_latency_ms.items() if key != "total"}
        if not candidates:
            return ""
        return max(candidates, key=candidates.get)

    @classmethod
    def _bottleneck_ms(cls, stage_latency_ms: dict[str, float]) -> float | None:
        stage = cls._bottleneck_stage(stage_latency_ms)
        if not stage:
            return None
        return float(stage_latency_ms.get(stage, 0.0))
