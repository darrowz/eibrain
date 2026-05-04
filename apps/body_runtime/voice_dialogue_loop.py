"""Continuous honjia voice dialogue loop."""

from __future__ import annotations

import re
import threading
import time

from apps.cognitive_runtime.app import CognitiveRuntimeApp
from apps.body_runtime.app import BodyRuntimeApp
from eibrain.protocol.actions import PlaySpeechAction
from eibrain.protocol.observations import AudioTranscriptFinal


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

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.body_runtime.update_voice_dialogue_state(
            enabled=True,
            running=True,
            phase="starting",
            last_status="starting",
            last_error="",
            wake_word=self.wake_word,
            sleep_word=self.sleep_word,
            conversation_active=self.conversation_active,
            last_reply="",
        )
        self._publish_engagement_state(phase="running" if self.conversation_active else "sleeping")
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

    def _publish_state(self, *, phase: str, last_status: str, **updates: object) -> None:
        payload = {
            "phase": phase,
            "last_status": last_status,
            "running": True,
            "enabled": True,
            "wake_word": self.wake_word,
            "sleep_word": self.sleep_word,
            "conversation_active": self.conversation_active,
        }
        payload.update(updates)
        payload.setdefault("last_reply", self.body_runtime.voice_dialogue_state.get("last_reply", ""))
        self.body_runtime.update_voice_dialogue_state(**payload)

    def _dispatch_ack_reply(self, text: str):
        actions = [
            PlaySpeechAction(
                ts=time.time(),
                source="voice_dialogue_loop",
                session_id=self.session_id,
                actor_id=self.actor_id,
                text=text,
            )
        ]
        return self.body_runtime.dispatch_actions(actions)

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
                self._publish_state(phase="listening", last_status="listening")
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
                        self._publish_state(
                            phase="idle",
                            last_status=status,
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
                            conversation_active=True,
                            last_transcript=self.wake_word,
                            last_reply=self.waking_phrase,
                        )
                        think_started = time.perf_counter()
                        self._dispatch_ack_reply(self.waking_phrase)
                        think_s = time.perf_counter() - think_started
                        total_s = time.perf_counter() - turn_started
                        stage_latency_ms = self._stage_latency_ms(
                            listen_asr_s=listen_asr_s,
                            think_s=think_s,
                            total_s=total_s,
                        )
                        self._publish_state(
                            phase="idle",
                            last_status="wake_acknowledged",
                            conversation_active=True,
                            last_reply=self.waking_phrase,
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
                    self._publish_state(
                        phase="idle",
                        last_status="no_transcript",
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
                    self._dispatch_ack_reply(self.sleeping_phrase)
                    think_s = time.perf_counter() - think_started
                    self._publish_state(
                        phase="idle",
                        last_status="sleep_acknowledged",
                        conversation_active=False,
                        last_transcript=self.sleep_word,
                        last_reply=self.sleeping_phrase,
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
                    self._publish_state(
                        phase="idle",
                        last_status="short_transcript_ignored",
                        last_transcript=transcript_for_cognitive,
                    )
                    self._sleep(self.empty_interval_s)
                    continue

                observation = self._replace_transcript(observation, transcript_for_cognitive)
                self._publish_state(
                    phase="thinking",
                    last_status="transcribed",
                    conversation_active=True,
                    last_transcript=observation.text,
                    last_error="",
                )
                think_started = time.perf_counter()
                actions = self.cognitive_runtime.handle_observation(observation)
                think_s = time.perf_counter() - think_started

                reply = ""
                for action in actions:
                    if getattr(action, "kind", "") == "play_speech_action":
                        reply = str(getattr(action, "text", "") or "")
                        break
                speak_started = time.perf_counter()
                if actions:
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
                self.body_runtime.update_voice_dialogue_state(
                    phase="error",
                    last_status="error",
                    last_error=str(exc),
                )
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
