"""Continuous honjia voice dialogue loop."""

from __future__ import annotations

import threading
import time

from apps.cognitive_runtime.app import CognitiveRuntimeApp
from apps.body_runtime.app import BodyRuntimeApp


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
    ) -> None:
        self.body_runtime = body_runtime
        self.cognitive_runtime = cognitive_runtime
        self.chunk_count = chunk_count
        self.idle_interval_s = idle_interval_s
        self.empty_interval_s = empty_interval_s
        self.session_id = session_id
        self.actor_id = actor_id
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
        )
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
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.body_runtime.is_speaking():
                    self.body_runtime.update_voice_dialogue_state(phase="speaking", last_status="playback_active")
                    self._sleep(self.idle_interval_s)
                    continue
                turn_started = time.perf_counter()
                self.body_runtime.update_voice_dialogue_state(phase="listening", last_status="listening")
                listen_started = time.perf_counter()
                observation = self.body_runtime.transcribe_audio_window(
                    chunk_count=self.chunk_count,
                    session_id=self.session_id,
                    actor_id=self.actor_id,
                )
                listen_asr_s = time.perf_counter() - listen_started
                transcript = observation.text.strip()
                if len(transcript) == 1:
                    self.body_runtime.update_voice_dialogue_state(
                        phase="idle",
                        last_status="short_transcript_ignored",
                        last_transcript=transcript,
                        last_latency_s={
                            "listen_asr": round(listen_asr_s, 2),
                            "total": round(time.perf_counter() - turn_started, 2),
                        },
                    )
                    self._sleep(self.empty_interval_s)
                    continue
                if not transcript:
                    self.body_runtime.update_voice_dialogue_state(
                        phase="idle",
                        last_status="no_transcript",
                        last_transcript="",
                        last_latency_s={
                            "listen_asr": round(listen_asr_s, 2),
                            "total": round(time.perf_counter() - turn_started, 2),
                        },
                    )
                    self._sleep(self.empty_interval_s)
                    continue
                self.body_runtime.update_voice_dialogue_state(
                    phase="thinking",
                    last_status="transcribed",
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
                self.body_runtime.update_voice_dialogue_state(
                    phase="idle",
                    last_status=status,
                    turn_count=turn_count,
                    last_error="",
                    last_latency_s={
                        "listen_asr": round(listen_asr_s, 2),
                        "think": round(think_s, 2),
                        "speak": round(speak_s, 2),
                        "total": round(time.perf_counter() - turn_started, 2),
                    },
                    last_completed_turn={
                        "turn_count": turn_count,
                        "transcript": transcript,
                        "reply": reply,
                        "status": status,
                        "latency_s": {
                            "listen_asr": round(listen_asr_s, 2),
                            "think": round(think_s, 2),
                            "speak": round(speak_s, 2),
                            "total": round(time.perf_counter() - turn_started, 2),
                        },
                        "completed_at_ts": time.time(),
                    },
                )
                self._sleep(self.idle_interval_s)
            except Exception as exc:  # pragma: no cover - hardware loop resilience
                self.body_runtime.update_voice_dialogue_state(
                    phase="error",
                    last_status="error",
                    last_error=str(exc),
                )
                self._sleep(max(2.0, self.empty_interval_s))

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(max(0.0, seconds))
