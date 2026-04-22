"""Continuous honjia voice dialogue loop."""

from __future__ import annotations

import threading
import time

from apps.cognitive_runtime.app import CognitiveRuntimeApp
from apps.body_runtime.app import BodyRuntimeApp
from eibrain.protocol.actions import PlaySpeechAction


class VoiceDialogueLoop:
    def __init__(
        self,
        *,
        body_runtime: BodyRuntimeApp,
        cognitive_runtime: CognitiveRuntimeApp,
        chunk_count: int = 3,
        idle_interval_s: float = 1.0,
        empty_interval_s: float = 1.5,
        no_transcript_feedback_interval_s: float = 12.0,
        session_id: str = "voice-dialogue-loop",
        actor_id: str = "darrow",
    ) -> None:
        self.body_runtime = body_runtime
        self.cognitive_runtime = cognitive_runtime
        self.chunk_count = chunk_count
        self.idle_interval_s = idle_interval_s
        self.empty_interval_s = empty_interval_s
        self.no_transcript_feedback_interval_s = no_transcript_feedback_interval_s
        self.session_id = session_id
        self.actor_id = actor_id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_no_transcript_feedback_at = 0.0

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
                if not transcript:
                    feedback = self._maybe_build_no_transcript_feedback()
                    if feedback is not None:
                        self.body_runtime.update_voice_dialogue_state(
                            phase="speaking",
                            last_status="heard_but_no_transcript",
                            last_reply=feedback.text,
                        )
                        self.body_runtime.dispatch_actions([feedback])
                    self.body_runtime.update_voice_dialogue_state(
                        phase="idle",
                        last_status="heard_but_no_transcript" if feedback is not None else "no_transcript",
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

    def _maybe_build_no_transcript_feedback(self) -> PlaySpeechAction | None:
        details = self._latest_audio_trace_details()
        if not details:
            return None
        heard_voice = bool(details.get("voice_activity")) or bool(details.get("asr_voice_activity"))
        if not heard_voice:
            return None
        now_ts = time.time()
        if now_ts - self._last_no_transcript_feedback_at < self.no_transcript_feedback_interval_s:
            return None
        self._last_no_transcript_feedback_at = now_ts
        return PlaySpeechAction(
            ts=now_ts,
            source="voice_dialogue.no_transcript_feedback",
            session_id=self.session_id,
            actor_id=self.actor_id,
            text="我听到了，但还没听清。请靠近一点，再说一遍。",
        )

    def _latest_audio_trace_details(self) -> dict[str, object]:
        recent_events = getattr(self.body_runtime, "recent_events", None)
        if not callable(recent_events):
            return {}
        for event in reversed(list(recent_events())):
            if not isinstance(event, dict) or event.get("kind") != "audio_transcript_final":
                continue
            details = event.get("details", {})
            if isinstance(details, dict):
                return details
        return {}
