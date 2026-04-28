"""Realtime voice session state for streaming dialogue."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


VOICE_PHASES = {
    "idle",
    "listening",
    "partial_asr",
    "thinking_stream",
    "speaking_stream",
    "barge_in",
    "completed",
    "error",
}


@dataclass(slots=True)
class RealtimeVoiceEvent:
    phase: str
    status: str
    transcript: str = ""
    reply_delta: str = ""
    detail: str = ""
    at_s: float = 0.0


@dataclass(slots=True)
class RealtimeVoiceSession:
    """Tracks one realtime voice turn without owning audio or network I/O."""

    session_id: str
    actor_id: str
    clock: object = time.perf_counter
    phase: str = "idle"
    status: str = "idle"
    transcript_partial: str = ""
    transcript_final: str = ""
    reply_text: str = ""
    interrupted: bool = False
    interrupt_reason: str = ""
    started_at_s: float | None = None
    phase_started_at_s: float | None = None
    first_audio_at_s: float | None = None
    first_partial_at_s: float | None = None
    final_asr_at_s: float | None = None
    first_reply_at_s: float | None = None
    first_speech_at_s: float | None = None
    completed_at_s: float | None = None
    events: list[RealtimeVoiceEvent] = field(default_factory=list)

    def start_listening(self) -> None:
        now_s = self._now()
        self.started_at_s = now_s
        self.first_audio_at_s = None
        self.first_partial_at_s = None
        self.final_asr_at_s = None
        self.first_reply_at_s = None
        self.first_speech_at_s = None
        self.completed_at_s = None
        self.transcript_partial = ""
        self.transcript_final = ""
        self.reply_text = ""
        self.interrupted = False
        self.interrupt_reason = ""
        self._transition("listening", "waiting_for_audio", at_s=now_s)

    def note_audio(self) -> None:
        now_s = self._now()
        if self.first_audio_at_s is None:
            self.first_audio_at_s = now_s
        self._record("listening", "audio_detected", at_s=now_s)

    def update_partial_transcript(self, text: str) -> None:
        now_s = self._now()
        if self.first_partial_at_s is None:
            self.first_partial_at_s = now_s
        self.transcript_partial = text.strip()
        self._transition("partial_asr", "partial_transcript", transcript=self.transcript_partial, at_s=now_s)

    def finalize_transcript(self, text: str) -> None:
        now_s = self._now()
        self.final_asr_at_s = now_s
        self.transcript_final = text.strip()
        self._transition("thinking_stream", "final_transcript", transcript=self.transcript_final, at_s=now_s)

    def append_reply_delta(self, delta: str) -> None:
        now_s = self._now()
        if self.first_reply_at_s is None:
            self.first_reply_at_s = now_s
        self.reply_text += delta
        self._transition("thinking_stream", "reply_delta", reply_delta=delta, at_s=now_s)

    def start_speaking(self) -> None:
        now_s = self._now()
        if self.first_speech_at_s is None:
            self.first_speech_at_s = now_s
        self._transition("speaking_stream", "speech_started", at_s=now_s)

    def interrupt(self, *, reason: str = "user_barge_in") -> None:
        now_s = self._now()
        self.interrupted = True
        self.interrupt_reason = reason
        self._transition("barge_in", "interrupted", detail=reason, at_s=now_s)

    def complete(self, *, status: str = "ok") -> None:
        now_s = self._now()
        self.completed_at_s = now_s
        self._transition("completed", status, at_s=now_s)

    def fail(self, error: str) -> None:
        now_s = self._now()
        self.completed_at_s = now_s
        self._transition("error", "error", detail=error, at_s=now_s)

    def snapshot(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "actor_id": self.actor_id,
            "phase": self.phase,
            "status": self.status,
            "transcript_partial": self.transcript_partial,
            "transcript_final": self.transcript_final,
            "reply_text": self.reply_text,
            "interrupted": self.interrupted,
            "interrupt_reason": self.interrupt_reason,
            "latency_ms": self.latency_ms(),
            "event_count": len(self.events),
        }

    def latency_ms(self) -> dict[str, float]:
        started = self.started_at_s
        if started is None:
            return {}
        result: dict[str, float] = {}
        if self.first_audio_at_s is not None:
            result["audio_detect"] = self._elapsed_ms(started, self.first_audio_at_s)
        if self.first_partial_at_s is not None:
            result["first_partial_asr"] = self._elapsed_ms(started, self.first_partial_at_s)
        if self.final_asr_at_s is not None:
            result["final_asr"] = self._elapsed_ms(started, self.final_asr_at_s)
        if self.first_reply_at_s is not None:
            result["first_reply_token"] = self._elapsed_ms(started, self.first_reply_at_s)
        if self.first_speech_at_s is not None:
            result["first_speech"] = self._elapsed_ms(started, self.first_speech_at_s)
        if self.completed_at_s is not None:
            result["total"] = self._elapsed_ms(started, self.completed_at_s)
        return result

    def _transition(
        self,
        phase: str,
        status: str,
        *,
        transcript: str = "",
        reply_delta: str = "",
        detail: str = "",
        at_s: float,
    ) -> None:
        if phase not in VOICE_PHASES:
            raise ValueError(f"unknown voice phase: {phase}")
        self.phase = phase
        self.status = status
        self.phase_started_at_s = at_s
        self._record(phase, status, transcript=transcript, reply_delta=reply_delta, detail=detail, at_s=at_s)

    def _record(
        self,
        phase: str,
        status: str,
        *,
        transcript: str = "",
        reply_delta: str = "",
        detail: str = "",
        at_s: float,
    ) -> None:
        self.events.append(
            RealtimeVoiceEvent(
                phase=phase,
                status=status,
                transcript=transcript,
                reply_delta=reply_delta,
                detail=detail,
                at_s=at_s,
            )
        )

    def _now(self) -> float:
        return float(self.clock())

    @staticmethod
    def _elapsed_ms(start_s: float, end_s: float) -> float:
        return round(max(0.0, end_s - start_s) * 1000, 2)
