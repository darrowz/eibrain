"""Streaming ear capture helpers."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess

from eibrain.protocol.observations import AudioTranscriptFinal


@dataclass(slots=True)
class ArecordStreamCapture:
    device: str
    sample_rate: int
    channels: int

    def build_command(self) -> list[str]:
        return [
            "arecord",
            "-D",
            self.device,
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate),
            "-c",
            str(self.channels),
            "-t",
            "raw",
        ]

    def read_chunks(self, chunk_count: int, *, chunk_bytes: int = 4096) -> list[bytes]:
        command = self.build_command() + ["-d", str(max(1, chunk_count))]
        completed = subprocess.run(command, capture_output=True, check=False)
        payload = completed.stdout or b""
        if chunk_count <= 1:
            return [payload]
        return [payload[i : i + chunk_bytes] for i in range(0, len(payload), chunk_bytes)][:chunk_count]


@dataclass(slots=True)
class EarStreamProcessor:
    capture: object
    recognizer: object

    def transcribe_window(
        self,
        *,
        chunk_count: int,
        session_id: str,
        actor_id: str,
    ) -> AudioTranscriptFinal:
        chunks = list(self.capture.read_chunks(chunk_count))
        text = self.recognizer.transcribe(
            chunks,
            sample_rate=self.capture.sample_rate,
            channels=self.capture.channels,
        )
        return AudioTranscriptFinal(
            ts=1.0,
            source="ear.asr",
            text=text,
            session_id=session_id,
            actor_id=actor_id,
        )
