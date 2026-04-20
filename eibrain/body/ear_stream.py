"""Streaming ear capture helpers."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
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

    def read_window(self, duration_s: int, *, chunk_bytes: int = 4096) -> list[bytes]:
        command = self.build_command() + ["-d", str(max(1, duration_s))]
        completed = subprocess.run(command, capture_output=True, check=False)
        payload = completed.stdout or b""
        if not payload:
            return []
        return [payload[i : i + chunk_bytes] for i in range(0, len(payload), chunk_bytes)]


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


def pcm_signal_stats(pcm_chunks: list[bytes], *, channels: int) -> dict[str, float | int | bool]:
    samples = array("h")
    for chunk in pcm_chunks:
        chunk_bytes = chunk[: len(chunk) - (len(chunk) % 2)]
        if chunk_bytes:
            samples.frombytes(chunk_bytes)
    if channels > 1 and samples:
        mono = array("h")
        for index in range(0, len(samples), channels):
            frame = samples[index : index + channels]
            if frame:
                mono.append(int(sum(frame) / len(frame)))
        samples = mono
    if not samples:
        return {
            "sample_count": 0,
            "peak_level": 0.0,
            "rms_level": 0.0,
            "dbfs": -120.0,
            "voice_activity": False,
        }
    peak = max(abs(sample) for sample in samples) / 32768.0
    rms = math.sqrt(sum(float(sample) * float(sample) for sample in samples) / len(samples)) / 32768.0
    dbfs = 20.0 * math.log10(max(rms, 1e-6))
    return {
        "sample_count": len(samples),
        "peak_level": round(peak, 6),
        "rms_level": round(rms, 6),
        "dbfs": round(dbfs, 2),
        "voice_activity": rms >= 0.015,
    }
