"""Reusable faster-whisper recognizer for live audio windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import wave

from eibrain.body.runtime_linux import resolve_faster_whisper_model_path


@dataclass(slots=True)
class FasterWhisperRecognizer:
    model_name: str
    language: str = "zh"
    compute_type: str = "int8"
    beam_size: int = 1
    vad_filter: bool = False
    device: str = "cpu"
    _model: object | None = field(default=None, init=False)

    def transcribe(self, pcm_chunks: list[bytes], *, sample_rate: int, channels: int) -> str:
        if not pcm_chunks:
            return ""
        wav_path = self._write_wav(pcm_chunks=pcm_chunks, sample_rate=sample_rate, channels=channels)
        try:
            segments, _info = self._get_model().transcribe(
                str(wav_path),
                language=self.language or None,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
            )
            return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        finally:
            wav_path.unlink(missing_ok=True)

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # pragma: no cover - host dependency

            self._model = WhisperModel(
                resolve_faster_whisper_model_path(self.model_name),
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    @staticmethod
    def _write_wav(*, pcm_chunks: list[bytes], sample_rate: int, channels: int) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(pcm_chunks))
        return wav_path
