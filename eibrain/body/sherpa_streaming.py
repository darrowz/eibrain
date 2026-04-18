"""Sherpa-ONNX streaming recognizer helpers."""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover - optional dependency in lightweight envs
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in lightweight envs
    np = None


def _pcm_to_float_samples(pcm_bytes: bytes, *, channels: int) -> list[float]:
    samples = array("h")
    samples.frombytes(pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % 2)])
    if channels <= 1:
        mono = samples
    else:
        mono = array("h")
        for idx in range(0, len(samples), channels):
            frame = samples[idx : idx + channels]
            if not frame:
                continue
            mono.append(int(sum(frame) / len(frame)))
    return [sample / 32768.0 for sample in mono]


def _to_waveform_buffer(samples: list[float]):
    if np is not None:
        return np.asarray(samples, dtype=np.float32)
    return array("f", samples)


@dataclass(slots=True)
class SherpaOnnxStreamingRecognizer:
    model_dir: str | Path
    expected_sample_rate: int = 16000
    model_type: str | None = None
    recognizer_factory: Callable[[int], Any] | None = None
    _recognizer_cache: dict[int, Any] = field(default_factory=dict, init=False)

    def transcribe(self, pcm_chunks: list[bytes], *, sample_rate: int, channels: int) -> str:
        recognizer = self._get_recognizer(self.expected_sample_rate)
        stream = recognizer.create_stream()
        for chunk in pcm_chunks:
            waveform = self._normalize_waveform(
                _pcm_to_float_samples(chunk, channels=channels),
                input_sample_rate=sample_rate,
            )
            if not waveform:
                continue
            stream.accept_waveform(
                sample_rate=self.expected_sample_rate,
                waveform=_to_waveform_buffer(waveform),
            )
            while hasattr(recognizer, "is_ready") and recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
        if hasattr(stream, "input_finished"):
            stream.input_finished()
        if hasattr(recognizer, "decode_stream"):
            recognizer.decode_stream(stream)
        result = getattr(stream, "result", None)
        return str(getattr(result, "text", "") or "").strip()

    def _normalize_waveform(self, samples: list[float], *, input_sample_rate: int) -> list[float]:
        if input_sample_rate == self.expected_sample_rate or not samples:
            return samples
        if input_sample_rate % self.expected_sample_rate == 0:
            step = max(1, input_sample_rate // self.expected_sample_rate)
            return samples[::step]
        ratio = input_sample_rate / self.expected_sample_rate
        output_length = max(1, int(len(samples) / ratio))
        return [samples[min(len(samples) - 1, int(round(index * ratio)))] for index in range(output_length)]

    def _get_recognizer(self, sample_rate: int):
        cached = self._recognizer_cache.get(sample_rate)
        if cached is not None:
            return cached
        recognizer = self._build_recognizer(sample_rate)
        self._recognizer_cache[sample_rate] = recognizer
        return recognizer

    def _build_recognizer(self, sample_rate: int):
        if self.recognizer_factory is not None:
            return self.recognizer_factory(sample_rate)
        import sherpa_onnx  # pragma: no cover - host dependency

        model_dir = Path(self.model_dir).expanduser()
        tokens = str(model_dir / "tokens.txt")
        encoder = str(model_dir / "encoder.onnx")
        decoder = str(model_dir / "decoder.onnx")
        joiner = str(model_dir / "joiner.onnx")
        model_type = self.model_type or self._infer_model_type(model_dir)

        if model_type in {"lstm", "conformer", "zipformer", "zipformer2"} and hasattr(sherpa_onnx, "OnlineRecognizer") and hasattr(sherpa_onnx.OnlineRecognizer, "from_transducer"):
            return sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                sample_rate=self.expected_sample_rate,
                model_type=model_type,
            )

        if hasattr(sherpa_onnx, "OfflineRecognizer") and hasattr(sherpa_onnx.OfflineRecognizer, "from_transducer"):
            return sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                sample_rate=self.expected_sample_rate,
                model_type=model_type,
            )

        if hasattr(sherpa_onnx, "OnlineRecognizer") and hasattr(sherpa_onnx.OnlineRecognizer, "from_transducer"):
            return sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                sample_rate=self.expected_sample_rate,
                model_type=model_type,
            )

        if not all(
            hasattr(sherpa_onnx, attr)
            for attr in (
                "OnlineRecognizer",
                "OnlineRecognizerConfig",
                "OnlineModelConfig",
                "OnlineTransducerModelConfig",
                "FeatureConfig",
            )
        ):
            raise RuntimeError("unsupported sherpa_onnx streaming API")

        config = sherpa_onnx.OnlineRecognizerConfig(
            feat_config=sherpa_onnx.FeatureConfig(sample_rate=sample_rate),
            model=sherpa_onnx.OnlineModelConfig(
                transducer=sherpa_onnx.OnlineTransducerModelConfig(
                    encoder=encoder,
                    decoder=decoder,
                    joiner=joiner,
                ),
                tokens=tokens,
            ),
        )
        return sherpa_onnx.OnlineRecognizer(config)

    @staticmethod
    def _infer_model_type(model_dir: Path) -> str:
        name = model_dir.name.lower()
        if "lstm" in name:
            return "lstm"
        if "zipformer2" in name:
            return "zipformer2"
        if "zipformer" in name:
            return "zipformer"
        if "conformer" in name:
            return "conformer"
        return "transducer"
