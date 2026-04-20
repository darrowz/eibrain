"""Ear organ implementation."""

from __future__ import annotations

import time

from eibrain.body.ear_stream import ArecordStreamCapture, pcm_signal_stats
from eibrain.body.organs.base import BaseOrgan
from eibrain.body.runtime_linux import transcribe_pcm_with_faster_whisper_subprocess
from eibrain.body.runtime_linux import transcribe_pcm_with_sherpa_subprocess
from eibrain.body.sherpa_streaming import SherpaOnnxStreamingRecognizer
from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth


class EarOrgan(BaseOrgan):
    name = "ear"
    subfunction_names = ("capture", "vad", "asr")

    def __init__(self, *, config=None) -> None:
        super().__init__(config=config)
        self._cache_ttl_s = self._read_float_config("capture", "refresh_interval_s", default=1.5)
        self._chunk_count = self._read_int_config("capture", "chunk_count", default=2)
        self._capture = self._build_capture()
        self._recognizer = self._build_recognizer()
        self._cached_heartbeat: OrganHealth | None = None
        self._cached_heartbeat_at = 0.0

    def heartbeat(self) -> OrganHealth:
        if not self._audio_runtime_enabled():
            return super().heartbeat()
        now_ts = time.time()
        if self._cached_heartbeat is not None and now_ts - self._cached_heartbeat_at < self._cache_ttl_s:
            return self._cached_heartbeat
        capture_state, chunks = self._capture_health(now_ts=now_ts)
        vad_state = self._vad_health(capture_state=capture_state, chunks=chunks, now_ts=now_ts)
        asr_state = self._asr_health(capture_state=capture_state, chunks=chunks, now_ts=now_ts)
        subfunctions = {
            "capture": capture_state,
            "vad": vad_state,
            "asr": asr_state,
        }
        statuses = [state.health for state in subfunctions.values()]
        if statuses and all(status == "healthy" for status in statuses):
            health = "healthy"
        elif any(status == "healthy" for status in statuses) or any(status == "degraded" for status in statuses):
            health = "degraded"
        else:
            health = "unavailable"
        self._cached_heartbeat = OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)
        self._cached_heartbeat_at = now_ts
        return self._cached_heartbeat

    def _audio_runtime_enabled(self) -> bool:
        return self._capture is not None and self._asr_provider() in {"sherpa_onnx", "faster_whisper"}

    def _build_capture(self) -> ArecordStreamCapture | None:
        capture_cfg = self.config.subfunctions.get("capture")
        if capture_cfg is None or capture_cfg.driver.kind == "noop":
            return None
        return ArecordStreamCapture(
            device=str(capture_cfg.driver.extra.get("device", "default")),
            sample_rate=int(capture_cfg.driver.extra.get("sample_rate", 16000)),
            channels=int(capture_cfg.driver.extra.get("channels", 1)),
        )

    def _build_recognizer(self) -> SherpaOnnxStreamingRecognizer | None:
        asr_cfg = self.config.subfunctions.get("asr")
        if asr_cfg is None or asr_cfg.driver.kind == "noop":
            return None
        if str(asr_cfg.driver.extra.get("provider", "sherpa_onnx")) != "sherpa_onnx":
            return None
        return SherpaOnnxStreamingRecognizer(
            model_dir=str(asr_cfg.driver.extra.get("model_dir", "")),
            model_type=str(asr_cfg.driver.extra.get("model_type", "") or "") or None,
        )

    def _capture_health(self, *, now_ts: float) -> tuple[SubfunctionHealth, list[bytes]]:
        if self._driver_kind("capture") == "noop":
            return self._subfunction_health("capture"), []
        probe = self.drivers["capture"].heartbeat()
        started = time.perf_counter()
        chunks: list[bytes] = []
        error = None
        if self._capture is not None:
            try:
                if hasattr(self._capture, "read_window"):
                    chunks = list(self._capture.read_window(self._chunk_count))
                else:
                    chunks = list(self._capture.read_chunks(self._chunk_count))
            except Exception as exc:  # pragma: no cover - hardware path
                error = str(exc)
        stats = pcm_signal_stats(chunks, channels=self._capture.channels if self._capture is not None else 1)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        details = self._merge_probe_details(
            probe=probe.details,
            elapsed_ms=elapsed_ms,
            status="healthy" if chunks else "capture_failed",
        )
        details.update(
            {
                "chunk_count": len(chunks),
                "requested_chunk_count": self._chunk_count,
                "sample_rate": self._capture.sample_rate if self._capture is not None else None,
                "channels": self._capture.channels if self._capture is not None else None,
                "capture_device": self._capture.device if self._capture is not None else None,
                "captured_at_ts": now_ts,
                "payload_bytes": sum(len(chunk) for chunk in chunks),
                **stats,
            }
        )
        if error:
            details["error"] = error
        if chunks:
            health = "healthy"
        else:
            health = "unavailable" if probe.status == "unavailable" else "degraded"
        return SubfunctionHealth(name="capture", health=health, details=details), chunks

    def _vad_health(
        self,
        *,
        capture_state: SubfunctionHealth,
        chunks: list[bytes],
        now_ts: float,
    ) -> SubfunctionHealth:
        if self._driver_kind("vad") == "noop":
            details = dict(self._subfunction_health("vad").details)
            details.update(
                {
                    "captured_at_ts": now_ts,
                    "voice_activity": bool(capture_state.details.get("voice_activity")),
                    "dbfs": capture_state.details.get("dbfs"),
                    "status": "observed" if chunks else "idle",
                    "speech_window_summary": self._summarize_audio(capture_state.details, transcript=""),
                }
            )
            health = "healthy" if chunks else "degraded"
            return SubfunctionHealth(name="vad", health=health, details=details)
        return self._subfunction_health("vad")

    def _asr_health(
        self,
        *,
        capture_state: SubfunctionHealth,
        chunks: list[bytes],
        now_ts: float,
    ) -> SubfunctionHealth:
        if self._driver_kind("asr") == "noop":
            return self._subfunction_health("asr")
        probe = self.drivers["asr"].heartbeat()
        started = time.perf_counter()
        transcript = ""
        error = None
        sample_count = int(capture_state.details.get("sample_count", 0) or 0)
        if (
            chunks
            and self._capture is not None
        ):
            try:
                provider = self._asr_provider()
                if provider == "sherpa_onnx" and self._recognizer is not None:
                    if isinstance(self._recognizer, SherpaOnnxStreamingRecognizer):
                        if sample_count >= int(self._recognizer.expected_sample_rate * 0.25):
                            result = transcribe_pcm_with_sherpa_subprocess(
                                pcm_bytes=b"".join(chunks),
                                model_dir=str(self._recognizer.model_dir),
                                model_type=self._recognizer.model_type,
                                sample_rate=self._capture.sample_rate,
                                channels=self._capture.channels,
                                chunk_bytes=max(1, len(chunks[0])) if chunks else 4096,
                            )
                            result_details = result.get("details", {})
                            if result.get("status") == "ok" and isinstance(result_details, dict):
                                transcript = str(result_details.get("text", "") or "")
                            elif isinstance(result_details, dict):
                                error = str(result_details.get("stderr") or result_details.get("reason") or "sherpa_subprocess_failed")
                    else:
                        transcript = self._recognizer.transcribe(
                            chunks,
                            sample_rate=self._capture.sample_rate,
                            channels=self._capture.channels,
                        )
                elif provider == "faster_whisper":
                    asr_cfg = self.config.subfunctions.get("asr")
                    asr_extra = asr_cfg.driver.extra if asr_cfg is not None else {}
                    result = transcribe_pcm_with_faster_whisper_subprocess(
                        pcm_bytes=b"".join(chunks),
                        model_name=str(asr_extra.get("model_name", "Systran/faster-whisper-tiny")),
                        sample_rate=self._capture.sample_rate,
                        channels=self._capture.channels,
                        language=str(asr_extra.get("language", "zh")),
                        compute_type=str(asr_extra.get("compute_type", "int8")),
                        beam_size=int(asr_extra.get("beam_size", 1)),
                        python_executable=str(asr_extra.get("python_executable", "/usr/bin/python3")),
                    )
                    result_details = result.get("details", {})
                    if result.get("status") == "ok" and isinstance(result_details, dict):
                        transcript = str(result_details.get("text", "") or "")
                    elif isinstance(result_details, dict):
                        error = str(result_details.get("stderr") or result_details.get("reason") or "faster_whisper_failed")
            except Exception as exc:  # pragma: no cover - hardware dependency
                error = str(exc)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        details = self._merge_probe_details(
            probe=probe.details,
            elapsed_ms=elapsed_ms,
            status="transcribed" if transcript else ("silence" if chunks else "capture_unavailable"),
        )
        details.update(
            {
                "captured_at_ts": now_ts,
                "transcript": transcript,
                "transcript_char_count": len(transcript),
                "voice_activity": capture_state.details.get("voice_activity"),
                "dbfs": capture_state.details.get("dbfs"),
                "sample_count": sample_count,
                "speech_window_summary": self._summarize_audio(capture_state.details, transcript=transcript),
            }
        )
        if error:
            details["error"] = error
        if (
            isinstance(self._recognizer, SherpaOnnxStreamingRecognizer)
            and sample_count < int(self._recognizer.expected_sample_rate * 0.25)
        ):
            details["status"] = "warming_up"
            details["speech_window_summary"] = "audio window too short for ASR decode"
        if transcript:
            health = "healthy"
        elif details.get("status") == "warming_up":
            health = "degraded"
        elif chunks:
            health = "degraded"
        else:
            health = "unavailable" if probe.status == "unavailable" else "degraded"
        return SubfunctionHealth(name="asr", health=health, details=details)

    def _asr_provider(self) -> str:
        config = self.config.subfunctions.get("asr")
        if config is None:
            return "disabled"
        return str(config.driver.extra.get("provider", "sherpa_onnx"))

    def _driver_kind(self, name: str) -> str:
        config = self.config.subfunctions.get(name)
        if config is None:
            return "noop"
        return str(config.driver.kind)

    def _read_float_config(self, subfunction_name: str, key: str, *, default: float) -> float:
        config = self.config.subfunctions.get(subfunction_name)
        if config is None:
            return default
        value = config.driver.extra.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _read_int_config(self, subfunction_name: str, key: str, *, default: int) -> int:
        config = self.config.subfunctions.get(subfunction_name)
        if config is None:
            return default
        value = config.driver.extra.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _merge_probe_details(*, probe: dict[str, object], elapsed_ms: float, status: str) -> dict[str, object]:
        merged = dict(probe)
        merged["driver"] = merged.get("driver", "command")
        merged["elapsed_ms"] = elapsed_ms
        merged["status"] = status
        nested = merged.get("details", {})
        if not isinstance(nested, dict):
            nested = {}
        merged["details"] = nested
        return merged

    @staticmethod
    def _summarize_audio(details: dict[str, object], *, transcript: str) -> str:
        dbfs = details.get("dbfs")
        voice_activity = details.get("voice_activity")
        if transcript:
            return f"heard speech at {dbfs} dBFS: {transcript}"
        if voice_activity:
            return f"voice activity detected at {dbfs} dBFS"
        return f"no clear speech activity ({dbfs} dBFS)"
