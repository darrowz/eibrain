"""Streaming ear capture helpers."""

from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass, field
import math
import subprocess
import time

from eibrain.protocol.observations import AudioTranscriptFinal


@dataclass(slots=True)
class ArecordStreamCapture:
    device: str
    sample_rate: int
    channels: int
    retry_count: int = 2
    retry_delay_s: float = 1.0
    lock_path: str = "/tmp/eibrain-arecord.lock"
    lock_timeout_s: float = 8.0
    streaming_vad: bool = False
    vad_frame_ms: int = 80
    vad_rms_threshold: float = 0.028
    vad_min_voice_ms: int = 160
    vad_end_silence_ms: int = 360
    vad_pre_roll_ms: int = 240
    last_returncode: int | None = None
    last_stderr: str = ""
    last_stdout_bytes: int = 0
    last_command: list[str] = field(default_factory=list)
    last_vad_triggered: bool = False
    last_vad_frame_count: int = 0
    last_vad_voice_frame_count: int = 0
    last_vad_elapsed_ms: float = 0.0

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
        payload = self._run_arecord(command)
        if chunk_count <= 1:
            return [payload]
        return [payload[i : i + chunk_bytes] for i in range(0, len(payload), chunk_bytes)][:chunk_count]

    def read_window(self, duration_s: int, *, chunk_bytes: int = 4096) -> list[bytes]:
        if self.streaming_vad:
            return self.read_voice_window(duration_s, chunk_bytes=chunk_bytes)
        command = self.build_command() + ["-d", str(max(1, duration_s))]
        payload = self._run_arecord(command)
        if not payload:
            return []
        return [payload[i : i + chunk_bytes] for i in range(0, len(payload), chunk_bytes)]

    def read_voice_window(self, max_duration_s: int, *, chunk_bytes: int = 4096) -> list[bytes]:
        command = self.build_command()
        frame_bytes = self._frame_bytes()
        max_frames = max(1, math.ceil(max_duration_s * 1000 / max(1, self.vad_frame_ms)))
        pre_roll_frames = max(1, math.ceil(self.vad_pre_roll_ms / max(1, self.vad_frame_ms)))
        min_voice_frames = max(1, math.ceil(self.vad_min_voice_ms / max(1, self.vad_frame_ms)))
        end_silence_frames = max(1, math.ceil(self.vad_end_silence_ms / max(1, self.vad_frame_ms)))
        all_frames: list[bytes] = []
        captured_frames: list[bytes] = []
        pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        triggered = False
        voice_frames = 0
        silence_after_voice = 0
        started = time.perf_counter()
        self.last_command = list(command)
        self.last_returncode = None
        self.last_stderr = ""
        self.last_stdout_bytes = 0
        try:
            with self._capture_lock():
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    if process.stdout is None:
                        return []
                    for _ in range(max_frames):
                        frame = process.stdout.read(frame_bytes)
                        if not frame:
                            break
                        all_frames.append(frame)
                        stats = pcm_signal_stats([frame], channels=self.channels)
                        is_voice = bool(stats["rms_level"] >= self.vad_rms_threshold)
                        if not triggered:
                            pre_roll.append(frame)
                            if not is_voice:
                                continue
                            triggered = True
                            captured_frames.extend(pre_roll)
                            voice_frames = 1
                            silence_after_voice = 0
                            continue
                        captured_frames.append(frame)
                        if is_voice:
                            voice_frames += 1
                            silence_after_voice = 0
                        else:
                            silence_after_voice += 1
                        if voice_frames >= min_voice_frames and silence_after_voice >= end_silence_frames:
                            break
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    stderr = process.stderr.read() if process.stderr is not None else b""
                    self.last_returncode = process.returncode
                    self.last_stderr = stderr.decode("utf-8", errors="replace").strip()
        except TimeoutError as exc:
            self.last_returncode = None
            self.last_stderr = str(exc)
            return []
        payload = b"".join(captured_frames if triggered else all_frames)
        self.last_stdout_bytes = len(payload)
        self.last_vad_triggered = triggered
        self.last_vad_frame_count = len(all_frames)
        self.last_vad_voice_frame_count = voice_frames
        self.last_vad_elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        if not payload:
            return []
        return [payload[i : i + chunk_bytes] for i in range(0, len(payload), chunk_bytes)]

    def _run_arecord(self, command: list[str]) -> bytes:
        self.last_command = list(command)
        attempts = max(1, self.retry_count + 1)
        payload = b""
        try:
            with self._capture_lock():
                for attempt in range(attempts):
                    completed = subprocess.run(command, capture_output=True, check=False)
                    payload = completed.stdout or b""
                    self._record_result(completed=completed, payload=payload)
                    if completed.returncode == 0 and payload:
                        return payload
                    if attempt + 1 < attempts:
                        time.sleep(max(0.0, self.retry_delay_s))
                return payload
        except TimeoutError as exc:
            self.last_returncode = None
            self.last_stderr = str(exc)
            self.last_stdout_bytes = 0
            return b""

    def _record_result(self, *, completed: subprocess.CompletedProcess[bytes], payload: bytes) -> None:
        self.last_returncode = completed.returncode
        stderr = completed.stderr or b""
        self.last_stderr = stderr.decode("utf-8", errors="replace").strip()
        self.last_stdout_bytes = len(payload)

    def _frame_bytes(self) -> int:
        bytes_per_sample = 2
        frame_bytes = int(self.sample_rate * self.channels * bytes_per_sample * max(1, self.vad_frame_ms) / 1000)
        alignment = max(1, self.channels * bytes_per_sample)
        return max(alignment, frame_bytes - (frame_bytes % alignment))

    def _capture_lock(self):
        class _NoopLock:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-Linux developer machines
            return _NoopLock()

        capture = self

        class _FileLock:
            def __init__(self) -> None:
                self._handle = None

            def __enter__(self):
                started = time.monotonic()
                self._handle = open(capture.lock_path, "a+", encoding="utf-8")
                while True:
                    try:
                        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return self
                    except BlockingIOError:
                        if time.monotonic() - started >= capture.lock_timeout_s:
                            raise TimeoutError(f"timed out waiting for audio capture lock: {capture.lock_path}")
                        time.sleep(0.1)

            def __exit__(self, exc_type, exc, traceback):
                if self._handle is not None:
                    try:
                        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        self._handle.close()
                return False

        return _FileLock()


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
