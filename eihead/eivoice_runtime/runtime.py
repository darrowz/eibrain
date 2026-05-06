from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from eibrain.protocol.joyinside_voice import audio_chunk

from .core import AudioFrame, EiVoiceRuntimeCore, OpusCodec
from .transport import VoiceStreamTransport


@dataclass(frozen=True)
class AcousticFrontendConfig:
    """Configurable placeholder for the acoustic front-end contract."""

    aec_enabled: bool = False
    aec_available: bool = False
    ns_enabled: bool = False
    ns_available: bool = False
    vad_enabled: bool = False
    vad_available: bool = False
    loopback_enabled: bool = False
    loopback_available: bool = False
    capture_enabled: bool = True
    capture_available: bool = True
    mode: str = "noop"
    warnings: tuple[str, ...] = ()

    def diagnostics(
        self,
        *,
        processed_frames: int = 0,
        dropped_frames: int = 0,
        last_frame_duration_ms: int | None = None,
    ) -> dict[str, Any]:
        warnings = list(self.warnings)
        warnings.extend(_frontend_component_warnings(self))
        return {
            "mode": self.mode,
            "capture": _frontend_component(
                enabled=self.capture_enabled,
                available=self.capture_available,
            ),
            "aec": _frontend_component(
                enabled=self.aec_enabled,
                available=self.aec_available,
            ),
            "ns": _frontend_component(
                enabled=self.ns_enabled,
                available=self.ns_available,
            ),
            "vad": _frontend_component(
                enabled=self.vad_enabled,
                available=self.vad_available,
            ),
            "loopback": _frontend_component(
                enabled=self.loopback_enabled,
                available=self.loopback_available,
            ),
            "healthy": self._healthy,
            "processed_frames": processed_frames,
            "dropped_frames": dropped_frames,
            "last_frame_duration_ms": last_frame_duration_ms,
            "warnings": list(dict.fromkeys(warnings)),
        }

    @property
    def _healthy(self) -> bool:
        return all(
            (
                self.capture_available if self.capture_enabled else True,
                self.aec_available if self.aec_enabled else True,
                self.ns_available if self.ns_enabled else True,
                self.vad_available if self.vad_enabled else True,
                self.loopback_available if self.loopback_enabled else True,
            )
        )


class NoOpAcousticFrontend:
    """Pass-through acoustic front-end with explicit AEC/NS/VAD diagnostics."""

    def __init__(self, config: AcousticFrontendConfig | None = None) -> None:
        self.config = config or AcousticFrontendConfig()
        self.processed_frames = 0
        self.dropped_frames = 0
        self.last_frame_duration_ms: int | None = None

    def process_capture(self, frame: AudioFrame) -> AudioFrame | None:
        self.processed_frames += 1
        self.last_frame_duration_ms = frame.duration_ms
        return frame

    def readiness(self) -> dict[str, Any]:
        return self.config.diagnostics(
            processed_frames=self.processed_frames,
            dropped_frames=self.dropped_frames,
            last_frame_duration_ms=self.last_frame_duration_ms,
        )


class AudioCaptureSource(Protocol):
    def read_frame(self) -> AudioFrame | None:
        ...


class AcousticFrontend(Protocol):
    def process_capture(self, frame: AudioFrame) -> AudioFrame | None:
        ...

    def readiness(self) -> dict[str, Any]:
        ...


class WsReceiveSource(Protocol):
    def read_frame(self) -> AudioFrame | None:
        ...


class PlaybackSink(Protocol):
    def play(self, frame: AudioFrame) -> None:
        ...

    def stop(self) -> None:
        ...


@dataclass
class RuntimeWorkerMetrics:
    step_once_calls: int = 0
    idle_steps: int = 0
    capture_frames: int = 0
    capture_empty_polls: int = 0
    capture_queue_full: int = 0
    encode_frames: int = 0
    encode_empty_polls: int = 0
    send_frames: int = 0
    send_empty_polls: int = 0
    send_rejected: int = 0
    receive_frames: int = 0
    receive_empty_polls: int = 0
    receive_queue_full: int = 0
    decode_frames: int = 0
    decode_empty_polls: int = 0
    decode_queue_full: int = 0
    playback_frames: int = 0
    playback_empty_polls: int = 0
    playback_interrupts: int = 0
    playback_frames_cleared: int = 0
    decode_frames_cleared: int = 0
    transport_inbound_events_cleared: int = 0
    last_capture_frame_duration_ms: int | None = None
    last_encode_frame_duration_ms: int | None = None
    last_send_frame_duration_ms: int | None = None
    last_receive_frame_duration_ms: int | None = None
    last_decode_frame_duration_ms: int | None = None
    last_playback_frame_duration_ms: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "step_once_calls": self.step_once_calls,
            "idle_steps": self.idle_steps,
            "capture_frames": self.capture_frames,
            "capture_empty_polls": self.capture_empty_polls,
            "capture_queue_full": self.capture_queue_full,
            "encode_frames": self.encode_frames,
            "encode_empty_polls": self.encode_empty_polls,
            "send_frames": self.send_frames,
            "send_empty_polls": self.send_empty_polls,
            "send_rejected": self.send_rejected,
            "receive_frames": self.receive_frames,
            "receive_empty_polls": self.receive_empty_polls,
            "receive_queue_full": self.receive_queue_full,
            "decode_frames": self.decode_frames,
            "decode_empty_polls": self.decode_empty_polls,
            "decode_queue_full": self.decode_queue_full,
            "playback_frames": self.playback_frames,
            "playback_empty_polls": self.playback_empty_polls,
            "playback_interrupts": self.playback_interrupts,
            "playback_frames_cleared": self.playback_frames_cleared,
            "decode_frames_cleared": self.decode_frames_cleared,
            "transport_inbound_events_cleared": self.transport_inbound_events_cleared,
            "last_capture_frame_duration_ms": self.last_capture_frame_duration_ms,
            "last_encode_frame_duration_ms": self.last_encode_frame_duration_ms,
            "last_send_frame_duration_ms": self.last_send_frame_duration_ms,
            "last_receive_frame_duration_ms": self.last_receive_frame_duration_ms,
            "last_decode_frame_duration_ms": self.last_decode_frame_duration_ms,
            "last_playback_frame_duration_ms": self.last_playback_frame_duration_ms,
        }


@dataclass
class EiVoiceRuntimeRunner:
    capture_source: AudioCaptureSource
    audio_frontend: AcousticFrontend
    playback_sink: PlaybackSink
    ws_receive_source: WsReceiveSource | None = None
    transport: VoiceStreamTransport | None = None
    codec: OpusCodec | None = None
    uid: str | None = None
    mid: str | None = None
    core: EiVoiceRuntimeCore = field(init=False)
    worker_metrics: RuntimeWorkerMetrics = field(default_factory=RuntimeWorkerMetrics, init=False)

    def __post_init__(self) -> None:
        self.core = EiVoiceRuntimeCore(codec=self.codec)

    def step_once(self) -> dict[str, bool]:
        self.worker_metrics.step_once_calls += 1
        result = {
            "capture": self.step_capture(),
            "encode": self.step_encode(),
        }
        if self.transport is not None:
            result["send"] = self.step_send()
        result["receive"] = self.step_receive()
        result["decode"] = self.step_decode()
        result["playback"] = self.step_playback()
        if not any(result.values()):
            self.worker_metrics.idle_steps += 1
        return result

    def step_capture(self) -> bool:
        frame = self.capture_source.read_frame()
        if frame is None:
            self.worker_metrics.capture_empty_polls += 1
            return False
        processed = self.audio_frontend.process_capture(frame)
        if processed is None:
            return False
        pushed = self.core.opus_encode_queue.push(processed, block=False)
        if not pushed:
            self.worker_metrics.capture_queue_full += 1
            return False
        self.worker_metrics.capture_frames += 1
        self.worker_metrics.last_capture_frame_duration_ms = processed.duration_ms
        return True

    def step_encode(self) -> bool:
        frame = self.core.opus_encode_queue.pop()
        if frame is None:
            self.worker_metrics.encode_empty_polls += 1
            return False
        encoded = self.core.codec.encode(frame)
        self.core.ws_send_queue.push(encoded, block=False)
        self.worker_metrics.encode_frames += 1
        self.worker_metrics.last_encode_frame_duration_ms = encoded.duration_ms
        return True

    def step_send(self) -> bool:
        if self.transport is None:
            self.worker_metrics.send_empty_polls += 1
            return False
        frame = self.core.ws_send_queue.pop()
        if frame is None:
            self.worker_metrics.send_empty_polls += 1
            return False
        event = audio_chunk(
            uid=self.uid,
            mid=self.mid,
            index=frame.sequence,
            audio_base64=base64.b64encode(frame.payload or frame.pcm).decode("ascii"),
        ).to_dict()
        content = event.setdefault("content", {})
        content["durationMs"] = frame.duration_ms
        content["sampleRateHz"] = frame.sample_rate_hz
        content["channels"] = frame.channels
        accepted = self.transport.send_event(event)
        if not accepted:
            self.worker_metrics.send_rejected += 1
            return False
        self.worker_metrics.send_frames += 1
        self.worker_metrics.last_send_frame_duration_ms = frame.duration_ms
        return True

    def step_receive(self) -> bool:
        if self.ws_receive_source is None and self.transport is None:
            self.worker_metrics.receive_empty_polls += 1
            return False
        frame = self._read_receive_frame()
        if frame is None:
            self.worker_metrics.receive_empty_polls += 1
            return False
        pushed = self.core.opus_decode_queue.push(frame, block=False)
        if not pushed:
            self.worker_metrics.receive_queue_full += 1
            return False
        self.worker_metrics.receive_frames += 1
        self.worker_metrics.last_receive_frame_duration_ms = frame.duration_ms
        return True

    def step_decode(self) -> bool:
        frame = self.core.opus_decode_queue.pop()
        if frame is None:
            self.worker_metrics.decode_empty_polls += 1
            return False
        decoded = self.core.codec.decode(frame)
        pushed = self.core.audio_playback_queue.push(decoded, block=False)
        if not pushed:
            self.worker_metrics.decode_queue_full += 1
            return False
        self.worker_metrics.decode_frames += 1
        self.worker_metrics.last_decode_frame_duration_ms = decoded.duration_ms
        return True

    def step_playback(self) -> bool:
        frame = self.core.audio_playback_queue.pop()
        if frame is None:
            self.worker_metrics.playback_empty_polls += 1
            return False
        self.playback_sink.play(frame)
        self.worker_metrics.playback_frames += 1
        self.worker_metrics.last_playback_frame_duration_ms = frame.duration_ms
        return True

    def interrupt_playback(self) -> int:
        self.playback_sink.stop()
        playback_cleared = _drain_queue(self.core.audio_playback_queue)
        decode_cleared = _drain_queue(self.core.opus_decode_queue)
        transport_cleared = 0
        if self.transport is not None:
            transport_cleared = len(self.transport.drain_inbound_events())
        self.worker_metrics.playback_interrupts += 1
        self.worker_metrics.playback_frames_cleared += playback_cleared
        self.worker_metrics.decode_frames_cleared += decode_cleared
        self.worker_metrics.transport_inbound_events_cleared += transport_cleared
        return playback_cleared + decode_cleared + transport_cleared

    def status(self) -> dict[str, Any]:
        status = dict(self.core.status())
        audio_frontend = self._audio_frontend_readiness()
        transport_status: dict[str, Any] = {}
        status["worker_metrics"] = self.worker_metrics.to_dict()
        status["audio_frontend"] = audio_frontend
        if self.transport is not None:
            transport_status = self.transport.status()
            status["transport"] = transport_status
        status["diagnostics"] = self._diagnostics(
            queues=status["queues"],
            audio_frontend=audio_frontend,
            transport_status=transport_status,
        )
        return status

    def _diagnostics(
        self,
        *,
        queues: Mapping[str, Any],
        audio_frontend: Mapping[str, Any],
        transport_status: Mapping[str, Any],
    ) -> dict[str, Any]:
        ws_send_queue = _mapping(queues.get("ws_send_queue"))
        opus_decode_queue = _mapping(queues.get("opus_decode_queue"))
        playback_queue = _mapping(queues.get("audio_playback_queue"))
        transport_state = _transport_state(transport_status)
        heartbeat = _heartbeat_status(transport_status)
        reconnect = _reconnect_status(transport_status)
        return {
            "schema": "eihead.eivoice_runtime.diagnostics.v1",
            "audio_frame": {
                "last_capture_duration_ms": self.worker_metrics.last_capture_frame_duration_ms,
                "last_encode_duration_ms": self.worker_metrics.last_encode_frame_duration_ms,
                "last_send_duration_ms": self.worker_metrics.last_send_frame_duration_ms,
                "last_receive_duration_ms": self.worker_metrics.last_receive_frame_duration_ms,
                "last_decode_duration_ms": self.worker_metrics.last_decode_frame_duration_ms,
                "last_playback_duration_ms": self.worker_metrics.last_playback_frame_duration_ms,
            },
            "queues": {str(name): dict(_mapping(queue)) for name, queue in queues.items()},
            "audio_frontend": dict(audio_frontend),
            "upstream": {
                "state": transport_state,
                "queue": "ws_send_queue",
                "queue_depth": _optional_int(ws_send_queue.get("depth")) or 0,
                "drop_count": _queue_drop_count(ws_send_queue),
                "last_frame_duration_ms": self.worker_metrics.last_send_frame_duration_ms
                or self.worker_metrics.last_encode_frame_duration_ms
                or self.worker_metrics.last_capture_frame_duration_ms,
            },
            "downstream": {
                "state": transport_state,
                "queue": "opus_decode_queue",
                "queue_depth": _optional_int(opus_decode_queue.get("depth")) or 0,
                "playback_queue_depth": _optional_int(playback_queue.get("depth")) or 0,
                "drop_count": _queue_drop_count(opus_decode_queue) + _queue_drop_count(playback_queue),
                "last_frame_duration_ms": self.worker_metrics.last_playback_frame_duration_ms
                or self.worker_metrics.last_decode_frame_duration_ms
                or self.worker_metrics.last_receive_frame_duration_ms,
            },
            "heartbeat": heartbeat,
            "reconnect": reconnect,
        }

    def _audio_frontend_readiness(self) -> dict[str, Any]:
        readiness = self.audio_frontend.readiness()
        if isinstance(readiness, dict):
            return dict(readiness)
        return {}

    def _read_receive_frame(self) -> AudioFrame | None:
        if self.ws_receive_source is not None:
            return self.ws_receive_source.read_frame()
        if self.transport is None:
            return None
        event = self.transport.receive_event()
        if not isinstance(event, Mapping):
            return None
        return _audio_frame_from_event(event)


def _audio_frame_from_event(event: Mapping[str, Any]) -> AudioFrame | None:
    content = event.get("content")
    if not isinstance(content, Mapping):
        content = {}
    metadata = _mapping(content.get("metadata"))
    encoded = (
        content.get("audioBase64")
        or content.get("audio_base64")
        or event.get("audioBase64")
        or event.get("audio_base64")
    )
    if not encoded:
        return None
    try:
        payload = base64.b64decode(str(encoded))
    except (ValueError, TypeError):
        return None
    sequence = _optional_int(
        content.get("index"),
        content.get("chunkIndex"),
        event.get("chunkIndex"),
        event.get("index"),
    )
    duration_ms = _optional_int(
        content.get("durationMs"),
        content.get("duration_ms"),
        metadata.get("durationMs"),
        metadata.get("duration_ms"),
        event.get("durationMs"),
        event.get("duration_ms"),
    )
    sample_rate_hz = _optional_int(
        content.get("sampleRateHz"),
        content.get("sample_rate_hz"),
        content.get("sampleRate"),
        metadata.get("sampleRateHz"),
        metadata.get("sample_rate_hz"),
        event.get("sampleRateHz"),
        event.get("sample_rate_hz"),
    )
    channels = _optional_int(
        content.get("channels"),
        metadata.get("channels"),
        event.get("channels"),
    )
    return AudioFrame(
        payload=payload,
        sequence=sequence or 0,
        duration_ms=duration_ms or 60,
        sample_rate_hz=sample_rate_hz or 16000,
        channels=channels or 1,
    )


def _frontend_component(*, enabled: bool, available: bool) -> dict[str, bool | str]:
    enabled_b = bool(enabled)
    available_b = bool(available)
    if enabled_b and available_b:
        state = "ready"
    elif enabled_b:
        state = "unavailable"
    else:
        state = "disabled"
    return {
        "enabled": enabled_b,
        "available": available_b,
        "state": state,
    }


def _frontend_component_warnings(config: AcousticFrontendConfig) -> list[str]:
    warnings: list[str] = []
    for label, enabled, available in (
        ("AEC", config.aec_enabled, config.aec_available),
        ("NS", config.ns_enabled, config.ns_available),
        ("VAD", config.vad_enabled, config.vad_available),
        ("loopback", config.loopback_enabled, config.loopback_available),
        ("capture", config.capture_enabled, config.capture_available),
    ):
        if enabled and not available:
            warnings.append(f"{label} configured but unavailable")
    return warnings


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _transport_state(transport_status: Mapping[str, Any]) -> str:
    if not transport_status:
        return "not_configured"
    connection = _mapping(transport_status.get("connection"))
    return str(transport_status.get("state") or connection.get("state") or "unknown")


def _heartbeat_status(transport_status: Mapping[str, Any]) -> dict[str, Any]:
    heartbeat = _mapping(transport_status.get("heartbeat"))
    defaults: dict[str, Any] = {
        "due": False,
        "awaiting_pong": False,
        "timed_out": False,
        "latency_ms": None,
    }
    defaults.update(heartbeat)
    return defaults


def _reconnect_status(transport_status: Mapping[str, Any]) -> dict[str, Any]:
    reconnect = _mapping(transport_status.get("reconnect"))
    defaults: dict[str, Any] = {
        "attempt": 0,
        "backoff_s": None,
        "next_retry_at": None,
        "ready": False,
        "reason": None,
    }
    defaults.update(reconnect)
    return defaults


def _queue_drop_count(queue: Mapping[str, Any]) -> int:
    return (
        (_optional_int(queue.get("dropped_oldest"), queue.get("droppedOldest")) or 0)
        + (_optional_int(queue.get("dropped_newest"), queue.get("droppedNewest")) or 0)
    )


def _drain_queue(queue: Any) -> int:
    cleared = 0
    while queue.pop() is not None:
        cleared += 1
    return cleared


def _optional_int(*values: Any) -> int | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
