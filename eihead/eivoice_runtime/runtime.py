from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from eibrain.protocol.joyinside_voice import audio_chunk

from .core import AudioFrame, EiVoiceRuntimeCore, OpusCodec
from .transport import VoiceStreamTransport


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

    def to_dict(self) -> dict[str, int]:
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
        return True

    def step_encode(self) -> bool:
        frame = self.core.opus_encode_queue.pop()
        if frame is None:
            self.worker_metrics.encode_empty_polls += 1
            return False
        encoded = self.core.codec.encode(frame)
        self.core.ws_send_queue.push(encoded, block=False)
        self.worker_metrics.encode_frames += 1
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
        )
        accepted = self.transport.send_event(event)
        if not accepted:
            self.worker_metrics.send_rejected += 1
            return False
        self.worker_metrics.send_frames += 1
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
        return True

    def step_playback(self) -> bool:
        frame = self.core.audio_playback_queue.pop()
        if frame is None:
            self.worker_metrics.playback_empty_polls += 1
            return False
        self.playback_sink.play(frame)
        self.worker_metrics.playback_frames += 1
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
        status["worker_metrics"] = self.worker_metrics.to_dict()
        status["audio_frontend"] = self._audio_frontend_readiness()
        if self.transport is not None:
            status["transport"] = self.transport.status()
        return status

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
    return AudioFrame(payload=payload, sequence=sequence or 0)


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
