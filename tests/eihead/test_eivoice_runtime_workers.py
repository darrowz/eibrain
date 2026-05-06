from __future__ import annotations

import base64
from dataclasses import replace

from eihead.eivoice_runtime import AudioFrame, EiVoiceRuntimeRunner, FakeWebSocketTransport


def _pcm_frame(sequence: int, payload: bytes | None = None) -> AudioFrame:
    return AudioFrame(
        pcm=f"pcm-{sequence}".encode("ascii"),
        payload=payload or b"",
        duration_ms=20,
        sample_rate_hz=16000,
        channels=1,
        sequence=sequence,
    )


class FakeCaptureSource:
    def __init__(self, *frames: AudioFrame) -> None:
        self.frames = list(frames)
        self.read_calls = 0

    def read_frame(self) -> AudioFrame | None:
        self.read_calls += 1
        if not self.frames:
            return None
        return self.frames.pop(0)


class FakeAudioFrontend:
    def __init__(self) -> None:
        self.processed_sequences: list[int] = []
        self._readiness = {
            "aec": {"enabled": True, "available": True},
            "ns": {"enabled": True, "available": True},
            "vad": {"enabled": True, "available": True},
            "loopback": {"enabled": True, "available": True},
            "warnings": [],
        }

    def process_capture(self, frame: AudioFrame) -> AudioFrame:
        self.processed_sequences.append(frame.sequence)
        return replace(frame, pcm=frame.pcm + b"-frontend")

    def readiness(self) -> dict[str, object]:
        return dict(self._readiness)


class FakeCodec:
    def __init__(self) -> None:
        self.encoded_sequences: list[int] = []
        self.decoded_sequences: list[int] = []

    def encode(self, frame: AudioFrame) -> AudioFrame:
        self.encoded_sequences.append(frame.sequence)
        return replace(frame, pcm=b"", payload=frame.pcm + b"-opus")

    def decode(self, frame: AudioFrame) -> AudioFrame:
        self.decoded_sequences.append(frame.sequence)
        return replace(frame, payload=b"", pcm=frame.payload + b"-decoded")


class FakeWsReceiveSource:
    def __init__(self, *frames: AudioFrame) -> None:
        self.frames = list(frames)
        self.read_calls = 0

    def read_frame(self) -> AudioFrame | None:
        self.read_calls += 1
        if not self.frames:
            return None
        return self.frames.pop(0)


class FakePlaybackSink:
    def __init__(self) -> None:
        self.played: list[AudioFrame] = []
        self.stop_calls = 0

    def play(self, frame: AudioFrame) -> None:
        self.played.append(frame)

    def stop(self) -> None:
        self.stop_calls += 1


def test_runner_step_methods_move_audio_through_worker_queues() -> None:
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(_pcm_frame(1)),
        audio_frontend=FakeAudioFrontend(),
        codec=FakeCodec(),
        ws_receive_source=FakeWsReceiveSource(_pcm_frame(7, payload=b"remote-7")),
        playback_sink=FakePlaybackSink(),
    )

    capture_result = runner.step_capture()
    encode_result = runner.step_encode()
    receive_result = runner.step_receive()
    decode_result = runner.step_decode()
    playback_result = runner.step_playback()

    sent_frame = runner.core.ws_send_queue.pop()

    assert capture_result is True
    assert encode_result is True
    assert receive_result is True
    assert decode_result is True
    assert playback_result is True
    assert sent_frame is not None
    assert sent_frame.sequence == 1
    assert sent_frame.payload == b"pcm-1-frontend-opus"
    assert runner.playback_sink.played[0].sequence == 7
    assert runner.playback_sink.played[0].pcm == b"remote-7-decoded"


def test_runner_step_once_executes_all_workers_and_status_merges_metrics() -> None:
    frontend = FakeAudioFrontend()
    codec = FakeCodec()
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(_pcm_frame(2)),
        audio_frontend=frontend,
        codec=codec,
        ws_receive_source=FakeWsReceiveSource(_pcm_frame(8, payload=b"remote-8")),
        playback_sink=FakePlaybackSink(),
    )

    step_result = runner.step_once()
    status = runner.status()

    assert step_result == {
        "capture": True,
        "encode": True,
        "receive": True,
        "decode": True,
        "playback": True,
    }
    assert status["state"] == "idle"
    assert status["worker_metrics"]["capture_frames"] == 1
    assert status["worker_metrics"]["encode_frames"] == 1
    assert status["worker_metrics"]["receive_frames"] == 1
    assert status["worker_metrics"]["decode_frames"] == 1
    assert status["worker_metrics"]["playback_frames"] == 1
    assert status["worker_metrics"]["step_once_calls"] == 1
    assert status["audio_frontend"] == frontend.readiness()
    assert status["queues"]["opus_encode_queue"]["depth"] == 0
    assert status["queues"]["opus_decode_queue"]["depth"] == 0


def test_runner_can_bridge_encoded_audio_to_transport_and_decode_inbound_tts() -> None:
    transport = FakeWebSocketTransport()
    transport.open()
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(_pcm_frame(11)),
        audio_frontend=FakeAudioFrontend(),
        codec=FakeCodec(),
        transport=transport,
        playback_sink=FakePlaybackSink(),
        uid="darrow",
        mid="mid-runner",
    )

    assert runner.step_capture() is True
    assert runner.step_encode() is True
    assert runner.step_send() is True

    outbound = transport.recv_from_client()
    assert outbound is not None
    assert outbound["contentType"] == "AUDIO"
    assert outbound["uid"] == "darrow"
    assert outbound["mid"] == "mid-runner"
    assert base64.b64decode(outbound["content"]["audioBase64"]) == b"pcm-11-frontend-opus"

    transport.deliver_from_server(
        {
            "contentType": "TTS",
            "content": {
                "eventType": "TTS",
                "index": 12,
                "audioBase64": base64.b64encode(b"remote-opus").decode("ascii"),
            },
        }
    )

    assert runner.step_receive() is True
    assert runner.step_decode() is True
    assert runner.step_playback() is True
    assert runner.playback_sink.played[0].sequence == 12
    assert runner.playback_sink.played[0].pcm == b"remote-opus-decoded"
    assert runner.status()["transport"]["transport"] == "fake_websocket"


def test_interrupt_playback_clears_pending_frames_stops_sink_and_tracks_metrics() -> None:
    sink = FakePlaybackSink()
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(),
        audio_frontend=FakeAudioFrontend(),
        codec=FakeCodec(),
        ws_receive_source=FakeWsReceiveSource(),
        playback_sink=sink,
    )

    assert runner.core.audio_playback_queue.push(_pcm_frame(3))
    assert runner.core.audio_playback_queue.push(_pcm_frame(4))

    cleared = runner.interrupt_playback()
    status = runner.status()

    assert cleared == 2
    assert runner.core.audio_playback_queue.pop() is None
    assert sink.stop_calls == 1
    assert status["worker_metrics"]["playback_interrupts"] == 1
    assert status["worker_metrics"]["playback_frames_cleared"] == 2


def test_interrupt_playback_clears_all_downstream_tts_buffers() -> None:
    transport = FakeWebSocketTransport()
    transport.open()
    sink = FakePlaybackSink()
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(),
        audio_frontend=FakeAudioFrontend(),
        codec=FakeCodec(),
        transport=transport,
        playback_sink=sink,
    )
    transport.deliver_from_server(
        {
            "contentType": "TTS",
            "content": {
                "eventType": "TTS",
                "index": 5,
                "audioBase64": base64.b64encode(b"transport-old-opus").decode("ascii"),
            },
        }
    )
    assert runner.core.opus_decode_queue.push(_pcm_frame(6, payload=b"decode-old-opus"))
    assert runner.core.audio_playback_queue.push(_pcm_frame(7, payload=b"playback-old-pcm"))

    cleared = runner.interrupt_playback()
    status = runner.status()

    assert cleared == 3
    assert runner.step_receive() is False
    assert runner.step_decode() is False
    assert runner.step_playback() is False
    assert sink.played == []
    assert sink.stop_calls == 1
    assert status["worker_metrics"]["playback_frames_cleared"] == 1
    assert status["worker_metrics"]["decode_frames_cleared"] == 1
    assert status["worker_metrics"]["transport_inbound_events_cleared"] == 1
    assert status["transport"]["queues"]["inbound_queue"]["depth"] == 0


def test_runner_reports_no_work_when_sources_and_queues_are_empty() -> None:
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(),
        audio_frontend=FakeAudioFrontend(),
        codec=FakeCodec(),
        ws_receive_source=FakeWsReceiveSource(),
        playback_sink=FakePlaybackSink(),
    )

    step_result = runner.step_once()
    status = runner.status()

    assert step_result == {
        "capture": False,
        "encode": False,
        "receive": False,
        "decode": False,
        "playback": False,
    }
    assert status["worker_metrics"]["capture_empty_polls"] == 1
    assert status["worker_metrics"]["receive_empty_polls"] == 1
    assert status["worker_metrics"]["idle_steps"] == 1
