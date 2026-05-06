from __future__ import annotations

import pytest


def _session():
    from eibrain.body.realtime_voice import RealtimeVoiceSession

    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    session = RealtimeVoiceSession(
        session_id="session-1",
        actor_id="darrow",
        clock=lambda: next(ticks),
        round_id="round-1",
        cancellation_token="token-1",
    )
    session.start_listening()
    return session


def _event(name: str, content: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "name": name,
        "type": "dialogue",
        "sessionId": "session-1",
        "roundId": "round-1",
        "cancellationToken": "token-1",
        "content": dict(content or {}),
    }


def test_adapter_applies_voice_asr_agent_tts_and_barge_in_mapping_events() -> None:
    from apps.body_runtime.voice_streaming_adapter import VoiceStreamingAdapter

    session = _session()
    adapter = VoiceStreamingAdapter(session)

    partial_trace = adapter.apply(_event("ei.voice.asr.partial", {"text": " 你好 ", "final": False}))
    final_trace = adapter.apply(_event("ei.voice.asr.final", {"text": "你好鸿途", "final": True}))
    delta_trace = adapter.apply(_event("ei.dialogue.agent.delta", {"delta": "你好"}))
    tts_trace = adapter.apply(_event("ei.voice.tts.sentence_start", {"text": "你好。"}))
    interrupt_trace = adapter.apply(_event("ei.voice.barge_in.detected", {"reason": "near_field_speech"}))

    assert session.transcript_partial == "你好"
    assert session.transcript_final == "你好鸿途"
    assert session.reply_text == "你好"
    assert session.first_speech_at_s is not None
    assert session.interrupted is True
    assert session.interrupt_reason == "near_field_speech"
    assert partial_trace["operation"] == "update_partial_transcript"
    assert final_trace["operation"] == "finalize_transcript"
    assert delta_trace["operation"] == "append_reply_delta"
    assert tts_trace["operation"] == "start_speaking"
    assert interrupt_trace["operation"] == "interrupt"


def test_adapter_accepts_event_envelope_and_legacy_dialogue_asr_names() -> None:
    from apps.body_runtime.voice_streaming_adapter import VoiceStreamingAdapter
    from eiprotocol.models import EventEnvelope, SourceRef

    session = _session()
    adapter = VoiceStreamingAdapter(session)
    envelope = EventEnvelope(
        event_id="evt-1",
        event_type="dialogue",
        name="ei.dialogue.asr.partial",
        time="2026-05-05T00:00:00+08:00",
        sequence=1,
        request_id="req-1",
        source=SourceRef(domain="eihead"),
        content={"text": "legacy partial", "cancellationToken": "token-1"},
        session_id="session-1",
        round_id="round-1",
    )

    adapter.apply(envelope)
    adapter.apply(_event("ei.dialogue.asr.final", {"text": "legacy final"}))

    assert session.transcript_partial == "legacy partial"
    assert session.transcript_final == "legacy final"


def test_adapter_exposes_asr_partial_and_final_diagnostics_in_live_trace() -> None:
    from apps.body_runtime.voice_streaming_adapter import VoiceStreamingAdapter

    session = _session()
    adapter = VoiceStreamingAdapter(session)

    partial_trace = adapter.apply(
        _event(
            "ei.voice.asr.partial",
            {
                "text": "hello",
                "latencyMs": 37,
                "frameIndex": 2,
                "framesReceived": 2,
                "framesProcessed": 1,
                "framesDropped": 1,
                "providerState": "streaming",
            },
        )
    )
    final_trace = adapter.apply(
        _event(
            "ei.voice.asr.final",
            {
                "text": "hello final",
                "latency_ms": 54,
                "frame_index": 3,
                "frames_received": 3,
                "frames_processed": 2,
                "frames_dropped": 1,
                "provider_state": "finalized",
            },
        )
    )

    assert partial_trace["live_trace"]["asr"]["partial_count"] == 1
    assert final_trace["live_trace"]["asr"]["final_count"] == 1
    assert final_trace["live_trace"]["asr"]["last_event"] == {
        "name": "ei.voice.asr.final",
        "session_id": "session-1",
        "round_id": "round-1",
        "latency_ms": 54,
        "frame_index": 3,
        "frames_received": 3,
        "frames_processed": 2,
        "frames_dropped": 1,
        "provider_state": "finalized",
    }
    assert adapter.snapshot()["live_trace"]["asr"]["last_event"]["provider_state"] == "finalized"


def test_adapter_maps_playback_started_and_interrupt_requested_aliases() -> None:
    from apps.body_runtime.voice_streaming_adapter import VoiceStreamingAdapter

    session = _session()
    adapter = VoiceStreamingAdapter(session)

    adapter.apply(_event("ei.voice.playback.started", {"state": "started"}))
    adapter.apply(_event("ei.dialogue.interrupt.requested", {"reason": "user_barge_in"}))

    assert session.first_speech_at_s is not None
    assert session.phase == "barge_in"
    assert session.interrupted is True
    assert session.interrupt_reason == "user_barge_in"


def test_adapter_records_audio_frame_tts_chunk_and_heartbeat_for_live_trace() -> None:
    from apps.body_runtime.voice_streaming_adapter import VoiceStreamingAdapter

    session = _session()
    adapter = VoiceStreamingAdapter(session)

    audio_trace = adapter.apply(_event("ei.voice.audio.frame", {"streamId": "mic", "chunkIndex": 0}))
    tts_chunk_trace = adapter.apply(_event("ei.voice.tts.chunk", {"streamId": "tts", "chunkIndex": 0}))
    heartbeat_trace = adapter.apply(
        {
            "name": "ei.voice.session.heartbeat",
            "type": "control",
            "sessionId": "session-1",
            "content": {"state": "capturing", "health": {"capture": {"status": "ok"}}},
        }
    )

    assert session.first_audio_at_s is not None
    assert audio_trace["operation"] == "note_audio"
    assert tts_chunk_trace["operation"] == "observe_tts_chunk"
    assert tts_chunk_trace["live_trace"]["tts_chunks"] == 1
    assert heartbeat_trace["operation"] == "observe_heartbeat"
    assert heartbeat_trace["applied"] is True
    assert heartbeat_trace["live_trace"]["last_heartbeat"]["state"] == "capturing"
    assert adapter.snapshot()["streaming"] == {
        "audio_frames": 1,
        "tts_chunks": 1,
        "last_heartbeat_state": "capturing",
    }
    assert [event.event_type for event in session.events[-3:]] == [
        "audio_detected",
        "tts_chunk",
        "voice_heartbeat",
    ]
    assert session.events[-1].payload["state"] == "capturing"


def test_adapter_maps_playback_stopped_to_complete_or_tts_stop_trace() -> None:
    from apps.body_runtime.voice_streaming_adapter import VoiceStreamingAdapter

    session = _session()
    adapter = VoiceStreamingAdapter(session)

    adapter.apply(_event("ei.voice.playback.started", {"state": "playing"}))
    complete_trace = adapter.apply(_event("ei.voice.playback.stopped", {"state": "stopped", "reason": "completed"}))

    assert complete_trace["operation"] == "complete_playback"
    assert session.phase == "completed"
    assert session.status == "playback_completed"

    session.start_listening(fresh_round=True, round_id="round-2", cancellation_token="token-2")
    adapter.apply(
        {
            **_event("ei.voice.playback.started", {"state": "playing"}),
            "roundId": "round-2",
            "cancellationToken": "token-2",
        }
    )
    stop_trace = adapter.apply(
        {
            **_event("ei.voice.playback.stopped", {"state": "stopped", "reason": "barge_in"}),
            "roundId": "round-2",
            "cancellationToken": "token-2",
        }
    )

    assert stop_trace["operation"] == "mark_tts_stopped"
    assert session.tts_stopped is True
    assert session.cancellation_chain[-1]["target"] == "tts"


def test_adapter_treats_duplicate_completed_playback_stop_as_idempotent() -> None:
    from apps.body_runtime.voice_streaming_adapter import VoiceStreamingAdapter

    session = _session()
    adapter = VoiceStreamingAdapter(session)

    adapter.apply(_event("ei.voice.playback.started", {"state": "playing"}))
    first_trace = adapter.apply(_event("ei.voice.playback.stopped", {"state": "stopped", "reason": "completed"}))
    second_trace = adapter.apply(_event("ei.voice.playback.stopped", {"state": "stopped", "reason": "completed"}))

    assert first_trace["operation"] == "complete_playback"
    assert second_trace["operation"] == "duplicate_playback_stop"
    assert second_trace["applied"] is False
    assert second_trace["terminal"] is True
    assert session.phase == "completed"
    assert session.status == "playback_completed"


def test_adapter_lets_realtime_session_reject_stale_round_token() -> None:
    from apps.body_runtime.voice_streaming_adapter import VoiceStreamingAdapter

    session = _session()
    adapter = VoiceStreamingAdapter(session)

    session.start_listening(
        fresh_round=True,
        round_id="round-2",
        cancellation_token="token-2",
    )

    with pytest.raises(RuntimeError, match="not current"):
        adapter.apply(_event("ei.voice.asr.partial", {"text": "late partial"}))

    assert session.transcript_partial == ""
