from __future__ import annotations

from eibrain.protocol.joyinside_voice import (
    audio_finish,
    audio_chunk,
    input_text_to_speech,
    interrupt,
    normalize_voice_event,
    parse_downstream_event,
    ping,
    text_message,
    to_eiprotocol_name,
    voice_chat_update,
)


def test_voice_chat_update_supports_audio_timbre_and_chat_config() -> None:
    event = voice_chat_update(
        uid="user-1",
        mid="msg-1",
        audio_input={"encoding": "pcm16", "sampleRate": 16000},
        audio_output={"encoding": "mp3", "sampleRate": 24000},
        timbre={"voice": "xiaoyi", "voiceSpeed": 1.1, "voiceVolume": 0.8},
        chat={"roleCode": "companion"},
    )

    payload = event.to_dict()

    assert payload["direction"] == "upstream"
    assert payload["contentType"] == "EVENT"
    assert payload["eventType"] == "CLIENT_VOICE_CHAT_UPDATE"
    assert payload["uid"] == "user-1"
    assert payload["mid"] == "msg-1"
    assert payload["content"]["audio"]["input"] == {"encoding": "pcm16", "sampleRate": 16000}
    assert payload["content"]["audio"]["output"] == {"encoding": "mp3", "sampleRate": 24000}
    assert payload["content"]["timbre"]["voiceSpeed"] == 1.1
    assert payload["content"]["timbre"]["voiceVolume"] == 0.8
    assert payload["content"]["chat"]["roleCode"] == "companion"
    assert to_eiprotocol_name(event) == "ei.voice.config.update.requested"


def test_joyinside_voice_helpers_are_exported_from_protocol_package() -> None:
    from eibrain.protocol import (
        JoyInsideVoiceEvent,
        normalize_voice_event as exported_normalize_voice_event,
        voice_chat_update as exported_voice_chat_update,
    )

    event = exported_voice_chat_update(uid="user-1", mid="msg-export")
    normalized = exported_normalize_voice_event(event, direction="upstream")

    assert isinstance(event, JoyInsideVoiceEvent)
    assert normalized["eiprotocolName"] == "ei.voice.config.update.requested"


def test_audio_chunk_supports_index_and_audio_base64() -> None:
    event = audio_chunk(uid="user-1", mid="msg-2", index=7, audio_base64="AAEC")

    payload = event.to_dict()
    normalized = normalize_voice_event(payload, direction="upstream")

    assert payload["contentType"] == "AUDIO"
    assert payload["content"]["index"] == 7
    assert payload["content"]["audioBase64"] == "AAEC"
    assert normalized["eiprotocolName"] == "ei.voice.audio.frame"
    assert normalized["audioBase64"] == "AAEC"
    assert normalized["chunkIndex"] == 7


def test_client_interrupt_maps_to_interrupt_requested() -> None:
    event = interrupt(uid="user-1", mid="msg-3", reason="barge_in")

    normalized = normalize_voice_event(event.to_dict(), direction="upstream")

    assert normalized["contentType"] == "EVENT"
    assert normalized["eventType"] == "CLIENT_INTERRUPT"
    assert normalized["eiprotocolName"] == "ei.dialogue.interrupt.requested"
    assert normalized["reason"] == "barge_in"


def test_other_upstream_constructors_map_to_stable_names() -> None:
    cases = [
        (text_message(uid="user-1", mid="msg-text", text="hello"), "TEXT", "ei.dialogue.user.text"),
        (audio_finish(uid="user-1", mid="msg-finish"), "EVENT", "ei.voice.audio.finish.requested"),
        (
            input_text_to_speech(uid="user-1", mid="msg-tts", text="please speak"),
            "EVENT",
            "ei.voice.tts.requested",
        ),
        (ping(uid="user-1", mid="msg-ping", timestamp=123), "PING", "ei.voice.session.heartbeat"),
    ]

    for event, expected_content_type, expected_name in cases:
        normalized = normalize_voice_event(event, direction="upstream")

        assert event.to_dict()["contentType"] == expected_content_type
        assert normalized["eiprotocolName"] == expected_name


def test_downstream_asr_text_type_controls_final_mapping() -> None:
    final_event = parse_downstream_event(
        {
            "uid": "user-1",
            "mid": "msg-4",
            "contentType": "ASR",
            "content": {"eventType": "ASR", "text": "hello", "textType": "IS_FINAL"},
        }
    )
    partial_event = parse_downstream_event(
        {
            "uid": "user-1",
            "mid": "msg-5",
            "contentType": "ASR",
            "content": {"eventType": "ASR", "text": "hel", "textType": "PARTIAL"},
        }
    )

    assert to_eiprotocol_name(final_event) == "ei.voice.asr.final"
    assert normalize_voice_event(final_event, direction="downstream")["final"] is True
    assert to_eiprotocol_name(partial_event) == "ei.voice.asr.partial"
    assert normalize_voice_event(partial_event, direction="downstream")["final"] is False


def test_downstream_agent_activity_tts_complete_interrupt_and_pong_mapping() -> None:
    cases = [
        ("CFG_BOT_EVENT", {"text": "config"}, "ei.voice.config.update", {"text": "config"}),
        ("SERVER_VOICE_CHAT_UPDATED", {}, "ei.voice.config.updated", {}),
        ("CALL_AGENT_START_EVENT", {}, "ei.voice.agent.started", {}),
        ("EMPTY_CONTENT", {}, "ei.voice.empty", {}),
        ("AGENT", {"text": "thinking"}, "ei.voice.agent.delta", {"text": "thinking"}),
        ("ACTIVITY", {"text": "wave"}, "ei.voice.activity.delta", {"text": "wave"}),
        ("TTS_SENTENCE_START", {"text": "hi"}, "ei.voice.tts.sentence_start", {"text": "hi"}),
        ("TTS", {"audioBase64": "AQID", "index": 3}, "ei.voice.tts.chunk", {"audioBase64": "AQID", "chunkIndex": 3}),
        ("TTS_COMPLETE", {}, "ei.voice.tts.complete", {"final": True}),
        ("COMPLETE", {}, "ei.dialogue.complete", {"final": True}),
        ("CALL_AGENT_INTERRUPTED", {"reason": "user_interrupt"}, "ei.dialogue.interrupt.applied", {"reason": "user_interrupt"}),
        ("PONG", {"timestamp": 123}, "ei.voice.session.heartbeat", {}),
    ]

    for event_type, content, expected_name, expected_fields in cases:
        normalized = normalize_voice_event(
            {
                "uid": "user-1",
                "mid": f"mid-{event_type}",
                "contentType": event_type,
                "content": {"eventType": event_type, **content},
            },
            direction="downstream",
        )

        assert normalized["eiprotocolName"] == expected_name
        for field, expected_value in expected_fields.items():
            assert normalized[field] == expected_value
