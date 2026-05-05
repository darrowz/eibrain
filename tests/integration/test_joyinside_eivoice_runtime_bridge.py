from __future__ import annotations


def test_joyinside_asr_event_flows_into_realtime_voice_bridge() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    normalized = normalize_voice_event(
        {
            "uid": "darrow",
            "mid": "msg-1",
            "contentType": "ASR",
            "content": {
                "eventType": "ASR",
                "text": "你好鸿途",
                "textType": "IS_FINAL",
            },
        },
        direction="downstream",
    )

    result = bridge.handle_event(normalized)

    assert normalized["eiprotocolName"] == "ei.voice.asr.final"
    assert result["lane"] == "slow_reasoning"
    assert result["blackboardPatch"]["asrFinal"] == "你好鸿途"


def test_joyinside_tts_chunk_flows_into_speaking_lane() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    bridge.handle_event({"type": "ASR_FINAL", "payload": {"text": "介绍一下你自己"}})
    normalized = normalize_voice_event(
        {
            "uid": "darrow",
            "mid": "msg-tts-1",
            "contentType": "TTS",
            "content": {
                "eventType": "TTS",
                "audioBase64": "AQID",
                "index": 4,
            },
        },
        direction="downstream",
    )

    result = bridge.handle_event(normalized)

    assert normalized["eiprotocolName"] == "ei.voice.tts.chunk"
    assert result["lane"] == "speaking"
    assert result["speechPlan"]["phase"] == "chunk"
    assert result["speechPlan"]["chunkIndex"] == 4
    assert result["speechPlan"]["metadata"]["audioBase64"] == "AQID"


def test_joyinside_interrupt_event_cancels_active_voice_round() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import interrupt, normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    partial = bridge.handle_event({"type": "ASR_PARTIAL", "payload": {"text": "介绍一下"}})
    normalized = normalize_voice_event(interrupt(uid="darrow", mid="msg-2", reason="barge_in"), direction="upstream")

    result = bridge.handle_event(normalized)

    assert normalized["eiprotocolName"] == "ei.dialogue.interrupt.requested"
    assert result["lane"] == "interrupt"
    assert result["interrupt"]["cancelOldRound"] is True
    assert result["interrupt"]["oldRound"]["roundId"] == partial["roundId"]


def test_joyinside_activity_event_preserves_payload_for_proactive_lane() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    normalized = normalize_voice_event(
        {
            "uid": "darrow",
            "mid": "msg-activity-1",
            "contentType": "ACTIVITY",
            "content": {
                "eventType": "ACTIVITY",
                "idleSeconds": 180,
                "emotion": {"mood": "sad"},
                "memoryCandidates": [{"id": "mem-1", "text": "喝水提醒", "importance": 0.9}],
            },
        },
        direction="downstream",
    )

    result = bridge.handle_event(normalized)

    assert normalized["payload"]["idleSeconds"] == 180
    assert result["lane"] == "proactive_activity"
    assert result["actions"][0]["proposal"]["should_emit"] is True
    assert result["memoryHints"][0]["candidate"]["id"] == "mem-1"


def test_joyinside_queue_health_preserves_payload_for_runtime_health_lane() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    normalized = normalize_voice_event(
        {
            "uid": "darrow",
            "mid": "msg-health-1",
            "contentType": "QUEUE_HEALTH",
            "content": {
                "eventType": "QUEUE_HEALTH",
                "status": "degraded",
                "queue": "ws_send_queue",
                "latencyMs": 850,
            },
        },
        direction="downstream",
    )

    result = bridge.handle_event(normalized)

    assert normalized["payload"]["status"] == "degraded"
    assert result["lane"] == "runtime_health"
    assert result["conversationState"] == "degraded"
    assert result["blackboardPatch"]["warnings"][0]["queue"] == "ws_send_queue"
    assert result["actions"][0]["pauseProactiveInteraction"] is True


def test_eivoice_runtime_status_feeds_monitoring_panel() -> None:
    from eihead.eivoice_runtime import AudioFrame, EiVoiceRuntimeCore
    from eihead.monitoring.eivoice_runtime import build_eivoice_runtime_panel

    runtime = EiVoiceRuntimeCore()
    runtime.state_machine.wake_detected()
    for sequence in range(27):
        runtime.ws_send_queue.push(AudioFrame(payload=b"x", sequence=sequence))

    panel = build_eivoice_runtime_panel(runtime.status())

    assert panel["state"] == "conversation"
    assert panel["queues"]["ws_send_queue"]["capacity"] == 25
    assert panel["queues"]["ws_send_queue"]["policy"] == "drop_oldest"
    assert panel["droppedTotal"] == 2
    assert panel["health"] == "degraded"
    assert panel["conversationState"] == "conversation"
    assert panel["wakeword"]["capacity_ms"] == 1500
    assert any("audio frontend readiness is missing" in warning for warning in panel["warnings"])


def _clock():
    values = iter([7000.0, 7000.01, 7000.02, 7000.03, 7000.04, 7000.05])

    def clock() -> float:
        return next(values)

    return clock
