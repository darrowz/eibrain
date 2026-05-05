from __future__ import annotations


def test_operator_console_cli_decodes_unicode_escaped_voice_words() -> None:
    from apps.operator_console.__main__ import _decode_cli_text

    assert _decode_cli_text(r"\u9e3f\u9014") == "\u9e3f\u9014"
    assert _decode_cli_text(r"\u7ed3\u675f\u5bf9\u8bdd") == "\u7ed3\u675f\u5bf9\u8bdd"


def test_operator_console_builds_runtime_status_report() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "node_id": "honjia",
            "organ_count": 1,
            "recent_event_count": 2,
            "degradation_mode": "normal",
            "capabilities": {"can_speak": True},
            "organs": {
                "mouth": {
                    "health": "healthy",
                    "subfunctions": {
                        "tts_playback": {
                            "health": "healthy",
                            "details": {
                                "driver": "command",
                                "status": "healthy",
                                "elapsed_ms": 12.5,
                                "details": {
                                    "label": "speaker:plughw:2,0",
                                    "device": "/dev/snd",
                                    "device_exists": True,
                                },
                            },
                        }
                    },
                }
            },
        },
        cognitive_snapshot={"last_reply": "hello", "learning_decision": "keep_policy"},
        traces=[{"trace_id": "t1", "kind": "audio_transcript_final"}],
    )

    assert report["system_health"] == "healthy"
    assert report["generated_at_ts"] > 0
    assert report["trace_count"] == 1
    assert report["degraded_organs"] == []
    assert report["summary"]["warning_count"] == 0
    assert report["summary"]["healthy_subfunction_count"] == 1
    assert report["summary"]["real_driver_count"] == 1
    assert report["body"]["degradation_mode"] == "normal"
    assert report["runtime_overview"]["node_id"] == "honjia"
    assert report["driver_breakdown"] == [{"driver": "command", "count": 1}]
    assert report["capability_status"] == [{"name": "can_speak", "enabled": True, "status": "enabled"}]
    assert report["probe_metrics"][0]["device"] == "/dev/snd"
    assert report["probe_metrics"][0]["device_exists"] is True
    assert report["dialogue_diagnostics"]["last_reply"] == "hello"


def test_operator_console_exposes_dialogue_loop_diagnostics() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {"can_hear_voice": True, "can_transcribe_speech": True, "can_speak": True},
            "organs": {},
            "voice_dialogue": {
                "enabled": True,
                "running": True,
                "conversation_active": True,
                "wake_word": "\u9e3f\u9014",
                "sleep_word": "\u7ed3\u675f\u5bf9\u8bdd",
                "phase": "thinking",
                "last_status": "transcribed",
                "turn_count": 3,
                "last_transcript": "福州小吃",
                "last_reply": "鱼丸和肉燕很有名。",
                "last_error": "",
                "current_phase_elapsed_s": 1.25,
                "last_latency_s": {"listen_asr": 4.0, "think": 2.0, "speak": 3.0, "total": 9.0},
                "last_stage_latency_ms": {"listen_asr": 4000.0, "think": 2000.0, "speak": 3000.0, "total": 9000.0},
                "last_bottleneck_stage": "listen_asr",
                "last_bottleneck_ms": 4000.0,
            },
        },
        cognitive_snapshot={
            "last_reply": "fallback reply",
            "learning_decision": "keep_policy",
            "last_llm_status": {"provider": "anthropic_compatible", "status": "ok", "error": ""},
        },
        traces=[],
    )

    dialogue = report["dialogue_diagnostics"]

    assert dialogue["running"] is True
    assert dialogue["conversation_active"] is True
    assert dialogue["wake_word"] == "\u9e3f\u9014"
    assert dialogue["sleep_word"] == "\u7ed3\u675f\u5bf9\u8bdd"
    assert dialogue["phase"] == "thinking"
    assert dialogue["turn_count"] == 3
    assert dialogue["last_transcript"] == "福州小吃"
    assert dialogue["last_reply"] == "鱼丸和肉燕很有名。"
    assert dialogue["current_phase_elapsed_s"] == 1.25
    assert dialogue["last_latency_s"]["total"] == 9.0
    assert dialogue["last_stage_latency_ms"]["listen_asr"] == 4000.0
    assert dialogue["last_bottleneck_stage"] == "listen_asr"
    assert dialogue["last_bottleneck_ms"] == 4000.0
    assert dialogue["last_llm_status"]["status"] == "ok"
    assert any(metric["id"] == "voice_dialogue.listen_asr" for metric in report["latency_metrics"])


def test_operator_console_exposes_voice_chain_readiness_in_dialogue_diagnostics() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    benchmark = {
        "turnCount": 2,
        "roundLeakCount": 0,
        "metrics": {
            "asrFinalMs": {"p95": 620.0, "threshold": 800.0, "pass": True},
            "firstAudioMs": {"p95": 1510.0, "threshold": 2000.0, "pass": True},
            "interruptStopMs": {"p95": 210.0, "threshold": 300.0, "pass": True},
        },
        "bottleneck": {"field": "firstAudioMs", "p95": 1510.0, "threshold": 2000.0},
    }
    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {"can_hear_voice": True, "can_transcribe_speech": True, "can_speak": True},
            "organs": {},
            "voice_dialogue": {
                "enabled": True,
                "running": True,
                "phase": "idle",
                "last_status": "reply_ready",
                "voice_chain_benchmark": benchmark,
            },
        },
        cognitive_snapshot={},
        traces=[],
    )

    readiness = report["dialogue_diagnostics"]["voice_chain_readiness"]
    assert readiness["source"] == "live_benchmark"
    assert readiness["live"] is True
    assert readiness["honjiaReady"] is True
    assert readiness["turnCount"] == 2
    assert readiness["summary"] == "ready: 2 live turns"
    assert readiness["bottleneck"]["field"] == "firstAudioMs"


def test_operator_console_backfills_live_ear_card_from_recent_audio_trace() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {},
            "organs": {
                "ear": {
                    "health": "healthy",
                    "subfunctions": {
                        "capture": {"health": "healthy", "details": {"status": "live_probe_skipped"}},
                        "vad": {"health": "healthy", "details": {"status": "live_probe_skipped"}},
                        "asr": {"health": "healthy", "details": {"status": "live_probe_skipped"}},
                    },
                }
            },
            "voice_dialogue": {"enabled": True, "running": True, "phase": "listening", "last_status": "listening"},
        },
        cognitive_snapshot={},
        traces=[
            {
                "kind": "audio_transcript_final",
                "details": {
                    "text": "你好鸿途",
                    "capture_device": "plughw:CARD=U4K,DEV=0",
                    "capture_elapsed_ms": 21.5,
                    "vad_elapsed_ms": 4012.5,
                    "asr_elapsed_ms": 5333.3,
                    "asr_decode_elapsed_ms": 320.4,
                    "asr_status": "transcribed",
                    "vad_triggered": True,
                    "streaming_vad": True,
                },
            }
        ],
    )

    ear = next(card for card in report["organ_cards"] if card["name"] == "ear")
    by_name = {entry["name"]: entry for entry in ear["subfunctions"]}
    assert by_name["capture"]["elapsed_ms"] == 21.5
    assert by_name["vad"]["elapsed_ms"] == 4012.5
    assert by_name["asr"]["elapsed_ms"] == 320.4
    assert by_name["asr"]["status"] == "transcribed"
    assert report["audio_diagnostics"]["capture_elapsed_ms"] == 21.5
    assert report["audio_diagnostics"]["asr_elapsed_ms"] == 5333.3
    assert report["audio_diagnostics"]["asr_decode_elapsed_ms"] == 320.4


def test_operator_console_backfills_dialogue_stage_latency_from_seconds() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {},
            "organs": {},
            "voice_dialogue": {
                "enabled": True,
                "running": True,
                "phase": "idle",
                "last_latency_s": {"listen_asr": 1.5, "total": 1.6},
            },
        },
        cognitive_snapshot={},
        traces=[],
    )

    dialogue = report["dialogue_diagnostics"]
    assert dialogue["last_stage_latency_ms"]["listen_asr"] == 1500.0
    assert dialogue["last_bottleneck_stage"] == "listen_asr"


def test_operator_console_exposes_memory_ownership_diagnostics() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={"degradation_mode": "normal", "capabilities": {}, "organs": {}},
        cognitive_snapshot={
            "memory_diagnostics": {
                "provider": "eimemory_rpc",
                "endpoint": "http://honxin:18081",
                "channel_owner": "eibrain",
                "agent_owner": "eibrain",
                "memory_owner": "eimemory",
                "last_query": "介绍下你自己",
                "task_context": {
                    "task_type": "brain.respond",
                    "recall_profile": "precision",
                    "allowed_sources": ["eibrain.audio_dialogue"],
                    "blocked_sources": ["eimemory.knowledge.claims"],
                },
                "last_recall": {"selected_count": 1},
            }
        },
        traces=[],
    )

    memory = report["memory_diagnostics"]
    assert memory["provider"] == "eimemory_rpc"
    assert memory["memory_owner"] == "eimemory"
    assert memory["last_query"] == "介绍下你自己"


def test_operator_console_exposes_realtime_memory_traces_for_monitoring() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    trace = {
        "schema": "eibrain.memory.closed_loop_trace.v1",
        "round_id": "round-42",
        "session_id": "session-1",
        "actor_id": "darrow",
        "recall": {
            "count": 1,
            "items": [
                {
                    "query": "用户偏好简短回复",
                    "summary": "Prefer short spoken replies.",
                    "selected_count": 1,
                    "selected_records": [{"record_id": "mem_1", "title": "Reply style", "source": "eibrain.preference"}],
                    "source_composition": {"eibrain.preference": 1},
                }
            ],
        },
        "writeback": {
            "count": 1,
            "items": [
                {
                    "status": "ok",
                    "summary": "用户偏好更短的语音回复。",
                    "source": "eibrain.semantic_candidate",
                    "memory_type": "semantic_candidate",
                    "diagnostics": {"record_id": "mem_2"},
                }
            ],
        },
        "errors": [],
    }
    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={"degradation_mode": "normal", "capabilities": {}, "organs": {}},
        cognitive_snapshot={"current": {"memory_traces": [trace]}, "scheduler": {"memory_trace_count": 1}},
        traces=[],
    )

    memory = report["memory_diagnostics"]
    panel = report["memory_trace_panel"]

    assert memory["memory_trace_count"] == 1
    assert memory["latest_trace_round_id"] == "round-42"
    assert memory["selected_count"] == 1
    assert memory["selected_records"][0]["record_id"] == "mem_1"
    assert memory["last_writeback"]["record_id"] == "mem_2"
    assert panel["count"] == 1
    assert panel["latest"]["round_id"] == "round-42"
    assert panel["latest"]["recall_count"] == 1
    assert panel["latest"]["writeback_count"] == 1
    assert panel["latest"]["error_count"] == 0
    assert panel["items"][0]["recall_items"][0]["query"] == "用户偏好简短回复"
    assert panel["items"][0]["writeback_items"][0]["record_id"] == "mem_2"


def test_operator_console_uses_live_voice_scheduler_memory_traces() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    trace = {
        "schema": "eibrain.memory.closed_loop_trace.v1",
        "round_id": "round-live-memory",
        "session_id": "session-live",
        "actor_id": "darrow",
        "recall": {
            "count": 1,
            "items": [
                {
                    "query": "用户刚才问响应性能",
                    "selected_count": 1,
                    "selected_records": [{"record_id": "mem_live_1", "source": "eibrain.audio_dialogue"}],
                    "source_composition": {"eibrain.audio_dialogue": 1},
                }
            ],
        },
        "writeback": {
            "count": 1,
            "items": [
                {
                    "status": "ok",
                    "summary": "用户关注语音响应性能。",
                    "record_id": "mem_live_2",
                }
            ],
        },
        "errors": [],
    }
    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {},
            "organs": {},
            "voice_dialogue": {
                "enabled": True,
                "running": True,
                "scheduler_state": {
                    "state": "active",
                    "memory_trace_count": 1,
                    "memory_traces": [trace],
                },
            },
        },
        cognitive_snapshot={},
        traces=[],
    )

    assert report["memory_trace_panel"]["count"] == 1
    assert report["memory_trace_panel"]["latest"]["round_id"] == "round-live-memory"
    assert report["memory_diagnostics"]["selected_records"][0]["record_id"] == "mem_live_1"
    assert report["memory_diagnostics"]["last_writeback"]["record_id"] == "mem_live_2"


def test_operator_console_marks_report_degraded_when_capabilities_missing() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "fixed_gaze",
            "capabilities": {
                "can_hear_voice": True,
                "can_transcribe_speech": False,
                "can_see_people": True,
                "can_identify_person": False,
                "can_speak": True,
                "can_orient_head": False,
            },
            "organs": {
                "neck": {"health": "degraded"},
            },
        },
        cognitive_snapshot={},
        traces=[],
    )

    assert report["system_health"] == "degraded"
    assert "can_transcribe_speech" in report["warnings"][0]
    assert "can_orient_head" in report["warnings"][1]
    assert report["degraded_organs"] == ["neck"]
    assert report["summary"]["degraded_organ_count"] == 1


def test_operator_console_extracts_probe_details_for_missing_hardware() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {"can_see_people": False},
            "organs": {
                "eye": {
                    "health": "degraded",
                    "subfunctions": {
                        "camera": {
                            "health": "unavailable",
                            "details": {
                                "driver": "command",
                                "status": "unavailable",
                                "elapsed_ms": 81.0,
                                "details": {
                                    "label": "camera",
                                    "device": "/dev/video0",
                                    "device_exists": False,
                                    "binary": "/usr/bin/v4l2-ctl",
                                },
                            },
                        }
                    },
                }
            },
        },
        cognitive_snapshot={},
        traces=[],
    )

    assert report["summary"]["unavailable_probe_count"] == 1
    assert report["probe_metrics"][0]["device"] == "/dev/video0"
    assert report["probe_metrics"][0]["device_exists"] is False
    assert report["probe_metrics"][0]["label"] == "camera"


def test_operator_console_exposes_visual_diagnostics() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {"can_see_people": True, "can_identify_person": False},
            "visual_tracking": {
                "running": True,
                "status": "tracking",
                "source": "state",
                "updated_at_ts": 124.0,
                "miss_count": 0,
                "last_outcome_status": "ok",
                "target": {
                    "label": "face",
                    "score": 0.91,
                    "target_x": 0.4,
                    "bbox": {
                        "x_min": 0.2,
                        "y_min": 0.1,
                        "x_max": 0.6,
                        "y_max": 0.7,
                    },
                },
            },
            "organs": {
                "eye": {
                    "health": "degraded",
                    "subfunctions": {
                        "camera": {
                            "health": "healthy",
                            "details": {
                                "driver": "command",
                                "status": "healthy",
                                "elapsed_ms": 35.0,
                                "frame_path": "/tmp/latest.jpg",
                                "frame_captured_at_ts": 123.0,
                                "details": {
                                    "label": "camera",
                                    "device": "/dev/video0",
                                    "device_exists": True,
                                },
                            },
                        },
                        "detection": {
                            "health": "healthy",
                            "details": {
                                "driver": "command",
                                "status": "healthy",
                                "backend": "gstreamer_hailo",
                                "service_status": "ok",
                                "state_path": "/tmp/eibrain-vision/state.json",
                                "state_age_s": 0.25,
                                "state_updated_at_ts": 122.9,
                                "elapsed_ms": 70.0,
                                "frame_path": "/tmp/latest.jpg",
                                "frame_captured_at_ts": 123.0,
                                "frame_updated_at_ts": 123.0,
                                "scene_summary": "1 face, 1 person",
                                "scene_labels": ["face", "person"],
                                "top_detection": {
                                    "label": "face",
                                    "score": 0.91,
                                    "bbox": {
                                        "x_min": 0.2,
                                        "y_min": 0.1,
                                        "x_max": 0.6,
                                        "y_max": 0.7,
                                    },
                                },
                                "detections": [
                                    {
                                        "label": "face",
                                        "score": 0.91,
                                        "bbox": {
                                            "x_min": 0.2,
                                            "y_min": 0.1,
                                            "x_max": 0.6,
                                            "y_max": 0.7,
                                        },
                                    }
                                ],
                            },
                        },
                        "identity": {
                            "health": "degraded",
                            "details": {
                                "driver": "command",
                                "status": "observing_unknown_face",
                                "elapsed_ms": 5.0,
                                "identity_summary": "1 unknown face candidate(s)",
                                "identity_candidates": [
                                    {
                                        "candidate_id": "unknown-face-1",
                                        "identity": "unknown",
                                        "score": 0.91,
                                    }
                                ],
                            },
                        },
                    },
                }
            },
        },
        cognitive_snapshot={},
        traces=[],
    )

    assert report["visual_diagnostics"]["frame_url"] == "/vision/latest.jpg"
    assert report["visual_diagnostics"]["detection_count"] == 1
    assert report["visual_diagnostics"]["detections"][0]["label"] == "face"
    assert report["visual_diagnostics"]["identity_candidates"][0]["candidate_id"] == "unknown-face-1"
    assert report["visual_diagnostics"]["tracking_status"] == "tracking"
    assert report["visual_diagnostics"]["tracking_source"] == "state"
    assert report["visual_diagnostics"]["tracking_target"]["label"] == "face"
    assert report["visual_diagnostics"]["backend"] == "gstreamer_hailo"
    assert report["visual_diagnostics"]["vision_service_status"] == "ok"
    assert report["visual_diagnostics"]["state_age_s"] == 0.25
    assert report["visual_diagnostics"]["state_path"] == "/tmp/eibrain-vision/state.json"
    assert report["visual_diagnostics"]["top_detection_bbox"]["x_min"] == 0.2


def test_operator_console_distinguishes_health_from_live_data() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {"can_see_people": True, "can_speak": True, "can_orient_head": True},
            "organs": {
                "eye": {
                    "health": "healthy",
                    "subfunctions": {
                        "camera": {"health": "healthy", "details": {"driver": "command", "status": "live_probe_skipped"}},
                        "detection": {"health": "healthy", "details": {"driver": "command", "status": "live_probe_skipped"}},
                    },
                },
                "neck": {
                    "health": "healthy",
                    "subfunctions": {
                        "tracking": {"health": "healthy", "details": {"driver": "command", "status": "live_probe_skipped"}},
                    },
                },
            },
        },
        cognitive_snapshot={},
        traces=[],
    )

    eye_card = next(card for card in report["organ_cards"] if card["name"] == "eye")
    neck_card = next(card for card in report["organ_cards"] if card["name"] == "neck")

    assert eye_card["health"] == "healthy"
    assert eye_card["data_health"] == "waiting"
    assert eye_card["data_status"] == "waiting_for_data"
    assert eye_card["live_data_subfunctions"] == 0
    assert eye_card["subfunctions"][0]["data_status"] == "waiting_for_data"
    assert neck_card["data_status"] == "waiting_for_data"
    assert report["summary"]["live_data_subfunction_count"] == 0
    assert report["summary"]["waiting_data_subfunction_count"] == 3
    assert report["visual_diagnostics"]["data_health"] == "waiting"
    assert report["visual_diagnostics"]["data_status"] == "waiting_for_frame"


def test_operator_console_treats_sleeping_vision_as_healthy_standby() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {"can_see_people": True},
            "organs": {
                "eye": {
                    "health": "healthy",
                    "subfunctions": {
                        "camera": {
                            "health": "healthy",
                            "details": {
                                "driver": "vision_state",
                                "status": "sleeping",
                                "service_status": "sleeping",
                                "backend": "vision_sleep_gate",
                                "state_path": "/tmp/eibrain-vision/state.json",
                            },
                        },
                        "detection": {
                            "health": "healthy",
                            "details": {
                                "driver": "vision_state",
                                "status": "sleeping",
                                "service_status": "sleeping",
                                "backend": "vision_sleep_gate",
                                "detections": [],
                            },
                        },
                    },
                }
            },
        },
        cognitive_snapshot={},
        traces=[],
    )

    assert report["system_health"] == "healthy"
    assert report["visual_diagnostics"]["data_status"] == "sleeping"
    assert report["visual_diagnostics"]["data_health"] == "healthy"
    assert report["visual_diagnostics"]["vision_service_status"] == "sleeping"


def test_operator_console_exposes_audio_diagnostics() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {"can_hear_voice": True, "can_transcribe_speech": True},
            "organs": {
                "ear": {
                    "health": "healthy",
                    "subfunctions": {
                        "capture": {
                            "health": "healthy",
                            "details": {
                                "driver": "command",
                                "status": "healthy",
                                "elapsed_ms": 25.0,
                                "capture_device": "hw:3,0",
                                "sample_rate": 48000,
                                "channels": 2,
                                "chunk_count": 2,
                                "payload_bytes": 8192,
                                "dbfs": -22.1,
                                "rms_level": 0.08,
                                "peak_level": 0.2,
                                "voice_activity": True,
                            },
                        },
                        "vad": {
                            "health": "healthy",
                            "details": {
                                "driver": "noop",
                                "status": "observed",
                                "speech_window_summary": "voice activity detected at -22.1 dBFS",
                            },
                        },
                        "asr": {
                            "health": "healthy",
                            "details": {
                                "driver": "command",
                                "status": "transcribed",
                                "elapsed_ms": 90.0,
                                "transcript": "ni hao honjia",
                                "voice_activity": True,
                                "speech_window_summary": "heard speech at -22.1 dBFS: ni hao honjia",
                            },
                        },
                    },
                }
            },
        },
        cognitive_snapshot={},
        traces=[],
    )

    assert report["audio_diagnostics"]["capture_device"] == "hw:3,0"
    assert report["audio_diagnostics"]["voice_activity"] is True
    assert report["audio_diagnostics"]["transcript"] == "ni hao honjia"


def test_operator_console_backfills_audio_diagnostics_from_recent_trace() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {"can_hear_voice": True, "can_transcribe_speech": True},
            "organs": {
                "ear": {
                    "health": "healthy",
                    "subfunctions": {
                        "capture": {"health": "healthy", "details": {"driver": "command", "status": "live_probe_skipped"}},
                        "vad": {"health": "healthy", "details": {"driver": "command", "status": "live_probe_skipped"}},
                        "asr": {"health": "healthy", "details": {"driver": "command", "status": "live_probe_skipped"}},
                    },
                }
            },
        },
        cognitive_snapshot={},
        traces=[
            {
                "kind": "audio_transcript_final",
                "recorded_at_ts": 123.0,
                "details": {
                    "text": "",
                    "speech_window_summary": "voice activity detected at -32.4 dBFS",
                    "dbfs": -32.4,
                    "rms_level": 0.024,
                    "peak_level": 0.77,
                    "payload_bytes": 576000,
                    "capture_device": "plughw:CARD=U4K,DEV=0",
                    "capture_elapsed_ms": 3140.0,
                    "asr_elapsed_ms": 0.01,
                    "asr_status": "below_asr_threshold",
                },
            }
        ],
    )

    assert report["audio_diagnostics"]["capture_device"] == "plughw:CARD=U4K,DEV=0"
    assert report["audio_diagnostics"]["dbfs"] == -32.4
    assert report["audio_diagnostics"]["capture_status"] == "recent_trace"
    assert report["audio_diagnostics"]["asr_status"] == "below_asr_threshold"
    assert report["summary"]["avg_latency_ms"] is not None
    assert report["latency_metrics"][0]["id"] == "ear.capture.recent"


def test_operator_console_exposes_neck_control_diagnostics() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={
            "degradation_mode": "normal",
            "capabilities": {"can_orient_head": True},
            "neck_control": {
                "state": "tracking",
                "active_intent": {"intent": "face_centering", "source": "vision"},
                "desired_angle": 12.5,
                "last_angle": 10.0,
                "suppressed_reason": "",
                "last_command_status": {"status": "sent", "driver": "servo"},
                "intent_count": 3,
            },
            "organs": {},
        },
        cognitive_snapshot={},
        traces=[],
    )

    neck = report["neck_control_diagnostics"]

    assert neck["enabled"] is True
    assert neck["state"] == "tracking"
    assert neck["active_intent"] == "face_centering"
    assert neck["active_source"] == "vision"
    assert neck["desired_angle"] == 12.5
    assert neck["last_angle"] == 10.0
    assert neck["suppressed_reason"] == ""
    assert neck["last_command_status"]["status"] == "sent"
    assert neck["last_command_status_label"] == "sent"
    assert neck["intent_count"] == 3


def test_operator_console_neck_control_diagnostics_are_backward_compatible() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={"degradation_mode": "normal", "capabilities": {}, "organs": {}},
        cognitive_snapshot={},
        traces=[],
    )

    assert report["neck_control_diagnostics"]["enabled"] is False
    assert report["neck_control_diagnostics"]["state"] == "unavailable"
    assert report["neck_control_diagnostics"]["intent_count"] == 0
