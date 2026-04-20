from __future__ import annotations


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
                                "elapsed_ms": 70.0,
                                "frame_path": "/tmp/latest.jpg",
                                "frame_captured_at_ts": 123.0,
                                "scene_summary": "1 face, 1 person",
                                "scene_labels": ["face", "person"],
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
