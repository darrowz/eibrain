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
