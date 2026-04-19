from __future__ import annotations


def test_operator_console_builds_runtime_status_report() -> None:
    from apps.operator_console.app import OperatorConsoleApp

    console = OperatorConsoleApp()
    report = console.build_status_report(
        body_snapshot={"degradation_mode": "normal", "capabilities": {"can_speak": True}},
        cognitive_snapshot={"last_reply": "hello", "learning_decision": "keep_policy"},
        traces=[{"trace_id": "t1", "kind": "audio_transcript_final"}],
    )

    assert report["system_health"] == "healthy"
    assert report["generated_at_ts"] > 0
    assert report["trace_count"] == 1
    assert report["degraded_organs"] == []
    assert report["body"]["degradation_mode"] == "normal"


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
