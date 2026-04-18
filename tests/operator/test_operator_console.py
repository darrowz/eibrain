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
    assert report["trace_count"] == 1
    assert report["body"]["degradation_mode"] == "normal"
