from __future__ import annotations

import sys


def test_body_runtime_snapshot_includes_subfunction_diagnostics(tmp_path) -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    script_path = tmp_path / "health_probe.py"
    script_path.write_text(
        "\n".join(
            [
                "import json",
                "print(json.dumps({'device': '/dev/video0', 'status': 'healthy', 'driver': 'camera_probe'}))",
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "body:",
                "  node_id: honjia",
                "  organs:",
                "    ear:",
                "      enabled: true",
                "      capture:",
                "        driver:",
                "          kind: command",
                "          command: python",
                "          health_command:",
                f"            - {sys.executable}",
                f"            - {script_path}",
                "      vad:",
                "        driver:",
                "          kind: noop",
                "      asr:",
                "        driver:",
                "          kind: noop",
            ]
        ),
        encoding="utf-8",
    )

    runtime = BodyRuntimeApp.from_config_path(config_path)
    snapshot = runtime.snapshot()
    capture = snapshot["organs"]["ear"]["subfunctions"]["capture"]

    assert capture["health"] == "healthy"
    assert capture["details"]["device"] == "/dev/video0"
    assert capture["details"]["driver"] == "camera_probe"
    assert "elapsed_ms" in capture["details"]


def test_body_runtime_records_recent_events_for_monitoring() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.protocol.actions import PlaySpeechAction

    runtime = BodyRuntimeApp()
    runtime.dispatch_actions(
        [PlaySpeechAction(ts=1.0, source="test", text="hello", session_id="s1", actor_id="user-1")]
    )

    events = runtime.recent_events()

    assert events
    assert events[-1]["kind"] == "speech_playback_completed"
    assert "recorded_at_ts" in events[-1]


def test_body_runtime_marks_command_action_error_from_json_status(tmp_path) -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.protocol.actions import MoveHeadAction

    script_path = tmp_path / "gimbal_fail.py"
    script_path.write_text(
        "\n".join(
            [
                "import json, sys",
                "sys.stdin.read()",
                "print(json.dumps({'status': 'error', 'details': {'error': 'missing smbus2'}}))",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "body:",
                "  organs:",
                "    neck:",
                "      enabled: true",
                "      motor:",
                "        driver:",
                "          kind: command",
                "          command:",
                f"            - {sys.executable}",
                f"            - {script_path}",
                "          health_command:",
                f"            - {sys.executable}",
                f"            - {script_path}",
                "      tracking:",
                "        driver:",
                "          kind: noop",
            ]
        ),
        encoding="utf-8",
    )

    runtime = BodyRuntimeApp.from_config_path(config_path)
    outcomes = runtime.dispatch_actions(
        [MoveHeadAction(ts=1.0, source="test", target_id="user-1", target_name="speaker", session_id="s1")]
    )

    assert outcomes[0].status == "error"
