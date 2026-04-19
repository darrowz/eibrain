from __future__ import annotations


def test_body_runtime_loads_from_yaml_and_dispatches_actions(tmp_path) -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.protocol.actions import MoveHeadAction, PlaySpeechAction

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "system:",
                "  project_name: eibrain",
                "body:",
                "  node_id: honjia",
                "  organs:",
                "    ear:",
                "      enabled: true",
                "      capture:",
                "        driver:",
                "          kind: noop",
                "      vad:",
                "        driver:",
                "          kind: noop",
                "      asr:",
                "        driver:",
                "          kind: noop",
                "    eye:",
                "      enabled: true",
                "      camera:",
                "        driver:",
                "          kind: noop",
                "      detection:",
                "        driver:",
                "          kind: noop",
                "      identity:",
                "        driver:",
                "          kind: noop",
                "    mouth:",
                "      enabled: true",
                "      tts_plan:",
                "        driver:",
                "          kind: noop",
                "      tts_playback:",
                "        driver:",
                "          kind: noop",
                "    neck:",
                "      enabled: true",
                "      motor:",
                "        driver:",
                "          kind: noop",
                "      tracking:",
                "        driver:",
                "          kind: noop",
            ]
        ),
        encoding="utf-8",
    )

    runtime = BodyRuntimeApp.from_config_path(config_path)
    snapshot = runtime.snapshot()
    outcomes = runtime.dispatch_actions(
        [
            PlaySpeechAction(ts=1.0, source="test", text="hello", session_id="s1"),
            MoveHeadAction(ts=1.0, source="test", target_id="user-1", target_name="speaker"),
        ]
    )

    assert snapshot["organ_count"] == 4
    assert snapshot["degradation_mode"] == "mute_companion"
    assert [outcome.kind for outcome in outcomes] == [
        "speech_playback_completed",
        "action_executed",
    ]
