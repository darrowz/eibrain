from __future__ import annotations


def test_body_state_manager_aggregates_health_capabilities_degradation_and_fallback() -> None:
    from eibrain.body.health import OrganHealth, SubfunctionHealth
    from eibrain.body.state import BodyStateManager

    manager = BodyStateManager(node_id="honjia-test", clock=lambda: 123.4)
    snapshot = manager.snapshot(
        [
            OrganHealth(
                organ="ear",
                health="healthy",
                subfunctions={
                    "capture": SubfunctionHealth(
                        name="capture",
                        health="healthy",
                        details={"driver": "command"},
                    ),
                    "asr": SubfunctionHealth(
                        name="asr",
                        health="healthy",
                        details={"driver": "command"},
                    ),
                },
            ),
            OrganHealth(
                organ="mouth",
                health="unavailable",
                subfunctions={
                    "tts_playback": SubfunctionHealth(
                        name="tts_playback",
                        health="unavailable",
                        details={"driver": "command"},
                    )
                },
            ),
        ],
    )

    assert snapshot["schema"] == "eibrain.body_state.v1"
    assert snapshot["node_id"] == "honjia-test"
    assert snapshot["updated_at_ts"] == 123.4
    assert snapshot["organ_count"] == 2
    assert snapshot["capabilities"]["can_hear_voice"] is True
    assert snapshot["capabilities"]["can_speak"] is False
    assert snapshot["degradation_mode"] == "mute_companion"
    assert snapshot["fallback_policy"]["mode"] == "mute_companion"
    assert "speech.play" in snapshot["fallback_policy"]["disabled_actions"]


def test_body_state_manager_preserves_runtime_sections_and_recent_events() -> None:
    from eibrain.body.state import BodyStateManager

    manager = BodyStateManager(node_id="honjia-test", clock=lambda: 222.0)
    snapshot = manager.snapshot(
        [],
        recent_events=[{"kind": "voice", "status": "ok"}],
        runtime={"voice_dialogue": {"phase": "idle"}},
    )

    assert snapshot["recent_event_count"] == 1
    assert snapshot["recent_events"] == [{"kind": "voice", "status": "ok"}]
    assert snapshot["runtime"] == {"voice_dialogue": {"phase": "idle"}}


def test_body_runtime_snapshot_exposes_structured_body_state_and_fallback_policy() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    snapshot = runtime.snapshot()

    assert snapshot["body_state"]["schema"] == "eibrain.body_state.v1"
    assert snapshot["body_state"]["fallback_policy"]["mode"] == "mute_companion"
    assert snapshot["fallback_policy"]["mode"] == "mute_companion"
    assert "speech.play" in snapshot["fallback_policy"]["disabled_actions"]
