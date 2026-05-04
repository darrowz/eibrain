from __future__ import annotations

from eihead.protocol import (
    ActionExecuted,
    MoveHeadAction,
    PlaySpeechAction,
    SpeechPlaybackCompleted,
    StopSpeechAction,
)
from eihead.runtime.app import HeadRuntimeApp


class _BodyRuntime:
    def __init__(self) -> None:
        self.dispatched: list[object] = []

    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "organ_count": 4,
            "capabilities": {"voice": True, "vision": True, "neck": True},
        }

    def dispatch_actions(self, actions: list[object]) -> list[object]:
        self.dispatched.extend(actions)
        action = actions[0]
        if isinstance(action, PlaySpeechAction):
            return [
                SpeechPlaybackCompleted(
                    ts=action.ts,
                    source="mouth.tts_playback",
                    status="ok",
                    session_id=action.session_id,
                    actor_id=action.actor_id,
                    target_id=action.target_id,
                )
            ]
        if isinstance(action, (MoveHeadAction, StopSpeechAction)):
            return [
                ActionExecuted(
                    ts=action.ts,
                    source="neck.motor" if isinstance(action, MoveHeadAction) else "mouth.tts_playback",
                    status="ok",
                    session_id=action.session_id,
                    actor_id=action.actor_id,
                    target_id=action.target_id,
                    action_kind=action.kind,
                    details={"target_angle": getattr(action, "target_angle", None)},
                )
            ]
        return []


class _FrameOnlyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-frame"}

    def latest_visual_frame_path(self) -> str:
        return "/tmp/honjia/latest.jpg"


class _SnapshotOnlyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-snapshot-only"}


class _LegacyBodyRuntime:
    def __init__(self) -> None:
        self.dispatched: list[object] = []

    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-legacy-body"}

    def dispatch_actions(self, actions: list[object]) -> list[object]:
        from eibrain.protocol.actions import MoveHeadAction as LegacyMoveHeadAction
        from eibrain.protocol.outcomes import ActionExecuted as LegacyActionExecuted

        self.dispatched.extend(actions)
        action = actions[0]
        if isinstance(action, LegacyMoveHeadAction):
            return [
                LegacyActionExecuted(
                    ts=action.ts,
                    source="legacy.neck",
                    status="ok",
                    action_kind=action.kind,
                    details={"target_angle": action.target_angle},
                )
            ]
        return []


def test_capabilities_returns_status_snapshot_shape_without_hardware() -> None:
    runtime = HeadRuntimeApp(body_runtime=_BodyRuntime(), config_path="config/test.yaml")

    payload = runtime.capabilities()

    assert payload["command"] == "capabilities"
    assert payload["runtime"] == "eihead"
    assert payload["node_id"] == "honjia-test"
    assert payload["body_runtime_node_id"] == "honjia-test"
    assert payload["summary"]["total"] == len(payload["capabilities"])
    assert payload["capabilities"]["neck"]["limits"]["tilt_deg"] is None


def test_handle_action_speak_delegates_to_body_runtime_dispatch() -> None:
    body_runtime = _BodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    outcome = runtime.handle_action(
        {"type": "speak", "text": "你好鸿途", "session_id": "s1", "actor_id": "darrow"},
        trace_id="trace-voice",
    )

    assert outcome["status"] == "accepted"
    assert outcome["success"] is True
    assert outcome["trace_id"] == "trace-voice"
    assert isinstance(body_runtime.dispatched[0], PlaySpeechAction)
    assert body_runtime.dispatched[0].text == "你好鸿途"
    assert outcome["details"]["delegate_outcomes"][0]["kind"] == "speech_playback_completed"


def test_handle_action_move_head_defaults_to_yaw_and_keeps_horizontal_only() -> None:
    body_runtime = _BodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    outcome = runtime.handle_action({"type": "move_head", "angle": 112, "target_name": "speaker"})

    assert outcome["status"] == "accepted"
    assert outcome["details"]["axis"] == "yaw"
    assert isinstance(body_runtime.dispatched[0], MoveHeadAction)
    assert body_runtime.dispatched[0].target_angle == 112
    assert body_runtime.dispatched[0].target_name == "speaker"


def test_handle_action_move_head_suppresses_jitter_within_min_angle_delta() -> None:
    body_runtime = _BodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime, ptz_min_angle_delta=4.0)

    first_outcome = runtime.handle_action({"type": "move_head", "angle": 112})
    second_outcome = runtime.handle_action({"type": "move_head", "angle": 115})
    third_outcome = runtime.handle_action({"type": "move_head", "angle": 119})

    assert first_outcome["status"] == "accepted"
    assert second_outcome["status"] == "skipped"
    assert second_outcome["details"]["reason"] == "ptz_jitter_suppressed"
    assert second_outcome["details"]["min_angle_delta"] == 4.0
    assert third_outcome["status"] == "accepted"
    assert len(body_runtime.dispatched) == 2


def test_handle_action_rejects_non_yaw_axis_without_dispatching() -> None:
    body_runtime = _BodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    outcome = runtime.handle_action({"type": "move_head", "axis": "pitch", "angle": 30})

    assert outcome["status"] == "unsupported"
    assert outcome["success"] is False
    assert outcome["details"]["axis"] == "pitch"
    assert body_runtime.dispatched == []


def test_handle_action_stop_speech_delegates_to_body_runtime_dispatch() -> None:
    body_runtime = _BodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    outcome = runtime.handle_action({"type": "stop_speech", "trace_id": "trace-stop"})

    assert outcome["status"] == "accepted"
    assert outcome["trace_id"] == "trace-stop"
    assert isinstance(body_runtime.dispatched[0], StopSpeechAction)
    assert outcome["details"]["delegate_outcomes"][0]["action_kind"] == "stop_speech_action"


def test_handle_action_capture_frame_uses_latest_visual_frame_path_fallback() -> None:
    runtime = HeadRuntimeApp(body_runtime=_FrameOnlyRuntime())

    outcome = runtime.handle_action({"type": "capture_frame"})

    assert outcome["status"] == "accepted"
    assert outcome["success"] is True
    assert outcome["delegated"] is True
    assert outcome["details"]["source"] == "latest_visual_frame_path"
    assert outcome["details"]["frame_path"] == "/tmp/honjia/latest.jpg"


def test_handle_action_without_dispatcher_returns_structured_skipped_outcome() -> None:
    runtime = HeadRuntimeApp(body_runtime=_SnapshotOnlyRuntime())

    outcome = runtime.handle_action({"type": "speak", "text": "hello"})

    assert outcome["status"] == "skipped"
    assert outcome["success"] is False
    assert outcome["details"]["reason"] == "body_runtime_dispatch_unavailable"


def test_handle_action_adapts_local_action_for_legacy_eibrain_body_delegate() -> None:
    body_runtime = _LegacyBodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    outcome = runtime.handle_action({"type": "move_head", "angle": 96})

    assert outcome["status"] == "accepted"
    assert outcome["success"] is True
    assert body_runtime.dispatched[0].__module__ == "eihead.protocol.actions"
    assert body_runtime.dispatched[1].__module__ == "eibrain.protocol.actions"
    assert outcome["details"]["protocol_action"] == "move_head_action"
    assert outcome["details"]["compat_protocol_action"] == "move_head_action"
    assert outcome["details"]["delegate_outcomes"][0]["source"] == "legacy.neck"
