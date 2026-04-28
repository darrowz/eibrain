from __future__ import annotations


def test_body_runtime_reports_capabilities_from_registered_organs() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    snapshot = runtime.snapshot()

    assert snapshot["organ_count"] == 4
    assert snapshot["degradation_mode"] == "mute_companion"
    assert snapshot["capabilities"]["can_hear_voice"] is False
    assert snapshot["capabilities"]["can_speak"] is False


def test_body_runtime_can_transcribe_audio_window() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    class _Capture:
        sample_rate = 16000
        channels = 1

        def read_chunks(self, chunk_count: int):
            return [b"a"] * chunk_count

    class _Recognizer:
        def transcribe(self, pcm_chunks, *, sample_rate: int, channels: int) -> str:
            return "streamed text"

    runtime = BodyRuntimeApp()
    runtime.ear_processor = runtime._build_ear_processor(capture=_Capture(), recognizer=_Recognizer())

    observation = runtime.transcribe_audio_window(
        chunk_count=2,
        session_id="session-1",
        actor_id="user-1",
    )

    assert observation.text == "streamed text"


def test_body_runtime_transcribes_from_ear_organ_heartbeat() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    runtime = BodyRuntimeApp()

    class _Ear:
        name = "ear"
        _chunk_count = 1
        _cached_heartbeat = None

        def heartbeat(self):
            return OrganHealth(
                organ="ear",
                health="healthy",
                subfunctions={
                    "asr": SubfunctionHealth(
                        name="asr",
                        health="healthy",
                        details={"transcript": "你好 honjia", "speech_window_summary": "heard speech"},
                    )
                },
            )

    runtime.organs = [_Ear()]

    observation = runtime.transcribe_audio_window(
        chunk_count=3,
        session_id="session-2",
        actor_id="user-2",
    )

    assert observation.text == "你好 honjia"
    assert runtime.recent_events()[-1]["status"] == "ok"


def test_body_runtime_maps_visual_target_to_move_head_action() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    runtime = BodyRuntimeApp()
    action = runtime.plan_visual_tracking_action(
        target_name="person",
        target_x=0.75,
        session_id="session-1",
        actor_id="user-1",
    )

    assert action.target_x == 0.75


def test_body_runtime_can_dispatch_visual_tracking_to_neck() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    runtime = BodyRuntimeApp()

    class _Eye:
        name = "eye"

        def supports_action(self, action) -> bool:
            return False

        def heartbeat(self):
            return OrganHealth(
                organ="eye",
                health="healthy",
                subfunctions={
                    "detection": SubfunctionHealth(
                        name="detection",
                        health="healthy",
                        details={
                            "detections": [
                                {
                                    "label": "person",
                                    "score": 0.6,
                                    "bbox": {"x_min": 0.0, "x_max": 0.4},
                                },
                                {
                                    "label": "face",
                                    "score": 0.8,
                                    "bbox": {"x_min": 0.6, "x_max": 0.8},
                                },
                            ]
                        },
                    )
                },
            )

    class _Neck:
        name = "neck"

        def supports_action(self, action) -> bool:
            return True

        def heartbeat(self):
            return OrganHealth(organ="neck", health="healthy", subfunctions={})

        def handle_action(self, action):
            from eibrain.protocol.outcomes import ActionExecuted

            return ActionExecuted(
                ts=action.ts,
                source="neck.motor",
                status="ok",
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                action_kind=action.kind,
                details={"target_x": action.target_x, "target_name": action.target_name},
            )

    runtime.organs = [_Eye(), _Neck()]

    outcome = runtime.track_visual_target_once(session_id="track-1", actor_id="vision-1")

    assert outcome is not None
    assert outcome.details["target_name"] == "face"
    assert outcome.details["target_x"] == 0.7


def test_body_runtime_state_tracking_reads_state_without_identity_probe(tmp_path) -> None:
    import json
    import time

    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.organs.eye.organ import EyeOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    frame_path = tmp_path / "latest.jpg"
    state_path = tmp_path / "state.json"
    frame_path.write_bytes(b"frame")
    state_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "backend": "gstreamer_hailo",
                "updated_at_ts": time.time(),
                "frame_path": str(frame_path),
                "frame_captured_at_ts": 50.0,
                "detections": [
                    {
                        "label": "face",
                        "score": 0.88,
                        "bbox": {"x_min": 0.6, "y_min": 0.1, "x_max": 0.8, "y_max": 0.5},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    eye = EyeOrgan(
        config=OrganConfig(
            enabled=True,
            subfunctions={
                "camera": SubfunctionConfig(
                    driver=DriverConfig(kind="command", command=["python"], extra={"provider": "vision_state", "state_path": str(state_path)}),
                ),
                "detection": SubfunctionConfig(
                    driver=DriverConfig(kind="command", command=["python"], extra={"provider": "vision_state", "state_path": str(state_path)}),
                ),
                "identity": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"])),
            },
        )
    )
    eye.drivers["identity"].heartbeat = lambda: (_ for _ in ()).throw(AssertionError("identity probe must not run"))

    class _Neck:
        name = "neck"

        def supports_action(self, action) -> bool:
            return True

        def heartbeat(self):
            from eibrain.body.health.organ_health import OrganHealth

            return OrganHealth(organ="neck", health="healthy", subfunctions={})

        def handle_action(self, action):
            from eibrain.protocol.outcomes import ActionExecuted

            return ActionExecuted(
                ts=action.ts,
                source="neck.motor",
                status="ok",
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                action_kind=action.kind,
                details={"target_x": action.target_x, "target_name": action.target_name},
            )

    runtime = BodyRuntimeApp()
    runtime.organs = [eye, _Neck()]

    outcome = runtime.track_visual_target_once(source="state")

    assert outcome is not None
    assert outcome.details["target_name"] == "face"
    assert outcome.details["target_x"] == 0.7
    assert runtime.visual_tracking_state["source"] == "state"


def test_body_runtime_updates_interaction_state_when_visual_target_locks() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    runtime = BodyRuntimeApp()

    class _Eye:
        name = "eye"

        def supports_action(self, action) -> bool:
            return False

        def heartbeat(self):
            return OrganHealth(
                organ="eye",
                health="healthy",
                subfunctions={
                    "detection": SubfunctionHealth(
                        name="detection",
                        health="healthy",
                        details={
                            "frame_captured_at_ts": 10.0,
                            "top_detection": {"label": "face", "score": 0.92},
                            "detections": [
                                {
                                    "label": "face",
                                    "score": 0.92,
                                    "bbox": {"x_min": 0.55, "x_max": 0.75},
                                }
                            ],
                        },
                    )
                },
            )

    class _Neck:
        name = "neck"

        def supports_action(self, action) -> bool:
            return True

        def heartbeat(self):
            return OrganHealth(organ="neck", health="healthy", subfunctions={})

        def handle_action(self, action):
            from eibrain.protocol.outcomes import ActionExecuted

            return ActionExecuted(
                ts=action.ts,
                source="neck.motor",
                status="ok",
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                action_kind=action.kind,
                details={"target_x": action.target_x, "target_name": action.target_name},
            )

    runtime.organs = [_Eye(), _Neck()]

    outcome = runtime.track_visual_target_once(session_id="track-2", actor_id="vision-2")

    assert outcome is not None
    assert runtime.interaction_state["current_mode"] == "attention"
    assert runtime.interaction_state["tracking_locked"] is True
    assert runtime.interaction_state["tracking_target_label"] == "face"
    assert runtime.snapshot()["interaction_state"]["current_mode"] == "attention"


def test_body_runtime_holds_neck_command_when_target_shift_is_small() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    runtime = BodyRuntimeApp()

    class _Eye:
        name = "eye"

        def __init__(self) -> None:
            self._detections = [
                {
                    "label": "face",
                    "score": 0.91,
                    "bbox": {"x_min": 0.50, "x_max": 0.70},
                },
                {
                    "label": "face",
                    "score": 0.90,
                    "bbox": {"x_min": 0.52, "x_max": 0.72},
                },
            ]
            self._index = 0

        def supports_action(self, action) -> bool:
            return False

        def heartbeat(self):
            detection = self._detections[min(self._index, len(self._detections) - 1)]
            self._index += 1
            return OrganHealth(
                organ="eye",
                health="healthy",
                subfunctions={
                    "detection": SubfunctionHealth(
                        name="detection",
                        health="healthy",
                        details={
                            "frame_captured_at_ts": 20.0,
                            "top_detection": detection,
                            "detections": [detection],
                        },
                    )
                },
            )

    class _Neck:
        name = "neck"

        def __init__(self) -> None:
            self.actions = []

        def supports_action(self, action) -> bool:
            return True

        def heartbeat(self):
            return OrganHealth(organ="neck", health="healthy", subfunctions={})

        def handle_action(self, action):
            from eibrain.protocol.outcomes import ActionExecuted

            self.actions.append(action)
            return ActionExecuted(
                ts=action.ts,
                source="neck.motor",
                status="ok",
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                action_kind=action.kind,
                details={"target_x": action.target_x},
            )

    eye = _Eye()
    neck = _Neck()
    runtime.organs = [eye, neck]

    first = runtime.track_visual_target_once(session_id="track-3", actor_id="vision-3")
    second = runtime.track_visual_target_once(session_id="track-4", actor_id="vision-4")

    assert first is not None
    assert second is None
    assert len(neck.actions) == 1
    assert runtime.interaction_state["current_mode"] == "attention"
    assert runtime.visual_tracking_state["status"] == "holding_target"
    assert runtime.snapshot()["visual_tracking"]["target"]["label"] == "face"


def test_body_runtime_ignores_non_preferred_visual_tracking_labels() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    runtime = BodyRuntimeApp()

    class _Eye:
        name = "eye"

        def supports_action(self, action) -> bool:
            return False

        def heartbeat(self):
            return OrganHealth(
                organ="eye",
                health="healthy",
                subfunctions={
                    "detection": SubfunctionHealth(
                        name="detection",
                        health="healthy",
                        details={
                            "frame_captured_at_ts": 123.0,
                            "detections": [
                                {
                                    "label": "potted plant",
                                    "score": 0.9,
                                    "bbox": {"x_min": 0.0, "x_max": 0.2},
                                }
                            ],
                            "top_detection": {"label": "potted plant"},
                        },
                    )
                },
            )

    runtime.organs = [_Eye()]

    outcome = runtime.track_visual_target_once(recenter_after_misses=5)

    assert outcome is None
    assert runtime.visual_tracking_state["status"] == "waiting_for_target"
    assert runtime.visual_tracking_state["target"] is None


def test_body_runtime_marks_visual_tracking_waiting_when_no_target() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    runtime = BodyRuntimeApp()

    class _Eye:
        name = "eye"

        def supports_action(self, action) -> bool:
            return False

        def heartbeat(self):
            return OrganHealth(
                organ="eye",
                health="healthy",
                subfunctions={
                    "detection": SubfunctionHealth(
                        name="detection",
                        health="healthy",
                        details={
                            "frame_captured_at_ts": 123.0,
                            "detection_count": 0,
                            "detections": [],
                            "top_detection": None,
                        },
                    )
                },
            )

    runtime.organs = [_Eye()]

    outcome = runtime.track_visual_target_once(recenter_after_misses=5)
    tracking = runtime.snapshot()["visual_tracking"]

    assert outcome is None
    assert tracking["status"] == "waiting_for_target"
    assert tracking["detection_count"] == 0
    assert tracking["miss_count"] == 1


def test_body_runtime_recenters_once_per_lost_visual_target_episode() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    runtime = BodyRuntimeApp()

    class _Eye:
        name = "eye"

        def supports_action(self, action) -> bool:
            return False

        def heartbeat(self):
            return OrganHealth(
                organ="eye",
                health="healthy",
                subfunctions={
                    "detection": SubfunctionHealth(
                        name="detection",
                        health="healthy",
                        details={"detections": [], "detection_count": 0, "top_detection": None},
                    )
                },
            )

    class _Neck:
        name = "neck"

        def __init__(self) -> None:
            self.actions = []

        def supports_action(self, action) -> bool:
            return True

        def heartbeat(self):
            return OrganHealth(organ="neck", health="healthy", subfunctions={})

        def handle_action(self, action):
            from eibrain.protocol.outcomes import ActionExecuted

            self.actions.append(action)
            return ActionExecuted(
                ts=action.ts,
                source="neck.motor",
                status="ok",
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                action_kind=action.kind,
                details={"target_name": action.target_name, "target_angle": action.target_angle},
            )

    neck = _Neck()
    runtime.organs = [_Eye(), neck]

    for _ in range(6):
        runtime.track_visual_target_once(recenter_after_misses=3)

    assert len(neck.actions) == 1
    assert neck.actions[0].target_name == "recenter"
    assert runtime.visual_tracking_state["status"] == "waiting_for_target"
    assert runtime.visual_tracking_state["miss_count"] == 6
