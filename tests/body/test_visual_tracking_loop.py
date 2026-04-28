from __future__ import annotations

import time


def test_visual_tracking_loop_calls_runtime_until_stopped() -> None:
    from apps.body_runtime.visual_tracking_loop import VisualTrackingLoop

    class _Runtime:
        def __init__(self) -> None:
            self.calls = 0

        def track_visual_target_once(self, *, session_id: str, actor_id: str):
            self.calls += 1
            return None

    runtime = _Runtime()
    loop = VisualTrackingLoop(body_runtime=runtime, interval_s=0.05)
    loop.start()
    try:
        deadline = time.time() + 1.0
        while runtime.calls < 2 and time.time() < deadline:
            time.sleep(0.02)
    finally:
        loop.stop()

    calls_after_stop = runtime.calls
    time.sleep(0.08)

    assert calls_after_stop >= 2
    assert runtime.calls == calls_after_stop


def test_visual_tracking_loop_passes_state_source_when_configured() -> None:
    from apps.body_runtime.visual_tracking_loop import VisualTrackingLoop

    class _Runtime:
        def __init__(self) -> None:
            self.sources: list[str] = []

        def track_visual_target_once(self, *, session_id: str, actor_id: str, source: str):
            self.sources.append(source)
            return None

    runtime = _Runtime()
    loop = VisualTrackingLoop(body_runtime=runtime, interval_s=0.05, source="state")
    loop.start()
    try:
        deadline = time.time() + 1.0
        while not runtime.sources and time.time() < deadline:
            time.sleep(0.02)
    finally:
        loop.stop()

    assert runtime.sources
    assert set(runtime.sources) == {"state"}


def test_visual_tracking_loop_skips_tracking_while_sleeping(tmp_path) -> None:
    from apps.body_runtime.engagement_state import EngagementStateReader
    from apps.body_runtime.engagement_state import EngagementStateWriter
    from apps.body_runtime.visual_tracking_loop import VisualTrackingLoop

    class _Runtime:
        def __init__(self) -> None:
            self.calls = 0

        def track_visual_target_once(self, *, session_id: str, actor_id: str):
            self.calls += 1
            return None

    state_path = tmp_path / "engagement.json"
    EngagementStateWriter(state_path).write(conversation_active=False, phase="idle")
    runtime = _Runtime()
    loop = VisualTrackingLoop(
        body_runtime=runtime,
        interval_s=0.05,
        sleeping_interval_s=0.05,
        engagement_reader=EngagementStateReader(state_path),
    )
    loop.start()
    try:
        time.sleep(0.12)
    finally:
        loop.stop()

    assert runtime.calls == 0


def test_visual_tracking_loop_pauses_runtime_once_while_sleeping(tmp_path) -> None:
    from apps.body_runtime.engagement_state import EngagementStateReader
    from apps.body_runtime.engagement_state import EngagementStateWriter
    from apps.body_runtime.visual_tracking_loop import VisualTrackingLoop

    class _Runtime:
        def __init__(self) -> None:
            self.calls = 0
            self.pauses: list[str] = []

        def track_visual_target_once(self, *, session_id: str, actor_id: str):
            self.calls += 1
            return None

        def pause_visual_tracking(self, *, reason: str):
            self.pauses.append(reason)

    state_path = tmp_path / "engagement.json"
    EngagementStateWriter(state_path).write(conversation_active=False, phase="idle")
    runtime = _Runtime()
    loop = VisualTrackingLoop(
        body_runtime=runtime,
        interval_s=0.05,
        sleeping_interval_s=0.05,
        engagement_reader=EngagementStateReader(state_path),
    )
    loop.start()
    try:
        time.sleep(0.16)
    finally:
        loop.stop()

    assert runtime.calls == 0
    assert runtime.pauses == ["engagement_inactive"]
