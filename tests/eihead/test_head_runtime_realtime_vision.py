from __future__ import annotations

import time
from typing import Any

import pytest

from eihead.protocol import RealtimeVisionObservation, VisionObservation
from eihead.runtime.app import HeadRuntimeApp


class FakeBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-test"}


class RealtimeObservationRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> RealtimeVisionObservation:
        return RealtimeVisionObservation(
            ts=1.0,
            source="eihead.honjia.eye.realtime",
            stream_id="front-main",
            status="tracking",
            frame_id="live-1",
        )


class MappingRealtimeRuntime(FakeBodyRuntime):
    eye_realtime = {
        "kind": "realtime_vision_observation",
        "mode": "realtime",
        "primary_mode": True,
        "stream_id": "front-main",
        "status": "tracking",
    }


class ToDictRealtimePayload:
    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "primary_mode": True,
            "stream_id": "front-main",
            "status": "tracking",
        }


class ToDictRealtimeRuntime(FakeBodyRuntime):
    def latest_realtime_vision(self) -> ToDictRealtimePayload:
        return ToDictRealtimePayload()


class LegacyVisionStateRuntime(FakeBodyRuntime):
    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "vision_state": {
                "schema": "eibrain.vision_state.v2",
                "frame_path": "/tmp/eibrain-vision/latest.jpg",
                "status": "tracking",
                "detections": [{"label": "person", "score": 0.9}],
            },
        }


class StaticVisionObservationRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> VisionObservation:
        return VisionObservation(ts=2.0, source="eihead.honjia.eye.compat", frame_id="still-1")


class MultiHookRuntime(FakeBodyRuntime):
    eye_realtime = {
        "kind": "realtime_vision_observation",
        "mode": "realtime_stream",
        "status": "not_wired",
        "not_wired": True,
    }

    def vision_realtime(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "primary_mode": True,
            "stream_id": "front-main",
            "status": "tracking",
        }


class ExpiredRealtimePayloadRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "status": "tracking",
            "timestamp": 100.0,
            "stream_id": "front-main",
        }


class MillisecondRealtimePayloadRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "status": "tracking",
            "timestamp": 1_714_800_001_234.0,
            "stream_id": "front-main",
            "not_wired": "false",
        }


LIVE_ADAPTER_STATUS = {
    "schema": "eihead.eye.realtime_status.v1",
    "mode": "realtime_stream",
    "status": "tracking",
    "backend": "gstreamer_hailo",
    "frame_count": 8,
    "detection_count": 2,
    "fps": 28.0,
    "last_frame_id": "frame-88",
    "placeholder": "false",
    "not_wired": "false",
    "compatibility_mode": "false",
    "message": "live adapter status",
}


class AdapterPayload:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = dict(payload)

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)


class LatestStatusAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        latest_status = AdapterPayload(LIVE_ADAPTER_STATUS)

    eye_realtime = Adapter()


class StatusMethodAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        def status(self) -> AdapterPayload:
            return AdapterPayload(LIVE_ADAPTER_STATUS)

    eye_realtime = Adapter()


class PollMethodAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        def poll(self) -> AdapterPayload:
            return AdapterPayload(LIVE_ADAPTER_STATUS)

    eye_realtime = Adapter()


class ToDictAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        def to_dict(self) -> dict[str, object]:
            return dict(LIVE_ADAPTER_STATUS)

    eye_realtime = Adapter()


class NotWiredAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        latest_status = AdapterPayload(
            {
                **LIVE_ADAPTER_STATUS,
                "status": "not_wired",
                "placeholder": "true",
                "not_wired": "true",
            }
        )

    eye_realtime = Adapter()


class CompatStaticAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        latest_status = AdapterPayload(
            {
                **LIVE_ADAPTER_STATUS,
                "mode": "compat/static",
                "compatibility_mode": "true",
            }
        )

    eye_realtime = Adapter()


@pytest.mark.parametrize(
    ("payload", "case_name"),
    [
        (
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime_stream",
                "status": "not_wired",
                "not_wired": True,
            },
            "not_wired",
        ),
        (
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime_stream",
                "status": "tracking",
                "placeholder": True,
            },
            "placeholder",
        ),
        (
            {
                "kind": "realtime_vision_observation",
                "mode": "compat/static",
                "status": "tracking",
            },
            "compat_static_mode",
        ),
        (
            {
                "kind": "vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "primary_mode": False,
            },
            "vision_observation",
        ),
        (
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "compatibility_mode": True,
            },
            "compatibility_mode",
        ),
    ],
)
def test_head_runtime_rejects_non_live_realtime_vision_payloads(
    payload: dict[str, Any],
    case_name: str,
) -> None:
    class Runtime(FakeBodyRuntime):
        def vision_realtime(self) -> dict[str, Any]:
            return payload

    runtime = HeadRuntimeApp(body_runtime=Runtime(), config_path=f"config/{case_name}.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_passes_through_realtime_vision_observation() -> None:
    runtime = HeadRuntimeApp(body_runtime=RealtimeObservationRuntime(), config_path="config/test.yaml")

    observation = runtime.vision_realtime()

    assert isinstance(observation, RealtimeVisionObservation)
    assert observation.stream_id == "front-main"


def test_head_runtime_passes_through_realtime_mapping() -> None:
    runtime = HeadRuntimeApp(body_runtime=MappingRealtimeRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() == MappingRealtimeRuntime.eye_realtime


def test_head_runtime_passes_through_realtime_to_dict_payload() -> None:
    runtime = HeadRuntimeApp(body_runtime=ToDictRealtimeRuntime(), config_path="config/test.yaml")

    observation = runtime.vision_realtime()

    assert isinstance(observation, ToDictRealtimePayload)
    assert observation.to_dict()["stream_id"] == "front-main"


def test_head_runtime_does_not_promote_legacy_vision_state_snapshot_to_realtime() -> None:
    runtime = HeadRuntimeApp(body_runtime=LegacyVisionStateRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_rejects_static_vision_observation() -> None:
    runtime = HeadRuntimeApp(body_runtime=StaticVisionObservationRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_uses_later_live_hook_when_earlier_hook_is_not_live() -> None:
    runtime = HeadRuntimeApp(body_runtime=MultiHookRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() == {
        "kind": "realtime_vision_observation",
        "mode": "realtime",
        "primary_mode": True,
        "stream_id": "front-main",
        "status": "tracking",
    }


@pytest.mark.parametrize(
    "runtime_cls",
    [
        LatestStatusAdapterRuntime,
        StatusMethodAdapterRuntime,
        PollMethodAdapterRuntime,
        ToDictAdapterRuntime,
    ],
)
def test_head_runtime_accepts_live_adapter_payload_forms(runtime_cls: type[FakeBodyRuntime]) -> None:
    runtime = HeadRuntimeApp(body_runtime=runtime_cls(), config_path="config/test.yaml")

    payload = runtime.vision_realtime()

    assert _payload_dict(payload) == LIVE_ADAPTER_STATUS


def test_head_runtime_rejects_not_wired_adapter_status_with_string_booleans() -> None:
    runtime = HeadRuntimeApp(body_runtime=NotWiredAdapterRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_rejects_stale_realtime_vision_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = HeadRuntimeApp(
        body_runtime=ExpiredRealtimePayloadRuntime(),
        config_path="config/test.yaml",
        realtime_vision_max_age_seconds=2.0,
    )
    _monotonic_timestamp = 120.0

    # Validate freshness window boundary explicitly.
    monkeypatch.setattr(time, "time", lambda: _monotonic_timestamp)

    assert runtime.vision_realtime() is None


def test_head_runtime_accepts_ms_realtime_timestamp_without_explicit_ms_field(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = HeadRuntimeApp(
        body_runtime=MillisecondRealtimePayloadRuntime(),
        config_path="config/test.yaml",
        realtime_vision_max_age_seconds=1200.0,
    )

    _monotonic_timestamp = 1_714_800_002.1
    monkeypatch.setattr(time, "time", lambda: _monotonic_timestamp)

    payload = runtime.vision_realtime()

    assert isinstance(payload, dict)
    assert payload["timestamp"] == 1_714_800_001_234.0


def test_head_runtime_rejects_compat_static_adapter_status() -> None:
    runtime = HeadRuntimeApp(body_runtime=CompatStaticAdapterRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def _payload_dict(payload: Any) -> dict[str, object]:
    assert payload is not None
    if isinstance(payload, dict):
        return payload
    assert hasattr(payload, "to_dict")
    return payload.to_dict()
