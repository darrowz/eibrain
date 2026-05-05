from __future__ import annotations

from pathlib import Path
from typing import Any

from eihead.runtime.app import HeadRuntimeApp
from eihead.runtime.native_providers import build_native_provider_statuses


class FakeBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-native-test", "organ_count": 4}


class FakeNativeProbe:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, provider_name: str, *, config: Any, environ: dict[str, str]) -> dict[str, object]:
        self.calls.append(provider_name)
        if provider_name == "eye":
            return {
                "status": "wired",
                "provider": environ["EIHEAD_TEST_EYE_PROVIDER"],
                "reason": "probe_reported_wired",
            }
        if provider_name == "ear":
            return {"status": "unknown", "reason": f"config_node={config.node_id}"}
        if provider_name == "mouth":
            return {"status": "unavailable", "reason": "tts_backend_disabled"}
        raise AssertionError(f"unexpected hardware probe for {provider_name}")


class FakeNeckAdapter:
    def apply_plan(self, plan: dict[str, object]) -> dict[str, object]:
        return {"status": "ok", "plan_status": plan.get("status")}


def test_from_config_path_reports_native_provider_boundaries_without_hardware(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    body_config_path = tmp_path / "eibrain.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia-native-test",
                "legacy:",
                f"  body_runtime_config_path: {body_config_path.as_posix()}",
                "native_providers:",
                "  eye:",
                "    enabled: true",
                "  ear:",
                "    enabled: true",
                "  mouth:",
                "    enabled: false",
                "  neck:",
                "    enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    captured: dict[str, str] = {}

    def fake_factory(path: str) -> FakeBodyRuntime:
        captured["body_config_path"] = path
        return FakeBodyRuntime()

    probe = FakeNativeProbe()

    runtime = HeadRuntimeApp.from_config_path(
        str(config_path),
        body_runtime_factory=fake_factory,
        native_provider_probe=probe,
        native_environ={"EIHEAD_TEST_EYE_PROVIDER": "fake-eye-adapter"},
    )

    snapshot = runtime.snapshot()
    native_providers = snapshot["native_providers"]

    assert captured["body_config_path"] == body_config_path.as_posix()
    assert snapshot["body_runtime"]["node_id"] == "honjia-native-test"
    assert native_providers["eye"] == {
        "status": "wired",
        "provider": "fake-eye-adapter",
        "reason": "probe_reported_wired",
    }
    assert native_providers["ear"]["status"] == "unknown"
    assert native_providers["ear"]["reason"] == "config_node=honjia-native-test"
    assert native_providers["mouth"]["status"] == "unavailable"
    assert native_providers["mouth"]["reason"] == "tts_backend_disabled"
    assert native_providers["neck"]["status"] == "unavailable"
    assert native_providers["neck"]["reason"] == "neck_servo_adapter_missing"
    assert "neck" not in probe.calls


def test_injected_neck_adapter_can_be_reported_as_wired_by_probe(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    config_path.write_text("node_id: honjia-native-test\n", encoding="utf-8")

    def fake_factory(path: str) -> FakeBodyRuntime:
        return FakeBodyRuntime()

    def probe(provider_name: str, *, config: Any, environ: dict[str, str]) -> dict[str, object]:
        if provider_name == "neck":
            return {"status": "wired", "provider": "fake-neck-servo", "reason": "adapter_injected"}
        return {"status": "unknown", "reason": "probe_not_configured"}

    runtime = HeadRuntimeApp.from_config_path(
        str(config_path),
        body_runtime_factory=fake_factory,
        native_provider_probe=probe,
        neck_servo_adapter=FakeNeckAdapter(),
    )

    assert runtime.snapshot()["native_providers"]["neck"] == {
        "status": "wired",
        "provider": "fake-neck-servo",
        "reason": "adapter_injected",
    }


def test_direct_runtime_construction_marks_uninjected_neck_unavailable() -> None:
    runtime = HeadRuntimeApp(body_runtime=FakeBodyRuntime(), config_path="config/test.yaml")

    native_providers = runtime.status()["native_providers"]

    assert native_providers["eye"]["status"] == "unknown"
    assert native_providers["ear"]["status"] == "unknown"
    assert native_providers["mouth"]["status"] == "unknown"
    assert native_providers["neck"] == {
        "status": "unavailable",
        "reason": "neck_servo_adapter_missing",
    }


def test_native_provider_probe_can_report_degraded_with_truthful_metadata() -> None:
    def probe(provider_name: str, *, config: Any, environ: dict[str, str]) -> dict[str, object]:
        if provider_name == "eye":
            return {
                "status": "degraded",
                "provider": "fake-eye-adapter",
                "source": "fake-native-live-probe",
                "checked_at": 5678.5,
                "reason": "low_frame_rate",
                "hardware_verified": True,
                "details": {"fps": 2},
            }
        return {"status": "unknown", "source": "fake-native-live-probe", "reason": "not_checked"}

    statuses = build_native_provider_statuses(
        config=None,
        environ={},
        probe=probe,
        neck_servo_adapter=FakeNeckAdapter(),
    )

    assert statuses["eye"] == {
        "status": "degraded",
        "provider": "fake-eye-adapter",
        "reason": "low_frame_rate",
        "source": "fake-native-live-probe",
        "checked_at": 5678.5,
        "last_checked": 5678.5,
        "hardware_verified": True,
        "details": {"fps": 2},
    }
