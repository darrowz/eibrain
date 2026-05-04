from __future__ import annotations

import io
import json
from pathlib import Path
import tomllib

from apps.head_runtime.app import HeadRuntimeApp as AppsHeadRuntimeApp
from eihead.runtime.app import HeadRuntimeApp
from eihead.runtime.legacy_body import LegacyBodyRuntimeAdapter
from eihead.runtime import cli
from eihead.protocol import VisionObservation


class FakeBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "organ_count": 4,
            "organs": {
                "ear": {"status": "mock"},
                "eye": {"status": "mock"},
            },
        }


class RealtimeVisionBodyRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "stream_id": "front-main",
            "status": "tracking",
        }


class StaticVisionBodyRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> VisionObservation:
        return VisionObservation(ts=1.0, source="eye.compat", frame_id="still-1")


def make_fake_head_runtime(config_path: str) -> HeadRuntimeApp:
    return HeadRuntimeApp(body_runtime=FakeBodyRuntime(), config_path=config_path)


def test_from_config_path_uses_legacy_body_config_for_eihead_config(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    body_config_path = tmp_path / "eibrain.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia-test",
                "legacy:",
                f"  body_runtime_config_path: {body_config_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    captured: dict[str, str] = {}

    def fake_factory(path: str) -> FakeBodyRuntime:
        captured["body_config_path"] = path
        return FakeBodyRuntime()

    runtime = HeadRuntimeApp.from_config_path(str(config_path), body_runtime_factory=fake_factory)

    assert runtime.config_path == str(config_path)
    assert captured["body_config_path"] == body_config_path.as_posix()


def test_from_config_path_keeps_eibrain_config_path_for_legacy_runtime() -> None:
    captured: dict[str, str] = {}

    def fake_factory(path: str) -> FakeBodyRuntime:
        captured["body_config_path"] = path
        return FakeBodyRuntime()

    HeadRuntimeApp.from_config_path("config/eibrain.honjia.yaml", body_runtime_factory=fake_factory)

    assert captured["body_config_path"] == "config/eibrain.honjia.yaml"


def test_legacy_body_adapter_loads_body_runtime_with_resolved_config(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    body_config_path = tmp_path / "eibrain.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia-test",
                "legacy:",
                f"  body_runtime_config_path: {body_config_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    captured: dict[str, str] = {}

    def fake_factory(path: str) -> FakeBodyRuntime:
        captured["body_config_path"] = path
        return FakeBodyRuntime()

    adapter = LegacyBodyRuntimeAdapter(body_runtime_factory=fake_factory)
    body_runtime = adapter.load_runtime(str(config_path))

    assert isinstance(body_runtime, FakeBodyRuntime)
    assert captured["body_config_path"] == body_config_path.as_posix()


def test_head_runtime_facade_does_not_embed_legacy_body_imports() -> None:
    app_source = Path("eihead/runtime/app.py").read_text(encoding="utf-8")
    cli_source = Path("eihead/runtime/cli.py").read_text(encoding="utf-8")

    assert "from apps.body_runtime" not in app_source
    assert "from apps.body_runtime" not in cli_source
    assert "from eibrain.protocol.actions" not in app_source
    assert "def _legacy_eibrain_action" not in app_source


def test_head_runtime_imports_and_wraps_body_snapshot() -> None:
    runtime = make_fake_head_runtime("config/test.yaml")

    snapshot = runtime.snapshot()

    assert AppsHeadRuntimeApp is HeadRuntimeApp
    assert snapshot["runtime"] == "eihead"
    assert snapshot["node_role"] == "head"
    assert snapshot["delegate"] == "apps.body_runtime.BodyRuntimeApp"
    assert snapshot["body_runtime"]["node_id"] == "honjia-test"
    assert snapshot["body_runtime"]["organ_count"] == 4


def test_head_runtime_realtime_vision_hook_is_explicit_and_does_not_fake_static_frames() -> None:
    runtime = make_fake_head_runtime("config/test.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_realtime_vision_hook_delegates_only_when_runtime_exposes_it() -> None:
    runtime = HeadRuntimeApp(body_runtime=RealtimeVisionBodyRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() == {
        "kind": "realtime_vision_observation",
        "mode": "realtime",
        "stream_id": "front-main",
        "status": "tracking",
    }


def test_head_runtime_realtime_vision_hook_rejects_static_compat_observation() -> None:
    runtime = HeadRuntimeApp(body_runtime=StaticVisionBodyRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def test_cli_status_uses_injected_runtime_without_hardware() -> None:
    stdout = io.StringIO()

    exit_code = cli.main(
        ["--config", "config/test.yaml", "status"],
        app_factory=make_fake_head_runtime,
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["command"] == "status"
    assert payload["runtime"] == "eihead"
    assert payload["config_path"] == "config/test.yaml"
    assert payload["body_runtime"]["node_id"] == "honjia-test"


def test_cli_serve_and_verify_dispatch_without_hardware() -> None:
    serve_payload = cli.dispatch(
        cli.build_parser().parse_args(["serve"]),
        app_factory=make_fake_head_runtime,
    )
    verify_payload = cli.dispatch(
        cli.build_parser().parse_args(["verify"]),
        app_factory=make_fake_head_runtime,
    )

    assert serve_payload["command"] == "serve"
    assert serve_payload["serve_mode"] == "compatibility_snapshot"
    assert verify_payload["command"] == "verify"
    assert verify_payload["checks"]["head_runtime_import"] == "ok"
    assert verify_payload["organ_count"] == 4


def test_verify_hardware_script_delegates_to_body_verifier(monkeypatch) -> None:
    called = {}

    def fake_verify() -> None:
        called["body_verify"] = True

    monkeypatch.setattr(cli, "_run_body_hardware_verifier", fake_verify)

    cli.verify_hardware_main()

    assert called == {"body_verify": True}


def test_pyproject_exposes_eihead_packages_and_scripts() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["project"]["scripts"]
    include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]

    assert "eibrain*" in include
    assert "apps*" in include
    assert "eihead*" in include
    assert scripts["eihead-runtime"] == "eihead.runtime.cli:main"
    assert scripts["eihead-verify-hardware"] == "eihead.runtime.cli:verify_hardware_main"
