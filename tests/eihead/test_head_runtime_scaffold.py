from __future__ import annotations

import io
import json
from pathlib import Path
import tomllib

from apps.head_runtime.app import HeadRuntimeApp as AppsHeadRuntimeApp
from eihead.runtime.app import HeadRuntimeApp
from eihead.runtime import cli


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


def make_fake_head_runtime(config_path: str) -> HeadRuntimeApp:
    return HeadRuntimeApp(body_runtime=FakeBodyRuntime(), config_path=config_path)


def test_head_runtime_imports_and_wraps_body_snapshot() -> None:
    runtime = make_fake_head_runtime("config/test.yaml")

    snapshot = runtime.snapshot()

    assert AppsHeadRuntimeApp is HeadRuntimeApp
    assert snapshot["runtime"] == "eihead"
    assert snapshot["node_role"] == "head"
    assert snapshot["delegate"] == "apps.body_runtime.BodyRuntimeApp"
    assert snapshot["body_runtime"]["node_id"] == "honjia-test"
    assert snapshot["body_runtime"]["organ_count"] == 4


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
