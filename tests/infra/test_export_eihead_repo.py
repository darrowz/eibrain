from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export-eihead-repo.py"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_eihead_repo", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_export_creates_required_standalone_layout(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"

    result = module.export_eihead_repo(target, repo_root=REPO_ROOT)

    assert result.target == target.resolve()
    required_paths = [
        ".gitignore",
        "eihead/runtime/cli.py",
        "eihead/runtime/http_api.py",
        "eihead/services/capability_registry.py",
        "eihead/runtime/app.py",
        "apps/head_runtime/__main__.py",
        "apps/body_runtime/app.py",
        "config/eibrain.honjia.yaml",
        "config/eihead.honjia.yaml",
        "deploy/systemd/eihead-runtime.service",
        "deploy/systemd/eihead-monitor.service",
        "eibrain/body/runtime_linux.py",
        "eibrain/cognition/__init__.py",
        "eibrain/cognition/realtime/__init__.py",
        "eibrain/cognition/realtime/turn.py",
        "eibrain/infra/config.py",
        "eibrain/verification/__init__.py",
        "eibrain/verification/body_checks.py",
        "eiprotocol/__init__.py",
        "eiprotocol/models.py",
        "eihead/ear/__init__.py",
        "eihead/ear/realtime.py",
        "eihead/mouth/__init__.py",
        "eihead/mouth/playback.py",
        "eihead/monitoring/voice.py",
        "eihead/monitoring/web.py",
        "docs/eihead-implementation-plan.md",
        "docs/eihead-deployment-plan.md",
        "docs/eihead-migration-audit.md",
        "EXPORT_MANIFEST.json",
        "pyproject.toml",
        "README.md",
    ]
    for rel_path in required_paths:
        assert (target / rel_path).is_file(), rel_path

    assert "eihead/runtime/cli.py" in result.copied
    assert "apps/head_runtime/__main__.py" in result.copied
    assert "apps/body_runtime/app.py" in result.copied
    assert "config/eibrain.honjia.yaml" in result.copied
    assert "config/eihead.honjia.yaml" in result.copied
    assert "eibrain/body/runtime_linux.py" in result.copied
    assert "eibrain/cognition/__init__.py" in result.copied
    assert "eibrain/cognition/realtime/__init__.py" in result.copied
    assert "eibrain/cognition/realtime/turn.py" in result.copied
    assert "eibrain/infra/config.py" in result.copied
    assert "eibrain/verification/__init__.py" in result.copied
    assert "eibrain/verification/body_checks.py" in result.copied
    assert "eiprotocol/__init__.py" in result.copied
    assert "eiprotocol/models.py" in result.copied
    assert "eihead/runtime/app.py" in result.copied
    assert "eihead/ear/__init__.py" in result.copied
    assert "eihead/ear/realtime.py" in result.copied
    assert "eihead/mouth/__init__.py" in result.copied
    assert "eihead/mouth/playback.py" in result.copied
    assert "eihead/monitoring/voice.py" in result.copied
    assert "eihead/monitoring/web.py" in result.copied
    assert "pyproject.toml" in result.generated
    assert "README.md" in result.generated
    assert ".gitignore" in result.generated
    assert "eibrain/protocol/__init__.py" in result.generated


def test_export_writes_machine_readable_manifest_for_honxin_sync(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"

    result = module.export_eihead_repo(target, repo_root=REPO_ROOT)
    manifest = json.loads((target / "EXPORT_MANIFEST.json").read_text(encoding="utf-8"))
    source_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert "EXPORT_MANIFEST.json" in result.generated
    assert result.manifest == "EXPORT_MANIFEST.json"
    assert manifest["schema_version"] == 1
    assert manifest["source"]["repo_root"] == str(REPO_ROOT)
    assert manifest["source"]["commit"] == source_commit
    assert isinstance(manifest["source"]["dirty"], bool)
    assert isinstance(manifest["source"]["status_short"], list)
    assert manifest["standalone_repo"]["expected_honxin_path"] == "/dev-project/eihead"
    assert manifest["standalone_repo"]["runtime_path"] == "/opt/eihead/current"
    assert manifest["future_capabilities"] == [
        {
            "area": "eye",
            "name": "eye.realtime_stream_detection",
            "status": "target",
            "target": "Realtime camera/Hailo stream detection feeding live RealtimeVisionObservation and monitor status data.",
            "compatibility_note": "Static image detection is retained only as a compatibility and test placeholder.",
        },
    ]
    assert manifest["migration_notes"] == [
        {
            "area": "eye",
            "applies_to": ["transitional_packages", "runtime_entrypoints"],
            "note": "The production eye direction for /dev-project/eihead is realtime stream detection; static image detection is not a deployment target.",
        },
    ]
    assert manifest["native_realtime_voice_files"] == [
        {
            "path": "eihead/ear/__init__.py",
            "role": "Native eihead ear realtime package boundary and exports.",
        },
        {
            "path": "eihead/ear/realtime.py",
            "role": "Native realtime voice ingestion pipeline contracts and status shapes.",
        },
        {
            "path": "eihead/mouth/__init__.py",
            "role": "Native eihead mouth package boundary and exports.",
        },
        {
            "path": "eihead/mouth/playback.py",
            "role": "Native speech synthesis/playback service with stop and busy-state reporting.",
        },
        {
            "path": "eihead/monitoring/voice.py",
            "role": "Monitor payload normalizer for offline/quasi-streaming closed-loop voice diagnostics and not-wired truthfulness.",
        },
    ]
    assert manifest["native_runtime_web_files"] == [
        {
            "path": "eihead/runtime/http_api.py",
            "role": "Runtime HTTP API surface for native head status/action requests.",
        },
        {
            "path": "eihead/monitoring/web.py",
            "role": "Web monitor composition including /api/voice/realtime and /api/audio/realtime voice panels.",
        },
        {
            "path": "eihead/runtime/app.py",
            "role": "Runtime service facade that binds realtime vision/voice/mouth status into monitor payloads.",
        },
    ]
    assert manifest["transitional_packages"] == [
        {
            "package": "apps.body_runtime",
            "paths": ["apps/body_runtime"],
            "reason": "Temporary honjia body runtime compatibility during eihead split.",
        },
        {
            "package": "eibrain.body",
            "paths": ["eibrain/body"],
            "reason": "Temporary honjia hardware/runtime implementation before native eihead modules replace it.",
        },
        {
            "package": "eibrain.cognition.realtime",
            "paths": ["eibrain/cognition/realtime"],
            "reason": "temporary realtime scheduler compatibility until eibrain/eihead protocol split is complete",
        },
        {
            "package": "eibrain.infra",
            "paths": ["eibrain/infra"],
            "reason": "Shared config helpers kept until eihead owns its deployment config layer.",
        },
        {
            "package": "eiprotocol",
            "paths": ["eiprotocol"],
            "reason": "Shared protocol MVP carried until /dev-project/eiprotocol becomes its own source repository.",
        },
        {
            "package": "eibrain.protocol",
            "paths": ["eibrain/protocol"],
            "reason": "Temporary protocol compatibility until eiprotocol is split into its own repo.",
        },
        {
            "package": "eibrain.verification",
            "paths": ["eibrain/verification"],
            "reason": "Head-side hardware verification helpers retained while verify_hardware CLI is transitional.",
        },
    ]
    assert manifest["runtime_entrypoints"] == [
        {
            "name": "eihead-runtime-http",
            "kind": "systemd-service",
            "service": "deploy/systemd/eihead-runtime.service",
            "console_script": "eihead-runtime",
            "module": "eihead.runtime.cli:main",
            "command": "eihead-runtime --config /etc/eihead/eihead.honjia.yaml http --host 0.0.0.0 --port 18081",
            "host": "0.0.0.0",
            "port": 18081,
        },
        {
            "name": "eihead-monitor",
            "kind": "systemd-service",
            "service": "deploy/systemd/eihead-monitor.service",
            "console_script": "eihead-runtime",
            "module": "eihead.runtime.cli:main",
            "command": "eihead-runtime --config /etc/eihead/eihead.honjia.yaml monitor --host 0.0.0.0 --port 18080",
            "host": "0.0.0.0",
            "port": 18080,
        },
        {
            "name": "apps.head_runtime",
            "kind": "python-module",
            "module": "apps.head_runtime.__main__:main",
            "command": "python -m apps.head_runtime",
            "compatibility": True,
        },
    ]


def test_export_manifest_paths_stay_relative_when_target_is_named_eihead(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead"

    result = module.export_eihead_repo(target, repo_root=REPO_ROOT)

    assert "apps/head_runtime/__main__.py" in result.copied
    assert "eihead/runtime/cli.py" in result.copied
    assert "eihead/apps/head_runtime/__main__.py" not in result.copied


def test_export_generates_standalone_pyproject_and_readme(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"

    module.export_eihead_repo(target, repo_root=REPO_ROOT)

    pyproject = (target / "pyproject.toml").read_text(encoding="utf-8")
    readme = (target / "README.md").read_text(encoding="utf-8")
    gitignore = (target / ".gitignore").read_text(encoding="utf-8")

    assert 'name = "eihead"' in pyproject
    assert 'eihead-runtime = "eihead.runtime.cli:main"' in pyproject
    assert '"apps.body_runtime*"' in pyproject
    assert '"eibrain.body*"' in pyproject
    assert '"eibrain.cognition*"' in pyproject
    assert '"eibrain.verification*"' in pyproject
    assert "eibrain-cognitive" not in pyproject
    assert "faster-whisper" not in pyproject
    assert "/dev-project/eihead" in readme
    assert "transitional `eibrain.body`" in readme
    assert "realtime stream detection" in readme
    assert "/api/voice/realtime" in readme
    assert "/api/audio/realtime" in readme
    assert "functional-not-complete" in readme
    assert "Realtime Cognitive" in readme
    assert "Scheduler" in readme
    assert "scheduler-backed functional stage" in readme
    assert "real streaming LLM/TTS" in readme
    assert "functional offline/quasi-streaming diagnostics" in readme
    assert "not hardware-verified real streaming" in readme
    assert "closed-loop voice diagnostics" in readme
    assert "round/scheduler/interrupt" in readme
    assert "hardware verification" in readme
    assert "Static image detection is compatibility/test-only" in readme
    assert "eihead-runtime http --host 0.0.0.0 --port 18081" in readme
    assert "__pycache__/" in gitignore
    assert "*.py[cod]" in gitignore


def test_export_documents_realtime_eye_adapter_monitor_and_truthfulness(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"

    result = module.export_eihead_repo(target, repo_root=REPO_ROOT)
    manifest = json.loads((target / "EXPORT_MANIFEST.json").read_text(encoding="utf-8"))
    readme = (target / "README.md").read_text(encoding="utf-8")

    assert (target / "eihead/eye/realtime.py").is_file()
    assert (target / "eihead/monitoring/realtime_vision.py").is_file()
    assert (target / "eihead/ear/realtime.py").is_file()
    assert (target / "eihead/mouth/playback.py").is_file()
    assert (target / "eihead/monitoring/voice.py").is_file()
    assert "eihead/eye/realtime.py" in result.copied
    assert "eihead/monitoring/realtime_vision.py" in result.copied
    assert "eihead/ear/realtime.py" in result.copied
    assert "eihead/mouth/playback.py" in result.copied
    assert "eihead/monitoring/voice.py" in result.copied
    assert manifest["native_realtime_eye_files"] == [
        {
            "path": "eihead/eye/adapters.py",
            "role": "Realtime GStreamer/Hailo adapter scaffold for /dev/video0 camera frames and /dev/hailo0 detections.",
        },
        {
            "path": "eihead/eye/gstreamer.py",
            "role": "Native realtime /dev/video0 GStreamer appsink reader and Hailo pipeline launcher.",
        },
        {
            "path": "eihead/eye/hailo_metadata.py",
            "role": "Native /dev/hailo0 Hailo ROI metadata parser for realtime detection boxes and scores.",
        },
        {
            "path": "eihead/eye/realtime.py",
            "role": "Realtime eye pipeline contracts shared by native adapters and monitor status.",
        },
        {
            "path": "eihead/monitoring/realtime_vision.py",
            "role": "Monitor payload normalizer for live realtime eye data and not-wired truthfulness.",
        },
    ]
    assert manifest["native_realtime_voice_files"] == [
        {
            "path": "eihead/ear/__init__.py",
            "role": "Native eihead ear realtime package boundary and exports.",
        },
        {
            "path": "eihead/ear/realtime.py",
            "role": "Native realtime voice ingestion pipeline contracts and status shapes.",
        },
        {
            "path": "eihead/mouth/__init__.py",
            "role": "Native eihead mouth package boundary and exports.",
        },
        {
            "path": "eihead/mouth/playback.py",
            "role": "Native speech synthesis/playback service with stop and busy-state reporting.",
        },
        {
            "path": "eihead/monitoring/voice.py",
            "role": "Monitor payload normalizer for offline/quasi-streaming closed-loop voice diagnostics and not-wired truthfulness.",
        },
    ]
    assert manifest["native_runtime_web_files"] == [
        {
            "path": "eihead/runtime/http_api.py",
            "role": "Runtime HTTP API surface for native head status/action requests.",
        },
        {
            "path": "eihead/monitoring/web.py",
            "role": "Web monitor composition including /api/voice/realtime and /api/audio/realtime voice panels.",
        },
        {
            "path": "eihead/runtime/app.py",
            "role": "Runtime service facade that binds realtime vision/voice/mouth status into monitor payloads.",
        },
    ]
    assert "eihead/eye/adapters.py" in result.copied
    assert "eihead/eye/adapters.py" in readme
    assert "eihead/eye/gstreamer.py" in result.copied
    assert "eihead/eye/gstreamer.py" in readme
    assert "eihead/eye/hailo_metadata.py" in result.copied
    assert "eihead/eye/hailo_metadata.py" in readme
    assert "eihead/eye/realtime.py" in readme
    assert "eihead/monitoring/realtime_vision.py" in readme
    assert "eihead/ear/realtime.py" in readme
    assert "eihead/ear/__init__.py" in readme
    assert "eihead/mouth/playback.py" in readme
    assert "eihead/mouth/__init__.py" in readme
    assert "eihead/monitoring/voice.py" in readme
    assert "eihead/runtime/http_api.py" in readme
    assert "/dev/video0" in readme
    assert "/dev/hailo0" in readme
    assert "not wired" in readme
    assert "/api/voice/realtime" in readme
    assert "/api/audio/realtime" in readme
    assert "functional-not-complete" in readme
    assert "functional offline/quasi-streaming diagnostics" in readme
    assert "not hardware-verified real streaming" in readme
    assert "closed-loop voice diagnostics" in readme
    assert "Realtime Cognitive" in readme
    assert "Scheduler" in readme
    assert "Static image detection is compatibility/test-only" in readme


def test_exported_transitional_runtime_imports_without_brain_runtime(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"

    module.export_eihead_repo(target, repo_root=REPO_ROOT)

    env = {**os.environ, "PYTHONPATH": str(target)}
    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "."],
        check=False,
        capture_output=True,
        text=True,
        cwd=target,
        env=env,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import apps.body_runtime.voice_dialogue_loop; "
                "import apps.body_runtime.verify_hardware; "
                "import eibrain.body.vad_policy; "
                "import eibrain.body.sherpa_streaming; "
                "import eibrain.verification; "
                "import eibrain.cognition.realtime; "
                "import eiprotocol; "
                "import eihead.monitoring.web; "
                "import eihead.runtime.app"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=target,
        env=env,
    )

    assert result.returncode == 0, result.stderr


def test_export_refuses_existing_target_without_force(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"
    target.mkdir()

    with pytest.raises(FileExistsError):
        module.export_eihead_repo(target, repo_root=REPO_ROOT)


def test_export_force_replaces_existing_target_after_validation(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"
    target.mkdir()
    marker = target / "old-file.txt"
    marker.write_text("old", encoding="utf-8")

    result = module.export_eihead_repo(target, repo_root=REPO_ROOT, force=True)

    assert result.force is True
    assert not marker.exists()
    assert (target / "eihead/runtime/cli.py").is_file()


def test_export_refuses_to_force_clean_source_repo_root(tmp_path: Path) -> None:
    module = _load_export_module()
    fake_repo = tmp_path / "fake-eibrain"
    fake_repo.mkdir()
    (fake_repo / "pyproject.toml").write_text('[project]\nname = "fake"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="source repo root"):
        module.export_eihead_repo(fake_repo, repo_root=fake_repo, force=True)


def test_cli_prints_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"

    assert module.main([str(target), "--repo-root", str(REPO_ROOT)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["target"] == str(target.resolve())
    assert output["force"] is False
    assert "eihead/runtime/cli.py" in output["copied"]
    assert output["generated"] == [
        ".gitignore",
        "pyproject.toml",
        "README.md",
        "eibrain/protocol/__init__.py",
        "EXPORT_MANIFEST.json",
    ]
    assert output["manifest"] == "EXPORT_MANIFEST.json"
