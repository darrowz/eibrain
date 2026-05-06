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
INDEPENDENT_PROTOCOL_MARKER = "independent eiprotocol fixture source\n"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_eihead_repo", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _create_protocol_git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "eiprotocol-repo"
    (repo / "eiprotocol").mkdir(parents=True)
    (repo / "eiprotocol" / "__init__.py").write_text(
        '"""fixture protocol package."""\n',
        encoding="utf-8",
    )
    (repo / "eiprotocol" / "independent_source_marker.txt").write_text(
        INDEPENDENT_PROTOCOL_MARKER,
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        "[project]\nname = \"eiprotocol\"\nversion = \"0.1.0\"\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], check=True, capture_output=True, text=True, cwd=repo)
    subprocess.run(["git", "add", "."], check=True, capture_output=True, text=True, cwd=repo)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Codex",
            "-c",
            "user.email=codex@example.invalid",
            "commit",
            "-m",
            "Initialize eiprotocol fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
    ).stdout.strip()
    return repo, commit


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
        "eibrain/cognition/vision_events.py",
        "eibrain/cognition/vision_realtime.py",
        "eibrain/cognition/vision_scene_graph.py",
        "eibrain/cognition/vision_voice_context.py",
        "eibrain/infra/config.py",
        "eibrain/memory/__init__.py",
        "eibrain/memory/visual_feedback.py",
        "eibrain/memory/visual_memory.py",
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
    assert "eibrain/cognition/vision_events.py" in result.copied
    assert "eibrain/cognition/vision_realtime.py" in result.copied
    assert "eibrain/cognition/vision_scene_graph.py" in result.copied
    assert "eibrain/cognition/vision_voice_context.py" in result.copied
    assert "eibrain/infra/config.py" in result.copied
    assert "eibrain/memory/__init__.py" in result.copied
    assert "eibrain/memory/visual_feedback.py" in result.copied
    assert "eibrain/memory/visual_memory.py" in result.copied
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
    assert manifest["protocol_sources"]["eiprotocol"]["repository"] == "eiprotocol"
    assert manifest["protocol_sources"]["eiprotocol"]["mode"] == "embedded_export"
    assert manifest["protocol_sources"]["eiprotocol"]["repo_root"] == str(REPO_ROOT)
    assert manifest["protocol_sources"]["eiprotocol"]["paths"] == ["eiprotocol"]
    assert manifest["protocol_sources"]["eiprotocol"]["commit"] == source_commit
    assert manifest["protocol_sources"]["legacy_eibrain_protocol"] == {
        "repository": "eibrain",
        "mode": "transitional_compatibility",
        "repo_root": str(REPO_ROOT),
        "paths": ["eibrain/protocol"],
        "commit": source_commit,
    }
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
            "package": "eibrain.voice",
            "paths": ["eibrain/voice"],
            "reason": "Shared voice-chain readiness helpers kept until eihead owns native voice diagnostics end-to-end.",
        },
        {
            "package": "eiprotocol",
            "paths": ["eiprotocol"],
            "reason": "Shared protocol MVP carried as an export copy; EXPORT_MANIFEST pins the independent /dev-project/eiprotocol revision when supplied.",
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


def test_export_manifest_records_native_completion_gates(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"

    module.export_eihead_repo(target, repo_root=REPO_ROOT)
    manifest = json.loads((target / "EXPORT_MANIFEST.json").read_text(encoding="utf-8"))

    gates = manifest["native_completion_gates"]
    assert [gate["module"] for gate in gates] == [
        "eye",
        "neck",
        "ear",
        "mouth",
        "realtime_cognitive_scheduler",
        "runtime",
        "export",
        "deploy",
    ]
    for gate in gates:
        assert gate["state"] in {"native_boundary", "transitional", "blocked_by_hardware_validation"}
        assert gate["owner_repo"] == "eihead"
        assert gate["blockers"]
        assert gate["next_acceptance"]
        assert gate["fake_completion_guard"] == (
            "Report not_wired, unknown, degraded, or blocked until the acceptance gate is verified."
        )

    eye_gate = next(gate for gate in gates if gate["module"] == "eye")
    assert "/dev/video0" in eye_gate["blockers"]
    assert "/dev/hailo0" in eye_gate["blockers"]
    assert "static image" in eye_gate["fake_completion_guard_note"]

    deploy_gate = next(gate for gate in gates if gate["module"] == "deploy")
    assert deploy_gate["state"] == "blocked_by_hardware_validation"
    assert "honjia Phase 0 baseline" in deploy_gate["next_acceptance"]

    scheduler_gate = next(gate for gate in gates if gate["module"] == "realtime_cognitive_scheduler")
    assert scheduler_gate["state"] == "native_boundary"
    assert scheduler_gate["code_level_complete"] is True
    assert scheduler_gate["hardware_verified"] is False
    assert scheduler_gate["honjia_cutover"] == "blocked_by_hardware_validation"
    assert "scheduler snapshot" in scheduler_gate["next_acceptance"]


def test_export_manifest_records_cutover_readiness_summary(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"

    module.export_eihead_repo(target, repo_root=REPO_ROOT)
    manifest = json.loads((target / "EXPORT_MANIFEST.json").read_text(encoding="utf-8"))

    readiness = manifest["cutover_readiness"]
    assert readiness["overall_state"] == "transitional"
    assert readiness["hardware_verified"] is False
    assert readiness["completion_policy"] == "blocked_until_honjia_hardware_acceptance"
    assert readiness["legacy_body_runtime_detached"] is False
    assert readiness["fake_completion_guard"] == (
        "Report not_wired, unknown, degraded, or blocked until the acceptance gate is verified."
    )

    providers = {entry["area"]: entry for entry in readiness["native_provider_modules"]}
    assert set(providers) == {"eye", "neck", "ear", "mouth", "runtime"}
    assert providers["eye"]["state"] == "native_boundary"
    assert providers["eye"]["hardware_verified"] is False
    assert "/dev/video0" in providers["eye"]["hardware_devices"]
    assert "/dev/hailo0" in providers["eye"]["hardware_devices"]
    assert "eihead.eye.gstreamer" in providers["eye"]["provider_modules"]
    assert providers["runtime"]["state"] == "transitional"
    assert "apps.body_runtime.BodyRuntimeApp" in providers["runtime"]["legacy_shim_dependencies"]

    endpoints = {entry["path"]: entry for entry in readiness["monitor_endpoints"]}
    assert {"/api/voice/realtime", "/api/audio/realtime", "/api/vision/realtime", "/api/eye/realtime"} <= set(
        endpoints
    )
    assert endpoints["/api/voice/realtime"]["port"] == 18080
    assert endpoints["/api/voice/realtime"]["provider_module"] == "eihead.monitoring.voice"
    assert endpoints["/api/audio/realtime"]["alias_for"] == "/api/voice/realtime"
    assert endpoints["/api/vision/realtime"]["provider_module"] == "eihead.monitoring.realtime_vision"
    assert endpoints["/api/eye/realtime"]["alias_for"] == "/api/vision/realtime"
    for endpoint in endpoints.values():
        assert endpoint["hardware_verified"] is False
        assert endpoint["completion_state"] in {"transitional", "blocked_by_hardware_validation"}


def test_export_manifest_records_code_completion_without_cutover_claim(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"

    module.export_eihead_repo(target, repo_root=REPO_ROOT)
    manifest = json.loads((target / "EXPORT_MANIFEST.json").read_text(encoding="utf-8"))
    readme = (target / "README.md").read_text(encoding="utf-8")
    audit = (target / "docs" / "eihead-migration-audit.md").read_text(encoding="utf-8")
    checklist = (target / "docs" / "eihead-code-completion-checklist.md").read_text(
        encoding="utf-8"
    )

    code_completion = manifest["code_completion"]
    assert code_completion["state"] == "code_level_complete_pending_honjia_validation"
    assert code_completion["software_closure"] == "complete"
    assert code_completion["honjia_cutover"] == "blocked_by_hardware_validation"
    assert code_completion["hardware_verified"] is False
    assert code_completion["legacy_body_runtime_detached"] is False
    assert code_completion["full_detachment_claim_allowed"] is False
    assert "Code-level completion is not honjia cutover completion" in code_completion["readiness_note"]

    software_closure = manifest["software_closure"]
    assert software_closure["scope"] == "Wave 3 P0/P1 software closure"
    assert software_closure["state"] == "code_level_complete_pending_honjia_validation"
    assert software_closure["code_level_complete"] is True
    assert software_closure["honjia_cutover_complete"] is False
    assert software_closure["hardware_verified"] is False
    assert software_closure["legacy_body_runtime_detached"] is False
    assert software_closure["full_detachment_claim_allowed"] is False
    assert software_closure["truthfulness_rule"] == (
        "Code-level completion means the software gates are represented and tested; "
        "honjia cutover remains blocked until real hardware validation records parity."
    )

    completed = {entry["id"]: entry for entry in software_closure["completed"]}
    assert set(completed) == {
        "p0_export_manifest_readiness",
        "p0_runtime_monitor_truthfulness",
        "p0_realtime_eye_boundary",
        "p1_voice_diagnostics_boundary",
        "p1_legacy_shim_policy",
    }
    for entry in completed.values():
        assert entry["status"] == "completed"
        assert entry["priority"] in {"P0", "P1"}
        assert entry["evidence"]

    hardware_blockers = {
        entry["id"]: entry for entry in software_closure["blocked_by_hardware_validation"]
    }
    assert set(hardware_blockers) == {
        "p0_honjia_phase0_parity",
        "p0_realtime_eye_hardware",
        "p0_neck_i2c_pan",
        "p1_ear_mouth_audio_loop",
        "p1_service_cutover_reboot_rollback",
    }
    for entry in hardware_blockers.values():
        assert entry["status"] == "blocked_by_hardware_validation"
        assert entry["priority"] in {"P0", "P1"}
        assert entry["requires"]

    legacy_blockers = {
        entry["package"]: entry for entry in software_closure["blocked_by_legacy_detachment"]
    }
    assert "apps.body_runtime" in legacy_blockers
    assert "eibrain.body" in legacy_blockers
    for entry in legacy_blockers.values():
        assert entry["status"] == "blocked_by_legacy_shim_removal"
        assert entry["blocks_full_detachment"] is True

    assert "Code-level completion is not honjia cutover completion" in readme
    assert "software_closure" in readme
    assert "honjia_cutover` is `blocked_by_hardware_validation" in readme
    assert "Code-Level Completion vs Honjia Cutover" in audit
    assert "software_closure" in audit
    assert "Code-level completion is not honjia cutover completion" in checklist
    assert "blocked_by_hardware_validation" in checklist
    assert "fully detached" in checklist


def test_export_marks_legacy_shims_as_transitional_not_detached(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"

    module.export_eihead_repo(target, repo_root=REPO_ROOT)
    manifest = json.loads((target / "EXPORT_MANIFEST.json").read_text(encoding="utf-8"))
    readme = (target / "README.md").read_text(encoding="utf-8")
    audit = (target / "docs" / "eihead-migration-audit.md").read_text(encoding="utf-8")

    shim_policy = manifest["cutover_readiness"]["legacy_shim_policy"]
    assert shim_policy["state"] == "transitional"
    assert shim_policy["legacy_body_runtime_detached"] is False
    assert shim_policy["legacy_paths_must_be_marked_as_shims"] is True
    assert shim_policy["full_detachment_claim_allowed"] is False

    shims = {entry["package"]: entry for entry in shim_policy["shims"]}
    assert set(shims) == {entry["package"] for entry in manifest["transitional_packages"]}
    for package in ("apps.body_runtime", "eibrain.body"):
        shim = shims[package]
        assert shim["classification"] == "transitional_shim"
        assert shim["blocks_full_detachment"] is True
        assert shim["removal_gate"]
        assert shim["paths"] == next(
            entry["paths"] for entry in manifest["transitional_packages"] if entry["package"] == package
        )

    assert "fake completion" in readme
    assert "cutover_readiness" in readme
    assert "legacy_shim_policy.legacy_body_runtime_detached` is `false`" in readme
    assert "blocked/transitional" in readme
    assert "hardware has not been verified on honjia" in readme
    assert "fake completion" in audit
    assert "cutover_readiness" in audit
    assert "legacy_shim_policy" in audit


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
    assert '"eibrain.memory*"' in pyproject
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
    assert "pins the independent protocol repository revision" in readme
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


def test_export_manifest_can_pin_independent_eiprotocol_repo_revision(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"
    protocol_repo, protocol_commit = _create_protocol_git_repo(tmp_path)

    module.export_eihead_repo(
        target,
        repo_root=REPO_ROOT,
        eiprotocol_repo_root=protocol_repo,
    )
    manifest = json.loads((target / "EXPORT_MANIFEST.json").read_text(encoding="utf-8"))

    assert manifest["protocol_sources"]["eiprotocol"] == {
        "repository": "eiprotocol",
        "mode": "independent_repo",
        "repo_root": str(protocol_repo.resolve()),
        "paths": ["eiprotocol"],
        "commit": protocol_commit,
        "dirty": False,
        "status_short": [],
    }


def test_export_copies_eiprotocol_from_independent_repo_when_supplied(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eihead-standalone"
    protocol_repo, _ = _create_protocol_git_repo(tmp_path)

    result = module.export_eihead_repo(
        target,
        repo_root=REPO_ROOT,
        eiprotocol_repo_root=protocol_repo,
    )

    assert (target / "eiprotocol" / "__init__.py").read_text(encoding="utf-8") == (
        '"""fixture protocol package."""\n'
    )
    assert (
        target / "eiprotocol" / "independent_source_marker.txt"
    ).read_text(encoding="utf-8") == INDEPENDENT_PROTOCOL_MARKER
    assert "eiprotocol/independent_source_marker.txt" in result.copied


def test_export_rejects_eibrain_root_as_independent_eiprotocol_repo(tmp_path: Path) -> None:
    module = _load_export_module()

    with pytest.raises(ValueError, match="must be independent"):
        module.export_eihead_repo(
            tmp_path / "eihead-standalone",
            repo_root=REPO_ROOT,
            eiprotocol_repo_root=REPO_ROOT,
        )


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
