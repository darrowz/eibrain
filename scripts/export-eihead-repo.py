#!/usr/bin/env python3
"""Export the eihead standalone repository layout from the eibrain tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
COPY_DIRS = (
    "eihead",
    "apps/head_runtime",
    # Transitional body runtime until honjia hardware code is fully renamed.
    "apps/body_runtime",
    "eibrain/body",
    "eibrain/cognition/realtime",
    "eibrain/infra",
    "eiprotocol",
    # Temporary compatibility until eiprotocol is split into its own repo.
    "eibrain/protocol",
    "eibrain/verification",
)
OPTIONAL_FILES = (
    "apps/__init__.py",
    "eibrain/__init__.py",
    "eibrain/cognition/__init__.py",
)
COPY_GLOBS = (
    "config/eibrain.honjia.yaml",
    "config/eihead*.yaml",
    "deploy/systemd/eihead-*.service",
    "docs/eihead-*.md",
)
MANIFEST_FILENAME = "EXPORT_MANIFEST.json"
EXPECTED_HONXIN_PATH = "/dev-project/eihead"
RUNTIME_PATH = "/opt/eihead/current"
SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

TRANSITIONAL_PACKAGES = (
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
)

RUNTIME_ENTRYPOINTS = (
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
)

FUTURE_CAPABILITIES = (
    {
        "area": "eye",
        "name": "eye.realtime_stream_detection",
        "status": "target",
        "target": "Realtime camera/Hailo stream detection feeding live RealtimeVisionObservation and monitor status data.",
        "compatibility_note": "Static image detection is retained only as a compatibility and test placeholder.",
    },
)

NATIVE_REALTIME_EYE_FILES = (
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
)

NATIVE_REALTIME_VOICE_FILES = (
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
)

NATIVE_RUNTIME_WEB_FILES = (
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
)

NATIVE_PROVIDER_MODULES = (
    {
        "area": "eye",
        "state": "native_boundary",
        "completion_gate": "eye",
        "provider_modules": [
            "eihead.eye.adapters",
            "eihead.eye.gstreamer",
            "eihead.eye.hailo_metadata",
            "eihead.eye.realtime",
            "eihead.monitoring.realtime_vision",
        ],
        "hardware_devices": ["/dev/video0", "/dev/hailo0"],
        "hardware_verified": False,
        "readiness_note": "Native boundary exists, but honjia realtime camera/Hailo validation is still required.",
    },
    {
        "area": "neck",
        "state": "transitional",
        "completion_gate": "neck",
        "provider_modules": ["eihead.neck.pan"],
        "hardware_devices": ["/dev/i2c-1"],
        "hardware_verified": False,
        "legacy_shim_dependencies": [
            "eibrain.body.neck_control",
            "eibrain.body.raspbot_driver",
            "eibrain.body.organs.neck",
        ],
        "readiness_note": "Pure pan planning is native; honjia Raspbot/I2C actuation still depends on legacy body shims.",
    },
    {
        "area": "ear",
        "state": "transitional",
        "completion_gate": "ear",
        "provider_modules": ["eihead.ear.realtime", "eihead.monitoring.voice"],
        "hardware_devices": ["plughw:CARD=U4K,DEV=0", "/dev/snd"],
        "hardware_verified": False,
        "legacy_shim_dependencies": [
            "apps.body_runtime.voice_dialogue_loop",
            "eibrain.body.sherpa_streaming",
            "eibrain.body.vad_policy",
        ],
        "readiness_note": "Native voice status contracts exist; real streaming microphone/VAD/ASR validation is not complete.",
    },
    {
        "area": "mouth",
        "state": "transitional",
        "completion_gate": "mouth",
        "provider_modules": ["eihead.mouth.playback", "eihead.monitoring.voice"],
        "hardware_devices": ["configured honjia speaker/playback device"],
        "hardware_verified": False,
        "legacy_shim_dependencies": ["apps.body_runtime.voice_dialogue_loop"],
        "readiness_note": "Native playback contracts exist; audible TTS/playback/stop validation is still required.",
    },
    {
        "area": "runtime",
        "state": "transitional",
        "completion_gate": "runtime",
        "provider_modules": [
            "eihead.runtime.app",
            "eihead.runtime.http_api",
            "eihead.monitoring.web",
        ],
        "hardware_devices": [],
        "hardware_verified": False,
        "legacy_shim_dependencies": ["apps.body_runtime.BodyRuntimeApp"],
        "readiness_note": "Runtime API and monitor are native wrappers while snapshot/action paths still delegate to legacy body runtime.",
    },
)

MONITOR_ENDPOINTS = (
    {
        "path": "/health",
        "method": "GET",
        "port": 18080,
        "provider_module": "eihead.monitoring.web",
        "completion_gate": "runtime",
        "completion_state": "transitional",
        "hardware_verified": False,
        "truthfulness_rule": "Return real health or explicit degraded/error state; never fake healthy.",
    },
    {
        "path": "/api/status",
        "method": "GET",
        "port": 18080,
        "provider_module": "eihead.runtime.app",
        "completion_gate": "runtime",
        "completion_state": "transitional",
        "hardware_verified": False,
        "truthfulness_rule": "Delegated compatibility must remain visible until native providers own status.",
    },
    {
        "path": "/api/capabilities",
        "method": "GET",
        "port": 18080,
        "provider_module": "eihead.runtime.app",
        "completion_gate": "runtime",
        "completion_state": "transitional",
        "hardware_verified": False,
        "truthfulness_rule": "Report device readiness from real probes or explicit unknown/offline/degraded states.",
    },
    {
        "path": "/api/vision/realtime",
        "method": "GET",
        "port": 18080,
        "provider_module": "eihead.monitoring.realtime_vision",
        "completion_gate": "eye",
        "completion_state": "blocked_by_hardware_validation",
        "hardware_verified": False,
        "truthfulness_rule": "Realtime boxes/scores/parser readiness must come from live /dev/video0 and /dev/hailo0 data or explicit not_wired/degraded states.",
    },
    {
        "path": "/api/eye/realtime",
        "method": "GET",
        "port": 18080,
        "provider_module": "eihead.monitoring.realtime_vision",
        "alias_for": "/api/vision/realtime",
        "completion_gate": "eye",
        "completion_state": "blocked_by_hardware_validation",
        "hardware_verified": False,
        "truthfulness_rule": "Alias for realtime eye diagnostics; static-image compatibility cannot satisfy this endpoint.",
    },
    {
        "path": "/api/voice/realtime",
        "method": "GET",
        "port": 18080,
        "provider_module": "eihead.monitoring.voice",
        "completion_gate": "ear_mouth",
        "completion_state": "transitional",
        "hardware_verified": False,
        "truthfulness_rule": "Voice diagnostics may show offline/quasi-streaming data, but missing streaming stages must remain not_wired/unknown/degraded.",
    },
    {
        "path": "/api/audio/realtime",
        "method": "GET",
        "port": 18080,
        "provider_module": "eihead.monitoring.voice",
        "alias_for": "/api/voice/realtime",
        "completion_gate": "ear_mouth",
        "completion_state": "transitional",
        "hardware_verified": False,
        "truthfulness_rule": "Alias for voice diagnostics; hardware-unverified closed-loop diagnostics are not real streaming completion.",
    },
    {
        "path": "/api/actions/recent",
        "method": "GET",
        "port": 18080,
        "provider_module": "eihead.monitoring.web",
        "completion_gate": "runtime",
        "completion_state": "transitional",
        "hardware_verified": False,
        "truthfulness_rule": "Report recent actions only when runtime exposes a real action log, otherwise not_wired.",
    },
    {
        "path": "/api/events/recent",
        "method": "GET",
        "port": 18080,
        "provider_module": "eihead.monitoring.web",
        "completion_gate": "runtime",
        "completion_state": "transitional",
        "hardware_verified": False,
        "truthfulness_rule": "Report recent events only when runtime exposes a real event journal, otherwise not_wired.",
    },
)

LEGACY_SHIM_REMOVAL_GATES = {
    "apps.body_runtime": "HeadRuntimeApp snapshot/actions no longer instantiate or import BodyRuntimeApp.",
    "eibrain.body": "Native eihead eye, neck, ear, and mouth providers pass honjia parity without eibrain.body imports.",
    "eibrain.cognition.realtime": "Voice round lifecycle and scheduler state are owned by eihead/eiprotocol contracts without eibrain scheduler imports.",
    "eibrain.infra": "eihead owns its deployment config layer and no longer needs shared eibrain infra helpers.",
    "eiprotocol": "eiprotocol is consumed as an independent package rather than a copied export payload.",
    "eibrain.protocol": "eihead and eibrain both consume eiprotocol directly without legacy eibrain.protocol compatibility exports.",
    "eibrain.verification": "Hardware verification CLI/checks are native to eihead or a shared verification package.",
}

MIGRATION_NOTES = (
    {
        "area": "eye",
        "applies_to": ["transitional_packages", "runtime_entrypoints"],
        "note": "The production eye direction for /dev-project/eihead is realtime stream detection; static image detection is not a deployment target.",
    },
)

FAKE_COMPLETION_GUARD = (
    "Report not_wired, unknown, degraded, or blocked until the acceptance gate is verified."
)
CODE_COMPLETION_STATE = "code_level_complete_pending_honjia_validation"

SOFTWARE_CLOSURE_COMPLETED = (
    {
        "id": "p0_export_manifest_readiness",
        "priority": "P0",
        "area": "export",
        "status": "completed",
        "evidence": [
            "EXPORT_MANIFEST.json records code_completion, software_closure, cutover_readiness, and legacy_shim_policy.",
            "Focused export tests assert code-level completion is not honjia cutover completion.",
        ],
    },
    {
        "id": "p0_runtime_monitor_truthfulness",
        "priority": "P0",
        "area": "runtime_monitor",
        "status": "completed",
        "evidence": [
            "native_runtime_web_files names the exported runtime and monitor API surface.",
            "monitor_endpoints require real data or explicit not_wired, unknown, degraded, or blocked states.",
        ],
    },
    {
        "id": "p0_realtime_eye_boundary",
        "priority": "P0",
        "area": "eye",
        "status": "completed",
        "evidence": [
            "native_realtime_eye_files exports the GStreamer, Hailo metadata, realtime contracts, and monitor adapter boundaries.",
            "Static image detection remains compatibility/test-only and cannot satisfy realtime eye completion.",
        ],
    },
    {
        "id": "p1_voice_diagnostics_boundary",
        "priority": "P1",
        "area": "ear_mouth",
        "status": "completed",
        "evidence": [
            "native_realtime_voice_files exports ear, mouth, and voice monitor diagnostic boundaries.",
            "Closed-loop diagnostics remain offline/quasi-streaming until honjia audio hardware proves the real loop.",
        ],
    },
    {
        "id": "p1_legacy_shim_policy",
        "priority": "P1",
        "area": "legacy_detachment",
        "status": "completed",
        "evidence": [
            "legacy_shim_policy names every copied legacy package as a transitional shim.",
            "full_detachment_claim_allowed stays false while legacy runtime copies remain required.",
        ],
    },
)

SOFTWARE_CLOSURE_HARDWARE_BLOCKERS = (
    {
        "id": "p0_honjia_phase0_parity",
        "priority": "P0",
        "area": "cutover",
        "status": "blocked_by_hardware_validation",
        "requires": [
            "Repeat the Phase 0 honjia baseline after export using real voice, vision, neck, monitor, and rollback evidence.",
        ],
    },
    {
        "id": "p0_realtime_eye_hardware",
        "priority": "P0",
        "area": "eye",
        "status": "blocked_by_hardware_validation",
        "requires": [
            "Validate continuous /dev/video0 frames and /dev/hailo0 detections on honjia.",
            "Record boxes, scores, frame age, parser readiness, stale state, and error state on port 18080.",
        ],
    },
    {
        "id": "p0_neck_i2c_pan",
        "priority": "P0",
        "area": "neck",
        "status": "blocked_by_hardware_validation",
        "requires": [
            "Validate /dev/i2c-1 Raspbot pan/yaw movement, settle behavior, and unsupported tilt truthfulness on honjia.",
        ],
    },
    {
        "id": "p1_ear_mouth_audio_loop",
        "priority": "P1",
        "area": "ear_mouth",
        "status": "blocked_by_hardware_validation",
        "requires": [
            "Validate U4K microphone capture, VAD/ASR turn telemetry, mouth-busy suppression, audible TTS playback, and stop_speech on honjia.",
        ],
    },
    {
        "id": "p1_service_cutover_reboot_rollback",
        "priority": "P1",
        "area": "deploy",
        "status": "blocked_by_hardware_validation",
        "requires": [
            "Start eihead runtime and monitor services on honjia, verify reboot persistence, and prove rollback restores the previous eibrain services.",
        ],
    },
)

NATIVE_COMPLETION_GATES = (
    {
        "module": "eye",
        "owner_repo": "eihead",
        "state": "native_boundary",
        "blockers": "/dev/video0 and /dev/hailo0 realtime stream validation on honjia.",
        "next_acceptance": "Realtime boxes, scores, frame age, parser readiness, and stale/error state are visible on port 18080.",
        "fake_completion_guard": FAKE_COMPLETION_GUARD,
        "fake_completion_guard_note": "static image detection remains compatibility-only and must not satisfy native realtime eye completion.",
    },
    {
        "module": "neck",
        "owner_repo": "eihead",
        "state": "transitional",
        "blockers": "/dev/i2c-1 Raspbot adapter validation and pan-only settle/no-oscillation test on honjia.",
        "next_acceptance": "move_head pan/yaw reaches the native adapter, tilt returns unsupported, and monitor shows target angle plus suppression reason.",
        "fake_completion_guard": FAKE_COMPLETION_GUARD,
        "fake_completion_guard_note": "Do not advertise tilt support unless real tilt hardware is installed and accepted.",
    },
    {
        "module": "ear",
        "owner_repo": "eihead",
        "state": "transitional",
        "blockers": "U4K microphone, VAD, ASR model, and mouth-busy suppression validation on honjia.",
        "next_acceptance": "Voice wake emits a traceable ASR turn with stage latency and no fake healthy state for missing streaming stages.",
        "fake_completion_guard": FAKE_COMPLETION_GUARD,
        "fake_completion_guard_note": "Closed-loop diagnostics are not hardware-verified realtime streaming until live audio proves them.",
    },
    {
        "module": "mouth",
        "owner_repo": "eihead",
        "state": "transitional",
        "blockers": "MiniMax TTS synthesis, local playback device, stop_speech, and busy-state validation on honjia.",
        "next_acceptance": "speak is audible, stop_speech interrupts or reports a real unsupported reason, and monitor shows real provider/playback state.",
        "fake_completion_guard": FAKE_COMPLETION_GUARD,
        "fake_completion_guard_note": "Synthesis configuration without audible playback is not completion.",
    },
    {
        "module": "runtime",
        "owner_repo": "eihead",
        "state": "transitional",
        "blockers": "HeadRuntimeApp still depends on delegated legacy body runtime for part of the snapshot/action path.",
        "next_acceptance": "snapshot/actions use native eye, neck, ear, and mouth providers without requiring apps.body_runtime.",
        "fake_completion_guard": FAKE_COMPLETION_GUARD,
        "fake_completion_guard_note": "Delegated compatibility must remain visible in status until native providers own the path.",
    },
    {
        "module": "export",
        "owner_repo": "eihead",
        "state": "transitional",
        "blockers": "Legacy package copies are still required until native runtime parity removes imports.",
        "next_acceptance": "Generated /dev-project/eihead installs and starts without apps.body_runtime or eibrain.body except named shims.",
        "fake_completion_guard": FAKE_COMPLETION_GUARD,
        "fake_completion_guard_note": "The export manifest must name every transitional package until its removal gate passes.",
    },
    {
        "module": "deploy",
        "owner_repo": "eihead",
        "state": "blocked_by_hardware_validation",
        "blockers": "honjia Phase 0 baseline, service cutover, reboot persistence, and rollback checks.",
        "next_acceptance": "honjia Phase 0 baseline repeats with equal or better voice, vision, neck, monitor, and rollback results.",
        "fake_completion_guard": FAKE_COMPLETION_GUARD,
        "fake_completion_guard_note": "Do not enable permanent eihead services until real-device parity is recorded.",
    },
)


PYPROJECT_TEMPLATE = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "eihead"
version = "0.1.0"
description = "Standalone head node runtime for the ei-hongtu system"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["PyYAML>=6.0"]

[project.scripts]
eihead-runtime = "eihead.runtime.cli:main"

[tool.setuptools.packages.find]
include = [
    "eihead*",
    "apps",
    "apps.head_runtime*",
    "apps.body_runtime*",
    "eibrain",
    "eibrain.body*",
    "eibrain.cognition*",
    "eibrain.infra*",
    "eibrain.protocol*",
    "eibrain.verification*",
    "eiprotocol*",
]
exclude = ["config*", "tests*", "docs*", "scripts*", "deploy*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""


README_TEMPLATE = """# eihead

Standalone head-node export for the ei-hongtu project.

This repository is generated from the eibrain monorepo by
`scripts/export-eihead-repo.py`. It contains the honjia-facing head runtime,
the `apps.head_runtime` compatibility entrypoint, a transitional copy of the
old body runtime, eihead systemd templates, and eihead migration/deployment
docs.

## Expected sync target

- Source of truth on honxin: `/dev-project/eihead`
- Runtime deployment path on honjia: `/opt/eihead/current`
- Runtime API: `eihead-runtime http --host 0.0.0.0 --port 18081`
- Native Web monitor: `eihead-runtime monitor --host 0.0.0.0 --port 18080`

## Eye direction

The production eye target for `/dev-project/eihead` is realtime stream detection:
continuous `/dev/video0` camera frames and `/dev/hailo0` detections feeding live
RealtimeVisionObservation payloads, runtime status, and the operator monitor.

The native voice boundary is under `eihead/ear` and `eihead/mouth`:
`eihead/ear/realtime.py`, `eihead/ear/__init__.py`,
`eihead/mouth/playback.py`, and `eihead/mouth/__init__.py`.
Its monitor adapter is `eihead/monitoring/voice.py`.
The monitor endpoint bridge is exported in `eihead/monitoring/web.py`, with
runtime facade support in `eihead/runtime/app.py`.

Native runtime and monitor surface includes:
- `GET /api/voice/realtime`
- `GET /api/audio/realtime`

Voice chain is now in a scheduler-backed functional stage using Realtime
Cognitive Scheduler for round lifecycle, scheduler status, and interrupt
visibility. Realtime Cognitive Scheduler compatibility is transitional. It
provides functional offline/quasi-streaming diagnostics for the closed-loop
voice diagnostics surface, but it is not hardware-verified real streaming.
The closed-loop voice diagnostics are functional offline/quasi-streaming diagnostics,
not hardware-verified real streaming or real streaming LLM/TTS.
It is still functional-not-complete: the loop has not been wired to real
streaming LLM/TTS, and the Web monitor should make round/scheduler/interrupt
state visible without presenting missing streaming stages as complete.

## Code completion vs cutover

Code-level completion is not honjia cutover completion. In
`EXPORT_MANIFEST.json`, `code_completion.software_closure` is `complete`, but
`code_completion.honjia_cutover` is `blocked_by_hardware_validation`.

The `software_closure` field lists which Wave 3 P0/P1 software gates are
complete at code level, which P0/P1 checks still require honjia hardware
validation, and which legacy shim removals still block any fully detached claim.
Do not describe this export as fully detached while
`legacy_body_runtime_detached` or `full_detachment_claim_allowed` is `false`.
Real cutover still requires recorded honjia parity for realtime eye, pan-only
neck, ear/mouth audio, services, reboot persistence, and rollback.

The standalone export intentionally includes the native realtime eye adapter and
monitor payload files:

- `eihead/eye/adapters.py`
- `eihead/eye/gstreamer.py`
- `eihead/eye/hailo_metadata.py`
- `eihead/eye/realtime.py`
- `eihead/monitoring/realtime_vision.py`

Native voice boundaries are exported as:

- `eihead/ear/__init__.py`
- `eihead/ear/realtime.py`
- `eihead/mouth/__init__.py`
- `eihead/mouth/playback.py`
- `eihead/monitoring/voice.py`
- `eihead/runtime/http_api.py`
- `eihead/monitoring/web.py`

The monitor truthfulness rule is strict: missing live wiring must be shown as
`not wired`, `not_wired`, `unknown`, or explicit offline/degraded data. Do not
show blank or fake-normal realtime vision status.

Static image detection is compatibility/test-only. Keep it only for old callers,
fixtures, and non-hardware tests; do not treat it as the deployment direction.

## Local commands

```bash
python -m pip install -e .
eihead-runtime status
eihead-runtime http --host 0.0.0.0 --port 18081
eihead-runtime monitor --host 0.0.0.0 --port 18080
```

The current runtime still carries a small `eibrain.protocol` compatibility
subset, transitional `eibrain.body` hardware code, and the minimal
`eibrain.cognition.realtime` scheduler primitives needed by the exported
`apps.body_runtime` voice chain, plus transitional hardware verification
helpers. The shared protocol package is also exported as `/dev-project/eiprotocol`;
when this exporter is given `--eiprotocol-repo-root`, `EXPORT_MANIFEST.json`
pins the independent protocol repository revision used by this eihead build.

`EXPORT_MANIFEST.json` also contains `native_completion_gates`. Treat those
gates as the source of truth for whether eye, neck, ear, mouth, runtime,
export, and deploy are complete. A module remains transitional or blocked until
its gate is verified on honjia; status and monitor payloads must say
`not_wired`, `unknown`, `degraded`, or `blocked` rather than implying fake
completion.

## Cutover readiness and fake completion

`EXPORT_MANIFEST.json` contains `cutover_readiness`, a machine-readable summary
for cutover review. It lists `native_provider_modules`, `monitor_endpoints`, and
`legacy_shim_policy` so reviewers can tell native boundaries from transitional
compatibility.

How to judge fake completion:
- If `cutover_readiness.hardware_verified` is `false`, the hardware has not been verified on honjia and the export remains blocked/transitional even if local tests or static fixtures pass.
- If `legacy_shim_policy.legacy_body_runtime_detached` is `false`, the export
  still carries legacy body runtime shims. Those paths must stay explicitly
  marked as transitional shims and must not be described as fully detached.
- Monitor endpoints are readiness probes, not proof of completion. A response
  is only acceptable when it shows real data or explicit `not_wired`, `unknown`,
  `degraded`, or `blocked` state for missing hardware or unwired stages.
"""


PROTOCOL_INIT_TEMPLATE = '''"""Minimal eibrain.protocol compatibility exports for transitional eihead."""

from .actions import MoveHeadAction, PlaySpeechAction, StopSpeechAction
from .observations import AudioTranscriptFinal
from .outcomes import ActionExecuted, SpeechPlaybackCompleted

__all__ = [
    "ActionExecuted",
    "AudioTranscriptFinal",
    "MoveHeadAction",
    "PlaySpeechAction",
    "SpeechPlaybackCompleted",
    "StopSpeechAction",
]
'''

GITIGNORE_TEMPLATE = """__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
"""


@dataclass(frozen=True)
class ExportResult:
    """Summary for a completed eihead export."""

    target: Path
    copied: tuple[str, ...]
    generated: tuple[str, ...]
    force: bool = False
    manifest: str = MANIFEST_FILENAME

    def to_dict(self) -> dict[str, object]:
        return {
            "target": str(self.target),
            "copied": list(self.copied),
            "generated": list(self.generated),
            "force": self.force,
            "manifest": self.manifest,
        }


def export_eihead_repo(
    target_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    eiprotocol_repo_root: str | Path | None = None,
    force: bool = False,
) -> ExportResult:
    """Export eihead standalone files into ``target_dir``.

    The source repository is only read. Existing targets are rejected unless
    ``force`` is true; forced cleanup is blocked for repo roots, repo parents,
    source-tree children, filesystem roots, and common broad directories.
    """

    source_root = _resolve_repo_root(repo_root)
    protocol_root = _resolve_optional_eiprotocol_repo_root(eiprotocol_repo_root)
    if protocol_root is not None and _same_path(protocol_root, source_root):
        raise ValueError("eiprotocol repo root must be independent from the eibrain repo root")
    target = _resolve_target(target_dir)

    if target.exists():
        if not force:
            raise FileExistsError(
                f"target already exists: {target}. Re-run with --force to replace it."
            )
        _validate_clean_target(target, source_root)
        if not target.is_dir():
            raise NotADirectoryError(f"target exists but is not a directory: {target}")
        shutil.rmtree(target)
    elif force:
        _validate_clean_target(target, source_root)

    target.mkdir(parents=True, exist_ok=False)

    copied: list[str] = []
    for rel_dir in COPY_DIRS:
        copy_root = (
            protocol_root
            if rel_dir == "eiprotocol" and protocol_root is not None
            else source_root
        )
        copied.extend(_copy_directory(copy_root, target, rel_dir))
    for rel_file in OPTIONAL_FILES:
        source_file = source_root / rel_file
        if source_file.exists():
            copied.append(_copy_file(source_file, target / rel_file, target_root=target))
    for pattern in COPY_GLOBS:
        copied.extend(_copy_glob(source_root, target, pattern))

    generated = list(_write_templates(target))
    generated.append(
        _write_export_manifest(
            target_root=target,
            source_root=source_root,
            eiprotocol_repo_root=protocol_root,
            copied=tuple(sorted(copied)),
            generated=tuple(generated),
        )
    )
    return ExportResult(
        target=target,
        copied=tuple(sorted(copied)),
        generated=tuple(generated),
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the standalone eihead repository layout."
    )
    parser.add_argument("target", help="Destination directory, for example /dev-project/eihead")
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help="Source eibrain repository root. Defaults to this script's parent repo.",
    )
    parser.add_argument(
        "--eiprotocol-repo-root",
        default=None,
        help=(
            "Optional independent eiprotocol repository root. When supplied, "
            "its git revision is pinned in EXPORT_MANIFEST.json."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing target directory after path validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = export_eihead_repo(
            args.target,
            repo_root=args.repo_root,
            eiprotocol_repo_root=args.eiprotocol_repo_root,
            force=args.force,
        )
    except Exception as exc:  # pragma: no cover - exercised by CLI callers.
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    root = Path(repo_root or DEFAULT_REPO_ROOT).expanduser().resolve(strict=True)
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"repo root does not contain pyproject.toml: {root}")
    return root


def _resolve_optional_eiprotocol_repo_root(repo_root: str | Path | None) -> Path | None:
    if repo_root in (None, ""):
        return None
    root = Path(repo_root).expanduser().resolve(strict=True)
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"eiprotocol repo root does not contain pyproject.toml: {root}")
    if not (root / "eiprotocol").is_dir():
        raise FileNotFoundError(f"eiprotocol repo root does not contain eiprotocol/: {root}")
    return root


def _resolve_target(target_dir: str | Path) -> Path:
    return Path(target_dir).expanduser().resolve(strict=False)


def _validate_clean_target(target: Path, repo_root: Path) -> None:
    target = target.resolve(strict=False)
    repo_root = repo_root.resolve(strict=True)

    if _same_path(target, repo_root):
        raise ValueError("refusing to clean the source repo root")
    if _path_contains(repo_root, target):
        raise ValueError("refusing to clean a directory inside the source repo")
    if _path_contains(target, repo_root):
        raise ValueError("refusing to clean an ancestor of the source repo")
    if target.parent == target:
        raise ValueError("refusing to clean a filesystem root")

    home = Path.home().resolve(strict=False)
    if _same_path(target, home) or _path_contains(target, home):
        raise ValueError("refusing to clean the user home directory or its ancestors")

    if len(target.parts) <= 2:
        raise ValueError(f"refusing to clean a broad path: {target}")

    broad_posix_paths = {
        Path("/dev"),
        Path("/dev-project"),
        Path("/dev-projiect"),
        Path("/etc"),
        Path("/home"),
        Path("/opt"),
        Path("/tmp"),
        Path("/usr"),
        Path("/var"),
    }
    for broad_path in broad_posix_paths:
        if _same_path(target, broad_path):
            raise ValueError(f"refusing to clean broad path: {target}")


def _copy_directory(source_root: Path, target_root: Path, rel_dir: str) -> list[str]:
    source_dir = source_root / rel_dir
    if not source_dir.is_dir():
        raise FileNotFoundError(f"required source directory is missing: {source_dir}")

    copied: list[str] = []
    for source_path in sorted(source_dir.rglob("*")):
        if _should_skip(source_path, source_dir):
            continue
        if not source_path.is_file():
            continue
        rel_path = source_path.relative_to(source_root)
        copied.append(_copy_file(source_path, target_root / rel_path, target_root=target_root))
    return copied


def _copy_glob(source_root: Path, target_root: Path, pattern: str) -> list[str]:
    matches = sorted(path for path in source_root.glob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"required source pattern matched no files: {pattern}")
    return [
        _copy_file(path, target_root / path.relative_to(source_root), target_root=target_root)
        for path in matches
    ]


def _copy_file(source_path: Path, target_path: Path, *, target_root: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return _relative_output_path(target_path, target_root=target_root)


def _write_templates(target_root: Path) -> tuple[str, ...]:
    templates = {
        ".gitignore": GITIGNORE_TEMPLATE,
        "pyproject.toml": PYPROJECT_TEMPLATE,
        "README.md": README_TEMPLATE,
        "eibrain/protocol/__init__.py": PROTOCOL_INIT_TEMPLATE,
    }
    generated: list[str] = []
    for rel_path, text in templates.items():
        path = target_root / rel_path
        path.write_text(text, encoding="utf-8", newline="\n")
        generated.append(rel_path)
    return tuple(generated)


def _write_export_manifest(
    *,
    target_root: Path,
    source_root: Path,
    eiprotocol_repo_root: Path | None,
    copied: tuple[str, ...],
    generated: tuple[str, ...],
) -> str:
    generated_with_manifest = (*generated, MANIFEST_FILENAME)
    source_state = _read_source_git_state(source_root)
    eiprotocol_state = _read_eiprotocol_source_state(
        source_root,
        eiprotocol_repo_root,
        source_state=source_state,
    )
    manifest = {
        "schema_version": 1,
        "source": {
            "repository": "eibrain",
            "repo_root": str(source_root),
            **source_state,
        },
        "protocol_sources": {
            "eiprotocol": eiprotocol_state,
            "legacy_eibrain_protocol": {
                "repository": "eibrain",
                "mode": "transitional_compatibility",
                "repo_root": str(source_root),
                "paths": ["eibrain/protocol"],
                "commit": source_state["commit"],
            },
        },
        "standalone_repo": {
            "name": "eihead",
            "expected_honxin_path": EXPECTED_HONXIN_PATH,
            "runtime_path": RUNTIME_PATH,
            "manifest_path": MANIFEST_FILENAME,
        },
        "transitional_packages": list(TRANSITIONAL_PACKAGES),
        "runtime_entrypoints": list(RUNTIME_ENTRYPOINTS),
        "native_realtime_eye_files": list(NATIVE_REALTIME_EYE_FILES),
        "native_realtime_voice_files": list(NATIVE_REALTIME_VOICE_FILES),
        "native_runtime_web_files": list(NATIVE_RUNTIME_WEB_FILES),
        "code_completion": _build_code_completion_summary(),
        "software_closure": _build_software_closure_summary(),
        "cutover_readiness": _build_cutover_readiness_summary(),
        "native_completion_gates": list(NATIVE_COMPLETION_GATES),
        "future_capabilities": list(FUTURE_CAPABILITIES),
        "migration_notes": list(MIGRATION_NOTES),
        "exported_paths": {
            "copied": list(copied),
            "generated": list(generated_with_manifest),
        },
    }
    path = target_root / MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return MANIFEST_FILENAME


def _build_code_completion_summary() -> dict[str, object]:
    return {
        "state": CODE_COMPLETION_STATE,
        "software_closure": "complete",
        "honjia_cutover": "blocked_by_hardware_validation",
        "hardware_verified": False,
        "legacy_body_runtime_detached": False,
        "full_detachment_claim_allowed": False,
        "readiness_note": (
            "Code-level completion is not honjia cutover completion. "
            "The Wave 3 P0/P1 software closure is represented and tested, "
            "but honjia cutover remains blocked until hardware validation "
            "and legacy shim removal gates are recorded."
        ),
    }


def _build_software_closure_summary() -> dict[str, object]:
    return {
        "scope": "Wave 3 P0/P1 software closure",
        "state": CODE_COMPLETION_STATE,
        "code_level_complete": True,
        "honjia_cutover_complete": False,
        "hardware_verified": False,
        "legacy_body_runtime_detached": False,
        "full_detachment_claim_allowed": False,
        "truthfulness_rule": (
            "Code-level completion means the software gates are represented and tested; "
            "honjia cutover remains blocked until real hardware validation records parity."
        ),
        "completed": list(SOFTWARE_CLOSURE_COMPLETED),
        "blocked_by_hardware_validation": list(SOFTWARE_CLOSURE_HARDWARE_BLOCKERS),
        "blocked_by_legacy_detachment": _build_legacy_detachment_blockers(),
    }


def _build_legacy_detachment_blockers() -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for package in TRANSITIONAL_PACKAGES:
        package_name = str(package["package"])
        blockers.append(
            {
                "package": package_name,
                "paths": list(package["paths"]),
                "status": "blocked_by_legacy_shim_removal",
                "blocks_full_detachment": True,
                "removal_gate": LEGACY_SHIM_REMOVAL_GATES[package_name],
            }
        )
    return blockers


def _build_cutover_readiness_summary() -> dict[str, object]:
    return {
        "overall_state": "transitional",
        "hardware_verified": False,
        "code_completion_state": CODE_COMPLETION_STATE,
        "software_closure_state": "complete",
        "honjia_cutover": "blocked_by_hardware_validation",
        "completion_policy": "blocked_until_honjia_hardware_acceptance",
        "legacy_body_runtime_detached": False,
        "fake_completion_guard": FAKE_COMPLETION_GUARD,
        "native_provider_modules": list(NATIVE_PROVIDER_MODULES),
        "monitor_endpoints": list(MONITOR_ENDPOINTS),
        "legacy_shim_policy": _build_legacy_shim_policy(),
    }


def _build_legacy_shim_policy() -> dict[str, object]:
    shims: list[dict[str, object]] = []
    for package in TRANSITIONAL_PACKAGES:
        package_name = str(package["package"])
        shims.append(
            {
                **package,
                "classification": "transitional_shim",
                "blocks_full_detachment": True,
                "removal_gate": LEGACY_SHIM_REMOVAL_GATES[package_name],
            }
        )
    return {
        "state": "transitional",
        "legacy_body_runtime_detached": False,
        "legacy_paths_must_be_marked_as_shims": True,
        "full_detachment_claim_allowed": False,
        "policy": (
            "Legacy copies are allowed only as explicitly named transitional shims. "
            "Their presence blocks any claim that eihead is fully detached from legacy body runtime."
        ),
        "shims": shims,
    }


def _read_source_git_state(source_root: Path) -> dict[str, object]:
    commit = _run_git(source_root, "rev-parse", "HEAD")
    status_short = _run_git(source_root, "status", "--short").splitlines()
    return {
        "commit": commit or "unknown",
        "dirty": bool(status_short),
        "status_short": status_short,
    }


def _read_eiprotocol_source_state(
    source_root: Path,
    eiprotocol_repo_root: Path | None,
    *,
    source_state: dict[str, object],
) -> dict[str, object]:
    if eiprotocol_repo_root is None:
        return {
            "repository": "eiprotocol",
            "mode": "embedded_export",
            "repo_root": str(source_root),
            "paths": ["eiprotocol"],
            "commit": source_state["commit"],
            "dirty": source_state["dirty"],
            "status_short": list(source_state.get("status_short", [])),
        }

    protocol_state = _read_source_git_state(eiprotocol_repo_root)
    return {
        "repository": "eiprotocol",
        "mode": "independent_repo",
        "repo_root": str(eiprotocol_repo_root),
        "paths": ["eiprotocol"],
        **protocol_state,
    }


def _run_git(source_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _relative_output_path(target_path: Path, *, target_root: Path) -> str:
    # Callers only need stable manifest paths, not absolute local machine paths.
    return target_path.relative_to(target_root).as_posix()


def _should_skip(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    if any(part in SKIP_DIR_NAMES for part in rel_parts):
        return True
    return path.suffix in SKIP_SUFFIXES


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return not _same_path(parent, child)
    except ValueError:
        return False


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


if __name__ == "__main__":
    raise SystemExit(main())
