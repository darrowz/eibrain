# eiprotocol v0.1.1 Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze `eiprotocol` as a contract-complete v0.1.1 shared package without pulling runtime, device, or model-provider implementation into the protocol layer.

**Architecture:** Keep `eiprotocol` transport-agnostic and dependency-clean. Add only event contracts, typed payload helpers, golden fixtures, validation/conformance reporting, and docs that make the v0.1.1 boundary explicit.

**Tech Stack:** Python dataclasses, pytest, JSON fixtures, standalone export scripts.

---

## Scope

In scope:

- Add missing v0.1.1 protocol contracts for realtime world-state vision and multimodal memory policy.
- Preserve existing v0.1 envelope compatibility.
- Keep `eiprotocol` free of imports from `eibrain`, `eihead`, hardware runtimes, or provider SDKs.
- Add conformance checks that compare catalog, fixtures, routing, validation, and standalone export.
- Update docs to mark v0.1.1 as a frozen protocol slice.

Out of scope:

- WebSocket/SSE/MQTT runtime implementation.
- Binary audio/video transport.
- Safety policy runtime execution.
- Hailo, ASR, TTS, LLM, eimemory indexing, or hardware adapter code.
- JoyInside role ecosystem or product content templates.

## Parallel Task Ownership

### Task A: Core v0.1.1 Protocol Events

**Files:**
- Modify: `eiprotocol/catalog.py`
- Modify: `eiprotocol/event_routing.py`
- Modify: `eiprotocol/models.py`
- Modify: `eiprotocol/builders.py`
- Modify: `eiprotocol/__init__.py`
- Create: `tests/protocol/test_eiprotocol_v011_vision_memory.py`
- Create/update: `tests/fixtures/eiprotocol/vision_scene.json`
- Create/update: `tests/fixtures/eiprotocol/vision_event.json`
- Create/update: `tests/fixtures/eiprotocol/memory_policy.json`

**Requirements:**
- Add `ei.observation.vision.scene`.
- Add `ei.observation.vision.event`.
- Add `ei.memory.policy.report`.
- Provide typed dataclasses and builders for all three.
- Ensure strict validation, codec roundtrip, fixture roundtrip, and routing classification.

### Task B: Conformance Report

**Files:**
- Create: `scripts/eiprotocol_conformance_report.py`
- Create: `tests/infra/test_eiprotocol_conformance_report.py`

**Requirements:**
- Report catalog count, fixture parity, routing parity, strict validation parity, builder/model coverage where detectable, and dependency cleanliness.
- Exit non-zero when required v0.1.1 fixtures or catalog routes are missing.
- Output JSON for automation and a stable text summary for humans.

### Task C: Standalone Export Hardening

**Files:**
- Modify: `scripts/export-eiprotocol-repo.py`
- Modify: `tests/infra/test_export_eiprotocol_repo.py`

**Requirements:**
- Ensure new v0.1.1 docs, fixtures, conformance script, and standalone-safe tests export cleanly.
- Standalone export must run protocol tests and conformance report without `eibrain` or `eihead`.

### Task D: Docs Freeze

**Files:**
- Modify: `docs/eiprotocol-v0.1-mvp.md`
- Modify: `docs/eiprotocol-hardening-checklist.md`
- Create: `docs/eiprotocol-v0.1.1-freeze.md`

**Requirements:**
- Define what v0.1.1 freezes and what remains v0.2+.
- Include event matrix for all cataloged events.
- State compatibility rules and non-goals clearly.

### Task E: Bridge Compatibility Audit

**Files:**
- Modify: `eibrain/protocol/eiprotocol_bridge.py`
- Modify: `tests/protocol/test_eiprotocol_bridge.py`

**Requirements:**
- Add bridge helpers only where needed for new generic payloads.
- Do not make `eiprotocol` depend on `eibrain`.
- Preserve old bridge tests and add v0.1.1 payload routing tests.

## Acceptance

- `python -m pytest -q tests/protocol tests/infra/test_export_eiprotocol_repo.py tests/infra/test_eiprotocol_conformance_report.py`
- `python -m pytest -q`
- `python -m compileall -q eiprotocol eibrain/protocol scripts tests`
- `python scripts/eiprotocol_conformance_report.py --strict`
- `git diff --check`
