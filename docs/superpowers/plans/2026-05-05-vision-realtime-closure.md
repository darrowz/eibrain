# Vision Realtime Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the code-level realtime vision loop for honjia so live frame detections can be monitored, evented to eibrain, and used by pan-only neck tracking.

**Architecture:** Keep eye service, monitoring, protocol bridge, neck tracking, and runtime integration as separate modules. Each module can be tested with fakes and must not require real GStreamer/Hailo hardware.

**Tech Stack:** Python standard library, existing eihead/eibrain/eiprotocol packages, pytest.

---

### Task 1: Realtime Eye Service

**Files:**
- Create: `eihead/eye/service.py`
- Modify: `eihead/eye/__init__.py`
- Test: `tests/eihead/test_eihead_eye_service.py`

- [ ] Write failing tests for service `poll_once()`, `status()`, `latest_observation()`, and not-wired propagation.
- [ ] Implement `RealtimeEyeService` around an adapter or pipeline-like object.
- [ ] Verify fake adapter tests pass without GStreamer/Hailo.

### Task 2: Web Visual Diagnostic Overlay

**Files:**
- Modify: `eihead/monitoring/realtime_vision.py`
- Modify: `eihead/monitoring/web.py`
- Test: `tests/eihead/test_eihead_monitoring_realtime_vision.py`
- Test: `tests/eihead/test_eihead_monitoring_web.py`

- [ ] Write failing tests for overlay boxes, score labels, target center/error, and no-live-image truthfulness.
- [ ] Add normalized overlay/visual diagnostic fields to `/api/vision/realtime`.
- [ ] Add a lightweight HTML visual diagnostic panel.

### Task 3: eiprotocol Vision Event Bridge

**Files:**
- Modify: `eiprotocol/models.py`
- Modify: `eiprotocol/builders.py`
- Modify: `eiprotocol/catalog.py`
- Modify: `eiprotocol/event_routing.py`
- Modify: `eiprotocol/__init__.py`
- Modify: `eibrain/protocol/eiprotocol_bridge.py`
- Test: `tests/protocol/test_eiprotocol_vision*.py`
- Fixture: `tests/fixtures/eiprotocol/realtime_vision_frame.json`

- [ ] Write failing tests for typed roundtrip and realtime payload to `ei.observation.vision.frame`.
- [ ] Preserve detections, boxes, scores, tracked target, latency, metadata, source and target direction.
- [ ] Run strict validation and fixture tests.

### Task 4: Vision Target Tracking and Pan Planner

**Files:**
- Create: `eihead/eye/tracking.py`
- Create: `eihead/neck/vision_follow.py`
- Modify: `eihead/eye/__init__.py`
- Modify: `eihead/neck/__init__.py`
- Test: `tests/eihead/test_eihead_vision_tracking.py`

- [ ] Write failing tests for target selection, deadband hold, smoothing, max step, and target loss.
- [ ] Implement `select_tracking_target()` and `plan_pan_follow_action()`.
- [ ] Verify the planner never emits tilt for honjia.

### Task 5: Runtime Native Eye Integration

**Files:**
- Modify: `eihead/runtime/app.py`
- Modify: `eihead/runtime/native_providers.py`
- Modify: `eihead/runtime/http_api.py`
- Test: `tests/eihead/test_head_runtime_realtime_vision.py`
- Test: `tests/eihead/test_head_runtime_native_providers.py`

- [ ] Write failing tests for native provider service polling and readiness in capabilities.
- [ ] Prefer realtime native eye provider/service data over legacy snapshots.
- [ ] Keep static legacy snapshots from being promoted to realtime ready.

### Task 6: Integration Verification and Audit

**Files:**
- Create: `docs/vision-realtime-closure-audit.md`

- [ ] Run focused tests for eye, monitor, protocol, runtime, and tracking.
- [ ] Run full pytest and compileall.
- [ ] Document completion percentage and hardware-only residual risks.
- [ ] Commit, sync honxin `/dev-project/eibrain`, export `eihead`/`eiprotocol`, then push GitHub.
