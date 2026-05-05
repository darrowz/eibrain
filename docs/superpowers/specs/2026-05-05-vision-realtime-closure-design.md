# Vision Realtime Closure Design

Date: 2026-05-05

## Goal

Complete the code-level realtime vision loop so honjia can move from static-frame diagnostics to a live `/dev/video0` + `/dev/hailo0` validation path, with detections visible on the Web monitor and consumable by eibrain/eiprotocol.

## Scope

This pass completes code-level wiring only. It must not claim honjia hardware validation until the camera, Hailo device, and pan servo are tested on-site.

## Architecture

The vision loop is split into five bounded pieces:

- `eihead.eye`: owns realtime frame/detection service state, Hailo/GStreamer adapter boundaries, and normalized detection payloads.
- `eihead.monitoring`: turns the latest realtime observation into Web/API diagnostics, including boxes, scores, target center, and readiness truthfulness.
- `eiprotocol` and `eibrain.protocol`: carry realtime `ei.observation.vision.frame` events from eihead to eibrain with frame metadata, detections, latency, and tracked target.
- `eihead.neck`: converts selected visual targets into pan-only follow actions with deadband, smoothing, and step limits.
- `eihead.runtime`: discovers native eye providers and exposes realtime status to `/api/vision/realtime` without promoting legacy static snapshots as live video.

## Data Flow

1. `GStreamerHailoRealtimeAdapter` reads frames from `/dev/video0`, parses Hailo ROI metadata from `/dev/hailo0`, and returns `RealtimeEyeStatus`.
2. `RealtimeEyeService` keeps the latest status/observation and exposes `poll()`, `status()`, and JSON-like realtime observations.
3. Runtime `vision_realtime()` resolves native provider/service status and passes it to Web diagnostics.
4. Web diagnostics normalizes boxes/scores and emits an overlay payload for visual diagnosis.
5. Bridge helpers convert realtime observations into `ei.observation.vision.frame` events.
6. Tracking planner selects the target and emits pan-only neck action suggestions for the neck lane.

## Truthfulness Rules

- Static image compatibility data must remain marked as compatibility mode and must not set `stream_ready=true`.
- Missing `/dev/video0`, `/dev/hailo0`, GStreamer, HEF, or frame reader must report `not_wired` or `degraded`.
- The Web page may show diagnostic boxes and overlay metadata without a live image, but must clearly say no live frame image is available.
- Pan follow logic must never output tilt actions on the current honjia hardware.

## Acceptance Criteria

- Tests cover service polling, not-wired/degraded propagation, overlay payload, protocol event roundtrip, runtime native provider integration, and pan-only tracking.
- `python -m pytest -q tests/eihead/test_eihead_eye_* tests/eihead/test_head_runtime_* tests/protocol/test_eiprotocol_vision*.py` passes.
- Full repository tests pass before final completion claim.
- honxin/GitHub sync happens only after local verification succeeds.

## Residual Hardware Gates

- Live FPS, frame age, Hailo parse error rate, and pan stability require honjia现场复测.
- HEF/postprocess paths may need device-specific config after code lands.
