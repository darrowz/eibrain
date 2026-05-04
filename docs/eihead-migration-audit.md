# eihead Migration Audit Matrix

Status date: 2026-05-04

This document audits the current eihead split after the standalone repository
scaffold exists. The target is executable migration guidance only. It does not
authorize code changes outside the eihead migration surface, and it keeps the
honjia real-device chain as the acceptance baseline.

## Current State

The eihead scaffold is present and useful, but it is still transitional. The
new `eihead` package has native runtime, HTTP, monitoring, config, capability,
and small local protocol pieces. The honjia hardware behavior is still owned by
the old body runtime copied into the export shape:

- Native eihead: `eihead.runtime`, `eihead.monitoring`, `eihead.services`,
  `eihead.protocol`, `apps.head_runtime`, `config/eihead.honjia.yaml`, and
  `deploy/systemd/eihead-*.service`.
- Transitional compatibility: `eihead.runtime.app.HeadRuntimeApp` delegates to
  `apps.body_runtime.BodyRuntimeApp`; `apps.head_runtime` is only an alias to
  the native wrapper.
- Legacy still carried by `/dev-project/eihead`: `apps.body_runtime`,
  `eibrain.body`, and a compatibility subset of `eibrain.protocol`.
- eibrain-side native bridge: `eibrain.infra.head_client.HeadClient` and
  `eibrain.infra.head_registry.HeadRegistry` are useful eibrain integration
  pieces, not honjia hardware ownership.

Until the legacy body modules are renamed and made import-independent, the
standalone eihead repository is an export shell around the proven honjia body
runtime for part of the chain, while ear/mouth are already native but not yet
fully complete.

The formal Eye target for `/dev-project/eihead` is realtime stream detection:
continuous `/dev/video0` camera frames and `/dev/hailo0` Hailo detections
producing live detections, `RealtimeVisionObservation` payloads, runtime status,
and monitor boxes. Static-image detection remains a
compatibility/test placeholder only and is not the deployment direction.
The native eye split now treats `eihead/eye/gstreamer.py` as the realtime
appsink reader boundary and `eihead/eye/hailo_metadata.py` as the Hailo ROI
metadata parser boundary; `eihead/eye/adapters.py` composes those pieces and
keeps missing hardware/plugins truthful as `not_wired`.
The new native voice boundary is `eihead/ear`, `eihead/mouth`, and
`eihead/monitoring/voice.py`, with endpoint hooks prepared in
`/api/voice/realtime` and `/api/audio/realtime`. This chain currently remains
functional-not-complete, but it has entered a scheduler-backed functional stage:
round lifecycle, scheduler state, and interrupt visibility can be shown in Web.
It is not yet wired to real streaming LLM/TTS, so missing stages must stay
explicit as `not_wired`, `unknown`, or `degraded`. The current closed-loop
voice diagnostics are functional offline/quasi-streaming diagnostics, not
hardware-verified real streaming.

## Audit Matrix

| Area | State | Current evidence | Next migration action | Acceptance gate |
| --- | --- | --- | --- | --- |
| Repository scaffold | Native scaffold | `eihead/`, `apps/head_runtime/`, `config/eihead.honjia.yaml`, `deploy/systemd/eihead-*.service` exist. | Keep as the target shape for all next moves. | Fresh exported `/dev-project/eihead` installs and runs `eihead-runtime status`. |
| Runtime CLI/API | Native wrapper | `eihead.runtime.cli` exposes `status`, `verify`, `http`, and `monitor`; HTTP serves `/health`, `/status`, `/capabilities`, `/actions`. | Keep stable while replacing the delegated internals module by module. | Endpoint payloads remain JSON object responses with explicit errors and no silent placeholder success. |
| Runtime app | Transitional | `HeadRuntimeApp` calls `apps.body_runtime.app.BodyRuntimeApp.from_config_path()` and reports `delegate: apps.body_runtime.BodyRuntimeApp`. | Replace delegate calls with native `eihead.eye`, `eihead.neck`, `eihead.ear`, and `eihead.mouth` services one at a time. | `HeadRuntimeApp.snapshot()` no longer requires `apps.body_runtime` after the final organ migration. |
| Native monitor | Native UI, legacy data source | `eihead.monitoring.web` owns the new monitor on `18080`, but status comes through the delegated body snapshot. | Keep the HTTP contract; change only data providers as organs become native. | Every field is real, `unknown`, or `not wired`; no fake `healthy` values. |
| Capability registry | Native | `eihead.services.capability_registry` declares honjia camera, Hailo, I2C, microphone, speaker, neck, ASR, TTS, and vision backend. | Feed it native per-organ probe results instead of only static path checks. | `/capabilities` shows online/degraded/offline with device paths and last-ok timestamps. |
| Local eihead protocol | Partly native | `eihead.protocol` has local action/outcome classes so eihead does not import `eibrain.protocol` for basic actions. | Replace the local mirror and eibrain compatibility classes with shared `eiprotocol` models. | eihead and eibrain import the same protocol package without cyclic dependency. |
| Shared protocol | Legacy source | Rich head contracts currently live under `eibrain.protocol`, and export keeps a compatibility subset. | Split to `/dev-project/eiprotocol` after organ behavior is stable enough to lock message names. | JSON round-trip tests pass for capability, observation, action, outcome, and feedback payloads. |
| Eye | Native boundary emerging | Camera/Hailo/realtime stream state logic still has legacy consumers in `apps.body_runtime.vision_hailo_service`, `eibrain.body.runtime_linux`, `eibrain.body.vision_state`, and `eibrain.body.organs.eye`; native export now carries `eihead/eye/realtime.py`, `eihead/eye/adapters.py`, `eihead/eye/gstreamer.py`, `eihead/eye/hailo_metadata.py`, and `eihead/monitoring/realtime_vision.py`. | Finish wiring native `eihead.eye` as realtime stream detection while preserving `/tmp/eibrain-vision/latest.jpg` and `/tmp/eibrain-vision/state.json` until consumers are changed. | honjia still captures `/dev/video0`, runs `/dev/hailo0`, continuously publishes detections, and renders boxes/scores/parser readiness on `18080`; static-image detection is compatibility/test-only. |
| Neck | Legacy | Pan/yaw control remains in `eibrain.body.neck_control`, `eibrain.body.raspbot_driver`, and `eibrain.body.organs.neck`. | Move to native `eihead.neck` with pan-only capability and explicit tilt rejection. | Manual pan command moves without oscillation; tilt requests fail as unsupported; `/dev/i2c-1` ownership is unchanged. |
| Ear | Partly native | Export now carries `eihead/ear/realtime.py`, `eihead/ear/__init__.py`, and `eihead/monitoring/voice.py`; runtime endpoints `/api/voice/realtime` and `/api/audio/realtime` are present in exports. Current closed-loop diagnostics are functional offline/quasi-streaming diagnostics, not hardware-verified real streaming. | Move to native `eihead.ear` end-to-end and keep round/scheduler/interrupt telemetry visible while real streaming LLM/TTS remains unwired. | Voice wake and one full ASR turn pass on honjia with measured stage latency, and no fake healthy speech state before streaming LLM/TTS completion. |
| Mouth | Partly native | TTS/playback contracts are now in `eihead/mouth/playback.py` and `eihead/mouth/__init__.py`, with monitor bridge at `eihead/monitoring/voice.py`. | Keep `speak`, `stop_speech`, and playback busy-state visibility while scheduler wiring finishes. | TTS is audible on honjia, stop works, and monitor shows synthesis/playback state from real data or explicit `not_wired/unknown`. |
| Export script | Transitional | `scripts/export-eihead-repo.py` intentionally copies `apps/body_runtime`, `eibrain/body`, `eibrain/infra`, `eibrain/protocol`, and minimal `eibrain/cognition/realtime` scheduler primitives. | Remove legacy copies only after each native replacement has parity; keep docs and config export. | Exported repo has no runtime dependency on `apps.body_runtime`, `eibrain.body`, or `eibrain.cognition.realtime` except named deprecation shims. |
| Deployment | Native templates, legacy fallback | `eihead-runtime.service` uses port `18081`; `eihead-monitor.service` keeps operator port `18080`; old eibrain service templates still exist for rollback. | Do not enable eihead services permanently until parity checks pass. | After reboot, eihead services own `18080`/`18081`; rollback can restore old eibrain body services in one downtime window. |

## Next Migration Order

The next round should run in this order: eye, neck, ear, mouth, protocol,
export, deploy. The order minimizes risk by moving observable, mostly
head-local data first, then action control, then the higher-risk audio loop,
then shared contracts and packaging.

### 1. Eye

Move the camera, Hailo, frame-state, and eye organ code into `eihead.eye` as
realtime stream detection. Keep the current state file paths during this step:

- `/tmp/eibrain-vision/latest.jpg`
- `/tmp/eibrain-vision/state.json`

Acceptance:

- `eihead-runtime status` reports frame age, detection count, top detection,
  backend, model, and stale/error state.
- Native eye status reports pipeline text/name, camera/Hailo device paths,
  readiness message, and parser error count when available.
- `http://honjia:18080` continues to show real camera/Hailo status and
  detection boxes.
- Static-image detection, if kept, is labelled compatibility/test-only and is
  not used as the native Eye acceptance path.
- Only one service owns `/dev/video0` and `/dev/hailo0`.
- The native eye module does not import `eibrain.body`; any temporary import is
  isolated behind an explicitly named compatibility adapter.

### 2. Neck

Move yaw/pan control into `eihead.neck`. Keep honjia as pan-only unless real
tilt hardware is installed and accepted later.

Acceptance:

- `move_head` with `axis: yaw` or `axis: pan` reaches the same Raspbot/I2C
  command path and reports an execution outcome.
- `axis: tilt` returns unsupported instead of pretending success.
- Manual pan-only test moves to target angle and settles without oscillation.
- Monitor shows target angle, last command status, and suppression reason from
  real state.

### 3. Ear

Move microphone capture, VAD policy, ASR recognizers, transcript cleanup, and
voice turn status into `eihead.ear`. This is after eye/neck because audio is
the most user-visible loop and should be migrated with more stable monitoring.

Acceptance:

- Voice wake produces a non-empty ASR turn with trace id and stage latency.
- Existing sherpa-ONNX model path, `plughw:CARD=U4K,DEV=0`, sample rate,
  channel count, VAD thresholds, and transcript replacements remain honored.
- Ear pauses or suppresses capture while mouth playback is active.
- Monitor shows VAD, recording, ASR text, confidence/status, and errors from
  real runtime state.

### 4. Mouth

Move TTS planning, synthesis, playback, stop, and speech-busy state into
`eihead.mouth`.

Acceptance:

- `speak` action produces audible playback on honjia and returns
  `eihead.execution_outcome.v1`.
- `stop_speech` interrupts or reports unsupported with a real reason.
- Ear can read mouth busy state before recording.
- Monitor shows provider/model, synthesis latency, playback status, last text
  length, and error tail without placeholder health.

### 5. Protocol

Split the shared contract to `/dev-project/eiprotocol` after the organ behavior
names are stable. The first `eiprotocol` cut should cover only what eihead and
eibrain both need:

- Capability manifest and device status.
- Audio turns and vision observations.
- Head actions: speak, stop speech, move head, set attention, capture frame.
- Execution outcomes and user feedback.
- Envelope fields: protocol version, kind, source, target, trace id,
  timestamp, and payload.

Acceptance:

- eihead does not require `eibrain.protocol` at runtime.
- eibrain does not import `eihead.protocol`.
- Unknown fields are accepted and preserved or ignored safely.
- JSON round-trip tests prove backward compatibility for current honjia
  payloads.

### 6. Export

Update `scripts/export-eihead-repo.py` only after the native modules pass their
parity gates. The export should progressively stop copying legacy code.

Acceptance:

- The generated `/dev-project/eihead` package installs in a clean environment.
- `eihead-runtime status`, `eihead-runtime http`, and `eihead-runtime monitor`
  start without `apps.body_runtime` or `eibrain.body`.
- A search of the exported runtime code shows legacy references only in
  deprecation shims, migration docs, and tests that assert removal.
- `docs/eihead-*.md`, `config/eihead*.yaml`, and `deploy/systemd/eihead-*.service`
  remain exported.

### 7. Deploy

Deploy only after export is clean enough to run without legacy body imports.
Use the existing short downtime strategy.

Acceptance:

- Before cutover, rerun the Phase 0 baseline from
  `docs/eihead-cutover-checklist.md`.
- Stop old eibrain monitor/body/vision services before starting eihead so
  ports and devices are not double-owned.
- Start `eihead-runtime.service` on `18081` and `eihead-monitor.service` on
  `18080`.
- Enable boot persistence only after voice, vision, neck, monitor, and rollback
  checks pass.
- Record deployed git revision, config path, data path, service names, and
  owner.

## Honjia Chain That Must Not Break

These are hard constraints for every migration PR and deployment attempt:

- Runtime paths stay separated: honxin source is `/dev-project/eibrain` or
  `/dev-project/eihead`; honjia runtime is `/opt/eihead/current`; honjia config
  is `/etc/eihead`; mutable runtime data should be under `/var/lib/eihead`
  once introduced.
- Operator monitor remains on honjia port `18080`; runtime API remains on
  `18081`.
- Camera and Hailo path remain `/dev/video0` and `/dev/hailo0`.
- Vision compatibility state remains readable from `/tmp/eibrain-vision` until
  all monitor and tracking consumers are migrated.
- Neck hardware remains pan/yaw-only on `/dev/i2c-1`; tilt is not advertised.
- Microphone remains `plughw:CARD=U4K,DEV=0` through `/dev/snd`; ASR remains
  sherpa-ONNX unless an explicit later change is accepted.
- Mouth playback remains audible through the configured default speaker path;
  TTS credentials and provider settings stay outside source control.
- eibrain cognition, personality, LLM routing, memory policy, and training
  policy stay in eibrain; eihead only owns head sensing, local actuation,
  health, and execution outcomes.
- Rollback remains service-level: stop `eihead-*`, start the previous
  `eibrain-*` services, and do not overwrite source repositories during
  rollback.

## Per-Stage Definition Of Done

A module is considered native only when all of these are true:

- Runtime import path is under `eihead.*`.
- It can run in the exported `/dev-project/eihead` repo without importing
  `apps.body_runtime` or `eibrain.body`.
- Any remaining legacy compatibility is named as a shim or adapter and has a
  removal condition.
- Status, capability, action, and outcome payloads include trace ids where the
  flow crosses eihead/eibrain.
- The `18080` monitor displays real data or explicit offline/unknown/not-wired
  states.
- Unit tests pass without honjia hardware using fakes, and honjia manual checks
  pass on the real chain before deployment.

The final split is accepted only when the Phase 0 baseline can be repeated
after deploy with equal or better results for voice wake, conversation, vision
detection, pan-only neck movement, and monitor truthfulness.
