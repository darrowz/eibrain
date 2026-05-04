# eiprotocol v0.1 MVP

Status date: 2026-05-05

This MVP is the first shared protocol slice for `eihead`, `eibrain`,
`eimemory`, `eiskills`, `eidocs`, and future `eibody` work. It is based on the
JoyInside-inspired event specification in
`C:/Users/Darrow/Documents/Codex/2026-05-04/joyinsdie-joyinside/docs/eiprotocol-v0.1.md`,
but keeps the first implementation intentionally small and testable.

Repository state after the split: `/dev-project/eibrain` remains the source
repo, `/dev-project/eiprotocol` is the exported shared protocol repo, and
`/dev-project/eihead` is the exported head repo.

## What Enters MVP

- A versioned event envelope with `specVersion=eiprotocol/0.1`.
- Round and trace fields: `requestId`, `sessionId`, `roundId`,
  `correlationId`, `causationId`, and `traceId`.
- Source and target identity fields for `eihead`, `eibrain`, `eimemory`,
  `eiskills`, `eidocs`, `eitraining`, `user`, and future modules.
- Capability reporting through `ei.capability.manifest.report`.
- Device health/status shapes for camera, Hailo, microphone, speaker, and
  pan-only neck.
- Audio turn events using `ei.dialogue.asr.partial` and
  `ei.dialogue.asr.final`.
- Realtime vision frame observations using `ei.observation.vision.frame`.
- Head actions using `ei.action.request`, including idempotency keys for
  side-effecting operations.
- Execution outcomes using `ei.outcome.execution`.
- User feedback using `ei.outcome.user.feedback`.
- Event transport binding for the next eihead/eibrain batch: HTTP JSON
  `POST /events` carrying one envelope per request.
- Basic validation for required envelope fields, turn-scoped `roundId`, and
  idempotency on side-effecting action events.

## JoyInside Reference Points Adopted

The JoyInside reference is strongest around realtime turn semantics. The MVP
adopts these parts now:

- Event-first naming: `ei.<plane>.<subject>.<verb>[.<detail>]`.
- Stream/round readiness: all turn-scoped payloads carry `roundId`.
- Interrupt-ready action model: actions and outcomes can be correlated without
  relying on implicit local state.
- Capability-driven brain behavior: the brain should read a manifest instead of
  assuming hardware or backend availability.
- Feedback loop readiness: execution outcomes and user feedback are explicit
  protocol events, so `eimemory` and `eitraining` can consume them later.

## Deferred From MVP

These are useful but intentionally not implementation blockers for v0.1:

- Realtime transport streaming: SSE/WebSocket/MQTT, binary audio/video chunks,
  replay, resume, and backpressure.
- Full conversation state-machine enforcement.
- Full policy and safety-gate runtime. The envelope has `policy`, but v0.1 does
  not make safety decisions a runtime dependency because the current migration
  scope explicitly excludes the safety layer.
- Memory recall/write event models beyond the envelope conventions.
- TTS audio chunk streaming and phoneme/action synchronization.
- Signed policy decisions and tamper-evident audit records.
- Multi-device linked sessions.

## Current Code Shape

The implementation lives in top-level package `eiprotocol`:

- `EventEnvelope`, `SourceRef`, `TargetRef`, `PolicyState`
- `CapabilityManifest`, `Capability`, `DeviceStatus`
- `AudioTurn`, `RealtimeVisionObservation`, `Detection`
- `HeadAction`, `ExecutionOutcome`, `UserFeedback`
- `validate_event`
- `EventDefinition`, `list_event_names`, `get_event_definition`
- `validate_event_strict`, `assert_event_valid`, `ValidationIssue`
- `build_event`, specialized builders, and `EventIdFactory`
- `dumps_event`, `loads_event`, and canonical JSON helpers
- `classify_event` routing metadata for all cataloged v0.1 events

The package is included in `pyproject.toml`, exported into standalone
`eiprotocol` builds by `scripts/export-eiprotocol-repo.py`, and carried into
standalone `eihead` builds by `scripts/export-eihead-repo.py`. The standalone
export includes protocol modules, JSON fixtures, standalone-safe tests, and
these protocol docs. This keeps the current monorepo working while establishing
`/dev-project/eiprotocol` as the shared protocol repository. When the eihead
export is given `--eiprotocol-repo-root`, its manifest pins the independent
protocol revision.

## Required Next Supplements

The reference document is broad enough for v0.1, but the project still needs
three follow-up supplements before real cutover:

1. Transport binding: HTTP JSON `POST /events` routes that carry the envelope
   between `eihead` and `eibrain`; realtime streaming transports stay deferred.
2. Compatibility adapters: keep expanding conversion between existing
   `eibrain.protocol` / `eihead.protocol` classes and `eiprotocol` envelope
   events as more realtime surfaces move out of compatibility mode.
3. Golden fixtures: keep expanding examples as new realtime transports and
   device modules are added. The current fixture set covers every cataloged
   v0.1 event name.

Transport acceptance for the next batch:

- `POST /events` accepts JSON envelopes for capability, observation, action,
  outcome, and feedback events.
- Responses are JSON and report `processed`, `not_wired`, or `not_processed`
  explicitly.
- Missing handlers include a reason and never return blank payloads or
  fake-normal success.

Acceptance for this MVP is intentionally code-level: catalog/fixture parity,
strict validation tests, JSON codec round-trips, standalone export tests,
standalone `eihead` export import smoke, and the `/events` transport
truthfulness checks must pass.
