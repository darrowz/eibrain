# ei-hongtu Topology

`ei-hongtu` is the umbrella system for Hongtu's embodied intelligence stack. It
keeps hardware-facing runtime, cognitive orchestration, memory, and future body
control in separate repositories while preserving one coherent agent identity.

## Repository Boundaries

```text
ei-hongtu
├── eiprotocol  shared events, observations, actions, and capabilities
├── eihead      (honjia)  real-device head runtime
├── eibrain     (honxin)  cognitive orchestration
├── eimemory    (honxin)  durable memory runtime
└── eibody      (reserved) future body/base runtime
```

### eiprotocol

`eiprotocol` should become the common contract layer across EI projects. It
prevents every project from inventing its own event, action, and observation
shape as the system grows.

Initial protocol families:

- `events`: runtime events, lifecycle events, execution outcomes, feedback
- `observations`: audio turns, vision frames, device state, memory recall results
- `actions`: speech, neck movement, attention, stop, pause, tool calls
- `capabilities`: device and model capability registration

`eibrain`, `eihead`, `eimemory`, `eiskills`, and `eidocs` should all depend on
this shared protocol instead of exchanging ad hoc dictionaries long term.

### eihead

`eihead` owns the honjia real-device head stack. It should absorb the current
hardware-facing modules from `eibrain`:

- camera and Hailo vision services
- microphone capture, VAD, ASR, and audio diagnostics
- TTS playback and mouth output
- neck/servo control
- local monitoring for real-device status
- honjia-specific systemd units and deployment configuration

`eihead` publishes observations and device state. It consumes action plans from
`eibrain`.

On startup, `eihead` should register its capabilities with `eibrain`:

- available cameras, microphones, speakers, neck/servo devices
- supported ASR, TTS, vision, embedding, and hardware backends
- current device health, driver state, latency, and degradation status
- physical and action limits such as angle ranges and rate limits

This lets the brain adapt to a swapped head device without hard-coding honjia
hardware assumptions.

### eibrain

`eibrain` owns cognition on honxin:

- dialogue policy and LLM routing
- multimodal turn fusion
- intent planning and action planning
- interaction state and agent identity
- memory policy coordination with `eimemory`
- skill invocation through `eiskills`
- knowledge/tool integration through `eidocs`

`eibrain` should not import honjia hardware drivers directly after the split.

### eimemory

`eimemory` owns long-term memory:

- working, episodic, semantic, and procedural memory records
- recall and writeback RPC for `eibrain`
- source intake and governance
- experience exports for `eitraining`

### eibody

`eibody` is reserved for future body/base runtime. It should later own hardware
that is not part of the head stack, such as chassis, arms, locomotion, or larger
actuator groups.

## Runtime Topology

```text
honjia / eihead
  camera + Hailo + microphone + speaker + neck servo
        │ observations / status
        ▼
honxin / eibrain
  multimodal fusion + dialogue + action planning
        │ recall / writeback
        ▼
honxin / eimemory
  durable memory + governance

eibrain ── skill intents ──> eiskills
eibrain ── knowledge tasks ─> eidocs
eimemory ─ experience ─────> eitraining ─ candidates ─> eiskills
eihead/eibrain execution outcomes ───────> eimemory/eitraining
```

## Split Contract

The split should be contract-first. The initial contract can stay JSON/HTTP so
the current honjia deployment remains easy to debug.

### eihead to eibrain

- `HeadObservation`
- `AudioTurn`
- `VisionFrame`
- `DeviceStatus`
- `AttentionState`
- `CapabilityManifest`
- `ExecutionOutcome`
- `UserFeedback`

### eibrain to eihead

- `SpeakAction`
- `MoveHeadAction`
- `AttentionIntent`
- `StopSpeechAction`
- `DeviceCommand`

## Deferred Safety And Permission Layer

The safety and permission layer is intentionally deferred for the first
`eihead` split. The immediate goal is to stabilize the protocol boundary,
capability registration, runtime monitoring, and feedback records without
changing live interaction behavior.

Keep the protocol extensible enough to add safety policy later, but do not
introduce new permission gates in the first migration.

## Feedback And Training Loop

Every meaningful action should produce an outcome record:

- what was attempted
- which module planned it
- which device or service executed it
- whether it succeeded
- latency and error details
- user satisfaction or correction when available
- suggested adjustment for next time

Outcomes should flow into `eimemory` as experience records and into
`eitraining` for replay, evaluation, skill promotion, rollback, and behavior
improvement.

## Migration Plan

1. Create `/dev-project/eihead` as the canonical honjia head runtime repository.
2. Define the first `eiprotocol` package or compatibility module.
3. Add `eihead` capability registration and eibrain capability ingestion.
4. Move `apps/body_runtime` into `eihead/apps/head_runtime`.
5. Move `eibrain/body` into `eihead/eihead/body`.
6. Move honjia configs and systemd units into `eihead/config` and
   `eihead/deploy/systemd`.
7. Keep compatibility wrappers in `eibrain` while honjia services transition.
8. Replace `eibrain-sync-honjia` with an `eihead` deployment sync command.
9. Route execution outcomes into `eimemory` and `eitraining`.
10. Defer safety and permission redesign until the split is stable.

## Non-goals For The First Split

- Do not redesign `eimemory` during the first `eihead` extraction.
- Do not move cognitive policy into `eihead`.
- Do not introduce a complex message bus before HTTP/status-file contracts are
  proven insufficient.
- Do not rename live honjia services until compatible service wrappers exist.
