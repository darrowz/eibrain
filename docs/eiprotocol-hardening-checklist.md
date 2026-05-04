# eiprotocol Hardening Checklist

Status date: 2026-05-05

This checklist defines the completion bar for the eiprotocol v0.1 MVP shared by
`eihead`, `eibrain`, `eimemory`, `eiskills`, `eidocs`, and future body modules.
The goal is a stable event contract that supports realtime multimodal dialogue,
interrupts, semantic actions, memory exchange, and outcome feedback without
forcing one transport or runtime implementation.

## Completion Target

The eiprotocol MVP can be treated as 98%+ complete when all items below are
implemented and verified:

- Event envelope round-trips through `EventEnvelope`, JSON codec, fixtures, and
  standalone package export.
- Known event names are discoverable from a catalog with plane, type, realtime,
  round-scoped, and side-effecting metadata.
- Strict validation catches invalid envelope fields, invalid enum values,
  missing idempotency keys, malformed timestamps, and unknown event names when
  requested.
- Builders exist for common event creation paths so callers do not handcraft
  incomplete envelopes.
- Routing classifies all v0.1 core events and preserves backward-compatible
  action request fields used by `eihead`.
- Golden fixtures cover control, capability, observation, dialogue, action,
  policy, memory, outcome, training, and error planes.
- Standalone export includes all eiprotocol modules, fixtures, and
  standalone-safe protocol tests.
- Existing `eibrain` compatibility bridge tests still pass in the monorepo.
- Unknown future events remain forward-compatible unless strict known-event
  validation is explicitly requested.

## Core Event Planes

- `control`: hello, ping, pong, resume, ack, error.
- `capability`: manifest report.
- `observation`: audio chunk, realtime vision frame.
- `dialogue`: ASR partial/final, agent delta/final, TTS delta/final,
  interrupt requested.
- `action`: request, dispatch, progress, complete, emergency stop.
- `policy`: policy decision.
- `memory`: recall request/result, write proposed/committed.
- `outcome`: execution outcome, user feedback.
- `training`: learning signal.
- `error`: protocol or runtime error event such as `ei.error.event`.

## Non-Goals

- WebSocket/SSE/MQTT transport implementation.
- Binary audio/video frame transport.
- Safety policy runtime execution.
- Memory indexing or retrieval internals.
- LLM, ASR, TTS, or vision model provider APIs.

These remain consumers of the protocol, not part of the protocol package.

## Acceptance Commands

Run from the source repository:

```bash
python -m pytest -q tests/protocol tests/infra/test_export_eiprotocol_repo.py
python -m compileall -q eiprotocol eibrain/protocol scripts
```

Run after exporting the standalone package:

```bash
python -m pytest -q
python -m compileall -q eiprotocol
```

## Integration Rules

- `eiprotocol` must not import `eibrain`, `eihead`, or device runtime modules.
- Compatibility bridges may import `eiprotocol`, but not the other way around.
- Validation helpers must return structured issues instead of throwing unless
  the caller chooses an assert-style API.
- Runtime routers may record known-but-unhandled events as diagnostics, but
  they must not report fake processing.
- Side-effecting action events must carry an idempotency key.
