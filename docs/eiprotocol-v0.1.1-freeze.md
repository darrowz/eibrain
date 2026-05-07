# eiprotocol v0.1.1 Freeze

Status date: 2026-05-05

This document freezes the eiprotocol v0.1.1 boundary. The goal is to make the
shared protocol package stable without treating the full JoyInside product
surface as required protocol implementation work.

v0.1.1 is a protocol contract freeze, not a runtime product freeze. Consumers
may build richer JoyInside behavior on top of these events, but the standalone
`eiprotocol` package only owns schemas, catalog metadata, routing metadata,
validation, fixtures, and exportable docs/tests for the frozen event surface.

## v0.1.1 Frozen Scope

The following areas are frozen for v0.1.1 and should be treated as the 100%
completion bar for the protocol package:

- Envelope: versioned event envelope fields, source/target refs, policy state,
  trace/correlation IDs, timestamps, canonical `ttlMs` output with `ttl_ms`
  input compatibility, and JSON round-trip behavior.
- Catalog: known event names and metadata for plane, type, realtime,
  round-scoped, and side-effecting semantics.
- Routing: classification metadata for every cataloged v0.1.1 event, including
  compatibility-safe action request routing.
- Validation: strict and non-strict validation for required envelope fields,
  enum values, timestamps, idempotency keys, and optional known-event checks.
- Fixtures: golden JSON examples for every frozen event family.
- Standalone export: protocol modules, fixtures, docs, and standalone-safe
  tests export cleanly without importing `eibrain`, `eihead`, or runtime
  device modules.
- Vision observations: `vision.frame`, `vision.scene`, and `vision.event`
  protocol events are part of the frozen observation surface; scene/event
  observations are session-addressable but not round-scoped by default.
- Memory policy: `memory.policy.report` is the frozen memory-policy exchange
  event. It reports policy posture and proposed/affected `writes`; it does not
  implement memory storage.
- Speech-action plan: speech-to-action planning shapes are frozen at the event
  contract level so dialogue, action request, and outcome events can be
  correlated.
- Realtime dialogue events: ASR partial/final, fast hypothesis, stable
  decision, agent deltas/finals, TTS deltas/finals, and interrupt requests are
  frozen as protocol events.

## v0.2+ Deferred Scope

The following areas are intentionally deferred to v0.2+ or product/runtime
repositories. They are not blockers for the v0.1.1 protocol freeze:

- WebSocket/SSE/MQTT runtime transport implementations.
- Binary media transport for audio, video, or other media payloads.
- Scene graph algorithms or perception model internals behind `vision.scene`.
- Safety runtime execution, policy engines, or live enforcement loops.
- Long-term memory implementation, retrieval, indexing, embedding, or storage.
- JoyInside role ecosystem, agent society behavior, persona orchestration, and
  product-level capability composition.

## Boundary Rule

If a feature requires a server loop, device driver, model provider, media
stream, policy engine, database, or JoyInside role runtime, it belongs outside
the v0.1.1 protocol package unless it can be represented as a small,
validated event contract with fixtures. The protocol may name the exchange; it
must not absorb the product implementation.
