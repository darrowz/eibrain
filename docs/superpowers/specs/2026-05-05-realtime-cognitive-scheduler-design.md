# Realtime Cognitive Scheduler Design

Status date: 2026-05-05

## Goal

Upgrade eibrain from a serial dialogue manager into a JoyInside-style realtime
cognitive scheduler: observe partial signals, produce fast non-committal
feedback, run slower reasoning in the background, arbitrate stable speech and
actions, and cancel stale rounds without leaking ghost replies.

This is a code-level completion target. Honjia hardware validation remains a
separate cutover gate.

## Current Baseline

- `eibrain.cognition.realtime.turn` already has a minimal
  `RealtimeTurnManager`, `TurnBlackboard`, cancellation tokens, fast hypotheses,
  stable decisions, an arbiter guard, and interruption summaries.
- `eiprotocol` already defines realtime dialogue events including ASR partial,
  fast hypothesis, stable decision, interrupt requested, TTS, memory, outcome,
  and training-oriented events.
- `apps.body_runtime` and `eihead.monitoring.voice` expose round, scheduler,
  interruption, and cancellation telemetry to the Web monitor.
- `eimemory` adapters and multimodal memory policy exist, but they are not yet
  orchestrated as realtime prefetch/writeback decisions.

## Target Architecture

The scheduler has three cooperating lanes that communicate only through a
per-round blackboard and stable protocol events:

- Lane A: `Listening + Fast Think`
  Writes non-stable hypotheses, micro-feedback suggestions, emotion hints, and
  memory prefetch requests from ASR partials, vision hints, and environment
  signals.
- Lane B: `Slow Thinking`
  Builds stable decisions from final text, hypotheses, memory candidates,
  persona state, and tool/action candidates. It is cancellation-aware.
- Lane C: `Speaking`
  Consumes only arbiter-approved stable speech/action plans. It stops when the
  round token is cancelled.

Core flow:

```text
observe -> hypothesize -> prefetch -> decide -> speak/act -> revise -> stop/resume
```

## Components

- `TurnBlackboard`: per-round state store for observations, hypotheses,
  memory candidates, persona state, emotion/environment state, speech/action
  plans, and cancellation state.
- `FastThinkEngine`: derives safe micro-feedback and intent hypotheses from
  partial observations. It cannot make final claims or commit facts.
- `MemoryOrchestrator`: prepares recall and writeback proposals without
  blocking realtime speech.
- `SlowReasoner`: converts final observations plus memory/persona context into
  stable response decisions.
- `ResponseArbiter`: decides which output can be spoken immediately, which must
  wait, and which stale output must be rejected.
- `InterruptionController`: cancels old LLM/RAG/TTS/action work by roundId and
  cancellation token, records interruption summaries, and starts the next round.
- `SpeechActionPlanner`: emits structured speech segments and action segments
  with offsets, emotion, style, fallback text, and device capability IDs.
- `PersonaRuntime`: runtime state for speaking style, voice, emotional policy,
  action style, and memory policy. It is not just a prompt string.
- `EmotionContextBuilder`: normalizes prosody, environment, and vision hints
  for fast/slow reasoning.
- `ProactiveActivityManager`: produces low-interruption proactive activity
  proposals only when usefulness and disturbance checks pass.

## Explicit Non-Goals

- No claim of completed honjia hardware validation.
- No forced safety policy expansion beyond a simple adapter seam.
- No replacement of the actual ASR/TTS provider stack in this phase.
- No requirement for a live WebSocket server in this phase; events and runtime
  APIs must make a future WebSocket transport straightforward.

## Acceptance Criteria

- Round-scoped calls include `round_id` and `cancellation_token`.
- Fast lane can emit a micro-feedback plan in <= 500 ms in synthetic tests.
- Slow lane can commit a stable decision without blocking fast lane state.
- Arbiter rejects unstable hypotheses and stale/cancelled round output.
- Interruption stops future speech/action consumption for the old round.
- Speech/action plans are structured and can be serialized to eiprotocol.
- Persona, emotion/environment, memory prefetch, and proactive activity have
  deterministic test coverage.
- Monitoring/export readiness reflects code-level completion while hardware
  cutover remains blocked.
