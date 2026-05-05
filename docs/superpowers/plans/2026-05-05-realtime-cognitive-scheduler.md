# Realtime Cognitive Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a JoyInside-inspired realtime cognitive scheduler with fast think, slow reason, response arbitration, interruption, persona/emotion context, speech-action planning, memory orchestration, proactive activity, protocol/export/monitoring support, and audit gates.

**Architecture:** Keep the existing `eibrain.cognition.realtime.turn` API compatible while splitting new logic into focused modules under `eibrain/cognition/realtime/`. All cross-module outputs are plain JSON-ready dataclasses or dictionaries so they can flow through eiprotocol and Web monitoring. Hardware validation remains a separate honjia gate.

**Tech Stack:** Python 3.13, dataclasses, pytest, eiprotocol builders/models, existing eihead monitor/export scripts.

---

## File Map

- `eibrain/cognition/realtime/events.py`: typed observations, event kinds, JSON helpers.
- `eibrain/cognition/realtime/blackboard.py`: richer TurnBlackboard state and snapshot helpers.
- `eibrain/cognition/realtime/fast.py`: FastThinkEngine and micro-feedback policy.
- `eibrain/cognition/realtime/emotion.py`: prosody, environment, and vision context normalization.
- `eibrain/cognition/realtime/memory.py`: realtime recall/writeback proposal orchestration.
- `eibrain/cognition/realtime/persona.py`: PersonaRuntime state and constraints.
- `eibrain/cognition/realtime/planner.py`: SpeechActionPlanner.
- `eibrain/cognition/realtime/arbiter.py`: ResponseArbiter.
- `eibrain/cognition/realtime/interruption.py`: InterruptionController integration helpers.
- `eibrain/cognition/realtime/slow.py`: SlowReasoner and stable decision assembly.
- `eibrain/cognition/realtime/activity.py`: ProactiveActivityManager.
- `eibrain/cognition/realtime/scheduler.py`: high-level RealtimeCognitiveScheduler facade.
- `eibrain/cognition/realtime/turn.py`: compatibility exports and delegation where needed.
- `eibrain/cognition/realtime/__init__.py`: public exports.
- `eiprotocol/models.py`, `eiprotocol/builders.py`, `eiprotocol/catalog.py`: protocol additions for speech-action plan, proactive activity, emotion context, memory prefetch, and cancellation outcome if missing.
- `eibrain/protocol/eiprotocol_bridge.py`: bridge functions for scheduler outputs.
- `apps/body_runtime/app.py`, `eihead/monitoring/voice.py`, `eihead/monitoring/web.py`: monitoring integration for scheduler lanes and acceptance metrics.
- `scripts/export-eihead-repo.py`: completion manifest updates.
- Tests under `tests/cognition/`, `tests/protocol/`, `tests/eihead/`, and `tests/infra/`.

## Parallel Work Packages

### Task 1: Blackboard, Events, Persona, Emotion

**Files:**
- Create: `eibrain/cognition/realtime/events.py`
- Create: `eibrain/cognition/realtime/blackboard.py`
- Create: `eibrain/cognition/realtime/persona.py`
- Create: `eibrain/cognition/realtime/emotion.py`
- Modify: `eibrain/cognition/realtime/__init__.py`
- Test: `tests/cognition/test_realtime_context.py`

- [ ] Write tests for JSON-ready observation normalization, persona constraints, emotion/environment context, and blackboard append/snapshot behavior.
- [ ] Implement dataclasses and helper functions.
- [ ] Run `python -m pytest -q tests/cognition/test_realtime_context.py`.

### Task 2: Fast Think and Memory Orchestration

**Files:**
- Create: `eibrain/cognition/realtime/fast.py`
- Create: `eibrain/cognition/realtime/memory.py`
- Modify: `eibrain/cognition/realtime/__init__.py`
- Test: `tests/cognition/test_realtime_fast_memory.py`

- [ ] Write tests proving partial ASR can produce non-stable hypotheses, safe micro-feedback, intent hints, and memory recall proposals.
- [ ] Implement FastThinkEngine and MemoryOrchestrator.
- [ ] Ensure FastThinkEngine never emits `stable=True` or final factual commitments.
- [ ] Run `python -m pytest -q tests/cognition/test_realtime_fast_memory.py tests/cognition/test_realtime_scheduler.py`.

### Task 3: Arbiter, Interruption, Speech-Action Planner

**Files:**
- Create: `eibrain/cognition/realtime/arbiter.py`
- Create: `eibrain/cognition/realtime/interruption.py`
- Create: `eibrain/cognition/realtime/planner.py`
- Modify: `eibrain/cognition/realtime/turn.py`
- Modify: `eibrain/cognition/realtime/__init__.py`
- Test: `tests/cognition/test_realtime_arbiter_interruption_planner.py`

- [ ] Write tests for stale round rejection, unstable hypothesis rejection, cancellation summaries, speech/action offsets, and fallback speech when an action is unavailable.
- [ ] Implement focused modules without duplicating existing `ResponseArbiter` behavior.
- [ ] Keep compatibility with existing `RealtimeTurnManager`.
- [ ] Run `python -m pytest -q tests/cognition/test_realtime_arbiter_interruption_planner.py tests/cognition/test_realtime_scheduler.py`.

### Task 4: Slow Reasoner and Scheduler Facade

**Files:**
- Create: `eibrain/cognition/realtime/slow.py`
- Create: `eibrain/cognition/realtime/activity.py`
- Create: `eibrain/cognition/realtime/scheduler.py`
- Modify: `eibrain/cognition/realtime/__init__.py`
- Test: `tests/cognition/test_realtime_cognitive_scheduler.py`

- [ ] Write tests for observe -> hypothesize -> prefetch -> decide -> speak/act -> interrupt flow.
- [ ] Implement SlowReasoner, ProactiveActivityManager, and RealtimeCognitiveScheduler facade.
- [ ] Ensure cancelled old rounds cannot commit new stable decisions.
- [ ] Run `python -m pytest -q tests/cognition/test_realtime_cognitive_scheduler.py tests/cognition/test_realtime_scheduler.py`.

### Task 5: Protocol, Bridge, Monitoring, Export

**Files:**
- Modify: `eiprotocol/catalog.py`
- Modify: `eiprotocol/models.py`
- Modify: `eiprotocol/builders.py`
- Modify: `eiprotocol/__init__.py`
- Modify: `eibrain/protocol/eiprotocol_bridge.py`
- Modify: `apps/body_runtime/app.py`
- Modify: `eihead/monitoring/voice.py`
- Modify: `eihead/monitoring/web.py`
- Modify: `scripts/export-eihead-repo.py`
- Test: `tests/protocol/test_eiprotocol_realtime_cognition.py`
- Test: `tests/body/test_body_runtime_voice_realtime.py`
- Test: `tests/eihead/test_eihead_monitoring_web.py`
- Test: `tests/infra/test_export_eihead_repo.py`

- [ ] Add/verify events for emotion context, memory prefetch, speech-action plan, proactive activity, and cancellation outcome.
- [ ] Add bridge conversion for scheduler snapshots and plans.
- [ ] Surface scheduler lane metrics in voice realtime monitoring without fake hardware claims.
- [ ] Update export manifest completion percentage and remaining blockers.
- [ ] Run focused protocol/body/monitor/export tests.

### Task 6: Ten-Round Audit and Completion Gate

**Files:**
- Modify as needed based on audit findings.
- Create: `docs/realtime-cognitive-scheduler-audit.md`
- Test: full suite.

- [ ] Run audit round 1: API/import/export consistency.
- [ ] Run audit round 2: stale round and cancellation leakage.
- [ ] Run audit round 3: fast lane safety and non-commitment.
- [ ] Run audit round 4: slow lane cancellation awareness.
- [ ] Run audit round 5: speech/action plan timing and fallback.
- [ ] Run audit round 6: persona/emotion/memory determinism.
- [ ] Run audit round 7: eiprotocol strict validation and standalone export.
- [ ] Run audit round 8: Web monitoring truthfulness.
- [ ] Run audit round 9: docs/export completion claims.
- [ ] Run audit round 10: full tests, compileall, diff check, secret scan.

## Completion Target

The code-level target is at least 90% coverage of the JoyInside-inspired design:

- Realtime round tracking and cancellation: complete.
- Fast think: complete at deterministic policy level.
- Slow reasoner: complete at cancellable planning level.
- Arbiter/interruption: complete at scheduler level.
- Speech-action planning: complete at structured plan level.
- Persona/emotion/memory/proactive activity: complete at deterministic MVP level.
- Hardware evidence: explicitly blocked until honjia现场 validation.
