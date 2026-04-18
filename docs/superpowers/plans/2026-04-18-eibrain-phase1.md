# eibrain Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 `eibrain` repository with a kernel-first embodied architecture, dual runtimes, modular body organs, state core, cognition flow, skills, and OpenClaw-ready memory boundaries.

**Architecture:** The repository centers on a shared embodied kernel and unified state store. `body-runtime` produces observations and executes actions while `cognitive-runtime` consumes observations, updates state, retrieves memory, plans intents, and dispatches skills.

**Tech Stack:** Python 3.14, pytest, dataclasses, typing protocols, package-based architecture.

---

### Task 1: Scaffold Repository Structure

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `eibrain/__init__.py`
- Create: `apps/body_runtime/__init__.py`
- Create: `apps/cognitive_runtime/__init__.py`
- Create: `tests/test_repo_imports.py`

- [ ] Write import and package structure tests first
- [ ] Create minimal package skeleton to satisfy imports
- [ ] Run import tests

### Task 2: Define Protocol Contracts

**Files:**
- Create: `eibrain/protocol/__init__.py`
- Create: `eibrain/protocol/base.py`
- Create: `eibrain/protocol/observations.py`
- Create: `eibrain/protocol/intents.py`
- Create: `eibrain/protocol/actions.py`
- Create: `eibrain/protocol/outcomes.py`
- Create: `eibrain/protocol/envelopes.py`
- Create: `tests/protocol/test_protocol_models.py`

- [ ] Write failing tests for core protocol objects and serialization helpers
- [ ] Implement minimal dataclass-based protocol models
- [ ] Run protocol tests

### Task 3: Build Unified State Models

**Files:**
- Create: `eibrain/state/__init__.py`
- Create: `eibrain/state/body.py`
- Create: `eibrain/state/world.py`
- Create: `eibrain/state/self_state.py`
- Create: `eibrain/state/session.py`
- Create: `eibrain/state/engagement.py`
- Create: `eibrain/state/embodied.py`
- Create: `tests/state/test_embodied_state.py`

- [ ] Write failing tests for state defaults and state transitions
- [ ] Implement minimal state dataclasses and helper methods
- [ ] Run state tests

### Task 4: Implement Kernel Core

**Files:**
- Create: `eibrain/kernel/__init__.py`
- Create: `eibrain/kernel/bus.py`
- Create: `eibrain/kernel/router.py`
- Create: `eibrain/kernel/lifecycle.py`
- Create: `eibrain/kernel/scheduler.py`
- Create: `eibrain/kernel/guards.py`
- Create: `tests/kernel/test_bus_and_router.py`

- [ ] Write failing tests for publish/subscribe, routing, and guard hooks
- [ ] Implement minimal synchronous kernel core
- [ ] Run kernel tests

### Task 5: Implement Body Organ and Health Core

**Files:**
- Create: `eibrain/body/__init__.py`
- Create: `eibrain/body/health/__init__.py`
- Create: `eibrain/body/health/organ_health.py`
- Create: `eibrain/body/health/capability_matrix.py`
- Create: `eibrain/body/health/degradation_manager.py`
- Create: `eibrain/body/health/fallback_policy.py`
- Create: `eibrain/body/state/__init__.py`
- Create: `eibrain/body/state/body_state_manager.py`
- Create: `tests/body/test_degradation_manager.py`

- [ ] Write failing tests for organ health aggregation and capability derivation
- [ ] Implement body health core
- [ ] Run body health tests

### Task 6: Implement Organ Interfaces and Phase 1 Stubs

**Files:**
- Create: `eibrain/body/organs/__init__.py`
- Create: `eibrain/body/organs/base.py`
- Create: `eibrain/body/organs/ear/__init__.py`
- Create: `eibrain/body/organs/ear/organ.py`
- Create: `eibrain/body/organs/eye/__init__.py`
- Create: `eibrain/body/organs/eye/organ.py`
- Create: `eibrain/body/organs/mouth/__init__.py`
- Create: `eibrain/body/organs/mouth/organ.py`
- Create: `eibrain/body/organs/neck/__init__.py`
- Create: `eibrain/body/organs/neck/organ.py`
- Create: `tests/body/test_organs.py`

- [ ] Write failing tests for organ heartbeat, health, and supported actions
- [ ] Implement stub organs with protocol-aware outputs
- [ ] Run organ tests

### Task 7: Implement Cognition Core

**Files:**
- Create: `eibrain/cognition/__init__.py`
- Create: `eibrain/cognition/fusion/__init__.py`
- Create: `eibrain/cognition/fusion/binder.py`
- Create: `eibrain/cognition/attention/__init__.py`
- Create: `eibrain/cognition/attention/manager.py`
- Create: `eibrain/cognition/dialogue/__init__.py`
- Create: `eibrain/cognition/dialogue/dialogue_manager.py`
- Create: `eibrain/cognition/dialogue/prompt_builder.py`
- Create: `eibrain/cognition/dialogue/llm_router.py`
- Create: `eibrain/cognition/policy/__init__.py`
- Create: `eibrain/cognition/policy/engine.py`
- Create: `eibrain/cognition/planner/__init__.py`
- Create: `eibrain/cognition/planner/intent_planner.py`
- Create: `tests/cognition/test_intent_planner.py`

- [ ] Write failing tests for observation binding, engagement evaluation, and intent planning
- [ ] Implement minimal cognition flow with stub LLM policy
- [ ] Run cognition tests

### Task 8: Implement Memory Contracts and OpenClaw Adapter Boundary

**Files:**
- Create: `eibrain/memory/__init__.py`
- Create: `eibrain/memory/contracts.py`
- Create: `eibrain/memory/working/__init__.py`
- Create: `eibrain/memory/working/store.py`
- Create: `eibrain/memory/episodic/__init__.py`
- Create: `eibrain/memory/episodic/store.py`
- Create: `eibrain/memory/semantic/__init__.py`
- Create: `eibrain/memory/semantic/store.py`
- Create: `eibrain/memory/adapters/__init__.py`
- Create: `eibrain/memory/adapters/base.py`
- Create: `eibrain/memory/adapters/openclaw.py`
- Create: `tests/memory/test_openclaw_adapter.py`

- [ ] Write failing tests for memory query/result contracts and adapter behavior
- [ ] Implement in-memory stores and OpenClaw stub adapter
- [ ] Run memory tests

### Task 9: Implement Skill Layer

**Files:**
- Create: `eibrain/skills/__init__.py`
- Create: `eibrain/skills/base.py`
- Create: `eibrain/skills/compiler.py`
- Create: `eibrain/skills/listen.py`
- Create: `eibrain/skills/reply.py`
- Create: `eibrain/skills/interrupt.py`
- Create: `eibrain/skills/orient.py`
- Create: `eibrain/skills/hold_attention.py`
- Create: `tests/skills/test_skill_compiler.py`

- [ ] Write failing tests for intent-to-action compilation
- [ ] Implement Phase 1 skills
- [ ] Run skill tests

### Task 10: Implement Runtime Assembly and Example Loop

**Files:**
- Create: `apps/body_runtime/app.py`
- Create: `apps/cognitive_runtime/app.py`
- Create: `eibrain/infra/__init__.py`
- Create: `eibrain/infra/config.py`
- Create: `eibrain/infra/logging.py`
- Create: `tests/integration/test_voice_loop.py`

- [ ] Write failing integration test for the minimal voice interaction loop
- [ ] Implement both runtime assembly apps and the end-to-end in-process demo loop
- [ ] Run the integration test and the full test suite

