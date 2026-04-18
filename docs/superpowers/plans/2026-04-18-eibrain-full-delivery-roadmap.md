# eibrain Full Delivery Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver `eibrain` from its current kernel-first foundation to a deployable, real-device, embodied intelligence system running across `honjia` and `honxin`.

**Architecture:** The system keeps the existing kernel-first split: `honjia` runs `body-runtime`, `honxin` runs `cognitive-runtime`, and both share a unified protocol/state model. The delivery path prioritizes stable deployment, real hardware adaptation, vision understanding, long-term memory, and finally adaptive behavior.

**Tech Stack:** Python 3.14, pytest, YAML configuration, local driver adapters, Anthropic-compatible MiniMax access, MiniMax MCP integration, OpenClaw adapter boundary, Linux deployment scripts.

---

## Current Baseline

- `Phase 1` kernel-first repository foundation is in place.
- Unified YAML configuration and deploy bootstrap exist.
- Text LLM path supports Anthropic-compatible MiniMax access.
- Body organ architecture, capability degradation, and dual runtime assembly already exist as scaffolding.
- `vision_llm` is wired as an experimental entry point, but not yet a production-trustworthy vision chain.

---

## Delivery Strategy

The remaining work should be delivered in six stages. Each stage ends in a deployable checkpoint, not just code completion.

### Stage 0: Deployment Hardening

**Purpose:** Make the current repository reproducibly deployable to both `honjia` and `honxin`.

**Why first:** The system already has enough moving parts that continuing feature work without a stable deployment layer will slow down every later stage.

**Inputs:**
- Existing unified config
- Current runtime entrypoints
- Current default bootstrap logic

**Outputs:**
- `.env.example`
- Linux deployment script set for `honjia` and `honxin`
- runtime startup docs
- health-check commands and expected outputs

**Scope:**
- normalize all runtime-required environment variables
- add deployment-time validation for required config sections
- add bootstrap outputs for model directories, logs, cache, and runtime temp data
- add service-friendly launch commands
- add smoke-check CLI path for text generation and body action routing

**Done when:**
- a fresh Linux host can be prepared from the repo and one YAML file
- both runtimes can be started with no manual path edits
- failures in config or missing env vars are surfaced clearly before runtime start

**Primary risks:**
- hidden assumptions from local Windows development
- incomplete env var defaults
- missing runtime directories for models and generated artifacts

---

### Stage 1: Real Body Driver Integration

**Purpose:** Replace the remaining stub/noop organ behavior with real `honjia` adapters.

**Why now:** `eibrain` cannot be judged as embodied until the body side actually hears, sees, speaks, and moves.

**Inputs:**
- Stage 0 deployable baseline
- current body organ interfaces
- migrated metadata from the legacy multimodality project

**Outputs:**
- real `ear` adapter for camera microphone capture
- real `mouth` adapter for speaker playback
- real `neck` adapter for gimbal execution
- real `eye` adapter for camera capture and detection

**Scope:**
- `ear.capture`: real audio source selection
- `ear.vad`: stable streaming VAD path
- `ear.asr`: local streaming `sherpa-onnx` integration
- `mouth.tts_playback`: real TTS output path
- `neck.motor`: real pan control execution
- `eye.camera`: frame capture
- `eye.detection`: detection path using existing edge-side assumptions

**Done when:**
- a person can speak near `honjia` and produce transcript observations
- `honxin` can produce a reply and `honjia` can audibly play it
- `honxin` can emit orient actions and `honjia` can move the gimbal
- an organ failure causes degradation, not total process failure

**Primary risks:**
- ALSA/audio device mismatch
- `sherpa-onnx` model bundle mismatch
- gimbal command timing and bounds safety
- camera/Hailo deployment dependencies on target Linux host

---

### Stage 2: Stable Vision Understanding Path

**Purpose:** Turn vision from “experimental hook” into a reliable production path.

**Why now:** The project’s embodied loop depends on more than ASR. We need a stable answer to “how does `honxin` actually understand what `honjia` sees?”

**Inputs:**
- Stage 1 real `eye` capture
- current `vision_llm` config entry
- MiniMax docs for MCP `understand_image` and `web_search`

**Outputs:**
- `MiniMax MCP Adapter`
- image handoff path from `honjia` to `honxin`
- structured vision result contract
- fallback policy between local detection and remote understanding

**Scope:**
- add `vision.mcp` config section
- support `uvx minimax-coding-plan-mcp`
- implement `understand_image(prompt, image_path_or_url)` adapter
- choose an image transfer policy:
  - default: `honjia` exports frame artifact, `honxin` accesses via URL or synced artifact path
- define result schema for caption, detected subject, confidence, and raw text
- keep `vision_llm` as experimental direct-model route, but do not make it the primary path

**Done when:**
- a frame captured by `honjia` can be understood by `honxin`
- the result is machine-readable and can feed cognition/state
- failure in MCP service degrades to local-only visual behavior

**Primary risks:**
- MCP process management on target Linux hosts
- image accessibility across machines
- external service rate limits or quota behavior

---

### Stage 3: Social Interaction Loop Completion

**Purpose:** Finish the first truly embodied interaction loop: listen, understand, reply, orient, interrupt, resume.

**Why now:** Once body and vision are real, the user-facing quality comes from turn-taking and embodied interaction timing.

**Inputs:**
- Stage 1 real organs
- Stage 2 stable vision path
- current engagement, intent, and skill layers

**Outputs:**
- full interaction state machine
- interrupt-safe speech behavior
- orient-to-speaker as a real runtime behavior
- cooldown and re-entry handling

**Scope:**
- tighten `idle -> noticing -> listening -> thinking -> speaking -> interrupted -> cooldown`
- add body-side reflex for immediate speech stop on user interruption
- use visual or audio speaker cues to orient neck during active interaction
- suppress conflicting actions while speaking or moving
- make session continuity explicit across transcript, reply, and body outcome

**Done when:**
- a user can interrupt the agent mid-reply and the system stops speaking quickly
- the system resumes listening without restarting the entire interaction
- the gimbal orients toward the current interaction target rather than moving blindly
- the loop behaves like one session, not disconnected events

**Primary risks:**
- race conditions between speak/listen/orient actions
- noisy interruption triggers
- cross-runtime state drift

---

### Stage 4: OpenClaw Memory Integration

**Purpose:** Turn the current memory boundary into a real long-term memory subsystem.

**Why now:** Before adaptive behavior makes sense, the system needs reliable persistence of interaction context.

**Inputs:**
- current memory contracts and in-memory adapter
- OpenClaw endpoint/auth shape
- stabilized social interaction loop

**Outputs:**
- production `OpenClawMemoryAdapter`
- retrieval path before reply planning
- write-back path after important interactions
- actor/session memory primitives

**Scope:**
- implement real OpenClaw read/write methods
- keep `working`, `episodic`, and `semantic` responsibilities separate
- define which events generate memory writes
- define memory retrieval budget for prompt building
- add failure isolation so OpenClaw outages do not break live interaction

**Done when:**
- the same user can interact twice and receive context-aware continuity
- the system can persist actor/session summaries across process restarts
- OpenClaw failure does not prevent live replies

**Primary risks:**
- slow memory calls hurting real-time response
- prompt bloat from unbounded retrieval
- memory pollution from low-value writes

---

### Stage 5: Learning and Evaluation Closure

**Purpose:** Upgrade the current learning scaffold into a usable improvement loop.

**Why now:** Learning should happen only after the system can reliably perceive, act, and remember.

**Inputs:**
- current learning/review scaffolding
- stabilized body/cognition/memory runtime behavior
- persistent logs and outcomes

**Outputs:**
- structured interaction review records
- evaluation metrics for speaking, interruption handling, and vision usefulness
- adaptation hooks for prompts, policies, and thresholds

**Scope:**
- add event trace packaging for review
- score interaction quality dimensions
- store actionable improvement summaries
- introduce bounded adaptation targets:
  - prompt templates
  - response style
  - interrupt thresholds
  - orient policy tuning
- keep hard safety boundaries outside learning control

**Done when:**
- the system can explain why a recent interaction failed or degraded
- review results can influence future runtime settings in a bounded way
- no learned setting can directly bypass body safety or operator controls

**Primary risks:**
- feedback loops that amplify bad behavior
- overfitting to small test scenarios
- hidden coupling between learned settings and runtime safety

---

### Stage 6: Production Readiness and Operator Tooling

**Purpose:** Make `eibrain` operable as a persistent real system instead of a development-only stack.

**Why last:** Production hardening matters most after the full embodied loop exists.

**Inputs:**
- all prior stages
- runtime logs, failure modes, and operational lessons

**Outputs:**
- operator runbook
- log and trace packaging
- startup and restart strategy
- incident-safe fallback modes

**Scope:**
- add structured logs for both runtimes
- add trace IDs across observation -> intent -> action -> outcome
- define “safe degraded” startup modes
- package service launch with systemd-ready or equivalent commands
- document operator workflows for:
  - restart
  - config validation
  - device health checking
  - MiniMax/MCP outage handling
  - OpenClaw outage handling

**Done when:**
- the system can be restarted and diagnosed by an operator without reading source code
- the operator can tell whether failure is in body, cognition, vision, or memory
- the system has a defined degraded mode for every major external dependency

**Primary risks:**
- hidden operational dependency chains
- incomplete observability across two machines
- mismatch between development logs and production triage needs

---

## Recommended Execution Order

1. Stage 0: Deployment Hardening
2. Stage 1: Real Body Driver Integration
3. Stage 2: Stable Vision Understanding Path
4. Stage 3: Social Interaction Loop Completion
5. Stage 4: OpenClaw Memory Integration
6. Stage 5: Learning and Evaluation Closure
7. Stage 6: Production Readiness and Operator Tooling

This order keeps the system moving from “can start” to “can perceive and act” to “can interact well” to “can remember and improve” to “can operate reliably.”

---

## Milestone Checkpoints

### Milestone A: Deployable Foundation

Reached after Stage 0.

**Meaning:** A clean host can start both runtimes from one repo and one YAML config.

### Milestone B: Real Embodied Loop

Reached after Stages 1-3.

**Meaning:** `honjia` and `honxin` can complete a real-world voice interaction with orienting and interruption behavior.

### Milestone C: Persistent Cognitive Agent

Reached after Stage 4.

**Meaning:** The system retains memory across interactions and process lifetimes.

### Milestone D: Adaptive Embodied System

Reached after Stage 5.

**Meaning:** The system can review its own behavior and apply bounded improvements.

### Milestone E: Operable Product Baseline

Reached after Stage 6.

**Meaning:** The system is deployable, diagnosable, and maintainable as an ongoing service.

---

## What Still Requires Human Confirmation

Only a small number of decisions still need explicit confirmation:

- whether image transfer between `honjia` and `honxin` should prefer shared filesystem, HTTP artifact serving, or synced artifact directory
- the exact OpenClaw production endpoint and auth shape
- the real Linux service management target (`systemd`, container, or manual process supervisor)

Everything else can continue under the current defaults.

---

## Immediate Next Step

The next concrete phase should be:

**Stage 2: Stable Vision Understanding Path**

Because:
- text access is already working
- body deployment scaffolding exists
- `coding-plan-vlm` direct routing is not yet trustworthy as the primary production path
- MiniMax officially recommends MCP tools for `understand_image` and `web_search`

That makes `MiniMax MCP Adapter` the best next implementation target.
