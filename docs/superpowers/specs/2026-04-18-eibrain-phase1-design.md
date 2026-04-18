# eibrain Phase 1 Design

## Goal

Build a new standalone repository, `eibrain`, as a kernel-first embodied intelligence system with two runtimes:

- `body-runtime` for `honjia`
- `cognitive-runtime` for `honxin`

Phase 1 focuses on a stable voice interaction loop with basic orient-to-speaker behavior.

## Core Principles

- Kernel-first architecture
- Single repository, dual runtimes
- State-driven embodied loop rather than linear event chaining
- Modular body organs with organ-level and subfunction-level degradation
- LLM introduced only inside cognition
- OpenClaw memory integration via adapter boundary
- Body failures degrade capabilities instead of crashing the whole system

## Runtime Boundaries

### body-runtime

Responsible for:

- audio capture / VAD / ASR
- camera / detection
- speech playback
- head motion
- organ state updates
- degradation evaluation
- local reflex behavior such as stop-speaking-on-user-interrupt

### cognitive-runtime

Responsible for:

- observation binding
- unified state updates
- engagement evaluation
- memory retrieval
- prompt construction
- LLM reasoning
- intent planning
- skill dispatch

## Layered Architecture

### kernel

Provides:

- bus
- envelope routing
- lifecycle
- scheduler
- guards

### protocol

Defines:

- observations
- intents
- actions
- outcomes
- state sync envelopes

### state

Defines the unified system state:

- `EmbodiedState`
- `BodyState`
- `WorldState`
- `SelfState`
- `SessionState`
- `EngagementState`
- `CapabilityMatrix`
- `DegradationMode`

### body

Contains organ implementations:

- `ear`
- `eye`
- `mouth`
- `neck`

Each organ exposes health and subfunction health.

### cognition

Contains:

- observation binder
- attention manager
- dialogue manager
- prompt builder
- llm router
- intent planner
- policy engine

### skills

Phase 1 skill set:

- `ListenSkill`
- `ReplySkill`
- `InterruptForUserSkill`
- `OrientToSpeakerSkill`
- `HoldAttentionSkill`

### memory

Contains:

- working memory
- episodic memory
- semantic memory
- OpenClaw adapter

## Body Organ Degradation Model

Phase 1 organs and subfunctions:

- ear: `capture`, `vad`, `asr`
- eye: `camera`, `detection`, `identity`
- mouth: `tts_plan`, `tts_playback`
- neck: `motor`, `tracking`

Each subfunction has one of:

- `healthy`
- `degraded`
- `unavailable`

The degradation manager derives:

- `CapabilityMatrix`
- `DegradationMode`

All cognitive decisions must depend on those derived values rather than assuming a complete body.

## LLM Placement

LLM is introduced in `cognition.dialogue.llm_router`.

It is only used after:

- state update
- memory retrieval
- prompt building

It supports:

- reply generation
- high-level interaction reasoning

It does not directly control hardware or write to the kernel state store.

## Memory Boundary

OpenClaw integration is reserved via a memory adapter.

The data flow is:

`EmbodiedState -> MemoryQuery -> OpenClaw Adapter -> MemoryResult -> Prompt Builder / Intent Planner`

OpenClaw does not directly participate in real-time state transitions.

## Phase 1 Main Loop

`Observation -> State Update -> Engagement Evaluation -> Intent Planning -> Skill Execution -> Action -> Body Execution -> Outcome -> State Update`

For the voice loop:

`speech start -> transcript -> listening -> thinking -> speak/orient intents -> play speech + move head -> interrupt handling -> back to listening`

## Out of Scope for Phase 1

- management web console
- mobile control plane
- multi-tenant orchestration
- generic IoT plugin platform
- rich RAG pipeline
- advanced learning loops
- full visual autonomy

