# Voice Streaming Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the voice chain from code-level batch dialogue closure to a JoyInside-like streaming voice chain that is ready for honjia live testing.

**Architecture:** Keep device I/O in eihead/eivoice, realtime cognition in eibrain, and event contracts in eiprotocol. The first deliverable is not product-grade AEC; it is a testable streaming boundary with ASR partial/final, LLM/TTS deltas, playback state, barge-in events, trace metrics, and fake-device tests.

**Tech Stack:** Python 3, pytest, eiprotocol builders/catalog/router, `apps.head_runtime`, `apps.body_runtime`, existing `RealtimeVoiceSession`, and fake capture/playback test doubles.

---

## File Structure

- `eiprotocol/catalog.py`: add missing voice-streaming event definitions and routes.
- `eiprotocol/builders.py`: add builder helpers for audio frame, TTS sentence/chunk, playback state, voice heartbeat, and barge-in events.
- `eiprotocol/event_routing.py`: route the new event names.
- `tests/protocol/test_eiprotocol_voice_streaming.py`: validate catalog, codec, builders, and routing.
- `apps/head_runtime/eivoice_gateway.py`: implement a fake-device-friendly eihead voice gateway.
- `tests/eihead/test_eivoice_gateway.py`: cover capture, state machine, playback queue, barge-in, heartbeat, reconnect state.
- `apps/body_runtime/voice_streaming_adapter.py`: map voice-streaming eiprotocol events into `RealtimeVoiceSession` operations and trace records.
- `tests/body/test_voice_streaming_adapter.py`: cover ASR partial/final, agent deltas, TTS playback events, interruption, stale round rejection.
- `apps/body_runtime/voice_chain_scenarios.py`: scenario runner for short Chinese utterances, follow-up turns, playback barge-in, and jitter.
- `tests/body/test_voice_chain_scenarios.py`: verify benchmark JSON shape and thresholds.
- `docs/realtime-cognitive-scheduler-audit.md`: document live-test boundary and honjia checklist updates.

## Task A: eiprotocol Voice Streaming Events

**Files:**
- Modify: `eiprotocol/catalog.py`
- Modify: `eiprotocol/builders.py`
- Modify: `eiprotocol/event_routing.py`
- Create: `tests/protocol/test_eiprotocol_voice_streaming.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert these event names have catalog definitions, strict validation, codec round-trip, and route names:

```python
VOICE_STREAMING_EVENTS = {
    "ei.voice.audio.frame": ("observation", "head_to_brain", "voice_audio_frame"),
    "ei.voice.asr.partial": ("dialogue", "head_to_brain", "voice_asr_partial"),
    "ei.voice.asr.final": ("dialogue", "head_to_brain", "voice_asr_final"),
    "ei.voice.tts.sentence_start": ("dialogue", "brain_to_head", "voice_tts_sentence_start"),
    "ei.voice.tts.chunk": ("dialogue", "brain_to_head", "voice_tts_chunk"),
    "ei.voice.playback.started": ("dialogue", "head_to_brain", "voice_playback_started"),
    "ei.voice.playback.stopped": ("dialogue", "head_to_brain", "voice_playback_stopped"),
    "ei.voice.barge_in.detected": ("dialogue", "head_to_brain", "voice_barge_in_detected"),
    "ei.voice.session.heartbeat": ("control", "bidirectional", "voice_session_heartbeat"),
}
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/protocol/test_eiprotocol_voice_streaming.py`

Expected: FAIL because the new names/routes/builders do not exist.

- [ ] **Step 3: Implement minimal protocol support**

Add catalog definitions and route names. Add builder helpers:

```python
build_voice_audio_frame_event(...)
build_voice_asr_event(..., final: bool)
build_voice_tts_sentence_start_event(...)
build_voice_tts_chunk_event(...)
build_voice_playback_state_event(..., started: bool)
build_voice_barge_in_detected_event(...)
build_voice_session_heartbeat_event(...)
```

Each builder must return a valid `EventEnvelope` and preserve `session_id`, `round_id`, `trace_id`, `streamId`, `chunkIndex`, `audioBase64`, `text`, `reason`, and `latencyMs` where applicable.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/protocol/test_eiprotocol_voice_streaming.py tests/protocol/test_eiprotocol_realtime_dialogue_head_events.py`

Expected: PASS.

## Task B: eihead/eivoice Gateway

**Files:**
- Create: `apps/head_runtime/eivoice_gateway.py`
- Create: `tests/eihead/test_eivoice_gateway.py`

- [ ] **Step 1: Write failing tests**

Test a fake capture/playback pipeline:

```python
gateway = EiVoiceGateway(session_id="s1", actor_id="u1", capture=fake_capture, playback=fake_playback)
events = gateway.capture_audio_frame()
assert events[0].name == "ei.voice.audio.frame"
gateway.enqueue_tts_chunk(...)
gateway.start_playback(...)
barge = gateway.probe_barge_in()
assert barge.name == "ei.voice.barge_in.detected"
```

Also test `heartbeat()` returns `ei.voice.session.heartbeat` with queue lengths, state, and health.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/eihead/test_eivoice_gateway.py`

Expected: FAIL because `apps.head_runtime.eivoice_gateway` does not exist.

- [ ] **Step 3: Implement gateway**

Implement pure-Python fake-device-friendly classes:

```python
class EiVoiceGateway:
    capture_audio_frame(self) -> list[EventEnvelope]
    accept_asr_partial(self, text: str) -> EventEnvelope
    accept_asr_final(self, text: str) -> EventEnvelope
    enqueue_tts_chunk(self, audio_base64: str, *, sentence_id: str = "") -> EventEnvelope
    start_playback(self) -> EventEnvelope
    stop_playback(self, reason: str = "completed") -> EventEnvelope
    probe_barge_in(self) -> EventEnvelope | None
    heartbeat(self) -> EventEnvelope
```

Keep this module independent from real honjia audio libraries. It should be ready to wrap U4K/arecord later, but tests must use fakes.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/eihead/test_eivoice_gateway.py`

Expected: PASS.

## Task C: eibrain Voice Streaming Adapter

**Files:**
- Create: `apps/body_runtime/voice_streaming_adapter.py`
- Create: `tests/body/test_voice_streaming_adapter.py`

- [ ] **Step 1: Write failing tests**

Test event application to `RealtimeVoiceSession`:

```python
adapter = VoiceStreamingAdapter(session)
adapter.apply(asr_partial_event)
assert session.transcript_partial == "你好"
adapter.apply(asr_final_event)
assert session.transcript_final == "你好鸿途"
adapter.apply(agent_delta_event)
assert session.reply_text == "你好"
adapter.apply(playback_started_event)
assert session.first_speech_at_s is not None
adapter.apply(barge_in_event)
assert session.interrupted is True
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/body/test_voice_streaming_adapter.py`

Expected: FAIL because the adapter module does not exist.

- [ ] **Step 3: Implement adapter**

Implement:

```python
class VoiceStreamingAdapter:
    def apply(self, event: Mapping[str, object] | EventEnvelope) -> dict[str, object]:
        ...
```

Map `ei.voice.asr.partial/final` and existing `ei.dialogue.asr.partial/final` to partial/final session operations. Map `ei.dialogue.agent.delta` to reply delta. Map `ei.voice.tts.sentence_start` and `ei.voice.playback.started` to speaking start. Map `ei.voice.barge_in.detected` and `ei.dialogue.interrupt.requested` to interrupt.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/body/test_voice_streaming_adapter.py tests/body/test_realtime_voice.py`

Expected: PASS.

## Task D: Voice Trace Dashboard and Scenario Runner

**Files:**
- Create: `apps/body_runtime/voice_chain_scenarios.py`
- Create: `tests/body/test_voice_chain_scenarios.py`
- Modify: `apps/body_runtime/voice_chain_benchmark.py`
- Modify: `docs/realtime-cognitive-scheduler-audit.md`

- [ ] **Step 1: Write failing tests**

Test that the scenario runner emits JSON with `scenario`, `turns`, `summary`, `thresholds`, and `honjiaReady`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/body/test_voice_chain_scenarios.py`

Expected: FAIL because the scenario runner does not exist.

- [ ] **Step 3: Implement scenario runner**

Implement deterministic scenario helpers for short utterance, fuzzy child utterance, playback barge-in, follow-up turn, and jitter. Do not call network providers or hardware.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/body/test_voice_chain_scenarios.py tests/body/test_voice_chain_benchmark.py`

Expected: PASS.

## Task E: Integration Verification

**Files:**
- No new production files unless review finds a gap.

- [ ] **Step 1: Run focused voice/protocol tests**

Run:

```text
python -m pytest -q tests/protocol/test_eiprotocol_voice_streaming.py tests/eihead/test_eivoice_gateway.py tests/body/test_voice_streaming_adapter.py tests/body/test_voice_chain_scenarios.py tests/body/test_realtime_voice.py tests/body/test_voice_dialogue_loop.py tests/body/test_voice_chain_benchmark.py
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run: `python -m pytest -q`

Expected: PASS.

- [ ] **Step 3: Audit and document true-device boundary**

Confirm the code reaches honjia true-device-test readiness, not product-grade AEC. Document required live checks: U4K capture, speaker echo, AEC/NS mode, MiniMax audible stop latency, ASR final latency, first audio latency, barge-in false trigger rate, and stale round leak count.

## Self-Review

- Spec coverage: protocol, eihead gateway, eibrain adapter, benchmark runner, and honjia test readiness are all mapped to tasks.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: all new event names use `ei.voice.*`; existing `ei.dialogue.*` aliases remain supported for compatibility.
- Scope check: this plan intentionally stops before real AEC implementation because AEC quality needs honjia live audio validation.
