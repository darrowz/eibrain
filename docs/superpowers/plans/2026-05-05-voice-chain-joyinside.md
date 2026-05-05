# Voice Chain JoyInside Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the honjia voice chain from a working batch dialogue loop toward a JoyInside-like realtime experience with truthful monitoring, latency benchmarking, and natural barge-in behavior.

**Architecture:** Keep the current `VoiceDialogueLoop`, `RealtimeVoiceSession`, MiniMax TTS, and eiprotocol event model. Add missing telemetry correctness, a lightweight benchmark harness, and a playback-time VAD interrupt seam before attempting full WebSocket ASR/TTS replacement.

**Tech Stack:** Python 3, pytest, eibrain body runtime, eihead monitoring contracts, sherpa-onnx/faster-whisper adapters, MiniMax TTS command driver.

---

## File Structure

- `apps/body_runtime/app.py`: fix ear processor telemetry field wiring and expose truthful capture/decode timings.
- `apps/body_runtime/voice_dialogue_loop.py`: add optional playback-time VAD polling that can request interrupt without dispatching stale replies.
- `apps/body_runtime/voice_chain_benchmark.py`: create an offline benchmark summary utility for ASR/LLM/TTS/interrupt latency traces.
- `tests/body/test_body_runtime_voice_realtime.py`: regression tests for truthful ear latency.
- `tests/body/test_voice_dialogue_loop.py`: regression tests for playback-time barge-in.
- `tests/body/test_voice_chain_benchmark.py`: tests for benchmark aggregation and threshold gates.
- `docs/superpowers/plans/2026-05-05-voice-chain-joyinside.md`: this implementation plan.

## Task A: Truthful Ear Telemetry

**Files:**
- Modify: `apps/body_runtime/app.py`
- Test: `tests/body/test_body_runtime_voice_realtime.py`

- [ ] **Step 1: Write failing test**

Add a test that builds a `BodyRuntimeApp` with a fake `ear_processor` exposing `last_capture_elapsed_ms`, `last_decode_elapsed_ms`, and `last_transcribe_elapsed_ms`. Call `transcribe_audio_window()` and assert the recent event details contain `capture_elapsed_ms`, `asr_decode_elapsed_ms`, and `asr_elapsed_ms`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/body/test_body_runtime_voice_realtime.py::test_ear_processor_event_reports_processor_latency_fields`

Expected: FAIL because `app.py` currently reads `last_capture_ms` and `last_asr_decode_ms`, which are not fields on `EarStreamProcessor`.

- [ ] **Step 3: Implement minimal fix**

Update `_record_ear_processor_event()` to read:

```python
"capture_elapsed_ms": getattr(self.ear_processor, "last_capture_elapsed_ms", None),
"asr_decode_elapsed_ms": getattr(self.ear_processor, "last_decode_elapsed_ms", None),
"asr_elapsed_ms": getattr(self.ear_processor, "last_transcribe_elapsed_ms", elapsed_ms),
```

Keep existing capture device, VAD, RMS, and transcript fields unchanged.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/body/test_body_runtime_voice_realtime.py tests/eihead/test_eihead_ear_realtime.py`

Expected: PASS.

## Task B: Voice Chain Benchmark

**Files:**
- Create: `apps/body_runtime/voice_chain_benchmark.py`
- Create: `tests/body/test_voice_chain_benchmark.py`

- [ ] **Step 1: Write failing tests**

Add tests for a pure function that accepts a list of turn dictionaries with fields such as `wakeToListenMs`, `asrFinalMs`, `firstTokenMs`, `firstAudioMs`, `interruptStopMs`, and `roundLeak`, then returns counts, averages, p95 values, threshold pass flags, and bottleneck labels.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/body/test_voice_chain_benchmark.py`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement minimal benchmark module**

Create:

```python
DEFAULT_THRESHOLDS = {
    "asrFinalMs": 800.0,
    "firstTokenMs": 700.0,
    "firstAudioMs": 2000.0,
    "interruptStopMs": 300.0,
}

def summarize_voice_chain(turns, *, thresholds=None):
    ...
```

The function must be standard-library-only, deterministic, and JSON-serializable. It should ignore missing numeric fields, count `roundLeak=True`, and compute p95 by nearest-rank.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/body/test_voice_chain_benchmark.py`

Expected: PASS.

## Task C: Playback-Time Barge-In Seam

**Files:**
- Modify: `apps/body_runtime/voice_dialogue_loop.py`
- Test: `tests/body/test_voice_dialogue_loop.py`

- [ ] **Step 1: Write failing test**

Add a fake body runtime where `is_speaking()` returns `True`, `probe_barge_in()` returns `{"detected": True, "reason": "playback_vad"}`, and `dispatch_actions()` records `StopSpeechAction`. Start the loop and assert it requests interrupt instead of only publishing `playback_active`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/body/test_voice_dialogue_loop.py::test_voice_dialogue_loop_detects_barge_in_while_speaking`

Expected: FAIL because `_run()` currently sleeps whenever `is_speaking()` is true.

- [ ] **Step 3: Implement minimal seam**

Add a small helper:

```python
def _maybe_interrupt_during_playback(self) -> bool:
    probe = getattr(self.body_runtime, "probe_barge_in", None)
    if not callable(probe):
        return False
    result = probe(session_id=self.session_id, actor_id=self.actor_id)
    if isinstance(result, dict) and result.get("detected"):
        self.request_interrupt(reason=str(result.get("reason") or "playback_barge_in"))
        return True
    return False
```

Call it inside the `is_speaking()` branch before sleeping. If it returns true, continue the loop without publishing a stale `playback_active` update.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/body/test_voice_dialogue_loop.py`

Expected: PASS.

## Task D: Integration Audit

**Files:**
- Modify only if Task A-C reveal integration gaps.

- [ ] **Step 1: Run combined focused suite**

Run:

```text
python -m pytest -q tests/body/test_realtime_voice.py tests/body/test_ear_stream.py tests/body/test_vad_policy.py tests/body/test_voice_dialogue_loop.py tests/body/test_body_runtime_voice_realtime.py tests/body/test_voice_chain_benchmark.py tests/eihead/test_eihead_ear_realtime.py tests/eihead/test_eihead_mouth_playback.py tests/integration/test_voice_loop.py tests/protocol/test_eiprotocol_realtime_dialogue_head_events.py
```

Expected: PASS.

- [ ] **Step 2: Run full repository suite**

Run: `python -m pytest -q`

Expected: PASS with the same skip profile as baseline.

- [ ] **Step 3: Commit**

Commit with message: `Advance JoyInside-like voice chain`

## Self-Review

- Spec coverage: truthful Web/ear telemetry is Task A; benchmark trace utility is Task B; playback-time natural interrupt seam is Task C; verification and submission are Task D.
- Placeholder scan: no TODO/TBD placeholders are present.
- Scope check: this plan deliberately avoids replacing ASR/TTS providers or requiring honjia live hardware, because the user is not on site.

## Second Wave Task D/E/F Status

- [x] **Task D: Code-level voice-chain completion record**

  Completed scope for this wave is limited to truthful ear telemetry, the voice-chain benchmark utility, and the playback-time barge-in probe/seam. This is a code-level closer, not on-site proof that honjia now matches a JoyInside free-dialogue experience.

- [x] **Task E: Verification boundary record**

  Focused code-level verification belongs to the implementation tasks. Hardware validation remains separate because local/unit tests cannot measure real U4K microphone echo, AEC/NS behavior, audible TTS stop latency, household false triggers, or continuous-dialogue first-packet latency.

- [x] **Task F: Configuration/documentation/risk boundary audit**

  Documentation now states that honjia hardware validation is required before making live-readiness claims. This task intentionally does not change runtime code.

- [ ] **Hardware validation follow-up**

  Validate on honjia: U4K microphone echo during speaker playback; AEC/NS suppression quality; MiniMax TTS stop actual audible latency; playback-time barge-in false-trigger rate; continuous-dialogue first-packet latency across repeated turns.
