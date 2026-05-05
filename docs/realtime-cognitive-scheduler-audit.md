# Realtime Cognitive Scheduler Audit

Date: 2026-05-05

Scope: JoyInside-inspired realtime cognition refactor for eibrain, eiprotocol, eihead monitoring, and the body runtime voice loop.

## Completion Assessment

Code-level completion is assessed at 92%.

Completed:

- Realtime cognitive scheduler with round id, cancellation token, lanes, blackboard state, and scheduler snapshots.
- Fast think, memory prefetch, slow reasoner, response arbiter, interruption controller, persona runtime, emotion context, speech/action planner, and proactive activity manager modules.
- Voice loop integration for ASR partial/final, microfeedback telemetry, interrupt cancellation chain, arbiter-gated speech/action dispatch, degraded reply status, and first-speech latency.
- Voice chain second wave code-level closer: truthful ear telemetry field wiring, a voice-chain benchmark summary utility, and a playback-time barge-in probe/seam for interrupting during speech playback.
- eiprotocol MVP events for realtime cognition, including emotion context, memory prefetch, speech/action plan, proactive activity, and cancellation-applied events.
- eihead Web monitoring projection for voice realtime state, scheduler lanes, interruption, cancellation chain, speech/action plans, and latency gates.
- Regression coverage across cognition, body runtime, eihead monitor, eiprotocol fixtures/builders/bridge, and standalone eiprotocol export.

Remaining non-code or hardware-gated work:

- Honjia on-device soak validation with real microphone, speaker, camera, Hailo, and gimbal hardware.
- U4K microphone echo validation, AEC/NS tuning, MiniMax TTS stop actual latency measurement, playback barge-in false-trigger-rate measurement, and continuous-dialogue first-packet latency measurement on honjia.
- Runtime benchmark tuning against the target metrics under live network and MiniMax/Qwen provider latency.
- Real video inference and continuous visual tracking are still planned as the next dedicated hardware phase.

## Voice Chain Second Wave Boundary

This wave makes the honjia voice chain code-level closer to a JoyInside-like realtime experience, but it is not an on-device AEC/free-conversation validation signoff. The completed scope is:

- Truthful ear telemetry: monitoring can now report capture, ASR decode, and ASR elapsed timing from the ear processor instead of optimistic or mismatched fields.
- Voice-chain benchmark: offline trace summaries can expose ASR final, first token, first audio, interrupt stop, round leak, p95, threshold, and bottleneck signals.
- Playback-time barge-in probe/seam: the loop has a hook for detecting user speech while TTS playback is active and requesting interrupt without dispatching stale replies.
- Monitor semantics: benchmark `turnCount` counts retained terminal traces after `(roundId, status)` de-duplication, while `roundLeakCount` specifically flags stale-round leakage. Unconfirmed TTS stop attempts remain visible as interruption events but do not create fake `interruptStopMs` latency.

Hardware validation required before claiming live JoyInside-equivalent behavior:

- U4K microphone echo under honjia speaker playback.
- AEC/NS effectiveness in the real room and device placement.
- MiniMax TTS stop actual latency from interrupt request to audible stop.
- False trigger rate for playback-time barge-in under household noise, TTS self-leak, and normal operator movement.
- Continuous-dialogue first-packet latency across repeated turns, not only isolated offline traces.
- Capture-backend timing on honjia, especially any backend that lacks `read_voice_window` and can only expose chunk-based probing.

## Ten Audit Rounds

1. Baseline and isolation audit: used a dedicated worktree branch for the realtime scheduler refactor, preserving the existing main working tree and avoiding direct overwrite of user changes.
2. Cognitive core audit: added and reviewed turn lifecycle, blackboard, cancellation token, scheduler snapshots, and lane separation.
3. Fast lane audit: checked that FastThink only writes hypotheses and microfeedback, not stable decisions or factual commitments.
4. Slow lane audit: checked final-ASR precedence, memory context intake, action-oriented decisions, and stable decision shape.
5. Arbiter/interruption audit: checked stale round rejection, cancellation semantics, action plan validation, and non-leaking interrupted rounds.
6. Speech-action audit: checked structured speech/action plans, fallback speech for action failure, timing fields, and semantic alignment diagnostics.
7. Voice runtime audit: checked ASR partial/final integration, reply dispatch status, interrupt stop telemetry, first speech latency, and Web monitor truthfulness.
8. eiprotocol audit: checked typed fixtures, strict validation, bridge event directions, standalone package export, and scheduler snapshot conversion.
9. Subagent finding audit: fixed P1/P2 findings around final-ASR override, action segment preservation, microfeedback timing, TTS stop confirmation, degraded reply status, typed fixture losslessness, and monitor lifecycle state.
10. Verification audit: ran focused regression and full repository tests after final fixes.

## Verification Evidence

Fresh verification commands run on 2026-05-05:

```text
python -m pytest -q tests/cognition/test_realtime_cognitive_scheduler.py tests/cognition/test_realtime_arbiter_interruption_planner.py tests/body/test_voice_dialogue_loop.py tests/body/test_body_runtime_voice_realtime.py tests/protocol/test_eiprotocol_realtime_cognition.py tests/protocol/test_eiprotocol_realtime_cognition_bridge.py tests/protocol/test_eiprotocol_fixtures.py
90 passed in 2.64s
```

```text
python -m pytest -q
795 passed, 2 skipped in 52.98s
```

## Residual Risks

- Live latency numbers can only be finalized after honjia现场复测; code now exposes the gates, but real ASR/LLM/TTS provider timing must be measured on device.
- The interrupt-to-stop metric now distinguishes dispatch latency from confirmed TTS stop; hardware backends that cannot confirm stop will report unconfirmed rather than fake success.
- The realtime visual lane is protocol/runtime-ready but still needs the next camera/Hailo video-stream implementation pass.
