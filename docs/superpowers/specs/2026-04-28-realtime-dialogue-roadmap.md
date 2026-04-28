# Realtime Dialogue Roadmap

## Why

The current honjia voice loop is stable, but it is still a batch pipeline:

`record window -> final ASR -> LLM -> TTS -> playback`

XiaoZhi-style systems feel smoother because they overlap the stages:

`stream audio -> partial ASR -> streaming LLM -> streaming/sentence TTS -> barge-in`

## Phase 1: Safe Foundations

- Add `RealtimeVoiceSession` as the shared state model for partial ASR, first reply token, first speech, total latency, and user barge-in.
- Add `VadEndpointPolicy` as a pure endpointing policy that can consume RMS today and Silero speech probability later.
- Add `AffectiveResponse` as the response contract for `text`, `emotion`, `speaking_style`, `gaze_intent`, and `memory_writeback`.

This phase should not replace the working honjia runtime.

## Phase 2: Quasi-Streaming

- Keep sherpa as the stable ASR provider.
- Lower endpoint latency by decoding on `max_capture_force_decode` for short wake phrases.
- Emit `partial_asr` when the recognizer can provide intermediate text; until then, expose the policy events in Web diagnostics.
- Split LLM output into sentence-level chunks and queue TTS as soon as the first sentence is ready.

## Phase 3: Full Realtime

- Use SileroVAD or equivalent probability-based VAD instead of RMS as the primary endpoint signal.
- Support playback interruption when a new speech segment arrives during TTS.
- Map `AffectiveResponse` to MiniMax voice parameters, Web state, and neck gaze behavior.

## Acceptance

- Wake phrase reliably activates from normal speaking distance.
- First visible ASR/partial state appears under 700 ms after voice start.
- First TTS audio starts before the whole LLM answer is complete.
- User speech during playback stops TTS and starts a new listening turn.
- Web shows stage latencies without double-counting capture, VAD, ASR, LLM, and TTS.
