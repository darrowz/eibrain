# eibrain / OpenClaw / eimemory Boundary

## Goal

Keep honjia's embodied brain coherent while allowing OpenClaw-style channels and eimemory long-term memory to cooperate without becoming duplicate brains.

## Ownership

1. `eibrain` owns honjia's real-time embodied turn: wake/sleep state, voice loop, vision target, neck control, mouth playback, and local diagnostics.
2. `OpenClaw` should be treated as a channel/gateway and orchestration peer for Feishu, MQTT, remote chat, and async tools.
3. `eimemory` is the canonical long-term memory backend. Neither eibrain nor OpenClaw should keep a separate authoritative durable memory.

## Conflict Rules

- No conflict exists when eibrain owns `voice.honjia`, OpenClaw owns async channels such as `feishu.bot`, and both write to eimemory.
- Conflict exists if eibrain and OpenClaw both try to own the same live honjia turn.
- Conflict exists if both systems write long-term memories without source labels.
- Conflict exists if OpenClaw memory and eimemory are both treated as authoritative.

## Minimum Memory Contract

Every recall context and writeback should preserve:

- `source_system`: `eibrain` or `openclaw`
- `channel_id`: `voice.honjia`, `feishu.bot`, `voice.xiaozhi`, etc.
- `agent_id`: `eibrain.voice`, `openclaw.feishu`, etc.
- `actor_id`: human or external actor
- `session_id`: local or channel session
- `memory_type`: `interaction.episode`, `user.preference`, `session.summary`, `body.diagnostic`, `visual.observation`
- `memory_contract_version`: currently `multimodal-memory.v1`

## Current Recommendation

Stabilize in this order:

1. Make voice latency and memory diagnostics visible on honjia Web `18080`.
2. Reduce ASR/VAD latency using a resident recognizer and quasi-streaming endpointing.
3. Add OpenClaw only as a channel/gateway, then point its durable memory calls to eimemory with explicit source metadata.
