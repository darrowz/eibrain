# eibrain

`eibrain` is a kernel-first embodied intelligence system with:

- `body-runtime` for `honjia`
- `cognitive-runtime` for `honxin`

Phase 1 targets a stable voice interaction loop with orient-to-speaker behavior,
modular body organs, and OpenClaw-ready memory boundaries.

## Deployable Configuration

All runtime configuration lives in a single YAML file:

- `config/eibrain.yaml`

Two CLI entrypoints are available after installation:

- `eibrain-body --config config/eibrain.yaml`
- `eibrain-cognitive --config config/eibrain.yaml --text "hello"`
- `eibrain-bootstrap --config config/eibrain.yaml`
- `eibrain-check-deployment --config config/eibrain.yaml`
- `eibrain-monitor --config config/eibrain.yaml`
- `eibrain-sync-honjia --target-host darrow@100.81.78.119 --restart-monitor`

The configuration supports:

- body organ subfunctions with `noop`, `command`, or `http` drivers
- cognition text and vision LLM provider settings
- MiniMax CLI-backed image understanding and web search configuration
- MiniMax MCP configuration preserved for optional fallback or later toolchain upgrades
- OpenClaw memory adapter settings
- honjia monitoring web settings
- environment variable expansion through `${VAR_NAME}`

## Memory Endpoint

`eibrain` should connect to a running `eimemory` service through `EIMEMORY_ENDPOINT`, for example `http://127.0.0.1:8091/`.

Keep these concerns separate:

- code checkout paths such as `/dev-project/eimemory`
- deployed runtime paths such as `/opt/eimemory/current`
- the actual integration contract, which is the RPC endpoint address

`eibrain` must not depend on the `eimemory` repository path at runtime.

## Standalone Package Install Order (Development)

When running `eibrain` with standalone shared packages, install in this order
in editable mode:

```bash
python -m pip install -e D:/github/ei-workspace/repos/eiprotocol
python -m pip install -e D:/github/ei-workspace/repos/eihead
python -m pip install -e D:/github/ei-workspace/repos/eibrain
```

This ensures runtime imports resolve from canonical standalone source trees:
`eiprotocol` first, then `eihead`, then `eibrain`.

## Default Deployment Layout

The default deployment root is `/home/${USER}/eibrain`, shared by both `honjia` and `honxin`.
You can bootstrap the layout with:

- `eibrain-bootstrap --config config/eibrain.yaml`
- `eibrain-check-deployment --config config/eibrain.yaml`
- `scripts/bootstrap-default-deployment.sh`
- `scripts/check-deployment.sh`

The bootstrap step creates:

- the shared runtime root
- body and cognitive runtime directories
- a local `sherpa-onnx` streaming ASR model directory with placeholder files:
  - `tokens.txt`
  - `encoder.onnx`
  - `decoder.onnx`
- `joiner.onnx`

## Deployment Sync

`honxin` is the only primary Git repository. `honjia` is a deployment target.

Use the sync helper on `honxin` after validated changes:

- `eibrain-sync-honjia --target-host darrow@100.81.78.119 --restart-monitor`

The sync helper deploys `config/eibrain.honjia.yaml` to `honjia` as `config/eibrain.yaml`,
so the deployment target keeps its real-device configuration instead of inheriting the
development defaults from the primary repository.

## honjia Monitoring

`eibrain-monitor` serves a lightweight monitoring page and JSON status feed for `honjia`.

- `GET /` returns an HTML dashboard
- `GET /status.json` returns machine-readable status
- `GET /healthz` returns the same JSON for quick probes

The listener is configured in `monitoring.host` / `monitoring.port`.

For the current honxin split-port deployment, keep `18080` for the proxy and run the local monitor on `18081` using `config/eibrain.honjia.local.yaml` plus `deploy/systemd/eibrain-monitor.service`.

## Voice Chain Validation Boundary

The second-wave voice-chain work is a code-level closer: truthful ear telemetry, a voice-chain benchmark, and a playback-time barge-in probe/seam are documented for honjia. Hardware validation is still required before calling the live experience JoyInside-equivalent, including U4K microphone echo, AEC/NS quality, MiniMax TTS stop latency, false-trigger rate, and continuous-dialogue first-packet latency.

## Model Defaults

- text LLM defaults to `MiniMax-M2.7-highspeed`
- vision LLM is configured separately and currently defaults to `coding-plan-vlm`
- `vision.provider` defaults to `minimax_cli` for stable image understanding on `honxin`
- `vision_llm` remains an experimental direct-model path until it is smoke-tested with real provider credentials

## Stage 2-6 Baseline

- Stage 2: primary vision path is `vision.cli` using `mmx vision describe` and `mmx search query`
- Stage 2 fallback: `vision.mcp` remains available using `uvx minimax-coding-plan-mcp`
- Stage 3: cognitive runtime supports orient and interrupt planning in addition to speech replies
- Stage 4: OpenClaw adapter supports in-memory and `http_json` read/write flows
- Stage 5: learning snapshot records review, score-derived recommendation, and policy decision
- Stage 6: operator console can summarize body snapshot, cognition snapshot, and recent traces
