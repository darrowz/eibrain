# eibrain <-> eimemory RPC Integration Design

## Summary

`eibrain` currently retrieves memory through a local `OpenClawMemoryAdapter` that only supports
`in_memory` state and a thin custom HTTP JSON surface. `eimemory` already exposes a dedicated
`eibrain` integration boundary through its SDK, RPC bridge, and RPC server.

This design connects `eibrain` to `eimemory` through a stable RPC/HTTP boundary instead of a
shared repository path or direct Python import from `/dev-project/eimemory`. The integration must
preserve the current `eibrain` memory contract while making deployment path independent from the
code checkout path.

## Goals

- Let `eibrain` retrieve memory context from `eimemory` over RPC/HTTP.
- Keep `eibrain` deployable without depending on `/dev-project/eimemory` as an import path.
- Preserve `eibrain`'s internal `MemoryQuery` and `MemoryResult` contract so cognition code stays
  simple.
- Allow graceful degradation when `eimemory` is unavailable.
- Make runtime configuration express deployment endpoints, not source-code checkout paths.
- Cover the new behavior with adapter, config, and runtime integration tests.

## Non-Goals

- Embedding the `eimemory` runtime directly inside the `eibrain` process.
- Making `eibrain` manage `eimemory` process lifecycle in this change.
- Reworking `eimemory`'s internal storage or recall algorithms.
- Fully replacing every legacy `openclaw`-named field in one pass.

## Current State

### eibrain

- `apps/cognitive_runtime/app.py` instantiates `OpenClawMemoryAdapter` directly.
- `eibrain/memory/adapters/openclaw.py` supports:
  - `in_memory`
  - `http_json` with ad-hoc endpoints such as `/retrieve_context`
- `MemoryResult` expects a compact summary plus optional relevant memories, actor profile, and
  session summary.
- Config currently models memory under `memory.openclaw`.

### eimemory

- `eimemory/adapters/eibrain/sdk.py` provides a direct client for recall/evolution against a local
  `Runtime` object.
- `eimemory/adapters/eibrain/rpc.py` exposes a typed RPC method surface:
  - `memory.recall`
  - `evolution.observe`
  - `evolution.get_active_policy`
- `eimemory/adapters/eibrain/rpc_server.py` exposes that RPC surface over HTTP POST.

## Recommended Approach

Adopt an RPC-first integration in `eibrain`.

`eibrain` will keep its own adapter boundary and select a provider from configuration. A new
provider, `eimemory_rpc`, will call the `eimemory` RPC server over HTTP POST using the method
surface already implemented in `eimemory`.

This preserves a clean system boundary:

- `eibrain` owns cognition behavior and memory-consumer contracts.
- `eimemory` owns storage, recall, and evolution behavior.
- Deployment can place each service in any directory because the connection happens through a
  configured endpoint.

## Architecture

### Provider model

`eibrain` memory configuration will support three modes:

- `in_memory`: existing local fallback behavior for development and degraded operation
- `http_json`: existing legacy custom HTTP adapter behavior kept for backward compatibility
- `eimemory_rpc`: new preferred integration path

The current `OpenClawMemoryAdapter` class will be refactored into a provider gateway or replaced by
an equivalent memory adapter factory. The important behavior is provider selection without forcing
cognition code to know which provider is active.

### New adapter boundary in eibrain

Add a dedicated RPC-aware adapter under `eibrain/memory/adapters/` responsible for:

- building RPC payloads from `MemoryQuery`
- calling the configured endpoint
- mapping the returned recall bundle into `MemoryResult`
- degrading safely on transport or payload failure

`apps/cognitive_runtime/app.py` should depend on the adapter interface, not instantiate the legacy
OpenClaw adapter unconditionally.

### Runtime boundary in eimemory

No protocol redesign is needed in `eimemory` for this change. `eibrain` should integrate with the
existing RPC methods exposed by `eimemory/adapters/eibrain/rpc.py` and
`eimemory/adapters/eibrain/rpc_server.py`.

If a small protocol addition is needed during implementation, it must be added to `eimemory` as an
extension of the existing RPC surface rather than by teaching `eibrain` to import `eimemory`
modules directly.

## Configuration Design

### Principle

Configuration must describe runtime deployment locations, not code checkout locations.

The following is explicitly forbidden in production logic:

- importing from `/dev-project/eimemory`
- storing `/dev-project/eimemory` inside `eibrain` business config as the integration mechanism
- assuming the deployed `eimemory` service lives beside the `eibrain` git checkout

### Proposed config shape

Keep the existing `memory.openclaw` block for compatibility, but extend it with the fields required
for RPC operation.

Expected fields:

- `provider`
- `endpoint`
- `api_key`
- `timeout_s`
- `agent_id` optional override
- `workspace_id` optional override

The runtime endpoint may be provided either in YAML or through environment expansion.
Typical deployment configuration should look like:

- development checkout:
  - endpoint points at a locally started `eimemory` RPC server
- deployed service:
  - endpoint points at the actual running service address, for example
    `http://127.0.0.1:8091/`

The endpoint is the contract. The deployment directory is not.

## Data Mapping

### Recall request mapping

`eibrain` `MemoryQuery` will map into `eimemory` `memory.recall` params.

Mapping rules:

- `query` -> `query`
- `session_id` -> included in `scope` so session-local memory can be retrieved
- `actor_id` -> included in `scope` so actor-scoped memory can be retrieved
- runtime cognition purpose -> `task_context`

The adapter should derive stable scope values using configuration defaults when query fields are
missing. This lets deployment operators define a stable agent/workspace identity once.

### Recall response mapping

`eimemory` recall returns a recall bundle, not the exact `MemoryResult` shape used in `eibrain`.
The adapter will map the bundle as follows:

- `summary`: synthesized from the most relevant recall items and active rules into a compact string
- `relevant_memories`: item summaries or titles from recall bundle items
- `actor_profile`: extracted from structured recall metadata when available, otherwise `{}`
- `session_summary`: extracted from bundle explanation or session-shaped items when available,
  otherwise `""`

The mapping should be deterministic and conservative. If the bundle contains richer information than
`MemoryResult` can represent, the adapter should preserve the highest-value subset instead of trying
to mirror the entire bundle.

### Degraded behavior

When RPC transport fails, returns invalid JSON, or returns an invalid payload shape, `eibrain` must
not crash the cognition runtime. It should fall back to the existing local degraded behavior:

- summary starts from the query text
- locally cached profile/session data may still be used if available
- failures remain observable through logs or diagnostics

## Deployment Model

### Code repositories

Current code repository paths:

- `/dev-project/eibrain`
- `/dev-project/eimemory`

These are development checkouts only.

### Real deployment paths

Real deployments may live elsewhere, for example:

- `/opt/eibrain/current`
- `/opt/eimemory/current`

or in separate services/containers.

The integration must continue to work as long as:

- `eimemory` is running and reachable at the configured endpoint
- `eibrain` is configured with that endpoint

No business logic should care whether the running code came from `/dev-project`, `/opt`, or a
container filesystem.

## Error Handling

The adapter must treat these conditions as recoverable:

- connection refused
- timeout
- non-200 response
- malformed JSON
- RPC payload with `ok: false`
- missing expected keys in the response

Recovery behavior:

- return a degraded `MemoryResult`
- do not raise into the main cognition loop for normal transport errors
- log enough detail to debug the failure boundary

## Testing Strategy

Implementation will follow TDD.

### eibrain tests

Add or update tests for:

- config loading of `eimemory_rpc` fields
- adapter request mapping from `MemoryQuery` to RPC payload
- adapter response mapping from recall bundle to `MemoryResult`
- graceful degradation on timeout/invalid payload/RPC error
- cognitive runtime selecting the configured provider correctly

### Cross-repo integration tests

Use `eimemory`'s existing RPC server in integration-style tests to prove that:

- `eibrain` can call a live `EIBrainRPCServer`
- a persisted memory in `eimemory` is reflected in `eibrain` memory retrieval
- scope mapping behaves consistently between the two repos

If a shared integration test cannot live in one repo cleanly, each repo should at least have one
boundary test that exercises the shared RPC contract from its own side.

## Implementation Phases

1. Extend `eibrain` config and adapter selection for `eimemory_rpc`.
2. Add failing adapter tests in `eibrain`.
3. Implement RPC client and response mapping in `eibrain`.
4. Update cognitive runtime wiring to use provider selection instead of a hard-coded legacy adapter.
5. Add live-boundary tests using `eimemory` RPC server.
6. Adjust deployment documentation/config examples to point to endpoints instead of repository paths.

## Risks and Mitigations

### Risk: mapping loses useful bundle detail

Mitigation:
Keep the first integration conservative, and only map the subset needed by current `eibrain`
consumers. Expand `MemoryResult` later if cognition genuinely needs more structure.

### Risk: old config naming causes confusion

Mitigation:
Support legacy `memory.openclaw` shape for compatibility, but document `eimemory_rpc` as the
preferred provider and move examples toward endpoint-oriented naming.

### Risk: developers confuse checkout path with runtime path again

Mitigation:
Document the distinction explicitly in config comments and deployment docs, and avoid any feature
that requires the `eimemory` repository path to be present on the `eibrain` host.

## Open Questions Resolved

- Direct SDK import from the `eimemory` repo is rejected because it couples code checkout paths and
  runtime behavior.
- RPC/HTTP is the preferred production boundary.
- Development and deployment use the same contract: configured endpoint, not repository path.
