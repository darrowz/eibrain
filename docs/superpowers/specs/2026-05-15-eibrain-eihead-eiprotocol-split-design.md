# eibrain / eihead / eiprotocol Physical Split Design

Date: 2026-05-15

## Goal

Make `eiprotocol` and `eihead` standalone physical repositories while keeping `eibrain` focused on cognition, embodied orchestration, memory policy, learning, and runtime coordination.

The split should remove duplicate ownership without breaking the current honxin/honjia workflow. `eiprotocol` becomes the canonical shared contract package first. `eihead` becomes the canonical honjia head runtime package second. `eibrain` then stops packaging embedded implementations of either package.

## Current State

The parent workspace already has standalone submodules:

- `D:\github\ei-workspace\repos\eibrain`
- `D:\github\ei-workspace\repos\eiprotocol`
- `D:\github\ei-workspace\repos\eihead`

The `eibrain` repository still physically contains and packages:

- `eiprotocol/`
- `eihead/`
- `apps/head_runtime/`
- `tests/protocol/`
- `tests/eihead/`

`pyproject.toml` currently exposes `eihead-runtime` and includes `eihead*` and `eiprotocol*` in the `eibrain` package find list. This is useful for compatibility, but it keeps the source-of-truth boundary ambiguous.

## Target Ownership

- `eiprotocol` owns event contracts, typed models, builders, catalog definitions, codec, validation, routing, golden fixtures, and conformance reporting.
- `eihead` owns honjia-local head runtime behavior: eye, ear, mouth, neck, native providers, device adapters, head HTTP API, event journal, capability registry, and head-local monitoring payloads.
- `eibrain` owns cognitive runtime, body orchestration boundary, memory adapter policy, learning loops, skills, embodied state, kernel scheduling, and cross-runtime operator views.
- `eimemory` remains the durable memory service. `eibrain` depends on its endpoint contract, not on the repository path.

## Non-Goals

- Do not redesign the runtime protocol while splitting repositories.
- Do not move eimemory into this split.
- Do not claim honjia hardware readiness from repository structure alone.
- Do not break the current `D:\github\ei-workspace` superproject workflow.
- Do not keep two authoritative copies of protocol or head runtime code after the compatibility window closes.

## Target Topology

```text
D:\github\ei-workspace
  repos/
    eiprotocol/        canonical shared protocol package
    eihead/            canonical honjia head runtime package
    eibrain/           cognition, orchestration, memory, learning
    eimemory/          durable memory service
    eidocs/
    eiskills/
    eitraining/
```

Runtime dependency direction should be:

```text
eihead     -> eiprotocol
eibrain    -> eiprotocol
eibrain    -> eihead only through runtime clients, status APIs, or optional dev/test helpers
eibrain    -> eimemory via endpoint
eihead     -> honjia hardware and local providers
```

`eiprotocol` must not import `eibrain`, `eihead`, hardware runtimes, LLM providers, memory backends, or deployment scripts.

## Import Boundary Rules

1. `eibrain` code may import `eiprotocol` from the standalone package.
2. `eihead` code may import `eiprotocol` from the standalone package.
3. `eiprotocol` code must remain dependency-clean and transport-agnostic.
4. `eibrain` should not import `eihead` internals for normal runtime behavior. Runtime communication should use protocol events, HTTP/status clients, snapshots, or explicit compatibility adapters.
5. `apps/head_runtime` should move to the `eihead` repository or become a temporary shim that delegates to `eihead.runtime.cli`.
6. `eibrain/protocol/eiprotocol_bridge.py` may remain in `eibrain` because it adapts eibrain-specific state into shared protocol events.
7. `tests/protocol` should move with `eiprotocol` unless a test specifically verifies eibrain's bridge behavior.
8. `tests/eihead` should move with `eihead` unless a test specifically verifies eibrain integration with head status or events.

## Migration Phases

### Phase 1: Canonicalize eiprotocol

Make `D:\github\ei-workspace\repos\eiprotocol` the only source of protocol implementation.

Expected changes:

- Move or verify protocol source files under the standalone `eiprotocol` repo.
- Move protocol fixtures and protocol-only tests to the standalone repo.
- Add or update `eiprotocol` packaging metadata, conformance command, and test command.
- Change `eibrain` development setup to install `repos/eiprotocol` in editable mode.
- Remove `eiprotocol*` from `eibrain` packaging once all imports resolve from the standalone package.
- Keep only eibrain-specific bridge tests in `eibrain/tests/protocol`.

Acceptance:

- `eiprotocol` tests pass in isolation without importing `eibrain` or `eihead`.
- `eibrain` tests that use protocol events pass while importing `eiprotocol` from `repos/eiprotocol`.
- A negative dependency check proves standalone `eiprotocol` has no imports from `eibrain`, `eihead`, `apps`, provider SDKs, or hardware modules.

### Phase 2: Canonicalize eihead

Make `D:\github\ei-workspace\repos\eihead` the only source of honjia head runtime implementation.

Expected changes:

- Move or verify `eihead/` package source under the standalone `eihead` repo.
- Move `apps/head_runtime/` into the `eihead` repo or replace it with a short deprecation shim during the compatibility window.
- Move `eihead-runtime` and `eihead-verify-hardware` script definitions to `eihead` packaging metadata.
- Move head runtime, monitoring, device, eye, ear, mouth, neck, and capability tests to the standalone repo.
- Keep eibrain integration tests only where they exercise body/cognitive interaction with head status or protocol events.
- Remove `eihead*` from `eibrain` packaging after consumers use the standalone package.

Acceptance:

- `eihead` tests pass in isolation with only `eiprotocol` as its EI-series package dependency.
- `eihead-runtime` can be installed and invoked from the standalone repo.
- `eibrain` no longer needs direct imports from `eihead` internals for its normal runtime path.
- Head-local hardware probes remain truthful: missing honjia devices report `not_wired` or `degraded`.

### Phase 3: Slim eibrain

Remove embedded implementation copies and keep `eibrain` as the brain/orchestration package.

Expected changes:

- Remove embedded `eiprotocol/` and `eihead/` implementation directories from `eibrain`.
- Update `pyproject.toml` package include list to stop including `eiprotocol*` and `eihead*`.
- Update README setup instructions to install EI workspace dependencies in order: `eiprotocol`, `eihead`, then `eibrain`.
- Keep `apps.body_runtime`, `apps.cognitive_runtime`, `apps.operator_console`, deployment helpers, memory adapters, cognition, learning, skills, body orchestration, state, and kernel code in `eibrain`.
- Re-home export scripts once standalone repos are authoritative. Export scripts should become validation helpers or be removed after the split is complete.

Acceptance:

- `eibrain` imports resolve with standalone `eiprotocol` and `eihead` installed.
- `python -m compileall -q apps eibrain tests` passes from the `eibrain` repo.
- The eibrain test suite passes without local embedded `eiprotocol/` or `eihead/` packages.
- The unified workspace graph still indexes all repositories through `D:\github\ei-workspace`.

## Operator Console Boundary

`apps.operator_console` can remain in `eibrain` if it is an aggregate view across body runtime, cognitive runtime, memory traces, and head status. Head-local monitoring primitives should live in `eihead.monitoring`.

The boundary is:

- `eihead.monitoring`: build truthful honjia-local payloads.
- `eibrain` operator console: compose body, cognition, memory, and head payloads into one operator-facing dashboard.

If the monitor is deployed only on honjia and does not need cognitive aggregation, move that monitor entrypoint to `eihead`. If it remains a cross-runtime dashboard, keep the entrypoint in `eibrain` and consume `eihead` through public status payloads.

## Compatibility Window

During migration, compatibility shims may exist, but each shim must:

- be small and explicit;
- delegate to the standalone package;
- emit no duplicate implementation logic;
- have a planned deletion phase;
- be covered by a test that proves it imports the standalone target.

The compatibility window ends when `eibrain` no longer packages `eiprotocol*` or `eihead*`.

## Development Workflow

In the parent workspace, use editable installs:

```bash
python -m pip install -e D:/github/ei-workspace/repos/eiprotocol
python -m pip install -e D:/github/ei-workspace/repos/eihead
python -m pip install -e D:/github/ei-workspace/repos/eibrain
```

Use `D:\github\ei-workspace` for cross-repository graph reasoning and isolated child repositories for focused package work.

## Validation Strategy

Run validation at three levels:

1. Package-local tests:
   - `eiprotocol` protocol and conformance tests.
   - `eihead` runtime, hardware-boundary, monitoring, and capability tests.
   - `eibrain` cognition, body orchestration, memory, learning, and bridge tests.
2. Dependency boundary checks:
   - `eiprotocol` has no imports from `eibrain`, `eihead`, or `apps`.
   - `eibrain` does not package embedded `eiprotocol*` or `eihead*`.
   - `eihead` depends on `eiprotocol`, not `eibrain`, for shared contracts.
3. Workspace integration:
   - editable installs from the parent workspace;
   - eibrain body/cognitive smoke tests;
   - eihead runtime smoke tests;
   - unified code-review graph rebuild from `D:\github\ei-workspace`.

## Risks

- Import shadowing can hide whether code is using embedded or standalone packages.
- Tests may pass because `eibrain` still contains fallback copies.
- Moving `apps/head_runtime` too early can break script entrypoints.
- Operator console ownership is mixed today and should be resolved deliberately.
- Hardware-facing tests must stay fakeable on non-honjia hosts.

## Recommendation

Proceed with a staged split:

1. Freeze and canonicalize `eiprotocol`.
2. Canonicalize `eihead`.
3. Slim `eibrain`.

This order reduces blast radius because protocol is the stable contract, head runtime is the device boundary, and eibrain can be cleaned up after both dependencies are real standalone packages.
