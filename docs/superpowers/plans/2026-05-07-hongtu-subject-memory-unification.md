# Hongtu Subject Memory Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make eibrain, eimemory, eihead/OpenClaw/Feishu treat Hongtu as one continuous subject with channel-specific observations instead of split personalities.

**Architecture:** eimemory remains the durable source of truth for identity and memory. eibrain owns runtime recall/write policy and injects a canonical subject context into every recall and memory write. OpenClaw/Feishu entries are classified as channels: user/agent conversation can become shared memory, while prompt injection traces and bridge audits stay diagnostic.

**Tech Stack:** Python 3.11, pytest, JSON-RPC eimemory adapter, eibrain memory policy, local eimemory compatibility layer.

---

## File Structure

- `eibrain/memory/subject.py`: new focused helper for Hongtu subject IDs, channel aliases, canonical scope, and memory-layer classification.
- `eibrain/memory/adapters/eimemory_rpc.py`: attach subject context to recall task context and writeback metadata without breaking existing RPC contracts.
- `eibrain/cognition/policy/multimodal_memory.py`: block audit/trace sources from normal persona recall, prefer meaningful OpenClaw/Feishu conversation sources, and expose memory-layer hints.
- `apps/cognitive_runtime/app.py`: surface subject/memory-layer diagnostics in existing memory trace data.
- `tests/memory/test_hongtu_subject.py`: tests for canonical subject scope, aliases, and source classification.
- `tests/memory/test_eimemory_rpc_adapter.py`: tests for subject context in recall/write RPC payloads.
- `tests/cognition/test_multimodal_memory_policy.py`: tests for OpenClaw/Feishu policy filtering.
- `../eimemory/identity.py`: compatibility helpers for canonical user aliases and legacy Hongtu scope recall.
- `../eimemory/adapters/eibrain/rpc.py`: use alias-aware recall scopes and preserve source channel metadata.
- `../eimemory/api/memory.py`: allow recall to search canonical Hongtu scope plus configured aliases when task context requests it.

## Task 1: eibrain Subject Identity Helper

**Files:**
- Create: `eibrain/memory/subject.py`
- Test: `tests/memory/test_hongtu_subject.py`

- [ ] **Step 1: Write failing tests**

```python
from eibrain.memory.subject import (
    DEFAULT_HONGTU_USER_ID,
    classify_memory_layer,
    normalize_hongtu_scope,
    subject_context,
)


def test_normalize_hongtu_scope_defaults_to_single_subject():
    scope = normalize_hongtu_scope({"agent_id": "honxin", "workspace_id": "honjia"})
    assert scope == {
        "tenant_id": "default",
        "agent_id": "hongtu",
        "workspace_id": "embodied",
        "user_id": DEFAULT_HONGTU_USER_ID,
    }


def test_subject_context_preserves_channel_aliases():
    ctx = subject_context(
        channel_id="feishu",
        actor_id="ou_644f810515d8ae7789de6a932d4de854",
        session_id="sess-1",
        source="openclaw.message_received",
    )
    assert ctx["subject_id"] == "hongtu"
    assert ctx["channel_id"] == "feishu"
    assert ctx["canonical_user_id"] == DEFAULT_HONGTU_USER_ID
    assert "ou_644f810515d8ae7789de6a932d4de854" in ctx["user_aliases"]


def test_classify_memory_layer_keeps_audits_out_of_persona():
    assert classify_memory_layer("openclaw.before_prompt_build", "conversation") == "trace"
    assert classify_memory_layer("ei_bridge.openclaw_feishu", "audit") == "channel_audit"
    assert classify_memory_layer("openclaw.agent_end", "conversation") == "episodic"
    assert classify_memory_layer("eibrain.identity", "profile") == "identity_core"
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/memory/test_hongtu_subject.py -q`

Expected: import failure because `eibrain.memory.subject` does not exist.

- [ ] **Step 3: Implement helper**

Create deterministic constants and pure functions only. Do not call network, disk, or RPC. Unknown non-empty users should be preserved as aliases while canonical user remains `darrow`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_hongtu_subject.py -q`

Expected: all tests pass.

## Task 2: eibrain Recall/Writeback Subject Context

**Files:**
- Modify: `eibrain/memory/adapters/eimemory_rpc.py`
- Test: `tests/memory/test_eimemory_rpc_adapter.py`

- [ ] **Step 1: Add tests for recall payload**

Assert `memory.recall` sends:

```json
{
  "scope": {
    "tenant_id": "default",
    "agent_id": "hongtu",
    "workspace_id": "embodied",
    "user_id": "darrow"
  },
  "task_context": {
    "subject_context": {
      "subject_id": "hongtu",
      "channel_id": "voice",
      "canonical_user_id": "darrow"
    }
  }
}
```

- [ ] **Step 2: Add tests for writeback metadata**

Assert `memory.ingest` metadata contains `subject_context`, `memory_layer`, `source_channel`, `session_id`, and raw actor alias when provided.

- [ ] **Step 3: Implement adapter changes**

Use `normalize_hongtu_scope()` for Hongtu runtime scope. Merge subject context into caller-provided task context without overwriting explicit non-empty keys.

- [ ] **Step 4: Run adapter tests**

Run: `pytest tests/memory/test_eimemory_rpc_adapter.py -q`

Expected: all tests pass.

## Task 3: eibrain Memory Policy Filters

**Files:**
- Modify: `eibrain/cognition/policy/multimodal_memory.py`
- Test: `tests/cognition/test_multimodal_memory_policy.py`

- [ ] **Step 1: Add tests for OpenClaw/Feishu source policy**

Normal voice/dialogue recall must prefer `openclaw.agent_end` and `openclaw.message_received`, but block `openclaw.before_prompt_build` and `ei_bridge.openclaw_feishu`.

- [ ] **Step 2: Add tests for diagnostic recall**

Diagnostic/task-trace recall may include audit sources only when the task type explicitly asks for diagnostics.

- [ ] **Step 3: Implement policy constants**

Add `audit_trace_sources`, `shared_dialogue_sources`, and `source_memory_layers`. Ensure protected persona keys cannot be overwritten by audit records.

- [ ] **Step 4: Run policy tests**

Run: `pytest tests/cognition/test_multimodal_memory_policy.py -q`

Expected: all tests pass.

## Task 4: eimemory Alias-Aware Recall Compatibility

**Files:**
- Modify: `../eimemory/identity.py`
- Modify: `../eimemory/adapters/eibrain/rpc.py`
- Modify: `../eimemory/api/memory.py`

- [ ] **Step 1: Add helper tests if local eimemory tests exist**

Search for existing eimemory tests. If none exist, add pure unit tests under the nearest existing test tree that verify alias-aware query scopes include canonical Darrow plus the known Feishu alias.

- [ ] **Step 2: Implement alias helpers**

Add a stable default alias map:

```python
DEFAULT_HONGTU_USER_ALIASES = {
    "darrow": ("darrow", "ou_644f810515d8ae7789de6a932d4de854"),
}
```

Add `hongtu_query_scopes_with_aliases(scope, aliases=None)` returning canonical and legacy scopes for each alias without duplicates.

- [ ] **Step 3: Wire recall**

When eibrain RPC receives `task_context.subject_context.user_aliases`, recall should search canonical Hongtu scope and aliases. Audit sources are still controlled by eibrain filters.

- [ ] **Step 4: Run focused tests**

Run the smallest available eimemory unit tests. If the local eimemory snapshot lacks test scaffolding, run `python -m compileall eimemory`.

## Task 5: Runtime Diagnostics

**Files:**
- Modify: `apps/cognitive_runtime/app.py`
- Optional Modify: `apps/operator_console/app.py`
- Optional Modify: `apps/operator_console/web.py`

- [ ] **Step 1: Add tests or targeted assertions**

Existing memory diagnostics should expose `subject_id`, `channel_id`, `canonical_user_id`, `memory_layer`, and blocked audit sources.

- [ ] **Step 2: Implement minimal diagnostics**

Do not redesign the Web UI. Add fields to the existing memory trace payload so the monitor can show why OpenClaw/Feishu memories were used or ignored.

- [ ] **Step 3: Run runtime tests**

Run relevant cognitive runtime/operator console tests plus full memory tests.

## Final Integration

- [ ] Run: `pytest tests/memory tests/cognition -q`
- [ ] Run: `python -m compileall eibrain apps eihead eiprotocol`
- [ ] Review `git diff --stat` for accidental deletions.
- [ ] Review source filtering manually: audit/trace sources blocked from normal dialogue recall.
- [ ] Sync to honxin `/dev-project` before any GitHub push.

