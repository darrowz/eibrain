# Honjia Hontu Voice Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route honjia voice replies through honxin OpenClaw/Hongtu instead of direct model keys in eibrain.

**Architecture:** honjia keeps the real-time hardware loop: ASR text in, TTS audio out. eibrain builds the short embodied prompt and calls an `openclaw_hontu` LLM provider. That provider executes the OpenClaw agent on honxin over SSH, receives JSON text, and returns it to the normal eibrain dialogue/TTS path.

**Tech Stack:** Python 3, pytest, subprocess, OpenClaw CLI, SSH, YAML config.

---

### Task 1: Add OpenClaw/Hongtu LLM Provider

**Files:**
- Modify: `eibrain/infra/config.py`
- Modify: `eibrain/cognition/dialogue/llm_router.py`
- Test: `tests/cognition/test_vision_llm_router.py`
- Test: `tests/infra/test_config_loader.py`

- [ ] **Step 1: Write failing router tests**

Add tests that monkeypatch `subprocess.run`, configure `LLMConfig(provider="openclaw_hontu")`, and assert the router calls `ssh honxin /home/darrow/n/bin/openclaw agent --json --message <prompt>` and extracts `result.payloads[0].text`.

- [ ] **Step 2: Add config fields**

Extend `LLMConfig` with `command`, `agent_id`, `session_id`, and `timeout_s`. Normalize `command` through `_parse_command()` in `load_config()`.

- [ ] **Step 3: Implement provider**

In `LLMRouter.generate()`, route `openclaw_hontu` to a subprocess-backed helper. Parse JSON stdout, return the first text payload, and record `last_status`, `last_error`, `last_text`, and elapsed time exactly like the existing providers.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest -q tests/cognition/test_vision_llm_router.py tests/infra/test_config_loader.py
```

Expected: all selected tests pass.

### Task 2: Switch honjia Config To Hongtu Channel

**Files:**
- Modify: `config/eibrain.honjia.yaml`
- Modify: `config/eibrain.honjia.remote.yaml`

- [ ] **Step 1: Set honjia cognition provider**

Change `cognition.llm.provider` to `openclaw_hontu` and set the command to:

```yaml
command:
  - ssh
  - -o
  - BatchMode=yes
  - -o
  - StrictHostKeyChecking=accept-new
  - honxin
  - env
  - PATH=/home/darrow/n/bin:/usr/local/bin:/usr/bin:/bin
  - /home/darrow/n/bin/openclaw
```

Keep `agent_id: main`, `session_id: eibrain-honjia-voice`, and `timeout_s: 45`.

- [ ] **Step 2: Preserve local fallback config**

Leave `config/eibrain.honjia.local.yaml` as a local direct-model fallback for bench work. The production honjia path uses OpenClaw/Hongtu.

### Task 3: Deploy And Verify On Honjia

**Files:**
- Remote: `/home/darrow/dev-project/eibrain`
- Remote: `/dev-project/eihead`

- [ ] **Step 1: Sync eibrain code to honjia**

Use the existing git remote workflow to pull or copy the changed eibrain files into `/home/darrow/dev-project/eibrain`.

- [ ] **Step 2: Repair honjia SSH trust to honxin**

Remove the stale `honxin` host key and add the current one, then verify:

```bash
ssh -o BatchMode=yes honxin '/home/darrow/n/bin/openclaw gateway call health --json | head -c 120'
```

Expected: command succeeds without password and returns JSON with `"ok": true`.

- [ ] **Step 3: Restart the 18082 dev runtime**

Restart honjia dev runtime with the same PYTHONPATH that fixed `sherpa_onnx`.

- [ ] **Step 4: Run live text probe**

Run `CognitiveRuntimeApp` on honjia with `config/eibrain.honjia.yaml` and transcript `系统联调，请只回复语音链路正常`. Expected reply text comes from OpenClaw and `last_llm_status.provider == "openclaw_hontu"`.

- [ ] **Step 5: Commit and push**

Commit eibrain changes and push to GitHub after focused tests and honjia probe pass.
