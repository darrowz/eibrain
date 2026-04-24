# eibrain <-> eimemory RPC Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect `eibrain` memory retrieval to `eimemory` through a deployment-safe RPC/HTTP boundary without coupling production code to the `/dev-project/eimemory` checkout path.

**Architecture:** `eibrain` keeps an internal memory-adapter boundary and adds a new `eimemory_rpc` provider that talks to `eimemory`'s HTTP RPC server. `eimemory` remains a separate runtime that can be started from any deployment directory, while `eibrain` consumes only a configured endpoint and degrades safely when the service is unavailable.

**Tech Stack:** Python 3, `urllib.request`, dataclass config loading, pytest, local HTTP RPC server from `eimemory.adapters.eibrain.rpc_server`

---

## File Structure

### eibrain repo

- Modify: `/dev-project/eibrain/eibrain/infra/config.py`
  Purpose: extend memory config so `eimemory_rpc` can be configured through endpoint-based runtime settings.
- Create: `/dev-project/eibrain/eibrain/memory/adapters/eimemory_rpc.py`
  Purpose: implement the RPC client adapter that maps `MemoryQuery` to `eimemory` recall RPC and maps the response back to `MemoryResult`.
- Create: `/dev-project/eibrain/eibrain/memory/adapters/factory.py`
  Purpose: centralize provider selection so cognition runtime does not hard-code the legacy adapter.
- Modify: `/dev-project/eibrain/apps/cognitive_runtime/app.py`
  Purpose: build memory adapters through the factory and keep cognition runtime independent from provider details.
- Modify: `/dev-project/eibrain/tests/infra/test_config_loader.py`
  Purpose: verify config loading for `eimemory_rpc` endpoint-oriented fields.
- Create: `/dev-project/eibrain/tests/memory/test_eimemory_rpc_adapter.py`
  Purpose: cover request mapping, response mapping, and degraded behavior.
- Modify: `/dev-project/eibrain/tests/memory/test_openclaw_adapter.py`
  Purpose: preserve legacy adapter coverage after factory/provider refactor.
- Modify: `/dev-project/eibrain/tests/cognition/test_deployable_cognitive_runtime.py`
  Purpose: verify runtime provider selection and integration behavior.
- Modify: `/dev-project/eibrain/config/eibrain.yaml`
- Modify: `/dev-project/eibrain/config/eibrain.honjia.yaml`
- Modify: `/dev-project/eibrain/config/eibrain.honjia.remote.yaml`
  Purpose: replace path-thinking with endpoint-thinking in shipped examples.
- Modify: `/dev-project/eibrain/README.md` or deployment docs if a better location already exists
  Purpose: document the code-path vs deployment-endpoint distinction.

### eimemory repo

- Modify: `/dev-project/eimemory/eimemory/cli/main.py`
  Purpose: add a supported CLI entry for serving the eibrain RPC boundary in deployment.
- Modify: `/dev-project/eimemory/README.md`
  Purpose: document how to start the RPC server and what endpoint `eibrain` should call.
- Modify: `/dev-project/eimemory/tests/test_adapters.py`
  Purpose: add CLI/startup coverage if a new serve command is added.
- Modify: `/dev-project/eimemory/tests/test_platform.py`
  Purpose: keep the existing RPC server boundary tests aligned with the shipped CLI/runtime contract.

### Cross-repo validation

- Create: `/dev-project/eibrain/tests/memory/test_eimemory_rpc_live.py`
  Purpose: verify a real `eimemory` RPC server can be called from `eibrain` in the development environment.

### Task 1: Extend Memory Config And Provider Selection

**Files:**
- Modify: `/dev-project/eibrain/eibrain/infra/config.py`
- Create: `/dev-project/eibrain/eibrain/memory/adapters/factory.py`
- Modify: `/dev-project/eibrain/tests/infra/test_config_loader.py`
- Modify: `/dev-project/eibrain/tests/cognition/test_deployable_cognitive_runtime.py`

- [ ] **Step 1: Write the failing config test**

```python
def test_load_config_reads_eimemory_rpc_fields(tmp_path) -> None:
    from eibrain.infra.config import load_config

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\\n".join(
            [
                "memory:",
                "  openclaw:",
                "    provider: eimemory_rpc",
                "    endpoint: http://127.0.0.1:8091/",
                "    timeout_s: 1.5",
                "    agent_id: honxin",
                "    workspace_id: honjia-prod",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.memory.openclaw.provider == "eimemory_rpc"
    assert config.memory.openclaw.endpoint == "http://127.0.0.1:8091/"
    assert config.memory.openclaw.agent_id == "honxin"
    assert config.memory.openclaw.workspace_id == "honjia-prod"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dev-project/eibrain && pytest tests/infra/test_config_loader.py::test_load_config_reads_eimemory_rpc_fields -v`
Expected: FAIL because `OpenClawConfig` does not yet expose `agent_id` / `workspace_id`.

- [ ] **Step 3: Write the failing provider-selection test**

```python
def test_cognitive_runtime_builds_eimemory_rpc_adapter_from_config(tmp_path) -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\\n".join(
            [
                "memory:",
                "  openclaw:",
                "    provider: eimemory_rpc",
                "    endpoint: http://127.0.0.1:8091/",
            ]
        ),
        encoding="utf-8",
    )

    runtime = CognitiveRuntimeApp.from_config_path(config_path)

    assert runtime.memory.__class__.__name__ == "EIMemoryRPCAdapter"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /dev-project/eibrain && pytest tests/cognition/test_deployable_cognitive_runtime.py::test_cognitive_runtime_builds_eimemory_rpc_adapter_from_config -v`
Expected: FAIL because the runtime still hard-codes `OpenClawMemoryAdapter`.

- [ ] **Step 5: Implement minimal config and factory changes**

```python
@dataclass(slots=True)
class OpenClawConfig:
    provider: str = "in_memory"
    endpoint: str = ""
    api_key: str = ""
    timeout_s: float = 5.0
    agent_id: str = ""
    workspace_id: str = ""
```

```python
from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
from eibrain.memory.adapters.openclaw import OpenClawMemoryAdapter


def build_memory_adapter(config):
    if config.provider == "eimemory_rpc":
        return EIMemoryRPCAdapter(config)
    return OpenClawMemoryAdapter(config)
```

```python
self.memory = build_memory_adapter(self.config.memory.openclaw)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /dev-project/eibrain && pytest tests/infra/test_config_loader.py::test_load_config_reads_eimemory_rpc_fields tests/cognition/test_deployable_cognitive_runtime.py::test_cognitive_runtime_builds_eimemory_rpc_adapter_from_config -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd /dev-project/eibrain
git add eibrain/infra/config.py eibrain/memory/adapters/factory.py apps/cognitive_runtime/app.py tests/infra/test_config_loader.py tests/cognition/test_deployable_cognitive_runtime.py
git commit -m "feat: add eimemory rpc provider selection"
```

### Task 2: Implement The eimemory RPC Adapter Boundary

**Files:**
- Create: `/dev-project/eibrain/eibrain/memory/adapters/eimemory_rpc.py`
- Create: `/dev-project/eibrain/tests/memory/test_eimemory_rpc_adapter.py`

- [ ] **Step 1: Write the failing request-mapping test**

```python
def test_eimemory_rpc_adapter_posts_recall_request(monkeypatch) -> None:
    import json

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
    from eibrain.memory.contracts import MemoryQuery

    captured = {}

    class _Response:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"items": [], "rules": [], "explanation": {}}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/", agent_id="honxin", workspace_id="robot"))
    adapter.retrieve_context(MemoryQuery(query="where is the user", session_id="session-1", actor_id="user-1"))

    assert captured["url"] == "http://127.0.0.1:8091/"
    assert captured["body"]["method"] == "memory.recall"
    assert captured["body"]["params"]["scope"]["agent_id"] == "honxin"
    assert captured["body"]["params"]["scope"]["workspace_id"] == "robot"
    assert captured["body"]["params"]["scope"]["session_id"] == "session-1"
    assert captured["body"]["params"]["scope"]["actor_id"] == "user-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dev-project/eibrain && pytest tests/memory/test_eimemory_rpc_adapter.py::test_eimemory_rpc_adapter_posts_recall_request -v`
Expected: FAIL because the adapter file does not exist yet.

- [ ] **Step 3: Write the failing response-mapping test**

```python
def test_eimemory_rpc_adapter_maps_recall_bundle_to_memory_result(monkeypatch) -> None:
    import json

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
    from eibrain.memory.contracts import MemoryQuery

    class _Response:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self) -> bytes:
            return json.dumps(
                {
                    "ok": True,
                    "result": {
                        "items": [
                            {"title": "Reply Style", "summary": "Prefer concise spoken replies."},
                            {"title": "Context", "summary": "The current speaker is in front of the robot."},
                        ],
                        "rules": [{"title": "Speak briefly"}],
                        "explanation": {"session_summary": "Recent dialogue is concise."},
                    },
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "eibrain.memory.adapters.eimemory_rpc.request.urlopen",
        lambda req, timeout=0: _Response(),
    )

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    result = adapter.retrieve_context(MemoryQuery(query="reply to user"))

    assert "Prefer concise spoken replies." in result.summary
    assert result.relevant_memories == [
        "Reply Style: Prefer concise spoken replies.",
        "Context: The current speaker is in front of the robot.",
    ]
    assert result.session_summary == "Recent dialogue is concise."
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /dev-project/eibrain && pytest tests/memory/test_eimemory_rpc_adapter.py::test_eimemory_rpc_adapter_maps_recall_bundle_to_memory_result -v`
Expected: FAIL because no response mapping exists yet.

- [ ] **Step 5: Implement minimal RPC adapter**

```python
class EIMemoryRPCAdapter:
    def __init__(self, config) -> None:
        self.config = config

    def retrieve_context(self, query: MemoryQuery) -> MemoryResult:
        payload = {
            "method": "memory.recall",
            "params": {
                "query": query.query,
                "limit": 8,
                "scope": self._scope(query),
                "task_context": {"task_type": "brain.respond", "goal": "retrieve memory for embodied response"},
            },
        }
        result = self._post(payload)
        return self._map_result(result)
```

```python
def _map_result(self, payload: dict[str, object]) -> MemoryResult:
    result = dict(payload.get("result", {}))
    items = [dict(item) for item in result.get("items", [])]
    rules = [dict(rule) for rule in result.get("rules", [])]
    explanation = dict(result.get("explanation", {}))
    memories = [f"{item.get('title', 'memory')}: {item.get('summary', '')}".strip() for item in items]
    summary_parts = [item.get("summary", "") for item in items[:2]] + [rule.get("title", "") for rule in rules[:1]]
    return MemoryResult(
        summary=" | ".join(part for part in summary_parts if part),
        relevant_memories=[item for item in memories if item and item != ":"],
        actor_profile={},
        session_summary=str(explanation.get("session_summary", "")),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /dev-project/eibrain && pytest tests/memory/test_eimemory_rpc_adapter.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd /dev-project/eibrain
git add eibrain/memory/adapters/eimemory_rpc.py tests/memory/test_eimemory_rpc_adapter.py
git commit -m "feat: add eimemory rpc memory adapter"
```

### Task 3: Add Graceful Degradation And Wire The Runtime End-To-End

**Files:**
- Modify: `/dev-project/eibrain/eibrain/memory/adapters/eimemory_rpc.py`
- Modify: `/dev-project/eibrain/apps/cognitive_runtime/app.py`
- Modify: `/dev-project/eibrain/tests/cognition/test_deployable_cognitive_runtime.py`
- Modify: `/dev-project/eibrain/tests/memory/test_openclaw_adapter.py`

- [ ] **Step 1: Write the failing degraded-behavior test**

```python
def test_eimemory_rpc_adapter_gracefully_degrades_on_transport_failure(monkeypatch) -> None:
    from urllib.error import URLError

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
    from eibrain.memory.contracts import MemoryQuery

    monkeypatch.setattr(
        "eibrain.memory.adapters.eimemory_rpc.request.urlopen",
        lambda req, timeout=0: (_ for _ in ()).throw(URLError("down")),
    )

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    result = adapter.retrieve_context(MemoryQuery(query="hello", session_id="s1", actor_id="user-1"))

    assert "hello" in result.summary
    assert result.session_summary == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dev-project/eibrain && pytest tests/memory/test_eimemory_rpc_adapter.py::test_eimemory_rpc_adapter_gracefully_degrades_on_transport_failure -v`
Expected: FAIL because failures are not mapped to degraded local results yet.

- [ ] **Step 3: Write the failing runtime behavior test**

```python
def test_cognitive_runtime_uses_eimemory_rpc_memory_in_handle_observation(monkeypatch, tmp_path) -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.protocol.observations import AudioTranscriptFinal
    from eibrain.memory.contracts import MemoryResult

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\\n".join(
            [
                "cognition:",
                "  llm:",
                "    provider: echo",
                "memory:",
                "  openclaw:",
                "    provider: eimemory_rpc",
                "    endpoint: http://127.0.0.1:8091/",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "eibrain.memory.adapters.eimemory_rpc.EIMemoryRPCAdapter.retrieve_context",
        lambda self, query: MemoryResult(summary="Prefer concise replies."),
    )

    runtime = CognitiveRuntimeApp.from_config_path(config_path)
    actions = runtime.handle_observation(
        AudioTranscriptFinal(ts=1.0, source="ear.asr", text="hello", session_id="s1", actor_id="user-1")
    )

    assert actions
    assert runtime.memory.__class__.__name__ == "EIMemoryRPCAdapter"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /dev-project/eibrain && pytest tests/cognition/test_deployable_cognitive_runtime.py::test_cognitive_runtime_uses_eimemory_rpc_memory_in_handle_observation -v`
Expected: FAIL until runtime wiring is complete.

- [ ] **Step 5: Implement degraded fallback and runtime cleanup**

```python
except (URLError, OSError, ValueError, KeyError, TypeError):
    return MemoryResult(
        summary=f"eimemory-context:{query.query}",
        relevant_memories=[],
        actor_profile={},
        session_summary="",
    )
```

```python
self.memory = build_memory_adapter(self.config.memory.openclaw)
```

- [ ] **Step 6: Run focused tests to verify they pass**

Run: `cd /dev-project/eibrain && pytest tests/memory/test_eimemory_rpc_adapter.py tests/cognition/test_deployable_cognitive_runtime.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd /dev-project/eibrain
git add eibrain/memory/adapters/eimemory_rpc.py apps/cognitive_runtime/app.py tests/memory/test_eimemory_rpc_adapter.py tests/cognition/test_deployable_cognitive_runtime.py tests/memory/test_openclaw_adapter.py
git commit -m "feat: wire cognitive runtime to eimemory rpc"
```

### Task 4: Ship A Supported eimemory RPC Server Entry Point

**Files:**
- Modify: `/dev-project/eimemory/eimemory/cli/main.py`
- Modify: `/dev-project/eimemory/tests/test_adapters.py`
- Modify: `/dev-project/eimemory/tests/test_platform.py`
- Modify: `/dev-project/eimemory/README.md`

- [ ] **Step 1: Write the failing CLI test**

```python
def test_cli_can_serve_eibrain_rpc(tmp_path, monkeypatch) -> None:
    from eimemory.cli.main import main as cli_main

    monkeypatch.setenv("EIMEMORY_ROOT", str(tmp_path / "runtime"))
    exit_code = cli_main(["serve-eibrain-rpc", "--host", "127.0.0.1", "--port", "0"])

    assert exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dev-project/eimemory && pytest tests/test_adapters.py::test_cli_can_serve_eibrain_rpc -v`
Expected: FAIL because no CLI command exists.

- [ ] **Step 3: Implement the minimal serve command**

```python
serve_rpc = sub.add_parser("serve-eibrain-rpc")
serve_rpc.add_argument("--host", default="127.0.0.1")
serve_rpc.add_argument("--port", type=int, default=8091)
```

```python
if parsed.command == "serve-eibrain-rpc":
    from eimemory.adapters.eibrain.rpc_server import EIBrainRPCServer

    server = EIBrainRPCServer(runtime, host=parsed.host, port=parsed.port)
    server.start()
    print(json.dumps({"ok": True, "host": server.address[0], "port": server.address[1]}, ensure_ascii=False))
    server.stop()
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /dev-project/eimemory && pytest tests/test_adapters.py::test_cli_can_serve_eibrain_rpc tests/test_platform.py::test_http_rpc_server_serves_recall_and_policy -v`
Expected: PASS

- [ ] **Step 5: Document the deployment-safe startup contract**

```md
### eibrain RPC server

Start the memory boundary from the deployed environment, not from a source checkout path:

```bash
EIMEMORY_ROOT=/var/lib/eimemory eimemory serve-eibrain-rpc --host 127.0.0.1 --port 8091
```

Point `eibrain` at `http://127.0.0.1:8091/` through config or environment variables.
```

- [ ] **Step 6: Commit**

```bash
cd /dev-project/eimemory
git add eimemory/cli/main.py tests/test_adapters.py tests/test_platform.py README.md
git commit -m "feat: add eibrain rpc serve command"
```

### Task 5: Add Live Boundary Validation And Deployment Docs

**Files:**
- Create: `/dev-project/eibrain/tests/memory/test_eimemory_rpc_live.py`
- Modify: `/dev-project/eibrain/config/eibrain.yaml`
- Modify: `/dev-project/eibrain/config/eibrain.honjia.yaml`
- Modify: `/dev-project/eibrain/config/eibrain.honjia.remote.yaml`
- Modify: `/dev-project/eibrain/README.md`

- [ ] **Step 1: Write the failing live-boundary test**

```python
def test_eimemory_rpc_adapter_reads_from_live_server(tmp_path) -> None:
    from eimemory.adapters.eibrain.rpc_server import EIBrainRPCServer
    from eimemory.api.runtime import Runtime

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
    from eibrain.memory.contracts import MemoryQuery

    runtime = Runtime.create(root=tmp_path / "eimemory-runtime")
    runtime.memory.ingest(
        text="Prefer concise embodied replies.",
        memory_type="preference",
        title="Reply style",
        scope={"agent_id": "honxin", "workspace_id": "robot"},
    )
    server = EIBrainRPCServer(runtime, host="127.0.0.1", port=0)
    server.start()
    try:
        adapter = EIMemoryRPCAdapter(
            OpenClawConfig(provider="eimemory_rpc", endpoint=f"http://{server.address[0]}:{server.address[1]}/", agent_id="honxin", workspace_id="robot")
        )
        result = adapter.retrieve_context(MemoryQuery(query="concise reply"))
    finally:
        server.stop()
        runtime.close()

    assert "concise" in result.summary.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dev-project/eibrain && PYTHONPATH=/dev-project/eimemory pytest tests/memory/test_eimemory_rpc_live.py::test_eimemory_rpc_adapter_reads_from_live_server -v`
Expected: FAIL until the adapter and scope mapping are correct.

- [ ] **Step 3: Update shipped config examples to endpoint-based values**

```yaml
memory:
  openclaw:
    provider: eimemory_rpc
    endpoint: ${EIMEMORY_ENDPOINT:-http://127.0.0.1:8091/}
    timeout_s: 3.0
    agent_id: honxin
    workspace_id: honjia
```

- [ ] **Step 4: Update deployment docs to separate code path from runtime path**

```md
- Source checkout example: `/dev-project/eimemory`
- Deployed service example: `/opt/eimemory/current`
- `eibrain` must never depend on either path directly; it only depends on `EIMEMORY_ENDPOINT`.
```

- [ ] **Step 5: Run full verification for both repos**

Run: `cd /dev-project/eibrain && pytest tests/infra/test_config_loader.py tests/memory/test_openclaw_adapter.py tests/memory/test_eimemory_rpc_adapter.py tests/cognition/test_deployable_cognitive_runtime.py -v`
Expected: PASS

Run: `cd /dev-project/eibrain && PYTHONPATH=/dev-project/eimemory pytest tests/memory/test_eimemory_rpc_live.py -v`
Expected: PASS

Run: `cd /dev-project/eimemory && pytest tests/test_adapters.py tests/test_platform.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /dev-project/eibrain
git add tests/memory/test_eimemory_rpc_live.py config/eibrain.yaml config/eibrain.honjia.yaml config/eibrain.honjia.remote.yaml README.md
git commit -m "docs: point eibrain memory to eimemory endpoint"
```

## Self-Review Checklist

- Spec coverage: config shape, RPC adapter, runtime wiring, graceful degradation, live boundary validation, and deployment-path separation are all covered by Tasks 1-5.
- Placeholder scan: this plan contains concrete file paths, test names, commands, expected failures, implementation snippets, and commit points.
- Type consistency: the plan consistently uses `EIMemoryRPCAdapter`, `build_memory_adapter`, `provider: eimemory_rpc`, and `memory.openclaw` config fields extended with `agent_id` and `workspace_id`.
