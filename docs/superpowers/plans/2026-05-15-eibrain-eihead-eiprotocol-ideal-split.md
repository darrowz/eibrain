# eibrain / eihead / eiprotocol Ideal Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the EI workspace toward the ideal physical structure in one deliberate pass: `eiprotocol` as the shared contract package, `eihead` as the honjia head runtime package, and `eibrain` as the cognition/body-orchestration package.

**Architecture:** Add boundary tests first so import shadowing cannot hide embedded copies. Then detach `eiprotocol`, detach `eihead` from copied `eibrain` code, and finally slim `eibrain` packaging and tests. Cross-repository runtime meaning flows through `eiprotocol`; internal package implementation remains normal Python APIs.

**Tech Stack:** Python 3.10+, setuptools, pytest, Git submodules under `D:\github\ei-workspace`, editable installs, existing EI packages.

---

## Touch / Do Not Touch Map

### Touch Now

- `D:\github\ei-workspace\repos\eiprotocol`
  - Keep as canonical protocol source.
  - Strengthen no-reverse-import tests.
  - Keep protocol fixtures and protocol-only tests here.
- `D:\github\ei-workspace\repos\eihead`
  - Remove embedded `eiprotocol/`.
  - Remove embedded `eibrain/`.
  - Remove `apps/body_runtime/` from packaged runtime.
  - Keep and improve native `eihead/` modules.
  - Keep `apps/head_runtime/` only as a temporary CLI compatibility shim if it delegates to `eihead.runtime`.
- `D:\github\ei-workspace\repos\eibrain`
  - Stop packaging `eihead*` and `eiprotocol*`.
  - Remove embedded `eihead/` and `eiprotocol/` after standalone imports pass.
  - Keep only eibrain-owned bridge/integration tests.
  - Retire or narrow export scripts once standalone repos are authoritative.

### Do Not Touch In This Split

- Do not change `eiprotocol` event names, payload semantics, catalog routes, or validation behavior unless a boundary test proves the existing contract is impossible to use.
- Do not redesign `eimemory` or the `EIMEMORY_ENDPOINT` runtime contract.
- Do not change honjia deployment ports: runtime HTTP stays `18081`, monitor stays `18080`.
- Do not claim hardware readiness or cutover completion.
- Do not rewrite `eibrain.cognition` internals just because the repository is being split.
- Do not move `eibrain.memory` durable-memory policy into `eihead`.
- Do not make `eiprotocol` depend on provider SDKs, hardware libraries, `eibrain`, `eihead`, or `apps`.
- Do not keep export-copy code as a second source of truth after the corresponding standalone package owns it.

### Move Or Replace

- `eihead` currently importing `eibrain.protocol.joyinside_voice` should get an `eihead.eivoice_runtime.joyinside_voice` module or equivalent local provider mapping.
- `eihead` currently importing `eibrain.voice.readiness` should get native `eihead.monitoring.voice_readiness` helpers.
- `eihead` currently carrying `apps.body_runtime` and `eibrain.body` should move only honjia-native device code into `eihead.devices`, `eihead.eye`, `eihead.ear`, `eihead.mouth`, and `eihead.neck`.
- Protocol-only tests currently under `eibrain/tests/protocol` should live in `eiprotocol/tests/protocol`.
- Head-only tests currently under `eibrain/tests/eihead` should live in `eihead/tests`.
- Eibrain bridge tests should stay under `eibrain/tests/protocol` only when they exercise `eibrain/protocol/eiprotocol_bridge.py`.

## Preflight

- [ ] **Step 1: Confirm all three repositories are clean**

Run:

```powershell
rtk git -C D:/github/ei-workspace/repos/eiprotocol status --short
rtk git -C D:/github/ei-workspace/repos/eihead status --short
rtk git -C D:/github/ei-workspace/repos/eibrain status --short
```

Expected: no output, or only the plan file if this plan is being committed before execution.

- [ ] **Step 2: Install editable dependencies in dependency order**

Run:

```powershell
python -m pip install -e D:/github/ei-workspace/repos/eiprotocol
python -m pip install -e D:/github/ei-workspace/repos/eihead
python -m pip install -e D:/github/ei-workspace/repos/eibrain
```

Expected: each package installs successfully.

---

### Task 1: Strengthen Protocol Source-Of-Truth Guardrails

**Files:**
- Modify: `D:\github\ei-workspace\repos\eiprotocol\tests\test_no_reverse_imports.py`
- Test: `D:\github\ei-workspace\repos\eiprotocol\tests\test_no_reverse_imports.py`

- [ ] **Step 1: Replace substring scanning with AST import scanning**

Use this complete test body:

```python
import ast
from pathlib import Path


FORBIDDEN_ROOTS = {"apps", "eibrain", "eihead", "eimemory"}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_eiprotocol_has_no_reverse_runtime_imports() -> None:
    package_root = Path(__file__).resolve().parents[1] / "eiprotocol"
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        forbidden = _import_roots(path) & FORBIDDEN_ROOTS
        if forbidden:
            relative = path.relative_to(package_root.parent)
            offenders.append(f"{relative}: {sorted(forbidden)}")

    assert offenders == []
```

- [ ] **Step 2: Run protocol boundary test**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eiprotocol/tests/test_no_reverse_imports.py -q
```

Expected: pass.

- [ ] **Step 3: Run full eiprotocol tests**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eiprotocol/tests -q
```

Expected: pass.

- [ ] **Step 4: Commit eiprotocol guardrail**

Run:

```powershell
rtk git -C D:/github/ei-workspace/repos/eiprotocol add tests/test_no_reverse_imports.py
rtk git -C D:/github/ei-workspace/repos/eiprotocol commit -m "test: enforce protocol dependency boundary"
```

Expected: commit succeeds.

---

### Task 2: Make eihead Consume Standalone eiprotocol

**Files:**
- Create: `D:\github\ei-workspace\repos\eihead\tests\test_protocol_dependency_boundary.py`
- Modify: `D:\github\ei-workspace\repos\eihead\pyproject.toml`
- Modify: `D:\github\ei-workspace\repos\eihead\README.md`
- Delete: `D:\github\ei-workspace\repos\eihead\eiprotocol\`

- [ ] **Step 1: Write failing eihead protocol boundary test**

Create `tests/test_protocol_dependency_boundary.py`:

```python
import importlib.util
from pathlib import Path


def test_eihead_has_no_embedded_eiprotocol_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "eiprotocol").exists()


def test_eihead_imports_protocol_from_standalone_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.find_spec("eiprotocol")

    assert spec is not None
    assert spec.origin is not None
    assert repo_root not in Path(spec.origin).resolve().parents
```

- [ ] **Step 2: Run test and confirm it fails before deletion**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eihead/tests/test_protocol_dependency_boundary.py -q
```

Expected: fail because `D:\github\ei-workspace\repos\eihead\eiprotocol` still exists.

- [ ] **Step 3: Update eihead packaging metadata**

Modify `D:\github\ei-workspace\repos\eihead\pyproject.toml` so the project section has:

```toml
dependencies = ["PyYAML>=6.0", "eiprotocol>=0.1.0"]
```

and the package find include list is reduced to:

```toml
include = [
    "eihead*",
    "apps",
    "apps.head_runtime*",
]
```

Do not include `eiprotocol*`, `eibrain*`, or `apps.body_runtime*`.

- [ ] **Step 4: Delete embedded protocol copy**

Delete:

```text
D:\github\ei-workspace\repos\eihead\eiprotocol\
```

Use PowerShell native removal, after verifying the resolved path:

```powershell
$target = Resolve-Path D:/github/ei-workspace/repos/eihead/eiprotocol
if ($target.Path -ne "D:\github\ei-workspace\repos\eihead\eiprotocol") { throw "unexpected delete target: $target" }
Remove-Item -LiteralPath $target.Path -Recurse -Force
```

- [ ] **Step 5: Update README protocol wording**

Replace wording that says the shared protocol package is exported into `eihead` with wording like:

~~~markdown
`eihead` consumes `eiprotocol` as a standalone dependency. Install both from the parent workspace during development:

```bash
python -m pip install -e D:/github/ei-workspace/repos/eiprotocol
python -m pip install -e D:/github/ei-workspace/repos/eihead
```
~~~

- [ ] **Step 6: Run eihead protocol boundary tests**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eihead/tests/test_protocol_dependency_boundary.py -q
```

Expected: pass.

- [ ] **Step 7: Run focused eihead tests that import eiprotocol**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eihead/tests -q
```

Expected: pass. If failures remain, stop and classify each failure as either a broken standalone `eiprotocol` dependency or an `eibrain` detachment blocker before continuing. Do not reintroduce embedded `eiprotocol`.

- [ ] **Step 8: Commit eihead protocol dependency**

Run:

```powershell
rtk git -C D:/github/ei-workspace/repos/eihead add pyproject.toml README.md tests/test_protocol_dependency_boundary.py
rtk git -C D:/github/ei-workspace/repos/eihead add -u eiprotocol
rtk git -C D:/github/ei-workspace/repos/eihead commit -m "refactor: consume standalone eiprotocol"
```

Expected: commit succeeds.

---

### Task 3: Remove eihead Imports From eibrain Compatibility Code

**Files:**
- Create: `D:\github\ei-workspace\repos\eihead\tests\test_no_eibrain_runtime_imports.py`
- Create: `D:\github\ei-workspace\repos\eihead\eihead\eivoice_runtime\joyinside_voice.py`
- Create: `D:\github\ei-workspace\repos\eihead\eihead\monitoring\voice_readiness.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\eivoice_runtime\transport.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\eivoice_runtime\runtime.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\monitoring\voice.py`
- Delete: `D:\github\ei-workspace\repos\eihead\eibrain\protocol\`
- Delete: `D:\github\ei-workspace\repos\eihead\eibrain\voice\`

- [ ] **Step 1: Write failing no-eibrain-imports test**

Create `tests/test_no_eibrain_runtime_imports.py`:

```python
import ast
from pathlib import Path


FORBIDDEN_ROOTS = {"eibrain"}
SCAN_ROOTS = ("eihead", "apps/head_runtime")


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_eihead_runtime_has_no_eibrain_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for scan_root in SCAN_ROOTS:
        root = repo_root / scan_root
        for path in sorted(root.rglob("*.py")):
            forbidden = _import_roots(path) & FORBIDDEN_ROOTS
            if forbidden:
                offenders.append(f"{path.relative_to(repo_root)}: {sorted(forbidden)}")

    assert offenders == []
```

- [ ] **Step 2: Run test and confirm it fails**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eihead/tests/test_no_eibrain_runtime_imports.py -q
```

Expected: fail on current `eihead/eivoice_runtime` and `eihead/monitoring/voice.py` imports.

- [ ] **Step 3: Move JoyInside voice helpers into eihead**

Create `eihead/eivoice_runtime/joyinside_voice.py` by copying the needed public helpers from `D:\github\ei-workspace\repos\eihead\eibrain\protocol\joyinside_voice.py`.

Required exports:

```python
__all__ = [
    "JoyInsideVoiceEvent",
    "audio_chunk",
    "ping",
]
```

If the source file exports more helpers used only by tests, keep them in this module too. Do not import `eibrain`.

- [ ] **Step 4: Update eihead eivoice imports**

Change imports:

```python
from eibrain.protocol.joyinside_voice import JoyInsideVoiceEvent, ping
```

to:

```python
from eihead.eivoice_runtime.joyinside_voice import JoyInsideVoiceEvent, ping
```

and change:

```python
from eibrain.protocol.joyinside_voice import audio_chunk
```

to:

```python
from eihead.eivoice_runtime.joyinside_voice import audio_chunk
```

- [ ] **Step 5: Move voice readiness helper into eihead**

Create `eihead/monitoring/voice_readiness.py` by copying the needed behavior from `D:\github\ei-workspace\repos\eihead\eibrain\voice\readiness.py`.

Required public function signature:

```python
def build_voice_chain_readiness(payload: object | None = None) -> dict[str, object]:
```

Copy the existing implementation from `D:\github\ei-workspace\repos\eihead\eibrain\voice\readiness.py` and keep behavior unchanged; only change ownership.

- [ ] **Step 6: Update voice monitoring import**

Change:

```python
from eibrain.voice.readiness import build_voice_chain_readiness
```

to:

```python
from eihead.monitoring.voice_readiness import build_voice_chain_readiness
```

- [ ] **Step 7: Run no-eibrain-imports test**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eihead/tests/test_no_eibrain_runtime_imports.py -q
```

Expected: pass for `eihead/` and `apps/head_runtime/`.

- [ ] **Step 8: Run focused voice and runtime tests**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eihead/tests -q
```

Expected: pass or reveal only legacy `apps.body_runtime` detachment work handled by Task 4.

- [ ] **Step 9: Commit eihead local helper ownership**

Run:

```powershell
rtk git -C D:/github/ei-workspace/repos/eihead add eihead/eivoice_runtime eihead/monitoring tests/test_no_eibrain_runtime_imports.py
rtk git -C D:/github/ei-workspace/repos/eihead commit -m "refactor: remove eibrain imports from eihead runtime"
```

Expected: commit succeeds.

---

### Task 4: Detach eihead From Legacy apps.body_runtime And Embedded eibrain

**Files:**
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\runtime\app.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\runtime\legacy_body.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\runtime\cli.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\runtime\composition.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\devices\*.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\eye\*.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\ear\*.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\mouth\*.py`
- Modify: `D:\github\ei-workspace\repos\eihead\eihead\neck\*.py`
- Modify: `D:\github\ei-workspace\repos\eihead\pyproject.toml`
- Modify: `D:\github\ei-workspace\repos\eihead\README.md`
- Modify: `D:\github\ei-workspace\repos\eihead\EXPORT_MANIFEST.json`
- Delete: `D:\github\ei-workspace\repos\eihead\apps\body_runtime\`
- Delete: `D:\github\ei-workspace\repos\eihead\eibrain\`

- [ ] **Step 1: Write failing embedded-copy test**

Create `D:\github\ei-workspace\repos\eihead\tests\test_no_embedded_eibrain_copy.py`:

```python
from pathlib import Path


def test_eihead_has_no_embedded_eibrain_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "eibrain").exists()


def test_eihead_has_no_legacy_body_runtime_app_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "apps" / "body_runtime").exists()
```

- [ ] **Step 2: Run test and confirm it fails**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eihead/tests/test_no_embedded_eibrain_copy.py -q
```

Expected: fail because embedded copies still exist.

- [ ] **Step 3: Make `HeadRuntimeApp` native-only by default**

Update `eihead/runtime/app.py` so snapshot/status/action paths use `eihead` native providers and do not instantiate `apps.body_runtime.BodyRuntimeApp`.

The runtime must still return truthful missing-provider states:

```python
{
    "status": "not_wired",
    "reason": "native_provider_unavailable",
}
```

for any missing eye, ear, mouth, or neck capability.

- [ ] **Step 4: Reduce `legacy_body.py` to an explicit deprecated shim or delete it**

If any test still requires `eihead.runtime.legacy_body`, make the file explicit:

```python
class LegacyBodyRuntimeAdapter:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "LegacyBodyRuntimeAdapter has been removed; use eihead native providers"
        )
```

If no test imports it, delete `eihead/runtime/legacy_body.py` and remove imports.

- [ ] **Step 5: Move hardware helper ownership into eihead native modules**

Only move helpers that are actually used by `eihead` runtime:

- camera/Hailo helpers -> `eihead.eye`
- U4K/audio status helpers -> `eihead.ear` or `eihead.devices.audio`
- playback helpers -> `eihead.mouth`
- Raspbot/I2C/yaw helpers -> `eihead.neck` or `eihead.devices.neck_servo`
- verification helpers -> `eihead.runtime.cli` subcommands or `eihead.devices`

Do not move eibrain cognition, memory, learning, or body orchestration code.

- [ ] **Step 6: Remove `apps.body_runtime` from eihead packaging**

Ensure `pyproject.toml` includes only:

```toml
include = [
    "eihead*",
    "apps",
    "apps.head_runtime*",
]
```

- [ ] **Step 7: Delete embedded app/body and eibrain copies**

Delete:

```text
D:\github\ei-workspace\repos\eihead\apps\body_runtime\
D:\github\ei-workspace\repos\eihead\eibrain\
```

Use resolved-path checks before recursive removal:

```powershell
$body = Resolve-Path D:/github/ei-workspace/repos/eihead/apps/body_runtime
if ($body.Path -ne "D:\github\ei-workspace\repos\eihead\apps\body_runtime") { throw "unexpected delete target: $body" }
Remove-Item -LiteralPath $body.Path -Recurse -Force

$brain = Resolve-Path D:/github/ei-workspace/repos/eihead/eibrain
if ($brain.Path -ne "D:\github\ei-workspace\repos\eihead\eibrain") { throw "unexpected delete target: $brain" }
Remove-Item -LiteralPath $brain.Path -Recurse -Force
```

- [ ] **Step 8: Update `EXPORT_MANIFEST.json` truthfully**

Set:

```json
{
  "code_completion": {
    "legacy_body_runtime_detached": true,
    "full_detachment_claim_allowed": false
  },
  "cutover_readiness": {
    "legacy_body_runtime_detached": true,
    "hardware_verified": false,
    "honjia_cutover": "blocked_by_hardware_validation"
  },
  "transitional_packages": []
}
```

Keep `hardware_verified: false` and `honjia_cutover: blocked_by_hardware_validation`.

- [ ] **Step 9: Update eihead README**

Remove wording that describes generated export copies. Say:

~~~markdown
`eihead` is the canonical honjia head runtime package. It depends on standalone `eiprotocol` for shared event contracts and does not vendor `eibrain`.
~~~

- [ ] **Step 10: Run eihead boundary and runtime tests**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eihead/tests/test_protocol_dependency_boundary.py D:/github/ei-workspace/repos/eihead/tests/test_no_eibrain_runtime_imports.py D:/github/ei-workspace/repos/eihead/tests/test_no_embedded_eibrain_copy.py -q
rtk pytest D:/github/ei-workspace/repos/eihead/tests -q
```

Expected: pass.

- [ ] **Step 11: Commit eihead detachment**

Run:

```powershell
rtk git -C D:/github/ei-workspace/repos/eihead add .
rtk git -C D:/github/ei-workspace/repos/eihead add -u
rtk git -C D:/github/ei-workspace/repos/eihead commit -m "refactor: detach eihead from embedded eibrain"
```

Expected: commit succeeds.

---

### Task 5: Slim eibrain Package Metadata And Head Entrypoints

**Files:**
- Create: `D:\github\ei-workspace\repos\eibrain\tests\test_dependency_boundaries.py`
- Modify: `D:\github\ei-workspace\repos\eibrain\pyproject.toml`
- Modify: `D:\github\ei-workspace\repos\eibrain\README.md`
- Delete or shim: `D:\github\ei-workspace\repos\eibrain\apps\head_runtime\`
- Delete: `D:\github\ei-workspace\repos\eibrain\eihead\`
- Delete: `D:\github\ei-workspace\repos\eibrain\eiprotocol\`

- [ ] **Step 1: Write failing eibrain package boundary test**

Create `tests/test_dependency_boundaries.py`:

```python
import importlib.util
from pathlib import Path


def _origin_for(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    assert spec is not None
    assert spec.origin is not None
    return Path(spec.origin).resolve()


def test_eibrain_does_not_vendor_eihead_or_eiprotocol() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "eihead").exists()
    assert not (repo_root / "eiprotocol").exists()


def test_eibrain_uses_standalone_eihead_and_eiprotocol_when_imported() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for module_name in ("eihead", "eiprotocol"):
        origin = _origin_for(module_name)
        assert repo_root not in origin.parents
```

- [ ] **Step 2: Run test and confirm it fails**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eibrain/tests/test_dependency_boundaries.py -q
```

Expected: fail because `eibrain/eihead` and `eibrain/eiprotocol` still exist.

- [ ] **Step 3: Update eibrain dependencies**

Modify `pyproject.toml` dependencies to include:

```toml
dependencies = [
    "PyYAML>=6.0",
    "faster-whisper>=1.2.1",
    "sherpa-onnx==1.12.33",
    "smbus2>=0.5.0",
    "eiprotocol>=0.1.0",
]
```

Do not add `eihead` as a hard dependency unless a remaining runtime import requires it. Prefer moving head-only code to `eihead`.

- [ ] **Step 4: Remove eihead console scripts from eibrain**

Remove these from `pyproject.toml`:

```toml
eihead-runtime = "eihead.runtime.cli:main"
eihead-verify-hardware = "eihead.runtime.cli:verify_hardware_main"
```

Those scripts belong in `D:\github\ei-workspace\repos\eihead\pyproject.toml`.

- [ ] **Step 5: Reduce eibrain package include list**

Change:

```toml
include = ["eibrain*", "apps*", "eihead*", "eiprotocol*"]
```

to:

```toml
include = ["eibrain*", "apps*"]
```

- [ ] **Step 6: Remove or delegate `apps.head_runtime`**

Preferred: delete `D:\github\ei-workspace\repos\eibrain\apps\head_runtime\` after its tests have moved to `eihead`.

Temporary shim only if needed:

```python
from eihead.runtime.cli import main, verify_hardware_main

__all__ = ["main", "verify_hardware_main"]
```

The shim must not contain implementation logic and must be scheduled for deletion.

- [ ] **Step 7: Delete embedded packages**

Delete:

```text
D:\github\ei-workspace\repos\eibrain\eihead\
D:\github\ei-workspace\repos\eibrain\eiprotocol\
```

Use resolved-path checks before recursive removal:

```powershell
$head = Resolve-Path D:/github/ei-workspace/repos/eibrain/eihead
if ($head.Path -ne "D:\github\ei-workspace\repos\eibrain\eihead") { throw "unexpected delete target: $head" }
Remove-Item -LiteralPath $head.Path -Recurse -Force

$protocol = Resolve-Path D:/github/ei-workspace/repos/eibrain/eiprotocol
if ($protocol.Path -ne "D:\github\ei-workspace\repos\eibrain\eiprotocol") { throw "unexpected delete target: $protocol" }
Remove-Item -LiteralPath $protocol.Path -Recurse -Force
```

- [ ] **Step 8: Update README install order**

Use:

~~~markdown
python -m pip install -e D:/github/ei-workspace/repos/eiprotocol
python -m pip install -e D:/github/ei-workspace/repos/eihead
python -m pip install -e D:/github/ei-workspace/repos/eibrain
~~~

State that `eibrain` owns cognition/body orchestration and consumes standalone protocol contracts.

- [ ] **Step 9: Run eibrain dependency tests**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eibrain/tests/test_repo_imports.py D:/github/ei-workspace/repos/eibrain/tests/test_dependency_boundaries.py -q
```

Expected: pass.

- [ ] **Step 10: Run focused eibrain tests**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eibrain/tests/body D:/github/ei-workspace/repos/eibrain/tests/cognition D:/github/ei-workspace/repos/eibrain/tests/memory D:/github/ei-workspace/repos/eibrain/tests/integration -q
```

Expected: pass or reveal tests that should move to `eihead` or `eiprotocol`.

- [ ] **Step 11: Commit eibrain package slimming**

Run:

```powershell
rtk git -C D:/github/ei-workspace/repos/eibrain add pyproject.toml README.md tests/test_dependency_boundaries.py
rtk git -C D:/github/ei-workspace/repos/eibrain add -u apps/head_runtime eihead eiprotocol
rtk git -C D:/github/ei-workspace/repos/eibrain commit -m "refactor: consume standalone ei packages"
```

Expected: commit succeeds.

---

### Task 6: Move Tests To Their Owning Repositories

**Files:**
- Move: protocol-only tests from `D:\github\ei-workspace\repos\eibrain\tests\protocol\` to `D:\github\ei-workspace\repos\eiprotocol\tests\protocol\`
- Move: protocol fixtures from `D:\github\ei-workspace\repos\eibrain\tests\fixtures\eiprotocol\` to `D:\github\ei-workspace\repos\eiprotocol\tests\fixtures\eiprotocol\`
- Move: head-only tests from `D:\github\ei-workspace\repos\eibrain\tests\eihead\` to `D:\github\ei-workspace\repos\eihead\tests\`
- Keep: `D:\github\ei-workspace\repos\eibrain\tests\protocol\test_eiprotocol_bridge.py`
- Keep: eibrain integration tests that exercise body/cognitive/memory behavior.

- [ ] **Step 1: Classify protocol tests**

Run:

```powershell
rtk rg "eibrain|apps|eihead" D:/github/ei-workspace/repos/eibrain/tests/protocol -g "*.py"
```

Expected: tests without these imports are protocol-only and should move to `eiprotocol`.

- [ ] **Step 2: Classify head tests**

Run:

```powershell
rtk rg "eibrain.cognition|eibrain.memory|apps.cognitive_runtime" D:/github/ei-workspace/repos/eibrain/tests/eihead -g "*.py"
```

Expected: most tests are head-only and should move to `eihead`. Any test that uses eibrain cognition or memory should be rewritten as an integration test or left in `eibrain/tests/integration`.

- [ ] **Step 3: Move protocol-only tests and fixtures**

Use PowerShell `Move-Item` with resolved source/target checks. Do not overwrite existing standalone tests; if a file already exists, compare content first and keep the canonical standalone version.

- [ ] **Step 4: Move head-only tests**

Move head-only test files into `D:\github\ei-workspace\repos\eihead\tests\`. Keep names stable unless they conflict with existing files.

- [ ] **Step 5: Update imports in moved tests**

Protocol tests should import only `eiprotocol`, standard library, and pytest.

Head tests should import only `eihead`, `eiprotocol`, standard library, and pytest unless they are explicitly marked integration.

- [ ] **Step 6: Run test suites**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eiprotocol/tests -q
rtk pytest D:/github/ei-workspace/repos/eihead/tests -q
rtk pytest D:/github/ei-workspace/repos/eibrain/tests -q
```

Expected: pass.

- [ ] **Step 7: Commit moved tests in each repository**

Run:

```powershell
rtk git -C D:/github/ei-workspace/repos/eiprotocol add tests
rtk git -C D:/github/ei-workspace/repos/eiprotocol commit -m "test: own protocol suite"

rtk git -C D:/github/ei-workspace/repos/eihead add tests
rtk git -C D:/github/ei-workspace/repos/eihead commit -m "test: own head runtime suite"

rtk git -C D:/github/ei-workspace/repos/eibrain add tests
rtk git -C D:/github/ei-workspace/repos/eibrain add -u tests
rtk git -C D:/github/ei-workspace/repos/eibrain commit -m "test: keep only brain-owned tests"
```

Expected: each commit succeeds when that repo has changes.

---

### Task 7: Retire Export Scripts As Sources Of Truth

**Files:**
- Modify or delete: `D:\github\ei-workspace\repos\eibrain\scripts\export-eiprotocol-repo.py`
- Modify or delete: `D:\github\ei-workspace\repos\eibrain\scripts\export-eihead-repo.py`
- Modify: `D:\github\ei-workspace\repos\eibrain\tests\infra\test_export_eiprotocol_repo.py`
- Modify: `D:\github\ei-workspace\repos\eibrain\tests\infra\test_export_eihead_repo.py`
- Modify: `D:\github\ei-workspace\repos\eibrain\README.md`

- [ ] **Step 1: Decide export script fate by source ownership**

If standalone repos are now canonical, export scripts should not generate standalone packages from `eibrain`.

Use this source-of-truth rule:

```text
delete the export script and its tests unless an external caller still invokes it
if an external caller still invokes it, keep only a non-generating retirement message
```

- [ ] **Step 2: If keeping a temporary script, make it non-generating**

The script should exit with:

```python
raise SystemExit(
    "This exporter is retired. Use D:/github/ei-workspace/repos/eiprotocol as the canonical repository."
)
```

or:

```python
raise SystemExit(
    "This exporter is retired. Use D:/github/ei-workspace/repos/eihead as the canonical repository."
)
```

- [ ] **Step 3: Update or delete export tests**

Delete tests that assert generated embedded packages. Keep only tests that assert the retirement message if the script remains.

- [ ] **Step 4: Run infra tests**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eibrain/tests/infra -q
```

Expected: pass.

- [ ] **Step 5: Commit export retirement**

Run:

```powershell
rtk git -C D:/github/ei-workspace/repos/eibrain add scripts tests/infra README.md
rtk git -C D:/github/ei-workspace/repos/eibrain add -u scripts tests/infra
rtk git -C D:/github/ei-workspace/repos/eibrain commit -m "chore: retire ei package exporters"
```

Expected: commit succeeds.

---

### Task 8: Workspace-Level Verification

**Files:**
- Verify: `D:\github\ei-workspace\.gitmodules`
- Verify: `D:\github\ei-workspace\repos\eiprotocol`
- Verify: `D:\github\ei-workspace\repos\eihead`
- Verify: `D:\github\ei-workspace\repos\eibrain`
- Verify: `D:\github\ei-workspace\.code-review-graph`

- [ ] **Step 1: Reinstall in dependency order**

Run:

```powershell
python -m pip install -e D:/github/ei-workspace/repos/eiprotocol
python -m pip install -e D:/github/ei-workspace/repos/eihead
python -m pip install -e D:/github/ei-workspace/repos/eibrain
```

Expected: installs succeed without local embedded package fallback.

- [ ] **Step 2: Run all local package tests**

Run:

```powershell
rtk pytest D:/github/ei-workspace/repos/eiprotocol/tests -q
rtk pytest D:/github/ei-workspace/repos/eihead/tests -q
rtk pytest D:/github/ei-workspace/repos/eibrain/tests -q
```

Expected: pass.

- [ ] **Step 3: Verify import origins**

Run:

```powershell
@'
import importlib.util
for name in ("eiprotocol", "eihead", "eibrain"):
    spec = importlib.util.find_spec(name)
    print(name, spec.origin if spec else None)
'@ | python -
```

Expected:

```text
eiprotocol D:\github\ei-workspace\repos\eiprotocol\eiprotocol\__init__.py
eihead D:\github\ei-workspace\repos\eihead\eihead\__init__.py
eibrain D:\github\ei-workspace\repos\eibrain\eibrain\__init__.py
```

- [ ] **Step 4: Rebuild unified graph**

Run from `D:\github\ei-workspace`:

```powershell
$env:CRG_RECURSE_SUBMODULES='true'
rtk code-review-graph build
rtk code-review-graph status
```

Expected: graph parses all three child repos and reports non-zero nodes/edges.

- [ ] **Step 5: Commit parent submodule pointers**

Run:

```powershell
rtk git -C D:/github/ei-workspace status --short
rtk git -C D:/github/ei-workspace add repos/eiprotocol repos/eihead repos/eibrain
rtk git -C D:/github/ei-workspace commit -m "chore: advance ei split submodules"
```

Expected: parent commit records updated submodule SHAs.

## Final Acceptance

- `eiprotocol` has no imports from `apps`, `eibrain`, `eihead`, or `eimemory`.
- `eihead` has no embedded `eiprotocol/`, no embedded `eibrain/`, and no packaged `apps.body_runtime`.
- `eibrain` has no embedded `eiprotocol/`, no embedded `eihead/`, and no `eihead-runtime` scripts.
- Protocol-only tests live in `eiprotocol`.
- Head-only tests live in `eihead`.
- Brain/orchestration tests live in `eibrain`.
- honjia hardware readiness remains blocked until real hardware validation.
- The parent workspace remains the integration and graph-reasoning surface.
