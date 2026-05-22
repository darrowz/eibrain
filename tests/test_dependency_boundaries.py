from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

BODY_RUNTIME_IMPORT_PREFIXES = ("apps.body_runtime", "eibrain.body")
ALLOWED_OPERATOR_CONSOLE_BODY_IMPORTS = {
    ("apps/operator_console/__main__.py", "apps.body_runtime.app", "BodyRuntimeApp"),
    ("apps/operator_console/__main__.py", "apps.body_runtime.engagement_state", "DEFAULT_ENGAGEMENT_STATE_PATH"),
    ("apps/operator_console/__main__.py", "apps.body_runtime.engagement_state", "EngagementStateReader"),
    ("apps/operator_console/__main__.py", "apps.body_runtime.engagement_state", "EngagementStateWriter"),
    ("apps/operator_console/__main__.py", "apps.body_runtime.visual_tracking_loop", "VisualTrackingLoop"),
    ("apps/operator_console/__main__.py", "apps.body_runtime.voice_dialogue_loop", "VoiceDialogueLoop"),
    ("apps/operator_console/__main__.py", "eibrain.body.realtime_audio", "ArecordRawChunkSource"),
    ("apps/operator_console/__main__.py", "eibrain.body.realtime_audio", "PcmRingBuffer"),
    ("apps/operator_console/__main__.py", "eibrain.body.realtime_audio", "RealtimeAudioCaptureWorker"),
    ("apps/operator_console/__main__.py", "eibrain.body.realtime_audio", "RealtimeWakeAudioPipeline"),
    ("apps/operator_console/__main__.py", "eibrain.body.realtime_audio", "RealtimeWakeDetector"),
    ("apps/operator_console/app.py", "apps.body_runtime.voice_provider_smoke", "build_voice_provider_smoke_report"),
}


def _resolve_spec_paths(spec: object) -> list[Path]:
    origin = getattr(spec, "origin", None)
    search_locations = getattr(spec, "submodule_search_locations", None)

    paths: list[Path] = []
    if isinstance(origin, str) and origin not in {"built-in", "frozen"}:
        paths.append(Path(origin).resolve(strict=False))
    if search_locations:
        for location in search_locations:
            paths.append(Path(location).resolve(strict=False))
    return paths


def _is_inside_repo(path: Path) -> bool:
    try:
        path.relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def _find_spec_outside_repo(package_name: str) -> object:
    original_sys_path = list(sys.path)
    filtered_sys_path = [
        path
        for path in original_sys_path
        if not _is_inside_repo(Path(path).resolve(strict=False))
    ]
    try:
        sys.path[:] = filtered_sys_path
        return importlib.util.find_spec(package_name)
    finally:
        sys.path[:] = original_sys_path


def _is_body_runtime_import(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in BODY_RUNTIME_IMPORT_PREFIXES
    )


def _collect_body_runtime_imports(root: Path) -> set[tuple[str, str, str]]:
    imports: set[tuple[str, str, str]] = set()
    for path in root.rglob("*.py"):
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_body_runtime_import(alias.name):
                        imports.add((rel_path, alias.name, alias.asname or alias.name.rsplit(".", 1)[-1]))
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if not _is_body_runtime_import(module_name):
                    continue
                for alias in node.names:
                    imports.add((rel_path, module_name, alias.asname or alias.name))
    return imports


def test_eibrain_does_not_vendor_eihead_or_eiprotocol() -> None:
    assert not (REPO_ROOT / "eihead").exists()
    assert not (REPO_ROOT / "eiprotocol").exists()


def test_eibrain_uses_standalone_eihead_and_eiprotocol_when_imported() -> None:
    for package_name in ("eihead", "eiprotocol"):
        spec = _find_spec_outside_repo(package_name)
        assert spec is not None, f"Expected importlib to resolve {package_name}"

        resolved_paths = _resolve_spec_paths(spec)
        assert resolved_paths, f"Expected resolved module path(s) for {package_name}"
        assert all(
            not _is_inside_repo(resolved_path) for resolved_path in resolved_paths
        ), f"{package_name} unexpectedly resolved inside eibrain repo: {resolved_paths}"


def test_body_runtime_packages_are_marked_as_compatibility_shims() -> None:
    import apps.body_runtime as body_runtime
    import eibrain.body as body_package

    for module in (body_runtime, body_package):
        assert module.COMPATIBILITY_SHIM is True
        assert module.DEPRECATED_RUNTIME_OWNER == "eihead"
        assert "eihead" in module.DEPRECATION_REASON


def test_cognitive_runtime_and_operator_console_do_not_expand_body_hot_path_imports() -> None:
    assert _collect_body_runtime_imports(REPO_ROOT / "apps" / "cognitive_runtime") == set()
    assert _collect_body_runtime_imports(REPO_ROOT / "apps" / "operator_console") == ALLOWED_OPERATOR_CONSOLE_BODY_IMPORTS
