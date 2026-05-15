from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


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
