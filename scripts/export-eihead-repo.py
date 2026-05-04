#!/usr/bin/env python3
"""Export the eihead standalone repository layout from the eibrain tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
COPY_DIRS = (
    "eihead",
    "apps/head_runtime",
    # Transitional body runtime until honjia hardware code is fully renamed.
    "apps/body_runtime",
    "eibrain/body",
    "eibrain/infra",
    # Temporary compatibility until eiprotocol is split into its own repo.
    "eibrain/protocol",
)
OPTIONAL_FILES = (
    "apps/__init__.py",
    "eibrain/__init__.py",
)
COPY_GLOBS = (
    "config/eibrain.honjia.yaml",
    "config/eihead*.yaml",
    "deploy/systemd/eihead-*.service",
    "docs/eihead-*.md",
)
MANIFEST_FILENAME = "EXPORT_MANIFEST.json"
EXPECTED_HONXIN_PATH = "/dev-project/eihead"
RUNTIME_PATH = "/opt/eihead/current"
SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

TRANSITIONAL_PACKAGES = (
    {
        "package": "apps.body_runtime",
        "paths": ["apps/body_runtime"],
        "reason": "Temporary honjia body runtime compatibility during eihead split.",
    },
    {
        "package": "eibrain.body",
        "paths": ["eibrain/body"],
        "reason": "Temporary honjia hardware/runtime implementation before native eihead modules replace it.",
    },
    {
        "package": "eibrain.infra",
        "paths": ["eibrain/infra"],
        "reason": "Shared config helpers kept until eihead owns its deployment config layer.",
    },
    {
        "package": "eibrain.protocol",
        "paths": ["eibrain/protocol"],
        "reason": "Temporary protocol compatibility until eiprotocol is split into its own repo.",
    },
)

RUNTIME_ENTRYPOINTS = (
    {
        "name": "eihead-runtime-http",
        "kind": "systemd-service",
        "service": "deploy/systemd/eihead-runtime.service",
        "console_script": "eihead-runtime",
        "module": "eihead.runtime.cli:main",
        "command": "eihead-runtime --config /etc/eihead/eihead.honjia.yaml http --host 0.0.0.0 --port 18081",
        "host": "0.0.0.0",
        "port": 18081,
    },
    {
        "name": "eihead-monitor",
        "kind": "systemd-service",
        "service": "deploy/systemd/eihead-monitor.service",
        "console_script": "eihead-runtime",
        "module": "eihead.runtime.cli:main",
        "command": "eihead-runtime --config /etc/eihead/eihead.honjia.yaml monitor --host 0.0.0.0 --port 18080",
        "host": "0.0.0.0",
        "port": 18080,
    },
    {
        "name": "apps.head_runtime",
        "kind": "python-module",
        "module": "apps.head_runtime.__main__:main",
        "command": "python -m apps.head_runtime",
        "compatibility": True,
    },
)


PYPROJECT_TEMPLATE = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "eihead"
version = "0.1.0"
description = "Standalone head node runtime for the ei-hongtu system"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["PyYAML>=6.0"]

[project.scripts]
eihead-runtime = "eihead.runtime.cli:main"

[tool.setuptools.packages.find]
include = [
    "eihead*",
    "apps",
    "apps.head_runtime*",
    "apps.body_runtime*",
    "eibrain",
    "eibrain.body*",
    "eibrain.infra*",
    "eibrain.protocol*",
]
exclude = ["config*", "tests*", "docs*", "scripts*", "deploy*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""


README_TEMPLATE = """# eihead

Standalone head-node export for the ei-hongtu project.

This repository is generated from the eibrain monorepo by
`scripts/export-eihead-repo.py`. It contains the honjia-facing head runtime,
the `apps.head_runtime` compatibility entrypoint, a transitional copy of the
old body runtime, eihead systemd templates, and eihead migration/deployment
docs.

## Expected sync target

- Source of truth on honxin: `/dev-project/eihead`
- Runtime deployment path on honjia: `/opt/eihead/current`
- Runtime API: `eihead-runtime http --host 0.0.0.0 --port 18081`
- Native Web monitor: `eihead-runtime monitor --host 0.0.0.0 --port 18080`

## Local commands

```bash
python -m pip install -e .
eihead-runtime status
eihead-runtime http --host 0.0.0.0 --port 18081
eihead-runtime monitor --host 0.0.0.0 --port 18080
```

The current runtime still carries a small `eibrain.protocol` compatibility
subset plus the transitional `eibrain.body` hardware code. Split that subset
into `eiprotocol` and rename the body runtime into native `eihead` modules once
the shared protocol repo is ready.
"""


PROTOCOL_INIT_TEMPLATE = '''"""Minimal eibrain.protocol compatibility exports for transitional eihead."""

from .actions import MoveHeadAction, PlaySpeechAction, StopSpeechAction
from .observations import AudioTranscriptFinal
from .outcomes import ActionExecuted, SpeechPlaybackCompleted

__all__ = [
    "ActionExecuted",
    "AudioTranscriptFinal",
    "MoveHeadAction",
    "PlaySpeechAction",
    "SpeechPlaybackCompleted",
    "StopSpeechAction",
]
'''


@dataclass(frozen=True)
class ExportResult:
    """Summary for a completed eihead export."""

    target: Path
    copied: tuple[str, ...]
    generated: tuple[str, ...]
    force: bool = False
    manifest: str = MANIFEST_FILENAME

    def to_dict(self) -> dict[str, object]:
        return {
            "target": str(self.target),
            "copied": list(self.copied),
            "generated": list(self.generated),
            "force": self.force,
            "manifest": self.manifest,
        }


def export_eihead_repo(
    target_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> ExportResult:
    """Export eihead standalone files into ``target_dir``.

    The source repository is only read. Existing targets are rejected unless
    ``force`` is true; forced cleanup is blocked for repo roots, repo parents,
    source-tree children, filesystem roots, and common broad directories.
    """

    source_root = _resolve_repo_root(repo_root)
    target = _resolve_target(target_dir)

    if target.exists():
        if not force:
            raise FileExistsError(
                f"target already exists: {target}. Re-run with --force to replace it."
            )
        _validate_clean_target(target, source_root)
        if not target.is_dir():
            raise NotADirectoryError(f"target exists but is not a directory: {target}")
        shutil.rmtree(target)
    elif force:
        _validate_clean_target(target, source_root)

    target.mkdir(parents=True, exist_ok=False)

    copied: list[str] = []
    for rel_dir in COPY_DIRS:
        copied.extend(_copy_directory(source_root, target, rel_dir))
    for rel_file in OPTIONAL_FILES:
        source_file = source_root / rel_file
        if source_file.exists():
            copied.append(_copy_file(source_file, target / rel_file, target_root=target))
    for pattern in COPY_GLOBS:
        copied.extend(_copy_glob(source_root, target, pattern))

    generated = list(_write_templates(target))
    generated.append(
        _write_export_manifest(
            target_root=target,
            source_root=source_root,
            copied=tuple(sorted(copied)),
            generated=tuple(generated),
        )
    )
    return ExportResult(
        target=target,
        copied=tuple(sorted(copied)),
        generated=tuple(generated),
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the standalone eihead repository layout."
    )
    parser.add_argument("target", help="Destination directory, for example /dev-project/eihead")
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help="Source eibrain repository root. Defaults to this script's parent repo.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing target directory after path validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = export_eihead_repo(args.target, repo_root=args.repo_root, force=args.force)
    except Exception as exc:  # pragma: no cover - exercised by CLI callers.
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    root = Path(repo_root or DEFAULT_REPO_ROOT).expanduser().resolve(strict=True)
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"repo root does not contain pyproject.toml: {root}")
    return root


def _resolve_target(target_dir: str | Path) -> Path:
    return Path(target_dir).expanduser().resolve(strict=False)


def _validate_clean_target(target: Path, repo_root: Path) -> None:
    target = target.resolve(strict=False)
    repo_root = repo_root.resolve(strict=True)

    if _same_path(target, repo_root):
        raise ValueError("refusing to clean the source repo root")
    if _path_contains(repo_root, target):
        raise ValueError("refusing to clean a directory inside the source repo")
    if _path_contains(target, repo_root):
        raise ValueError("refusing to clean an ancestor of the source repo")
    if target.parent == target:
        raise ValueError("refusing to clean a filesystem root")

    home = Path.home().resolve(strict=False)
    if _same_path(target, home) or _path_contains(target, home):
        raise ValueError("refusing to clean the user home directory or its ancestors")

    if len(target.parts) <= 2:
        raise ValueError(f"refusing to clean a broad path: {target}")

    broad_posix_paths = {
        Path("/dev"),
        Path("/dev-project"),
        Path("/dev-projiect"),
        Path("/etc"),
        Path("/home"),
        Path("/opt"),
        Path("/tmp"),
        Path("/usr"),
        Path("/var"),
    }
    for broad_path in broad_posix_paths:
        if _same_path(target, broad_path):
            raise ValueError(f"refusing to clean broad path: {target}")


def _copy_directory(source_root: Path, target_root: Path, rel_dir: str) -> list[str]:
    source_dir = source_root / rel_dir
    if not source_dir.is_dir():
        raise FileNotFoundError(f"required source directory is missing: {source_dir}")

    copied: list[str] = []
    for source_path in sorted(source_dir.rglob("*")):
        if _should_skip(source_path, source_dir):
            continue
        if not source_path.is_file():
            continue
        rel_path = source_path.relative_to(source_root)
        copied.append(_copy_file(source_path, target_root / rel_path, target_root=target_root))
    return copied


def _copy_glob(source_root: Path, target_root: Path, pattern: str) -> list[str]:
    matches = sorted(path for path in source_root.glob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"required source pattern matched no files: {pattern}")
    return [
        _copy_file(path, target_root / path.relative_to(source_root), target_root=target_root)
        for path in matches
    ]


def _copy_file(source_path: Path, target_path: Path, *, target_root: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return _relative_output_path(target_path, target_root=target_root)


def _write_templates(target_root: Path) -> tuple[str, ...]:
    templates = {
        "pyproject.toml": PYPROJECT_TEMPLATE,
        "README.md": README_TEMPLATE,
        "eibrain/protocol/__init__.py": PROTOCOL_INIT_TEMPLATE,
    }
    generated: list[str] = []
    for rel_path, text in templates.items():
        path = target_root / rel_path
        path.write_text(text, encoding="utf-8", newline="\n")
        generated.append(rel_path)
    return tuple(generated)


def _write_export_manifest(
    *,
    target_root: Path,
    source_root: Path,
    copied: tuple[str, ...],
    generated: tuple[str, ...],
) -> str:
    generated_with_manifest = (*generated, MANIFEST_FILENAME)
    source_state = _read_source_git_state(source_root)
    manifest = {
        "schema_version": 1,
        "source": {
            "repository": "eibrain",
            "repo_root": str(source_root),
            **source_state,
        },
        "standalone_repo": {
            "name": "eihead",
            "expected_honxin_path": EXPECTED_HONXIN_PATH,
            "runtime_path": RUNTIME_PATH,
            "manifest_path": MANIFEST_FILENAME,
        },
        "transitional_packages": list(TRANSITIONAL_PACKAGES),
        "runtime_entrypoints": list(RUNTIME_ENTRYPOINTS),
        "exported_paths": {
            "copied": list(copied),
            "generated": list(generated_with_manifest),
        },
    }
    path = target_root / MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return MANIFEST_FILENAME


def _read_source_git_state(source_root: Path) -> dict[str, object]:
    commit = _run_git(source_root, "rev-parse", "HEAD")
    status_short = _run_git(source_root, "status", "--short").splitlines()
    return {
        "commit": commit or "unknown",
        "dirty": bool(status_short),
        "status_short": status_short,
    }


def _run_git(source_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _relative_output_path(target_path: Path, *, target_root: Path) -> str:
    # Callers only need stable manifest paths, not absolute local machine paths.
    return target_path.relative_to(target_root).as_posix()


def _should_skip(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    if any(part in SKIP_DIR_NAMES for part in rel_parts):
        return True
    return path.suffix in SKIP_SUFFIXES


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return not _same_path(parent, child)
    except ValueError:
        return False


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


if __name__ == "__main__":
    raise SystemExit(main())
