#!/usr/bin/env python3
"""Export the eiprotocol standalone repository layout from the eibrain tree."""

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
COPY_DIRS = ("eiprotocol",)
COPY_FILES = ("tests/protocol/test_eiprotocol_mvp.py",)
SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

PYPROJECT_TEMPLATE = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "eiprotocol"
version = "0.1.0"
description = "Shared eiprotocol v0.1 MVP contracts for EI projects"
readme = "README.md"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
include = ["eiprotocol*"]
exclude = ["tests*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

README_TEMPLATE = """# eiprotocol

Standalone shared protocol contracts for EI projects.

This repository is generated from the eibrain monorepo by
`scripts/export-eiprotocol-repo.py`.

## Source

- Source repository: eibrain
- Source commit: {source_commit}
- Source dirty: {source_dirty}

## Local commands

```bash
python -m pip install -e .
python -m pytest -q
```

The package currently contains the v0.1 MVP event envelope and JSON-friendly
models used by `eihead`, `eibrain`, and future EI service repositories.
"""

GITIGNORE_TEMPLATE = """__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
"""


@dataclass(frozen=True)
class ExportResult:
    """Summary for a completed eiprotocol export."""

    target: Path
    copied: tuple[str, ...]
    generated: tuple[str, ...]
    force: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "target": str(self.target),
            "copied": list(self.copied),
            "generated": list(self.generated),
            "force": self.force,
        }


def export_eiprotocol_repo(
    target_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> ExportResult:
    """Export eiprotocol standalone files into ``target_dir``."""

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
    for rel_file in COPY_FILES:
        copied.append(_copy_required_file(source_root, target, rel_file))

    source_state = _read_source_git_state(source_root)
    generated = _write_templates(target, source_state=source_state)
    return ExportResult(
        target=target,
        copied=tuple(sorted(copied)),
        generated=generated,
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the standalone eiprotocol repository layout."
    )
    parser.add_argument(
        "target",
        help="Destination directory, for example /dev-project/eiprotocol",
    )
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
        result = export_eiprotocol_repo(args.target, repo_root=args.repo_root, force=args.force)
    except Exception as exc:  # pragma: no cover - exercised by CLI callers.
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    root = Path(repo_root or DEFAULT_REPO_ROOT).expanduser().resolve(strict=True)
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"repo root does not contain pyproject.toml: {root}")
    if not (root / "eiprotocol").is_dir():
        raise FileNotFoundError(f"repo root does not contain eiprotocol/: {root}")
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


def _copy_required_file(source_root: Path, target_root: Path, rel_file: str) -> str:
    source_file = source_root / rel_file
    if not source_file.is_file():
        raise FileNotFoundError(f"required source file is missing: {source_file}")
    return _copy_file(source_file, target_root / rel_file, target_root=target_root)


def _copy_file(source_path: Path, target_path: Path, *, target_root: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return _relative_output_path(target_path, target_root=target_root)


def _write_templates(target_root: Path, *, source_state: dict[str, object]) -> tuple[str, ...]:
    templates = {
        "pyproject.toml": PYPROJECT_TEMPLATE,
        "README.md": README_TEMPLATE.format(
            source_commit=source_state["commit"],
            source_dirty=str(source_state["dirty"]).lower(),
        ),
        ".gitignore": GITIGNORE_TEMPLATE,
    }
    generated: list[str] = []
    for rel_path, text in templates.items():
        path = target_root / rel_path
        path.write_text(text, encoding="utf-8", newline="\n")
        generated.append(rel_path)
    return tuple(generated)


def _read_source_git_state(source_root: Path) -> dict[str, object]:
    commit = _run_git(source_root, "rev-parse", "HEAD")
    status_short = _run_git(source_root, "status", "--short").splitlines()
    return {
        "commit": commit or "unknown",
        "dirty": bool(status_short),
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
