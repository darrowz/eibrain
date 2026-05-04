from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export-eiprotocol-repo.py"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_eiprotocol_repo", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_export_creates_standalone_eiprotocol_layout(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eiprotocol-standalone"

    result = module.export_eiprotocol_repo(target, repo_root=REPO_ROOT)

    assert result.target == target.resolve()
    required_paths = [
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "eiprotocol/__init__.py",
        "eiprotocol/models.py",
        "tests/protocol/test_eiprotocol_event_routing.py",
        "tests/protocol/test_eiprotocol_fixtures.py",
        "tests/protocol/test_eiprotocol_mvp.py",
    ]
    for rel_path in required_paths:
        assert (target / rel_path).is_file(), rel_path

    assert "eiprotocol/__init__.py" in result.copied
    assert "eiprotocol/models.py" in result.copied
    assert "tests/protocol/test_eiprotocol_event_routing.py" in result.copied
    assert "tests/protocol/test_eiprotocol_fixtures.py" in result.copied
    assert "tests/protocol/test_eiprotocol_mvp.py" in result.copied
    assert result.generated == ("pyproject.toml", "README.md", ".gitignore")
    assert not (target / "eibrain").exists()
    assert not (target / "eiprotocol/__pycache__").exists()


def test_export_generates_package_metadata_and_readme_source_commit(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eiprotocol-standalone"

    module.export_eiprotocol_repo(target, repo_root=REPO_ROOT)

    pyproject = (target / "pyproject.toml").read_text(encoding="utf-8")
    readme = (target / "README.md").read_text(encoding="utf-8")
    gitignore = (target / ".gitignore").read_text(encoding="utf-8")

    assert 'name = "eiprotocol"' in pyproject
    assert 'include = ["eiprotocol*"]' in pyproject
    assert 'testpaths = ["tests"]' in pyproject
    assert 'name = "eibrain"' not in pyproject
    assert _source_commit() in readme
    assert "scripts/export-eiprotocol-repo.py" in readme
    assert "__pycache__/" in gitignore
    assert "*.py[cod]" in gitignore


def test_export_refuses_existing_target_without_force(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eiprotocol-standalone"
    target.mkdir()

    with pytest.raises(FileExistsError):
        module.export_eiprotocol_repo(target, repo_root=REPO_ROOT)


def test_export_force_replaces_existing_target(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eiprotocol-standalone"
    target.mkdir()
    stale_file = target / "old-file.txt"
    stale_file.write_text("old", encoding="utf-8")

    result = module.export_eiprotocol_repo(target, repo_root=REPO_ROOT, force=True)

    assert result.force is True
    assert not stale_file.exists()
    assert (target / "eiprotocol/models.py").is_file()


def test_exported_repo_imports_eiprotocol_from_target_and_round_trips_json(
    tmp_path: Path,
) -> None:
    module = _load_export_module()
    target = tmp_path / "eiprotocol-standalone"
    module.export_eiprotocol_repo(target, repo_root=REPO_ROOT)

    env = {**os.environ, "PYTHONPATH": str(target)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from eiprotocol import AudioTurn, EventEnvelope, SourceRef; "
                "event = AudioTurn(text='hello', language='en-US').to_event("
                "event_id='evt_1', request_id='req_1', session_id='ses_1', "
                "round_id='rnd_1', sequence=1, "
                "source=SourceRef(domain='eihead', instance_id='honjia'), "
                "time='2026-05-04T10:32:00+08:00'); "
                "payload = json.loads(event.to_json()); "
                "restored = EventEnvelope.from_dict(payload); "
                "assert restored.to_dict() == payload; "
                "assert restored.round_id == 'rnd_1'; "
                "assert payload['content']['text'] == 'hello'"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=target,
        env=env,
    )

    assert result.returncode == 0, result.stderr


def test_exported_repo_runs_mvp_fixture_and_event_routing_tests(tmp_path: Path) -> None:
    module = _load_export_module()
    target = tmp_path / "eiprotocol-standalone"
    module.export_eiprotocol_repo(target, repo_root=REPO_ROOT)

    env = {**os.environ, "PYTHONPATH": str(target)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/protocol/test_eiprotocol_mvp.py",
            "tests/protocol/test_eiprotocol_fixtures.py",
            "tests/protocol/test_eiprotocol_event_routing.py",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=target,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_prints_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_export_module()
    target = tmp_path / "eiprotocol-standalone"

    assert module.main([str(target), "--repo-root", str(REPO_ROOT)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["target"] == str(target.resolve())
    assert output["force"] is False
    assert "eiprotocol/models.py" in output["copied"]
    assert output["generated"] == ["pyproject.toml", "README.md", ".gitignore"]
