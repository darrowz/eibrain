from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export-eihead-repo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_eihead_repo", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_export_script_is_retired() -> None:
    module = _load_module()

    with pytest.raises(SystemExit) as exc_info:
        module.main([])

    message = str(exc_info.value)
    assert "retired" in message.lower()
    assert "D:/github/ei-workspace/repos/eihead" in message
