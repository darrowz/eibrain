from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from eiprotocol.catalog import list_event_names


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "eiprotocol_conformance_report.py"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "eiprotocol"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("eiprotocol_conformance_report", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_report_uses_dynamic_catalog_and_current_fixtures() -> None:
    module = _load_report_module()

    report = module.build_report(REPO_ROOT)
    expected_extra_fixtures = _extra_fixture_entries()

    assert report["catalog_event_count"] == len(list_event_names())
    assert report["fixture_count"] == len(list(FIXTURE_DIR.glob("*.json")))
    assert report["missing_fixtures"] == []
    assert report["extra_fixtures"] == expected_extra_fixtures
    assert report["routing_missing"] == []
    assert report["strict_validation_failures"] == []
    assert report["dependency_violations"] == []
    assert report["status"] == "pass"


def test_json_cli_prints_required_report_fields() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert {
        "catalog_event_count",
        "fixture_count",
        "missing_fixtures",
        "extra_fixtures",
        "routing_missing",
        "strict_validation_failures",
        "dependency_violations",
        "status",
    }.issubset(report)
    assert report["catalog_event_count"] == len(list_event_names())


def test_strict_cli_fails_when_report_has_blocking_violations(monkeypatch) -> None:
    module = _load_report_module()

    def fake_report(repo_root: Path) -> dict[str, object]:
        return {
            "catalog_event_count": 2,
            "fixture_count": 1,
            "missing_fixtures": ["ei.example.missing"],
            "extra_fixtures": [{"fixture": "future.json", "name": "ei.future.event"}],
            "routing_missing": [{"event_name": "ei.example.unrouted", "reason": "not_processed"}],
            "strict_validation_failures": [{"fixture": "bad.json", "errors": ["name is required"]}],
            "dependency_violations": [{"file": "eiprotocol/bad.py", "module": "eibrain"}],
            "status": "fail",
        }

    monkeypatch.setattr(module, "build_report", fake_report)

    assert module.main(["--strict"]) == 1


def test_strict_cli_passes_for_clean_current_report() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--strict"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_dependency_scan_reports_eiprotocol_imports_of_runtime_packages(tmp_path: Path) -> None:
    module = _load_report_module()
    package_dir = tmp_path / "eiprotocol"
    package_dir.mkdir()
    (package_dir / "safe.py").write_text("from .catalog import list_event_names\n", encoding="utf-8")
    (package_dir / "bad.py").write_text("import eibrain.foo\nfrom eihead import client\n", encoding="utf-8")

    violations = module.scan_dependency_violations(package_dir, tmp_path)

    assert violations == [
        {"file": "eiprotocol/bad.py", "module": "eibrain.foo"},
        {"file": "eiprotocol/bad.py", "module": "eihead"},
    ]


def _extra_fixture_entries() -> list[dict[str, str]]:
    catalog_events = set(list_event_names())
    entries: list[dict[str, str]] = []
    for fixture_path in sorted(FIXTURE_DIR.glob("*.json")):
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        name = payload.get("name", "")
        if name not in catalog_events:
            entries.append({"fixture": fixture_path.name, "name": name})
    return entries
