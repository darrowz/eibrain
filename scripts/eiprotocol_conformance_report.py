from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RELATIVE_DIR = Path("tests") / "fixtures" / "eiprotocol"
WORKSPACE_PROTOCOL_FIXTURE_DIR = Path("..") / "eiprotocol" / "tests" / "fixtures" / "eiprotocol"
RUNTIME_IMPORT_ROOTS = {"eibrain", "eihead"}
ROUND_SCOPED_TYPES = {"dialogue", "action", "memory", "outcome", "training"}


def build_report(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    _ensure_repo_importable(repo_root)

    from eiprotocol.catalog import get_event_definition, list_event_names
    from eiprotocol.event_routing import classify_event
    from eiprotocol.validation import validate_event_strict

    catalog_events = sorted(list_event_names())
    catalog_event_set = set(catalog_events)
    fixture_dir = _fixture_dir(repo_root)
    fixtures = load_fixture_payloads(fixture_dir)

    fixture_event_names = {
        str(payload.get("name"))
        for _fixture_name, payload in fixtures
        if isinstance(payload.get("name"), str) and payload.get("name")
    }
    missing_fixtures = sorted(catalog_event_set - fixture_event_names)
    extra_fixtures = [
        {"fixture": fixture_name, "name": _fixture_event_name(payload)}
        for fixture_name, payload in fixtures
        if _fixture_event_name(payload) not in catalog_event_set
    ]

    routing_missing: list[dict[str, str]] = []
    for event_name in catalog_events:
        definition = get_event_definition(event_name)
        if definition is None:
            routing_missing.append({"event_name": event_name, "reason": "missing_catalog_definition"})
            continue

        route = classify_event(_minimal_event(event_name, definition))
        if route.get("status") != "routed":
            routing_missing.append(
                {
                    "event_name": event_name,
                    "reason": str(route.get("reason") or route.get("status") or "unknown"),
                }
            )

    strict_validation_failures: list[dict[str, Any]] = []
    for fixture_name, payload in fixtures:
        if _fixture_event_name(payload) not in catalog_event_set:
            continue
        issues = validate_event_strict(payload, known_event_required=True)
        if issues:
            strict_validation_failures.append(
                {
                    "fixture": fixture_name,
                    "name": _fixture_event_name(payload),
                    "errors": [issue.to_dict() for issue in issues],
                }
            )

    dependency_violations = scan_dependency_violations(_protocol_package_dir(repo_root), repo_root)

    status = "fail" if any(
        (
            missing_fixtures,
            extra_fixtures,
            routing_missing,
            strict_validation_failures,
            dependency_violations,
        )
    ) else "pass"

    return {
        "catalog_event_count": len(catalog_events),
        "fixture_count": len(fixtures),
        "missing_fixtures": missing_fixtures,
        "extra_fixtures": extra_fixtures,
        "routing_missing": routing_missing,
        "strict_validation_failures": strict_validation_failures,
        "dependency_violations": dependency_violations,
        "status": status,
    }


def load_fixture_payloads(fixture_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    fixtures: list[tuple[str, dict[str, Any]]] = []
    if not fixture_dir.exists():
        return fixtures

    for fixture_path in sorted(fixture_dir.glob("*.json")):
        with fixture_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        fixtures.append((fixture_path.name, payload if isinstance(payload, dict) else {}))
    return fixtures


def _fixture_dir(repo_root: Path) -> Path:
    local_dir = repo_root / FIXTURE_RELATIVE_DIR
    if local_dir.exists():
        return local_dir
    return (repo_root / WORKSPACE_PROTOCOL_FIXTURE_DIR).resolve()


def _protocol_package_dir(repo_root: Path) -> Path:
    local_package = repo_root / "eiprotocol"
    if local_package.exists():
        return local_package
    return (repo_root / ".." / "eiprotocol" / "eiprotocol").resolve()


def scan_dependency_violations(package_dir: Path, repo_root: Path) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    if not package_dir.exists():
        return violations

    for source_path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_runtime_import(alias.name):
                        violations.append(_dependency_violation(repo_root, source_path, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if _is_runtime_import(node.module):
                    violations.append(_dependency_violation(repo_root, source_path, node.module))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report eiprotocol v0.1.1 conformance coverage.")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--strict", action="store_true", help="exit non-zero for blocking conformance issues")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    report = build_report(args.repo_root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text_report(report)

    if args.strict and _has_strict_failures(report):
        return 1
    return 0


def _ensure_repo_importable(repo_root: Path) -> None:
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)


def _fixture_event_name(payload: Mapping[str, Any]) -> str:
    value = payload.get("name")
    return value if isinstance(value, str) else ""


def _minimal_event(event_name: str, definition: Any) -> dict[str, Any]:
    round_id = "rnd_conformance_001" if definition.round_scoped or definition.event_type in ROUND_SCOPED_TYPES else ""
    return {
        "specVersion": "eiprotocol/0.1",
        "id": f"evt_{event_name.replace('ei.', '').replace('.', '_')}",
        "type": definition.event_type,
        "name": event_name,
        "time": "2026-05-04T10:30:00+08:00",
        "sequence": 1,
        "requestId": "req_conformance_001",
        "sessionId": "ses_conformance_001",
        "roundId": round_id,
        "correlationId": "",
        "causationId": "",
        "traceId": "trc_conformance_001",
        "source": {"domain": "eiprotocol", "instanceId": "conformance", "metadata": {}},
        "target": {"domain": "eiprotocol", "instanceId": "conformance", "metadata": {}},
        "priority": "normal",
        "ttlMs": None,
        "mode": {},
        "content": {
            field_name: _minimal_content_value(field_name)
            for field_name in definition.required_content_fields
        },
        "policy": {
            "decision": "not_required",
            "riskLevel": "L0",
            "decisionId": "",
            "requiredAck": False,
            "reason": "",
            "expiresAt": "",
            "extensions": {},
        },
        "extensions": {},
    }


def _minimal_content_value(field_name: str) -> Any:
    if field_name in {"capabilities", "results", "speechSegments", "actionSegments"}:
        return []
    if field_name in {"proposal", "target", "components", "scope"}:
        return {}
    if field_name in {"final", "success", "stable", "shouldEmit"}:
        return True
    if field_name in {"chunkIndex", "limit", "progress", "resultCount"}:
        return 1
    if field_name == "riskLevel":
        return "L1"
    if field_name == "decision":
        return "allow"
    if field_name == "status":
        return "ok"
    if field_name in {"sentAt", "reportedAt", "observedAt"}:
        return "2026-05-04T10:30:00+08:00"
    return f"{field_name}_conformance"


def _is_runtime_import(module_name: str) -> bool:
    root = module_name.split(".", 1)[0]
    return root in RUNTIME_IMPORT_ROOTS


def _dependency_violation(repo_root: Path, source_path: Path, module_name: str) -> dict[str, str]:
    return {
        "file": source_path.relative_to(repo_root).as_posix(),
        "module": module_name,
    }


def _has_strict_failures(report: Mapping[str, Any]) -> bool:
    return any(
        bool(report.get(key))
        for key in (
            "missing_fixtures",
            "extra_fixtures",
            "routing_missing",
            "strict_validation_failures",
            "dependency_violations",
        )
    )


def _print_text_report(report: Mapping[str, Any]) -> None:
    print(f"status: {report['status']}")
    print(f"catalog_event_count: {report['catalog_event_count']}")
    print(f"fixture_count: {report['fixture_count']}")
    for key in (
        "missing_fixtures",
        "extra_fixtures",
        "routing_missing",
        "strict_validation_failures",
        "dependency_violations",
    ):
        value = report[key]
        if value:
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: []")


if __name__ == "__main__":
    raise SystemExit(main())
