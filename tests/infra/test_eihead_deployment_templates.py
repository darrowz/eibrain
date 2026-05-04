from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
PLAN_PATH = REPO_ROOT / "docs" / "eihead-deployment-plan.md"


def _unit_text(name: str) -> str:
    return (SYSTEMD_DIR / name).read_text(encoding="utf-8")


def test_eihead_runtime_service_uses_target_paths_and_http_port() -> None:
    text = _unit_text("eihead-runtime.service")

    assert "User=darrow" in text
    assert "WorkingDirectory=/opt/eihead/current" in text
    assert "Environment=PYTHONUNBUFFERED=1" in text
    assert "Environment=EIHEAD_CONFIG=/etc/eihead/eihead.honjia.yaml" in text
    assert "Environment=EIHEAD_EXPORT_MANIFEST=/opt/eihead/current/EXPORT_MANIFEST.json" in text
    assert "EnvironmentFile=-/etc/eihead/eihead.env" in text
    assert "ExecStart=/opt/eihead/current/.venv/bin/eihead-runtime" in text
    assert "--config /etc/eihead/eihead.honjia.yaml" in text
    assert " http --host 0.0.0.0 --port 18081" in text
    assert "Restart=always" in text


def test_eihead_monitor_service_keeps_18080_compatibility_without_old_unit_edits() -> None:
    text = _unit_text("eihead-monitor.service")

    assert "User=darrow" in text
    assert "WorkingDirectory=/opt/eihead/current" in text
    assert "Environment=PYTHONUNBUFFERED=1" in text
    assert "Environment=EIHEAD_RUNTIME_URL=http://127.0.0.1:18081" in text
    assert "Environment=EIHEAD_EXPORT_MANIFEST=/opt/eihead/current/EXPORT_MANIFEST.json" in text
    assert "Environment=EIHEAD_MONITOR_PORT=18080" in text
    assert "EnvironmentFile=-/etc/eihead/eihead.env" in text
    assert "Requires=eihead-runtime.service" in text
    assert "ExecStart=/opt/eihead/current/.venv/bin/eihead-runtime" in text
    assert "--config /etc/eihead/eihead.honjia.yaml" in text
    assert " monitor --host 0.0.0.0 --port 18080" in text
    assert "apps.operator_console" not in text
    assert "Restart=always" in text
    assert "eibrain-monitor.service" not in text


def test_eihead_systemd_templates_have_no_destructive_commands() -> None:
    forbidden_patterns = [
        r"\brm\b",
        r"\bgit\s+reset\b",
        r"\bgit\s+checkout\b",
        r"\bmkfs\b",
        r"\bdd\b",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bpoweroff\b",
        r"\bhalt\b",
        r"\bkill\b",
        r"\bsystemctl\s+(?:--user\s+)?(?:start|stop|restart|enable|disable|mask|unmask)\b",
    ]
    for name in ("eihead-runtime.service", "eihead-monitor.service"):
        text = _unit_text(name)
        for pattern in forbidden_patterns:
            assert not re.search(pattern, text)


def test_eihead_deployment_plan_documents_downtime_cutover_paths_and_ports() -> None:
    text = PLAN_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "honxin `/dev-project` is the code source of truth",
        "It is not a runtime path",
        "honjia `/opt/eihead/current`",
        "honjia `/etc/eihead/eihead.honjia.yaml`",
        "honjia `18081`",
        "honjia `18080`",
        "monitoring.port: 18080",
        "eihead-native monitor",
        "short downtime cutover",
        "stop the old eibrain head-side",
        "eibrain-monitor.service",
        "eibrain-vision-hailo.service",
        "brain-runtime.service",
        "eihead-runtime.service",
        "eihead-monitor.service",
    ]
    for token in required_tokens:
        assert token in text
