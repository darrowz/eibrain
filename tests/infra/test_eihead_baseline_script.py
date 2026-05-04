from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "capture-eihead-baseline.sh"
CHECKLIST_PATH = REPO_ROOT / "docs" / "eihead-cutover-checklist.md"


def _script_text() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_baseline_script_exists_and_uses_safe_shell_options() -> None:
    text = _script_text()

    assert text.startswith("#!/usr/bin/env bash\n")
    assert re.search(r"^set -eu\b", text, re.MULTILINE)
    assert "set -o pipefail" in text


def test_baseline_script_contains_required_read_only_probes() -> None:
    text = _script_text()
    required_tokens = [
        "hostname",
        "date -Is",
        "git -C",
        "rev-parse --short HEAD",
        "systemctl --user status",
        "systemctl status",
        "ss -ltnup",
        "netstat -ltnup",
        "/dev/video0",
        "/dev/hailo0",
        "/dev/i2c-1",
        "/dev/snd",
        "curl -fsS --max-time 3",
        "http://127.0.0.1:18080",
        "/health",
        "/status",
        "/api/health",
        "/api/status",
        "journalctl --user -u",
        "journalctl -u",
    ]

    for token in required_tokens:
        assert token in text


def test_baseline_script_gracefully_degrades_when_commands_are_missing() -> None:
    text = _script_text()

    assert "have()" in text
    assert "command -v" in text
    assert "missing command" in text
    assert "non-fatal" in text
    assert "|| echo" in text


def test_baseline_script_has_no_destructive_commands() -> None:
    text = _script_text()
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

    for pattern in forbidden_patterns:
        assert not re.search(pattern, text)


def test_cutover_checklist_covers_all_phases_and_deployment_paths() -> None:
    text = CHECKLIST_PATH.read_text(encoding="utf-8")

    for phase in range(8):
        assert f"Phase {phase}" in text

    required_paths = [
        "/dev-project/eibrain",
        "/dev-project/eihead",
        "/dev-project/eiprotocol",
        "/home/darrow/eibrain",
        "/opt/eihead/current",
        "/var/lib/eihead",
        "/etc/eihead",
    ]
    for path in required_paths:
        assert path in text
