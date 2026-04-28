from __future__ import annotations

from pathlib import Path
import sys


def test_sync_honjia_cli_runs_expected_ssh_and_scp(monkeypatch, capsys) -> None:
    repo_source_root = Path(__file__).resolve().parents[2]
    if str(repo_source_root) not in sys.path:
        sys.path.insert(0, str(repo_source_root))
    from apps import sync_honjia

    calls: list[list[str]] = []

    def _fake_run(command: list[str]) -> None:
        calls.append(command)

    monkeypatch.setattr(sync_honjia, "_run", _fake_run)
    monkeypatch.setattr(sync_honjia, "_current_revision", lambda repo_root: "abc123")
    monkeypatch.setattr(
        "sys.argv",
        ["sync-honjia", "--target-host", "darrow@honjia", "--restart-services", "--include-tests"],
    )

    assert sync_honjia.main() == 0
    output = capsys.readouterr().out
    assert output.strip().startswith("synced=darrow@honjia:")
    assert "revision=abc123" in output
    assert calls[0][:2] == ["ssh", "darrow@honjia"]
    cleanup_calls = [
        command for command in calls if command[:2] == ["ssh", "darrow@honjia"] and "rm -rf --" in command[2]
    ]
    assert cleanup_calls
    assert "/eibrain/apps" in cleanup_calls[0][2]
    assert "/eibrain/tests" in cleanup_calls[0][2]
    assert any(command[:2] == ["scp", "-r"] for command in calls)
    assert any(str(command[-1]).endswith("/config/eibrain.honjia.yaml") for command in calls if command[:1] == ["scp"])
    assert any(str(command[-1]).endswith("/config/eibrain.yaml") for command in calls if command[:1] == ["scp"])
    assert any(command[:2] == ["ssh", "darrow@honjia"] and "/REVISION" in command[2] for command in calls)
    assert any(str(command[2]).endswith("tests") for command in calls if command[:2] == ["scp", "-r"])
    assert any(command[:5] == ["ssh", "darrow@honjia", "systemctl", "--user", "restart"] and command[-1] == "eibrain-vision-hailo.service" for command in calls)
    assert any(
        command[:2] == ["ssh", "darrow@honjia"] and command[2].startswith("python3 -c ")
        for command in calls
    )
    assert calls[-1] == ["ssh", "darrow@honjia", "systemctl", "--user", "restart", "eibrain-monitor.service"]
