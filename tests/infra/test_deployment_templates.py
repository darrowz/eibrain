from __future__ import annotations

from pathlib import Path


def test_eibrain_monitor_systemd_template_uses_local_deploy_config() -> None:
    unit_text = Path('deploy/systemd/eibrain-monitor.service').read_text(encoding='utf-8')

    assert 'ExecStart=/opt/eibrain/.venv/bin/python -m apps.operator_console --config /opt/eibrain/current/config/eibrain.honjia.local.yaml' in unit_text
    assert 'WorkingDirectory=/opt/eibrain/current' in unit_text
    assert '/dev-project/eibrain' not in unit_text


def test_honjia_local_config_moves_monitor_to_18081() -> None:
    config_text = Path('config/eibrain.honjia.local.yaml').read_text(encoding='utf-8')

    assert 'port: 18081' in config_text
    assert 'provider: eimemory_rpc' in config_text
