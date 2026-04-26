from __future__ import annotations

from pathlib import Path


def test_eibrain_monitor_systemd_template_uses_local_deploy_config() -> None:
    unit_text = Path('deploy/systemd/eibrain-monitor.service').read_text(encoding='utf-8')

    assert 'WorkingDirectory=/home/darrow/dev-project/eibrain' in unit_text
    assert 'ExecStart=/home/darrow/dev-project/eibrain/.venv/bin/python -m apps.operator_console' in unit_text
    assert '--config /home/darrow/dev-project/eibrain/config/eibrain.honjia.yaml' in unit_text
    assert '--disable-visual-tracking-loop' in unit_text


def test_eibrain_vision_hailo_systemd_template_owns_camera_pipeline() -> None:
    unit_text = Path('deploy/systemd/eibrain-vision-hailo.service').read_text(encoding='utf-8')

    assert 'WorkingDirectory=/home/darrow/dev-project/eibrain' in unit_text
    assert 'ExecStart=/usr/bin/python3 -m apps.body_runtime.vision_hailo_service' in unit_text
    assert '--backend gstreamer' in unit_text


def test_honjia_local_config_moves_monitor_to_18081() -> None:
    config_text = Path('config/eibrain.honjia.local.yaml').read_text(encoding='utf-8')

    assert 'port: 18081' in config_text
    assert 'provider: eimemory_rpc' in config_text
