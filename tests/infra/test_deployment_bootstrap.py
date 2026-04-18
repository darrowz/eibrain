from __future__ import annotations


def test_bootstrap_deployment_creates_runtime_and_streaming_model_layout(tmp_path) -> None:
    from eibrain.infra.config import EIBrainConfig
    from eibrain.infra.deployment import bootstrap_default_deployment

    config = EIBrainConfig()
    config.deployment.root_dir = str(tmp_path / "eibrain")
    config.deployment.body_runtime_dir = ""
    config.deployment.cognitive_runtime_dir = ""
    config.deployment.__post_init__()

    result = bootstrap_default_deployment(config)

    assert result.root_dir.exists()
    assert result.body_runtime_dir.exists()
    assert result.cognitive_runtime_dir.exists()
    assert result.sherpa_model_dir.exists()
    assert (result.sherpa_model_dir / "README.md").exists()
    assert (result.sherpa_model_dir / "tokens.txt").exists()
