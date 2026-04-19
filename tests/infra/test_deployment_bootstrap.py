from __future__ import annotations

from pathlib import Path
import shutil
import sys


def _make_tmp_dir(name: str) -> Path:
    path = Path.cwd() / ".tmp-test-artifacts" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_bootstrap_deployment_creates_runtime_and_streaming_model_layout() -> None:
    repo_source_root = Path(__file__).resolve().parents[2]
    if str(repo_source_root) not in sys.path:
        sys.path.insert(0, str(repo_source_root))
    from eibrain.infra.config import EIBrainConfig
    from eibrain.infra.deployment import bootstrap_default_deployment

    tmp_path = _make_tmp_dir("deployment-bootstrap")
    try:
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
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
