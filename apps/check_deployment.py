"""CLI entrypoint for deployment validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from eibrain.infra.config import load_config
from eibrain.infra.deployment import bootstrap_default_deployment


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and bootstrap the eibrain deployment layout.")
    parser.add_argument("--config", default="config/eibrain.yaml", help="Path to unified YAML config")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root_dir / config_path

    env_example = root_dir / ".env.example"
    if not config_path.is_file():
        raise SystemExit(f"missing config: {config_path}")
    if not env_example.is_file():
        raise SystemExit(f"missing env example: {env_example}")

    config = load_config(config_path)
    layout = bootstrap_default_deployment(config)
    print(f"config={config_path}")
    print(f"env_example={env_example}")
    print(f"root_dir={layout.root_dir}")
    print(f"body_runtime_dir={layout.body_runtime_dir}")
    print(f"cognitive_runtime_dir={layout.cognitive_runtime_dir}")
    print(f"sherpa_model_dir={layout.sherpa_model_dir}")
    print("deployment-check=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
