"""CLI entrypoint for default deployment bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

from eibrain.infra.config import load_config
from eibrain.infra.deployment import bootstrap_default_deployment


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the default eibrain deployment layout.")
    parser.add_argument("--config", default="config/eibrain.yaml", help="Path to unified YAML config")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    layout = bootstrap_default_deployment(config)
    print(f"root_dir={layout.root_dir}")
    print(f"body_runtime_dir={layout.body_runtime_dir}")
    print(f"cognitive_runtime_dir={layout.cognitive_runtime_dir}")
    print(f"sherpa_model_dir={layout.sherpa_model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
