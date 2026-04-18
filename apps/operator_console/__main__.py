"""CLI entrypoint for honjia monitoring web."""

from __future__ import annotations

import argparse

from apps.body_runtime.app import BodyRuntimeApp
from eibrain.infra.config import load_config

from .web import MonitoringWebServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Start eibrain honjia monitoring web")
    parser.add_argument("--config", default="config/eibrain.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    runtime = BodyRuntimeApp(config=config)
    server = MonitoringWebServer(
        runtime=runtime,
        host=config.monitoring.host,
        port=config.monitoring.port,
    )
    server.start()
    try:
        print(f"monitoring web listening on http://{config.monitoring.host}:{server.port}")
        server._thread.join()  # type: ignore[union-attr]
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
