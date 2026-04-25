"""Sync deployable runtime files from honxin primary repo to honjia."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


DEPLOY_PATHS = (
    "apps",
    "eibrain",
    "scripts",
    "pyproject.toml",
    "README.md",
    ".env.example",
)

OPTIONAL_PATHS = (
    "tests",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync deployable eibrain files to honjia.")
    parser.add_argument("--target-host", required=True, help="honjia ssh target, e.g. darrow@honjia")
    parser.add_argument("--target-dir", default="/home/darrow/dev-project/eibrain", help="Remote deployment dir")
    parser.add_argument(
        "--config-source",
        default="config/eibrain.honjia.yaml",
        help="Source config file in the honxin repo to deploy as honjia config/eibrain.yaml",
    )
    parser.add_argument("--include-tests", action="store_true", help="Also sync the tests directory for remote validation")
    parser.add_argument("--restart-monitor", action="store_true", help="Restart eibrain-monitor after sync")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_source = repo_root / args.config_source
    if not config_source.is_file():
        raise SystemExit(f"missing config source: {config_source}")
    _run(["ssh", args.target_host, f"mkdir -p {args.target_dir}"])
    _cleanup_target(args.target_host, args.target_dir)
    for rel in DEPLOY_PATHS:
        source = repo_root / rel
        _run(["scp", "-r", str(source), f"{args.target_host}:{args.target_dir}/"])
    if args.include_tests:
        for rel in OPTIONAL_PATHS:
            source = repo_root / rel
            _run(["scp", "-r", str(source), f"{args.target_host}:{args.target_dir}/"])
    config_target = f"{args.target_dir}/{args.config_source}"
    config_target_dir = str(Path(config_target).parent).replace("\\", "/")
    _run(["ssh", args.target_host, f"mkdir -p {config_target_dir}"])
    _run(["scp", str(config_source), f"{args.target_host}:{config_target}"])
    _run(["ssh", args.target_host, f"mkdir -p {args.target_dir}/config"])
    _run(["scp", str(config_source), f"{args.target_host}:{args.target_dir}/config/eibrain.yaml"])
    if args.restart_monitor:
        _run(["ssh", args.target_host, "systemctl", "--user", "restart", "eibrain-monitor.service"])
    print(f"synced={args.target_host}:{args.target_dir}")
    print(f"config_source={config_source}")
    print(f"include_tests={args.include_tests}")
    return 0


def _cleanup_target(target_host: str, target_dir: str) -> None:
    remote_paths = " ".join(
        shlex.quote(_remote_path(target_dir, rel)) for rel in (*DEPLOY_PATHS, *OPTIONAL_PATHS)
    )
    _run(["ssh", target_host, f"rm -rf -- {remote_paths}"])


def _remote_path(target_dir: str, rel: str) -> str:
    return f"{target_dir.rstrip('/')}/{rel}"


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
