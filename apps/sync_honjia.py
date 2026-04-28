"""Sync deployable runtime files from honxin primary repo to honjia."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


DEPLOY_PATHS = (
    "apps",
    "eibrain",
    "deploy",
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
    parser.add_argument("--restart-vision", action="store_true", help="Restart eibrain-vision-hailo after sync")
    parser.add_argument("--restart-services", action="store_true", help="Restart vision first, then monitor")
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
    revision = _current_revision(repo_root)
    revision_path = args.target_dir.rstrip("/") + "/REVISION"
    _run(["ssh", args.target_host, f"printf '%s\\n' {shlex.quote(revision)} > {shlex.quote(revision_path)}"])
    config_target = f"{args.target_dir}/{args.config_source}"
    config_target_dir = str(Path(config_target).parent).replace("\\", "/")
    _run(["ssh", args.target_host, f"mkdir -p {config_target_dir}"])
    _run(["scp", str(config_source), f"{args.target_host}:{config_target}"])
    _run(["ssh", args.target_host, f"mkdir -p {args.target_dir}/config"])
    _run(["scp", str(config_source), f"{args.target_host}:{args.target_dir}/config/eibrain.yaml"])
    if args.restart_services or args.restart_vision:
        _run(["ssh", args.target_host, "systemctl", "--user", "restart", "eibrain-vision-hailo.service"])
    if args.restart_services:
        freshness_check = (
            "import json,time,pathlib; p=pathlib.Path('/tmp/eibrain-vision/state.json'); deadline=time.time()+10\n"
            "while time.time()<deadline:\n"
            "    if p.exists():\n"
            "        data=json.loads(p.read_text()); ts=float(data.get('updated_at_ts',0) or 0)\n"
            "        if time.time()-ts<5: raise SystemExit(0)\n"
            "    time.sleep(0.5)\n"
            "raise SystemExit('vision state did not become fresh')"
        )
        _run(
            [
                "ssh",
                args.target_host,
                shlex.join(["python3", "-c", freshness_check]),
            ]
        )
    if args.restart_services or args.restart_monitor:
        _run(["ssh", args.target_host, "systemctl", "--user", "restart", "eibrain-monitor.service"])
    print(f"synced={args.target_host}:{args.target_dir}")
    print(f"revision={revision}")
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


def _current_revision(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    revision = result.stdout.strip()
    return revision or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
