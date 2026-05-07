# EI Series Deployment Standard

This document defines the honxin deployment layout for the EI series projects.
It keeps source repositories, runtime releases, mutable state, logs, and service
ownership separate so eibrain, eimemory, eihead, eiprotocol, eidocs, eiskills,
eitraining, and OpenClaw can be operated consistently.

## Canonical Hosts

- `honxin`: orchestration, source repositories, memory service, OpenClaw gateway,
  scheduled knowledge/skill jobs, and monitoring tunnels.
- `honjia`: device-side runtime for camera, voice, gimbal, Hailo, and local
  monitoring.

## Directory Layout

| Purpose | Path | Notes |
| --- | --- | --- |
| Source repository | `/dev-project/<project>` | Git working copy and primary edit location on honxin. |
| Runtime root | `/opt/<project>` | User-owned runtime root for releases and shared assets. |
| Immutable release | `/opt/<project>/releases/<git-short-sha>` | rsync copy of source at deploy time. |
| Active release | `/opt/<project>/current` | Symlink to the active release. |
| Shared runtime assets | `/opt/<project>/{bin,models,run,logs}` | Project-specific helpers, models, PID/state files, and logs when not using `/var/log`. |
| Mutable state | `/var/lib/<project>` | Long-lived databases, governance state, indexes, and runtime state. |
| Configuration | `/etc/<project>` | Runtime config not committed to source. |
| User services | `/home/darrow/.config/systemd/user/*.service` | All honxin services run as the `darrow` user via `systemctl --user`. |

## Project Roles

| Project | Source | Runtime | Service role |
| --- | --- | --- | --- |
| `eibrain` | `/dev-project/eibrain` | `/opt/eibrain/current` | Brain code and honjia monitor proxy/service templates. |
| `eimemory` | `/dev-project/eimemory` | `/opt/eimemory/current` | Memory RPC, localhost proxy, governance console, nightly jobs. |
| `eihead` | `/dev-project/eihead` | Library/device repo | Deployed on honjia for hardware runtime; no honxin daemon by default. |
| `eiprotocol` | `/dev-project/eiprotocol` | Library/spec repo | Shared protocol contracts; no daemon. |
| `eidocs` | `/dev-project/eidocs` | CLI/user install | `eidocs-worker.timer` and `eidocs-prune.timer`. |
| `eiskills` | `/dev-project/eiskills` | CLI/user install | audit and auto-evolve timers. |
| `eitraining` | `/dev-project/eitraining` | Batch/tooling repo | No always-on honxin daemon by default. |
| `openclaw` | npm package with EI patch repo `/dev-project/openclaw-lark-image-vision-summary` | `/opt/openclaw/current` symlink to installed package | Feishu/OpenClaw gateway and watchdog timer. |

## Release Procedure

1. Commit code in `/dev-project/<project>` and make sure the working tree is
   clean except intentionally ignored local config such as `.claude/`.
2. Create a release directory:

   ```bash
   /dev-project/eibrain/scripts/deploy-ei-release.sh <project> /dev-project/<project> /opt/<project>
   ```

3. Run project-specific install hooks:

   - `eibrain`: install/editable into `/opt/eibrain/.venv` when Python package
     entry points changed.
   - `eimemory`: copy the previous release `.venv` with `COPY_VENV=1`, then run
     `/opt/eimemory/current/.venv/bin/python -m pip install -e /opt/eimemory/current`.
   - OpenClaw: keep npm package management as the package source, but expose it
     through `/opt/openclaw/current`.

4. Restart or reload the matching user service.
5. Verify the expected process, port, and health endpoint.
6. Push honxin source to GitHub after local verification.

## Service Policy

All long-running honxin processes should be managed by user systemd.

```bash
systemctl --user daemon-reload
systemctl --user enable --now eimemory-rpc.service
systemctl --user enable --now eimemory-rpc-localhost.service
systemctl --user enable --now eimemory-console.service
systemctl --user enable --now eimemory-nightly.timer
systemctl --user enable --now openclaw-gateway.service
systemctl --user enable --now openclaw-stuck-watchdog.timer
```

Use these checks after deploy:

```bash
systemctl --user --no-pager status eimemory-rpc.service eimemory-rpc-localhost.service openclaw-gateway.service
ss -ltnp | grep -E ':(8091|8765|18789|18080)\b'
readlink -f /opt/eibrain/current /opt/eimemory/current /opt/openclaw/current
```

## Current Honxin Port Map

| Port | Owner | Purpose |
| --- | --- | --- |
| `8091` | eimemory | eibrain RPC on Tailscale address plus localhost proxy. |
| `8765` | eimemory | Governance console. |
| `18789` | OpenClaw | Gateway. |
| `18080` | eibrain monitor proxy | SSH tunnel to honjia monitor. |

## Stop And Resume

Pause memory and OpenClaw before migration:

```bash
systemctl --user disable --now eimemory-rpc.service eimemory-rpc-localhost.service eimemory-console.service eimemory-nightly.timer
systemctl --user disable --now openclaw-gateway.service openclaw-stuck-watchdog.timer
```

Resume after deployment:

```bash
systemctl --user enable --now eimemory-rpc.service eimemory-rpc-localhost.service eimemory-console.service eimemory-nightly.timer
systemctl --user enable --now openclaw-gateway.service openclaw-stuck-watchdog.timer
```

## Rollback

Rollback is a symlink switch plus service restart:

```bash
ln -sfn /opt/<project>/releases/<previous-sha> /opt/<project>/current
systemctl --user restart <service-name>
```

Do not edit files directly under `/opt/<project>/current`; make changes in
`/dev-project/<project>`, commit, deploy a new release, then switch `current`.
