# EI Series Deployment Standard

This document defines the honxin deployment layout for the EI series projects.
It keeps source repositories, runtime releases, mutable state, logs, and service
ownership separate for the Python EI projects. OpenClaw is intentionally managed
as the Node global gateway runtime, not as an `/opt/openclaw/current` release.

Current baseline checked on `honxin` at 2026-05-09:

- EI Python projects use `/dev-project/<project>` plus `/opt/<project>/current`.
- `eibrain` includes the frozen `eiprotocol` package under
  `/opt/eibrain/current/eiprotocol`.
- OpenClaw runs from `/home/darrow/n/lib/node_modules/openclaw` via
  `openclaw-gateway.service`.

## Canonical Hosts

- `honxin`: orchestration, source repositories, memory service, OpenClaw gateway,
  scheduled knowledge jobs, and release management.
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
| `eibrain` | `/dev-project/eibrain` | `/opt/eibrain/current` | Brain code, CLI tools, honjia config templates, and embedded `eiprotocol`; no honxin daemon by default after `eihead` moved device monitoring to honjia. |
| `eiprotocol` | `/dev-project/eibrain/eiprotocol` | `/opt/eibrain/current/eiprotocol` | Shared event/action/observation contracts bundled with eibrain v0.1.1; no daemon. |
| `eimemory` | `/dev-project/eimemory` | `/opt/eimemory/current` | Memory RPC for eibrain/OpenClaw on `127.0.0.1:8091`. |
| `eihead` | exported from `/dev-project/eibrain` | `/opt/eihead/current` on honjia | Deployed on honjia for hardware runtime and monitor; no honxin daemon by default. |
| `eidocs` | `/dev-project/eidocs` | `/opt/eidocs/current` plus user CLI install | `eidocs-worker.timer` and `eidocs-prune.timer`. |
| `eiskills` | `/dev-project/eiskills` | `/opt/eiskills/current` | Skill repository and tooling; no honxin daemon in the current baseline. |
| `eitraining` | `/dev-project/eitraining` | `/opt/eitraining/current` | Batch/tooling repo; no always-on honxin daemon by default. |
| `openclaw` | Node global package plus EI patch/reference repos under `/dev-project` | `/home/darrow/n/lib/node_modules/openclaw` | Feishu/OpenClaw gateway managed by `openclaw-gateway.service`; do not require `/opt/openclaw/current`. |

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
   - OpenClaw: keep npm package management as the package source. The canonical
     runtime path is `/home/darrow/n/lib/node_modules/openclaw`; no
     `/opt/openclaw/current` symlink is required.

4. Restart or reload the matching user service.
5. Verify the expected process, port, and health endpoint.
6. Push honxin source to GitHub after local verification.

## Service Policy

All long-running honxin processes should be managed by user systemd.

```bash
systemctl --user daemon-reload
systemctl --user enable --now eimemory-rpc.service
systemctl --user enable --now eidocs-worker.timer
systemctl --user enable --now eidocs-prune.timer
systemctl --user enable --now openclaw-gateway.service
```

Use these checks after deploy:

```bash
systemctl --user --no-pager status eimemory-rpc.service eidocs-worker.timer openclaw-gateway.service
ss -ltnp | grep -E ':(8091|18789|18791)\b'
readlink -f /opt/eibrain/current /opt/eimemory/current /opt/eidocs/current /opt/eiskills/current /opt/eitraining/current
/opt/eibrain/current/.venv/bin/python /opt/eibrain/current/scripts/eiprotocol_conformance_report.py --strict
```

## Current Honxin Port Map

| Port | Owner | Purpose |
| --- | --- | --- |
| `8091` | eimemory | eibrain/OpenClaw memory RPC bound to `127.0.0.1`. |
| `18789` | OpenClaw | Gateway. |
| `18791` | OpenClaw | Local browser/control sidecar. |

`8765` and `18080` are not honxin baseline listeners in the current deployment.
`18080` belongs to honjia `eihead-monitor.service`.

## Stop And Resume

Pause memory and OpenClaw before migration:

```bash
systemctl --user disable --now eimemory-rpc.service eidocs-worker.timer eidocs-prune.timer
systemctl --user disable --now openclaw-gateway.service
```

Resume after deployment:

```bash
systemctl --user enable --now eimemory-rpc.service eidocs-worker.timer eidocs-prune.timer
systemctl --user enable --now openclaw-gateway.service
```

## Rollback

Rollback is a symlink switch plus service restart:

```bash
ln -sfn /opt/<project>/releases/<previous-sha> /opt/<project>/current
systemctl --user restart <service-name>
```

Do not edit files directly under `/opt/<project>/current`; make changes in
`/dev-project/<project>`, commit, deploy a new release, then switch `current`.
