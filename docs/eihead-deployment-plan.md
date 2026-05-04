# eihead Deployment Plan

This plan defines the deployment shape for moving honjia head hardware from
the monolithic eibrain body runtime into an eihead service pair. It is a plan
and template set only; it does not perform deployment.

## Repository And Runtime Paths

- honxin `/dev-project` is the code source of truth. It is not a runtime path.
- honxin `/dev-project/eibrain` keeps the current source tree while eihead is
  being extracted.
- honxin `/dev-project/eihead` is the target source repository once the split
  becomes independent.
- honjia `/opt/eihead/current` is the target runtime path for the deployed
  release.
- honjia `/etc/eihead/eihead.honjia.yaml` is the default runtime config.
- honjia `/etc/eihead/eihead.env` is the optional environment override file.
- honjia `18081` is reserved for the eihead runtime HTTP API.
- honjia `18080` remains the operator Web monitoring URL through the
  compatibility monitor.

## Service Templates

- `deploy/systemd/eihead-runtime.service` starts the runtime API with
  `eihead-runtime --config /etc/eihead/eihead.honjia.yaml http --host 0.0.0.0 --port 18081`.
- `deploy/systemd/eihead-monitor.service` starts the compatibility Web monitor
  after `eihead-runtime.service` and keeps the user-facing Web port on `18080`.
- The templates run as user `darrow` from `/opt/eihead/current`.
- The templates do not edit, remove, or override existing eibrain service
  files.
- The compatibility monitor still reads the monitor bind address from
  `/etc/eihead/eihead.honjia.yaml`; keep `monitoring.port: 18080` there until
  an eihead-native proxy replaces the wrapper.
- No safety or permission gating is introduced in this phase.

## Cutover Strategy

The current honjia node has no production business load, so the preferred
strategy is a short downtime cutover rather than a side-by-side migration. This
reduces port conflicts and makes acceptance easier.

During the downtime window, it is acceptable to stop the old eibrain head-side
services before starting eihead:

```bash
sudo systemctl stop eibrain-monitor.service eibrain-vision-hailo.service
systemctl --user stop eibrain-monitor.service eibrain-vision-hailo.service brain-runtime.service
```

Some units may not exist on a given honjia image. Treat "unit not found" as
non-fatal, then confirm ports and devices are free before starting eihead.

Start the new services only after the old monitor/body/vision ownership is
released:

```bash
sudo systemctl daemon-reload
sudo systemctl start eihead-runtime.service
sudo systemctl start eihead-monitor.service
```

Enable boot persistence only after acceptance passes:

```bash
sudo systemctl enable eihead-runtime.service eihead-monitor.service
```

## Acceptance Checks

- `systemctl status eihead-runtime.service` is active.
- `systemctl status eihead-monitor.service` is active.
- `curl http://127.0.0.1:18081/status` returns eihead runtime status.
- `curl http://127.0.0.1:18081/capabilities` returns the honjia capability
  manifest when the runtime API supports it.
- `curl http://127.0.0.1:18080` opens the Web monitor.
- `/dev/video0`, `/dev/hailo0`, `/dev/i2c-1`, microphone, and speaker state
  appear in the Web monitor as real data, degraded data, or explicit offline
  data; blank "normal" placeholders are not acceptable.
- Voice, vision frame capture, Hailo detection, and horizontal neck movement
  are manually verified after service start.

## Rollback

Rollback is service-level only. Stop `eihead-monitor.service` and
`eihead-runtime.service`, then start the previous eibrain services from the same
downtime window. Do not overwrite honxin source repositories during rollback.
