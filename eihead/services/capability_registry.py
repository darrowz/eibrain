"""Pure software capability registry for an eihead node.

The registry keeps the legacy plain dictionary manifest stable for existing
monitoring callers, and can also emit an eiprotocol v0.1 capability event for
the eihead/eibrain boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from eiprotocol import EventEnvelope


CapabilityConfig = Mapping[str, Any]
PathExists = Callable[[str], bool]
Clock = Callable[[], float]

ONLINE = "online"
OFFLINE = "offline"
DEGRADED = "degraded"
VALID_STATUSES = {ONLINE, OFFLINE, DEGRADED}
EIPROTOCOL_MANIFEST_EVENT = "ei.capability.manifest.report"


DEFAULT_CAPABILITIES: dict[str, dict[str, Any]] = {
    "camera": {
        "kind": "hardware",
        "paths": ["/dev/video0"],
        "limits": {"streams": 1},
    },
    "hailo": {
        "kind": "accelerator",
        "paths": ["/dev/hailo0"],
        "limits": {"device_count": 1},
    },
    "i2c": {
        "kind": "bus",
        "paths": ["/dev/i2c-1"],
        "limits": {"bus": 1},
    },
    "microphone": {
        "kind": "hardware",
        "paths": [],
        "limits": {"channels": 1},
    },
    "speaker": {
        "kind": "hardware",
        "paths": [],
        "limits": {"channels": 1},
    },
    "neck": {
        "kind": "actuator",
        "paths": ["/dev/i2c-1"],
        "limits": {"pan_deg": [0, 180], "tilt_deg": None},
    },
    "asr": {
        "kind": "software",
        "paths": [],
        "limits": {"streaming": False},
    },
    "tts": {
        "kind": "software",
        "paths": [],
        "limits": {"streaming": False},
    },
    "vision_backend": {
        "kind": "software",
        "paths": [],
        "limits": {"realtime": False},
    },
}


@dataclass(slots=True)
class CapabilityProbeResult:
    """Normalized status for one eihead capability."""

    name: str
    kind: str
    status: str
    latency_ms: float | None = None
    last_ok_ts: float | None = None
    error: str | None = None
    limits: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "last_ok_ts": self.last_ok_ts,
            "error": self.error,
            "limits": dict(self.limits),
            "details": dict(self.details),
        }


class CapabilityRegistry:
    """Builds a CapabilityManifest for honjia without touching live services."""

    schema = "eihead.capability_manifest.v1"

    def __init__(
        self,
        config: CapabilityConfig | None = None,
        *,
        node_id: str | None = None,
        path_exists: PathExists | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = _normalize_config(config or {})
        self.node_id = node_id or str(self._config.get("node_id") or "honjia")
        self._path_exists = path_exists or os.path.exists
        self._clock = clock or time.time

    @classmethod
    def from_config(
        cls,
        config: CapabilityConfig | None = None,
        *,
        path_exists: PathExists | None = None,
        clock: Clock | None = None,
    ) -> "CapabilityRegistry":
        return cls(config, path_exists=path_exists, clock=clock)

    def capability_names(self) -> list[str]:
        configured = set(_capability_section(self._config))
        return sorted(set(DEFAULT_CAPABILITIES) | configured)

    def probe(self, name: str) -> CapabilityProbeResult:
        default = DEFAULT_CAPABILITIES.get(name, {"kind": "custom", "paths": [], "limits": {}})
        config = _capability_section(self._config).get(name, {})
        merged = _merge_capability(default, config)
        started = self._clock()

        status, error, path_details = self._resolve_status(name, merged)
        finished = self._clock()
        latency_ms = round(max(finished - started, 0.0) * 1000, 3)
        last_ok_ts = finished if status in {ONLINE, DEGRADED} else _optional_float(merged.get("last_ok_ts"))

        details = dict(merged.get("details") or {})
        details.update(path_details)
        _copy_optional(details, merged, "backend")
        _copy_optional(details, merged, "provider")
        _copy_optional(details, merged, "model")

        return CapabilityProbeResult(
            name=name,
            kind=str(merged.get("kind") or default["kind"]),
            status=status,
            latency_ms=latency_ms,
            last_ok_ts=last_ok_ts,
            error=error,
            limits=dict(merged.get("limits") or {}),
            details=details,
        )

    def manifest(self) -> dict[str, Any]:
        generated_at_ts = self._clock()
        capabilities = {name: self.probe(name).to_dict() for name in self.capability_names()}
        return {
            "schema": self.schema,
            "node_id": self.node_id,
            "generated_at_ts": generated_at_ts,
            "capabilities": capabilities,
        }

    def eiprotocol_manifest(
        self,
        *,
        event_id: str | None = None,
        request_id: str | None = None,
        sequence: int = 0,
        time: str | None = None,
    ) -> EventEnvelope:
        return manifest_to_eiprotocol_event(
            self.manifest(),
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )

    def to_json(self) -> str:
        return json.dumps(self.manifest(), ensure_ascii=False, sort_keys=True)

    def _resolve_status(self, name: str, config: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
        if config.get("enabled") is False:
            return OFFLINE, "disabled", {"paths": _paths_from_config(config), "available_paths": []}

        explicit_status = config.get("status")
        explicit_error = config.get("error")
        if explicit_status is not None:
            return _coerce_status(str(explicit_status)), _optional_str(explicit_error), {
                "paths": _paths_from_config(config),
                "available_paths": list(config.get("available_paths") or []),
            }

        paths = _paths_from_config(config)
        if paths:
            available = [path for path in paths if self._path_exists(path)]
            if len(available) == len(paths):
                return ONLINE, None, {"paths": paths, "available_paths": available}
            if available:
                return DEGRADED, _path_error(name, paths, available), {
                    "paths": paths,
                    "available_paths": available,
                }
            return OFFLINE, _path_error(name, paths, available), {"paths": paths, "available_paths": []}

        if config.get("enabled") is True or _has_declaration(config):
            return ONLINE, None, {"paths": [], "available_paths": []}

        return OFFLINE, "not_configured", {"paths": [], "available_paths": []}


def manifest_from_config(
    config: CapabilityConfig | None = None,
    *,
    path_exists: PathExists | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    return CapabilityRegistry.from_config(config, path_exists=path_exists, clock=clock).manifest()


def manifest_to_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True)


def manifest_to_eiprotocol_event(
    manifest: Mapping[str, Any],
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int = 0,
    time: str | None = None,
) -> EventEnvelope:
    from eiprotocol import Capability, CapabilityManifest, DeviceStatus, SourceRef

    node_id = str(manifest.get("node_id") or "honjia")
    capabilities, backends = _eiprotocol_capability_lists(manifest)
    resolved_event_id = event_id or f"evt_capability_manifest_{_stable_token(node_id)}"
    eiprotocol_manifest = CapabilityManifest(
        manifest_id=f"capability_manifest.{node_id}",
        manifest_version=str(manifest.get("schema") or CapabilityRegistry.schema),
        device={
            "deviceId": node_id,
            "nodeId": node_id,
            "source": f"eihead.{node_id}",
            "model": "honjia",
        },
        runtime={
            "generatedAtTs": manifest.get("generated_at_ts"),
        },
        modalities={
            "vision": ["camera.front"],
            "audioInput": ["microphone.default"],
            "audioOutput": ["speaker.default"],
            "actuation": ["neck.pan"],
            "bus": ["bus.i2c"],
            "accelerator": ["accelerator.hailo"],
        },
        capabilities=[Capability.from_dict(item) for item in capabilities],
        backends=[Capability.from_dict(item) for item in backends],
        health=DeviceStatus(
            status=_overall_status(manifest),
            message=f"registry manifest for {node_id}",
            checked_at_ms=_timestamp_ms(manifest.get("generated_at_ts")),
        ),
        metadata={
            "registrySchema": manifest.get("schema"),
            "registryNodeId": node_id,
        },
    )
    return eiprotocol_manifest.to_event(
        event_id=resolved_event_id,
        request_id=request_id or resolved_event_id,
        sequence=sequence,
        source=SourceRef(domain="eihead", instance_id=node_id, device_id=node_id),
        time=time or _event_time(manifest.get("generated_at_ts")),
    )


def _normalize_config(config: CapabilityConfig) -> dict[str, Any]:
    normalized = dict(config)
    if "capabilities" not in normalized:
        capabilities = {
            key: value
            for key, value in normalized.items()
            if key in DEFAULT_CAPABILITIES and isinstance(value, Mapping)
        }
        if capabilities:
            normalized["capabilities"] = capabilities
    return normalized


def _capability_section(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    section = config.get("capabilities") or {}
    if not isinstance(section, Mapping):
        return {}
    return {str(name): dict(value) for name, value in section.items() if isinstance(value, Mapping)}


def _merge_capability(default: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(default)
    merged.update(config)
    merged["limits"] = {**dict(default.get("limits") or {}), **dict(config.get("limits") or {})}
    if "path" in config and "paths" not in config:
        merged.pop("paths", None)
    return merged


def _paths_from_config(config: Mapping[str, Any]) -> list[str]:
    paths = config.get("paths", config.get("path", []))
    if isinstance(paths, str):
        return [paths]
    if paths is None:
        return []
    return [str(path) for path in paths]


def _coerce_status(status: str) -> str:
    normalized = status.strip().lower()
    return normalized if normalized in VALID_STATUSES else DEGRADED


def _path_error(name: str, paths: list[str], available: list[str]) -> str:
    missing = [path for path in paths if path not in available]
    return f"{name} missing paths: {', '.join(missing)}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _has_declaration(config: Mapping[str, Any]) -> bool:
    declarative_keys = {"backend", "provider", "model", "limits", "details"}
    return any(key in config for key in declarative_keys)


def _copy_optional(target: dict[str, Any], source: Mapping[str, Any], key: str) -> None:
    if key in source:
        target[key] = source[key]


def _eiprotocol_capability_lists(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry_capabilities = manifest.get("capabilities")
    if not isinstance(registry_capabilities, Mapping):
        return [], []

    capabilities: list[dict[str, Any]] = []
    backends: list[dict[str, Any]] = []
    for name in sorted(registry_capabilities):
        raw_capability = registry_capabilities[name]
        if not isinstance(raw_capability, Mapping):
            continue
        capability = _registry_capability_to_eiprotocol(str(name), raw_capability)
        if str(raw_capability.get("kind") or "") == "software":
            backends.append(capability)
        else:
            capabilities.append(capability)
    return capabilities, backends


def _registry_capability_to_eiprotocol(name: str, capability: Mapping[str, Any]) -> dict[str, Any]:
    details = capability.get("details") if isinstance(capability.get("details"), Mapping) else {}
    metadata = {
        "registryName": name,
        "registryKind": capability.get("kind"),
        "latencyMs": capability.get("latency_ms"),
        "lastOkTs": capability.get("last_ok_ts"),
        "error": capability.get("error"),
        "paths": list(details.get("paths") or []),
        "availablePaths": list(details.get("available_paths") or []),
    }
    metadata.update({key: value for key, value in dict(details).items() if key not in {"paths", "available_paths"}})
    return {
        "capabilityId": _eiprotocol_capability_id(name),
        "kind": _eiprotocol_kind(name, capability),
        "provider": str(details.get("provider") or ""),
        "model": str(details.get("model") or details.get("backend") or ""),
        "version": "",
        "devicePath": _device_path(details),
        "actions": _eiprotocol_actions(name),
        "status": str(capability.get("status") or "unknown"),
        "limits": _eiprotocol_limits(name, capability),
        "metadata": metadata,
    }


def _eiprotocol_capability_id(name: str) -> str:
    return {
        "camera": "camera.front",
        "hailo": "accelerator.hailo",
        "i2c": "bus.i2c",
        "microphone": "microphone.default",
        "speaker": "speaker.default",
        "neck": "neck.pan",
        "asr": "asr.default",
        "tts": "tts.default",
        "vision_backend": "vision.default",
    }.get(name, name)


def _eiprotocol_kind(name: str, capability: Mapping[str, Any]) -> str:
    return {
        "camera": "camera",
        "hailo": "vision_accelerator",
        "i2c": "bus",
        "microphone": "audio_input",
        "speaker": "audio_output",
        "neck": "actuator",
        "asr": "asr",
        "tts": "tts",
        "vision_backend": "vision",
    }.get(name, str(capability.get("kind") or name))


def _eiprotocol_actions(name: str) -> list[str]:
    return {
        "camera": ["frame_capture", "video_stream"],
        "hailo": ["inference"],
        "i2c": ["device_bus"],
        "microphone": ["audio_capture"],
        "speaker": ["audio_playback"],
        "neck": ["move_head"],
    }.get(name, [])


def _eiprotocol_limits(name: str, capability: Mapping[str, Any]) -> dict[str, Any]:
    limits = capability.get("limits") if isinstance(capability.get("limits"), Mapping) else {}
    if name != "neck":
        return dict(limits)

    pan = limits.get("pan_deg")
    tilt = limits.get("tilt_deg")
    pan_min, pan_max = _range_pair(pan, 0, 180)
    mapped = {
        "axis": "pan",
        "minAngle": pan_min,
        "maxAngle": pan_max,
        "tiltSupported": tilt is not None,
    }
    if tilt is not None:
        tilt_min, tilt_max = _range_pair(tilt, 0, 0)
        mapped["minTiltAngle"] = tilt_min
        mapped["maxTiltAngle"] = tilt_max
    return mapped


def _range_pair(value: Any, default_min: int, default_max: int) -> tuple[Any, Any]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0], value[1]
    return default_min, default_max


def _device_path(details: Mapping[str, Any]) -> str:
    paths = details.get("paths") or details.get("available_paths") or []
    if isinstance(paths, str):
        return paths
    if isinstance(paths, list) and paths:
        return str(paths[0])
    return ""


def _overall_status(manifest: Mapping[str, Any]) -> str:
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, Mapping) or not capabilities:
        return OFFLINE
    statuses = {
        str(capability.get("status") or OFFLINE)
        for capability in capabilities.values()
        if isinstance(capability, Mapping)
    }
    if OFFLINE in statuses:
        return DEGRADED
    if DEGRADED in statuses:
        return DEGRADED
    return ONLINE


def _timestamp_ms(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number * 1000)


def _event_time(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "1970-01-01T00:00:00.000Z"
    return datetime.fromtimestamp(number, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _stable_token(value: object) -> str:
    token = str(value).strip().replace(" ", "_")
    return token or "event"


__all__ = [
    "CapabilityProbeResult",
    "CapabilityRegistry",
    "DEGRADED",
    "EIPROTOCOL_MANIFEST_EVENT",
    "DEFAULT_CAPABILITIES",
    "OFFLINE",
    "ONLINE",
    "manifest_from_config",
    "manifest_to_eiprotocol_event",
    "manifest_to_json",
]
