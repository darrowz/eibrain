"""Pure software capability registry for an eihead node.

The registry intentionally emits plain dictionaries.  Protocol line A can wrap
these payloads later without making this module block on a hard dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Callable, Mapping


CapabilityConfig = Mapping[str, Any]
PathExists = Callable[[str], bool]
Clock = Callable[[], float]

ONLINE = "online"
OFFLINE = "offline"
DEGRADED = "degraded"
VALID_STATUSES = {ONLINE, OFFLINE, DEGRADED}


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


__all__ = [
    "CapabilityProbeResult",
    "CapabilityRegistry",
    "DEGRADED",
    "DEFAULT_CAPABILITIES",
    "OFFLINE",
    "ONLINE",
    "manifest_from_config",
    "manifest_to_json",
]
