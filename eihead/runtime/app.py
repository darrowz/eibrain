"""Head runtime scaffold.

This module intentionally delegates to ``apps.body_runtime`` for the first
migration pass. Keeping the wrapper small lets us introduce the eihead package
without changing the proven honjia body runtime path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from eihead.monitoring import build_status_snapshot
from eihead.protocol import MoveHeadAction, PlaySpeechAction, StopSpeechAction
from eihead.services import CapabilityRegistry

DEFAULT_CONFIG_PATH = "config/eibrain.yaml"
BodyRuntimeFactory = Callable[[str], Any]


def _default_body_runtime_factory(config_path: str) -> Any:
    from apps.body_runtime.app import BodyRuntimeApp

    return BodyRuntimeApp.from_config_path(config_path)


def _body_runtime_config_path(config_path: str) -> str:
    """Resolve the legacy body config when the user passes an eihead config."""

    if not Path(config_path).name.startswith("eihead"):
        return config_path
    try:
        from eihead.runtime.config import EiheadConfigError, load_eihead_config

        config = load_eihead_config(config_path)
    except (EiheadConfigError, OSError):
        return config_path
    return config.legacy.body_runtime_config_path or config_path


@dataclass(slots=True)
class HeadRuntimeApp:
    """Compatibility wrapper for the future standalone eihead runtime."""

    body_runtime: Any
    config_path: str = DEFAULT_CONFIG_PATH
    delegate_name: str = "apps.body_runtime.BodyRuntimeApp"

    @classmethod
    def from_config_path(
        cls,
        path: str = DEFAULT_CONFIG_PATH,
        *,
        body_runtime_factory: BodyRuntimeFactory | None = None,
    ) -> "HeadRuntimeApp":
        factory = body_runtime_factory or _default_body_runtime_factory
        body_config_path = _body_runtime_config_path(str(path))
        return cls(body_runtime=factory(body_config_path), config_path=str(path))

    def snapshot(self) -> dict[str, Any]:
        body_snapshot = self._body_snapshot()
        return {
            "runtime": "eihead",
            "node_role": "head",
            "status": "ok",
            "config_path": self.config_path,
            "delegate": self.delegate_name,
            "body_runtime": body_snapshot,
        }

    def status(self) -> dict[str, Any]:
        return {
            "command": "status",
            **self.snapshot(),
        }

    def capabilities(self) -> dict[str, Any]:
        body_snapshot = self._body_snapshot()
        node_id = _string_or_default(body_snapshot.get("node_id"), "honjia")
        manifest = CapabilityRegistry({"node_id": node_id}).manifest()
        status_snapshot = build_status_snapshot(manifest)
        return {
            "command": "capabilities",
            "runtime": "eihead",
            "node_role": "head",
            "config_path": self.config_path,
            "delegate": self.delegate_name,
            "body_runtime_node_id": node_id,
            "body_runtime_capabilities": body_snapshot.get("capabilities", {}),
            **status_snapshot,
        }

    def handle_action(self, action: Mapping[str, Any] | Any, trace_id: str | None = None) -> dict[str, Any]:
        normalized, effective_trace_id = self._normalize_action(action, trace_id=trace_id)
        action_type = self._action_type(normalized)
        action_id = _string_or_default(normalized.get("action_id") or normalized.get("id"), "")

        if action_type == "speak":
            text = _string_or_default(_action_value(normalized, "text"), "")
            if not text.strip():
                return self._action_outcome(
                    action_id=action_id,
                    action_type=action_type,
                    trace_id=effective_trace_id,
                    status="skipped",
                    success=False,
                    details={"reason": "missing_text"},
                )
            protocol_action = PlaySpeechAction(
                ts=time.time(),
                source="eihead.runtime",
                text=text,
                session_id=_optional_string(_action_value(normalized, "session_id")),
                actor_id=_optional_string(_action_value(normalized, "actor_id")),
                target_id=_optional_string(_action_value(normalized, "target_id")),
            )
            return self._dispatch_protocol_action(
                protocol_action,
                action_id=action_id,
                action_type=action_type,
                trace_id=effective_trace_id,
                details={"text_char_count": len(text)},
            )

        if action_type == "move_head":
            axis = _string_or_default(_action_value(normalized, "axis"), "yaw").strip().lower() or "yaw"
            if axis != "yaw":
                return self._action_outcome(
                    action_id=action_id,
                    action_type=action_type,
                    trace_id=effective_trace_id,
                    status="unsupported",
                    success=False,
                    details={"axis": axis, "reason": "honjia currently exposes yaw/pan only"},
                )
            target_angle = _action_value(normalized, "target_angle")
            if target_angle is None:
                target_angle = _action_value(normalized, "angle")
            protocol_action = MoveHeadAction(
                ts=time.time(),
                source="eihead.runtime",
                session_id=_optional_string(_action_value(normalized, "session_id")),
                actor_id=_optional_string(_action_value(normalized, "actor_id")),
                target_id=_optional_string(_action_value(normalized, "target_id")),
                target_name=_string_or_default(_action_value(normalized, "target_name"), "manual"),
                target_x=_optional_float(_action_value(normalized, "target_x")),
                target_angle=_optional_int(target_angle),
            )
            return self._dispatch_protocol_action(
                protocol_action,
                action_id=action_id,
                action_type=action_type,
                trace_id=effective_trace_id,
                details={"axis": "yaw"},
            )

        if action_type == "stop_speech":
            protocol_action = StopSpeechAction(
                ts=time.time(),
                source="eihead.runtime",
                session_id=_optional_string(_action_value(normalized, "session_id")),
                actor_id=_optional_string(_action_value(normalized, "actor_id")),
                target_id=_optional_string(_action_value(normalized, "target_id")),
            )
            return self._dispatch_protocol_action(
                protocol_action,
                action_id=action_id,
                action_type=action_type,
                trace_id=effective_trace_id,
            )

        if action_type == "capture_frame":
            return self._capture_frame_outcome(
                action_id=action_id,
                trace_id=effective_trace_id,
            )

        return self._action_outcome(
            action_id=action_id,
            action_type=action_type or "unknown",
            trace_id=effective_trace_id,
            status="unsupported",
            success=False,
            details={"reason": "unsupported_action_type"},
        )

    def serve(self) -> dict[str, Any]:
        return {
            "command": "serve",
            "serve_mode": "compatibility_snapshot",
            **self.snapshot(),
        }

    def verify(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        body_runtime = snapshot.get("body_runtime", {})
        organ_count = body_runtime.get("organ_count") if isinstance(body_runtime, Mapping) else None
        return {
            "command": "verify",
            "runtime": "eihead",
            "status": "ok",
            "checks": {
                "head_runtime_import": "ok",
                "body_runtime_delegate": "ok",
                "body_runtime_snapshot": "ok",
            },
            "organ_count": organ_count,
            "config_path": self.config_path,
            "delegate": self.delegate_name,
            "body_runtime": body_runtime,
        }

    def _body_snapshot(self) -> dict[str, Any]:
        if not hasattr(self.body_runtime, "snapshot"):
            raise TypeError("body_runtime must expose snapshot() for eihead compatibility")
        snapshot = self.body_runtime.snapshot()
        if not isinstance(snapshot, Mapping):
            raise TypeError("body_runtime.snapshot() must return a mapping")
        return dict(snapshot)

    def _normalize_action(
        self,
        action: Mapping[str, Any] | Any,
        *,
        trace_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        if isinstance(action, Mapping):
            payload = dict(action)
        elif hasattr(action, "to_dict") and callable(action.to_dict):
            payload = dict(action.to_dict())
        elif is_dataclass(action):
            payload = asdict(action)
        else:
            return {"type": "unsupported", "raw_type": type(action).__name__}, trace_id

        nested = payload.get("action")
        if isinstance(nested, Mapping):
            effective_trace_id = trace_id or _optional_string(payload.get("trace_id"))
            return dict(nested), effective_trace_id
        return payload, trace_id or _optional_string(payload.get("trace_id"))

    def _action_type(self, action: Mapping[str, Any]) -> str:
        raw = _string_or_default(
            action.get("type") or action.get("action_type") or action.get("kind"),
            "",
        )
        normalized = raw.strip().lower()
        aliases = {
            "play_speech": "speak",
            "play_speech_action": "speak",
            "speech": "speak",
            "move_head_action": "move_head",
            "pan": "move_head",
            "stop_speech_action": "stop_speech",
            "stop_tts": "stop_speech",
        }
        return aliases.get(normalized, normalized)

    def _dispatch_protocol_action(
        self,
        protocol_action: Any,
        *,
        action_id: str,
        action_type: str,
        trace_id: str | None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        dispatch_actions = getattr(self.body_runtime, "dispatch_actions", None)
        if not callable(dispatch_actions):
            return self._action_outcome(
                action_id=action_id,
                action_type=action_type,
                trace_id=trace_id,
                status="skipped",
                success=False,
                details={
                    **dict(details or {}),
                    "reason": "body_runtime_dispatch_unavailable",
                    "protocol_action": protocol_action.kind,
                },
            )
        try:
            delegate_outcomes = dispatch_actions([protocol_action])
        except Exception as exc:  # pragma: no cover - exercised by integration when hardware fails.
            return self._action_outcome(
                action_id=action_id,
                action_type=action_type,
                trace_id=trace_id,
                status="error",
                success=False,
                details={
                    **dict(details or {}),
                    "error": str(exc),
                    "protocol_action": protocol_action.kind,
                },
            )

        serialized = [_serialize_outcome(outcome) for outcome in delegate_outcomes or []]
        compat_action = None
        if not serialized:
            compat_action = _legacy_eibrain_action(protocol_action)
            if compat_action is not None:
                try:
                    delegate_outcomes = dispatch_actions([compat_action])
                except Exception as exc:  # pragma: no cover - exercised by integration when hardware fails.
                    return self._action_outcome(
                        action_id=action_id,
                        action_type=action_type,
                        trace_id=trace_id,
                        status="error",
                        success=False,
                        details={
                            **dict(details or {}),
                            "error": str(exc),
                            "protocol_action": protocol_action.kind,
                            "compat_protocol_action": getattr(compat_action, "kind", ""),
                        },
                    )
                serialized = [_serialize_outcome(outcome) for outcome in delegate_outcomes or []]
        return self._action_outcome(
            action_id=action_id,
            action_type=action_type,
            trace_id=trace_id,
            status="accepted" if serialized else "skipped",
            success=bool(serialized),
            delegated=True,
            details={
                **dict(details or {}),
                "protocol_action": protocol_action.kind,
                "compat_protocol_action": getattr(compat_action, "kind", "") if compat_action is not None else "",
                "delegate_outcomes": serialized,
            },
        )

    def _capture_frame_outcome(self, *, action_id: str, trace_id: str | None) -> dict[str, Any]:
        capture_frame = getattr(self.body_runtime, "capture_frame", None)
        if callable(capture_frame):
            try:
                frame = capture_frame()
            except Exception as exc:  # pragma: no cover - exercised by integration when camera fails.
                return self._action_outcome(
                    action_id=action_id,
                    action_type="capture_frame",
                    trace_id=trace_id,
                    status="error",
                    success=False,
                    details={"error": str(exc)},
                )
            return self._action_outcome(
                action_id=action_id,
                action_type="capture_frame",
                trace_id=trace_id,
                status="accepted",
                success=True,
                delegated=True,
                details={"frame": _serialize_outcome(frame)},
            )

        latest_visual_frame_path = getattr(self.body_runtime, "latest_visual_frame_path", None)
        if callable(latest_visual_frame_path):
            frame_path = latest_visual_frame_path()
            if frame_path:
                return self._action_outcome(
                    action_id=action_id,
                    action_type="capture_frame",
                    trace_id=trace_id,
                    status="accepted",
                    success=True,
                    delegated=True,
                    details={"frame_path": str(frame_path), "source": "latest_visual_frame_path"},
                )
            return self._action_outcome(
                action_id=action_id,
                action_type="capture_frame",
                trace_id=trace_id,
                status="skipped",
                success=False,
                delegated=True,
                details={"reason": "no_latest_visual_frame"},
            )

        return self._action_outcome(
            action_id=action_id,
            action_type="capture_frame",
            trace_id=trace_id,
            status="unsupported",
            success=False,
            details={"reason": "capture_frame_unavailable"},
        )

    def _action_outcome(
        self,
        *,
        action_id: str,
        action_type: str,
        trace_id: str | None,
        status: str,
        success: bool,
        details: Mapping[str, Any] | None = None,
        delegated: bool = False,
    ) -> dict[str, Any]:
        return {
            "schema": "eihead.execution_outcome.v1",
            "runtime": "eihead",
            "node_role": "head",
            "action_id": action_id,
            "action_type": action_type,
            "trace_id": trace_id or "",
            "status": status,
            "success": success,
            "delegated": delegated,
            "details": dict(details or {}),
        }


def _action_value(action: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in action:
        return action[key]
    params = action.get("params")
    if isinstance(params, Mapping) and key in params:
        return params[key]
    return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _string_or_default(value: Any, default: str) -> str:
    text = _optional_string(value)
    return text if text is not None else default


def _serialize_outcome(outcome: Any) -> dict[str, Any]:
    if isinstance(outcome, Mapping):
        return dict(outcome)
    if hasattr(outcome, "to_dict") and callable(outcome.to_dict):
        return dict(outcome.to_dict())
    if is_dataclass(outcome):
        return asdict(outcome)
    return {"value": outcome}


def _legacy_eibrain_action(action: Any) -> Any | None:
    """Convert local eihead actions for the old in-repo body runtime.

    The standalone eihead package must not require eibrain. During the
    transitional split, however, the existing body organs still check action
    classes with ``isinstance(eibrain.protocol.*)``. Keep this import optional
    so standalone exports run without eibrain while the current monorepo
    delegate remains compatible.
    """

    try:
        from eibrain.protocol.actions import MoveHeadAction as LegacyMoveHeadAction
        from eibrain.protocol.actions import PlaySpeechAction as LegacyPlaySpeechAction
        from eibrain.protocol.actions import StopSpeechAction as LegacyStopSpeechAction
    except ImportError:
        return None

    if isinstance(action, PlaySpeechAction):
        return LegacyPlaySpeechAction(
            ts=action.ts,
            source=action.source,
            text=action.text,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
        )
    if isinstance(action, MoveHeadAction):
        return LegacyMoveHeadAction(
            ts=action.ts,
            source=action.source,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
            target_name=action.target_name,
            target_x=action.target_x,
            target_angle=action.target_angle,
        )
    if isinstance(action, StopSpeechAction):
        return LegacyStopSpeechAction(
            ts=action.ts,
            source=action.source,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
        )
    return None
