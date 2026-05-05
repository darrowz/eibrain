"""Head runtime scaffold.

This module intentionally delegates to ``apps.body_runtime`` for the first
migration pass. Keeping the wrapper small lets us introduce the eihead package
without changing the proven honjia body runtime path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
import time
from typing import Any, Mapping

from eiprotocol.event_routing import classify_event
from eihead.monitoring import build_status_snapshot
from eihead.neck import PanMoveCommand, PanNeckState, plan_pan_move
from eihead.protocol import MoveHeadAction, PlaySpeechAction, StopSpeechAction, serialize_message
from eihead.services import CapabilityRegistry
from .event_journal import EventJournal
from .legacy_body import (
    DEFAULT_BODY_RUNTIME_DELEGATE,
    BodyRuntimeFactory,
    LegacyBodyRuntimeAdapter,
)
from .native_providers import (
    NativeProviderProbe,
    build_native_provider_statuses,
    normalize_native_provider_statuses,
)

DEFAULT_CONFIG_PATH = "config/eibrain.yaml"
DEFAULT_REALTIME_VISION_MAX_AGE_SECONDS = 2.0
DEFAULT_PTZ_MIN_ANGLE_DELTA = 2.0
REALTIME_VISION_ATTRS = (
    "eye_realtime",
    "vision_realtime",
    "realtime_vision",
    "latest_eye_realtime",
    "latest_vision_realtime",
    "latest_realtime_vision",
)
APP_REALTIME_VISION_ATTRS = (
    "eye_realtime",
    "realtime_vision",
    "latest_eye_realtime",
    "latest_vision_realtime",
    "latest_realtime_vision",
)
REALTIME_VISION_CONTAINER_KEYS = ("eye", "vision", "realtime_vision", "body_runtime")
CAPABILITY_NATIVE_PROVIDER_MAP = {
    "camera": "eye",
    "vision_backend": "eye",
    "microphone": "ear",
    "asr": "ear",
    "speaker": "mouth",
    "tts": "mouth",
    "neck": "neck",
}


@dataclass(slots=True)
class HeadRuntimeApp:
    """Compatibility wrapper for the future standalone eihead runtime."""

    body_runtime: Any
    config_path: str = DEFAULT_CONFIG_PATH
    delegate_name: str = DEFAULT_BODY_RUNTIME_DELEGATE
    legacy_body_adapter: LegacyBodyRuntimeAdapter = field(default_factory=LegacyBodyRuntimeAdapter, repr=False)
    realtime_vision_max_age_seconds: float = DEFAULT_REALTIME_VISION_MAX_AGE_SECONDS
    ptz_min_angle_delta: float = DEFAULT_PTZ_MIN_ANGLE_DELTA
    event_journal: EventJournal = field(default_factory=EventJournal, repr=False)
    neck_servo_adapter: Any | None = field(default=None, repr=False)
    neck_pan_state: PanNeckState = field(default_factory=PanNeckState, repr=False)
    native_providers: Mapping[str, Any] | None = field(default=None, repr=False)
    _ptz_last_target_angle: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.native_providers = normalize_native_provider_statuses(
            self.native_providers,
            neck_servo_adapter=self.neck_servo_adapter,
        )

    @classmethod
    def from_config_path(
        cls,
        path: str = DEFAULT_CONFIG_PATH,
        *,
        body_runtime_factory: BodyRuntimeFactory | None = None,
        native_provider_probe: NativeProviderProbe | None = None,
        native_environ: Mapping[str, str] | None = None,
        neck_servo_adapter: Any | None = None,
    ) -> "HeadRuntimeApp":
        adapter = (
            LegacyBodyRuntimeAdapter(body_runtime_factory=body_runtime_factory)
            if body_runtime_factory is not None
            else LegacyBodyRuntimeAdapter()
        )
        native_config = _load_optional_eihead_config(str(path))
        return cls(
            body_runtime=adapter.load_runtime(str(path)),
            config_path=str(path),
            delegate_name=adapter.delegate_name,
            legacy_body_adapter=adapter,
            neck_servo_adapter=neck_servo_adapter,
            native_providers=build_native_provider_statuses(
                config=native_config,
                environ=native_environ,
                probe=native_provider_probe,
                neck_servo_adapter=neck_servo_adapter,
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        body_snapshot, body_snapshot_check = self._body_snapshot_or_error()
        native_providers = dict(self.native_providers or {})
        checks, check_details, status = _runtime_check_summary(
            delegate_name=self.delegate_name,
            native_providers=native_providers,
            body_snapshot_check=body_snapshot_check,
        )
        return {
            "runtime": "eihead",
            "node_role": "head",
            "ok": status == "ok",
            "status": status,
            "config_path": self.config_path,
            "delegate": self.delegate_name,
            "checks": checks,
            "check_details": check_details,
            "native_providers": native_providers,
            "body_runtime": body_snapshot,
        }

    def status(self) -> dict[str, Any]:
        return {
            "command": "status",
            **self.snapshot(),
        }

    def health(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        body_runtime = snapshot.get("body_runtime", {})
        payload: dict[str, Any] = {
            "ok": snapshot.get("ok") is True,
            "status": _string_or_default(snapshot.get("status"), "unknown"),
            "runtime": "eihead",
            "node_role": "head",
            "source": "snapshot",
            "checked_at_ts": float(time.time()),
            "checks": snapshot.get("checks", {}),
            "check_details": snapshot.get("check_details", {}),
            "native_providers": snapshot.get("native_providers", {}),
        }
        if isinstance(body_runtime, Mapping) and "node_id" in body_runtime:
            payload["node_id"] = body_runtime["node_id"]
        return payload

    def capabilities(self) -> dict[str, Any]:
        body_snapshot = self._body_snapshot()
        node_id = _string_or_default(body_snapshot.get("node_id"), "honjia")
        manifest = CapabilityRegistry(
            {"node_id": node_id},
            probe=_capability_live_probe_from_native_providers(self.native_providers or {}),
        ).manifest()
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

    def vision_realtime(self) -> Mapping[str, Any] | Any | None:
        """Return explicit realtime eye payloads only.

        Legacy body snapshots can contain static frame paths and image-derived
        detections. Those are intentionally not promoted here; the monitor
        should say not-wired until a realtime eye adapter exposes a live stream
        hook.
        """

        now_ts = float(time.time())
        for candidate in self._realtime_vision_candidates():
            payload = _resolve_realtime_payload_candidate(candidate)
            if _is_realtime_vision_payload(
                payload,
                now_ts=now_ts,
                max_age_seconds=self.realtime_vision_max_age_seconds,
            ):
                return payload
        return None

    def voice_status(self) -> Mapping[str, Any] | Any | None:
        """Return native voice diagnostics when available, else a body snapshot fallback."""

        for attr_name in (
            "voice_realtime",
            "voice_status",
            "latest_voice_realtime",
            "latest_voice_status",
        ):
            if not hasattr(self.body_runtime, attr_name):
                continue
            source = getattr(self.body_runtime, attr_name)
            payload = _resolve_realtime_payload_candidate(source() if callable(source) else source)
            if payload is not None:
                return payload

        body_snapshot = self._body_snapshot()
        voice_dialogue = body_snapshot.get("voice_dialogue")
        organs = body_snapshot.get("organs") if isinstance(body_snapshot.get("organs"), Mapping) else {}
        ear = organs.get("ear") if isinstance(organs, Mapping) and isinstance(organs.get("ear"), Mapping) else None
        mouth = organs.get("mouth") if isinstance(organs, Mapping) and isinstance(organs.get("mouth"), Mapping) else None
        if isinstance(voice_dialogue, Mapping) or ear is not None or mouth is not None:
            payload: dict[str, Any] = {}
            if isinstance(voice_dialogue, Mapping):
                payload["voice_dialogue"] = dict(voice_dialogue)
            if ear is not None:
                payload["ear"] = dict(ear)
            if mouth is not None:
                payload["mouth"] = dict(mouth)
            return payload
        return None

    def voice_realtime(self) -> Mapping[str, Any] | Any | None:
        return self.voice_status()

    def recent_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return self.event_journal.recent(limit)

    def event_summary(self) -> dict[str, Any]:
        return self.event_journal.summary()

    def handle_event(self, event: Mapping[str, Any] | Any, trace_id: str | None = None) -> dict[str, Any]:
        route = classify_event(event)
        effective_trace_id = trace_id or _event_trace_id(event)
        common = _event_outcome_common(route, trace_id=effective_trace_id)

        if route.get("status") == "invalid":
            outcome = {
                **common,
                "ok": False,
                "accepted": False,
                "processed": False,
                "status": "not_processed",
                "reason": "invalid_event",
                "errors": list(route.get("errors") or []),
            }
            self.event_journal.append(event, outcome, trace_id=effective_trace_id)
            return outcome

        if route.get("status") == "not_processed":
            reason = _string_or_default(route.get("reason"), "unsupported_event_name")
            outcome = {
                **common,
                "ok": False,
                "accepted": False,
                "processed": False,
                "status": "not_processed",
                "reason": reason,
            }
            self.event_journal.append(event, outcome, trace_id=effective_trace_id)
            return outcome

        route_name = _string_or_default(route.get("route"), "")
        if route_name == "action_request":
            action = _action_from_event_route(route)
            action_outcome = self.handle_action(action, trace_id=effective_trace_id)
            accepted = action_outcome.get("status") == "accepted" or action_outcome.get("success") is True
            outcome = {
                **common,
                "ok": bool(action_outcome.get("success")),
                "accepted": bool(accepted),
                "processed": True,
                "status": _string_or_default(action_outcome.get("status"), "unknown"),
                "route": route_name,
                "action_outcome": action_outcome,
            }
            reason = _action_outcome_reason(action_outcome)
            if reason:
                outcome["reason"] = reason
            self.event_journal.append(event, outcome, trace_id=effective_trace_id)
            return outcome

        if route.get("status") == "routed":
            outcome = {
                **common,
                "ok": True,
                "accepted": True,
                "processed": False,
                "status": "recorded",
                "reason": "recorded_for_diagnostics",
                "route": route_name,
            }
            self.event_journal.append(event, outcome, trace_id=effective_trace_id)
            return outcome

        outcome = {
            **common,
            "ok": False,
            "accepted": False,
            "processed": False,
            "status": "not_processed",
            "reason": "unsupported_event_route",
            "route": route_name,
        }
        self.event_journal.append(event, outcome, trace_id=effective_trace_id)
        return outcome

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
            if self.neck_servo_adapter is not None:
                return self._handle_native_neck_action(
                    normalized,
                    action_id=action_id,
                    trace_id=effective_trace_id,
                )

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
            target_angle = _optional_int(target_angle)

            ptz_suppressed_reason = self._maybe_suppress_ptz_jitter(target_angle)
            if ptz_suppressed_reason is not None:
                return self._action_outcome(
                    action_id=action_id,
                    action_type=action_type,
                    trace_id=effective_trace_id,
                    status="skipped",
                    success=False,
                    details=ptz_suppressed_reason,
                )

            protocol_action = MoveHeadAction(
                ts=time.time(),
                source="eihead.runtime",
                session_id=_optional_string(_action_value(normalized, "session_id")),
                actor_id=_optional_string(_action_value(normalized, "actor_id")),
                target_id=_optional_string(_action_value(normalized, "target_id")),
                target_name=_string_or_default(_action_value(normalized, "target_name"), "manual"),
                target_x=_optional_float(_action_value(normalized, "target_x")),
                target_angle=target_angle,
            )
            outcome = self._dispatch_protocol_action(
                protocol_action,
                action_id=action_id,
                action_type=action_type,
                trace_id=effective_trace_id,
                details={"axis": "yaw"},
            )
            if outcome.get("success") and target_angle is not None:
                self._ptz_last_target_angle = target_angle
            return outcome

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

    def _handle_native_neck_action(
        self,
        action: Mapping[str, Any],
        *,
        action_id: str,
        trace_id: str | None,
    ) -> dict[str, Any]:
        axis = _string_or_default(_action_value(action, "axis"), "pan").strip().lower() or "pan"
        target_angle = _action_value(action, "target_angle")
        if target_angle is None:
            target_angle = _action_value(action, "angle")

        plan = plan_pan_move(
            PanMoveCommand(
                axis=axis,
                target_angle=_optional_float(target_angle),
                target_x=_optional_float(_action_value(action, "target_x")),
                source="eihead.runtime",
                action_id=action_id,
                trace_id=trace_id or "",
                metadata=_action_metadata(action),
            ),
            self.neck_pan_state,
        )
        self.neck_pan_state = PanNeckState.from_dict(plan.get("state", {}))

        if not bool(plan.get("success")):
            reason = _string_or_default(plan.get("reason"), _string_or_default(plan.get("status"), "invalid"))
            return self._action_outcome(
                action_id=action_id,
                action_type="move_head",
                trace_id=trace_id,
                status=_string_or_default(plan.get("status"), "invalid"),
                success=False,
                details={
                    "axis": axis,
                    "reason": reason,
                    "neck_plan": plan,
                },
            )

        apply_plan = getattr(self.neck_servo_adapter, "apply_plan", None)
        if not callable(apply_plan):
            return self._action_outcome(
                action_id=action_id,
                action_type="move_head",
                trace_id=trace_id,
                status="skipped",
                success=False,
                details={
                    "axis": "pan",
                    "reason": "neck_servo_adapter_unavailable",
                    "neck_plan": plan,
                },
            )

        try:
            servo_outcome = apply_plan(plan)
        except Exception as exc:  # pragma: no cover - exercised by integration when hardware fails.
            return self._action_outcome(
                action_id=action_id,
                action_type="move_head",
                trace_id=trace_id,
                status="error",
                success=False,
                details={
                    "axis": "pan",
                    "reason": "neck_servo_adapter_error",
                    "error": str(exc),
                    "neck_plan": plan,
                },
            )

        if isinstance(servo_outcome, Mapping):
            servo_details = dict(servo_outcome)
        else:
            servo_details = _serialize_outcome(servo_outcome)
        servo_status = _string_or_default(servo_details.get("status"), "")

        if servo_status == "ok":
            target_angle_value = _optional_float(plan.get("action", {}).get("target_angle"))
            if target_angle_value is not None:
                self.neck_pan_state = replace(
                    self.neck_pan_state,
                    current_angle=target_angle_value,
                    target_angle=target_angle_value,
                )
            status = "accepted"
            success = True
        elif servo_status == "suppressed":
            status = "skipped"
            success = True
        elif servo_status == "unavailable":
            status = "skipped"
            success = False
        else:
            status = servo_status or _string_or_default(plan.get("status"), "skipped")
            success = False

        return self._action_outcome(
            action_id=action_id,
            action_type="move_head",
            trace_id=trace_id,
            status=status,
            success=success,
            delegated=True,
            details={
                "axis": "pan",
                "reason": _native_neck_reason(plan, servo_details),
                "neck_plan": plan,
                "neck_servo": servo_details,
            },
        )

    def _maybe_suppress_ptz_jitter(self, target_angle: int | None) -> dict[str, Any] | None:
        target_angle_int = target_angle
        if target_angle_int is None:
            return None
        min_angle_delta = float(self.ptz_min_angle_delta)
        previous_angle = self._ptz_last_target_angle
        if previous_angle is None:
            return None
        if min_angle_delta <= 0:
            return None
        if abs(target_angle_int - previous_angle) <= min_angle_delta:
            return {
                "axis": "yaw",
                "reason": "ptz_jitter_suppressed",
                "previous_target_angle": previous_angle,
                "target_angle": target_angle_int,
                "min_angle_delta": min_angle_delta,
            }
        return None

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
            "ok": snapshot.get("ok") is True,
            "status": _string_or_default(snapshot.get("status"), "unknown"),
            "checks": snapshot.get("checks", {}),
            "check_details": snapshot.get("check_details", {}),
            "organ_count": organ_count,
            "config_path": self.config_path,
            "delegate": self.delegate_name,
            "native_providers": snapshot.get("native_providers", {}),
            "body_runtime": body_runtime,
        }

    def _body_snapshot_or_error(self) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            return self._body_snapshot(), {"status": "ok"}
        except Exception as exc:
            error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            return (
                {
                    "status": "blocked",
                    "reason": "body_runtime_snapshot_failed",
                    "error": error,
                },
                {
                    "status": "blocked",
                    "reason": "body_runtime_snapshot_failed",
                    "error": error,
                },
            )

    def _body_snapshot(self) -> dict[str, Any]:
        if not hasattr(self.body_runtime, "snapshot"):
            raise TypeError("body_runtime must expose snapshot() for eihead compatibility")
        snapshot = self.body_runtime.snapshot()
        if not isinstance(snapshot, Mapping):
            raise TypeError("body_runtime.snapshot() must return a mapping")
        return dict(snapshot)

    def _realtime_vision_candidates(self) -> list[Any]:
        candidates: list[Any] = []
        candidates.extend(_attr_payload_candidates(self, APP_REALTIME_VISION_ATTRS))
        candidates.extend(_native_provider_realtime_candidates(self.native_providers or {}))
        candidates.extend(_attr_payload_candidates(self.body_runtime, REALTIME_VISION_ATTRS))
        try:
            body_snapshot = self._body_snapshot()
        except Exception:
            body_snapshot = {}
        candidates.extend(_mapping_realtime_candidates(body_snapshot))
        return candidates

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
            compat_action = self.legacy_body_adapter.compat_action(protocol_action)
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


def _runtime_check_summary(
    *,
    delegate_name: str,
    native_providers: Mapping[str, Any],
    body_snapshot_check: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any], str]:
    delegate_check, delegate_details = _delegate_check(delegate_name)
    native_check, native_details = _native_provider_check(native_providers)
    body_check = _string_or_default(body_snapshot_check.get("status"), "unknown")
    checks = {
        "head_runtime_import": "ok",
        "body_runtime_delegate": delegate_check,
        "body_runtime_snapshot": body_check,
        "native_provider_boundaries": native_check,
    }
    check_details = {
        "body_runtime_delegate": delegate_details,
        "body_runtime_snapshot": dict(body_snapshot_check),
        "native_provider_boundaries": native_details,
    }
    return checks, check_details, _overall_runtime_status(checks.values())


def _delegate_check(delegate_name: str) -> tuple[str, dict[str, Any]]:
    if delegate_name == DEFAULT_BODY_RUNTIME_DELEGATE:
        return (
            "degraded",
            {
                "delegate": delegate_name,
                "reason": "legacy_body_runtime_delegate_active",
            },
        )
    if not delegate_name:
        return "unknown", {"delegate": delegate_name, "reason": "delegate_unknown"}
    return "ok", {"delegate": delegate_name}


def _native_provider_check(native_providers: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    provider_states: dict[str, str] = {}
    non_wired: dict[str, str] = {}
    for provider_name, provider_payload in native_providers.items():
        provider_state = _provider_status(provider_payload)
        provider_states[str(provider_name)] = provider_state
        if provider_state != "wired":
            non_wired[str(provider_name)] = provider_state

    if non_wired:
        return (
            "degraded",
            {
                "reason": "native_provider_not_wired",
                "providers": provider_states,
                "non_wired": non_wired,
            },
        )
    return "ok", {"providers": provider_states}


def _provider_status(provider_payload: Any) -> str:
    if isinstance(provider_payload, Mapping):
        return _string_or_default(provider_payload.get("status"), "unknown").strip().lower() or "unknown"
    return "unknown"


def _overall_runtime_status(checks: Any) -> str:
    states = {_string_or_default(state, "unknown").strip().lower() for state in checks}
    if states & {"blocked", "error", "failed"}:
        return "blocked"
    if states & {"degraded", "unknown", "unavailable"}:
        return "degraded"
    return "ok"


def _attr_payload_candidates(source_object: Any, attr_names: tuple[str, ...]) -> list[Any]:
    candidates: list[Any] = []
    for attr_name in attr_names:
        if not hasattr(source_object, attr_name):
            continue
        source = getattr(source_object, attr_name)
        candidates.append(source() if callable(source) else source)
    return candidates


def _native_provider_realtime_candidates(native_providers: Mapping[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    eye_provider = native_providers.get("eye") if isinstance(native_providers, Mapping) else None
    if isinstance(eye_provider, Mapping):
        candidates.extend(_mapping_realtime_candidates(eye_provider))
    return candidates


def _mapping_realtime_candidates(payload: Any) -> list[Any]:
    if not isinstance(payload, Mapping):
        return []
    candidates: list[Any] = []

    for attr_name in REALTIME_VISION_ATTRS:
        if attr_name in payload:
            candidates.append(payload[attr_name])
    if "realtime" in payload:
        candidates.append(payload["realtime"])

    details = payload.get("details")
    if isinstance(details, Mapping):
        candidates.append(details)
        candidates.extend(_mapping_realtime_candidates(details))

    for container_key in REALTIME_VISION_CONTAINER_KEYS:
        container = payload.get(container_key)
        if container_key == "realtime_vision" and container is not None:
            candidates.append(container)
        if isinstance(container, Mapping):
            candidates.append(container)
            candidates.extend(_mapping_realtime_candidates(container))

    organs = payload.get("organs")
    if isinstance(organs, Mapping):
        for organ_key in ("eye", "vision"):
            organ_payload = organs.get(organ_key)
            if isinstance(organ_payload, Mapping):
                candidates.append(organ_payload)
                candidates.extend(_mapping_realtime_candidates(organ_payload))
    return candidates


def _capability_live_probe_from_native_providers(native_providers: Mapping[str, Any]):
    providers = dict(native_providers or {})

    def probe(name: str, *, config: dict[str, Any], static_status: dict[str, Any]) -> dict[str, Any] | None:
        provider_name = CAPABILITY_NATIVE_PROVIDER_MAP.get(name)
        if provider_name is None:
            return None
        provider_payload = providers.get(provider_name)
        if not isinstance(provider_payload, Mapping):
            return None

        native_status = _provider_status(provider_payload)
        hardware_verified = _bool_or_none(provider_payload.get("hardware_verified")) is True
        capability_status = _capability_status_from_native_provider(native_status, hardware_verified=hardware_verified)
        details: dict[str, Any] = {
            "native_provider": provider_name,
            "native_status": native_status,
        }
        native_details = provider_payload.get("details")
        if isinstance(native_details, Mapping):
            details.update({f"native_{key}": value for key, value in native_details.items()})

        return {
            "status": capability_status,
            "source": _string_or_default(provider_payload.get("source"), "native_provider"),
            "reason": _string_or_default(
                provider_payload.get("reason"),
                "native_provider_status",
            ),
            "checked_at": _optional_float(provider_payload.get("checked_at")),
            "last_checked": _optional_float(provider_payload.get("last_checked")),
            "hardware_verified": hardware_verified,
            "provider": provider_payload.get("provider"),
            "details": details,
            "native_provider_status": native_status,
            "static_status": static_status.get("status"),
            "config_kind": config.get("kind"),
        }

    return probe


def _capability_status_from_native_provider(native_status: str, *, hardware_verified: bool) -> str:
    if native_status == "wired":
        return "live" if hardware_verified else "online"
    if native_status in {"degraded", "unavailable", "unknown"}:
        return native_status
    return "unknown"


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _load_optional_eihead_config(path: str) -> Any | None:
    filename = path.replace("\\", "/").rsplit("/", 1)[-1]
    if not filename.startswith("eihead"):
        return None
    try:
        from .config import EiheadConfigError, load_eihead_config

        return load_eihead_config(path)
    except (EiheadConfigError, OSError):
        return None


def _event_outcome_common(route: Mapping[str, Any], *, trace_id: str | None) -> dict[str, Any]:
    return {
        "runtime": "eihead",
        "node_role": "head",
        "trace_id": trace_id or "",
        "event_name": _string_or_default(route.get("eventName"), ""),
        "event_type": _string_or_default(route.get("eventType"), ""),
    }


def _event_trace_id(event: Mapping[str, Any] | Any) -> str | None:
    if isinstance(event, Mapping):
        return _optional_string(event.get("traceId") or event.get("trace_id"))

    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            return _optional_string(payload.get("traceId") or payload.get("trace_id"))

    return _optional_string(getattr(event, "trace_id", None) or getattr(event, "traceId", None))


def _action_from_event_route(route: Mapping[str, Any]) -> dict[str, Any]:
    action_id = _string_or_default(route.get("actionId"), "")
    action_type = _string_or_default(route.get("actionType"), "")
    target = _string_or_default(route.get("target"), "")
    params = _params_with_action_aliases(route.get("params"), action_type=action_type)
    action: dict[str, Any] = {
        "id": action_id,
        "action_id": action_id,
        "type": action_type,
        "action_type": action_type,
        "target": target,
        "params": params,
        "risk_level": _string_or_default(route.get("riskLevel"), ""),
        "idempotency_key": _string_or_default(route.get("idempotencyKey"), ""),
    }
    if target:
        action["target_name"] = target
    return action


def _params_with_action_aliases(params: Any, *, action_type: str) -> dict[str, Any]:
    normalized = dict(params) if isinstance(params, Mapping) else {}
    for key, value in list(normalized.items()):
        snake_key = _camel_to_snake(str(key))
        normalized.setdefault(snake_key, value)

    if action_type == "move_head" and "target_angle" in normalized:
        normalized.setdefault("angle", normalized["target_angle"])
    return normalized


def _camel_to_snake(text: str) -> str:
    result: list[str] = []
    for index, char in enumerate(text):
        if char.isupper() and index > 0 and text[index - 1] != "_":
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _action_outcome_reason(outcome: Mapping[str, Any]) -> str:
    details = outcome.get("details")
    if isinstance(details, Mapping):
        return _string_or_default(details.get("reason"), "")
    return ""


def _action_value(action: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in action:
        return action[key]
    params = action.get("params")
    if isinstance(params, Mapping) and key in params:
        return params[key]
    return default


def _action_metadata(action: Mapping[str, Any]) -> dict[str, Any]:
    metadata = action.get("metadata")
    if isinstance(metadata, Mapping):
        return dict(metadata)
    params = action.get("params")
    if isinstance(params, Mapping) and isinstance(params.get("metadata"), Mapping):
        return dict(params["metadata"])
    return {}


def _native_neck_reason(plan: Mapping[str, Any], servo_details: Mapping[str, Any]) -> str:
    servo_reason = _string_or_default(servo_details.get("reason"), "")
    if servo_reason:
        return servo_reason
    return _string_or_default(plan.get("reason"), "")


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


def _is_realtime_vision_payload(
    payload: Any,
    *,
    now_ts: float,
    max_age_seconds: float,
) -> bool:
    if payload is None:
        return False
    data = _payload_mapping(payload)
    if data is None:
        return False

    kind = _normalized_payload_text(data.get("kind"))
    mode = _normalized_payload_text(data.get("mode"))
    status = _normalized_payload_text(data.get("status"))
    schema = _normalized_payload_text(data.get("schema"))
    source = _normalized_payload_text(data.get("source"))

    if (
        _truthy_payload_flag(data.get("not_wired"))
        or _truthy_payload_flag(data.get("placeholder"))
        or status in {"not_wired", "offline", "missing", "placeholder", "unavailable"}
    ):
        return False
    if (
        kind == "vision_observation"
        or data.get("primary_mode") is False
        or _truthy_payload_flag(data.get("compatibility_mode"))
        or mode in {"compat", "compat/static", "static", "snapshot", "vision_state"}
        or "vision_state" in schema
        or source == "vision_state"
    ):
        return False
    if not (kind == "realtime_vision_observation" or mode in {"realtime", "realtime_stream"}):
        return False
    return _is_realtime_payload_fresh(data, now_ts=now_ts, max_age_seconds=max_age_seconds)


def _resolve_realtime_payload_candidate(payload: Any, *, seen: set[int] | None = None) -> Any:
    if payload is None:
        return None
    seen = seen or set()
    candidate_id = id(payload)
    if candidate_id in seen:
        return payload
    seen.add(candidate_id)

    latest_status = getattr(payload, "latest_status", None)
    if latest_status is not None:
        resolved = _resolve_realtime_payload_candidate(latest_status, seen=seen)
        if resolved is not None:
            return resolved

    for method_name in ("status", "poll"):
        method = getattr(payload, method_name, None)
        if not callable(method):
            continue
        try:
            resolved = _resolve_realtime_payload_candidate(method(), seen=seen)
        except TypeError:
            continue
        if resolved is not None:
            return resolved

    return payload


def _payload_mapping(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, Mapping):
        return dict(payload)
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        data = payload.to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    if is_dataclass(payload):
        return asdict(payload)
    try:
        serialized = serialize_message(payload)
    except TypeError:
        return None
    return dict(serialized) if isinstance(serialized, Mapping) else None


def _is_realtime_payload_fresh(data: Mapping[str, Any], *, now_ts: float, max_age_seconds: float) -> bool:
    if max_age_seconds <= 0:
        return True
    capture_ts = _extract_realtime_capture_timestamp(data)
    if capture_ts is None:
        return True
    if now_ts < capture_ts:
        return True
    return now_ts - capture_ts <= max_age_seconds


def _extract_realtime_capture_timestamp(data: Mapping[str, Any]) -> float | None:
    for key in (
        "last_frame_captured_at_ts",
        "captured_at_ts",
        "timestamp_ms",
        "timestamp",
    ):
        value = _coerce_realtime_timestamp(data.get(key))
        if value is not None:
            return value

    stream = data.get("stream")
    if isinstance(stream, Mapping):
        for key in ("last_frame_captured_at_ts", "captured_at_ts", "timestamp_ms", "timestamp"):
            value = _coerce_realtime_timestamp(stream.get(key))
            if value is not None:
                return value
    health = data.get("health")
    if isinstance(health, Mapping):
        for key in ("last_frame_captured_at_ts", "captured_at_ts", "timestamp_ms", "timestamp"):
            value = _coerce_realtime_timestamp(health.get(key))
            if value is not None:
                return value
    return None


def _coerce_realtime_timestamp(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        timestamp = float(raw)
    except (TypeError, ValueError):
        return None
    absolute_timestamp = abs(timestamp)
    if absolute_timestamp <= 2_000_000_000:
        return timestamp
    if absolute_timestamp <= 2_000_000_000_000:
        return timestamp / 1000.0
    return None


def _normalized_payload_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _truthy_payload_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
