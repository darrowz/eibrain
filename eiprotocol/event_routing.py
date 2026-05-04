"""Pure eiprotocol event envelope routing helpers."""

from __future__ import annotations

from typing import Any, Mapping

from .models import validate_event


_SUPPORTED_EVENT_ROUTES = {
    "ei.capability.manifest.report": "capability_manifest",
    "ei.dialogue.asr.partial": "asr_partial",
    "ei.dialogue.asr.final": "asr_final",
    "ei.observation.vision.frame": "realtime_vision_frame",
    "ei.action.request": "action_request",
    "ei.outcome.execution": "execution_outcome",
    "ei.outcome.user.feedback": "user_feedback",
}

_ACTION_CONTENT_FIELDS = (
    "actionId",
    "actionType",
    "target",
    "params",
    "riskLevel",
    "idempotencyKey",
)


def classify_event(event: Any) -> dict[str, Any]:
    """Classify an eiprotocol envelope into a JSON-friendly route description."""
    payload, coercion_errors = _coerce_event(event)
    if coercion_errors:
        return _invalid_route(payload, coercion_errors)

    validation_errors = validate_event(payload)
    if validation_errors:
        return _invalid_route(payload, validation_errors)

    event_name = _text(payload.get("name"))
    event_type = _text(payload.get("type"))
    route = _SUPPORTED_EVENT_ROUTES.get(event_name)
    if route is None:
        return {
            "status": "not_processed",
            "reason": "unsupported_event_name",
            "eventName": event_name,
            "eventType": event_type,
        }

    route_description: dict[str, Any] = {
        "status": "routed",
        "route": route,
        "eventName": event_name,
        "eventType": event_type,
    }
    if route == "action_request":
        route_description.update(_action_fields(payload))
    return route_description


def _coerce_event(event: Any) -> tuple[dict[str, Any], list[str]]:
    if isinstance(event, Mapping):
        return dict(event), []

    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception as exc:  # pragma: no cover - defensive for EventEnvelope-like inputs.
            return {}, [f"to_dict failed: {exc}"]
        if isinstance(payload, Mapping):
            return dict(payload), []
        return {}, ["to_dict must return a mapping"]

    return {}, ["event must be a mapping or provide to_dict()"]


def _invalid_route(payload: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    return {
        "status": "invalid",
        "reason": "invalid_event",
        "eventName": _text(payload.get("name")),
        "eventType": _text(payload.get("type")),
        "errors": list(errors),
    }


def _action_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, Mapping):
        return {field: "" for field in _ACTION_CONTENT_FIELDS}

    params = content.get("params")
    risk_level = content.get("riskLevel")
    if not risk_level:
        policy = payload.get("policy")
        if isinstance(policy, Mapping):
            risk_level = policy.get("riskLevel")

    return {
        "actionId": _text(content.get("actionId")),
        "actionType": _text(content.get("actionType")),
        "target": _text(content.get("target")),
        "params": dict(params) if isinstance(params, Mapping) else {},
        "riskLevel": _text(risk_level),
        "idempotencyKey": _text(content.get("idempotencyKey")),
    }


def _text(value: Any) -> str:
    return str(value or "")


__all__ = ["classify_event"]
