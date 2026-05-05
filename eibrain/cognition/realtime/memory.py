"""Inert memory proposal orchestration for realtime cognition."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


class MemoryOrchestrator:
    """Build round-scoped recall/writeback proposals without I/O side effects."""

    def __init__(
        self,
        *,
        memory_service: Any | None = None,
        default_channels: Iterable[str] = ("voice",),
        default_priority: str = "normal",
    ) -> None:
        self.memory_service = memory_service
        self.default_channels = self._channels(default_channels)
        self.default_priority = _clean_text(default_priority) or "normal"

    def build_recall_request(
        self,
        turn: Any,
        *,
        query: str,
        channels: Iterable[str] | None = None,
        priority: str | int | float | None = None,
        reason: str,
        limit: int = 3,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._base_payload(
            turn,
            kind="recall_request",
            query=query,
            channels=channels,
            priority=priority,
            reason=reason,
            metadata=metadata,
        )
        payload.update(
            {
                "limit": int(limit),
                "requester": "realtime_memory_orchestrator",
                "external_call": False,
                "stable": False,
            }
        )
        return self._record(turn, payload)

    def build_writeback_proposal(
        self,
        turn: Any,
        *,
        query: str,
        channels: Iterable[str] | None = None,
        priority: str | int | float | None = None,
        reason: str,
        summary: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        requires_commit: bool = True,
    ) -> dict[str, Any]:
        payload = self._base_payload(
            turn,
            kind="writeback_proposal",
            query=query,
            channels=channels,
            priority=priority,
            reason=reason,
            metadata=metadata,
        )
        payload.update(
            {
                "summary": _clean_text(summary) or _clean_text(query),
                "external_call": False,
                "requires_commit": bool(requires_commit),
                "stable": False,
            }
        )
        return self._record(turn, payload)

    def _base_payload(
        self,
        turn: Any,
        *,
        kind: str,
        query: str,
        channels: Iterable[str] | None,
        priority: str | int | float | None,
        reason: str,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        cleaned_query = _clean_text(query)
        if not cleaned_query:
            raise ValueError("memory proposal query is required")
        cleaned_reason = _clean_text(reason)
        if not cleaned_reason:
            raise ValueError("memory proposal reason is required")
        return _json_ready(
            {
                "kind": kind,
                "round_id": str(_value(turn, "round_id", "")),
                "cancellation_token": _value(turn, "cancellation_token"),
                "query": cleaned_query,
                "channels": self._channels(channels or self.default_channels),
                "priority": self._priority(priority),
                "reason": cleaned_reason,
                "metadata": dict(metadata or {}),
                "source": "memory_orchestrator",
            }
        )

    def _record(self, turn: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
        proposal = _json_ready(dict(payload))
        if hasattr(turn, "append_memory"):
            return turn.append_memory(proposal)
        current = _value(turn, "memory_candidates")
        if isinstance(current, list):
            current.append(proposal)
        elif isinstance(turn, dict):
            turn.setdefault("memory_candidates", []).append(proposal)
        return proposal

    def _priority(self, value: str | int | float | None) -> str | int | float:
        if value is None:
            return self.default_priority
        if isinstance(value, (int, float)):
            return value
        return _clean_text(value) or self.default_priority

    @staticmethod
    def _channels(channels: Iterable[str]) -> list[str]:
        if isinstance(channels, str):
            channels = (channels,)
        normalized: list[str] = []
        for channel in channels:
            cleaned = _clean_text(channel)
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized or ["voice"]


__all__ = ["MemoryOrchestrator"]
