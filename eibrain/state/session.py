"""Session state models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SessionState:
    active_session_id: str | None = None
