"""Compatibility exports for the optional standalone eiskills package."""

from __future__ import annotations

try:
    from eiskills.base import Skill
except ModuleNotFoundError:  # pragma: no cover - compatibility path when eiskills is absent
    class Skill:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def __call__(self, *args, **kwargs):  # pragma: no cover - marker for compatibility
            return self


__all__ = ["Skill"]

