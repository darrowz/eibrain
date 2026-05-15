#!/usr/bin/env python3
"""Retired eihead exporter shim."""

from __future__ import annotations

from typing import Sequence


RETIRED_MESSAGE = (
    "scripts/export-eihead-repo.py is retired. "
    "Use the canonical standalone repository at "
    "D:/github/ei-workspace/repos/eihead."
)


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    raise SystemExit(RETIRED_MESSAGE)


if __name__ == "__main__":
    main()
