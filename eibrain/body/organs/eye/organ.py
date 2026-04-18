"""Eye organ implementation."""

from __future__ import annotations

from eibrain.body.organs.base import BaseOrgan


class EyeOrgan(BaseOrgan):
    name = "eye"
    subfunction_names = ("camera", "detection", "identity")
