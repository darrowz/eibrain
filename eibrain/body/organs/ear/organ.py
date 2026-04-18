"""Ear organ implementation."""

from __future__ import annotations

from eibrain.body.organs.base import BaseOrgan


class EarOrgan(BaseOrgan):
    name = "ear"
    subfunction_names = ("capture", "vad", "asr")
