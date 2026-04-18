"""MiniMax CLI adapter for image understanding and search."""

from __future__ import annotations

import json
import os
import subprocess

from eibrain.infra.config import MiniMaxCLIConfig
from eibrain.vision.minimax_mcp import VisionUnderstandingResult


class MiniMaxCLIAdapter:
    def __init__(self, config: MiniMaxCLIConfig, runner=None) -> None:
        self.config = config
        self.runner = runner or self._default_runner

    def understand_image(self, *, prompt: str, image_url: str) -> VisionUnderstandingResult:
        payload = self.runner(
            self.config.command
            + [
                "vision",
                "describe",
                "--image",
                image_url,
                "--prompt",
                prompt,
                "--output",
                "json",
                "--non-interactive",
            ],
            self._env(),
        )
        result = json.loads(payload or "{}")
        return VisionUnderstandingResult(
            summary=str(result.get("summary") or result.get("content", "")),
            primary_subject=str(result.get("primary_subject", "")),
            confidence=float(result.get("confidence", 0.0)),
        )

    def search_web(self, query: str) -> dict[str, object]:
        payload = self.runner(
            self.config.command
            + [
                "search",
                "query",
                "--q",
                query,
                "--output",
                "json",
                "--non-interactive",
            ],
            self._env(),
        )
        return json.loads(payload or "{}")

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["MINIMAX_API_KEY"] = self.config.api_key
        env["MINIMAX_API_HOST"] = self.config.base_url
        return env

    def _default_runner(self, command: list[str], env: dict[str, str]) -> str:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        return completed.stdout
