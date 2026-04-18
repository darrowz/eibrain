"""MiniMax MCP adapter for image understanding."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess

from eibrain.infra.config import MiniMaxMCPConfig


@dataclass(slots=True)
class VisionUnderstandingResult:
    summary: str = ""
    primary_subject: str = ""
    confidence: float = 0.0


class MiniMaxMCPAdapter:
    def __init__(self, config: MiniMaxMCPConfig, runner=None) -> None:
        self.config = config
        self.runner = runner or self._default_runner

    def understand_image(self, *, prompt: str, image_url: str) -> VisionUnderstandingResult:
        payload = {
            "tool": "understand_image",
            "arguments": {
                "prompt": prompt,
                "image_url": image_url,
            },
        }
        result = self.runner(self.config.command, self._env(), payload)
        return VisionUnderstandingResult(
            summary=str(result.get("summary", "")),
            primary_subject=str(result.get("primary_subject", "")),
            confidence=float(result.get("confidence", 0.0)),
        )

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["MINIMAX_API_KEY"] = self.config.api_key
        env["MINIMAX_API_HOST"] = self.config.api_host
        if self.config.base_path:
            env["MINIMAX_MCP_BASE_PATH"] = self.config.base_path
        if self.config.resource_mode:
            env["MINIMAX_API_RESOURCE_MODE"] = self.config.resource_mode
        return env

    def _default_runner(self, command: list[str], env: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        return json.loads(completed.stdout or "{}")
