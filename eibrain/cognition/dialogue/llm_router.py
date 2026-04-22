"""Configurable LLM router."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib import request

from eibrain.infra.config import LLMConfig


class LLMRouter:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            return ""
        if self.config.provider == "echo":
            return f"reply: {prompt.splitlines()[-1]}"
        if self._uses_anthropic_api():
            try:
                return self._generate_anthropic_compatible(prompt)
            except (URLError, OSError, ValueError, KeyError):
                pass
        if self._uses_chat_api():
            try:
                return self._generate_openai_compatible(prompt)
            except (URLError, OSError, ValueError, KeyError):
                pass
        return ""

    def generate_vision(self, prompt: str, image_urls: list[str]) -> str:
        if not prompt.strip():
            return ""
        if not image_urls:
            return self.generate(prompt)
        if self._uses_anthropic_api() and self.config.supports_vision:
            try:
                return self._generate_anthropic_compatible(prompt, image_urls=image_urls)
            except (URLError, OSError, ValueError, KeyError):
                pass
        if self._uses_chat_api() and self.config.supports_vision:
            try:
                return self._generate_openai_compatible(prompt, image_urls=image_urls)
            except (URLError, OSError, ValueError, KeyError):
                pass
        joined = ", ".join(image_urls)
        return f"vision-reply: {prompt} [{joined}]"

    def _uses_chat_api(self) -> bool:
        return self.config.provider in {"openai_compatible", "minimax", "qwen"} and bool(self.config.endpoint)

    def _uses_anthropic_api(self) -> bool:
        return self.config.provider == "anthropic_compatible" and bool(self.config.endpoint)

    def _anthropic_messages_url(self) -> str:
        endpoint = self.config.endpoint.rstrip("/")
        if endpoint.endswith("/v1/messages"):
            return endpoint
        return f"{endpoint}/v1/messages"

    def _generate_anthropic_compatible(self, prompt: str, image_urls: list[str] | None = None) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if image_urls:
            for image_url in image_urls:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url,
                        },
                    }
                )

        body = json.dumps(
            {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.config.temperature,
            }
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }
        req = request.Request(self._anthropic_messages_url(), data=body, method="POST", headers=headers)
        with request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return self._extract_anthropic_text(payload)

    @staticmethod
    def _extract_anthropic_text(payload: dict[str, object]) -> str:
        parts = payload.get("content", [])
        if parts and isinstance(parts, list):
            text_parts = [
                str(item.get("text", ""))
                for item in parts
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
            ]
            if text_parts:
                return "".join(text_parts).strip()
            for item in parts:
                if isinstance(item, dict) and item.get("type") == "text":
                    return str(item.get("text", ""))
            for item in parts:
                if isinstance(item, dict) and item.get("text"):
                    return str(item.get("text", ""))
        if isinstance(payload.get("text"), str):
            return str(payload["text"]).strip()
        return ""

    def _generate_openai_compatible(self, prompt: str, image_urls: list[str] | None = None) -> str:
        content: list[dict[str, str]] | str
        if image_urls:
            content = [{"type": "text", "text": prompt}]
            content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
        else:
            content = prompt
        body = json.dumps(
            {
                "model": self.config.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = request.Request(self.config.endpoint, data=body, method="POST", headers=headers)
        with request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])
