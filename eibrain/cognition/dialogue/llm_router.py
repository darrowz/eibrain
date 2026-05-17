"""Configurable LLM router."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from urllib.error import URLError
from urllib import request

from eibrain.infra.config import LLMConfig


class LLMRouter:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.last_status = "idle"
        self.last_error = ""
        self.last_provider = self.config.provider
        self.last_text = ""
        self.last_elapsed_ms: float | None = None

    def generate(self, prompt: str) -> str:
        started = time.perf_counter()
        self.last_provider = self.config.provider
        self.last_error = ""
        self.last_text = ""
        try:
            if not prompt.strip():
                self.last_status = "empty_prompt"
                return ""
            if self.config.provider == "echo":
                self.last_status = "ok"
                self.last_text = f"reply: {prompt.splitlines()[-1]}"
                return self.last_text
            if self._uses_openclaw_hontu():
                try:
                    self.last_text = self._generate_openclaw_hontu(prompt)
                    self.last_status = "ok" if self.last_text else "empty_response"
                    return self.last_text
                except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                    self.last_status = "error"
                    self.last_error = f"{type(exc).__name__}: {exc}"
            if self._uses_anthropic_api():
                try:
                    self.last_text = self._generate_anthropic_compatible(prompt)
                    self.last_status = "ok" if self.last_text else "empty_response"
                    return self.last_text
                except (URLError, OSError, ValueError, KeyError) as exc:
                    self.last_status = "error"
                    self.last_error = f"{type(exc).__name__}: {exc}"
            if self._uses_chat_api():
                try:
                    self.last_text = self._generate_openai_compatible(prompt)
                    self.last_status = "ok" if self.last_text else "empty_response"
                    return self.last_text
                except (URLError, OSError, ValueError, KeyError) as exc:
                    self.last_status = "error"
                    self.last_error = f"{type(exc).__name__}: {exc}"
            if self.last_status not in {"error", "empty_response"}:
                self.last_status = "unavailable"
            return ""
        finally:
            self.last_elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

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

    def _uses_openclaw_hontu(self) -> bool:
        return self.config.provider == "openclaw_hontu" and bool(self.config.command)

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

    def _generate_openclaw_hontu(self, prompt: str) -> str:
        command = list(self.config.command)
        timeout_s = float(self.config.timeout_s or 30.0)
        cli_timeout = str(int(timeout_s)) if timeout_s.is_integer() else str(timeout_s)
        full_command = self._build_openclaw_command(
            command,
            [
                "agent",
                "--agent",
                self.config.agent_id or "main",
                "--session-id",
                self.config.session_id or "eibrain-honjia-voice",
                "--message",
                prompt,
                "--json",
                "--timeout",
                cli_timeout,
            ],
        )
        completed = subprocess.run(
            full_command,
            text=True,
            capture_output=True,
            timeout=timeout_s + 5.0,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise ValueError(f"openclaw command failed ({completed.returncode}): {stderr}")
        payload = self._parse_openclaw_json(completed.stdout)
        return self._extract_openclaw_text(payload)

    @staticmethod
    def _build_openclaw_command(command: list[str], agent_args: list[str]) -> list[str]:
        if command and command[0] == "ssh":
            remote_start = len(command) - 1
            if "env" in command[1:]:
                remote_start = command.index("env")
            remote_command = [*command[remote_start:], *agent_args]
            quoted_remote = " ".join(shlex.quote(part) for part in remote_command)
            return [*command[:remote_start], quoted_remote]
        return [*command, *agent_args]

    @staticmethod
    def _parse_openclaw_json(output: str) -> dict[str, object]:
        text = output.strip()
        if not text:
            raise ValueError("openclaw command returned empty stdout")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < start:
                raise ValueError("openclaw command returned non-json stdout") from None
            parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("openclaw command returned non-object json")
        return parsed

    @staticmethod
    def _extract_openclaw_text(payload: dict[str, object]) -> str:
        result = payload.get("result")
        if isinstance(result, dict):
            payloads = result.get("payloads")
            if isinstance(payloads, list):
                for item in payloads:
                    if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                        return item["text"].strip()
            if isinstance(result.get("text"), str):
                return str(result["text"]).strip()
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
