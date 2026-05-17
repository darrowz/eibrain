"""Configurable LLM router."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
from urllib.error import URLError
from urllib import request

from eibrain.infra.config import LLMConfig


_GATEWAY_DEVICE_IDENTITY_JS = r"""
const fs = require("fs");
const os = require("os");
const path = require("path");
const crypto = require("crypto");

const ED25519_SPKI_PREFIX = Buffer.from("302a300506032b6570032100", "hex");
const input = JSON.parse(fs.readFileSync(0, "utf8"));

function base64UrlEncode(buf) {
  return buf.toString("base64").replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/g, "");
}

function derivePublicKeyRaw(publicKeyPem) {
  const spki = crypto.createPublicKey(publicKeyPem).export({ type: "spki", format: "der" });
  if (spki.length === ED25519_SPKI_PREFIX.length + 32 && spki.subarray(0, ED25519_SPKI_PREFIX.length).equals(ED25519_SPKI_PREFIX)) {
    return spki.subarray(ED25519_SPKI_PREFIX.length);
  }
  return spki;
}

function fingerprintPublicKey(publicKeyPem) {
  return crypto.createHash("sha256").update(derivePublicKeyRaw(publicKeyPem)).digest("hex");
}

function loadOrCreateIdentity(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      const parsed = JSON.parse(fs.readFileSync(filePath, "utf8"));
      if (parsed?.version === 1 && typeof parsed.deviceId === "string" && typeof parsed.publicKeyPem === "string" && typeof parsed.privateKeyPem === "string") {
        return parsed;
      }
    }
  } catch {}
  const { publicKey, privateKey } = crypto.generateKeyPairSync("ed25519");
  const publicKeyPem = publicKey.export({ type: "spki", format: "pem" });
  const privateKeyPem = privateKey.export({ type: "pkcs8", format: "pem" });
  const identity = {
    version: 1,
    deviceId: fingerprintPublicKey(publicKeyPem),
    publicKeyPem,
    privateKeyPem,
    createdAtMs: Date.now(),
  };
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(identity, null, 2)}\n`, { mode: 0o600 });
  try { fs.chmodSync(filePath, 0o600); } catch {}
  return identity;
}

function normalizeMetadata(value) {
  return typeof value === "string" && value.trim() ? value.trim().replace(/[A-Z]/g, c => String.fromCharCode(c.charCodeAt(0) + 32)) : "";
}

const identityPath = input.identityPath || path.join(os.homedir(), ".config", "eibrain", "openclaw-device.json");
const identity = loadOrCreateIdentity(identityPath);
const signedAtMs = Date.now();
const scopes = Array.isArray(input.scopes) ? input.scopes : [];
const payload = [
  "v3",
  identity.deviceId,
  input.clientId || "cli",
  input.clientMode || "cli",
  input.role || "operator",
  scopes.join(","),
  String(signedAtMs),
  input.token || "",
  input.nonce || "",
  normalizeMetadata(input.platform),
  normalizeMetadata(input.deviceFamily),
].join("|");
const signature = base64UrlEncode(crypto.sign(null, Buffer.from(payload, "utf8"), crypto.createPrivateKey(identity.privateKeyPem)));
process.stdout.write(JSON.stringify({
  id: identity.deviceId,
  publicKey: base64UrlEncode(derivePublicKeyRaw(identity.publicKeyPem)),
  signature,
  signedAt: signedAtMs,
  nonce: input.nonce || "",
}));
"""


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
            if self._uses_openclaw_gateway_ws():
                try:
                    self.last_text = self._generate_openclaw_gateway_ws(prompt)
                    self.last_status = "ok" if self.last_text else "empty_response"
                    return self.last_text
                except (OSError, ValueError, TimeoutError, KeyError) as exc:
                    self.last_status = "error"
                    self.last_error = f"{type(exc).__name__}: {exc}"
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

    def _uses_openclaw_gateway_ws(self) -> bool:
        return self.config.provider == "openclaw_gateway_ws" and bool(self.config.endpoint and self.config.api_key)

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
        agent_args = [
            "agent",
            "--agent",
            self.config.agent_id or "main",
            "--session-id",
            self.config.session_id or "eibrain-honjia-voice",
            "--message",
            prompt,
            "--json",
        ]
        if self.config.thinking:
            agent_args.extend(["--thinking", self.config.thinking])
        agent_args.extend(["--timeout", cli_timeout])
        full_command = self._build_openclaw_command(
            command,
            agent_args,
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

    def _generate_openclaw_gateway_ws(self, prompt: str) -> str:
        timeout_s = float(self.config.timeout_s or 30.0)
        websocket = self._open_gateway_websocket(self.config.endpoint, timeout_s)
        try:
            challenge = self._wait_gateway_challenge(websocket, timeout_s)
            challenge_payload = challenge.get("payload", {})
            nonce = challenge_payload.get("nonce", "") if isinstance(challenge_payload, dict) else ""
            if not isinstance(nonce, str) or not nonce:
                raise ValueError("openclaw gateway challenge did not include a nonce")
            scopes = ["operator.write"]
            device = self._build_gateway_device(nonce=nonce, scopes=scopes)
            self._gateway_request(
                websocket,
                "connect",
                {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": "cli",
                        "displayName": "eibrain:honjia-voice",
                        "version": "eibrain",
                        "platform": "linux",
                        "mode": "cli",
                        "instanceId": str(uuid.uuid4()),
                    },
                    "caps": [],
                    "auth": {"token": self.config.api_key},
                    "role": "operator",
                    "scopes": scopes,
                    "device": device,
                },
                timeout_s,
            )
            agent_params: dict[str, object] = {
                "message": prompt,
                "agentId": self.config.agent_id or "main",
                "sessionId": self.config.session_id or "eibrain-honjia-voice",
                "idempotencyKey": str(uuid.uuid4()),
                "timeout": int(timeout_s),
            }
            if self.config.thinking:
                agent_params["thinking"] = self.config.thinking
            payload = self._gateway_request(
                websocket,
                "agent",
                agent_params,
                timeout_s + 5.0,
                expect_final=True,
            )
            return self._extract_openclaw_text(payload)
        finally:
            close = getattr(websocket, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _open_gateway_websocket(url: str, timeout_s: float):
        try:
            import websocket
        except ImportError as exc:
            raise OSError("websocket-client is required for openclaw_gateway_ws provider") from exc
        return websocket.create_connection(url, timeout=timeout_s, suppress_origin=True)

    def _build_gateway_device(self, *, nonce: str, scopes: list[str]) -> dict[str, object]:
        node_command = os.environ.get("EIBRAIN_GATEWAY_DEVICE_NODE", "node")
        payload = {
            "nonce": nonce,
            "scopes": scopes,
            "token": self.config.api_key,
            "clientId": "cli",
            "clientMode": "cli",
            "role": "operator",
            "platform": "linux",
            "identityPath": os.path.expanduser("~/.config/eibrain/openclaw-device.json"),
        }
        completed = subprocess.run(
            [node_command, "-e", _GATEWAY_DEVICE_IDENTITY_JS],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=5.0,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise OSError(f"openclaw gateway device identity failed ({completed.returncode}): {stderr}")
        parsed = json.loads(completed.stdout)
        if not isinstance(parsed, dict):
            raise ValueError("openclaw gateway device identity returned non-object json")
        return parsed

    def _wait_gateway_challenge(self, websocket, timeout_s: float) -> dict[str, object]:
        deadline = time.monotonic() + timeout_s
        while True:
            frame = self._read_gateway_frame(websocket, deadline)
            if frame.get("type") == "event" and frame.get("event") == "connect.challenge":
                return frame

    def _gateway_request(
        self,
        websocket,
        method: str,
        params: dict[str, object],
        timeout_s: float,
        *,
        expect_final: bool = False,
    ) -> dict[str, object]:
        request_id = str(uuid.uuid4())
        websocket.send(
            json.dumps(
                {
                    "type": "req",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
                ensure_ascii=False,
            )
        )
        return self._wait_gateway_response(websocket, request_id, method, timeout_s, expect_final=expect_final)

    def _wait_gateway_response(
        self,
        websocket,
        request_id: str,
        method: str,
        timeout_s: float,
        *,
        expect_final: bool = False,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout_s
        while True:
            frame = self._read_gateway_frame(websocket, deadline)
            if frame.get("type") != "res" or frame.get("id") != request_id:
                continue
            if not frame.get("ok"):
                raise ValueError(f"openclaw gateway {method} failed: {self._gateway_error_message(frame)}")
            payload = frame.get("payload", {})
            if not isinstance(payload, dict):
                raise ValueError(f"openclaw gateway {method} returned non-object payload")
            if expect_final and payload.get("status") == "accepted":
                continue
            return payload

    @staticmethod
    def _read_gateway_frame(websocket, deadline: float) -> dict[str, object]:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError("openclaw gateway response timed out")
        settimeout = getattr(websocket, "settimeout", None)
        if callable(settimeout):
            settimeout(max(0.1, remaining_s))
        raw = websocket.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        frame = json.loads(str(raw))
        if not isinstance(frame, dict):
            raise ValueError("openclaw gateway returned non-object frame")
        return frame

    @staticmethod
    def _gateway_error_message(frame: dict[str, object]) -> str:
        error = frame.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
            if message:
                return str(message)
        if error:
            return str(error)
        return "unknown error"

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
