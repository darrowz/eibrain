from __future__ import annotations

import json


def test_llm_router_supports_experimental_vision_provider() -> None:
    from eibrain.cognition.dialogue.llm_router import LLMRouter
    from eibrain.infra.config import LLMConfig

    router = LLMRouter(
        LLMConfig(
            provider="minimax",
            model="coding-plan-vlm",
            experimental=True,
            supports_vision=True,
        )
    )

    result = router.generate_vision(
        prompt="describe the image",
        image_urls=["https://example.com/camera-frame.jpg"],
    )

    assert "vision-reply" in result
    assert "camera-frame.jpg" in result


def test_llm_router_supports_anthropic_compatible_text_and_vision(monkeypatch) -> None:
    from eibrain.cognition.dialogue.llm_router import LLMRouter
    from eibrain.infra.config import LLMConfig

    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"content": [{"type": "text", "text": "anthropic-ok"}]}).encode("utf-8")

    def _fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("eibrain.cognition.dialogue.llm_router.request.urlopen", _fake_urlopen)

    router = LLMRouter(
        LLMConfig(
            provider="anthropic_compatible",
            model="MiniMax-M2.7-highspeed",
            endpoint="https://api.minimaxi.com/anthropic",
            api_key="secret",
            supports_vision=True,
        )
    )

    text_result = router.generate("hello")
    vision_result = router.generate_vision("describe", ["https://example.com/frame.jpg"])

    assert text_result == "anthropic-ok"
    assert vision_result == "anthropic-ok"
    assert captured["url"] == "https://api.minimaxi.com/anthropic/v1/messages"
    normalized_headers = {str(key).lower(): value for key, value in captured["headers"].items()}
    assert normalized_headers["x-api-key"] == "secret"
    assert captured["body"]["model"] == "MiniMax-M2.7-highspeed"
    assert captured["body"]["messages"][0]["content"][0]["type"] == "text"
    assert captured["body"]["messages"][0]["content"][1]["type"] == "image"
    assert captured["timeout"] == 30


def test_llm_router_extracts_text_after_thinking_block(monkeypatch) -> None:
    from eibrain.cognition.dialogue.llm_router import LLMRouter
    from eibrain.infra.config import LLMConfig

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "content": [
                        {"type": "thinking", "thinking": "private reasoning"},
                        {"type": "text", "text": "这是一个自然回答"},
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr("eibrain.cognition.dialogue.llm_router.request.urlopen", lambda req, timeout=10: _Response())

    router = LLMRouter(
        LLMConfig(
            provider="anthropic_compatible",
            model="MiniMax-M2.7-highspeed",
            endpoint="https://api.minimaxi.com/anthropic",
            api_key="secret",
        )
    )

    assert router.generate("今天天气怎么样？") == "这是一个自然回答"


def test_llm_router_falls_back_when_remote_provider_errors(monkeypatch) -> None:
    from urllib.error import URLError

    from eibrain.cognition.dialogue.llm_router import LLMRouter
    from eibrain.infra.config import LLMConfig

    def _failing_urlopen(req, timeout=10):
        raise URLError("provider unavailable")

    monkeypatch.setattr("eibrain.cognition.dialogue.llm_router.request.urlopen", _failing_urlopen)

    router = LLMRouter(
        LLMConfig(
            provider="anthropic_compatible",
            model="MiniMax-M2.7-highspeed",
            endpoint="https://api.minimaxi.com/anthropic",
            api_key="secret",
            supports_vision=True,
        )
    )

    text_result = router.generate("hello world")
    vision_result = router.generate_vision("describe", ["https://example.com/frame.jpg"])

    assert text_result == ""
    assert vision_result.startswith("vision-reply:")


def test_llm_router_echo_provider_is_the_only_text_echo() -> None:
    from eibrain.cognition.dialogue.llm_router import LLMRouter
    from eibrain.infra.config import LLMConfig

    router = LLMRouter(LLMConfig(provider="echo"))

    assert router.generate("hello world") == "reply: hello world"


def test_llm_router_uses_openai_compatible_image_url_shape(monkeypatch) -> None:
    import json

    from eibrain.cognition.dialogue.llm_router import LLMRouter
    from eibrain.infra.config import LLMConfig

    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")

    def _fake_urlopen(req, timeout=10):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("eibrain.cognition.dialogue.llm_router.request.urlopen", _fake_urlopen)

    router = LLMRouter(
        LLMConfig(
            provider="openai_compatible",
            model="qwen3.6-plus",
            endpoint="https://example.com/v1/chat/completions",
            api_key="secret",
            supports_vision=True,
        )
    )

    result = router.generate_vision("describe", ["https://example.com/frame.jpg"])

    assert result == "ok"
    content = captured["body"]["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "https://example.com/frame.jpg"


def test_llm_router_calls_openclaw_hontu_agent_command(monkeypatch) -> None:
    from types import SimpleNamespace

    from eibrain.cognition.dialogue.llm_router import LLMRouter
    from eibrain.infra.config import LLMConfig

    captured: dict[str, object] = {}

    def _fake_run(command, input=None, text=None, capture_output=None, timeout=None, encoding=None):
        captured["command"] = command
        captured["input"] = input
        captured["text"] = text
        captured["capture_output"] = capture_output
        captured["timeout"] = timeout
        captured["encoding"] = encoding
        payload = {
            "status": "ok",
            "result": {
                "payloads": [
                    {
                        "text": "语音链路正常",
                    }
                ]
            },
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")

    monkeypatch.setattr("eibrain.cognition.dialogue.llm_router.subprocess.run", _fake_run)

    router = LLMRouter(
        LLMConfig(
            provider="openclaw_hontu",
            command=[
                "ssh",
                "honxin",
                "env",
                "PATH=/home/darrow/n/bin:/usr/local/bin:/usr/bin:/bin",
                "/home/darrow/n/bin/openclaw",
            ],
            agent_id="main",
            session_id="eibrain-honjia-voice",
            timeout_s=45.0,
        )
    )

    result = router.generate("你是鸿途。\n[user] 测试语音链路")

    assert result == "语音链路正常"
    command = captured["command"]
    assert command[:2] == ["ssh", "honxin"]
    assert len(command) == 3
    remote_command = command[-1]
    assert "env PATH=/home/darrow/n/bin:/usr/local/bin:/usr/bin:/bin /home/darrow/n/bin/openclaw agent" in remote_command
    assert "--agent main" in remote_command
    assert "--session-id eibrain-honjia-voice" in remote_command
    assert "'你是鸿途。\n[user] 测试语音链路'" in remote_command
    assert "--json --timeout 45" in remote_command
    assert captured["text"] is True
    assert captured["capture_output"] is True
    assert captured["timeout"] == 50.0
    assert captured["encoding"] == "utf-8"
    assert router.last_status == "ok"
    assert router.last_provider == "openclaw_hontu"
    assert router.last_text == "语音链路正常"


def test_llm_router_records_openclaw_hontu_command_failure(monkeypatch) -> None:
    from types import SimpleNamespace

    from eibrain.cognition.dialogue.llm_router import LLMRouter
    from eibrain.infra.config import LLMConfig

    def _fake_run(command, input=None, text=None, capture_output=None, timeout=None, encoding=None):
        return SimpleNamespace(returncode=255, stdout="", stderr="ssh: connect failed")

    monkeypatch.setattr("eibrain.cognition.dialogue.llm_router.subprocess.run", _fake_run)

    router = LLMRouter(
        LLMConfig(
            provider="openclaw_hontu",
            command=["ssh", "honxin", "/home/darrow/n/bin/openclaw"],
            timeout_s=3.0,
        )
    )

    assert router.generate("hello") == ""
    assert router.last_status == "error"
    assert "openclaw command failed" in router.last_error
