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

    assert text_result == "reply: hello world"
    assert vision_result.startswith("vision-reply:")


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
