from __future__ import annotations


def test_minimax_cli_adapter_invokes_vision_describe() -> None:
    from eibrain.infra.config import MiniMaxCLIConfig
    from eibrain.vision.minimax_cli import MiniMaxCLIAdapter

    captured: dict[str, object] = {}

    def _runner(command: list[str], env: dict[str, str]) -> str:
        captured["command"] = command
        captured["env"] = env
        return '{"summary":"a person near a camera","primary_subject":"person","confidence":0.91}'

    adapter = MiniMaxCLIAdapter(
        MiniMaxCLIConfig(api_key="secret", base_url="https://api.minimaxi.com"),
        runner=_runner,
    )

    result = adapter.understand_image(
        prompt="who is in the frame?",
        image_url="https://example.com/frame.jpg",
    )

    assert result.summary == "a person near a camera"
    assert result.primary_subject == "person"
    assert captured["command"][:4] == [
        "mmx",
        "vision",
        "describe",
        "--image",
    ]
    assert "--prompt" in captured["command"]


def test_minimax_cli_adapter_invokes_search_query() -> None:
    from eibrain.infra.config import MiniMaxCLIConfig
    from eibrain.vision.minimax_cli import MiniMaxCLIAdapter

    captured: dict[str, object] = {}

    def _runner(command: list[str], env: dict[str, str]) -> str:
        captured["command"] = command
        return '{"summary":"search ok","items":["result-1"]}'

    adapter = MiniMaxCLIAdapter(
        MiniMaxCLIConfig(api_key="secret", base_url="https://api.minimaxi.com"),
        runner=_runner,
    )

    result = adapter.search_web("MiniMax AI")

    assert result["summary"] == "search ok"
    assert captured["command"][:3] == ["mmx", "search", "query"]


def test_minimax_cli_adapter_parses_real_cli_content_shape() -> None:
    from eibrain.infra.config import MiniMaxCLIConfig
    from eibrain.vision.minimax_cli import MiniMaxCLIAdapter

    def _runner(command: list[str], env: dict[str, str]) -> str:
        return '{"content":"A close-up portrait of an orange tabby cat looking at the camera outdoors.","base_resp":{"status_code":0,"status_msg":"success"}}'

    adapter = MiniMaxCLIAdapter(
        MiniMaxCLIConfig(api_key="secret", base_url="https://api.minimaxi.com"),
        runner=_runner,
    )

    result = adapter.understand_image(
        prompt="who is in the frame?",
        image_url="https://example.com/frame.jpg",
    )

    assert result.summary.startswith("A close-up portrait")


def test_minimax_cli_adapter_falls_back_to_plain_text_payload() -> None:
    from eibrain.infra.config import MiniMaxCLIConfig
    from eibrain.vision.minimax_cli import MiniMaxCLIAdapter

    def _runner(command: list[str], env: dict[str, str]) -> str:
        return "chair detected near the back wall"

    adapter = MiniMaxCLIAdapter(
        MiniMaxCLIConfig(api_key="secret", base_url="https://api.minimaxi.com"),
        runner=_runner,
    )

    result = adapter.understand_image(
        prompt="who is in the frame?",
        image_url="https://example.com/frame.jpg",
    )

    assert result.summary == "chair detected near the back wall"


def test_minimax_mcp_adapter_falls_back_to_plain_text_payload() -> None:
    from eibrain.infra.config import MiniMaxMCPConfig
    from eibrain.vision.minimax_mcp import MiniMaxMCPAdapter

    def _runner(command: list[str], env: dict[str, str], payload: dict[str, object]):
        return "target detected near window"

    adapter = MiniMaxMCPAdapter(
        MiniMaxMCPConfig(enabled=True, api_key="secret"),
        runner=_runner,
    )

    result = adapter.understand_image(
        prompt="describe frame",
        image_url="https://example.com/frame.jpg",
    )

    assert result.summary == "target detected near window"
