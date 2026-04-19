from __future__ import annotations


def test_command_driver_heartbeat_degrades_when_binary_missing() -> None:
    from eibrain.body.drivers.command import CommandDriver
    from eibrain.infra.config import DriverConfig

    driver = CommandDriver(
        DriverConfig(
            kind="command",
            command=["missing-binary"],
            extra={"health_command": ["missing-binary", "--health"]},
        )
    )

    result = driver.heartbeat()

    assert result.status == "unavailable"
    assert result.details["reason"] == "command_not_found"
    assert "elapsed_ms" in result.details


def test_command_driver_invoke_returns_error_when_binary_missing() -> None:
    from eibrain.body.drivers.command import CommandDriver
    from eibrain.infra.config import DriverConfig

    driver = CommandDriver(
        DriverConfig(
            kind="command",
            command=["missing-binary"],
        )
    )

    result = driver.invoke("move_head", {"target_name": "speaker"})

    assert result.status == "error"
    assert result.details["reason"] == "command_not_found"
    assert "elapsed_ms" in result.details
