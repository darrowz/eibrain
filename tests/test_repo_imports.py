from importlib import import_module


def test_core_packages_are_importable() -> None:
    expected_modules = [
        "eibrain",
        "apps.body_runtime",
        "apps.cognitive_runtime",
    ]

    for module_name in expected_modules:
        module = import_module(module_name)
        assert module is not None
