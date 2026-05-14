from pathlib import Path

FORBIDDEN = ("from eibrain.cognition", "import eibrain.cognition", "from eibrain.learning", "import eibrain.learning")


def test_body_does_not_import_cognition_or_learning_policy() -> None:
    offenders: list[str] = []
    for path in (Path("eibrain") / "body").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in FORBIDDEN):
            offenders.append(str(path))
    assert offenders == []
