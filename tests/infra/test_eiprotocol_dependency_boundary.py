from pathlib import Path


FORBIDDEN = (
    "eiprotocol/",
    "eiprotocol\\",
    " import eiprotocol as",
)


def test_eibrain_embedded_eiprotocol_is_marked_transitional() -> None:
    readme = Path("docs/eiprotocol-v0.1.1-freeze.md").read_text(encoding="utf-8")
    assert "standalone" in readme.lower()
    readme_text = readme.lower()
    assert "transport agnostic" in readme_text or "transport-agnostic" in readme_text


def test_runtime_code_uses_public_eiprotocol_imports() -> None:
    roots = [Path("eibrain"), Path("eihead")]
    private_imports: list[str] = []

    for root in roots:
        for path in root.rglob("*.py"):
            if path.match("*/protocol/*"):
                continue
            text = path.read_text(encoding="utf-8")
            if "from eiprotocol." in text or "import eiprotocol." in text:
                continue
            if any(token in text for token in FORBIDDEN):
                private_imports.append(str(path))

    assert private_imports == []
