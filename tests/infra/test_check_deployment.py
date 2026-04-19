from __future__ import annotations

from pathlib import Path
import shutil
import sys


def _make_tmp_dir(name: str) -> Path:
    path = Path.cwd() / ".tmp-test-artifacts" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_check_deployment_cli_validates_and_bootstraps(monkeypatch, capsys) -> None:
    repo_source_root = Path(__file__).resolve().parents[2]
    if str(repo_source_root) not in sys.path:
        sys.path.insert(0, str(repo_source_root))
    from apps import check_deployment

    tmp_path = _make_tmp_dir("check-deployment")
    try:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        (repo_root / ".env.example").write_text("ANTHROPIC_MODEL=test\n", encoding="utf-8")
        config_dir = repo_root / "config"
        config_dir.mkdir()
        config_path = config_dir / "eibrain.yaml"
        runtime_dir = tmp_path / "runtime"
        config_path.write_text(
            "\n".join(
                [
                    "deployment:",
                    f"  root_dir: {runtime_dir}",
                    "body:",
                    "  organs:",
                    "    ear:",
                    "      asr:",
                    "        driver:",
                    "          model_dir: ''",
                ]
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(check_deployment, "__file__", str(repo_root / "apps" / "check_deployment.py"))
        monkeypatch.setattr("sys.argv", ["check-deployment", "--config", "config/eibrain.yaml"])

        assert check_deployment.main() == 0
        output = capsys.readouterr().out
        assert "deployment-check=ok" in output
        assert "config=" in output
        assert runtime_dir.exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
