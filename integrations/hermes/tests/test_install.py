from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memori_hermes.install import (  # noqa: E402
    install_plugin,
    is_installed,
    main,
    plugin_target_dir,
    uninstall_plugin,
)


def test_install_plugin_copies_provider_to_hermes_plugins(tmp_path: Path) -> None:
    target = install_plugin(hermes_home_path=tmp_path)

    assert target == tmp_path / "plugins" / "memori"
    assert (target / "__init__.py").is_file()
    assert (target / "plugin.yaml").is_file()
    assert (target / "client.py").is_file()
    assert (target / "tools.py").is_file()
    assert is_installed(hermes_home_path=tmp_path)


def test_install_plugin_uses_hermes_home_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    target = install_plugin()

    assert target == tmp_path / "plugins" / "memori"
    assert is_installed()


def test_install_plugin_requires_force_for_existing_target(tmp_path: Path) -> None:
    install_plugin(hermes_home_path=tmp_path)

    try:
        install_plugin(hermes_home_path=tmp_path)
    except FileExistsError as exc:
        assert "--force" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileExistsError")


def test_install_plugin_force_replaces_existing_target(tmp_path: Path) -> None:
    target = install_plugin(hermes_home_path=tmp_path)
    marker = target / "old.txt"
    marker.write_text("old", encoding="utf-8")

    install_plugin(hermes_home_path=tmp_path, force=True)

    assert not marker.exists()
    assert is_installed(hermes_home_path=tmp_path)


def test_install_plugin_excludes_cache_files(tmp_path: Path) -> None:
    target = install_plugin(hermes_home_path=tmp_path)

    assert not list(target.rglob("__pycache__"))
    assert not list(target.rglob("*.pyc"))


def test_uninstall_plugin_removes_memori_directory_only(tmp_path: Path) -> None:
    target = install_plugin(hermes_home_path=tmp_path)
    sibling = tmp_path / "plugins" / "other"
    sibling.mkdir()

    removed = uninstall_plugin(hermes_home_path=tmp_path)

    assert removed == target
    assert not target.exists()
    assert sibling.exists()


def test_status_command_returns_zero_when_installed(tmp_path: Path, capsys) -> None:
    install_plugin(hermes_home_path=tmp_path)

    result = main(["--hermes-home", str(tmp_path), "status"])

    captured = capsys.readouterr()
    assert result == 0
    assert "is installed" in captured.out


def test_status_command_returns_one_when_missing(tmp_path: Path, capsys) -> None:
    result = main(["--hermes-home", str(tmp_path), "status"])

    captured = capsys.readouterr()
    assert result == 1
    assert "is not installed" in captured.out


def test_install_command_defaults_to_install(tmp_path: Path, capsys) -> None:
    result = main(["--hermes-home", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "Installed Memori Hermes provider" in captured.out
    assert is_installed(hermes_home_path=tmp_path)


def test_plugin_target_dir_expands_user(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/tmp/hermes-memori-home")

    target = plugin_target_dir("~/custom-hermes")

    assert str(target) == os.path.expanduser("~/custom-hermes/plugins/memori")
