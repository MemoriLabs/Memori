from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import memori_hermes._paths as paths  # noqa: E402
from memori_hermes.install import (  # noqa: E402
    hermes_home,
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
    assert (target / "_paths.py").is_file()
    assert (target / "client.py").is_file()
    assert (target / "tools.py").is_file()
    assert is_installed(hermes_home_path=tmp_path)


def test_installed_plugin_loads_with_hermes_style_namespace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = install_plugin(hermes_home_path=tmp_path)
    module_name = "_hermes_user_memory.memori"
    parent = types.ModuleType("_hermes_user_memory")
    parent.__path__ = []  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "_hermes_user_memory", parent)
    spec = importlib.util.spec_from_file_location(
        module_name,
        target / "__init__.py",
        submodule_search_locations=[str(target)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)

    spec.loader.exec_module(module)

    class Collector:
        provider = None

        def register_memory_provider(self, provider):
            self.provider = provider

    collector = Collector()
    module.register(collector)

    assert collector.provider is not None
    assert collector.provider.name == "memori"


def test_install_plugin_uses_hermes_home_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(paths, "_hermes_home_from_hermes", lambda: None)

    target = install_plugin()

    assert target == tmp_path / "plugins" / "memori"
    assert is_installed()


def test_hermes_home_prefers_explicit_argument_over_resolver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    explicit_home = tmp_path / "explicit-home"
    env_home = tmp_path / "env-home"
    resolver_home = tmp_path / "resolver-home"
    monkeypatch.setenv("HERMES_HOME", str(env_home))
    monkeypatch.setattr(paths, "_hermes_home_from_hermes", lambda: resolver_home)

    assert plugin_target_dir(explicit_home) == explicit_home / "plugins" / "memori"


def test_hermes_home_prefers_hermes_resolver_over_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_home = tmp_path / "env-home"
    resolver_home = tmp_path / "resolver-home"
    monkeypatch.setenv("HERMES_HOME", str(env_home))
    monkeypatch.setattr(paths, "_hermes_home_from_hermes", lambda: resolver_home)

    assert hermes_home() == resolver_home


def test_hermes_home_uses_hermes_resolver_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    resolver_home = tmp_path / "resolver-home"
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(paths, "_hermes_home_from_hermes", lambda: resolver_home)

    assert hermes_home() == resolver_home


def test_hermes_home_uses_windows_platform_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    local_appdata = tmp_path / "AppData" / "Local"
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
    monkeypatch.setattr(paths, "_hermes_home_from_hermes", lambda: None)
    monkeypatch.setattr(paths.sys, "platform", "win32")

    assert hermes_home() == local_appdata / "hermes"


def test_hermes_home_uses_posix_platform_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(paths, "_hermes_home_from_hermes", lambda: None)
    monkeypatch.setattr(paths.sys, "platform", "darwin")

    assert hermes_home() == tmp_path / ".hermes"


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
