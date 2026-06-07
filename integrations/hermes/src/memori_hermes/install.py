"""Installer CLI for the Memori Hermes memory provider."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from . import _paths
except ImportError:  # pragma: no cover - supports direct file execution.
    import _paths  # type: ignore[no-redef]

PLUGIN_NAME = _paths.PLUGIN_NAME

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def hermes_home() -> Path:
    """Return the Hermes home directory used for user-installed plugins."""
    return _paths.resolve_hermes_home()


def plugin_source_dir() -> Path:
    """Return the installed memori_hermes package directory."""
    return Path(__file__).resolve().parent


def plugin_target_dir(hermes_home_path: str | Path | None = None) -> Path:
    """Return the Hermes memory plugin destination for Memori."""
    return _paths.plugin_target_dir(hermes_home_path)


def _ignore_copy_names(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(name)
        if name in EXCLUDED_DIRS or path.suffix in EXCLUDED_SUFFIXES:
            ignored.add(name)
    return ignored


def install_plugin(
    *,
    hermes_home_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Install the Memori provider into Hermes' user plugin directory."""
    source = plugin_source_dir()
    target = plugin_target_dir(hermes_home_path)

    if target.exists():
        if not force:
            raise FileExistsError(
                f"{target} already exists. Re-run with --force to replace it."
            )
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=_ignore_copy_names)
    return target


def uninstall_plugin(*, hermes_home_path: str | Path | None = None) -> Path:
    """Remove the Memori provider from Hermes' user plugin directory."""
    target = plugin_target_dir(hermes_home_path)
    if target.exists():
        shutil.rmtree(target)
    return target


def is_installed(*, hermes_home_path: str | Path | None = None) -> bool:
    """Return whether the Memori provider is installed for Hermes discovery."""
    target = plugin_target_dir(hermes_home_path)
    return (target / "__init__.py").is_file() and (target / "plugin.yaml").is_file()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-memori",
        description="Install the Memori memory provider for Hermes Agent.",
    )
    parser.add_argument(
        "--hermes-home",
        help=(
            "Hermes home directory. Defaults to HERMES_HOME, Hermes' own "
            "resolver, or the platform default."
        ),
    )

    subparsers = parser.add_subparsers(dest="command")

    install = subparsers.add_parser(
        "install",
        help="Install Memori into Hermes' memory provider plugin directory.",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing Memori plugin directory.",
    )

    subparsers.add_parser(
        "uninstall",
        help="Remove Memori from Hermes' memory provider plugin directory.",
    )
    subparsers.add_parser(
        "status",
        help="Show whether Memori is installed for Hermes memory discovery.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the hermes-memori installer CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command or "install"

    try:
        if command == "install":
            target = install_plugin(
                hermes_home_path=args.hermes_home,
                force=getattr(args, "force", False),
            )
            print(f"Installed Memori Hermes provider to {target}")
            print("Next steps:")
            print("  hermes config set memory.provider memori")
            print("  hermes memory setup")
            print("  hermes memory status")
            return 0

        if command == "uninstall":
            target = uninstall_plugin(hermes_home_path=args.hermes_home)
            print(f"Removed Memori Hermes provider from {target}")
            return 0

        if command == "status":
            target = plugin_target_dir(args.hermes_home)
            if is_installed(hermes_home_path=args.hermes_home):
                print(f"Memori Hermes provider is installed at {target}")
                return 0
            print(f"Memori Hermes provider is not installed at {target}")
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
