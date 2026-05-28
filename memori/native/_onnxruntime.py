"""ONNX Runtime bootstrap for the native Rust extension."""

import hashlib
import logging
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_ORT_VERSION = "1.23.2"
_ORT_LOCK_TIMEOUT_SECONDS = 120.0
_ORT_DOWNLOAD_ATTEMPTS = 3
_ORT_ASSET_BY_PLATFORM: dict[tuple[str, str], tuple[str, str]] = {
    (
        "linux",
        "x86_64",
    ): (
        "onnxruntime-linux-x64-1.23.2.tgz",
        "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
    ),
    (
        "linux",
        "amd64",
    ): (
        "onnxruntime-linux-x64-1.23.2.tgz",
        "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
    ),
    (
        "linux",
        "aarch64",
    ): (
        "onnxruntime-linux-aarch64-1.23.2.tgz",
        "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c",
    ),
    (
        "linux",
        "arm64",
    ): (
        "onnxruntime-linux-aarch64-1.23.2.tgz",
        "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c",
    ),
    (
        "android",
        "aarch64",
    ): (
        "onnxruntime-android-1.23.2.aar",
        "82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827",
    ),
    (
        "android",
        "arm64",
    ): (
        "onnxruntime-android-1.23.2.aar",
        "82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827",
    ),
    (
        "android",
        "x86_64",
    ): (
        "onnxruntime-android-1.23.2.aar",
        "82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827",
    ),
    (
        "android",
        "amd64",
    ): (
        "onnxruntime-android-1.23.2.aar",
        "82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827",
    ),
    (
        "darwin",
        "x86_64",
    ): (
        "onnxruntime-osx-x86_64-1.23.2.tgz",
        "d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6",
    ),
    (
        "darwin",
        "arm64",
    ): (
        "onnxruntime-osx-arm64-1.23.2.tgz",
        "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856",
    ),
    (
        "windows",
        "x86_64",
    ): (
        "onnxruntime-win-x64-1.23.2.zip",
        "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd",
    ),
    (
        "windows",
        "amd64",
    ): (
        "onnxruntime-win-x64-1.23.2.zip",
        "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd",
    ),
    (
        "windows",
        "arm64",
    ): (
        "onnxruntime-win-arm64-1.23.2.zip",
        "1cfe88b6435df3b5fb0e9f6bd7d6f5df1e887b6174de7f6e2a47bab956f3f168",
    ),
}


def _current_platform_system() -> str:
    if sys.platform == "android":
        return "android"
    return platform.system().lower()


def _onnxruntime_asset_for_current_platform() -> tuple[str, str] | None:
    return _ORT_ASSET_BY_PLATFORM.get(
        (_current_platform_system(), platform.machine().lower())
    )


def _onnxruntime_lib_filename() -> str:
    system = _current_platform_system()
    if system == "windows":
        return "onnxruntime.dll"
    if system == "darwin":
        return "libonnxruntime.dylib"
    return "libonnxruntime.so"


def _android_abi_for_machine(machine: str) -> str | None:
    normalized = machine.lower()
    if normalized in {"aarch64", "arm64"}:
        return "arm64-v8a"
    if normalized in {"x86_64", "amd64"}:
        return "x86_64"
    return None


def _resolve_onnxruntime_lib_path(lib_dir: Path) -> Path | None:
    direct_path = lib_dir / _onnxruntime_lib_filename()
    if direct_path.exists():
        return direct_path

    system = _current_platform_system()
    if system == "android":
        abi = _android_abi_for_machine(platform.machine())
        if abi is not None:
            abi_path = lib_dir / "jni" / abi / _onnxruntime_lib_filename()
            if abi_path.exists():
                return abi_path

    if system == "darwin":
        fallback_pattern = "libonnxruntime.*.dylib"
    elif system == "windows":
        fallback_pattern = "onnxruntime*.dll"
    else:
        fallback_pattern = "libonnxruntime.so.*"

    for candidate in sorted(lib_dir.glob(fallback_pattern)):
        if candidate.is_file():
            return candidate
    for candidate in sorted(lib_dir.rglob(fallback_pattern)):
        if candidate.is_file():
            return candidate
    for candidate in sorted(lib_dir.rglob(_onnxruntime_lib_filename())):
        if candidate.is_file():
            return candidate
    return None


def _is_within_directory(directory: Path, candidate: Path) -> bool:
    directory = directory.resolve()
    candidate = candidate.resolve()
    return directory == candidate or directory in candidate.parents


def _extract_onnxruntime_archive(archive_path: Path, destination: Path) -> None:
    if archive_path.suffix in {".zip", ".aar"}:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                if not member.filename:
                    continue
                target = destination / member.filename
                if not _is_within_directory(destination, target):
                    raise RuntimeError("Unsafe path in ONNX Runtime zip archive")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        return
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.name:
                continue
            target = destination / member.name
            if not _is_within_directory(destination, target):
                raise RuntimeError("Unsafe path in ONNX Runtime tar archive")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not (member.isfile() or member.islnk()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _download_urls_for_asset(asset_name: str) -> tuple[str, str]:
    if asset_name.endswith(".aar"):
        maven = (
            "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/"
            f"onnxruntime-android/{_ORT_VERSION}/{asset_name}"
        )
        return (maven, maven)

    github = (
        "https://github.com/microsoft/onnxruntime/releases/download/"
        f"v{_ORT_VERSION}/{asset_name}"
    )
    sourceforge = (
        "https://sourceforge.net/projects/onnx-runtime.mirror/files/"
        f"v{_ORT_VERSION}/{asset_name}/download"
    )
    return github, sourceforge


def _download_asset_with_retries(asset_name: str, destination: Path) -> bool:
    urls = _download_urls_for_asset(asset_name)
    for attempt in range(1, _ORT_DOWNLOAD_ATTEMPTS + 1):
        for url in urls:
            try:
                with requests.get(url, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    with destination.open("wb") as file_handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file_handle.write(chunk)
                return True
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to download %s (attempt %s/%s) from %s",
                    asset_name,
                    attempt,
                    _ORT_DOWNLOAD_ATTEMPTS,
                    url,
                )
    return False


def _acquire_cache_lock(lock_path: Path) -> bool:
    deadline = time.monotonic() + _ORT_LOCK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.2)
    return False


def _release_cache_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _configure_onnxruntime_env(lib_path: Path) -> None:
    os.environ["ORT_DYLIB_PATH"] = str(lib_path)
    if _current_platform_system() == "windows":
        try:
            os.add_dll_directory(str(lib_path.parent))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to add ONNX Runtime directory to DLL search path")


def _ensure_onnxruntime_dylib() -> None:
    current = os.environ.get("ORT_DYLIB_PATH")
    if current and Path(current).exists():
        _configure_onnxruntime_env(Path(current))
        return
    if os.environ.get("MEMORI_ORT_AUTO_DOWNLOAD", "1").lower() in {"0", "false", "no"}:
        return

    asset_info = _onnxruntime_asset_for_current_platform()
    if asset_info is None:
        return
    asset_name, expected_sha = asset_info

    cache_root = Path.home() / ".cache" / "memori" / "onnxruntime" / _ORT_VERSION
    asset_root = (
        asset_name.removesuffix(".tgz").removesuffix(".zip").removesuffix(".aar")
    )
    install_dir = cache_root / asset_root
    lib_path = _resolve_onnxruntime_lib_path(install_dir)
    if lib_path is not None:
        _configure_onnxruntime_env(lib_path)
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / ".download.lock"
    if not _acquire_cache_lock(lock_path):
        logger.warning("Timed out waiting for ONNX Runtime cache lock")
        return
    try:
        existing_lib_path = _resolve_onnxruntime_lib_path(install_dir)
        if existing_lib_path is not None:
            _configure_onnxruntime_env(existing_lib_path)
            return

        with tempfile.NamedTemporaryFile(
            suffix=Path(asset_name).suffix, dir=cache_root, delete=False
        ) as tmp_file:
            archive_path = Path(tmp_file.name)
        try:
            if not _download_asset_with_retries(asset_name, archive_path):
                return
            actual_sha = _compute_sha256(archive_path)
            if actual_sha != expected_sha:
                logger.error(
                    "ONNX Runtime checksum mismatch for %s: expected %s got %s",
                    asset_name,
                    expected_sha,
                    actual_sha,
                )
                return

            extract_root = Path(
                tempfile.mkdtemp(prefix="onnxruntime-extract-", dir=cache_root)
            )
            try:
                _extract_onnxruntime_archive(archive_path, extract_root)
                extracted_dir = extract_root / asset_root
                source_dir = extracted_dir if extracted_dir.exists() else extract_root
                final_dir = install_dir
                if not final_dir.exists():
                    if source_dir == extract_root:
                        shutil.copytree(source_dir, final_dir)
                    else:
                        os.replace(source_dir, final_dir)
            finally:
                shutil.rmtree(extract_root, ignore_errors=True)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to prepare ONNX Runtime binaries")
            return
        finally:
            archive_path.unlink(missing_ok=True)

        resolved_lib_path = _resolve_onnxruntime_lib_path(install_dir)
        if resolved_lib_path is not None:
            _configure_onnxruntime_env(resolved_lib_path)
    finally:
        _release_cache_lock(lock_path)
