"""Native extension discovery and model name normalization."""

import importlib.util
import logging
import os
import sys
from importlib.machinery import ExtensionFileLoader
from pathlib import Path

from memori.native._onnxruntime import _ensure_onnxruntime_dylib

logger = logging.getLogger(__name__)


def _try_import_memori_python() -> bool:
    _ensure_onnxruntime_dylib()
    env_path = os.environ.get("MEMORI_PYTHON_LIB")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    cargo_target_dir = os.environ.get("CARGO_TARGET_DIR")
    if cargo_target_dir:
        target = Path(cargo_target_dir)
        candidates.extend(
            [
                target / "release" / "libmemori_python.dylib",
                target / "release" / "libmemori_python.so",
                target / "release" / "memori_python.dll",
                target / "debug" / "libmemori_python.dylib",
                target / "debug" / "libmemori_python.so",
                target / "debug" / "memori_python.dll",
            ]
        )

    candidates.extend(
        [
            Path("target/release/libmemori_python.dylib"),
            Path("target/release/libmemori_python.so"),
            Path("target/release/memori_python.dll"),
            Path("target/debug/libmemori_python.dylib"),
            Path("target/debug/libmemori_python.so"),
            Path("target/debug/memori_python.dll"),
            Path("core/target/release/libmemori_python.dylib"),
            Path("core/target/release/libmemori_python.so"),
            Path("core/target/release/memori_python.dll"),
            Path("core/target/debug/libmemori_python.dylib"),
            Path("core/target/debug/libmemori_python.so"),
            Path("core/target/debug/memori_python.dll"),
        ]
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            loader = ExtensionFileLoader("memori_python", str(candidate))
            spec = importlib.util.spec_from_loader("memori_python", loader)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            sys.modules["memori_python"] = module
            logger.debug("Loaded memori_python from %s", candidate)
            return True
        except ImportError:
            continue

    try:
        import memori_python  # noqa: F401  # ty: ignore[unresolved-import]

        logger.debug(
            "Loaded memori_python from %s",
            getattr(memori_python, "__file__", "unknown"),
        )
        return True
    except ImportError:
        return False


def _normalize_model_name(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    normalized = model_name.strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered in {"all-minilm-l6-v2", "allminilml6v2"}:
        return None
    return normalized
