from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from memori.provisioning._models import ProvisionResult


class ProvisionCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_cache_path()

    def get(self, key: str) -> ProvisionResult | None:
        data = self._read()
        raw = data.get(key)
        if not isinstance(raw, dict):
            return None

        try:
            result = ProvisionResult.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            return None

        if result.is_expired():
            data.pop(key, None)
            self._write(data)
            return None
        return result

    def set(self, key: str, result: ProvisionResult) -> None:
        data = self._read()
        data[key] = result.to_dict()
        self._write(data)

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

        if not isinstance(raw, dict):
            return {}
        return raw


def cache_key(
    provider: str, tag: str | None = None, cache_key: str | None = None
) -> str:
    if cache_key:
        return f"{provider}:{cache_key}"
    if tag:
        return f"{provider}:{tag}"
    return provider


def default_cache_path() -> Path:
    home = os.environ.get("MEMORI_HOME") or os.environ.get("HOME")
    if home is None:
        raise RuntimeError("neither MEMORI_HOME nor HOME environment variable is set")
    return Path(home) / ".memori" / "provisioning.json"
