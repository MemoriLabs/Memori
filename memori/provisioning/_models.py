from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ProvisionResult:
    provider: str
    family: str
    dsn: str
    connect_args: dict[str, Any] = field(default_factory=dict)
    claim_url: str | None = None
    expires_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvisionResult:
        return cls(
            provider=str(data["provider"]),
            family=str(data["family"]),
            dsn=str(data["dsn"]),
            connect_args=dict(data.get("connect_args") or {}),
            claim_url=data.get("claim_url"),
            expires_at=data.get("expires_at"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "family": self.family,
            "dsn": self.dsn,
            "connect_args": self.connect_args,
            "claim_url": self.claim_url,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        if not isinstance(self.expires_at, str):
            return True

        parsed = _parse_datetime(self.expires_at)
        if parsed is None:
            return True

        resolved_now = now or datetime.now(timezone.utc)
        if resolved_now.tzinfo is None:
            resolved_now = resolved_now.replace(tzinfo=timezone.utc)

        return parsed <= resolved_now


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
