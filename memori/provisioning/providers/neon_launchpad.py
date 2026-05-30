from __future__ import annotations

import os
from typing import Any

import requests

from memori.provisioning._models import ProvisionResult
from memori.provisioning._registry import Registry

DEFAULT_NEON_LAUNCHPAD_URL = "https://neon.new/api/v1/database"


@Registry.register_provider("neon-launchpad")
def provision_neon_launchpad(
    *,
    tag: str = "memori",
    timeout: int = 30,
    url: str | None = None,
    **_kwargs: Any,
) -> ProvisionResult:
    response = requests.post(
        url or os.environ.get("MEMORI_NEON_LAUNCHPAD_URL") or DEFAULT_NEON_LAUNCHPAD_URL,
        json={"ref": tag},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_neon_launchpad_response(response.json())


def parse_neon_launchpad_response(data: dict[str, Any]) -> ProvisionResult:
    dsn = data.get("connection_string")
    if not isinstance(dsn, str) or not dsn:
        raise ValueError("Neon Launchpad response did not include a connection string")
        
    return ProvisionResult(
        provider="neon-launchpad",
        family="postgres",
        dsn=dsn,
        connect_args={},
        claim_url=data["claim_url"] if isinstance(data.get("claim_url"), str) else None,
        expires_at=data["expires_at"] if isinstance(data.get("expires_at"), str) else None,
        metadata={
            "id": data.get("id"),
            "status": data.get("status"),
            "neon_project_id": data.get("neon_project_id"),
        },
    )
