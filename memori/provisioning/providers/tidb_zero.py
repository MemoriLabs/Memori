from __future__ import annotations

import os
from typing import Any

import requests

from memori.provisioning._models import ProvisionResult
from memori.provisioning._registry import Registry

DEFAULT_TIDB_ZERO_URL = "https://zero.tidbapi.com/v1beta1/instances"


@Registry.register_provider("tidb-zero")
def provision_tidb_zero(
    *,
    tag: str = "memori",
    timeout: int = 30,
    url: str | None = None,
    api_key: str | None = None,
    **_kwargs: Any,
) -> ProvisionResult:
    headers = {"Content-Type": "application/json"}
    resolved_api_key = api_key or os.environ.get("TIDB_ZERO_API_KEY")
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"

    response = requests.post(
        url or os.environ.get("MEMORI_TIDB_ZERO_URL") or DEFAULT_TIDB_ZERO_URL,
        headers=headers,
        json={"tag": tag},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_tidb_zero_response(response.json())


def parse_tidb_zero_response(data: dict[str, Any]) -> ProvisionResult:
    instance = data.get("instance")
    if not isinstance(instance, dict):
        raise ValueError("TiDB Zero response did not include an instance")

    dsn = instance.get("connectionString")
    if not isinstance(dsn, str) or not dsn:
        raise ValueError("TiDB Zero response did not include a connection string")

    claim_info = instance.get("claimInfo") or {}
    claim_url = claim_info.get("claimUrl") if isinstance(claim_info, dict) else None
    expires_at = instance.get("expiresAt")
    connection = _safe_connection_metadata(instance.get("connection"))

    return ProvisionResult(
        provider="tidb-zero",
        family="mysql",
        dsn=dsn,
        connect_args={"ssl": {}},
        claim_url=claim_url if isinstance(claim_url, str) else None,
        expires_at=expires_at if isinstance(expires_at, str) else None,
        metadata={
            "id": instance.get("id"),
            "connection": connection,
        },
    )


def _safe_connection_metadata(connection: Any) -> dict[str, Any] | None:
    if not isinstance(connection, dict):
        return None
    return {
        key: value
        for key, value in connection.items()
        if key.lower() not in {"password", "pwd"}
    }
