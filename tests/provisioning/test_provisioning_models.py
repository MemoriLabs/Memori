from datetime import datetime, timezone

from memori.provisioning._models import ProvisionResult
from memori.provisioning.providers.tidb_zero import parse_tidb_zero_response


def test_parse_tidb_zero_response_full_payload():
    result = parse_tidb_zero_response(
        {
            "instance": {
                "id": "inst-1",
                "connection": {"host": "example.tidbcloud.com"},
                "connectionString": "mysql://user:pass@example.tidbcloud.com/db",
                "claimInfo": {"claimUrl": "https://tidbcloud.com/tidbs/claim/abc"},
                "expiresAt": "2026-06-01T00:00:00Z",
            }
        }
    )

    assert result == ProvisionResult(
        provider="tidb-zero",
        family="mysql",
        dsn="mysql://user:pass@example.tidbcloud.com/db",
        connect_args={"ssl": {}},
        claim_url="https://tidbcloud.com/tidbs/claim/abc",
        expires_at="2026-06-01T00:00:00Z",
        metadata={
            "id": "inst-1",
            "connection": {"host": "example.tidbcloud.com"},
        },
    )


def test_parse_tidb_zero_response_removes_connection_password_from_metadata():
    result = parse_tidb_zero_response(
        {
            "instance": {
                "connection": {
                    "host": "example.tidbcloud.com",
                    "username": "user",
                    "password": "secret",
                },
                "connectionString": "mysql://user:secret@example.tidbcloud.com/db",
            }
        }
    )

    assert result.metadata["connection"] == {
        "host": "example.tidbcloud.com",
        "username": "user",
    }


def test_parse_tidb_zero_response_missing_optional_claim_and_expiry():
    result = parse_tidb_zero_response(
        {"instance": {"connectionString": "mysql://user:pass@host/db"}}
    )

    assert result.provider == "tidb-zero"
    assert result.family == "mysql"
    assert result.connect_args == {"ssl": {}}
    assert result.claim_url is None
    assert result.expires_at is None


def test_provision_result_expiry_detection():
    result = ProvisionResult(
        provider="tidb-zero",
        family="mysql",
        dsn="mysql://user:pass@host/db",
        expires_at="2026-01-01T00:00:00Z",
    )

    assert result.is_expired(datetime(2026, 1, 2, tzinfo=timezone.utc)) is True
    assert result.is_expired(datetime(2025, 12, 31, tzinfo=timezone.utc)) is False


def test_provision_result_malformed_expiry_is_expired():
    result = ProvisionResult(
        provider="tidb-zero",
        family="mysql",
        dsn="mysql://user:pass@host/db",
        expires_at="surprise",
    )

    assert result.is_expired() is True
