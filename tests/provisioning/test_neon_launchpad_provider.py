import pytest
import requests

from memori.provisioning.providers.neon_launchpad import (
    DEFAULT_NEON_LAUNCHPAD_URL,
    provision_neon_launchpad,
)


def test_provision_neon_launchpad_posts_to_api(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "connection_string": "postgresql://user:password@host:port/dbname",
        "claim_url": "https://neon.new/claim/abc",
        "expires_at": "2026-06-01T00:00:00Z",
    }

    post = mocker.patch(
        "memori.provisioning.providers.neon_launchpad.requests.post",
        return_value=response,
    )

    result = provision_neon_launchpad(tag="memori-test", timeout=7)

    post.assert_called_once_with(
        DEFAULT_NEON_LAUNCHPAD_URL,
        json={"ref": "memori-test"},
        timeout=7,
    )
    response.raise_for_status.assert_called_once_with()
    assert result.dsn == "postgresql://user:password@host:port/dbname"
    assert result.family == "postgresql"
    assert result.claim_url == "https://neon.new/claim/abc"


def test_provision_neon_launchpad_supports_url_override(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "connection_string": "postgresql://user:password@host:port/dbname",
        "claim_url": "https://neon.new/claim/abc",
        "expires_at": "2026-06-01T00:00:00Z",
    }
    post = mocker.patch(
        "memori.provisioning.providers.neon_launchpad.requests.post",
        return_value=response,
    )

    provision_neon_launchpad(url="https://custom.example.com/db")

    post.assert_called_once_with(
        "https://custom.example.com/db",
        json={"ref": "memori"},
        timeout=30,
    )


def test_provision_neon_launchpad_propagates_http_errors(mocker):
    response = mocker.Mock()
    response.raise_for_status.side_effect = requests.HTTPError("500")
    mocker.patch(
        "memori.provisioning.providers.neon_launchpad.requests.post",
        return_value=response,
    )
    # TEST

    with pytest.raises(requests.HTTPError):
        provision_neon_launchpad()
