import pytest
import requests

from memori.provisioning.providers.tidb_zero import (
    DEFAULT_TIDB_ZERO_URL,
    provision_tidb_zero,
)


def test_provision_tidb_zero_posts_to_v1beta1_endpoint(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "instance": {
            "connectionString": "mysql://user:secret@example.com/memori",
            "claimInfo": {"claimUrl": "https://tidbcloud.com/tidbs/claim/abc"},
            "expiresAt": "2026-06-01T00:00:00Z",
        }
    }
    post = mocker.patch(
        "memori.provisioning.providers.tidb_zero.requests.post",
        return_value=response,
    )

    result = provision_tidb_zero(tag="memori-test", timeout=7)

    post.assert_called_once_with(
        DEFAULT_TIDB_ZERO_URL,
        headers={"Content-Type": "application/json"},
        json={"tag": "memori-test"},
        timeout=7,
    )
    response.raise_for_status.assert_called_once_with()
    assert result.dsn == "mysql://user:secret@example.com/memori"
    assert result.connect_args == {"ssl": {}}
    assert result.claim_url == "https://tidbcloud.com/tidbs/claim/abc"


def test_provision_tidb_zero_supports_url_override_and_bearer_token(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "instance": {"connectionString": "mysql://user:secret@example.com/memori"}
    }
    post = mocker.patch(
        "memori.provisioning.providers.tidb_zero.requests.post",
        return_value=response,
    )

    provision_tidb_zero(
        tag="memori-test",
        timeout=7,
        url="https://example.com/instances",
        api_key="tidb-token",
    )

    post.assert_called_once_with(
        "https://example.com/instances",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer tidb-token",
        },
        json={"tag": "memori-test"},
        timeout=7,
    )


def test_provision_tidb_zero_propagates_http_errors(mocker):
    response = mocker.Mock()
    response.raise_for_status.side_effect = requests.HTTPError("500")
    mocker.patch(
        "memori.provisioning.providers.tidb_zero.requests.post",
        return_value=response,
    )
    # TEST

    with pytest.raises(requests.HTTPError):
        provision_tidb_zero()
