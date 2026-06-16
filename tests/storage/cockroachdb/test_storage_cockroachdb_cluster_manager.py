import sys
from unittest.mock import MagicMock, patch

from memori._config import Config
from memori.storage.cockroachdb._cluster_manager import ClusterManager


def test_start_finalize_post_uses_extended_timeout():
    config = Config()
    cli = MagicMock()
    manager = ClusterManager(config)

    cluster_uuid = "test-cluster-uuid"
    started = {"cluster": {"uuid": cluster_uuid, "id": 123}}
    finalized = {
        "status": 1,
        "connection": {"string": "postgresql://user:pass@host/db"},
        "claim": {"url": "https://claim.example.com"},
    }

    mock_api = MagicMock()
    mock_api.post.side_effect = [started, finalized]
    mock_psycopg = MagicMock()

    with (
        patch.object(manager, "cluster_is_started", return_value=False),
        patch("builtins.input", return_value="Y"),
        patch(
            "memori.storage.cockroachdb._cluster_manager.Api",
            return_value=mock_api,
        ),
        patch("memori.storage.cockroachdb._cluster_manager.time.sleep"),
        patch("memori.storage.cockroachdb._cluster_manager.Manager") as mock_manager,
        patch("memori.storage.cockroachdb._cluster_manager.Builder") as mock_builder,
        patch.dict(sys.modules, {"psycopg": mock_psycopg}),
        patch.object(manager.files, "write_id"),
    ):
        mock_manager.return_value.start.return_value = MagicMock()
        manager.start(cli)

    assert mock_api.post.call_count == 2

    start_call = mock_api.post.call_args_list[0]
    assert start_call.args == ("cockroachdb/cluster/start",)
    assert "timeout" not in start_call.kwargs

    finalize_call = mock_api.post.call_args_list[1]
    assert finalize_call.args == (f"cockroachdb/cluster/finalize/{cluster_uuid}",)
    assert finalize_call.kwargs["json"] == {"cluster": {"id": 123}}
    assert finalize_call.kwargs["timeout"] == 60

    mock_builder.return_value.disable_banner.return_value.execute.assert_called_once()
