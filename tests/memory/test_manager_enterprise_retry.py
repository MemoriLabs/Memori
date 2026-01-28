import pytest

from memori._config import Config
from memori._exceptions import MemoriApiError
from memori.memory._manager import Manager


def test_manager_enterprise_retries_until_201(mocker):
    cfg = Config()
    cfg.enterprise = True
    cfg.request_num_backoff = 3
    cfg.request_backoff_factor = 0.01

    api = mocker.Mock()
    api.post.side_effect = [500, 200, 201]
    mocker.patch("memori.memory._manager.Api", return_value=api)
    sleep = mocker.patch("memori.memory._manager.time.sleep")

    out = Manager(cfg)._handle_enterprise({"messages": []})

    assert out is None
    assert api.post.call_count == 3
    assert sleep.call_count == 2


def test_manager_enterprise_raises_after_exhausting_attempts(mocker):
    cfg = Config()
    cfg.enterprise = True
    cfg.request_num_backoff = 2
    cfg.request_backoff_factor = 0

    api = mocker.Mock()
    api.post.return_value = 200
    mocker.patch("memori.memory._manager.Api", return_value=api)
    mocker.patch("memori.memory._manager.time.sleep")

    with pytest.raises(MemoriApiError, match="Expected 201"):
        Manager(cfg)._handle_enterprise({"messages": []})
