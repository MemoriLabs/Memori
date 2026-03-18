import pathlib
from unittest.mock import patch

import pytest

from memori import Memori

from .cloud_helpers import clear_mock_state, mocked_post

try:
    from dotenv import load_dotenv

    current_dir = pathlib.Path.cwd()
    env_path = current_dir / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


@pytest.fixture(autouse=True)
def universal_memori_mock():
    """Automatically mock network requests for ALL tests and reset state before/after."""
    clear_mock_state()
    with patch("memori._network.Api.post", new=mocked_post):
        yield
    clear_mock_state()


@pytest.fixture
def memori_setup(request):
    test_name = request.node.name.replace("test_", "").replace("_", "-")
    memori_client = Memori()
    entity_id = f"user-{test_name}"
    process_id = f"process-{test_name}"
    memori_client.attribution(entity_id=entity_id, process_id=process_id)
    return memori_client, entity_id, process_id
