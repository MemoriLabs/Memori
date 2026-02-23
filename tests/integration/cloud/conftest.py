import os
import time

import pytest

MEMORI_API_KEY = os.environ.get("MEMORI_API_KEY")

requires_memori_api_key = pytest.mark.skipif(
    not MEMORI_API_KEY,
    reason="MEMORI_API_KEY environment variable not set (required for cloud tests)",
)


@pytest.fixture
def cloud_test_mode():
    """Set MEMORI_TEST_MODE=1 so cloud API calls hit staging.

    Production cloud-api.memorilabs.ai does not exist yet.
    Only staging-cloud-api.memorilabs.ai is live.
    """
    original = os.environ.get("MEMORI_TEST_MODE")
    os.environ["MEMORI_TEST_MODE"] = "1"
    yield
    if original is None:
        os.environ.pop("MEMORI_TEST_MODE", None)
    else:
        os.environ["MEMORI_TEST_MODE"] = original


@pytest.fixture
def cloud_memori_instance(sqlite_session_factory, cloud_test_mode):
    """Create a Memori instance in cloud mode with local SQLite for verification.

    Uses conn for local storage (conversation/message verification) but sets
    config.cloud = True so augmentation and recall hit the staging cloud API.
    Requires MEMORI_API_KEY and MEMORI_TEST_MODE=1 (set automatically).
    """
    if not MEMORI_API_KEY:
        pytest.skip("MEMORI_API_KEY not set (required for cloud tests)")

    from memori import Memori

    mem = Memori(conn=sqlite_session_factory)
    mem.config.cloud = True
    mem.config.storage.build()

    yield mem

    mem.close()
    time.sleep(0.2)


@pytest.fixture
def cloud_registered_openai_client(cloud_memori_instance, openai_client):
    cloud_memori_instance.llm.register(openai_client)
    cloud_memori_instance.attribution(
        entity_id="cloud-test-entity", process_id="cloud-test-process"
    )
    return openai_client


@pytest.fixture
def cloud_registered_async_openai_client(cloud_memori_instance, async_openai_client):
    cloud_memori_instance.llm.register(async_openai_client)
    cloud_memori_instance.attribution(
        entity_id="cloud-test-entity", process_id="cloud-test-process"
    )
    return async_openai_client


@pytest.fixture
def cloud_registered_anthropic_client(cloud_memori_instance, anthropic_client):
    cloud_memori_instance.llm.register(anthropic_client)
    cloud_memori_instance.attribution(
        entity_id="cloud-test-entity", process_id="cloud-test-process"
    )
    return anthropic_client


@pytest.fixture
def cloud_registered_async_anthropic_client(
    cloud_memori_instance, async_anthropic_client
):
    cloud_memori_instance.llm.register(async_anthropic_client)
    cloud_memori_instance.attribution(
        entity_id="cloud-test-entity", process_id="cloud-test-process"
    )
    return async_anthropic_client


@pytest.fixture
def cloud_registered_google_client(cloud_memori_instance, google_client):
    cloud_memori_instance.llm.register(google_client)
    cloud_memori_instance.attribution(
        entity_id="cloud-test-entity", process_id="cloud-test-process"
    )
    return google_client


@pytest.fixture
def cloud_registered_xai_client(cloud_memori_instance, xai_client):
    cloud_memori_instance.llm.register(xai_client)
    cloud_memori_instance.attribution(
        entity_id="cloud-test-entity", process_id="cloud-test-process"
    )
    return xai_client


@pytest.fixture
def cloud_registered_async_xai_client(cloud_memori_instance, async_xai_client):
    cloud_memori_instance.llm.register(async_xai_client)
    cloud_memori_instance.attribution(
        entity_id="cloud-test-entity", process_id="cloud-test-process"
    )
    return async_xai_client


@pytest.fixture
def cloud_registered_bedrock_client(cloud_memori_instance, bedrock_client):
    cloud_memori_instance.llm.register(chatbedrock=bedrock_client)
    cloud_memori_instance.attribution(
        entity_id="cloud-test-entity", process_id="cloud-test-process"
    )
    return bedrock_client
