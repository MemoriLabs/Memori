import os
import time

import pytest

MEMORI_API_KEY = os.environ.get("MEMORI_API_KEY")

requires_memori_api_key = pytest.mark.skipif(
    not MEMORI_API_KEY,
    reason="MEMORI_API_KEY environment variable not set (required for hosted tests)",
)


@pytest.fixture
def hosted_test_mode():
    """Set MEMORI_TEST_MODE=1 so hosted API calls hit staging.

    Production hosted-api.memorilabs.ai does not exist yet.
    Only staging-hosted-api.memorilabs.ai is live.
    """
    original = os.environ.get("MEMORI_TEST_MODE")
    os.environ["MEMORI_TEST_MODE"] = "1"
    yield
    if original is None:
        os.environ.pop("MEMORI_TEST_MODE", None)
    else:
        os.environ["MEMORI_TEST_MODE"] = original


@pytest.fixture
def hosted_memori_instance(sqlite_session_factory, hosted_test_mode):
    """Create a Memori instance in hosted mode with local SQLite for verification.

    Uses conn for local storage (conversation/message verification) but sets
    config.hosted = True so augmentation and recall hit the staging hosted API.
    Requires MEMORI_API_KEY and MEMORI_TEST_MODE=1 (set automatically).
    """
    if not MEMORI_API_KEY:
        pytest.skip("MEMORI_API_KEY not set (required for hosted tests)")

    from memori import Memori

    mem = Memori(conn=sqlite_session_factory)
    mem.config.hosted = True
    mem.config.storage.build()

    yield mem

    mem.close()
    time.sleep(0.2)


@pytest.fixture
def hosted_registered_openai_client(hosted_memori_instance, openai_client):
    hosted_memori_instance.llm.register(openai_client)
    hosted_memori_instance.attribution(
        entity_id="hosted-test-entity", process_id="hosted-test-process"
    )
    return openai_client


@pytest.fixture
def hosted_registered_async_openai_client(hosted_memori_instance, async_openai_client):
    hosted_memori_instance.llm.register(async_openai_client)
    hosted_memori_instance.attribution(
        entity_id="hosted-test-entity", process_id="hosted-test-process"
    )
    return async_openai_client


@pytest.fixture
def hosted_registered_anthropic_client(hosted_memori_instance, anthropic_client):
    hosted_memori_instance.llm.register(anthropic_client)
    hosted_memori_instance.attribution(
        entity_id="hosted-test-entity", process_id="hosted-test-process"
    )
    return anthropic_client


@pytest.fixture
def hosted_registered_async_anthropic_client(
    hosted_memori_instance, async_anthropic_client
):
    hosted_memori_instance.llm.register(async_anthropic_client)
    hosted_memori_instance.attribution(
        entity_id="hosted-test-entity", process_id="hosted-test-process"
    )
    return async_anthropic_client


@pytest.fixture
def hosted_registered_google_client(hosted_memori_instance, google_client):
    hosted_memori_instance.llm.register(google_client)
    hosted_memori_instance.attribution(
        entity_id="hosted-test-entity", process_id="hosted-test-process"
    )
    return google_client


@pytest.fixture
def hosted_registered_xai_client(hosted_memori_instance, xai_client):
    hosted_memori_instance.llm.register(xai_client)
    hosted_memori_instance.attribution(
        entity_id="hosted-test-entity", process_id="hosted-test-process"
    )
    return xai_client


@pytest.fixture
def hosted_registered_async_xai_client(hosted_memori_instance, async_xai_client):
    hosted_memori_instance.llm.register(async_xai_client)
    hosted_memori_instance.attribution(
        entity_id="hosted-test-entity", process_id="hosted-test-process"
    )
    return async_xai_client


@pytest.fixture
def hosted_registered_bedrock_client(hosted_memori_instance, bedrock_client):
    hosted_memori_instance.llm.register(chatbedrock=bedrock_client)
    hosted_memori_instance.attribution(
        entity_id="hosted-test-entity", process_id="hosted-test-process"
    )
    return bedrock_client
