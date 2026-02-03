import os
import time

import pytest


@pytest.fixture
def hosted_memori_enterprise():
    """Set MEMORI_ENTERPRISE=1 for hosted mode, restore on teardown."""
    original = os.environ.get("MEMORI_ENTERPRISE")
    os.environ["MEMORI_ENTERPRISE"] = "1"
    yield
    if original is None:
        os.environ.pop("MEMORI_ENTERPRISE", None)
    else:
        os.environ["MEMORI_ENTERPRISE"] = original


@pytest.fixture
def hosted_memori_instance(sqlite_session_factory, hosted_memori_enterprise):
    """Create a Memori instance in hosted/enterprise mode.

    Does NOT mock the augmentation API — hits real production API.
    Does NOT set MEMORI_TEST_MODE — uses production endpoint.
    """
    from memori import Memori

    mem = Memori(conn=sqlite_session_factory)
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
