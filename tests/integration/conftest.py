"""Shared fixtures for integration tests."""

import os
import time
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

requires_openai = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY environment variable not set",
)

requires_anthropic = pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY environment variable not set",
)

# Check if google-genai is installed (the new unified SDK)
try:
    import importlib.util

    GOOGLE_SDK_AVAILABLE = importlib.util.find_spec("google.genai") is not None
except ImportError:
    GOOGLE_SDK_AVAILABLE = False

requires_google = pytest.mark.skipif(
    not GOOGLE_API_KEY or not GOOGLE_SDK_AVAILABLE,
    reason="GOOGLE_API_KEY not set or google-genai not installed",
)

requires_xai = pytest.mark.skipif(
    not XAI_API_KEY,
    reason="XAI_API_KEY environment variable not set",
)

# AWS Bedrock requires AWS credentials
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Check if langchain-aws is installed
try:
    BEDROCK_SDK_AVAILABLE = importlib.util.find_spec("langchain_aws") is not None
except ImportError:
    BEDROCK_SDK_AVAILABLE = False

requires_bedrock = pytest.mark.skipif(
    not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY) or not BEDROCK_SDK_AVAILABLE,
    reason="AWS credentials not set or langchain-aws not installed",
)


@pytest.fixture(scope="session")
def openai_api_key():
    """Return the OpenAI API key or skip if not available."""
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")
    return OPENAI_API_KEY


@pytest.fixture
def sqlite_session_factory(tmp_path):
    """Create a file-based SQLite database session factory for isolated testing.

    Uses a temporary file instead of in-memory to avoid threading issues with
    the background augmentation threads.
    """
    db_path = tmp_path / "test_memori.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield Session

    # Allow background threads to finish their pending operations
    time.sleep(0.2)
    engine.dispose()


@pytest.fixture
def memori_test_mode():
    """Enable Memori test mode for the duration of the test."""
    original = os.environ.get("MEMORI_TEST_MODE")
    os.environ["MEMORI_TEST_MODE"] = "1"
    yield
    if original is None:
        os.environ.pop("MEMORI_TEST_MODE", None)
    else:
        os.environ["MEMORI_TEST_MODE"] = original


@pytest.fixture
def openai_client(openai_api_key):
    """Create a fresh OpenAI sync client."""
    from openai import OpenAI

    return OpenAI(api_key=openai_api_key)


@pytest.fixture
def async_openai_client(openai_api_key):
    """Create a fresh OpenAI async client."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=openai_api_key)


@pytest.fixture
def memori_instance(sqlite_session_factory, memori_test_mode):
    """Create a Memori instance with SQLite for isolated testing."""
    from memori import Memori

    mem = Memori(conn=sqlite_session_factory)
    mem.config.storage.build()

    yield mem

    # Allow background operations to complete before teardown
    time.sleep(0.1)


@pytest.fixture
def registered_openai_client(memori_instance, openai_client):
    """Create a Memori-wrapped OpenAI sync client with attribution."""
    memori_instance.llm.register(openai_client)
    memori_instance.attribution(entity_id="test-entity", process_id="test-process")
    return openai_client


@pytest.fixture
def registered_async_openai_client(memori_instance, async_openai_client):
    """Create a Memori-wrapped OpenAI async client with attribution."""
    memori_instance.llm.register(async_openai_client)
    memori_instance.attribution(entity_id="test-entity", process_id="test-process")
    return async_openai_client


@pytest.fixture
def registered_streaming_openai_client(memori_instance, openai_api_key):
    """Create a Memori-wrapped OpenAI sync client for streaming tests.

    Note: Streaming is enabled by passing stream=True to the API call,
    not to the register() method.
    """
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)
    memori_instance.llm.register(client)
    memori_instance.attribution(entity_id="test-entity", process_id="test-process")
    return client


@pytest.fixture
def registered_async_streaming_client(memori_instance, openai_api_key):
    """Create a Memori-wrapped OpenAI async client for streaming tests.

    Note: Streaming is enabled by passing stream=True to the API call,
    not to the register() method.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_api_key)
    memori_instance.llm.register(client)
    memori_instance.attribution(entity_id="test-entity", process_id="test-process")
    return client


@pytest.fixture(scope="session")
def anthropic_api_key():
    """Return the Anthropic API key or skip if not available."""
    if not ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return ANTHROPIC_API_KEY


@pytest.fixture
def anthropic_client(anthropic_api_key):
    """Create a fresh Anthropic sync client."""
    from anthropic import Anthropic

    return Anthropic(api_key=anthropic_api_key)


@pytest.fixture
def async_anthropic_client(anthropic_api_key):
    """Create a fresh Anthropic async client."""
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=anthropic_api_key)


@pytest.fixture
def registered_anthropic_client(memori_instance, anthropic_client):
    """Create a Memori-wrapped Anthropic sync client with attribution."""
    memori_instance.llm.register(anthropic_client)
    memori_instance.attribution(entity_id="test-entity", process_id="test-process")
    return anthropic_client


@pytest.fixture
def registered_async_anthropic_client(memori_instance, async_anthropic_client):
    """Create a Memori-wrapped Anthropic async client with attribution."""
    memori_instance.llm.register(async_anthropic_client)
    memori_instance.attribution(entity_id="test-entity", process_id="test-process")
    return async_anthropic_client


@pytest.fixture(scope="session")
def google_api_key():
    """Return the Google API key or skip if not available."""
    if not GOOGLE_API_KEY:
        pytest.skip("GOOGLE_API_KEY not set")
    return GOOGLE_API_KEY


@pytest.fixture
def google_client(google_api_key):
    """Create a fresh Google GenAI client using the new unified SDK."""
    if not GOOGLE_SDK_AVAILABLE:
        pytest.skip("google-genai not installed (pip install google-genai)")

    from google import genai

    client = genai.Client(api_key=google_api_key)
    yield client
    # Clean up client resources
    client.close()


@pytest.fixture
def registered_google_client(memori_instance, google_client):
    """Create a Memori-wrapped Google client with attribution."""
    memori_instance.llm.register(google_client)
    memori_instance.attribution(entity_id="test-entity", process_id="test-process")
    return google_client


@pytest.fixture(scope="session")
def xai_api_key():
    """Return the xAI API key or skip if not available."""
    if not XAI_API_KEY:
        pytest.skip("XAI_API_KEY not set")
    return XAI_API_KEY


@pytest.fixture
def xai_client(xai_api_key):
    """Create a fresh xAI sync client (OpenAI-compatible)."""
    from openai import OpenAI

    return OpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1",
    )


@pytest.fixture
def async_xai_client(xai_api_key):
    """Create a fresh xAI async client (OpenAI-compatible)."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1",
    )


@pytest.fixture
def registered_xai_client(memori_instance, xai_client):
    """Create a Memori-wrapped xAI sync client with attribution."""
    memori_instance.llm.register(xai_client)
    memori_instance.attribution(entity_id="test-entity", process_id="test-process")
    return xai_client


@pytest.fixture
def registered_async_xai_client(memori_instance, async_xai_client):
    """Create a Memori-wrapped xAI async client with attribution."""
    memori_instance.llm.register(async_xai_client)
    memori_instance.attribution(entity_id="test-entity", process_id="test-process")
    return async_xai_client


@pytest.fixture(scope="session")
def aws_credentials():
    """Return AWS credentials or skip if not available."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        pytest.skip("AWS credentials not set")
    return {
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
        "region_name": AWS_REGION,
    }


@pytest.fixture
def bedrock_client(aws_credentials):
    """Create a fresh Bedrock ChatBedrock client via LangChain."""
    if not BEDROCK_SDK_AVAILABLE:
        pytest.skip("langchain-aws not installed (pip install langchain-aws)")

    from langchain_aws import ChatBedrock  # type: ignore[import-not-found]

    return ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name=aws_credentials["region_name"],
    )


@pytest.fixture
def registered_bedrock_client(memori_instance, bedrock_client):
    """Create a Memori-wrapped Bedrock client with attribution.

    Note: Bedrock uses named parameter registration: chatbedrock=client
    """
    memori_instance.llm.register(chatbedrock=bedrock_client)
    memori_instance.attribution(entity_id="test-entity", process_id="test-process")
    return bedrock_client


@dataclass
class CapturedPayload:
    """Container for captured AA payloads with validation methods."""

    payloads: list = field(default_factory=list)

    def capture(self, payload: dict) -> dict:
        """Capture a payload and return mock response."""
        self.payloads.append(payload)
        # Return a valid mock response structure
        return {
            "entity": {"facts": [], "triples": []},
            "process": {"attributes": []},
            "conversation": {"summary": None},
        }

    @property
    def last(self) -> dict | None:
        """Get the last captured payload."""
        return self.payloads[-1] if self.payloads else None

    @property
    def count(self) -> int:
        """Get number of captured payloads."""
        return len(self.payloads)

    def validate_structure(self, payload: dict | None = None) -> list[str]:
        """Validate payload structure against expected AA schema.

        Returns list of validation errors (empty if valid).
        """
        errors = []
        payload = payload or self.last

        if not payload:
            return ["No payload to validate"]

        # Check top-level keys
        if "conversation" not in payload:
            errors.append("Missing 'conversation' key")
        if "meta" not in payload:
            errors.append("Missing 'meta' key")

        # Validate conversation structure
        if "conversation" in payload:
            conv = payload["conversation"]
            if "messages" not in conv:
                errors.append("Missing 'conversation.messages'")
            elif not isinstance(conv["messages"], list):
                errors.append("'conversation.messages' must be a list")

        # Validate meta structure
        if "meta" in payload:
            meta = payload["meta"]

            # Check required meta keys
            required_meta = [
                "attribution",
                "framework",
                "llm",
                "platform",
                "sdk",
                "storage",
            ]
            for key in required_meta:
                if key not in meta:
                    errors.append(f"Missing 'meta.{key}'")

            # Validate attribution
            if "attribution" in meta:
                attr = meta["attribution"]
                if "entity" not in attr or "id" not in attr.get("entity", {}):
                    errors.append("Missing 'meta.attribution.entity.id'")
                if "process" not in attr or "id" not in attr.get("process", {}):
                    errors.append("Missing 'meta.attribution.process.id'")

                # Validate hashed IDs (should be 64-char SHA256 or None)
                entity_id = attr.get("entity", {}).get("id")
                if entity_id is not None and len(entity_id) != 64:
                    errors.append(
                        f"Entity ID not hashed: {len(entity_id)} chars, expected 64"
                    )

                process_id = attr.get("process", {}).get("id")
                if process_id is not None and len(process_id) != 64:
                    errors.append(
                        f"Process ID not hashed: {len(process_id)} chars, expected 64"
                    )

            # Validate LLM structure
            if "llm" in meta:
                llm = meta["llm"]
                if "model" not in llm:
                    errors.append("Missing 'meta.llm.model'")
                elif "provider" not in llm.get("model", {}):
                    errors.append("Missing 'meta.llm.model.provider'")

            # Validate SDK structure
            if "sdk" in meta:
                sdk = meta["sdk"]
                if sdk.get("lang") != "python":
                    lang = sdk.get("lang")
                    errors.append(f"Expected sdk.lang='python', got '{lang}'")

            # Validate storage structure
            if "storage" in meta:
                storage = meta["storage"]
                if "dialect" not in storage:
                    errors.append("Missing 'meta.storage.dialect'")
                if "cockroachdb" not in storage:
                    errors.append("Missing 'meta.storage.cockroachdb'")

        return errors

    def is_valid(self, payload: dict | None = None) -> bool:
        """Check if payload is valid."""
        return len(self.validate_structure(payload)) == 0


@pytest.fixture
def aa_payload_capture():
    """Fixture that captures AA payloads without sending to API.

    Use this to validate payload structure without making actual API calls.
    The mock returns a valid response structure so the pipeline continues.
    """
    captured = CapturedPayload()

    async def mock_augmentation(payload: dict) -> dict:
        return captured.capture(payload)

    with patch("memori._network.Api.augmentation_async", new=mock_augmentation):
        yield captured


@pytest.fixture
def memori_instance_with_capture(
    sqlite_session_factory, memori_test_mode, aa_payload_capture
):
    """Create a Memori instance that captures AA payloads for validation.

    Use this when you want to validate payload formatting without sending
    requests to the AA API (even staging).
    """
    from memori import Memori

    mem = Memori(conn=sqlite_session_factory)
    mem.config.storage.build()

    yield mem, aa_payload_capture

    # Allow background operations to complete before teardown
    time.sleep(0.1)
