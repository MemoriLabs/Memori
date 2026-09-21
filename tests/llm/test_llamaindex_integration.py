"""Unit tests for the Memori × LlamaIndex integration.

All tests use mocks — no real LLM calls or Memori Cloud API requests are made.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the integration package importable without pip-installing it.
# The integration lives at integrations/llamaindex/src/ relative to the
# repository root, which is 3 levels up from this test file.
_REPO_ROOT = Path(__file__).parent.parent.parent
_INTEGRATION_SRC = _REPO_ROOT / "integrations" / "llamaindex" / "src"
if str(_INTEGRATION_SRC) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_SRC))

from memori import Memori  # noqa: E402
from memori._exceptions import UnsupportedLLMProviderError  # noqa: E402
from memori.llm._constants import (  # noqa: E402
    LLAMAINDEX_ANTHROPIC_LLM_PROVIDER,
    LLAMAINDEX_FRAMEWORK_PROVIDER,
    LLAMAINDEX_OPENAI_LLM_PROVIDER,
)
from memori.llm._utils import client_is_llamaindex  # noqa: E402
from memori.llm.clients.frameworks import LlamaIndex as LlamaIndexClient  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memori_instance(mocker):
    """Return a Memori instance with mocked storage and augmentation."""
    mock_conn = mocker.MagicMock()
    mocker.patch("memori.storage.Manager.start", return_value=mocker.MagicMock())
    mocker.patch(
        "memori.memory.augmentation.Manager.start", return_value=mocker.MagicMock()
    )
    return Memori(conn=mock_conn)


def _make_llamaindex_openai_llm(mocker, module="llama_index.llms.openai"):
    """Build a mock LlamaIndex OpenAI LLM with a patching-ready _client."""
    mock_llm = mocker.MagicMock()
    type(mock_llm).__module__ = module
    # Simulate the internal OpenAI client
    mock_inner = mocker.MagicMock()
    type(mock_inner).__module__ = "openai"
    mock_inner._version = "2.8.1"
    del mock_inner._memori_installed
    del mock_inner.base_url
    mock_llm._client = mock_inner
    del mock_llm.llm
    # Prevent async detection from raising
    mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)
    return mock_llm, mock_inner


def _make_llamaindex_anthropic_llm(mocker, module="llama_index.llms.anthropic"):
    """Build a mock LlamaIndex Anthropic LLM with a patching-ready _client."""
    mock_llm = mocker.MagicMock()
    type(mock_llm).__module__ = module
    mock_inner = mocker.MagicMock()
    type(mock_inner).__module__ = "anthropic"
    del mock_inner._memori_installed
    # Ensure anthropic version is importable
    mock_anthropic_module = mocker.MagicMock()
    mock_anthropic_module.__version__ = "0.75.0"
    mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic_module})
    mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)
    mock_llm._client = mock_inner
    del mock_llm.llm
    return mock_llm, mock_inner


# ---------------------------------------------------------------------------
# client_is_llamaindex detector
# ---------------------------------------------------------------------------


def test_client_is_llamaindex_detects_llms_openai(mocker):
    """client_is_llamaindex returns True for llama_index.llms.openai.*."""
    obj = mocker.MagicMock()
    type(obj).__module__ = "llama_index.llms.openai"
    assert client_is_llamaindex(obj) is True


def test_client_is_llamaindex_detects_core_llms(mocker):
    """client_is_llamaindex returns True for llama_index.core.llms.*."""
    obj = mocker.MagicMock()
    type(obj).__module__ = "llama_index.core.llms.openai"
    assert client_is_llamaindex(obj) is True


def test_client_is_llamaindex_rejects_non_llamaindex(mocker):
    """client_is_llamaindex returns False for non-LlamaIndex objects."""
    obj = mocker.MagicMock()
    type(obj).__module__ = "openai"
    assert client_is_llamaindex(obj) is False

    type(obj).__module__ = "langchain_openai"
    assert client_is_llamaindex(obj) is False

    type(obj).__module__ = "agno.models.openai"
    assert client_is_llamaindex(obj) is False


# ---------------------------------------------------------------------------
# LlamaIndex client class — OpenAI
# ---------------------------------------------------------------------------


def test_llamaindex_client_registers_openai(memori_instance, mocker):
    """LlamaIndex(config).register(llm) patches the inner OpenAI client."""
    mock_llm, mock_inner = _make_llamaindex_openai_llm(mocker)

    client = LlamaIndexClient(memori_instance.config)
    client.register(mock_llm)

    assert hasattr(mock_inner, "_memori_installed")
    assert mock_inner._memori_installed is True
    assert memori_instance.config.framework.provider == LLAMAINDEX_FRAMEWORK_PROVIDER
    assert memori_instance.config.llm.provider == LLAMAINDEX_OPENAI_LLM_PROVIDER


def test_llamaindex_client_registers_anthropic(memori_instance, mocker):
    """LlamaIndex(config).register(llm) patches the inner Anthropic client."""
    mock_llm, mock_inner = _make_llamaindex_anthropic_llm(mocker)

    client = LlamaIndexClient(memori_instance.config)
    client.register(mock_llm)

    assert hasattr(mock_inner, "_memori_installed")
    assert mock_inner._memori_installed is True
    assert memori_instance.config.framework.provider == LLAMAINDEX_FRAMEWORK_PROVIDER
    assert memori_instance.config.llm.provider == LLAMAINDEX_ANTHROPIC_LLM_PROVIDER


def test_llamaindex_client_idempotent(memori_instance, mocker):
    """Registering the same LLM twice does not raise or double-wrap."""
    mock_llm, mock_inner = _make_llamaindex_openai_llm(mocker)

    client = LlamaIndexClient(memori_instance.config)
    client.register(mock_llm)
    client.register(mock_llm)  # second call is a no-op

    assert mock_inner._memori_installed is True


def test_llamaindex_client_raises_on_unsupported_llm(memori_instance, mocker):
    """LlamaIndex client raises UnsupportedLLMProviderError for unknown LLMs."""
    mock_llm = mocker.MagicMock()
    type(mock_llm).__module__ = "llama_index.llms.some_unknown_provider"
    # No _client, _aclient with recognised sub-attributes
    del mock_llm._client
    del mock_llm._aclient

    client = LlamaIndexClient(memori_instance.config)
    with pytest.raises(UnsupportedLLMProviderError):
        client.register(mock_llm)


# ---------------------------------------------------------------------------
# Auto-detection via memori.llm.register()
# ---------------------------------------------------------------------------


def test_llm_register_auto_detects_llamaindex_openai(memori_instance, mocker):
    """memori.llm.register(llm) auto-detects a LlamaIndex OpenAI LLM."""
    mock_llm, mock_inner = _make_llamaindex_openai_llm(mocker)

    result = memori_instance.llm.register(mock_llm)

    assert result is memori_instance
    assert hasattr(mock_inner, "_memori_installed")
    assert mock_inner._memori_installed is True


def test_llm_register_returns_memori_for_llamaindex(memori_instance, mocker):
    """memori.llm.register(llm) returns the Memori instance for chaining."""
    mock_llm, _ = _make_llamaindex_openai_llm(mocker)

    result = memori_instance.llm.register(mock_llm)

    assert result is memori_instance
    assert isinstance(result, Memori)


def test_llm_register_llamaindex_allows_chaining(memori_instance, mocker):
    """llm.register(llm).attribution(...) chain works for LlamaIndex LLMs."""
    mock_llm, _ = _make_llamaindex_openai_llm(mocker)

    result = memori_instance.llm.register(mock_llm).attribution(
        entity_id="user_42", process_id="test-agent"
    )

    assert isinstance(result, Memori)
    assert result.config.entity_id == "user_42"
    assert result.config.process_id == "test-agent"


# ---------------------------------------------------------------------------
# MemoriChatEngine wrapper
# ---------------------------------------------------------------------------


def test_memori_chat_engine_registers_llm_on_init(memori_instance, mocker):
    """MemoriChatEngine registers the underlying LLM client at construction."""
    mock_llm, mock_inner = _make_llamaindex_openai_llm(mocker)

    mock_engine = mocker.MagicMock()
    mock_engine.llm = mock_llm

    from chat_engine import MemoriChatEngine

    _ = MemoriChatEngine(
        chat_engine=mock_engine,
        memori=memori_instance,
        entity_id="user_1",
    )

    assert hasattr(mock_inner, "_memori_installed")
    assert mock_inner._memori_installed is True
    assert memori_instance.config.entity_id == "user_1"


def test_memori_chat_engine_delegates_chat(memori_instance, mocker):
    """MemoriChatEngine.chat() delegates to the wrapped engine."""
    mock_llm, _ = _make_llamaindex_openai_llm(mocker)

    mock_engine = mocker.MagicMock()
    mock_engine.llm = mock_llm
    mock_engine.chat.return_value = "Hello!"

    from chat_engine import MemoriChatEngine

    engine = MemoriChatEngine(
        chat_engine=mock_engine,
        memori=memori_instance,
        entity_id="user_1",
    )
    response = engine.chat("Hi there")

    mock_engine.chat.assert_called_once_with("Hi there")
    assert response == "Hello!"


def test_memori_chat_engine_reset_creates_new_session(memori_instance, mocker):
    """MemoriChatEngine.reset() calls memori.new_session()."""
    mock_llm, _ = _make_llamaindex_openai_llm(mocker)

    mock_engine = mocker.MagicMock()
    mock_engine.llm = mock_llm

    from chat_engine import MemoriChatEngine

    engine = MemoriChatEngine(
        chat_engine=mock_engine,
        memori=memori_instance,
        entity_id="user_1",
    )
    old_session = memori_instance.config.session_id
    engine.reset()

    assert memori_instance.config.session_id != old_session


def test_memori_chat_engine_no_llm_warns(memori_instance, mocker, caplog):
    """MemoriChatEngine logs a warning when no LLM is found on the engine."""
    import logging

    mock_engine = mocker.MagicMock()
    del mock_engine.llm
    del mock_engine._llm
    del mock_engine.llm_predictor

    from chat_engine import MemoriChatEngine

    with caplog.at_level(logging.WARNING):
        MemoriChatEngine(
            chat_engine=mock_engine,
            memori=memori_instance,
            entity_id="user_1",
        )

    assert "could not find an LLM" in caplog.text


# ---------------------------------------------------------------------------
# MemoriAgent wrapper
# ---------------------------------------------------------------------------


def test_memori_agent_registers_llm_on_init(memori_instance, mocker):
    """MemoriAgent registers the underlying LLM client at construction."""
    mock_llm, mock_inner = _make_llamaindex_openai_llm(mocker)

    mock_agent = mocker.MagicMock()
    mock_agent.llm = mock_llm

    from agent import MemoriAgent

    _ = MemoriAgent(
        agent=mock_agent,
        memori=memori_instance,
        entity_id="user_2",
        process_id="react-agent",
    )

    assert hasattr(mock_inner, "_memori_installed")
    assert mock_inner._memori_installed is True
    assert memori_instance.config.entity_id == "user_2"
    assert memori_instance.config.process_id == "react-agent"


def test_memori_agent_delegates_chat(memori_instance, mocker):
    """MemoriAgent.chat() delegates to the wrapped agent."""
    mock_llm, _ = _make_llamaindex_openai_llm(mocker)

    mock_agent = mocker.MagicMock()
    mock_agent.llm = mock_llm
    mock_agent.chat.return_value = "Agent response"

    from agent import MemoriAgent

    agent = MemoriAgent(
        agent=mock_agent,
        memori=memori_instance,
        entity_id="user_2",
    )
    result = agent.chat("Do something")

    mock_agent.chat.assert_called_once_with("Do something")
    assert result == "Agent response"


def test_memori_agent_new_session(memori_instance, mocker):
    """MemoriAgent.new_session() creates a new Memori session."""
    mock_llm, _ = _make_llamaindex_openai_llm(mocker)

    mock_agent = mocker.MagicMock()
    mock_agent.llm = mock_llm

    from agent import MemoriAgent

    agent = MemoriAgent(
        agent=mock_agent,
        memori=memori_instance,
        entity_id="user_2",
    )
    old_session = memori_instance.config.session_id
    agent.new_session()

    assert memori_instance.config.session_id != old_session


# ---------------------------------------------------------------------------
# MemoriCallbackHandler
# ---------------------------------------------------------------------------


def test_callback_handler_captures_turn_on_llm_end(memori_instance, mocker):
    """MemoriCallbackHandler captures an agent turn when LLM event ends."""
    memori_instance.attribution(entity_id="user_cb", process_id="cb-agent")
    mock_capture = mocker.patch.object(memori_instance, "capture_agent_turn")

    from callback_handler import (
        MemoriCallbackHandler,
    )

    handler = MemoriCallbackHandler(memori_instance, project_id="proj-1")

    # Simulate CBEventType.LLM
    try:
        from llama_index.core.callbacks.schema import CBEventType

        event_type = CBEventType.LLM
    except ImportError:
        # LlamaIndex not installed — use a mock event type
        event_type = mocker.MagicMock()
        event_type.__eq__ = lambda self, other: True
        mock_schema_module = mocker.MagicMock()
        mock_schema_module.CBEventType.LLM = event_type
        mocker.patch.dict(
            "sys.modules", {"llama_index.core.callbacks.schema": mock_schema_module}
        )

    # on_event_start: set pending user message
    mock_msg = mocker.MagicMock()
    mock_msg.role = "user"
    mock_msg.content = "Hello agent"
    handler.on_event_start(event_type, payload={"messages": [mock_msg]})

    assert handler._pending_user_message == "Hello agent"

    # on_event_end: capture the turn
    mock_response = mocker.MagicMock()
    mock_response.message.content = "Hello user!"
    handler.on_event_end(event_type, payload={"response": mock_response})

    mock_capture.assert_called_once_with(
        user_content="Hello agent",
        assistant_content="Hello user!",
        project_id="proj-1",
        platform="llamaindex",
    )


def test_callback_handler_skips_capture_without_project_id(memori_instance, mocker):
    """MemoriCallbackHandler does not capture if project_id is not set."""
    mock_capture = mocker.patch.object(memori_instance, "capture_agent_turn")

    from callback_handler import (
        MemoriCallbackHandler,
    )

    handler = MemoriCallbackHandler(memori_instance, project_id=None)
    handler._pending_user_message = "Hello"

    try:
        from llama_index.core.callbacks.schema import CBEventType

        event_type = CBEventType.LLM
    except ImportError:
        event_type = mocker.MagicMock()
        event_type.__eq__ = lambda self, other: True
        mock_schema_module = mocker.MagicMock()
        mock_schema_module.CBEventType.LLM = event_type
        mocker.patch.dict(
            "sys.modules", {"llama_index.core.callbacks.schema": mock_schema_module}
        )

    mock_response = mocker.MagicMock()
    mock_response.message.content = "Reply"
    handler.on_event_end(event_type, payload={"response": mock_response})

    mock_capture.assert_not_called()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_memori_llamaindex_config_defaults():
    """MemoriLlamaIndexConfig has sensible defaults."""
    from config import MemoriLlamaIndexConfig

    cfg = MemoriLlamaIndexConfig(entity_id="u1")

    assert cfg.entity_id == "u1"
    assert cfg.process_id == "llamaindex"
    assert cfg.retrieval_k == 5
    assert cfg.auto_retrieve_memories is True
    assert cfg.enable_chat_memory is True
    assert cfg.enable_agent_memory is True
    assert cfg.session_id is None
    assert cfg.project_id is None


def test_memori_llamaindex_config_custom():
    """MemoriLlamaIndexConfig respects custom values."""
    from config import MemoriLlamaIndexConfig

    cfg = MemoriLlamaIndexConfig(
        entity_id="u2",
        process_id="my-bot",
        retrieval_k=10,
        session_id="sess-abc",
        project_id="proj-xyz",
    )

    assert cfg.retrieval_k == 10
    assert cfg.session_id == "sess-abc"
    assert cfg.project_id == "proj-xyz"


# ---------------------------------------------------------------------------
# memori.llamaindex property
# ---------------------------------------------------------------------------


def test_memori_instance_has_llamaindex_property(memori_instance):
    """Memori instances expose the .llamaindex provider attribute."""
    from memori.llm._providers import LlamaIndex as LlamaIndexProvider

    assert hasattr(memori_instance, "llamaindex")
    assert isinstance(memori_instance.llamaindex, LlamaIndexProvider)


def test_memori_llamaindex_register_deprecated(memori_instance, mocker):
    """memori.llamaindex.register() emits a DeprecationWarning."""
    mock_llm, _ = _make_llamaindex_openai_llm(mocker)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        memori_instance.llamaindex.register(mock_llm)
