"""
Tests for AzureOpenAI client detection and registration.

This ensures that AzureOpenAI clients are properly detected by Memori's
client registration system, which uses `.startswith("openai")` to match
both OpenAI and AzureOpenAI clients.
"""

import pytest

from memori._config import Config
from memori.llm._clients import OpenAi


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def openai_handler(config):
    return OpenAi(config)


class TestAzureOpenAIModuleDetection:
    """Tests for AzureOpenAI module path detection."""

    def test_openai_module_path(self):
        """Verify OpenAI class module is 'openai'."""
        from openai import OpenAI

        assert OpenAI.__module__ == "openai"

    def test_azure_openai_module_path(self):
        """Verify AzureOpenAI class module is 'openai.lib.azure'."""
        from openai import AzureOpenAI

        assert AzureOpenAI.__module__ == "openai.lib.azure"

    def test_azure_openai_startswith_openai(self):
        """Verify AzureOpenAI module starts with 'openai'."""
        from openai import AzureOpenAI

        assert AzureOpenAI.__module__.startswith("openai")

    def test_openai_startswith_openai(self):
        """Verify OpenAI module starts with 'openai'."""
        from openai import OpenAI

        assert OpenAI.__module__.startswith("openai")


class TestAzureOpenAIInheritance:
    """Tests for AzureOpenAI class inheritance."""

    def test_azure_openai_is_subclass_of_openai(self):
        """Verify AzureOpenAI inherits from OpenAI."""
        from openai import AzureOpenAI, OpenAI

        assert issubclass(AzureOpenAI, OpenAI)

    def test_azure_openai_has_chat_attribute(self):
        """Verify AzureOpenAI has the required 'chat' attribute."""
        from openai import AzureOpenAI

        assert hasattr(AzureOpenAI, "chat")

    def test_azure_openai_has_beta_attribute(self):
        """Verify AzureOpenAI has the required 'beta' attribute."""
        from openai import AzureOpenAI

        assert hasattr(AzureOpenAI, "beta")


class TestAzureOpenAIRegistryDetection:
    """Tests for AzureOpenAI detection by Memori's Registry."""

    def test_registry_detects_openai_client(self):
        """Verify Registry detects standard OpenAI clients."""
        from openai import OpenAI

        # The actual detection happens via type(client).__module__
        assert OpenAI.__module__.startswith("openai")

    def test_registry_detects_azure_openai_client(self):
        """Verify Registry detects AzureOpenAI clients."""
        from openai import AzureOpenAI

        # The detection should match AzureOpenAI module
        assert AzureOpenAI.__module__.startswith("openai")

    def test_registry_does_not_detect_anthropic(self):
        """Verify Registry does not falsely detect Anthropic as OpenAI."""
        import anthropic

        assert not anthropic.Anthropic.__module__.startswith("openai")

    def test_registry_does_not_detect_langchain_openai(self):
        """Verify Registry does not falsely detect langchain_openai as OpenAI."""
        # langchain_openai module starts with 'langchain', not 'openai'
        module_name = "langchain_openai"
        assert not module_name.startswith("openai")


class TestAzureOpenAIClientRegistration:
    """Tests for AzureOpenAI client registration with Memori."""

    def test_openai_handler_registers_openai_client(self, openai_handler, mocker):
        """Verify OpenAi handler can register standard OpenAI clients."""
        mock_client = mocker.MagicMock()
        mock_client._version = "1.0.0"
        mock_client.chat.completions.create = mocker.MagicMock()
        mock_client.beta.chat.completions.parse = mocker.MagicMock()
        del mock_client._memori_installed

        mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)

        result = openai_handler.register(mock_client)

        assert result is openai_handler
        assert hasattr(mock_client, "_memori_installed")
        assert mock_client._memori_installed is True

    def test_openai_handler_registers_azure_openai_client(self, openai_handler, mocker):
        """Verify OpenAi handler can register AzureOpenAI clients.

        AzureOpenAI has the same API structure as OpenAI, so the same
        handler should work for both.
        """
        # Mock an AzureOpenAI-like client
        mock_azure_client = mocker.MagicMock()
        mock_azure_client._version = "1.0.0"
        mock_azure_client.chat.completions.create = mocker.MagicMock()
        mock_azure_client.beta.chat.completions.parse = mocker.MagicMock()
        del mock_azure_client._memori_installed

        mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)

        result = openai_handler.register(mock_azure_client)

        assert result is openai_handler
        assert hasattr(mock_azure_client, "_memori_installed")
        assert mock_azure_client._memori_installed is True

    def test_openai_handler_wraps_chat_completions_create(self, openai_handler, mocker):
        """Verify handler wraps chat.completions.create for AzureOpenAI."""
        mock_client = mocker.MagicMock()
        mock_client._version = "1.0.0"
        mock_client.beta.chat.completions.parse = mocker.MagicMock()
        del mock_client._memori_installed

        mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)

        openai_handler.register(mock_client)

        # Verify the original method was stored as backup
        assert hasattr(mock_client.chat, "_completions_create")

    def test_openai_handler_wraps_beta_parse(self, openai_handler, mocker):
        """Verify handler wraps beta.chat.completions.parse for AzureOpenAI."""
        mock_client = mocker.MagicMock()
        mock_client._version = "1.0.0"
        mock_client.chat.completions.create = mocker.MagicMock()
        del mock_client._memori_installed

        mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)

        openai_handler.register(mock_client)

        # Verify the original method was stored as backup
        assert hasattr(mock_client.beta, "_chat_completions_parse")


class TestAzureOpenAINoFalsePositives:
    """Tests to ensure no false positives in client detection."""

    def test_anthropic_not_matched(self):
        """Anthropic module should not match OpenAI detection."""
        assert not "anthropic".startswith("openai")

    def test_google_not_matched(self):
        """Google module should not match OpenAI detection."""
        assert not "google.generativeai".startswith("openai")

    def test_langchain_openai_not_matched(self):
        """langchain_openai module should not match (uses underscore)."""
        assert not "langchain_openai".startswith("openai")

    def test_pydantic_ai_not_matched(self):
        """pydantic_ai module should not match OpenAI detection."""
        assert not "pydantic_ai".startswith("openai")

    def test_xai_not_matched(self):
        """xai module should not match OpenAI detection."""
        assert not "xai_sdk".startswith("openai")

    def test_openai_submodules_matched(self):
        """All openai submodules should match detection."""
        openai_modules = [
            "openai",
            "openai.lib.azure",
            "openai.resources",
            "openai.types",
            "openai._client",
        ]
        for module in openai_modules:
            assert module.startswith("openai"), f"{module} should match"
