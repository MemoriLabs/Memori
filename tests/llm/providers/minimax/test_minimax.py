"""
Tests for MiniMax client detection and registration.

MiniMax provides an OpenAI-compatible API, so clients are created using
the OpenAI SDK with a custom base_url pointing to api.minimax.io.
Memori detects MiniMax as a platform via _detect_platform().
"""

import pytest

from memori._config import Config
from memori.llm._clients import OpenAi, _detect_platform


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def openai_handler(config):
    return OpenAi(config)


class TestMiniMaxPlatformDetection:
    """Tests for MiniMax platform detection via base_url."""

    def test_detect_minimax_io_base_url(self):
        """MiniMax international API should be detected."""

        class FakeClient:
            base_url = "https://api.minimax.io/v1"

        assert _detect_platform(FakeClient()) == "minimax"

    def test_detect_minimaxi_com_base_url(self):
        """MiniMax China API should be detected."""

        class FakeClient:
            base_url = "https://api.minimaxi.com/v1"

        assert _detect_platform(FakeClient()) == "minimax"

    def test_detect_minimax_case_insensitive(self):
        """Detection should be case-insensitive."""

        class FakeClient:
            base_url = "https://api.MiniMax.io/v1"

        assert _detect_platform(FakeClient()) == "minimax"

    def test_no_base_url_returns_none(self):
        """Client without base_url should return None."""

        class FakeClient:
            pass

        assert _detect_platform(FakeClient()) is None

    def test_openai_base_url_not_detected_as_minimax(self):
        """OpenAI base_url should not be detected as MiniMax."""

        class FakeClient:
            base_url = "https://api.openai.com/v1"

        assert _detect_platform(FakeClient()) != "minimax"

    def test_other_platforms_not_detected_as_minimax(self):
        """Other platform URLs should not be detected as MiniMax."""
        urls = [
            "https://api.deepseek.com/v1",
            "https://api.x.ai/v1",
            "https://integrate.api.nvidia.com/v1",
        ]
        for url in urls:

            class FakeClient:
                base_url = url

            result = _detect_platform(FakeClient())
            assert result != "minimax", f"URL {url} should not be detected as minimax"


class TestMiniMaxClientRegistration:
    """Tests for MiniMax client registration with Memori using OpenAI handler."""

    def test_openai_handler_registers_minimax_client(self, openai_handler, mocker):
        """Verify OpenAi handler can register MiniMax-configured clients."""
        mock_client = mocker.MagicMock()
        mock_client._version = "1.0.0"
        mock_client.chat.completions.create = mocker.MagicMock()
        mock_client.beta.chat.completions.parse = mocker.MagicMock()
        mock_client.base_url = "https://api.minimax.io/v1"
        del mock_client._memori_installed

        mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)

        result = openai_handler.register(mock_client)

        assert result is openai_handler
        assert hasattr(mock_client, "_memori_installed")
        assert mock_client._memori_installed is True

    def test_minimax_platform_set_after_registration(self, openai_handler, config, mocker):
        """Verify platform is set to 'minimax' after client registration."""
        mock_client = mocker.MagicMock()
        mock_client._version = "1.0.0"
        mock_client.chat.completions.create = mocker.MagicMock()
        mock_client.beta.chat.completions.parse = mocker.MagicMock()
        mock_client.base_url = "https://api.minimax.io/v1"
        del mock_client._memori_installed

        mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)

        openai_handler.register(mock_client)

        assert config.platform.provider == "minimax"

    def test_minimax_handler_wraps_chat_completions_create(
        self, openai_handler, mocker
    ):
        """Verify handler wraps chat.completions.create for MiniMax."""
        mock_client = mocker.MagicMock()
        mock_client._version = "1.0.0"
        mock_client.beta.chat.completions.parse = mocker.MagicMock()
        mock_client.base_url = "https://api.minimax.io/v1"
        del mock_client._memori_installed

        mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)

        openai_handler.register(mock_client)

        assert hasattr(mock_client.chat, "_completions_create")

    def test_minimax_handler_wraps_beta_parse(self, openai_handler, mocker):
        """Verify handler wraps beta.chat.completions.parse for MiniMax."""
        mock_client = mocker.MagicMock()
        mock_client._version = "1.0.0"
        mock_client.chat.completions.create = mocker.MagicMock()
        mock_client.base_url = "https://api.minimax.io/v1"
        del mock_client._memori_installed

        mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)

        openai_handler.register(mock_client)

        assert hasattr(mock_client.beta, "_chat_completions_parse")

    def test_minimax_registration_is_idempotent(self, openai_handler, mocker):
        """Verify multiple registrations don't re-wrap methods."""
        mock_client = mocker.MagicMock()
        mock_client._version = "1.0.0"
        mock_client.chat.completions.create = mocker.MagicMock()
        mock_client.beta.chat.completions.parse = mocker.MagicMock()
        mock_client.base_url = "https://api.minimax.io/v1"
        del mock_client._memori_installed

        mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)

        openai_handler.register(mock_client)
        original_create = mock_client.chat.completions.create

        openai_handler.register(mock_client)

        assert mock_client.chat.completions.create == original_create
        assert mock_client._memori_installed is True


class TestMiniMaxAutoRegistration:
    """Tests for MiniMax auto-detection via llm.register()."""

    def test_llm_register_auto_detects_minimax_client(self, mocker):
        """Test that llm.register() auto-detects MiniMax client (OpenAI with minimax base_url)."""
        from memori import Memori

        mock_conn = mocker.MagicMock()
        mocker.patch("memori.storage.Manager.start", return_value=mocker.MagicMock())
        mocker.patch(
            "memori.memory.augmentation.Manager.start",
            return_value=mocker.MagicMock(),
        )
        memori_instance = Memori(conn=mock_conn)

        mock_client = mocker.MagicMock()
        type(mock_client).__module__ = "openai"
        mock_client._version = "2.8.1"
        mock_client.chat.completions.create = mocker.MagicMock()
        mock_client.beta.chat.completions.parse = mocker.MagicMock()
        mock_client.base_url = "https://api.minimax.io/v1"
        del mock_client._memori_installed

        mocker.patch("asyncio.get_running_loop", side_effect=RuntimeError)

        result = memori_instance.llm.register(mock_client)

        assert result is memori_instance
        assert hasattr(mock_client, "_memori_installed")
        assert mock_client._memori_installed is True
        assert memori_instance.config.platform.provider == "minimax"
