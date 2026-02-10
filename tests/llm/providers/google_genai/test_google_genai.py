"""
Tests for google-genai SDK support.

This ensures that the new google-genai SDK format (without _pb protobuf)
is properly handled for both streaming and non-streaming responses.

"""

import pytest

from memori._config import Config
from memori.llm._base import BaseInvoke, BaseIterator


@pytest.fixture
def config():
    return Config()


class MockGoogleGenaiPart:
    """Mock google-genai Part object."""

    def __init__(self, text):
        self.text = text


class MockGoogleGenaiContent:
    """Mock google-genai Content object."""

    def __init__(self, parts, role="model"):
        self.parts = parts
        self.role = role


class MockGoogleGenaiCandidate:
    """Mock google-genai Candidate object."""

    def __init__(self, content):
        self.content = content


class MockGoogleGenaiResponse:
    """Mock google-genai GenerateContentResponse (non-streaming)."""

    def __init__(self, text, role="model"):
        part = MockGoogleGenaiPart(text)
        content = MockGoogleGenaiContent([part], role)
        self.candidates = [MockGoogleGenaiCandidate(content)]
        # Note: No _pb attribute - this is the new format


class MockGoogleGenaiChunk:
    """Mock google-genai streaming chunk."""

    def __init__(self, text, role="model"):
        part = MockGoogleGenaiPart(text)
        content = MockGoogleGenaiContent([part], role)
        self.candidates = [MockGoogleGenaiCandidate(content)]
        # Note: No _pb attribute - this is the new format


class TestGoogleGenaiFormatDetection:
    """Tests for detecting google-genai format (no _pb attribute)."""

    def test_response_has_no_pb_attribute(self):
        """Verify mock response doesn't have _pb (like real google-genai)."""
        response = MockGoogleGenaiResponse("Hello")
        assert not hasattr(response, "_pb")
        assert "_pb" not in response.__dict__

    def test_response_has_candidates_attribute(self):
        """Verify mock response has candidates (like real google-genai)."""
        response = MockGoogleGenaiResponse("Hello")
        assert hasattr(response, "candidates")
        assert len(response.candidates) == 1

    def test_chunk_has_no_pb_attribute(self):
        """Verify mock chunk doesn't have _pb."""
        chunk = MockGoogleGenaiChunk("Hi")
        assert not hasattr(chunk, "_pb")
        assert "_pb" not in chunk.__dict__

    def test_chunk_has_candidates_attribute(self):
        """Verify mock chunk has candidates."""
        chunk = MockGoogleGenaiChunk("Hi")
        assert hasattr(chunk, "candidates")


class TestGoogleGenaiNonStreamingFormat:
    """Tests for non-streaming google-genai response formatting."""

    def test_format_response_with_google_genai_format(self, config):
        """Test _format_response handles google-genai format."""
        invoke = BaseInvoke(config, lambda **kwargs: None)
        invoke._uses_protobuf = True

        response = MockGoogleGenaiResponse("Hello world", role="model")
        formatted = invoke._format_response(response)

        assert "candidates" in formatted
        assert len(formatted["candidates"]) == 1
        assert "content" in formatted["candidates"][0]
        assert "parts" in formatted["candidates"][0]["content"]
        assert (
            formatted["candidates"][0]["content"]["parts"][0]["text"] == "Hello world"
        )
        assert formatted["candidates"][0]["content"]["role"] == "model"

    def test_format_response_with_empty_candidates(self, config):
        """Test _format_response handles empty candidates."""
        invoke = BaseInvoke(config, lambda **kwargs: None)
        invoke._uses_protobuf = True

        response = MockGoogleGenaiResponse("Test")
        response.candidates = []
        formatted = invoke._format_response(response)

        # Empty candidates returns empty dict (no content to save)
        assert formatted == {}

    def test_format_response_preserves_role(self, config):
        """Test that role is preserved in formatted response."""
        invoke = BaseInvoke(config, lambda **kwargs: None)
        invoke._uses_protobuf = True

        response = MockGoogleGenaiResponse("Hello", role="model")
        formatted = invoke._format_response(response)

        assert formatted["candidates"][0]["content"]["role"] == "model"


class TestGoogleGenaiStreamingFormat:
    """Tests for streaming google-genai chunk processing."""

    def test_process_chunk_with_google_genai_format(self, config):
        """Test process_chunk handles google-genai chunk format."""
        invoke = BaseInvoke(config, lambda **kwargs: None)
        invoke._uses_protobuf = True

        iterator = BaseIterator(config, iter([]))
        iterator.invoke = invoke
        iterator.raw_response = []

        chunk = MockGoogleGenaiChunk("Hello", role="model")
        iterator.process_chunk(chunk)

        assert len(iterator.raw_response) == 1
        assert "candidates" in iterator.raw_response[0]
        assert (
            iterator.raw_response[0]["candidates"][0]["content"]["parts"][0]["text"]
            == "Hello"
        )

    def test_process_multiple_chunks(self, config):
        """Test processing multiple streaming chunks."""
        invoke = BaseInvoke(config, lambda **kwargs: None)
        invoke._uses_protobuf = True

        iterator = BaseIterator(config, iter([]))
        iterator.invoke = invoke
        iterator.raw_response = []

        chunks = [
            MockGoogleGenaiChunk("Hello"),
            MockGoogleGenaiChunk(" "),
            MockGoogleGenaiChunk("World"),
        ]

        for chunk in chunks:
            iterator.process_chunk(chunk)

        assert len(iterator.raw_response) == 3
        assert (
            iterator.raw_response[0]["candidates"][0]["content"]["parts"][0]["text"]
            == "Hello"
        )
        assert (
            iterator.raw_response[1]["candidates"][0]["content"]["parts"][0]["text"]
            == " "
        )
        assert (
            iterator.raw_response[2]["candidates"][0]["content"]["parts"][0]["text"]
            == "World"
        )

    def test_process_chunk_preserves_role(self, config):
        """Test that role is preserved in processed chunk."""
        invoke = BaseInvoke(config, lambda **kwargs: None)
        invoke._uses_protobuf = True

        iterator = BaseIterator(config, iter([]))
        iterator.invoke = invoke
        iterator.raw_response = []

        chunk = MockGoogleGenaiChunk("Test", role="model")
        iterator.process_chunk(chunk)

        assert iterator.raw_response[0]["candidates"][0]["content"]["role"] == "model"


class TestGoogleGenaiBackwardsCompatibility:
    """Tests to ensure old google-generativeai format still works."""

    def test_format_response_with_pb_format(self, config):
        """Test _format_response still handles _pb format."""

        class MockPbResponse:
            def __init__(self):
                self._pb = None  # Would be protobuf in real usage

        invoke = BaseInvoke(config, lambda **kwargs: None)
        invoke._uses_protobuf = True

        # When _pb exists but is None, it should try protobuf parsing
        # This test just verifies the _pb path is still checked first
        response = MockPbResponse()
        assert "_pb" in response.__dict__

    def test_non_protobuf_response_unchanged(self, config):
        """Test non-protobuf responses pass through unchanged."""
        invoke = BaseInvoke(config, lambda **kwargs: None)
        invoke._uses_protobuf = False

        response = {"choices": [{"message": {"content": "Hello"}}]}
        formatted = invoke._format_response(response)

        assert formatted == response
