from unittest.mock import Mock, patch

import memori.embeddings._sentence_transformers as st_core


def test_get_model_downloads_from_huggingface():
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformer"
    ) as mock_transformer:
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        result = st_core.get_sentence_transformers_embedder(
            "all-mpnet-base-v2"
        )._get_model()

        assert result is mock_model
        mock_transformer.assert_called_once_with("all-mpnet-base-v2")


def test_get_model_caching():
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformer"
    ) as mock_transformer:
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        embedder = st_core.get_sentence_transformers_embedder("test-model")
        result1 = embedder._get_model()
        result2 = embedder._get_model()

        assert result1 is result2
        mock_transformer.assert_called_once()


def test_get_model_different_models():
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformer"
    ) as mock_transformer:
        mock_model1 = Mock()
        mock_model2 = Mock()
        mock_transformer.side_effect = [mock_model1, mock_model2]

        result1 = st_core.get_sentence_transformers_embedder("model-1")._get_model()
        result2 = st_core.get_sentence_transformers_embedder("model-2")._get_model()

        assert result1 is not result2
        assert mock_transformer.call_count == 2
