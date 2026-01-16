r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                 perfectam memoriam
                      memorilabs.ai
"""

import struct
from unittest.mock import Mock, patch

import numpy as np
import pytest

import memori.embeddings._sentence_transformers as st_core
from memori._config import Config
from memori.embeddings import embed_texts, format_embedding_for_db


def test_format_embedding_for_db_mysql():
    embedding = [1.0, 2.0, 3.0]
    result = format_embedding_for_db(embedding, "mysql")
    assert isinstance(result, bytes)
    unpacked = struct.unpack("<3f", result)
    assert list(unpacked) == pytest.approx(embedding)


def test_format_embedding_for_db_postgresql():
    embedding = [1.0, 2.0, 3.0]
    result = format_embedding_for_db(embedding, "postgresql")
    assert isinstance(result, bytes)
    unpacked = struct.unpack("<3f", result)
    assert list(unpacked) == pytest.approx(embedding)


def test_format_embedding_for_db_cockroachdb():
    embedding = [1.0, 2.0, 3.0]
    result = format_embedding_for_db(embedding, "cockroachdb")
    assert isinstance(result, bytes)
    unpacked = struct.unpack("<3f", result)
    assert list(unpacked) == pytest.approx(embedding)


def test_format_embedding_for_db_sqlite():
    embedding = [1.0, 2.0, 3.0]
    result = format_embedding_for_db(embedding, "sqlite")
    assert isinstance(result, bytes)
    unpacked = struct.unpack("<3f", result)
    assert list(unpacked) == pytest.approx(embedding)


def test_format_embedding_for_db_mongodb(mocker):
    embedding = [1.0, 2.0, 3.0]
    # Mock bson.Binary to test MongoDB path
    mock_bson = mocker.MagicMock()
    mock_binary = mocker.MagicMock()
    mock_bson.Binary.return_value = mock_binary
    mocker.patch.dict("sys.modules", {"bson": mock_bson})

    result = format_embedding_for_db(embedding, "mongodb")
    # Should return bson.Binary wrapped bytes
    assert result == mock_binary
    # Verify bson.Binary was called with packed bytes
    mock_bson.Binary.assert_called_once()
    call_args = mock_bson.Binary.call_args[0][0]
    assert isinstance(call_args, bytes)
    unpacked = struct.unpack("<3f", call_args)
    assert list(unpacked) == pytest.approx(embedding)


def test_format_embedding_for_db_mongodb_no_bson():
    """Test MongoDB fallback when bson is not available."""
    embedding = [1.0, 2.0, 3.0]
    # Don't mock bson, let ImportError happen
    result = format_embedding_for_db(embedding, "mongodb")
    # Should return raw bytes as fallback
    assert isinstance(result, bytes)
    unpacked = struct.unpack("<3f", result)
    assert list(unpacked) == pytest.approx(embedding)


def test_format_embedding_for_db_unknown_dialect():
    embedding = [1.0, 2.0, 3.0]
    result = format_embedding_for_db(embedding, "unknown_db")
    assert isinstance(result, bytes)
    unpacked = struct.unpack("<3f", result)
    assert list(unpacked) == pytest.approx(embedding)


def test_format_embedding_for_db_high_dimensional():
    embedding = [float(i) for i in range(768)]
    result_mysql = format_embedding_for_db(embedding, "mysql")
    assert isinstance(result_mysql, bytes)
    unpacked_mysql = struct.unpack("<768f", result_mysql)
    assert list(unpacked_mysql) == pytest.approx(embedding)

    result_postgres = format_embedding_for_db(embedding, "postgresql")
    assert isinstance(result_postgres, bytes)
    unpacked_postgres = struct.unpack("<768f", result_postgres)
    assert list(unpacked_postgres) == pytest.approx(embedding)


def test_get_model_caches_model():
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformer"
    ) as mock_transformer:
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        embedder = st_core.get_sentence_transformers_embedder("test-model")
        model1 = embedder._get_model()
        model2 = embedder._get_model()

        assert model1 is model2
        mock_transformer.assert_called_once_with("test-model")


def test_get_model_different_models():
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformer"
    ) as mock_transformer:
        mock_model_1 = Mock()
        mock_model_2 = Mock()
        mock_transformer.side_effect = [mock_model_1, mock_model_2]

        model1 = st_core.get_sentence_transformers_embedder("model-1")._get_model()
        model2 = st_core.get_sentence_transformers_embedder("model-2")._get_model()

        assert model1 is not model2
        assert mock_transformer.call_count == 2


def test_embed_texts_single_string():
    cfg = Config()
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model"
    ) as mock_get_model:
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        result = embed_texts(
            "Hello world",
            model=cfg.embeddings.model,
            fallback_dimension=cfg.embeddings.fallback_dimension,
        )

        assert len(result) == 1
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])
        mock_model.encode.assert_called_once()


def test_embed_texts_list_of_strings():
    cfg = Config()
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model"
    ) as mock_get_model:
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        result = embed_texts(
            ["Hello", "World"],
            model=cfg.embeddings.model,
            fallback_dimension=cfg.embeddings.fallback_dimension,
        )

        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])
        assert result[1] == pytest.approx([0.4, 0.5, 0.6])


def test_embed_texts_empty_list():
    cfg = Config()
    st_core._EMBEDDER_CACHE.clear()
    result = embed_texts(
        [],
        model=cfg.embeddings.model,
        fallback_dimension=cfg.embeddings.fallback_dimension,
    )
    assert result == []


def test_embed_texts_empty_string():
    cfg = Config()
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model"
    ) as mock_get_model:
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        result = embed_texts(
            "",
            model=cfg.embeddings.model,
            fallback_dimension=cfg.embeddings.fallback_dimension,
        )

        assert len(result) == 1
        mock_model.encode.assert_called_once_with(
            [""], convert_to_numpy=True, normalize_embeddings=True
        )


def test_embed_texts_filters_empty_strings():
    cfg = Config()
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model"
    ) as mock_get_model:
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        result = embed_texts(
            ["Hello", "", "World", ""],
            model=cfg.embeddings.model,
            fallback_dimension=cfg.embeddings.fallback_dimension,
        )

        assert len(result) == 2
        mock_model.encode.assert_called_once_with(
            ["Hello", "World"], convert_to_numpy=True, normalize_embeddings=True
        )


def test_embed_texts_chunks_long_text_and_pools(mocker):
    st_core._EMBEDDER_CACHE.clear()
    mock_model = mocker.Mock()
    mock_model.get_max_seq_length.return_value = 6
    mock_model.tokenizer = mocker.Mock()
    mock_model.tokenizer.return_value = {"input_ids": list(range(20))}
    mock_model.tokenizer.decode.side_effect = ["c1", "c2", "c3", "c4", "c5"]

    # Chunk embeddings: unit vectors along two axes to make mean+renorm predictable.
    mock_model.encode.return_value = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )

    mocker.patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model",
        return_value=mock_model,
    )

    out = embed_texts(["x" * 10], model="test-model", fallback_dimension=2)

    assert len(out) == 1
    # Mean is [0.6, 0.4], then renormalized to unit length.
    assert out[0] == pytest.approx([0.832050, 0.554700], rel=1e-4)
    mock_model.encode.assert_called_with(
        ["c1", "c2", "c3", "c4", "c5"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def test_embed_texts_custom_model():
    st_core._EMBEDDER_CACHE.clear()
    with patch("memori.embeddings._api.get_sentence_transformers_embedder") as mock_get:
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_get.return_value = mock_embedder

        result = embed_texts("test", model="custom-model", fallback_dimension=1024)

        mock_get.assert_called_once_with("custom-model")
        mock_embedder.embed.assert_called_once_with(["test"], fallback_dimension=1024)
        assert len(result) == 1


def test_embed_texts_model_load_failure():
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model"
    ) as mock_get_model:
        mock_get_model.side_effect = OSError("Model not found")

        result = embed_texts(
            ["Hello", "World"], model="missing-model", fallback_dimension=7
        )

        assert len(result) == 2
        assert result[0] == [0.0] * 7
        assert result[1] == [0.0] * 7


def test_embed_texts_encode_failure():
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model"
    ) as mock_get_model:
        mock_model = Mock()
        mock_model.encode.side_effect = RuntimeError("Encoding failed")
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_get_model.return_value = mock_model

        result = embed_texts(["Hello"], model="test-model", fallback_dimension=1024)

        assert len(result) == 1
        assert result[0] == [0.0] * 384


def test_embed_texts_shape_error_retries_and_pools(mocker):
    st_core._EMBEDDER_CACHE.clear()
    mock_model = mocker.Mock()
    # First call (convert_to_numpy=True) fails like numpy stack error
    # Then we retry per-text (convert_to_numpy=True) which succeeds.
    mock_model.encode.side_effect = [
        ValueError("all input arrays must have the same shape"),
        np.ones((1, 4), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
    ]
    mock_model.get_sentence_embedding_dimension.return_value = 4
    mocker.patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model",
        return_value=mock_model,
    )

    out = embed_texts(["a", "b"], model="test-model", fallback_dimension=1024)

    assert out == [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    assert mock_model.encode.call_count == 3


def test_embed_texts_encode_failure_with_dimension_error():
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model"
    ) as mock_get_model:
        mock_model = Mock()
        mock_model.encode.side_effect = RuntimeError("Encoding failed")
        mock_model.get_sentence_embedding_dimension.side_effect = RuntimeError(
            "Dimension error"
        )
        mock_get_model.return_value = mock_model

        result = embed_texts(["Hello"], model="test-model", fallback_dimension=11)

        assert len(result) == 1
        assert result[0] == [0.0] * 11


def test_embed_texts_model_load_runtime_error():
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model"
    ) as mock_get_model:
        mock_get_model.side_effect = RuntimeError("Runtime error")

        result = embed_texts("test", model="test-model", fallback_dimension=9)

        assert len(result) == 1
        assert result[0] == [0.0] * 9


def test_embed_texts_model_load_value_error():
    st_core._EMBEDDER_CACHE.clear()
    with patch(
        "memori.embeddings._sentence_transformers.SentenceTransformersEmbedder._get_model"
    ) as mock_get_model:
        mock_get_model.side_effect = ValueError("Value error")

        result = embed_texts("test", model="test-model", fallback_dimension=9)

        assert len(result) == 1
        assert result[0] == [0.0] * 9


@pytest.mark.asyncio
async def test_embed_texts_async_single_string():
    cfg = Config()
    mock_result = [[0.1, 0.2, 0.3]]

    async def mock_run_in_executor(executor, func, *args):
        return mock_result

    with patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = mock_run_in_executor

        result = await embed_texts(
            "Hello world",
            model=cfg.embeddings.model,
            fallback_dimension=cfg.embeddings.fallback_dimension,
            async_=True,
        )

        assert len(result) == 1
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_embed_texts_async_list():
    cfg = Config()
    mock_result = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    async def mock_run_in_executor(executor, func, *args):
        return mock_result

    with patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = mock_run_in_executor

        result = await embed_texts(
            ["Hello", "World"],
            model=cfg.embeddings.model,
            fallback_dimension=cfg.embeddings.fallback_dimension,
            async_=True,
        )

        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])
        assert result[1] == pytest.approx([0.4, 0.5, 0.6])


@pytest.mark.asyncio
async def test_embed_texts_async_custom_model():
    mock_result = [[0.1, 0.2, 0.3]]

    async def mock_run_in_executor(executor, func, *args):
        return mock_result

    with patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = mock_run_in_executor

        result = await embed_texts(
            "test", model="custom-model", fallback_dimension=1024, async_=True
        )

        assert len(result) == 1
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])
