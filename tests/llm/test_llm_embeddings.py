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
from unittest.mock import patch

import pytest

from memori._config import Config
from memori._rust_core import RustCoreAdapterError
from memori.embeddings import TEI, embed_texts, format_embedding_for_db


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
    mock_bson = mocker.MagicMock()
    mock_binary = mocker.MagicMock()
    mock_bson.Binary.return_value = mock_binary
    mocker.patch.dict("sys.modules", {"bson": mock_bson})

    result = format_embedding_for_db(embedding, "mongodb")

    assert result == mock_binary
    mock_bson.Binary.assert_called_once()
    call_args = mock_bson.Binary.call_args[0][0]
    assert isinstance(call_args, bytes)
    unpacked = struct.unpack("<3f", call_args)
    assert list(unpacked) == pytest.approx(embedding)


def test_format_embedding_for_db_mongodb_no_bson():
    embedding = [1.0, 2.0, 3.0]
    result = format_embedding_for_db(embedding, "mongodb")
    assert isinstance(result, bytes)
    unpacked = struct.unpack("<3f", result)
    assert list(unpacked) == pytest.approx(embedding)


def test_format_embedding_for_db_oceanbase_uses_pyobvector(mocker):
    embedding = [1.0, 2.0, 3.0]
    mock_vector = mocker.MagicMock()
    mock_vector._to_db.return_value = "vector-bytes"
    mock_util = mocker.MagicMock(Vector=mock_vector)
    mock_pkg = mocker.MagicMock(util=mock_util)
    mocker.patch.dict(
        "sys.modules", {"pyobvector": mock_pkg, "pyobvector.util": mock_util}
    )

    result = format_embedding_for_db(embedding, "oceanbase")

    assert result == "vector-bytes"
    mock_vector._to_db.assert_called_once_with(embedding)


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


def test_embed_texts_single_string(mocker):
    cfg = Config()
    native = mocker.patch(
        "memori.embeddings._api.embed_texts_native",
        return_value=[[0.1, 0.2, 0.3]],
    )

    result = embed_texts("Hello world", model=cfg.embeddings.model)

    assert result == [[0.1, 0.2, 0.3]]
    native.assert_called_once_with(["Hello world"], model=cfg.embeddings.model)


def test_embed_texts_list_of_strings(mocker):
    cfg = Config()
    native = mocker.patch(
        "memori.embeddings._api.embed_texts_native",
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )

    result = embed_texts(["Hello", "World"], model=cfg.embeddings.model)

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    native.assert_called_once_with(["Hello", "World"], model=cfg.embeddings.model)


def test_embed_texts_empty_list(mocker):
    cfg = Config()
    native = mocker.patch("memori.embeddings._api.embed_texts_native")

    result = embed_texts([], model=cfg.embeddings.model)

    assert result == []
    native.assert_not_called()


def test_embed_texts_empty_string(mocker):
    cfg = Config()
    native = mocker.patch(
        "memori.embeddings._api.embed_texts_native",
        return_value=[[0.1, 0.2, 0.3]],
    )

    result = embed_texts("", model=cfg.embeddings.model)

    assert result == [[0.1, 0.2, 0.3]]
    native.assert_called_once_with([""], model=cfg.embeddings.model)


def test_embed_texts_filters_empty_strings(mocker):
    cfg = Config()
    native = mocker.patch(
        "memori.embeddings._api.embed_texts_native",
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )

    result = embed_texts(["Hello", "", "World", ""], model=cfg.embeddings.model)

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    native.assert_called_once_with(["Hello", "World"], model=cfg.embeddings.model)


def test_embed_texts_custom_model(mocker):
    native = mocker.patch(
        "memori.embeddings._api.embed_texts_native",
        return_value=[[0.1, 0.2, 0.3]],
    )

    result = embed_texts("test", model="custom-model")

    native.assert_called_once_with(["test"], model="custom-model")
    assert result == [[0.1, 0.2, 0.3]]


def test_embed_texts_propagates_native_errors(mocker):
    native = mocker.patch(
        "memori.embeddings._api.embed_texts_native",
        side_effect=RuntimeError("unsupported native model"),
    )

    with pytest.raises(RuntimeError, match="unsupported native model"):
        embed_texts("test", model="unsupported-model")

    native.assert_called_once_with(["test"], model="unsupported-model")


def test_embed_texts_propagates_unavailable_native_backend(mocker):
    native = mocker.patch(
        "memori.embeddings._api.embed_texts_native",
        side_effect=RustCoreAdapterError("Rust embeddings are unavailable"),
    )

    with pytest.raises(RustCoreAdapterError, match="Rust embeddings are unavailable"):
        embed_texts("test", model="all-MiniLM-L6-v2")

    native.assert_called_once_with(["test"], model="all-MiniLM-L6-v2")


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
            async_=True,
        )

        assert result == [[0.1, 0.2, 0.3]]


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
            async_=True,
        )

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.asyncio
async def test_embed_texts_async_custom_model():
    mock_result = [[0.1, 0.2, 0.3]]

    async def mock_run_in_executor(executor, func, *args):
        return mock_result

    with patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = mock_run_in_executor

        result = await embed_texts("test", model="custom-model", async_=True)

        assert result == [[0.1, 0.2, 0.3]]


def test_embed_texts_uses_tei_remote(mocker):
    tei = TEI(url="http://localhost:8080/v1/embeddings")
    mock_post = mocker.patch("memori.embeddings._tei.requests.post")
    mock_response = mocker.Mock()
    mock_response.json.side_effect = [
        {"data": [{"embedding": [1.0, 0.0]}]},
        {"data": [{"embedding": [0.0, 1.0]}]},
    ]
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    out = embed_texts(["a", "b"], model="tei-model", tei=tei)

    assert out == [[1.0, 0.0], [0.0, 1.0]]
    assert mock_post.call_count == 2
    first_kwargs = mock_post.call_args_list[0].kwargs
    second_kwargs = mock_post.call_args_list[1].kwargs
    assert first_kwargs["json"] == {"input": ["a"], "model": "tei-model"}
    assert second_kwargs["json"] == {"input": ["b"], "model": "tei-model"}
    assert first_kwargs["timeout"] == 30.0
    assert second_kwargs["timeout"] == 30.0


def test_embed_texts_tei_token_chunks_and_pools(mocker):
    tei = TEI(url="http://localhost:8080/v1/embeddings")

    tokenizer = mocker.Mock()
    tokenizer.return_value = {"input_ids": [[0, 1, 2, 3]]}
    tokenizer.decode.side_effect = ["c1", "c2"]

    mock_post = mocker.patch("memori.embeddings._tei.requests.post")
    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]
    }
    mock_post.return_value = mock_response

    out = embed_texts(
        "abcd", model="tei-model", tei=tei, tokenizer=tokenizer, chunk_size=2
    )

    assert len(out) == 1
    assert out[0] == pytest.approx([0.707106, 0.707106], rel=1e-5)
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"input": ["c1", "c2"], "model": "tei-model"}
    assert kwargs["timeout"] == 30.0
