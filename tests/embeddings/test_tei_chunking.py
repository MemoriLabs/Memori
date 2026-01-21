r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                 perfectam memoriam
                      memorilabs.ai
"""

import numpy as np
import pytest

from memori.embeddings._chunking import chunk_text_by_tokens
from memori.embeddings._tei_embed import embed_texts_via_tei


def test_chunk_text_by_tokens_list_input_ids(mocker):
    tokenizer = mocker.Mock()
    tokenizer.return_value = {"input_ids": [[0, 1, 2, 3]]}
    tokenizer.decode.side_effect = ["c1", "c2"]

    out = chunk_text_by_tokens(text="abcd", tokenizer=tokenizer, chunk_size=2)

    assert out == ["c1", "c2"]


def test_chunk_text_by_tokens_numpy_input_ids(mocker):
    tokenizer = mocker.Mock()
    tokenizer.return_value = {"input_ids": np.array([[0, 1, 2, 3]], dtype=np.int64)}
    tokenizer.decode.side_effect = ["c1", "c2"]

    out = chunk_text_by_tokens(text="abcd", tokenizer=tokenizer, chunk_size=2)

    assert out == ["c1", "c2"]


def test_chunk_text_by_tokens_falls_back_when_missing_input_ids(mocker):
    tokenizer = mocker.Mock()
    tokenizer.return_value = {}
    tokenizer.decode.return_value = "unused"

    out = chunk_text_by_tokens(text="abcd", tokenizer=tokenizer, chunk_size=2)

    assert out == ["abcd"]


def test_embed_texts_via_tei_no_tokenizer_calls_server_once(mocker):
    tei = mocker.Mock()
    tei.embed.side_effect = [[[1.0, 2.0]], [[3.0, 4.0]]]

    out = [
        embed_texts_via_tei(text=t, model="m", tei=tei, tokenizer=None)
        for t in ["a", "b"]
    ]

    assert out == [[1.0, 2.0], [3.0, 4.0]]
    assert tei.embed.call_count == 2
    tei.embed.assert_any_call(["a"], model="m")
    tei.embed.assert_any_call(["b"], model="m")


def test_embed_texts_via_tei_tokenizer_chunks_and_pools(mocker):
    tei = mocker.Mock()
    # Two chunks => mean([1,0],[0,1]) renorm => [0.707..., 0.707...]
    tei.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]

    tokenizer = mocker.Mock()
    tokenizer.return_value = {"input_ids": [[0, 1, 2, 3]]}
    tokenizer.decode.side_effect = ["c1", "c2"]

    out = embed_texts_via_tei(
        text="abcd",
        model="m",
        tei=tei,
        tokenizer=tokenizer,
        chunk_size=2,
    )

    assert out == pytest.approx([0.707106, 0.707106], rel=1e-5)
    tei.embed.assert_called_once_with(["c1", "c2"], model="m")
