from memori._embedding_input import is_embeddable_text, normalize_embed_texts_input
from memori.embeddings._utils import prepare_text_inputs


def test_is_embeddable_text_matches_rust_visibility_rules():
    assert is_embeddable_text("hello")
    assert is_embeddable_text("  a  ")
    assert not is_embeddable_text("")
    assert not is_embeddable_text("   ")
    assert not is_embeddable_text("\u200b")
    assert is_embeddable_text("\u200bword")


def test_normalize_embed_texts_input_preserves_order():
    assert normalize_embed_texts_input("solo") == ["solo"]
    assert normalize_embed_texts_input(["a", "", "b"]) == ["a", "", "b"]


def test_prepare_text_inputs_filters_non_embeddable_text():
    assert prepare_text_inputs(["hello", "", "   ", "world"]) == ["hello", "world"]
