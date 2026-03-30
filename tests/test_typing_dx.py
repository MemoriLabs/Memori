import inspect
from pathlib import Path

import memori
from memori import LlmRegistry, Memori


def test_package_includes_py_typed_marker() -> None:
    marker = Path(memori.__file__).with_name("py.typed")
    assert marker.is_file()


def test_public_api_has_annotations_for_primary_methods() -> None:
    annotated_methods = {
        Memori.__init__: ["conn", "debug_truncate"],
        Memori.attribution: ["entity_id", "process_id"],
        Memori.new_session: [],
        Memori.set_session: ["session_id"],
        Memori.recall: ["query", "limit"],
        Memori.embed_texts: ["texts", "async_"],
        LlmRegistry.register: [
            "client",
            "openai_chat",
            "claude",
            "gemini",
            "xai",
            "chatbedrock",
            "chatgooglegenai",
            "chatopenai",
            "chatvertexai",
        ],
    }

    for method, param_names in annotated_methods.items():
        signature = inspect.signature(method)
        assert signature.return_annotation is not inspect.Signature.empty
        for name in param_names:
            assert signature.parameters[name].annotation is not inspect.Signature.empty


def test_public_api_has_hover_docstrings_for_primary_methods() -> None:
    documented_methods = [
        Memori.__init__,
        Memori.attribution,
        Memori.new_session,
        Memori.set_session,
        Memori.recall,
        Memori.embed_texts,
        LlmRegistry.register,
    ]

    for method in documented_methods:
        assert inspect.getdoc(method)
