from memori.llm.helpers.query_extraction import extract_user_query


def test_extract_openai_multimodal_returns_string_not_list():
    """
    Behavioral Guarantee: The extractor must never return a list, even when
    the user payload contains an OpenAI multimodal vision array.
    It should extract text parts and ignore image objects.
    """
    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Can you analyze this diagram?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/diagram.jpg"},
                    },
                    {"type": "text", "text": "What does the top box say?"},
                ],
            }
        ]
    }

    result = extract_user_query(kwargs)

    # Contract 1: Must never return a list
    assert isinstance(result, str)
    
    # Contract 2: Must extract all text and safely ignore non-text blocks
    assert result == "Can you analyze this diagram? What does the top box say?"


def test_extract_text_from_parts_ignores_non_strings():
    """
    Behavioral Guarantee: The helper must never append non-strings to the
    extracted text list, preventing TypeError crashes on string joins.
    """
    from memori.llm.helpers.query_extraction import extract_text_from_parts
    from types import SimpleNamespace

    # Provide a list of parts that includes malformed text values
    parts = [
        "valid string",
        {"text": None},
        {"text": []},
        {"text": {"nested": "dict"}},
        {"text": "valid dict string"},
        SimpleNamespace(text=None),
        SimpleNamespace(text=123),
        SimpleNamespace(text="valid object string"),
    ]

    result = extract_text_from_parts(parts)

    assert isinstance(result, str)
    assert result == "valid string valid dict string valid object string"
