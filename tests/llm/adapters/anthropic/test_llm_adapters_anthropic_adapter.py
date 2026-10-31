from memori.llm.adapters.anthropic._adapter import Adapter


def test_get_formatted_query():
    assert Adapter().get_formatted_query({}) == []
    assert Adapter().get_formatted_query({"conversation": {"query": {}}}) == []

    assert Adapter().get_formatted_query(
        {
            "conversation": {
                "query": {
                    "messages": [
                        {"content": "abc", "role": "user"},
                        {"content": "def", "role": "assistant"},
                    ]
                }
            }
        }
    ) == [{"content": "abc", "role": "user"}, {"content": "def", "role": "assistant"}]


def test_get_formatted_response_unstreamed():
    assert Adapter().get_formatted_response({}) == []
    assert Adapter().get_formatted_query({"conversation": {"response": {}}}) == []

    assert Adapter().get_formatted_response(
        {
            "conversation": {
                "response": {
                    "content": [
                        {"text": "abc", "type": "text"},
                        {"text": "def", "type": "text"},
                    ],
                    "role": "user",
                }
            }
        }
    ) == [
        {"role": "user", "text": "abc", "type": "text"},
        {"role": "user", "text": "def", "type": "text"},
    ]


def test_get_formatted_response_unstreamed_skips_non_text_blocks():
    assert Adapter().get_formatted_response(
        {
            "conversation": {
                "response": {
                    "content": [
                        {"thinking": "...", "signature": "sig", "type": "thinking"},
                        {"text": "Let me check the weather.", "type": "text"},
                        {
                            "id": "toolu_01A09q90qw90lq9178350182",
                            "input": {"location": "San Francisco, CA"},
                            "name": "get_current_weather",
                            "type": "tool_use",
                        },
                    ],
                    "role": "assistant",
                }
            }
        }
    ) == [
        {"role": "assistant", "text": "Let me check the weather.", "type": "text"},
    ]


def test_get_formatted_query_with_injected_messages():
    assert Adapter().get_formatted_query(
        {
            "conversation": {
                "query": {
                    "_memori_injected_count": 2,
                    "messages": [
                        {"content": "injected 1", "role": "user"},
                        {"content": "injected 2", "role": "assistant"},
                        {"content": "new message", "role": "user"},
                        {"content": "new response", "role": "assistant"},
                    ],
                }
            }
        }
    ) == [
        {"content": "new message", "role": "user"},
        {"content": "new response", "role": "assistant"},
    ]
