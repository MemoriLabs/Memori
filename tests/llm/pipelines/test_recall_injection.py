from copy import deepcopy

import pytest

from memori._config import Config
from memori.llm._base import BaseInvoke
from memori.llm.pipelines.recall_injection import (
    format_recalled_fact_lines,
    format_recalled_summary_lines,
    inject_recalled_facts,
)

# --- format_recalled_fact_lines tests ---


def test_fact_lines_from_plain_string():
    result = format_recalled_fact_lines(["User likes pizza"])

    assert result == ["- User likes pizza"]


def test_format_recalled_fact_lines_skips_empty_string():
    result = format_recalled_fact_lines([""])

    assert result == []


def test_format_recalled_fact_lines_from_dict_with_date():
    fact_dict = {
        "content": "User prefers concise answers",
        "date_created": "2026-05-12T10:30:00Z",
    }
    result = format_recalled_fact_lines([fact_dict])

    assert result == ["- User prefers concise answers. Stated at 2026-05-12 10:30"]


def test_format_recalled_fact_lines_from_dict_without_date():
    fact_dict = {
        "content": "User prefers concise answers",
        "date_created": None,
    }
    result = format_recalled_fact_lines([fact_dict])

    assert result == ["- User prefers concise answers"]


def test_format_recalled_fact_lines_skips_dict_with_empty_content():
    fact_dict = {
        "content": "",
        "date_created": None,
    }
    result = format_recalled_fact_lines([fact_dict])

    assert result == []


def test_format_recalled_fact_lines_from_object():
    class FactObject:
        content = "User prefers structured data"
        date_created = "2026-05-13T12:00:00Z"

    result = format_recalled_fact_lines([FactObject()])

    assert result == ["- User prefers structured data. Stated at 2026-05-13 12:00"]


def test_format_recalled_fact_lines_skips_object_missing_attributes():
    class EmptyObject:
        pass

    result = format_recalled_fact_lines([EmptyObject()])

    assert result == []


def test_format_recalled_fact_lines_mixed_types():
    class FactObject:
        content = "Third fact"
        date_created = "2026-05-14T10:00:00Z"

    facts = [
        "First fact",
        {"content": "Second fact", "date_created": "2026-05-13T10:00:00Z"},
        FactObject(),
    ]

    result = format_recalled_fact_lines(facts)

    assert result == [
        "- First fact",
        "- Second fact. Stated at 2026-05-13 10:00",
        "- Third fact. Stated at 2026-05-14 10:00",
    ]


# --- format_recalled_summary_lines tests ---


def test_format_recalled_summary_lines_with_date():
    fact = {
        "content": "Irrelevant for summary extraction",
        "summaries": [
            {
                "content": "User is a software engineer.",
                "date_created": "2026-05-14T10:30:00Z",
            }
        ],
    }

    result = format_recalled_summary_lines([fact])

    assert result == ["- [2026-05-14 10:30]\n  User is a software engineer."]


def test_format_recalled_summary_lines_deduplicates():
    facts = [
        {
            "content": "Fact A",
            "summaries": [{"content": "Shared topic summary."}],
        },
        {
            "content": "Fact B",
            "summaries": [{"content": "Shared topic summary."}],
        },
    ]

    result = format_recalled_summary_lines(facts)

    assert result == ["- Shared topic summary."]


def test_format_recalled_summary_lines_orders_primary_first():
    facts = [
        {
            "content": "Fact 1",
            "summaries": [
                {"content": "Fact 1 Primary Summary"},
                {"content": "Fact 1 Additional Summary"},
            ],
        },
        {
            "content": "Fact 2",
            "summaries": [
                {"content": "Fact 2 Primary Summary"},
                {"content": "Fact 2 Additional Summary"},
            ],
        },
    ]

    result = format_recalled_summary_lines(facts)

    assert result == [
        "- Fact 1 Primary Summary",
        "- Fact 2 Primary Summary",
        "- Fact 1 Additional Summary",
        "- Fact 2 Additional Summary",
    ]


@pytest.fixture
def recalled_invoke(mocker):
    config = Config()
    config.cloud = True
    config.entity_id = "public-test-user"
    config.llm.provider = "openai"
    config.recall_relevance_threshold = 0.0
    recall = mocker.patch("memori.memory.recall.Recall")
    recall.return_value.search_facts.return_value = {
        "facts": [{"content": "User prefers concise replies", "similarity": 1.0}],
        "messages": [],
    }
    return BaseInvoke(config, "create"), recall


@pytest.mark.parametrize(
    "parts",
    [
        [],
        [{"type": "text", "text": "Base instructions"}],
        [
            {
                "type": "text",
                "text": "Base instructions",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        ],
    ],
)
def test_inject_recalled_facts_preserves_structured_system_parts(
    recalled_invoke, parts
):
    invoke, recall = recalled_invoke
    original = deepcopy(parts)
    user_message = {"role": "user", "content": "Hello"}
    kwargs = {
        "messages": [{"role": "system", "content": parts}, user_message],
        "model": "test-model",
    }

    result = inject_recalled_facts(invoke, kwargs)

    assert result is kwargs
    assert result["messages"][1] is user_message
    assert result["model"] == "test-model"
    content = result["messages"][0]["content"]
    assert content[:-1] == original
    assert all(
        actual is expected for actual, expected in zip(content, parts, strict=False)
    )
    assert parts == original
    assert content[-1]["type"] == "text"
    assert "<memori_context>" in content[-1]["text"]
    assert "User prefers concise replies" in content[-1]["text"]
    recall.return_value.search_facts.assert_called_once_with("Hello")


def test_inject_recalled_facts_string_system_matches_structured_text(recalled_invoke):
    invoke, _ = recalled_invoke
    structured = {
        "messages": [
            {"role": "system", "content": []},
            {"role": "user", "content": "Hello"},
        ]
    }
    plain = {
        "messages": [
            {"role": "system", "content": "Base instructions"},
            {"role": "user", "content": "Hello"},
        ]
    }

    inject_recalled_facts(invoke, structured)
    inject_recalled_facts(invoke, plain)

    assert plain["messages"][0]["content"] == (
        "Base instructions" + structured["messages"][0]["content"][0]["text"]
    )


def test_inject_recalled_facts_no_facts_leaves_structured_content_untouched(
    recalled_invoke,
):
    invoke, recall = recalled_invoke
    recall.return_value.search_facts.return_value = {"facts": [], "messages": []}
    parts = [{"type": "text", "text": "Base instructions"}]
    kwargs = {
        "messages": [
            {"role": "system", "content": parts},
            {"role": "user", "content": "Hello"},
        ]
    }
    original = deepcopy(kwargs)

    assert inject_recalled_facts(invoke, kwargs) == original
    assert kwargs["messages"][0]["content"] is parts
