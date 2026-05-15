from memori.llm.pipelines.recall_injection import (
    format_recalled_fact_lines,
    format_recalled_summary_lines,
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
