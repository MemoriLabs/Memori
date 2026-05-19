from memori._utils import bytes_to_json, format_date_created, generate_uniq, merge_chunk


def test_merge_chunk():
    merged = merge_chunk({}, {"a": 1, "b": [1, 2], "c": {"x": "foo"}})
    merged = merge_chunk(merged, {"b": [3], "c": {"y": "bar"}, "d": 5})
    merged = merge_chunk(merged, {"a": 2, "c": {"x": "baz"}})

    assert merged == {"a": 2, "b": [1, 2, 3], "c": {"x": "baz", "y": "bar"}, "d": 5}


def test_bytes_to_json():
    assert bytes_to_json(
        {
            b"name": b"Alice",
            "info": {
                b"age": 47,
                b"email": b"alice@domain.com",
                "tags": [b"abc", b"def"],
            },
            "body": b'{"ghi": "jkl"}',
        }
    ) == {
        "body": {"ghi": "jkl"},
        "info": {
            "age": 47,
            "email": "alice@domain.com",
            "tags": [
                "abc",
                "def",
            ],
        },
        "name": "Alice",
    }


def test_generate_uniq():
    assert generate_uniq(None) is None  # type: ignore[arg-type]
    assert generate_uniq([]) is None

    assert (
        generate_uniq(["Abc. Def", "ghi", "123"])
        == "84596a42b9169065d9f1bf2015e508beab38dd6af0814cc20572cf3b256f763d"
    )
    assert (
        generate_uniq(["abc def", "GHI", "123 "])
        == "84596a42b9169065d9f1bf2015e508beab38dd6af0814cc20572cf3b256f763d"
    )


def test_format_date_created_iso_z():
    assert format_date_created("2026-05-14T10:30:00Z") == "2026-05-14 10:30"


def test_format_date_created_none():
    assert format_date_created(None) is None
