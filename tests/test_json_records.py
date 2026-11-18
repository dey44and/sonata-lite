"""Checks for bounded, incremental reading of nuScenes metadata arrays."""

import json

import pytest

from datasets._json import iter_json_records


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 64, 65536])
def test_records_across_chunk_boundaries(tmp_path, chunk_size):
    records = [
        {"token": "first", "nested": {"values": [1, True, None]}},
        {"text": 'Escaped quote: " and backslash: \\', "number": -1.25e-12},
        {"unicode": "București 🚙 東京", "braces": "{ [ , ] }"},
        {"long": "x" * 300},
    ]
    path = tmp_path / "sample_data.json"
    path.write_text(
        " \n" + json.dumps(records, ensure_ascii=False, indent=2) + "\t\n",
        encoding="utf-8",
    )

    assert list(iter_json_records(path, chunk_size=chunk_size)) == records


@pytest.mark.parametrize("text", ["[]", " [ \n\t ] \r\n"])
def test_empty_array(tmp_path, text):
    path = tmp_path / "empty.json"
    path.write_text(text, encoding="utf-8")

    assert list(iter_json_records(path, chunk_size=1)) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "top-level JSON array"),
        ("{}", "top-level JSON array"),
        ("[", "unexpected end"),
        ("[{\"token\":", "invalid JSON object"),
        ("[{\"token\": 1}", "unexpected end"),
        ("[{},]", "trailing commas"),
        ("[{},", "unexpected end"),
        ("[{} {}]", "expected ',' or ']'"),
        ("[{\"broken\": }]", "invalid JSON object"),
        ("[] {}", "trailing content"),
        ("[{}]]", "trailing content"),
        ("[1]", "object record"),
        ("[null]", "object record"),
        ("[[]]", "object record"),
        ("[{}, false]", "object record"),
        ("[{\"value\": NaN}]", "invalid JSON constant"),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 5, 65536])
def test_rejects_invalid_tables(tmp_path, text, message, chunk_size):
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message) as error:
        list(iter_json_records(path, chunk_size=chunk_size))

    assert str(path) in str(error.value)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_requires_positive_integer_chunk_size(tmp_path, chunk_size):
    path = tmp_path / "records.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        list(iter_json_records(path, chunk_size=chunk_size))


def test_missing_table_preserves_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_json_records(tmp_path / "missing.json"))


def test_reads_lazily_with_bounded_requests(tmp_path, monkeypatch):
    """Producing one record must not read the rest of a large table."""
    path = tmp_path / "large.json"
    path.write_text(json.dumps([{"token": i} for i in range(5000)]))
    original_open = type(path).open
    read_sizes = []

    class TrackedStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size):
            read_sizes.append(size)
            return self.stream.read(size)

    def tracked_open(self, *args, **kwargs):
        return TrackedStream(original_open(self, *args, **kwargs))

    monkeypatch.setattr(type(path), "open", tracked_open)
    records = iter_json_records(path, chunk_size=32)
    assert next(records) == {"token": 0}
    assert sum(read_sizes) < path.stat().st_size
    assert all(size == 32 for size in read_sizes)
    records.close()
