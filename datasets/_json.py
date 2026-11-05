"""Incremental reading of nuScenes JSON metadata tables."""

import json
from collections.abc import Iterator
from pathlib import Path


def iter_json_records(path: str | Path, *, chunk_size: int = 65536) -> Iterator[dict]:
    """Yield objects from a UTF-8 JSON array without loading the full table.

    ``chunk_size`` counts decoded characters. The text stream handles UTF-8
    sequences split across underlying byte reads. Consumed text is discarded
    before each read, bounding the parser buffer by a chunk plus the current
    record. Consume the iterator fully to validate the array's closing bracket
    and reject trailing content.
    """
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")

    path = Path(path)

    def reject_constant(value):
        raise ValueError(f"{path}: invalid JSON constant {value!r}")

    decoder = json.JSONDecoder(parse_constant=reject_constant)

    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        position = 0
        eof = False

        def refill():
            nonlocal buffer, position, eof
            chunk = stream.read(chunk_size)
            buffer = buffer[position:] + chunk
            position = 0
            eof = not chunk

        def skip_whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n":
                    position += 1
                if position < len(buffer) or eof:
                    return
                refill()

        def fail(message):
            raise ValueError(f"{path}: {message}")

        skip_whitespace()
        if position == len(buffer) or buffer[position] != "[":
            fail("expected a top-level JSON array")
        position += 1
        skip_whitespace()

        if position < len(buffer) and buffer[position] == "]":
            position += 1
        else:
            record_index = 0
            while True:
                skip_whitespace()
                if position == len(buffer):
                    fail("unexpected end of JSON array")
                if buffer[position] != "{":
                    fail(f"expected an object record at array index {record_index}")

                while True:
                    try:
                        record, end = decoder.raw_decode(buffer, position)
                    except json.JSONDecodeError as error:
                        if eof:
                            fail(
                                f"invalid JSON object at array index {record_index}: "
                                f"{error.msg}"
                            )
                        refill()
                    else:
                        position = end
                        yield record
                        record_index += 1
                        break

                skip_whitespace()
                if position == len(buffer):
                    fail("unexpected end of JSON array; expected ',' or ']'")
                separator = buffer[position]
                position += 1
                if separator == "]":
                    break
                if separator != ",":
                    fail("expected ',' or ']' after an object record")
                skip_whitespace()
                if position < len(buffer) and buffer[position] == "]":
                    fail("trailing commas are not permitted in JSON arrays")

        skip_whitespace()
        if position < len(buffer):
            fail("unexpected trailing content after JSON array")
