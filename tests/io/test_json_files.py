"""Contracts for the shared JSON document reader and its error policies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.io.json_files import read_json

_DIGIT_LIMIT_SKIP = pytest.mark.skipif(
    sys.get_int_max_str_digits() == 0, reason="integer digit limit is disabled"
)


class _Malformed(ValueError):
    pass


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _integer_past_digit_limit() -> bytes:
    """Return a JSON array holding an integer one digit past the parser limit."""
    return ("[" + "1" * (sys.get_int_max_str_digits() + 1) + "]").encode()


def test_reads_any_json_value_without_shape_checks(tmp_path: Path) -> None:
    path = _write_bytes(tmp_path / "value.json", '{"city": "München"}'.encode())
    assert read_json(path) == {"city": "München"}

    listed = _write_bytes(tmp_path / "list.json", b"[1, 2]")
    assert read_json(listed) == [1, 2]


@pytest.mark.parametrize(
    "payload",
    [b'{"truncated": ', b"\xff\xfe not utf-8", b""],
    ids=["malformed-json", "invalid-utf8", "empty-file"],
)
def test_lenient_policy_returns_none_for_unusable_content(tmp_path: Path, payload: bytes) -> None:
    path = _write_bytes(tmp_path / "broken.json", payload)
    assert read_json(path) is None


def test_lenient_policy_returns_none_for_missing_file_and_directory(tmp_path: Path) -> None:
    assert read_json(tmp_path / "absent.json") is None
    assert read_json(tmp_path) is None


@_DIGIT_LIMIT_SKIP
def test_lenient_policy_returns_none_for_integer_past_digit_limit(tmp_path: Path) -> None:
    path = _write_bytes(tmp_path / "oversized.json", _integer_past_digit_limit())
    assert read_json(path) is None


@_DIGIT_LIMIT_SKIP
def test_strict_policy_propagates_digit_limit_error_without_factory(tmp_path: Path) -> None:
    factory_calls: list[json.JSONDecodeError] = []

    def factory(error: json.JSONDecodeError) -> Exception:
        factory_calls.append(error)
        return _Malformed("unused")

    path = _write_bytes(tmp_path / "oversized.json", _integer_past_digit_limit())

    with pytest.raises(ValueError, match="digits") as raised:
        read_json(path, on_malformed=factory)

    assert not isinstance(raised.value, _Malformed)
    assert factory_calls == []


def test_strict_policy_raises_factory_error_chained_to_parser_error(tmp_path: Path) -> None:
    path = _write_bytes(tmp_path / "broken.json", b'{"truncated": ')

    with pytest.raises(_Malformed, match="Expecting value") as raised:
        read_json(path, on_malformed=lambda error: _Malformed(str(error)))

    assert isinstance(raised.value.__cause__, json.JSONDecodeError)


def test_strict_policy_passes_the_parser_error_to_the_factory(tmp_path: Path) -> None:
    seen: list[json.JSONDecodeError] = []

    def factory(error: json.JSONDecodeError) -> Exception:
        seen.append(error)
        return _Malformed("wrapped")

    path = _write_bytes(tmp_path / "broken.json", b"not json")

    with pytest.raises(_Malformed, match="wrapped"):
        read_json(path, on_malformed=factory)

    assert len(seen) == 1
    assert isinstance(seen[0], json.JSONDecodeError)


def test_strict_policy_lets_unreadable_content_propagate(tmp_path: Path) -> None:
    factory_calls: list[Exception] = []

    def factory(error: json.JSONDecodeError) -> Exception:
        factory_calls.append(error)
        return _Malformed("unused")

    invalid = _write_bytes(tmp_path / "invalid.json", b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        read_json(invalid, on_malformed=factory)

    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "absent.json", on_malformed=factory)

    assert factory_calls == []
