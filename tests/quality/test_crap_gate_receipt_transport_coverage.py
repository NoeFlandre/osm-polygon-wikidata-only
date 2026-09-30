from __future__ import annotations

import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikipedia import transport as wikipedia_transport
from osm_polygon_wikidata_only.enrichment.wikipedia.models import FetchResult
from osm_polygon_wikidata_only.grid5000.sentence_protocol import (
    _read_checkpoint_metadata,
)
from osm_polygon_wikidata_only.hf import v1_language_splits
from osm_polygon_wikidata_only.hf.v1_language_splits import V1LanguageSplitError
from osm_polygon_wikidata_only.pipeline._wikidata_recovery import audit_receipts


def test_recovery_receipt_loader_rejects_bad_json_and_accepts_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipts.json"
    assert audit_receipts.load_receipts(path) == ({}, False)

    path.write_text("{broken", encoding="utf-8")
    assert audit_receipts.load_receipts(path) == ({}, False)
    path.write_text(
        '{"contract_version":"wikidata-enrichment-integrity-v2","regions":{"region":{}}}',
        encoding="utf-8",
    )
    assert audit_receipts.load_receipts(path) == ({"region": {}}, True)


def test_receipt_classification_parser_rejects_unknown_values() -> None:
    assert audit_receipts.parse_receipt_classifications({"Q1": "invalid-state"}) is None


def test_checkpoint_metadata_reader_rejects_invalid_and_non_object_payloads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.json"
    for content in ("{broken", "[]"):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid checkpoint metadata"):
            _read_checkpoint_metadata(path)

    path.write_text('{"identity":{}}', encoding="utf-8")
    assert _read_checkpoint_metadata(path) == {"identity": {}}


def test_v1_language_stream_wraps_unexpected_errors_and_preserves_domain_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = SimpleNamespace(language_column="language")
    args = (tmp_path / "source.parquet", spec, pa.schema([]), tmp_path, 1, {}, {}, {})

    def fail_open(_path: Path) -> object:
        raise OSError("unreadable shard")

    monkeypatch.setattr(v1_language_splits, "open_parquet", fail_open)
    with pytest.raises(V1LanguageSplitError, match="Could not partition language column"):
        v1_language_splits._stream_source_file(*args)

    domain_error = V1LanguageSplitError("invalid source schema")

    def fail_validation(_path: Path) -> object:
        raise domain_error

    monkeypatch.setattr(v1_language_splits, "open_parquet", fail_validation)
    with pytest.raises(V1LanguageSplitError, match="invalid source schema") as raised:
        v1_language_splits._stream_source_file(*args)
    assert raised.value is domain_error


def test_wikipedia_request_translates_http_and_network_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = wikipedia_transport.HttpWikipediaClient(
        Settings(), scheduler=SimpleNamespace(), session=SimpleNamespace()
    )
    monkeypatch.setattr(wikipedia_transport, "with_retries", lambda call, **_kwargs: call())

    def http_error(_url: str) -> object:
        raise urllib.error.HTTPError(
            "https://en.wikipedia.org/api", 429, "slow down", Message(), None
        )

    monkeypatch.setattr(client, "_http_get", http_error)
    data, result = client._request_article_data("https://en.wikipedia.org/api", fallback=False)
    assert data is None
    assert result is not None and result.status == "rate_limited"

    def network_error(_url: str) -> object:
        raise OSError("offline")

    monkeypatch.setattr(client, "_http_get", network_error)
    data, result = client._request_article_data("https://en.wikipedia.org/api", fallback=True)
    assert data is None
    assert result is not None and result.status == "http_error"
    assert result.error.startswith("parse fallback failed:")


def test_wikipedia_parse_fallback_preserves_error_and_handles_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = wikipedia_transport.HttpWikipediaClient(
        Settings(), scheduler=SimpleNamespace(), session=SimpleNamespace()
    )
    original = FetchResult("empty_text", None)
    rejected = FetchResult("http_error", None, "fallback unavailable")
    monkeypatch.setattr(client, "_request_article_data", lambda *_args, **_kwargs: (None, rejected))
    assert (
        client._parse_fallback(
            "en",
            "enwiki",
            "Title",
            {},
            original,
            "url",
            wikidata_label="",
            wikidata_description="",
        )
        is rejected
    )

    monkeypatch.setattr(
        client,
        "_request_article_data",
        lambda *_args, **_kwargs: ({"parse": {"text": ""}}, None),
    )
    result = client._parse_fallback(
        "en",
        "enwiki",
        "Title",
        {},
        original,
        "url",
        wikidata_label="",
        wikidata_description="",
    )
    assert result.status == "empty_text"
    assert "exact-revision parse were empty" in result.error
