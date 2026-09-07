"""Metadata remains UTF-8 when the host's default text encoding is ASCII."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.grid5000 import ner_controller
from osm_polygon_wikidata_only.ner.pipeline import Contract
from scripts import grid5000_geographic_ner as cli
from scripts import prepare_geographic_ner_pilot as pilot


@pytest.fixture
def ascii_default(monkeypatch):
    original = Path.open

    def open_ascii(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "ascii"
        return original(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", open_ascii)


@pytest.mark.parametrize(
    "reader",
    [
        ner_controller._read_receipt,
        ner_controller._read_mapping,
        ner_controller._read_contract_payload,
    ],
)
def test_controller_json_is_utf8(tmp_path, ascii_default, reader):
    path = tmp_path / "metadata.json"
    value = {"languages": ["en"], "text": "Montréal"}
    path.write_bytes(json.dumps(value, ensure_ascii=False).encode())
    assert reader(path) == value


@pytest.mark.parametrize("reader", [ner_controller._runtime_lock_blocks, cli._read_lock])
def test_lock_comments_are_utf8(tmp_path, ascii_default, reader):
    path = tmp_path / "lock.txt"
    path.write_bytes(("# café\npackage==1 --hash=sha256:" + "a" * 64 + "\n").encode())
    assert reader(path)


def test_staging_contract_reads_and_writes_utf8(tmp_path, ascii_default):
    contract = Contract(languages=("en",), implementation_revision="révision")
    path = tmp_path / "contract.json"
    path.write_bytes(json.dumps(asdict(contract), ensure_ascii=False).encode())
    assert cli._read_contract(path) == contract
    cli._write_contract(path, contract)
    assert json.loads(path.read_bytes()) == json.loads(json.dumps(asdict(contract)))


def test_manifest_with_unicode_metadata_is_utf8(tmp_path, ascii_default):
    sentence = tmp_path / "wikipedia/sentences/region.parquet"
    sentence.parent.mkdir(parents=True)
    sentence.touch()
    path = tmp_path / "manifest.json"
    path.write_bytes(
        json.dumps(
            {
                "note": "Montréal",
                "regions": [
                    {"project": "wikipedia", "stem": "region", "supported_languages": ["en"]}
                ],
            },
            ensure_ascii=False,
        ).encode()
    )
    assert list(pilot._manifest_sentence_files(tmp_path, path)) == [("wikipedia", sentence)]
