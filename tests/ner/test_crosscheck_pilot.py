"""Offline tests for the deterministic secondary-model pilot preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.ner.pipeline import (
    LABEL,
    WIKINEURAL_LANGUAGES,
    WIKINEURAL_MODEL_ID,
    WIKINEURAL_MODEL_REVISION,
    Contract,
)
from scripts import prepare_geographic_ner_crosscheck as crosscheck


def test_crosscheck_contract_is_secondary_and_language_explicit() -> None:
    contract = crosscheck.build_contract()

    assert contract.languages == WIKINEURAL_LANGUAGES
    assert contract.model_id == WIKINEURAL_MODEL_ID
    assert contract.model_revision == WIKINEURAL_MODEL_REVISION
    assert contract.label == LABEL
    assert contract.validation_status == "silver_unvalidated"
    assert contract.implementation_revision == "geographic-ner-silver-v1"


def test_crosscheck_preparation_copies_input_and_writes_contract(tmp_path) -> None:
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    pq.write_table(pa.table({"sentence_id": ["s1"], "text": ["Paris"]}), pilot / "input.parquet")
    (pilot / "selection.json").write_text(json.dumps({"sample_size": 1}))
    output = tmp_path / "nested" / "crosscheck"

    assert crosscheck.prepare(pilot, output) == output

    assert pq.read_table(output / "input.parquet").to_pylist() == [
        {
            "sentence_id": "s1",
            "text": "Paris",
        }
    ]
    assert (output / "selection.json").read_text() == (pilot / "selection.json").read_text()
    payload = json.loads((output / "contract.json").read_text())
    assert payload == {
        "implementation_revision": "geographic-ner-silver-v1",
        "label": LABEL,
        "languages": list(WIKINEURAL_LANGUAGES),
        "model_id": WIKINEURAL_MODEL_ID,
        "model_revision": WIKINEURAL_MODEL_REVISION,
        "threshold": 0.5,
        "validation_status": "silver_unvalidated",
    }


def test_crosscheck_preparation_preserves_non_ascii_contract_text(tmp_path, monkeypatch) -> None:
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "input.parquet").write_bytes(b"input")
    (pilot / "selection.json").write_text("{}")
    contract = Contract(
        languages=("en",),
        validation_status="silver_unvalidated",
        implementation_revision="é",
        model_id=WIKINEURAL_MODEL_ID,
        model_revision=WIKINEURAL_MODEL_REVISION,
        label=LABEL,
    )
    monkeypatch.setattr(crosscheck, "build_contract", lambda: contract)

    output = crosscheck.prepare(pilot, tmp_path / "output")

    expected = {
        "implementation_revision": "é",
        "label": LABEL,
        "languages": list(WIKINEURAL_LANGUAGES),
        "model_id": WIKINEURAL_MODEL_ID,
        "model_revision": WIKINEURAL_MODEL_REVISION,
        "threshold": 0.5,
        "validation_status": "silver_unvalidated",
    }
    assert (output / "contract.json").read_text(encoding="utf-8") == (
        json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )


@pytest.mark.parametrize("missing", ["input.parquet", "selection.json"])
def test_crosscheck_preparation_rejects_missing_primary_file(tmp_path, missing) -> None:
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    for name in ("input.parquet", "selection.json"):
        if name != missing:
            (pilot / name).write_bytes(b"present")

    with pytest.raises(ValueError, match=rf"Primary pilot is missing {missing}"):
        crosscheck.prepare(pilot, tmp_path / "output")


def test_crosscheck_preparation_rejects_an_existing_output(tmp_path) -> None:
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "input.parquet").write_bytes(b"input")
    (pilot / "selection.json").write_text("{}")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(ValueError, match="Cross-check directory already exists"):
        crosscheck.prepare(pilot, output)


def test_crosscheck_cli_prepares_the_requested_directory(tmp_path, capsys) -> None:
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    pq.write_table(pa.table({"sentence_id": ["s1"]}), pilot / "input.parquet")
    (pilot / "selection.json").write_text("{}")
    output = tmp_path / "crosscheck"

    assert crosscheck.main(["--pilot-dir", str(pilot), "--output-dir", str(output)]) == 0

    assert json.loads(capsys.readouterr().out) == {"output_dir": str(output), "status": "prepared"}


def test_crosscheck_parser_requires_two_path_arguments() -> None:
    parser = crosscheck._parser()
    actions = {action.dest: action for action in parser._actions}

    assert parser.description == crosscheck.__doc__
    for name in ("pilot_dir", "output_dir"):
        assert actions[name].type is Path
        assert actions[name].required is True

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--pilot-dir", "pilot"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "output"])
