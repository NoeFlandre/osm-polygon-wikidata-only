from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.grid5000.sentence_protocol import (
    import_checkpoint_tree,
    validate_manifest_extension,
    validate_sentence_output,
)
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2.sentence_logic import sentence_schema


def _write_table(path: Path, schema: pa.Schema, *, text: str = "First.") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field.name: None for field in schema}
    if "text" in schema.names:
        row["text"] = text
    pq.write_table(pa.Table.from_pylist([row], schema=schema), path)


def _manifest(
    regions: list[dict[str, object]], *, model_revision: str = "rev-a"
) -> dict[str, object]:
    return {
        "contract_version": "v2-sentence-splitting-v1",
        "segmenter": "sat-3l-sm",
        "model_id": "segment-any-text/sat-3l-sm",
        "model_revision": model_revision,
        "segmenter_version": "2.2.1",
        "supported_languages": ["en", "fr"],
        "unsupported_languages": [],
        "unsupported_language_policy": "one unsplit row; never passed to SaT",
        "regions": regions,
    }


def _identity(*, model_revision: str = "rev-a") -> dict[str, object]:
    return {
        "contract_version": "v2-sentence-checkpoints-v1",
        "stem": "alpha-latest",
        "project": "wikipedia",
        "input_fingerprint": "input-a",
        "model_id": "segment-any-text/sat-3l-sm",
        "model_revision": model_revision,
        "batch_size": 256,
    }


def _write_checkpoint(
    root: Path,
    *,
    identity: dict[str, object] | None = None,
    invalid_batch: bool = False,
    second_batch_symlink: Path | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": identity or _identity(),
        "complete": False,
        "batch_count": 2,
        "row_count": 2,
    }
    (root / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    _write_table(root / "batch-00000000.parquet", sentence_schema())
    if invalid_batch:
        _write_table(root / "batch-00000001.parquet", section_schema())
    elif second_batch_symlink is not None:
        (root / "batch-00000001.parquet").symlink_to(second_batch_symlink)
    else:
        _write_table(root / "batch-00000001.parquet", sentence_schema(), text="Second.")


def test_manifest_extension_preserves_existing_regions_and_adds_selected_stem() -> None:
    local = _manifest([{"stem": "done-latest", "project": "wikipedia", "sentence_rows": 2}])
    incoming = _manifest(
        [
            {"stem": "done-latest", "project": "wikipedia", "sentence_rows": 2},
            {"stem": "alpha-latest", "project": "wikipedia", "sentence_rows": 3},
        ]
    )

    validate_manifest_extension(local, incoming, selected_stems=("alpha-latest",))


@pytest.mark.parametrize(
    ("incoming", "message"),
    [
        (_manifest([]), "removed"),
        (
            _manifest([{"stem": "done-latest", "project": "wikipedia", "sentence_rows": 3}]),
            "changed",
        ),
        (
            {
                **_manifest([{"stem": "done-latest", "project": "wikipedia", "sentence_rows": 2}]),
                "model_revision": "rev-b",
            },
            "model_revision",
        ),
        (
            {
                **_manifest([{"stem": "done-latest", "project": "wikipedia", "sentence_rows": 2}]),
                "unsupported_language_policy": "split everything",
            },
            "policy",
        ),
    ],
)
def test_manifest_extension_rejects_removal_or_contract_changes(
    incoming: dict[str, object], message: str
) -> None:
    local = _manifest([{"stem": "done-latest", "project": "wikipedia", "sentence_rows": 2}])

    with pytest.raises(ValueError, match=message):
        validate_manifest_extension(local, incoming, selected_stems=("alpha-latest",))


def test_manifest_extension_requires_selected_stems() -> None:
    payload = _manifest([])

    with pytest.raises(ValueError, match="selected"):
        validate_manifest_extension(payload, payload, selected_stems=("alpha-latest",))


@pytest.mark.parametrize(
    ("regions", "message"),
    [
        ({}, "regions must be a list"),
        (["invalid"], "region must be an object"),
        ([{}], "needs string"),
        (
            [
                {"stem": "alpha-latest", "project": "wikipedia"},
                {"stem": "alpha-latest", "project": "wikipedia"},
            ],
            "duplicate",
        ),
    ],
)
def test_manifest_extension_rejects_malformed_region_entries(regions: object, message: str) -> None:
    local = _manifest([])
    incoming = {**_manifest([]), "regions": regions}

    with pytest.raises(ValueError, match=message):
        validate_manifest_extension(local, incoming, selected_stems=("alpha-latest",))


def test_sentence_output_requires_schema_and_sha256(tmp_path: Path) -> None:
    valid = tmp_path / "valid.parquet"
    _write_table(valid, sentence_schema())

    validate_sentence_output(valid, expected_sha256=sha256_file(valid))

    with pytest.raises(ValueError, match="schema"):
        invalid_schema = tmp_path / "invalid-schema.parquet"
        _write_table(invalid_schema, section_schema())
        validate_sentence_output(invalid_schema, expected_sha256=sha256_file(invalid_schema))
    with pytest.raises(ValueError, match="SHA-256"):
        validate_sentence_output(valid, expected_sha256="0" * 64)


def test_import_checkpoint_tree_validates_then_preserves_existing_batches(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    local = tmp_path / "local"
    _write_checkpoint(incoming)
    _write_checkpoint(local)
    _write_table(local / "batch-00000002.parquet", sentence_schema(), text="Older.")

    imported = import_checkpoint_tree(incoming, local, expected_identity=_identity())

    assert imported == (0, 1)
    assert (local / "batch-00000002.parquet").is_file()
    assert json.loads((local / "metadata.json").read_text())["identity"] == _identity()


@pytest.mark.parametrize("bad_identity", [_identity(model_revision="rev-b"), {"bad": True}])
def test_import_checkpoint_tree_rejects_identity_mismatch(
    tmp_path: Path, bad_identity: dict[str, object]
) -> None:
    incoming = tmp_path / "incoming"
    _write_checkpoint(incoming, identity=bad_identity)

    with pytest.raises(ValueError, match="identity"):
        import_checkpoint_tree(incoming, tmp_path / "local", expected_identity=_identity())


def test_import_checkpoint_tree_rejects_bad_schema_and_path_traversal(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.parquet"
    _write_table(outside, section_schema())

    invalid_schema_root = tmp_path / "invalid-schema"
    _write_checkpoint(invalid_schema_root, invalid_batch=True)
    with pytest.raises(ValueError, match="schema"):
        import_checkpoint_tree(
            invalid_schema_root,
            tmp_path / "local-schema",
            expected_identity=_identity(),
        )

    traversal_root = tmp_path / "traversal"
    _write_checkpoint(traversal_root, second_batch_symlink=outside)
    with pytest.raises(ValueError, match="outside"):
        import_checkpoint_tree(
            traversal_root,
            tmp_path / "local-traversal",
            expected_identity=_identity(),
        )


def test_import_checkpoint_tree_rejects_invalid_tree_layout(tmp_path: Path) -> None:
    missing_metadata = tmp_path / "missing-metadata"
    missing_metadata.mkdir()
    _write_table(missing_metadata / "batch-00000000.parquet", sentence_schema())
    with pytest.raises(ValueError, match=r"metadata\.json"):
        import_checkpoint_tree(
            missing_metadata,
            tmp_path / "local-missing-metadata",
            expected_identity=_identity(),
        )

    unexpected = tmp_path / "unexpected"
    _write_checkpoint(unexpected)
    (unexpected / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected checkpoint artifact"):
        import_checkpoint_tree(
            unexpected,
            tmp_path / "local-unexpected",
            expected_identity=_identity(),
        )

    with pytest.raises(ValueError, match="root is missing"):
        import_checkpoint_tree(
            tmp_path / "missing-root",
            tmp_path / "local-missing-root",
            expected_identity=_identity(),
        )


def test_import_checkpoint_tree_rejects_noncontiguous_batches(tmp_path: Path) -> None:
    incoming = tmp_path / "noncontiguous"
    _write_checkpoint(incoming)
    (incoming / "batch-00000001.parquet").unlink()
    _write_table(incoming / "batch-00000002.parquet", sentence_schema())

    with pytest.raises(ValueError, match="contiguous"):
        import_checkpoint_tree(incoming, tmp_path / "local", expected_identity=_identity())


def test_failed_checkpoint_import_leaves_prior_local_state_unchanged(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    local = tmp_path / "local"
    _write_checkpoint(incoming, invalid_batch=True)
    _write_checkpoint(local)
    before_metadata = (local / "metadata.json").read_bytes()
    before_batch = (local / "batch-00000000.parquet").read_bytes()

    with pytest.raises(ValueError):
        import_checkpoint_tree(incoming, local, expected_identity=_identity())

    assert (local / "metadata.json").read_bytes() == before_metadata
    assert (local / "batch-00000000.parquet").read_bytes() == before_batch
