from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.grid5000.sentence_protocol import (
    is_safe_run_id,
    plan_sentence_batches,
    sentence_source_paths,
    sha256_manifest,
    validate_cleanup_target,
)


def _write_section(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field.name: None for field in section_schema()}
    row.update({"section_id": path.stem, "language": "en", "text": text})
    pq.write_table(pa.Table.from_pylist([row], schema=section_schema()), path)


@pytest.fixture
def staged_v2(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    processed_v2 = tmp_path / "processed_v2"
    manifest_path = processed_v2 / "manifests" / "processed_pbfs.json"
    manifest_path.parent.mkdir(parents=True)
    stems = ("delta-latest", "beta-latest", "gamma-latest", "alpha-latest")
    for stem in stems:
        _write_section(
            processed_v2 / "wikipedia" / "sections" / f"{stem}.parquet",
            f"Wikipedia source for {stem}",
        )
    _write_section(
        processed_v2 / "wikivoyage" / "sections" / "beta-latest.parquet",
        "Wikivoyage source for beta-latest",
    )
    json_path = {
        "regions": {stem: {"sections_path": f"wikipedia/sections/{stem}.parquet"} for stem in stems}
    }
    manifest_path.write_text(json.dumps(json_path), encoding="utf-8")
    sentence_manifest = {
        "regions": [
            {"stem": "gamma-latest", "project": "wikipedia"},
            {"stem": "delta-latest", "project": "wikipedia"},
            {"stem": "delta-latest", "project": "wikivoyage"},
        ]
    }
    return processed_v2, sentence_manifest


def test_plan_excludes_completed_stems_and_packs_optional_wikivoyage(
    staged_v2: tuple[Path, dict[str, object]],
) -> None:
    processed_v2, sentence_manifest = staged_v2
    alpha_paths = sentence_source_paths(processed_v2, "alpha-latest")
    beta_paths = sentence_source_paths(processed_v2, "beta-latest")
    expected_bytes = sum(path.stat().st_size for path in (*alpha_paths, *beta_paths))

    batches = plan_sentence_batches(
        processed_v2,
        sentence_manifest,
        max_stems=4,
        max_input_bytes=expected_bytes,
    )

    assert batches[0].stems == ("alpha-latest", "beta-latest")
    assert batches[0].input_bytes == expected_bytes
    assert all(
        stem not in {"gamma-latest", "delta-latest"} for batch in batches for stem in batch.stems
    )
    assert all(len(batch.stems) <= 4 for batch in batches)


def test_plan_allows_one_oversized_stem_and_requires_positive_limits(
    staged_v2: tuple[Path, dict[str, object]],
) -> None:
    processed_v2, sentence_manifest = staged_v2
    batches = plan_sentence_batches(
        processed_v2,
        sentence_manifest,
        max_stems=1,
        max_input_bytes=1,
    )

    assert batches[0].stems == ("alpha-latest",)
    assert batches[0].input_bytes > 1
    with pytest.raises(ValueError, match="max_stems"):
        plan_sentence_batches(processed_v2, sentence_manifest, max_stems=0)
    with pytest.raises(ValueError, match="max_input_bytes"):
        plan_sentence_batches(processed_v2, sentence_manifest, max_input_bytes=0)


def test_plan_accepts_missing_manifest_and_rejects_malformed_regions(
    staged_v2: tuple[Path, dict[str, object]],
) -> None:
    processed_v2, _ = staged_v2

    assert plan_sentence_batches(processed_v2, None, max_stems=1)
    with pytest.raises(ValueError, match="regions must be a list"):
        plan_sentence_batches(processed_v2, {"regions": {}}, max_stems=1)
    with pytest.raises(ValueError, match="region must be an object"):
        plan_sentence_batches(processed_v2, {"regions": ["invalid"]}, max_stems=1)
    with pytest.raises(ValueError, match="needs string"):
        plan_sentence_batches(processed_v2, {"regions": [{}]}, max_stems=1)


def test_sentence_sources_reject_invalid_stems_and_missing_wikipedia(
    staged_v2: tuple[Path, dict[str, object]],
) -> None:
    processed_v2, _ = staged_v2

    with pytest.raises(ValueError, match="Invalid sentence stem"):
        sentence_source_paths(processed_v2, "../outside")
    with pytest.raises(FileNotFoundError, match="Wikipedia section source is missing"):
        sentence_source_paths(processed_v2, "missing-latest")


def test_safe_run_ids_and_cleanup_targets_are_bounded(tmp_path: Path) -> None:
    run_root = tmp_path / "run-20260827-01"

    assert is_safe_run_id("run-20260827-01")
    assert not is_safe_run_id("../outside")
    assert not is_safe_run_id("Run-20260827")
    assert validate_cleanup_target(run_root, run_root / "jobs" / "123")
    assert not validate_cleanup_target(run_root, run_root)
    assert not validate_cleanup_target(run_root, run_root.parent / "other")


def test_sha256_manifest_is_sorted_and_relative(tmp_path: Path) -> None:
    first = tmp_path / "z.txt"
    second = tmp_path / "nested" / "a.txt"
    second.parent.mkdir()
    first.write_bytes(b"z")
    second.write_bytes(b"a")

    records = sha256_manifest([first, second], root=tmp_path)

    assert [record.relative_path for record in records] == ["nested/a.txt", "z.txt"]
    assert records[0].size == 1
    assert records[0].sha256 == hashlib.sha256(b"a").hexdigest()

    outside = tmp_path.parent / "outside.txt"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="outside root"):
        sha256_manifest([outside], root=tmp_path)
    with pytest.raises(FileNotFoundError):
        sha256_manifest([tmp_path / "missing.txt"], root=tmp_path)
    with pytest.raises(ValueError, match="Duplicate artifact path"):
        sha256_manifest([first, first], root=tmp_path)
