from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.grid5000.sentence_protocol import (
    SentenceBatch,
    _pack_batches,
    _required_projects,
    _should_flush,
    _validate_stem,
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
    with pytest.raises(ValueError, match=r"^max_stems must be positive$"):
        plan_sentence_batches(processed_v2, sentence_manifest, max_stems=0)
    with pytest.raises(ValueError, match=r"^max_input_bytes must be positive$"):
        plan_sentence_batches(processed_v2, sentence_manifest, max_input_bytes=0)


def test_plan_accepts_missing_manifest_and_rejects_malformed_regions(
    staged_v2: tuple[Path, dict[str, object]],
) -> None:
    processed_v2, _ = staged_v2

    assert plan_sentence_batches(processed_v2, None, max_stems=1)
    assert plan_sentence_batches(processed_v2, {}, max_stems=1)
    with pytest.raises(ValueError, match=r"^Sentence manifest regions must be a list$"):
        plan_sentence_batches(processed_v2, {"regions": {}}, max_stems=1)
    with pytest.raises(ValueError, match=r"^Sentence manifest region must be an object$"):
        plan_sentence_batches(processed_v2, {"regions": ["invalid"]}, max_stems=1)
    with pytest.raises(
        ValueError, match=r"^Sentence manifest region needs string stem and project$"
    ):
        plan_sentence_batches(processed_v2, {"regions": [{}]}, max_stems=1)


@pytest.mark.parametrize("region", [{"project": "wikipedia"}, {"stem": "alpha-latest"}])
def test_plan_rejects_regions_with_one_missing_identity_field(
    staged_v2: tuple[Path, dict[str, object]], region: dict[str, object]
) -> None:
    processed_v2, _ = staged_v2

    with pytest.raises(
        ValueError, match=r"^Sentence manifest region needs string stem and project$"
    ):
        plan_sentence_batches(processed_v2, {"regions": [region]}, max_stems=1)


def test_sentence_sources_reject_invalid_stems_and_missing_wikipedia(
    staged_v2: tuple[Path, dict[str, object]],
) -> None:
    processed_v2, _ = staged_v2

    with pytest.raises(ValueError, match="Invalid sentence stem"):
        sentence_source_paths(processed_v2, "../outside")
    with pytest.raises(FileNotFoundError, match="Wikipedia section source is missing"):
        sentence_source_paths(processed_v2, "missing-latest")


def test_sentence_sources_return_canonical_wikipedia_and_optional_wikivoyage_paths(
    staged_v2: tuple[Path, dict[str, object]],
) -> None:
    processed_v2, _ = staged_v2

    assert sentence_source_paths(processed_v2, "beta-latest") == (
        processed_v2 / "wikipedia" / "sections" / "beta-latest.parquet",
        processed_v2 / "wikivoyage" / "sections" / "beta-latest.parquet",
    )
    assert sentence_source_paths(processed_v2, "alpha-latest") == (
        processed_v2 / "wikipedia" / "sections" / "alpha-latest.parquet",
    )


@pytest.mark.parametrize("stem", ["", ".", "..", "a/b", r"a\b"])
def test_validate_stem_rejects_empty_dot_and_path_like_values(stem: str) -> None:
    with pytest.raises(ValueError, match=r"^Invalid sentence stem:"):
        _validate_stem(stem)


def test_required_projects_match_the_source_file_contract(
    staged_v2: tuple[Path, dict[str, object]],
) -> None:
    processed_v2, _ = staged_v2

    assert _required_projects(sentence_source_paths(processed_v2, "beta-latest")) == {
        "wikipedia",
        "wikivoyage",
    }
    assert _required_projects(sentence_source_paths(processed_v2, "alpha-latest")) == {"wikipedia"}


def test_completed_stem_does_not_stop_planning_later_pending_stems(
    staged_v2: tuple[Path, dict[str, object]],
) -> None:
    processed_v2, _ = staged_v2
    sentence_manifest = {"regions": [{"stem": "alpha-latest", "project": "wikipedia"}]}

    batches = plan_sentence_batches(processed_v2, sentence_manifest, max_stems=1)

    assert [batch.stems for batch in batches] == [
        ("beta-latest",),
        ("delta-latest",),
        ("gamma-latest",),
    ]


def test_pack_batches_resets_byte_count_and_indexes_each_batch() -> None:
    batches = _pack_batches(
        [("alpha-latest", 3), ("beta-latest", 4), ("gamma-latest", 5)],
        max_stems=4,
        max_input_bytes=5,
    )

    assert batches == (
        SentenceBatch(index=0, stems=("alpha-latest",), input_bytes=3),
        SentenceBatch(index=1, stems=("beta-latest",), input_bytes=4),
        SentenceBatch(index=2, stems=("gamma-latest",), input_bytes=5),
    )


def test_should_flush_requires_existing_stems_and_checks_both_limits() -> None:
    assert not _should_flush((), current_bytes=100, next_bytes=100, max_stems=1, max_input_bytes=1)
    assert _should_flush(
        ("alpha-latest",), current_bytes=1, next_bytes=1, max_stems=1, max_input_bytes=100
    )
    assert _should_flush(
        ("alpha-latest",), current_bytes=1, next_bytes=2, max_stems=10, max_input_bytes=2
    )
    assert not _should_flush(
        ("alpha-latest",), current_bytes=1, next_bytes=1, max_stems=10, max_input_bytes=2
    )


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
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError) as error:
        sha256_manifest([missing], root=tmp_path)
    assert error.value.args == (missing,)
    with pytest.raises(ValueError, match="Duplicate artifact path"):
        sha256_manifest([first, first], root=tmp_path)
