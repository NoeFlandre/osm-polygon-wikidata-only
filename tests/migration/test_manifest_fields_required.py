"""Phase 2.5 / Defect 4: Manifest amendment 5 (link_schema_version +
link_artifact_sha256) is not implemented.

The approved design requires:

* ``link_schema_version`` and ``link_count`` added to the processed
  manifest entry ``<data-root>/processed/manifests/processed_pbfs.json``
  for the migrated PBF.
* ``link_schema_version`` and ``link_artifact_sha256`` (64-hex SHA-256
  of the canonical link file) added to the augmentation manifest entry
  keyed by stem.
* NO secondary ``manifests/link_manifest.json``. Any such artifact must
  not be created by migration.

Additionally:

* Pending-publication intent must be written for the migrated stem.
* Metadata-refresh marker must be written for the migrated stem,
  carrying the per-region ``link_artifact_sha256`` from the augmentation
  manifest.

Malformed existing manifest JSON must block (never silently replace
corruption with an empty manifest). The manifest update must merge
with existing entries (preserving unrelated fields/stems).
"""

from __future__ import annotations

import codecs
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.pipeline import link_migration
from osm_polygon_wikidata_only.pipeline._link_migration import artifacts
from tests.migration._builders import write_document, write_legacy_link, write_polygon


def _setup_processed(processed: Path, stem: str) -> None:
    for sub in (
        "polygons",
        "polygon_articles",
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
        "manifests",
    ):
        (processed / sub).mkdir(parents=True, exist_ok=True)
    write_polygon(processed, stem)
    write_document(processed, stem)
    write_legacy_link(processed, stem)
    from osm_polygon_wikidata_only.augmentation.schema import document_schema, section_schema

    empty_sections = pa.Table.from_pylist([], schema=section_schema())
    pq.write_table(  # type: ignore[no-untyped-call]
        empty_sections,
        processed / "wikipedia" / "sections" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=document_schema()),
        processed / "wikivoyage" / "documents" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikivoyage" / "sections" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"_placeholder": []}),
        processed / "wikidata" / "facts" / f"{stem}.parquet",
    )
    source_pbf = f"{stem}.osm.pbf"
    (processed / "manifests" / "processed_pbfs.json").write_text(
        json.dumps(
            {
                source_pbf: {
                    "source_pbf": source_pbf,
                    "region": stem.removesuffix("-latest"),
                    "polygons_path": f"polygons/{stem}.parquet",
                    "articles_path": f"wikipedia/documents/{stem}.parquet",
                    "polygon_articles_path": f"polygon_articles/{stem}.parquet",
                    "extraction_version": "test",
                    "processed_at": "2026-07-24T00:00:00Z",
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# 1. NO secondary manifests/link_manifest.json is created
# ---------------------------------------------------------------------------


def test_apply_does_not_create_link_manifest_json(tmp_path: Path) -> None:
    """Migration must NOT create a secondary
    ``manifests/link_manifest.json`` -- the approved design updates the
    EXISTING ``processed_pbfs.json`` instead.
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    # Pre-seed the processed manifest with an entry for the PBF.
    processed_manifest = processed / "manifests" / "processed_pbfs.json"
    processed_manifest.write_text(
        json.dumps(
            {
                f"{stem}.osm.pbf": {
                    "source_pbf": f"{stem}.osm.pbf",
                    "region": "r",
                    "polygons_path": f"polygons/{stem}.parquet",
                    "articles_path": f"wikipedia/documents/{stem}.parquet",
                    "polygon_articles_path": f"polygon_articles/{stem}.parquet",
                    "extraction_version": "test",
                    "processed_at": "2026-07-24T00:00:00Z",
                }
            }
        )
    )

    plan = link_migration.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    link_migration.apply_link_migration(processed)

    link_manifest = processed / "manifests" / "link_manifest.json"
    assert not link_manifest.exists(), (
        f"Migration must NOT create {link_manifest}; the design updates processed_pbfs.json"
    )


# ---------------------------------------------------------------------------
# 2. processed_pbfs.json: link_schema_version + link_count
# ---------------------------------------------------------------------------


def test_apply_writes_link_schema_version_and_count_to_processed_manifest(tmp_path: Path) -> None:
    """The processed manifest entry for the migrated PBF must include
    ``link_schema_version`` and ``link_count`` (with the actual local
    table row count), and other fields must be preserved.
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    processed_manifest = processed / "manifests" / "processed_pbfs.json"
    pre_existing = {
        "source_pbf": f"{stem}.osm.pbf",
        "region": "r",
        "polygons_path": f"polygons/{stem}.parquet",
        "articles_path": f"wikipedia/documents/{stem}.parquet",
        "polygon_articles_path": f"polygon_articles/{stem}.parquet",
        "extraction_version": "test",
        "processed_at": "2026-07-24T00:00:00Z",
        "unrelated_field": {"preserve": "me"},
    }
    processed_manifest.write_text(
        json.dumps({f"{stem}.osm.pbf": pre_existing}, indent=2, sort_keys=True) + "\n"
    )

    link_migration.apply_link_migration(processed)

    payload = json.loads(processed_manifest.read_text())
    entry = payload[f"{stem}.osm.pbf"]

    # Required fields.
    assert entry["link_schema_version"] == "polygon-document-links-v1", (
        f"Processed manifest must include link_schema_version; got entry={entry}"
    )
    assert entry["link_count"] == 1, f"link_count must equal local table rows; got {entry}"

    # Every pre-existing field must be preserved.
    for key, value in pre_existing.items():
        if key == "processed_at":
            # processed_at is updated by the migration.
            continue
        assert entry.get(key) == value, (
            f"Field {key!r} must be preserved; got {entry.get(key)} != {value}"
        )

    # No secondary manifest is created.
    link_manifest = processed / "manifests" / "link_manifest.json"
    assert not link_manifest.exists(), f"Migration must NOT create {link_manifest}"


# ---------------------------------------------------------------------------
# 3. augmentation manifest: link_schema_version + link_artifact_sha256
# ---------------------------------------------------------------------------


def test_apply_writes_link_schema_version_and_sha256_to_augmentation_manifest(
    tmp_path: Path,
) -> None:
    """The augmentation manifest entry for the migrated stem must
    include ``link_schema_version`` and ``link_artifact_sha256``
    (64-hex SHA-256 of the canonical link file).
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)
    aug_manifest = processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    aug_manifest.parent.mkdir(parents=True, exist_ok=True)
    completed_at = "2026-07-24T00:00:00Z"
    pre_existing = {
        "counts": {"legacy_documents": 12},
        "completed_at": completed_at,
        "extra_contract_field": "preserve me",
    }
    aug_manifest.write_text(json.dumps({stem: pre_existing}, indent=2, sort_keys=True) + "\n")

    link_migration.apply_link_migration(processed)

    payload = json.loads(aug_manifest.read_text())
    entry = payload[stem]
    # All required augmentation-manifest fields.
    required_fields = {
        "contract_version",
        "core_hashes",
        "paths",
        "counts",
        "completed_at",
        "link_schema_version",
        "link_artifact_sha256",
    }
    missing = required_fields - set(entry.keys())
    assert not missing, (
        f"Augmentation manifest entry is missing required fields: {missing}; got {entry}"
    )

    assert entry.get("link_schema_version") == "polygon-document-links-v1", (
        f"Augmentation manifest must include link_schema_version; got {entry}"
    )
    sha = entry.get("link_artifact_sha256")
    assert isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), (
        f"link_artifact_sha256 must be 64 lowercase hex; got {sha!r}"
    )

    # The SHA must equal the hash of the canonical link file.
    link_path = processed / "polygon_articles" / f"{stem}.parquet"
    expected = hashlib.sha256(link_path.read_bytes()).hexdigest()
    assert sha == expected, (
        f"link_artifact_sha256 must equal canonical link hash; got {sha} vs {expected}"
    )
    assert entry["completed_at"] == completed_at
    assert entry["extra_contract_field"] == "preserve me"
    assert entry["counts"] == {
        "legacy_documents": 12,
        "polygon_articles": 1,
        "wikivoyage_documents": 0,
        "wikivoyage_sections": 0,
    }

    from osm_polygon_wikidata_only.augmentation.orchestrator import sidecar_paths
    from osm_polygon_wikidata_only.config.paths import DataRoot

    data_root = DataRoot(tmp_path)
    expected_paths = [str(path.relative_to(processed)) for path in sidecar_paths(data_root, stem)]
    assert entry["paths"] == expected_paths
    polygons_path = processed / "polygons" / f"{stem}.parquet"
    documents_path = processed / "wikipedia" / "documents" / f"{stem}.parquet"
    assert entry["core_hashes"] == {
        str(polygons_path): hashlib.sha256(polygons_path.read_bytes()).hexdigest(),
        str(documents_path): hashlib.sha256(documents_path.read_bytes()).hexdigest(),
    }


def test_new_augmentation_entry_has_a_timestamp_when_no_previous_value_exists(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    link_migration.apply_link_migration(processed)

    path = processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    entry = json.loads(path.read_text())[stem]
    completed_at = datetime.fromisoformat(entry["completed_at"])
    assert completed_at.tzinfo is not None


def test_manifest_json_reader_requests_utf8_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"city": "München"}', encoding="utf-8")
    original_read_text = Path.read_text
    encodings: list[str | None] = []

    def read_text(
        file_path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        encodings.append(encoding)
        return original_read_text(file_path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert artifacts._load_json_object(path, "manifest.json") == {"city": "München"}
    assert len(encodings) == 1
    assert encodings[0] is not None
    assert codecs.lookup(encodings[0]).name == "utf-8"


# ---------------------------------------------------------------------------
# 4. Augmentation manifest must merge (preserve other stems)
# ---------------------------------------------------------------------------


def test_augmentation_manifest_merges_preserves_other_stems(tmp_path: Path) -> None:
    """Updating the augmentation manifest for stem A must NOT erase
    prior entries for stem B.
    """
    processed = tmp_path / "processed"
    stem_a = "alpha-latest"
    _setup_processed(processed, stem_a)

    aug_manifest = processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    aug_manifest.parent.mkdir(parents=True, exist_ok=True)
    pre_existing = {
        "beta-latest": {
            "stale": False,
            "core_hashes": {},
        },
    }
    aug_manifest.write_text(json.dumps(pre_existing, indent=2, sort_keys=True) + "\n")

    link_migration.apply_link_migration(processed)

    payload = json.loads(aug_manifest.read_text())
    assert "beta-latest" in payload, (
        f"Augmentation manifest must preserve stem beta-latest; got {payload}"
    )
    assert payload["beta-latest"] == pre_existing["beta-latest"]


# ---------------------------------------------------------------------------
# 5. Pending-publication intent is written
# ---------------------------------------------------------------------------


def test_apply_writes_pending_publication_intent(tmp_path: Path) -> None:
    """After migration, the pending-publication manifest must include
    the migrated stem.
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    # No prior pending manifest.
    link_migration.apply_link_migration(processed)

    from osm_polygon_wikidata_only.config.paths import DataRoot
    from osm_polygon_wikidata_only.pipeline import pending_publications as pp

    dr = DataRoot(tmp_path)
    stems = pp.load_pending_publications(dr)
    assert stem in stems, f"Pending publications must include the migrated stem {stem}; got {stems}"


# ---------------------------------------------------------------------------
# 6. Metadata-refresh marker is written
# ---------------------------------------------------------------------------


def test_apply_writes_metadata_refresh_marker(tmp_path: Path) -> None:
    """After migration, the metadata-refresh marker must include the
    migrated stem with the correct ``fingerprint_hashes`` entry
    (the augmentation manifest's link_artifact_sha256).
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    link_migration.apply_link_migration(processed)

    from osm_polygon_wikidata_only.config.paths import DataRoot
    from osm_polygon_wikidata_only.pipeline import pending_publications as pp

    dr = DataRoot(tmp_path)
    marker = pp.load_metadata_refresh_marker(dr)
    assert marker is not None, "Metadata-refresh marker must be written after migration"

    # Every required marker field must be present and well-formed.
    assert set(marker.keys()) == {"stems", "fingerprint_hashes"}, (
        f"Marker keys must be exactly 'stems' and 'fingerprint_hashes'; got {set(marker.keys())}"
    )
    assert marker["stems"] == [stem], (
        f"Marker stems must be the sorted list of migrated stems; got {marker['stems']}"
    )
    assert set(marker["fingerprint_hashes"].keys()) == {stem}, (
        f"Marker fingerprint_hashes keys must equal the stems list; got {marker['fingerprint_hashes'].keys()}"
    )
    sha = marker["fingerprint_hashes"][stem]
    link_path = processed / "polygon_articles" / f"{stem}.parquet"
    expected = hashlib.sha256(link_path.read_bytes()).hexdigest()
    assert sha == expected, (
        f"Marker fingerprint_hashes[{stem}] must equal canonical link hash; got {sha} vs {expected}"
    )


def test_pending_publications_merge_multiple_stems_in_sorted_order(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    stems = ["beta-latest", "alpha-latest"]
    for stem in stems:
        _setup_processed(processed, stem)

    link_migration.apply_link_migration(processed)

    pending_path = processed / "manifests" / "pending_migration_publications.json"
    payload = json.loads(pending_path.read_text())
    expected_stems = sorted(stems)
    expected_hashes = {
        stem: hashlib.sha256(
            (processed / "polygon_articles" / f"{stem}.parquet").read_bytes()
        ).hexdigest()
        for stem in expected_stems
    }
    assert payload == {
        "contract_version": "pending-publications-v1",
        "stems": expected_stems,
        "metadata_refresh": {
            "stems": expected_stems,
            "fingerprint_hashes": expected_hashes,
        },
    }

    from osm_polygon_wikidata_only.augmentation.rejection_ledger import LEDGER_CONTRACT_VERSION

    ledger_path = processed / "integrity" / "rejection_ledger.json"
    assert json.loads(ledger_path.read_text()) == {
        "contract_version": LEDGER_CONTRACT_VERSION,
        "records": [],
    }


# ---------------------------------------------------------------------------
# 7. Malformed existing processed manifest JSON blocks migration
# ---------------------------------------------------------------------------


def test_malformed_processed_manifest_blocks_migration(tmp_path: Path) -> None:
    """A malformed existing ``processed_pbfs.json`` must block the
    migration -- never silently replace corruption with an empty manifest.
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    processed_manifest = processed / "manifests" / "processed_pbfs.json"
    processed_manifest.write_text("{ this is not valid JSON")

    # Plan should still succeed (planning is read-only and a separate
    # file from the migration). The apply stage must refuse to
    # silently overwrite a malformed manifest.
    with pytest.raises((ValueError, json.JSONDecodeError)) as error:
        link_migration.apply_link_migration(processed)
    assert str(error.value).startswith("processed_pbfs.json: ")


@pytest.mark.parametrize(
    ("relative_path", "diagnostic"),
    [
        (
            "augmentation/manifests/augmentation_manifest.json",
            "augmentation_manifest.json",
        ),
        (
            "manifests/pending_migration_publications.json",
            "pending_migration_publications.json",
        ),
    ],
)
def test_malformed_migration_manifests_keep_their_file_diagnostic(
    tmp_path: Path,
    relative_path: str,
    diagnostic: str,
) -> None:
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)
    malformed = processed / relative_path
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{ invalid JSON")

    with pytest.raises(ValueError) as error:
        link_migration.apply_link_migration(processed)
    assert str(error.value).startswith(f"{diagnostic}: ")
