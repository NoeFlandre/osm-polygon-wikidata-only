from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema, polygon_schema
from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub

# Intentionally importing what might not exist yet to verify red state
from osm_polygon_wikidata_only.hf.publication import (
    CorePublicationArtifacts,
    PublicationValidationError,
    assemble_metadata_only_upload,
    load_existing_core_artifacts,
)
from osm_polygon_wikidata_only.hf.reconciliation import ReconciliationPlanner
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.repo_layout import canonical_region_paths


def test_canonical_region_paths() -> None:
    paths = canonical_region_paths("mexico-latest")
    assert paths["polygons/mexico-latest.parquet"] == "polygons/mexico-latest.parquet"
    assert (
        paths["polygon_articles/mexico-latest.parquet"] == "polygon_articles/mexico-latest.parquet"
    )
    assert (
        paths["wikipedia/documents/mexico-latest.parquet"]
        == "wikipedia/documents/mexico-latest.parquet"
    )
    assert len(paths) == 7


def test_remote_inventory_fetch_success() -> None:
    stub = StubHfHub(remote_files={"polygons/mexico-latest.parquet", "README.md"})
    inventory = RemoteInventory.fetch(repo_id="test/repo", hub=stub)
    assert inventory.contains("polygons/mexico-latest.parquet")
    assert not inventory.contains("polygons/hungary-latest.parquet")
    assert inventory.files == {"polygons/mexico-latest.parquet", "README.md"}


def test_remote_inventory_fetch_failure() -> None:
    class FailingHub(StubHfHub):
        def list_repo_files(self, repo_id: str, *, repo_type: str = "dataset") -> list[str]:
            raise RuntimeError("Network timeout")

    with pytest.raises(UploadError, match="Network timeout"):
        RemoteInventory.fetch(repo_id="test/repo", hub=FailingHub())


def test_remote_inventory_fetch_explicit_hub_no_credentials() -> None:
    # Explicit hub test double is supplied, resolver raises an error but fetch must succeed without resolving
    def failing_resolver(token: str | None) -> str:
        raise ValueError("Credentials not configured")

    stub = StubHfHub(remote_files={"polygons/mexico-latest.parquet"})
    inventory = RemoteInventory.fetch(
        repo_id="test/repo",
        hub=stub,
        token="some-token",
        _resolve_token=failing_resolver,
    )
    assert inventory.contains("polygons/mexico-latest.parquet")


def test_load_existing_core_artifacts_success(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    # Write valid schema parquet files
    poly_table = pa.Table.from_pylist(
        [{"polygon_id": "1", "lat": 1.0, "lon": 2.0}], schema=polygon_schema()
    )
    pq.write_table(poly_table, data_root.processed_polygons / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    links_table = pa.Table.from_pylist(
        [{"polygon_id": "1", "article_id": "a1"}], schema=polygon_article_schema()
    )
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    # Write manifest entry
    manifest_data = {
        f"{stem}.osm.pbf": {
            "source_pbf": f"{stem}.osm.pbf",
            "region": stem,
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
            "wikipedia_documents_path": f"wikipedia/documents/{stem}.parquet",
            "polygon_count": 1,
            "article_count": 1,
            "link_count": 1,
        }
    }
    (data_root.processed_manifests / "processed_pbfs.json").write_text(json.dumps(manifest_data))

    # Also wikipedia/documents for completed augmented region
    doc_table = pa.Table.from_pylist(
        [
            {
                "document_id": "1",
                "article_id": "a1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "es",
            }
        ],
        schema=wikipedia_document_schema(),
    )
    doc_dir = data_root.processed / "wikipedia" / "documents"
    doc_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(doc_table, doc_dir / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    artifacts = load_existing_core_artifacts(data_root, stem)
    assert isinstance(artifacts, CorePublicationArtifacts)
    assert artifacts.polygons_path == data_root.processed_polygons / f"{stem}.parquet"
    assert artifacts.polygon_articles_path == data_root.processed_links / f"{stem}.parquet"
    assert artifacts.wikipedia_documents_path == doc_dir / f"{stem}.parquet"


def test_load_existing_core_artifacts_validation_errors(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"

    # Missing files raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_existing_core_artifacts(data_root, stem)

    # Incomplete schema parquet file raises ValueError
    data_root.processed_polygons.mkdir(parents=True, exist_ok=True)
    (data_root.processed_polygons / f"{stem}.parquet").write_text("not-parquet")
    (data_root.processed_links / f"{stem}.parquet").write_text("not-parquet")
    manifest_entry: dict[str, dict[str, object]] = {f"{stem}.osm.pbf": {}}
    (data_root.processed_manifests / "processed_pbfs.json").write_text(json.dumps(manifest_entry))

    with pytest.raises(ValueError):
        load_existing_core_artifacts(data_root, stem)


def test_load_existing_core_artifacts_manifest_validation_gaps(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"

    # Set up valid files
    poly_table = pa.Table.from_pylist([{"polygon_id": "1"}], schema=polygon_schema())
    pq.write_table(poly_table, data_root.processed_polygons / f"{stem}.parquet")  # type: ignore[no-untyped-call]
    links_table = pa.Table.from_pylist([{"polygon_id": "1"}], schema=polygon_article_schema())
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    # Write invalid manifest entries to verify project-specific validation error
    manifest_entry = {
        f"{stem}.osm.pbf": {
            "source_pbf": "wrong-source.osm.pbf",  # Mismatch source_pbf
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
        }
    }
    (data_root.processed_manifests / "processed_pbfs.json").write_text(json.dumps(manifest_entry))

    with pytest.raises(PublicationValidationError, match="source_pbf mismatch"):
        load_existing_core_artifacts(data_root, stem)

    # Mismatch polygons path
    manifest_entry[f"{stem}.osm.pbf"]["source_pbf"] = f"{stem}.osm.pbf"
    manifest_entry[f"{stem}.osm.pbf"]["polygons_path"] = "polygons/wrong-name.parquet"
    (data_root.processed_manifests / "processed_pbfs.json").write_text(json.dumps(manifest_entry))

    with pytest.raises(PublicationValidationError, match="polygons_path mismatch"):
        load_existing_core_artifacts(data_root, stem)


def test_reconciliation_planner_mexico_state(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"

    # Set up local complete files
    poly_table = pa.Table.from_pylist([{"polygon_id": "1"}], schema=polygon_schema())
    pq.write_table(poly_table, data_root.processed_polygons / f"{stem}.parquet")  # type: ignore[no-untyped-call]
    links_table = pa.Table.from_pylist([{"polygon_id": "1"}], schema=polygon_article_schema())
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    manifest_data = {
        f"{stem}.osm.pbf": {
            "source_pbf": f"{stem}.osm.pbf",
            "region": stem,
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
            "wikipedia_documents_path": f"wikipedia/documents/{stem}.parquet",
            "polygon_count": 1,
            "article_count": 1,
            "link_count": 1,
        }
    }
    (data_root.processed_manifests / "processed_pbfs.json").write_text(json.dumps(manifest_data))

    # Complete local augmentation
    wikipedia_documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    wikipedia_sections_path = data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet"
    wikivoyage_documents_path = data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
    wikivoyage_sections_path = data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"
    wikidata_facts_path = data_root.processed / "wikidata" / "facts" / f"{stem}.parquet"

    for p in [
        wikipedia_documents_path,
        wikipedia_sections_path,
        wikivoyage_documents_path,
        wikivoyage_sections_path,
        wikidata_facts_path,
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Dummy parquet table
        pq.write_table(pa.table({"x": [1]}), p)  # type: ignore[no-untyped-call]

    # Augmentation manifest
    aug_manifest = {
        stem: {
            "contract_version": "text-sidecars-v1",
            "core_hashes": {
                str(data_root.processed_polygons / f"{stem}.parquet"): "a" * 64,
                str(wikipedia_documents_path): "b" * 64,
            },
            "counts": {
                "wikipedia_documents": 1,
                "wikipedia_sections": 1,
                "wikivoyage_documents": 1,
                "wikivoyage_sections": 1,
                "wikidata_facts": 1,
            },
        }
    }
    aug_manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    aug_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    aug_manifest_path.write_text(json.dumps(aug_manifest))

    # Remote missing polygons and links, but has augmentation sidecars
    inventory = RemoteInventory(
        {
            f"wikipedia/documents/{stem}.parquet",
            f"wikipedia/sections/{stem}.parquet",
            f"wikivoyage/documents/{stem}.parquet",
            f"wikivoyage/sections/{stem}.parquet",
            f"wikidata/facts/{stem}.parquet",
            "README.md",
            "manifests/processed_pbfs.json",
            "manifests/augmentation_manifest.json",
        }
    )

    # Planner
    planner = ReconciliationPlanner(data_root, inventory, stems={stem})
    plan = planner.plan()

    assert (stem, "polygons") in plan.missing
    assert (stem, "polygon_articles") in plan.missing
    assert (stem, "wikipedia/documents") not in plan.missing


def test_corpus_identity_preserved(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"

    poly_table = pa.Table.from_pylist([{"polygon_id": "1"}], schema=polygon_schema())
    pq.write_table(poly_table, data_root.processed_polygons / f"{stem}.parquet")  # type: ignore[no-untyped-call]
    links_table = pa.Table.from_pylist([{"polygon_id": "1"}], schema=polygon_article_schema())
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    manifest_data = {
        f"{stem}.osm.pbf": {
            "source_pbf": f"{stem}.osm.pbf",
            "region": stem,
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
        }
    }
    (data_root.processed_manifests / "processed_pbfs.json").write_text(json.dumps(manifest_data))

    # Remote missing everything
    inventory = RemoteInventory(set())
    planner = ReconciliationPlanner(data_root, inventory, stems={stem})
    plan = planner.plan()

    # Verify present/missing sets contain exact seven canonical corpus identifiers
    # instead of collapsed names like "wikipedia"
    missing_corpora = {corp for _, corp in plan.missing}
    assert "polygons" in missing_corpora
    assert "polygon_articles" in missing_corpora


def test_incomplete_augmentation_resumability_under_missing_completion(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"

    poly_table = pa.Table.from_pylist([{"polygon_id": "1"}], schema=polygon_schema())
    pq.write_table(poly_table, data_root.processed_polygons / f"{stem}.parquet")  # type: ignore[no-untyped-call]
    links_table = pa.Table.from_pylist([{"polygon_id": "1"}], schema=polygon_article_schema())
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    manifest_data = {
        f"{stem}.osm.pbf": {
            "source_pbf": f"{stem}.osm.pbf",
            "region": stem,
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
        }
    }
    (data_root.processed_manifests / "processed_pbfs.json").write_text(json.dumps(manifest_data))

    # Partial sidecar file exists but NO completion manifest entry
    wikipedia_sections_path = data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet"
    wikipedia_sections_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), wikipedia_sections_path)  # type: ignore[no-untyped-call]

    inventory = RemoteInventory(set())
    planner = ReconciliationPlanner(data_root, inventory, stems={stem})
    # Should not raise an error, but classify as AUGMENT
    plan = planner.plan()
    assert stem in plan.stems_to_augment


def test_fail_closed_under_completed_manifest_missing_files(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"

    poly_table = pa.Table.from_pylist([{"polygon_id": "1"}], schema=polygon_schema())
    pq.write_table(poly_table, data_root.processed_polygons / f"{stem}.parquet")  # type: ignore[no-untyped-call]
    links_table = pa.Table.from_pylist([{"polygon_id": "1"}], schema=polygon_article_schema())
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    manifest_data = {
        f"{stem}.osm.pbf": {
            "source_pbf": f"{stem}.osm.pbf",
            "region": stem,
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
        }
    }
    (data_root.processed_manifests / "processed_pbfs.json").write_text(json.dumps(manifest_data))

    # Augmentation manifest claims region completed
    aug_manifest = {stem: {"contract_version": "text-sidecars-v1"}}
    aug_manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    aug_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    aug_manifest_path.write_text(json.dumps(aug_manifest))

    # But required canonical document is missing!
    inventory = RemoteInventory(set())
    planner = ReconciliationPlanner(data_root, inventory, stems={stem})
    with pytest.raises(ValueError, match="missing required canonical documents file"):
        planner.plan()


def test_malformed_augmentation_manifest_json_raises(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"

    # Malformed JSON in augmentation_manifest.json
    aug_manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    aug_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    aug_manifest_path.write_text("invalid{json")

    inventory = RemoteInventory(set())
    planner = ReconciliationPlanner(data_root, inventory, stems={stem})
    with pytest.raises(ValueError, match="Malformed augmentation manifest JSON"):
        planner.plan()


def _setup_minimal_region(data_root: DataRoot, stem: str) -> None:
    data_root.processed_polygons.mkdir(parents=True, exist_ok=True)
    data_root.processed_links.mkdir(parents=True, exist_ok=True)
    data_root.processed_manifests.mkdir(parents=True, exist_ok=True)

    poly_table = pa.Table.from_pylist(
        [{"polygon_id": "1", "lat": 19.0, "lon": -99.0}], schema=polygon_schema()
    )
    pq.write_table(poly_table, data_root.processed_polygons / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    links_table = pa.Table.from_pylist(
        [{"polygon_id": "1", "article_id": "a1"}], schema=polygon_article_schema()
    )
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    manifest_data = {
        f"{stem}.osm.pbf": {
            "source_pbf": f"{stem}.osm.pbf",
            "region": stem,
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
            "wikipedia_documents_path": f"wikipedia/documents/{stem}.parquet",
            "polygon_count": 1,
            "article_count": 1,
            "link_count": 1,
        }
    }
    (data_root.processed_manifests / "processed_pbfs.json").write_text(json.dumps(manifest_data))


def test_metadata_only_upload_contract(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    _setup_minimal_region(data_root, "mexico-latest")

    # assemble_metadata_only_upload must put README.md as the absolute final add operation
    ops = assemble_metadata_only_upload(
        data_root=data_root,
        repo_id="test/repo",
    )
    assert len(ops) > 0
    final_op = ops[-1]
    assert final_op.action == "add"
    assert final_op.path_in_repo == "README.md"


def test_repository_refresh_uses_current_three_map_contract(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    plan = ReconciliationPlanner(
        data_root,
        RemoteInventory(set()),
        stems=set(),
    ).plan()

    assert "assets/coverage_map.png" in plan.repository_refresh
    assert "assets/geographic_text_density.png" in plan.repository_refresh
    assert "assets/geographic_polygon_count.png" not in plan.repository_refresh
    assert "assets/geographic_wikipedia_text_coverage.png" not in plan.repository_refresh
