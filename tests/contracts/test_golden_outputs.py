"""Golden output tests against checked-in fixtures.

The dataset card Markdown and the unified-sync publication file list
are public, stable artifacts. Any drift is a regression; the golden
files in ``tests/fixtures/golden/`` capture the exact expected
output.
"""

from __future__ import annotations

import json
from pathlib import Path

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.cli.commands import _sync_upload_files
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    ARTICLE_DESCRIPTIONS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_ARTICLE_DESCRIPTIONS,
    POLYGON_COLUMNS,
    POLYGON_DESCRIPTIONS,
)
from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN = FIXTURE_ROOT / "golden"
FIXTURE_PROCESSED = FIXTURE_ROOT / "processed"
STEM = "monaco-latest"
REPO_ID = "NoeFlandre/osm-polygon-wikidata-only"


def _render_card() -> str:
    return render_dataset_card(
        repo_id=REPO_ID,
        stats={"polygon_count": 1, "article_count": 1, "unique_wikidata_count": 1},
        polygon_columns=list(POLYGON_COLUMNS),
        polygon_descriptions=POLYGON_DESCRIPTIONS,
        article_columns=list(ARTICLE_COLUMNS),
        article_descriptions=ARTICLE_DESCRIPTIONS,
        link_columns=list(POLYGON_ARTICLE_COLUMNS),
        link_descriptions=POLYGON_ARTICLE_DESCRIPTIONS,
        maintainer="Noé Flandre",
    )


def _seed_data_root(tmp: Path) -> DataRoot:
    """Populate ``tmp`` so :func:`_sync_upload_files` can build its snapshots.

    Mirrors the exact paths :func:`process_extracted_pbf` and
    :func:`augment_region` write in production; we copy the small
    committed parquet fixtures rather than re-running the pipeline.
    """
    data_root = DataRoot(tmp)
    data_root.ensure()

    polygons = data_root.processed_polygons / f"{STEM}.parquet"
    articles = data_root.processed_articles / f"{STEM}.parquet"
    polygon_articles = data_root.processed_links / f"{STEM}.parquet"
    polygons.parent.mkdir(parents=True, exist_ok=True)
    articles.parent.mkdir(parents=True, exist_ok=True)
    polygon_articles.parent.mkdir(parents=True, exist_ok=True)

    polygons.write_bytes((FIXTURE_PROCESSED / "polygons" / f"{STEM}.parquet").read_bytes())
    articles.write_bytes((FIXTURE_PROCESSED / "articles" / f"{STEM}.parquet").read_bytes())
    polygon_articles.write_bytes(
        (FIXTURE_PROCESSED / "polygon_articles" / f"{STEM}.parquet").read_bytes()
    )

    wikipedia_docs = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    wikipedia_sections = data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet"
    wikivoyage_docs = data_root.processed / "wikivoyage" / "documents" / f"{STEM}.parquet"
    wikivoyage_sections = data_root.processed / "wikivoyage" / "sections" / f"{STEM}.parquet"
    wikidata_facts = data_root.processed / "wikidata" / "facts" / f"{STEM}.parquet"

    wikipedia_docs.parent.mkdir(parents=True, exist_ok=True)
    wikipedia_sections.parent.mkdir(parents=True, exist_ok=True)
    wikivoyage_docs.parent.mkdir(parents=True, exist_ok=True)
    wikivoyage_sections.parent.mkdir(parents=True, exist_ok=True)
    wikidata_facts.parent.mkdir(parents=True, exist_ok=True)

    wikipedia_docs.write_bytes(
        (FIXTURE_PROCESSED / "wikipedia" / "documents" / f"{STEM}.parquet").read_bytes()
    )
    wikipedia_sections.write_bytes(
        (FIXTURE_PROCESSED / "wikipedia" / "sections" / f"{STEM}.parquet").read_bytes()
    )
    wikivoyage_docs.write_bytes(
        (FIXTURE_PROCESSED / "wikivoyage" / "documents" / f"{STEM}.parquet").read_bytes()
    )
    wikivoyage_sections.write_bytes(
        (FIXTURE_PROCESSED / "wikivoyage" / "sections" / f"{STEM}.parquet").read_bytes()
    )
    wikidata_facts.write_bytes(
        (FIXTURE_PROCESSED / "wikidata" / "facts" / f"{STEM}.parquet").read_bytes()
    )

    aug_manifest_dir = data_root.processed / "augmentation" / "manifests"
    aug_manifest_dir.mkdir(parents=True, exist_ok=True)
    aug_manifest = aug_manifest_dir / "augmentation_manifest.json"
    aug_manifest.write_text(
        (FIXTURE_PROCESSED / "manifests" / "processed_pbfs.json").read_text(),
        encoding="utf-8",
    )

    processed_manifests_dir = data_root.processed_manifests
    processed_manifests_dir.mkdir(parents=True, exist_ok=True)
    (processed_manifests_dir / "processed_pbfs.json").write_text(
        (FIXTURE_PROCESSED / "manifests" / "processed_pbfs.json").read_text(),
        encoding="utf-8",
    )

    return data_root


def _split_publication(files: list[tuple[Path, str]]) -> dict[str, list[str]]:
    """Mirror the documented split: core artifacts before augmentation.

    The production assembly path always emits the seven core artifacts
    (parquets + manifest + coverage maps) before the seven augmentation
    artifacts (parquets + manifests + README). The split is fixed by
    the documented ``core``/``augmentation`` keys.
    """
    remote = [remote for _, remote in files]
    cut = len(remote) - 7 if len(remote) >= 14 else 7
    return {"core": remote[:cut], "augmentation": remote[cut:]}


def _build_result_objects(data_root: DataRoot) -> tuple[ProcessResult, AugmentationResult]:
    polygons = data_root.processed_polygons / f"{STEM}.parquet"
    articles = data_root.processed_articles / f"{STEM}.parquet"
    polygon_articles = data_root.processed_links / f"{STEM}.parquet"
    manifest_path = data_root.processed_manifests / "processed_pbfs.json"
    aug_manifest = data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    core = ProcessResult(
        polygons_path=polygons,
        articles_path=articles,
        polygon_articles_path=polygon_articles,
        manifest_path=manifest_path,
        polygon_count=1,
        article_count=1,
        link_count=1,
        manifest_entry={
            "polygon_count": 1,
            "article_count": 1,
            "unique_wikidata_count": 1,
        },
        stage_timings_s={},
    )
    augmentation = AugmentationResult(
        wikipedia_documents_path=data_root.processed
        / "wikipedia"
        / "documents"
        / f"{STEM}.parquet",
        wikipedia_sections_path=data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet",
        wikivoyage_documents_path=data_root.processed
        / "wikivoyage"
        / "documents"
        / f"{STEM}.parquet",
        wikivoyage_sections_path=data_root.processed
        / "wikivoyage"
        / "sections"
        / f"{STEM}.parquet",
        wikidata_facts_path=data_root.processed / "wikidata" / "facts" / f"{STEM}.parquet",
        manifest_path=aug_manifest,
        counts={},
    )
    return core, augmentation


def test_dataset_card_matches_golden() -> None:
    """The dataset card Markdown must match the checked-in golden file exactly."""
    golden_path = GOLDEN / "dataset_card.md"
    assert golden_path.exists(), "dataset card golden fixture missing"
    assert _render_card() == golden_path.read_text(encoding="utf-8")


def test_publication_file_list_matches_golden(tmp_path: Path) -> None:
    """The unified-sync publication file list is locked by a golden JSON.

    This test invokes the production assembly path used by
    ``sync-dir`` (``osm_polygon_wikidata_only.cli.commands._sync_upload_files``)
    with deterministic stubs derived from the committed parquet
    fixtures. It captures the remote-path ordering returned by the
    real production function and compares it byte-for-byte against
    ``tests/fixtures/golden/publication_file_list.json``.
    """
    data_root = _seed_data_root(tmp_path)
    core, augmentation = _build_result_objects(data_root)

    files = _sync_upload_files(data_root, REPO_ID, STEM, augmentation, core)
    remote_by_split = _split_publication(files)

    golden_path = GOLDEN / "publication_file_list.json"
    assert golden_path.exists(), "publication golden fixture missing"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert remote_by_split == golden, (
        f"sync-dir publication order drifted: got {remote_by_split!r}, want {golden!r}"
    )
