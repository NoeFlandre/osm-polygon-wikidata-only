"""Documentation safety checks: no leaked secrets, no broken links or assets."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from scripts.assemble_docs_site import PRESENTATION_FILES

REPOSITORY = Path(__file__).resolve().parents[1]


def _public_markdown_files() -> Iterator[Path]:
    yield REPOSITORY / "README.md"
    yield from sorted((REPOSITORY / "docs").glob("*.md"))


def test_public_docs_omit_private_operator_details() -> None:
    """Public Markdown should not publish local internals or cache layouts."""
    forbidden = (
        "/Users/",
        "/Volumes/",
        "Seagate",
        "superpowers/",
        "cache/",
        "processed/articles/",
        "cli/_sync/",
        "hf/_dataset_stats/",
        "hf/_geographic/",
        "hf/_publication/",
        "hf/_uploader/",
        "pipeline/_link_migration/",
        "pipeline/_wikidata_recovery/",
        "hf._",
        "pipeline._",
        "cli._",
        "secret-value",
    )
    for document in _public_markdown_files():
        text = document.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"{marker!r} leaked into {document}"


def test_mkdocs_navigation_points_to_existing_public_pages() -> None:
    config = yaml.safe_load((REPOSITORY / "mkdocs.yml").read_text(encoding="utf-8"))
    nav = config["nav"]
    targets = [
        target
        for entry in nav
        for target in entry.values()
        if isinstance(target, str) and target.endswith(".md")
    ]
    assert targets
    for target in targets:
        assert (REPOSITORY / "docs" / target).is_file(), target
    assert config["exclude_docs"].split()


def test_superpowers_archive_is_removed_ignored_and_not_excluded() -> None:
    archive = REPOSITORY / "docs/superpowers"
    assert not any(path.is_file() for path in archive.rglob("*"))
    assert (
        "docs/superpowers/" in (REPOSITORY / ".gitignore").read_text(encoding="utf-8").splitlines()
    )
    config = yaml.safe_load((REPOSITORY / "mkdocs.yml").read_text(encoding="utf-8"))
    assert all("superpowers" not in pattern for pattern in config["exclude_docs"].split())


def test_public_docs_never_contain_test_password() -> None:
    documents = [
        REPOSITORY / "README.md",
        REPOSITORY / "SECURITY.md",
        REPOSITORY / "docs/development.md",
        REPOSITORY / "docs/architecture.md",
    ]

    for document in documents:
        assert "secret-value" not in document.read_text(encoding="utf-8")


def test_public_docs_do_not_expose_personal_storage_layout() -> None:
    for document in (
        REPOSITORY / "README.md",
        REPOSITORY / "docs/architecture.md",
    ):
        text = document.read_text(encoding="utf-8")
        assert "/Volumes/" not in text
        assert "Seagate" not in text
        assert "external drive" not in text.lower()


@pytest.mark.repository
def test_published_presentation_sources_exist() -> None:
    """`presentations/` is excluded from the Docker context by design."""
    for path in PRESENTATION_FILES:
        assert (REPOSITORY / path).is_file(), f"missing tracked Pages source {path}"


def test_readme_image_references_have_repository_assets() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    references = re.findall(r"!\[[^\]]*\]\((?!https?://)([^)\s]+)\)", readme)
    assert references
    for path in references:
        assert (REPOSITORY / path).is_file(), f"missing README asset {path}"


def test_dataset_citation_files_are_valid_and_point_to_their_hubs() -> None:
    expected = {
        REPOSITORY
        / "docs/citations/osm-polygon-wikidata-only.cff": "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only",
        REPOSITORY
        / "docs/citations/osm-polygon-wikidata-and-wikipedia.cff": "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia",
    }
    for path, dataset_url in expected.items():
        citation = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert citation["cff-version"] == "1.2.0"
        assert citation["type"] == "dataset"
        assert citation["authors"]
        assert citation["url"] == dataset_url
        assert citation["repository-code"] == (
            "https://github.com/NoeFlandre/osm-polygon-wikidata-only"
        )


def test_readme_is_a_concise_first_run_guide_with_factual_project_scope() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    lines = readme.splitlines()

    assert len(lines) < 200
    quick_start = lines.index("## Quick start")
    assert quick_start < 16
    assert "uv sync --frozen" in readme
    assert "monaco-latest.osm.pbf" in readme
    assert "download.geofabrik.de/europe/monaco-latest.osm.pbf" in readme
    assert "process-pbf" in readme
    assert "--data-root" in readme
    assert "--limit 100" in readme
    assert "--no-full-text" in readme
    assert "osm-polygon-wikidata-only" in readme
    assert "osm-polygon-wikidata-and-wikipedia" in readme
    assert "V1" in readme and "V2" in readme
    assert "polygons/<stem>.parquet" in readme
    assert "wikipedia/documents/<stem>.parquet" in readme
    assert "polygon_document_links/<stem>.parquet" in readme
    for target in (
        "docs/architecture.md",
        "docs/api.md",
        "docs/development.md",
        "docs/dataset-snapshot.md",
        "docs/citations/osm-polygon-wikidata-only.cff",
        "docs/citations/osm-polygon-wikidata-and-wikipedia.cff",
        "CITATION.cff",
    ):
        assert target in readme
    assert "OSM_POLYGON_DATA_ROOT" in readme
    assert "HF_TOKEN" in readme
    assert "docker compose run" in readme
    assert "Apache-2.0" in readme
    assert "ODbL 1.0" in readme
    assert "CC0 1.0" in readme
    assert "CC BY-SA 4.0" in readme
    assert "GFDL" in readme
    assert "Wikimedia Terms of Use" in readme
    assert "Wikivoyage copyleft policy" in readme
    assert "Snapshot metric" not in readme
    assert "Small snapshot table" not in readme


def test_dataset_snapshot_page_owns_metrics_and_links_both_dataset_cards() -> None:
    snapshot = (REPOSITORY / "docs/dataset-snapshot.md").read_text(encoding="utf-8")

    assert "Snapshot metric" in snapshot
    assert "Small snapshot table" in snapshot
    assert "final-dataset-snapshot" in snapshot
    assert "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only" in snapshot
    assert (
        "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia" in snapshot
    )
