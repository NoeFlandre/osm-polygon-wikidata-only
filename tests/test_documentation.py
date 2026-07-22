"""Reader-facing documentation contract tests."""

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def test_readme_documents_complete_wikimedia_bot_password_workflow() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")

    required_text = (
        "https://meta.wikimedia.org/wiki/Special:BotPasswords",
        "WIKIMEDIA_BOT_USERNAME",
        "WIKIMEDIA_BOT_PASSWORD",
        "WIKIMEDIA_REQUESTS_PER_MINUTE",
        "--skip-existing",
        "--push",
        "revoke",
        "Do not commit",
    )
    for text in required_text:
        assert text in readme


def test_security_and_development_docs_cover_bot_password_handling() -> None:
    security = (REPOSITORY / "SECURITY.md").read_text(encoding="utf-8")
    development = (REPOSITORY / "docs/development.md").read_text(encoding="utf-8")

    assert "WIKIMEDIA_BOT_PASSWORD" in security
    assert "revoke" in security.lower()
    assert "browser cookies" in security.lower()
    assert "WIKIMEDIA_BOT_USERNAME" in development
    assert "WIKIMEDIA_REQUESTS_PER_MINUTE" in development
    assert "live credentials" in development.lower()


def test_public_docs_never_contain_test_password() -> None:
    documents = [
        REPOSITORY / "README.md",
        REPOSITORY / "SECURITY.md",
        REPOSITORY / "docs/development.md",
        REPOSITORY / "docs/architecture.md",
    ]

    for document in documents:
        assert "secret-value" not in document.read_text(encoding="utf-8")


def test_public_docs_explain_enrichment_progress_heartbeat() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8").lower()
    architecture = (REPOSITORY / "docs/architecture.md").read_text(encoding="utf-8").lower()

    for document in (readme, architecture):
        assert "two-minute" in document
        assert "qid" in document
        assert "wikipedia site" in document
        assert "articles attempted" in document
        assert "eta" in document
        assert "request pacing" in document


def test_readme_documents_geographic_coverage_section() -> None:
    """The README must reference both visualization assets and explain them concisely."""
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    assert "## Geographic coverage" in readme
    assert "assets/geographic_wikipedia_text_coverage.png" in readme
    assert "assets/geographic_polygon_count.png" in readme
    assert readme.count("assets/geographic_wikipedia_text_coverage.png") == 1
    assert readme.count("assets/geographic_polygon_count.png") == 1
    # The coverage section must mention the formula, denominator, and conditioning.
    assert "denominator" in readme.lower()
    assert "wikidata" in readme.lower()
    # No opacity encoding claim.
    assert "opacity encodes" not in readme.lower()


def test_readme_states_both_geographic_coverage_formulas() -> None:
    """The README must spell out the coverage_rate and polygon_count formulas once each."""
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    coverage_section = readme.split("## Geographic coverage", 1)[1].split("\n## ", 1)[0]

    # Coverage rate formula
    assert "coverage_rate" in coverage_section
    assert "covered_polygons" in coverage_section
    assert "all_dataset_polygons" in coverage_section
    # A covered polygon definition appears once.
    assert "non-empty text" in coverage_section

    # Polygon count formula
    assert "polygon_count" in coverage_section
    assert "centroid" in coverage_section.lower()
    assert "H3 cell" in coverage_section

    # The conditioning clause appears once and is not duplicated.
    conditioning_phrase = "OSM `wikidata=*` tag"
    assert coverage_section.count(conditioning_phrase) == 1
    assert "wikidata" in coverage_section.lower()


def test_architecture_documents_geographic_coverage_generation() -> None:
    architecture = (REPOSITORY / "docs/architecture.md").read_text(encoding="utf-8")
    assert "Geographic coverage" in architecture
    assert "geographic_polygon_count.png" in architecture
    assert "geographic_wikipedia_text_coverage.png" in architecture
    assert "H3" in architecture
    assert "20 polygons" in architecture or "twenty polygons" in architecture.lower()
    assert "logarithmic" in architecture.lower()
    # The architecture doc must no longer claim opacity encodes polygon count.
    assert "opacity encodes" not in architecture.lower()


def test_readme_documents_five_augmentation_sidecars() -> None:
    """The source README must mention the five augmentation sidecars in the
    Output schema section without claiming hardcoded counts."""
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    for path in (
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
    ):
        assert path in readme, f"missing augmentation path {path} in README"


def test_readme_documents_regenerated_dataset_card() -> None:
    """The source README must explain that the published dataset card is
    regenerated automatically and reports factual statistics derived from
    the local Parquet files."""
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    assert "Generated dataset card" in readme
    # 'regenerated' is on one line and 'automatically' on the next.
    assert "regenerated" in readme and "automatically" in readme
    assert "local finalized Parquet" in readme or "local finalized Parquet files" in readme
    assert "write_readme_snapshot" in readme


def test_architecture_documents_augmentation_readme_recomputation() -> None:
    """Architecture doc must document that ``write_readme_snapshot``
    recomputes both core and augmentation statistics."""
    architecture = (REPOSITORY / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "write_readme_snapshot" in architecture
    assert "core and augmentation stats" in architecture or (
        "recomputes both" in architecture.lower()
    )


def test_readme_describes_current_public_workflow_without_migration_language() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")

    assert "at most eight requests in flight" in readme
    assert "groups of 25 QIDs" in readme
    assert "cache/wikidata_recovery/checkpoints" in readme
    assert "1,300+ tracked tests" in readme
    assert "lossless" not in readme.lower()
    assert "at most three requests in flight" not in readme
    assert "suite is fast (< 2 s)" not in readme


def test_readme_repository_layout_names_current_focused_modules() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    layout = readme.split("## Repository layout", 1)[1].split("\n## ", 1)[0]

    for name in (
        "augmentation/",
        "enrichment/wikidata/",
        "enrichment/wikipedia/",
        "hf/_dataset_stats/",
        "hf/_geographic/",
        "hf/_uploader/",
        "pipeline/_wikidata_recovery/",
    ):
        assert name in layout
    assert "tests/               # pytest suite (114+" not in layout


def test_developer_docs_use_current_test_paths_and_quality_gate() -> None:
    development = (REPOSITORY / "docs/development.md").read_text(encoding="utf-8")

    assert "tests/enrichment/test_wikimedia_auth.py" in development
    assert "tests/cli/test_dependencies.py" in development
    assert "git diff --check" in development
    assert "uv build" in development


def test_architecture_names_current_private_ownership_boundaries() -> None:
    architecture = (REPOSITORY / "docs/architecture.md").read_text(encoding="utf-8")

    assert "hf._publication.models" in architecture
    assert "pipeline._wikidata_recovery.storage" in architecture
    assert "RecoveryRepairResult" in architecture
