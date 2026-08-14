"""Reader-facing documentation contract tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml

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


def test_home_page_explains_v1_and_v2_contracts() -> None:
    index = (REPOSITORY / "docs/index.md").read_text(encoding="utf-8")
    assert "V1" in index
    assert "V2" in index
    assert "without a Wikidata QID" in index
    assert "polygon_document_links/<stem>.parquet" in index
    assert "link_sources" in index
    assert "wikipedia_tag_refs" in index
    assert "docs/citations/osm-polygon-wikidata-only.cff" in index
    assert "docs/citations/osm-polygon-wikidata-and-wikipedia.cff" in index


def test_pages_workflow_builds_strict_site_and_deploys_artifact() -> None:
    workflow = yaml.safe_load(
        (REPOSITORY / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    )
    build = workflow["jobs"]["build"]
    deploy = workflow["jobs"]["deploy"]
    build_run = "\n".join(step.get("run", "") for step in build["steps"] if isinstance(step, dict))
    assert "mkdocs build --strict --site-dir site" in build_run
    assert any(step.get("uses") == "actions/upload-pages-artifact@v3" for step in build["steps"])
    assert build["permissions"] == {"contents": "read", "pages": "read"}
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert any(step.get("uses") == "actions/deploy-pages@v4" for step in deploy["steps"])


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
    assert readme.count("| `--skip-existing` |") == 1


def test_local_presentation_build_outputs_are_ignored() -> None:
    """Generated slide captures must not pollute a contributor worktree."""
    gitignore = (REPOSITORY / ".gitignore").read_text(encoding="utf-8")

    assert "/presentations/captures/" in gitignore
    assert "/presentations/output/" in gitignore


def test_security_and_development_docs_cover_bot_password_handling() -> None:
    security = (REPOSITORY / "SECURITY.md").read_text(encoding="utf-8")
    development = (REPOSITORY / "docs/development.md").read_text(encoding="utf-8")

    assert "WIKIMEDIA_BOT_PASSWORD" in security
    assert "revoke" in security.lower()
    assert "browser cookies" in security.lower()
    assert "WIKIMEDIA_BOT_USERNAME" in development
    assert "WIKIMEDIA_BOT_PASSWORD" in development
    assert "WIKIMEDIA_REQUESTS_PER_MINUTE" in development
    assert "all-or-nothing" in development
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


def test_public_docs_do_not_expose_personal_storage_layout() -> None:
    for document in (
        REPOSITORY / "README.md",
        REPOSITORY / "docs/architecture.md",
    ):
        text = document.read_text(encoding="utf-8")
        assert "/Volumes/" not in text
        assert "Seagate" not in text
        assert "external drive" not in text.lower()


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
    """The README references the three current public maps only."""
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    assert "## Geographic coverage" in readme
    assert readme.count("assets/geographic_text_presence.png") == 1
    assert readme.count("assets/coverage_map.png") == 1
    assert readme.count("assets/geographic_text_density.png") == 1
    assert "assets/geographic_wikipedia_text_coverage.png" not in readme
    assert "assets/geographic_polygon_count.png" not in readme


def test_readme_states_combined_geographic_density_definition() -> None:
    """The README defines the raw cross-project H3 count."""
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    coverage_section = readme.split("## Geographic coverage", 1)[1].split("\n## ", 1)[0]

    assert "raw number of polygons" in coverage_section
    assert "Wikipedia or Wikivoyage" in coverage_section
    assert "non-empty text" in coverage_section
    assert "counted once" in coverage_section
    assert "not a proportion" in coverage_section
    assert "H3 cell" in coverage_section


def test_architecture_documents_geographic_coverage_generation() -> None:
    architecture = (REPOSITORY / "docs/architecture.md").read_text(encoding="utf-8")
    assert "Geographic coverage" in architecture
    assert "geographic_text_density.png" in architecture
    assert "H3" in architecture
    assert "logarithmic" in architecture.lower()


def test_pages_workflow_publishes_dataset_presentation() -> None:
    workflow = (REPOSITORY / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    for path in (
        "presentations/dataset.html",
        "presentations/codebase.html",
        "presentations/assets/coverage_map.png",
        "presentations/assets/text_density.png",
        "presentations/assets/text_presence.png",
    ):
        assert path in workflow
        assert (REPOSITORY / path).is_file(), f"missing tracked Pages source {path}"


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


def test_readme_image_references_have_repository_assets() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    for path in (
        "assets/dataset_hero.png",
        "assets/geographic_text_presence.png",
        "assets/coverage_map.png",
        "assets/geographic_text_density.png",
    ):
        assert path in readme
        assert (REPOSITORY / path).is_file(), f"missing README asset {path}"


def test_readme_places_trackio_snapshot_after_hero_and_intro() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    hero = "![OSM Polygon Wikidata dataset overview](assets/dataset_hero.png)"
    intro = "Extract polygonal OpenStreetMap features"
    trackio = "## Trackio snapshot"
    codebase_presentation = (
        "https://noeflandre.github.io/osm-polygon-wikidata-only/presentations/codebase.html"
    )
    project = "## What this project does"
    assert readme.index(hero) < readme.index(intro) < readme.index(trackio) < readme.index(project)
    assert codebase_presentation in readme


def test_readme_places_v1_blog_post_after_hero() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    hero = "![OSM Polygon Wikidata dataset overview](assets/dataset_hero.png)"
    blog = "V1 blog post: [How to describe a place on Earth with Wikidata]"
    intro = "Extract polygonal OpenStreetMap features"
    assert blog in readme
    assert readme.index(hero) < readme.index(blog) < readme.index(intro)


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


def test_architecture_documents_augmentation_readme_recomputation() -> None:
    """Architecture doc must describe factual card recomputation."""
    architecture = (REPOSITORY / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "core and" in architecture
    assert "augmentation statistics" in architecture
    assert "finalized tables" in architecture


def test_architecture_qualifies_skip_existing_behavior() -> None:
    architecture = (REPOSITORY / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "--skip-existing` skips completed local processing" in architecture
    assert "remote reconciliation" in architecture


def test_readme_describes_current_public_workflow_without_migration_language() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")

    assert "at most twelve requests in flight" in readme
    assert "(anonymous runs default to three)" in readme
    assert "groups of 25 QIDs" in readme
    assert "checkpoints" in readme.lower()
    assert "The tracked test suite is deterministic and requires no live network" in readme
    assert "1,300+ tracked tests" not in readme
    assert "lossless" not in readme.lower()
    assert "at most three requests in flight" not in readme
    assert "cache/" not in readme
    assert "processed/articles/" not in readme
    assert "suite is fast (< 2 s)" not in readme


def test_readme_repository_layout_names_current_focused_modules() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    layout = readme.split("## Repository layout", 1)[1].split("\n## ", 1)[0]

    for name in (
        "augmentation/",
        "domain/",
        "io/",
        "cli/",
        "enrichment/",
        "hf/",
        "pipeline/",
        "utils/",
        "v2/",
    ):
        assert name in layout
    assert "tests/               # pytest suite (114+" not in layout


def test_developer_docs_use_current_test_paths_and_quality_gate() -> None:
    development = (REPOSITORY / "docs/development.md").read_text(encoding="utf-8")

    assert "tests/enrichment/test_wikimedia_auth.py" in development
    assert "tests/cli/test_dependencies.py" in development
    assert "git diff --check" in development
    assert "uv build" in development
    assert (
        "uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q" in development
    )


def test_architecture_keeps_private_modules_out_of_public_docs() -> None:
    architecture = (REPOSITORY / "docs/architecture.md").read_text(encoding="utf-8")
    assert "hf._" not in architecture
    assert "pipeline._" not in architecture
    assert "cli._" not in architecture


def test_current_documentation_uses_uv_ruff_and_ty_quality_gate() -> None:
    documents = (
        REPOSITORY / "README.md",
        REPOSITORY / "docs/development.md",
        REPOSITORY / "docs/architecture.md",
    )
    combined = "\n".join(document.read_text(encoding="utf-8") for document in documents)

    assert "uv run ruff check src tests scripts" in combined
    assert "uv run ty check src scripts" in combined
    assert "mypy" not in combined.lower()


def test_public_docs_explain_the_complete_project_tooling_stack() -> None:
    documents = (
        REPOSITORY / "README.md",
        REPOSITORY / "docs/development.md",
        REPOSITORY / "docs/architecture.md",
    )
    combined = "\n".join(document.read_text(encoding="utf-8") for document in documents).lower()

    for tool in (
        "uv",
        "ruff",
        "ty",
        "pytest",
        "pre-commit",
        "typer",
        "rich",
        "tqdm",
        "just",
        "github actions",
    ):
        assert tool in combined
    assert "osm-polygon-wikidata-only-audit-remote" in combined
