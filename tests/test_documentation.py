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
