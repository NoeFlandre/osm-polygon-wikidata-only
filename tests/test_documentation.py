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
