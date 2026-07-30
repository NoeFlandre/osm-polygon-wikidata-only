"""Ownership contracts for dataset publication models."""

from osm_polygon_wikidata_only.hf import publication
from osm_polygon_wikidata_only.hf._publication import artifacts
from osm_polygon_wikidata_only.hf._publication.models import (
    CorePublicationArtifacts,
    PublicationValidationError,
)


def test_publication_facade_reexports_models_by_identity() -> None:
    assert publication.CorePublicationArtifacts is CorePublicationArtifacts
    assert publication.PublicationValidationError is PublicationValidationError


def test_publication_uses_focused_artifact_validation_and_loading() -> None:
    assert publication._validate_core_artifacts is artifacts.validate_core_artifacts
    assert publication._validate_augmentation_artifacts is artifacts.validate_augmentation_artifacts
    assert publication.load_existing_core_artifacts is artifacts.load_existing_core_artifacts
