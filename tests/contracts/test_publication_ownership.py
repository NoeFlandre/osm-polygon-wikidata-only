"""Ownership contracts for dataset publication models."""

from osm_polygon_wikidata_only.hf import publication
from osm_polygon_wikidata_only.hf._publication.models import (
    CorePublicationArtifacts,
    PublicationValidationError,
)


def test_publication_facade_reexports_models_by_identity() -> None:
    assert publication.CorePublicationArtifacts is CorePublicationArtifacts
    assert publication.PublicationValidationError is PublicationValidationError
