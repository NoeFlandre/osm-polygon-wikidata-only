"""Durability contracts for normal augmentation checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.augmentation.models import Document, Section, WikidataFact


def _document() -> Document:
    return Document(
        document_id="Q1:wikipedia:en:10:20",
        article_id="Q1:en:10:20",
        wikidata="Q1",
        project="wikipedia",
        language="en",
        site="enwiki",
        title="Title",
        url="https://example.test",
        page_id=10,
        revision_id=20,
        revision_timestamp="2026-01-01T00:00:00Z",
        retrieved_at="2026-01-01T00:00:00Z",
        full_text="Text",
        full_text_format="plain",
        article_length_chars=4,
        article_length_words=1,
        article_length_tokens_estimate=1,
        license="CC BY-SA 4.0",
        attribution="Wikipedia contributors",
        source_api="https://example.test/w/api.php",
        fetch_status="ok",
        fetch_error="",
        content_hash="abc",
    )


def _section() -> Section:
    document = _document()
    return Section(
        section_id="section-1",
        document_id=document.document_id,
        article_id=document.article_id,
        wikidata=document.wikidata,
        project=document.project,
        language=document.language,
        site=document.site,
        page_id=document.page_id,
        revision_id=document.revision_id,
        section_index=0,
        heading="Lead",
        anchor="",
        level=0,
        parent_section_id="",
        section_path="Lead",
        text="Text",
        text_length_chars=4,
        text_length_words=1,
        text_length_tokens_estimate=1,
        content_hash="def",
        license=document.license,
        attribution=document.attribution,
    )


def _voyage_document() -> Document:
    values = _document().to_dict()
    values.update(
        document_id="Q1:wikivoyage:en:10:20",
        article_id="",
        project="wikivoyage",
        site="enwikivoyage",
    )
    return Document(**values)


def _fact() -> WikidataFact:
    return WikidataFact(
        fact_id="fact-1",
        wikidata="Q1",
        property_id="P31",
        property_label_en="instance of",
        property_labels="{}",
        value_type="wikibase-entityid",
        value_entity_id="Q2",
        value_label_en="value",
        value_labels="{}",
        value_text="",
        numeric_value=None,
        unit_entity_id="",
        rank="normal",
        qualifiers="[]",
        references="[]",
        retrieved_at="2026-01-01T00:00:00Z",
        source_api="https://example.test/w/api.php",
    )


def test_plan_key_is_deterministic_and_input_sensitive() -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import augmentation_plan_key

    kwargs = {
        "core_hashes": {"/data/polygons.parquet": "a" * 64},
        "qids": ("Q1",),
        "document_identities": (("Q1:wikipedia:en:10:20", 20, "abc"),),
    }

    first = augmentation_plan_key(**kwargs)

    assert first == augmentation_plan_key(**kwargs)
    assert len(first) == 64
    assert first != augmentation_plan_key(
        **{**kwargs, "qids": ("Q1", "Q2")},
    )


def test_section_metadata_parser_normalizes_valid_documents_and_rejects_bad_values() -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        _section_batch_expected_documents,
    )

    assert _section_batch_expected_documents(
        {"documents": [["doc", "20", "hash"], ["other", 21, "hash2"]]}
    ) == (("doc", 20, "hash"), ("other", 21, "hash2"))
    assert _section_batch_expected_documents({"documents": object()}) is None
    assert _section_batch_expected_documents(None) is None


def test_section_batch_identity_helpers_validate_known_unique_sections() -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        _section_ids_are_known,
        _section_ids_are_unique,
    )

    section = _section()
    expected = ((section.document_id, section.revision_id, "abc"),)

    assert _section_ids_are_known([section], expected)
    assert _section_ids_are_unique([section])
    assert not _section_ids_are_known([section], (("other", 1, "hash"),))
    assert not _section_ids_are_unique([section, section])


def test_checkpoint_payload_helpers_reject_malformed_rows(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        _read_json_object,
        _validated_entity_payload,
        _validated_voyage_documents,
    )

    payload = tmp_path / "payload.json"
    payload.write_text('{"Q1": {"id": "Q1"}}', encoding="utf-8")
    assert _read_json_object(payload) == {"Q1": {"id": "Q1"}}
    payload.write_text("not-json", encoding="utf-8")
    assert _read_json_object(payload) is None

    assert _validated_entity_payload({"Q1": {"id": "Q1"}}, ("Q1",)) == {"Q1": {"id": "Q1"}}
    assert _validated_entity_payload({"Q2": {}}, ("Q1",)) is None
    assert _validated_entity_payload({"Q1": []}, ("Q1",)) is None

    voyage = _voyage_document()
    assert _validated_voyage_documents([voyage.to_dict()]) == [voyage]
    assert _validated_voyage_documents([_document().to_dict()]) is None
    assert _validated_voyage_documents([voyage.to_dict(), voyage.to_dict()]) is None


def test_checkpoint_constructor_validation_helpers_preserve_contract() -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        _validate_augmentation_stem,
        _validate_plan_key,
    )

    assert _validate_augmentation_stem("england-latest") == "england-latest"
    assert _validate_plan_key("a" * 64) == "a" * 64
    with pytest.raises(ValueError, match="stem"):
        _validate_augmentation_stem("../escape")
    with pytest.raises(ValueError, match="plan key"):
        _validate_plan_key("not-a-plan-key")


@pytest.mark.parametrize("stem", ["", ".", "..", "../escape", "nested/stem", r"nested\stem"])
def test_checkpoint_store_rejects_unsafe_stem(tmp_path: Path, stem: str) -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        AugmentationCheckpointStore,
    )

    with pytest.raises(ValueError, match="stem"):
        AugmentationCheckpointStore(tmp_path / "cache", stem, "a" * 64)


def test_checkpoint_paths_stay_under_supplied_data_cache(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        AugmentationCheckpointStore,
    )

    cache_root = tmp_path / "seagate-data-root" / "cache" / "augmentation_checkpoints"
    store = AugmentationCheckpointStore(cache_root, "england-latest", "a" * 64)

    assert store.plan_root.resolve().is_relative_to(cache_root.resolve())
    assert not store.plan_root.resolve().is_relative_to(Path.cwd().resolve())


def test_entities_round_trip_and_mismatched_qids_are_not_reused(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        AugmentationCheckpointStore,
    )

    store = AugmentationCheckpointStore(tmp_path, "england-latest", "a" * 64)
    entities = {"Q1": {"id": "Q1", "sitelinks": {}, "claims": {}}}

    store.save_entities(("Q1",), entities)

    assert store.load_entities(("Q1",)) == entities
    assert store.load_entities(("Q2",)) is None


def test_entities_checkpoint_allows_authoritatively_missing_qids(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        AugmentationCheckpointStore,
    )

    store = AugmentationCheckpointStore(tmp_path, "england-latest", "a" * 64)
    entities = {"Q1": {"id": "Q1", "sitelinks": {}, "claims": {}}}

    store.save_entities(("Q1", "Q2"), entities)

    assert store.load_entities(("Q1", "Q2")) == entities


def test_documents_sections_and_facts_round_trip_with_exact_inputs(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        AugmentationCheckpointStore,
        document_identities,
        entities_digest,
    )

    store = AugmentationCheckpointStore(tmp_path, "england-latest", "a" * 64)
    document = _document()
    digest = entities_digest({"Q1": {"id": "Q1"}})
    identities = document_identities([document])

    voyage_document = _voyage_document()
    store.save_voyage_documents(digest, [voyage_document])
    store.save_section_batch(0, identities, [_section()])
    store.save_facts(digest, [_fact()])

    assert store.load_voyage_documents(digest) == [voyage_document]
    assert store.load_voyage_documents("different") is None
    assert store.load_section_batch(0, identities) == [_section()]
    assert store.load_section_batch(0, ()) is None
    assert store.load_facts(digest) == [_fact()]
    assert store.load_facts("different") is None


def test_incomplete_or_corrupt_checkpoint_is_not_reused(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        AugmentationCheckpointStore,
    )

    store = AugmentationCheckpointStore(tmp_path, "england-latest", "a" * 64)
    store.save_entities(("Q1",), {"Q1": {"id": "Q1"}})
    (store.plan_root / "entities" / "metadata.json").write_text("{broken")

    assert store.load_entities(("Q1",)) is None


def test_malformed_section_metadata_is_not_reused(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        AugmentationCheckpointStore,
        document_identities,
    )

    store = AugmentationCheckpointStore(tmp_path, "england-latest", "a" * 64)
    identities = document_identities([_document()])
    store.save_section_batch(0, identities, [_section()])
    metadata = store.plan_root / "sections" / "batch-000000" / "metadata.json"
    metadata.write_text('{"contract_version":"augmentation-checkpoints-v1","documents":[1]}\n')

    assert store.load_section_batch(0, identities) is None


def test_clear_removes_only_one_region(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.augmentation.checkpoints import (
        AugmentationCheckpointStore,
    )

    first = AugmentationCheckpointStore(tmp_path, "england-latest", "a" * 64)
    second = AugmentationCheckpointStore(tmp_path, "scotland-latest", "b" * 64)
    first.save_entities(("Q1",), {"Q1": {"id": "Q1"}})
    second.save_entities(("Q2",), {"Q2": {"id": "Q2"}})

    first.clear()

    assert not first.region_root.exists()
    assert second.load_entities(("Q2",)) == {"Q2": {"id": "Q2"}}
