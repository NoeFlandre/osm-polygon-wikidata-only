"""Phase 2.5 / Defect 9: Canonical planning must use the canonical schema.

``_classify_stem`` must construct the planned canonical table with
``polygon_document_link_schema()`` explicitly so the planning digest,
schema and column metadata all match what the apply stage writes.
The on-disk canonical Parquet schema must match the canonical schema
exactly (order, types, nullability, metadata).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.domain.polygon_document_links import (
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.pipeline import link_migration as lm
from osm_polygon_wikidata_only.pipeline._link_migration import planning
from tests.migration._builders import (
    write_legacy_polygon_article,
    write_polygon_qid_membership,
    write_wikipedia_document,
)


def _setup_minimal_processed(processed: Path, stem: str) -> None:
    """Set up the minimal artifacts needed to classify a stem."""
    for sub in (
        "polygons",
        "polygon_articles",
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
        "manifests",
    ):
        (processed / sub).mkdir(parents=True, exist_ok=True)

    write_polygon_qid_membership(
        processed / "polygons" / f"{stem}.parquet",
        "p1",
        ["Q1"],
        source_pbf=f"{stem}.osm.pbf",
    )
    write_legacy_polygon_article(
        processed / "polygon_articles" / f"{stem}.parquet",
        "p1",
        "Q1",
        source_pbf=f"{stem}.osm.pbf",
    )
    write_wikipedia_document(processed / "wikipedia" / "documents" / f"{stem}.parquet", "Q1")

    (processed / "manifests" / "processed_pbfs.json").write_text(
        f'{{"{stem}.osm.pbf": {{"source_pbf": "{stem}.osm.pbf"}}}}'
    )
    from osm_polygon_wikidata_only.augmentation.schema import (
        document_schema,
        section_schema,
    )

    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikipedia" / "sections" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikivoyage" / "sections" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=document_schema()),
        processed / "wikivoyage" / "documents" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"_placeholder": []}),
        processed / "wikidata" / "facts" / f"{stem}.parquet",
    )


def test_classify_stem_planning_uses_canonical_schema(tmp_path: Path) -> None:
    """The MIGRATABLE branch of ``_classify_stem`` must construct the
    planned canonical table with the canonical schema explicitly.
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_minimal_processed(processed, stem)

    sp = planning.classify_stem(stem, processed)
    assert sp.classification == lm.StemClassification.MIGRATABLE, (
        f"Stem should be MIGRATABLE; got {sp}"
    )

    # The planning digest is computed from the canonical_table. We
    # verify the staging of the canonical table inside the apply path
    # yields the exact canonical schema.
    plan = lm.plan_link_migration(processed)
    assert plan.is_safe_to_apply

    # Re-run the apply. After apply the on-disk schema must equal the
    # canonical schema exactly.
    lm.apply_link_migration(processed)
    on_disk = pq.read_table(  # type: ignore[no-untyped-call]
        processed / "polygon_articles" / f"{stem}.parquet"
    )
    assert on_disk.schema.equals(polygon_document_link_schema(), check_metadata=True), (
        f"On-disk schema must equal the canonical schema; got {on_disk.schema}"
    )


def test_classify_stem_canonical_branch_strict_schema_check(tmp_path: Path) -> None:
    """When the link table is already canonical, ``_classify_stem``
    must still verify the schema strictly (types, metadata) -- a
    lookalike schema must be BLOCKED.
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_minimal_processed(processed, stem)

    # First migrate to produce a real canonical link file.
    lm.apply_link_migration(processed)

    # Now corrupt the metadata to make the on-disk schema differ
    # only in field metadata.
    canonical_with_metadata = polygon_document_link_schema()
    # Add bogus metadata to the first field to simulate a lookalike.
    field0 = canonical_with_metadata.field(0).with_metadata({"custom": "bogus"})
    lookalike_schema = pa.schema(
        [field0, *canonical_with_schema_iter(canonical_with_metadata)],
    )

    # Write a lookalike canonical file with bogus metadata.
    on_disk = pq.read_table(  # type: ignore[no-untyped-call]
        processed / "polygon_articles" / f"{stem}.parquet"
    )
    lookalike = pa.Table.from_arrays(on_disk.columns, schema=lookalike_schema)
    pq.write_table(  # type: ignore[no-untyped-call]
        lookalike, processed / "polygon_articles" / f"{stem}.parquet"
    )

    sp = planning.classify_stem(stem, processed)
    assert sp.classification == lm.StemClassification.BLOCKED, (
        f"Lookalike schema must be BLOCKED; got {sp}"
    )


def canonical_with_schema_iter(schema: pa.Schema):
    """Yield fields 1..N of a schema (skipping the first which the
    test already mutated)."""
    for i in range(1, len(schema)):
        yield schema.field(i)
