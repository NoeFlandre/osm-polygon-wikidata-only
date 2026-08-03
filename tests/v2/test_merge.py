import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import polygon_document_link_schema
from osm_polygon_wikidata_only.domain.schema import empty_row, polygon_schema
from osm_polygon_wikidata_only.enrichment.wikipedia.models import FetchResult
from osm_polygon_wikidata_only.enrichment.wikipedia.transport import InMemoryWikipediaClient
from osm_polygon_wikidata_only.v2.extractor import V2ExtractedPbf, V2PbfStem, candidate_to_v2_row
from osm_polygon_wikidata_only.v2.reuse import merge_v2_region
from osm_polygon_wikidata_only.v2.v1_index import build_v1_reuse_index
from tests.v2.test_direct_enrichment import _article, _v1_row


def _write(path: Path, schema: pa.Schema, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_merge_reuses_v1_and_fetches_only_missing_direct_pages(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    stem = "region-latest"
    existing_polygon_id = f"{stem}:way:1"
    polygons = empty_row(tuple(field.name for field in polygon_schema()))
    polygons.update(
        {
            "polygon_id": existing_polygon_id,
            "region": "region",
            "source_pbf": f"{stem}.osm.pbf",
            "osm_type": "way",
            "osm_id": 1,
            "wikidata": "Q42",
            "tags": json.dumps({"wikipedia": "en:Existing"}),
        }
    )
    voyage_polygon = dict(polygons)
    voyage_polygon.update(
        {
            "polygon_id": f"{stem}:way:3",
            "osm_id": 3,
            "wikidata": "Q99",
            "tags": json.dumps({}),
        }
    )
    _write(
        root.processed_polygons / f"{stem}.parquet",
        polygon_schema(),
        [polygons, voyage_polygon],
    )

    document = _v1_row()
    _write(
        root.processed / "wikipedia/documents" / f"{stem}.parquet",
        wikipedia_document_schema(),
        [document],
    )
    link = empty_row(tuple(field.name for field in polygon_document_link_schema()))
    link.update(
        {
            "polygon_id": existing_polygon_id,
            "document_id": document["document_id"],
            "project": "wikipedia",
            "wikidata": "Q42",
            "language": "en",
            "source_pbf": f"{stem}.osm.pbf",
            "region": "region",
            "osm_type": "way",
            "osm_id": 1,
            "page_id": 1,
            "revision_id": 2,
        }
    )
    _write(root.processed_links / f"{stem}.parquet", polygon_document_link_schema(), [link])
    voyage_link = dict(link)
    voyage_link.update(
        {
            "polygon_id": voyage_polygon["polygon_id"],
            "document_id": "Q99:wikivoyage:en:3:4",
            "project": "wikivoyage",
            "wikidata": "Q99",
            "osm_id": 3,
            "page_id": 3,
            "revision_id": 4,
        }
    )
    _write(
        root.processed_links / f"{stem}.parquet",
        polygon_document_link_schema(),
        [link, voyage_link],
    )

    direct = candidate_to_v2_row(
        (
            "way",
            2,
            {"wikipedia": "en:New page"},
            '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}',
        ),
        source_pbf_stem=stem,
        region="region",
        source_pbf=f"{stem}.osm.pbf",
        extracted_at="2026-01-01T00:00:00Z",
    )
    assert direct is not None
    extracted = V2ExtractedPbf(
        V2PbfStem(tmp_path / f"{stem}.osm.pbf", stem, "region"),
        (direct,),
        0.0,
    )

    class CountingClient(InMemoryWikipediaClient):
        calls = 0

        def fetch_article(self, *args: object, **kwargs: object) -> FetchResult:
            self.calls += 1
            return super().fetch_article(*args, **kwargs)  # type: ignore[arg-type]

    client = CountingClient({("enwiki", "New page"): FetchResult("ok", _article("New page"))})
    merge_v2_region(
        root,
        extracted,
        index=build_v1_reuse_index(root.processed),
        wikipedia_client=client,
    )

    assert client.calls == 1
    links = pq.read_table(
        root.processed_v2 / "polygon_document_links" / f"{stem}.parquet"
    ).to_pylist()
    documents = pq.read_table(
        root.processed_v2 / "wikipedia/documents" / f"{stem}.parquet"
    ).to_pylist()
    assert len(links) == 3
    assert len(documents) == 2
    direct_document = next(row for row in documents if row["wikidata"] is None)
    assert direct_document["title"] == "New page"
    assert any(row["document_id"] == direct_document["document_id"] for row in links)
    polygon_rows = pq.read_table(root.processed_v2 / "polygons" / f"{stem}.parquet").to_pylist()
    voyage_row = next(
        row for row in polygon_rows if row["polygon_id"] == voyage_polygon["polygon_id"]
    )
    assert voyage_row["has_wikipedia"] is False
