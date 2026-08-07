import json
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.models import Section
from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import polygon_document_link_schema
from osm_polygon_wikidata_only.domain.schema import empty_row, polygon_schema
from osm_polygon_wikidata_only.enrichment.wikipedia.models import FetchResult
from osm_polygon_wikidata_only.enrichment.wikipedia.transport import InMemoryWikipediaClient
from osm_polygon_wikidata_only.v2.extractor import V2ExtractedPbf, V2PbfStem, candidate_to_v2_row
from osm_polygon_wikidata_only.v2.reuse import merge_v2_region, reconcile_v2_region
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest
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
    existing_section = Section(
        "existing-section",
        document["document_id"],
        document["article_id"],
        document["wikidata"],
        "wikipedia",
        "en",
        "enwiki",
        1,
        2,
        0,
        "",
        "",
        0,
        "",
        "[]",
        "Existing section",
        16,
        2,
        3,
        "section-hash",
        "CC BY-SA",
        "",
    ).to_dict()
    _write(
        root.processed / "wikipedia/sections" / f"{stem}.parquet",
        section_schema(),
        [existing_section],
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

    class SectionClient:
        def parse_html(self, project: str, language: str, revision_id: int) -> str:
            assert project == "wikipedia"
            assert language == "en"
            assert revision_id == 20
            return "<p>Direct section text</p><h2>Heading</h2><p>More text</p>"

    client = CountingClient({("enwiki", "New page"): FetchResult("ok", _article("New page"))})
    merge_v2_region(
        root,
        extracted,
        index=build_v1_reuse_index(root.processed),
        wikipedia_client=client,
        section_client=SectionClient(),
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
    sections = pq.read_table(root.processed_v2 / "wikipedia/sections" / f"{stem}.parquet")
    assert tuple(sections.schema.names) == (
        "section_id",
        "document_id",
        "article_id",
        "wikidata",
        "project",
        "language",
        "site",
        "page_id",
        "revision_id",
        "section_index",
        "heading",
        "anchor",
        "level",
        "parent_section_id",
        "section_path",
        "text",
        "text_length_chars",
        "text_length_words",
        "text_length_tokens_estimate",
        "content_hash",
        "license",
        "attribution",
    )
    assert sections.num_rows > 0
    assert {row["document_id"] for row in sections.to_pylist()} == {
        document["document_id"],
        direct_document["document_id"],
    }
    polygon_rows = pq.read_table(root.processed_v2 / "polygons" / f"{stem}.parquet").to_pylist()
    voyage_row = next(
        row for row in polygon_rows if row["polygon_id"] == voyage_polygon["polygon_id"]
    )
    assert voyage_row["has_wikipedia"] is False


def test_merge_uses_current_pbf_wikidata_and_drops_stale_v1_links(tmp_path: Path) -> None:
    """A changed OSM tag must replace stale V1 identity data, not abort V2."""
    root = DataRoot(tmp_path)
    root.ensure()
    stem = "region-latest"
    polygon_id = f"{stem}:way:130311194"

    old_polygon = empty_row(tuple(field.name for field in polygon_schema()))
    old_polygon.update(
        {
            "polygon_id": polygon_id,
            "region": "region",
            "source_pbf": f"{stem}.osm.pbf",
            "osm_type": "way",
            "osm_id": 130311194,
            "wikidata": "Q42",
            "tags": "{}",
        }
    )
    _write(root.processed_polygons / f"{stem}.parquet", polygon_schema(), [old_polygon])

    stale_link = empty_row(tuple(field.name for field in polygon_document_link_schema()))
    stale_link.update(
        {
            "polygon_id": polygon_id,
            "document_id": "Q42:wikipedia:en:1:2",
            "project": "wikipedia",
            "wikidata": "Q42",
            "language": "en",
            "source_pbf": f"{stem}.osm.pbf",
            "region": "region",
            "osm_type": "way",
            "osm_id": 130311194,
            "page_id": 1,
            "revision_id": 2,
        }
    )
    _write(root.processed_links / f"{stem}.parquet", polygon_document_link_schema(), [stale_link])

    current = candidate_to_v2_row(
        (
            "way",
            130311194,
            {"wikidata": "Q43", "name": "Current name"},
            '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}',
        ),
        source_pbf_stem=stem,
        region="region",
        source_pbf=f"{stem}.osm.pbf",
        extracted_at="2026-01-01T00:00:00Z",
    )
    assert current is not None
    index = build_v1_reuse_index(root.processed)
    try:
        merge_v2_region(
            root,
            V2ExtractedPbf(
                V2PbfStem(tmp_path / f"{stem}.osm.pbf", stem, "region"),
                (current,),
                0.0,
            ),
            index=index,
            wikipedia_client=None,
        )
    finally:
        index.close()

    polygons = pq.read_table(root.processed_v2 / "polygons" / f"{stem}.parquet").to_pylist()
    links = pq.read_table(
        root.processed_v2 / "polygon_document_links" / f"{stem}.parquet"
    ).to_pylist()
    assert polygons[0]["wikidata"] == "Q43"
    assert links == []


def test_merge_fetches_direct_wikipedia_pages_concurrently_and_deterministically(
    tmp_path: Path,
) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    stem = "region-latest"
    first = candidate_to_v2_row(
        (
            "way",
            1,
            {"wikipedia": "en:First page"},
            '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}',
        ),
        source_pbf_stem=stem,
        region="region",
        source_pbf=f"{stem}.osm.pbf",
        extracted_at="2026-01-01T00:00:00Z",
    )
    second = candidate_to_v2_row(
        (
            "way",
            2,
            {"wikipedia": "en:Second page"},
            '{"type":"Polygon","coordinates":[[[2,0],[3,0],[3,1],[2,0]]]}',
        ),
        source_pbf_stem=stem,
        region="region",
        source_pbf=f"{stem}.osm.pbf",
        extracted_at="2026-01-01T00:00:00Z",
    )
    assert first is not None and second is not None
    extracted = V2ExtractedPbf(
        V2PbfStem(tmp_path / f"{stem}.osm.pbf", stem, "region"),
        (first, second),
        0.0,
    )

    class ConcurrentClient:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._barrier = threading.Barrier(2)
            self.active = 0
            self.max_active = 0

        def fetch_article(self, _language: str, _site: str, title: str, **_kwargs: object):
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                self._barrier.wait(timeout=2)
                page_id = 1 if title == "First page" else 2
                return FetchResult("ok", _article(title, page_id=page_id))
            finally:
                with self._lock:
                    self.active -= 1

    class SectionClient:
        def parse_html(self, _project: str, _language: str, _revision_id: int) -> str:
            return "<p>text</p>"

    client = ConcurrentClient()
    merge_v2_region(
        root,
        extracted,
        index=build_v1_reuse_index(root.processed),
        wikipedia_client=client,
        section_client=SectionClient(),
        direct_workers=2,
    )

    assert client.max_active == 2
    documents = pq.read_table(
        root.processed_v2 / "wikipedia/documents" / f"{stem}.parquet"
    ).to_pylist()
    assert [row["title"] for row in documents] == ["First page", "Second page"]


def test_merge_batches_v1_title_lookups_across_region(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    stem = "region-latest"
    extracted_rows = []
    for position in range(3):
        row = candidate_to_v2_row(
            (
                "way",
                position + 1,
                {"wikipedia": f"en:Page {position}"},
                '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}',
            ),
            source_pbf_stem=stem,
            region="region",
            source_pbf=f"{stem}.osm.pbf",
            extracted_at="2026-01-01T00:00:00Z",
        )
        assert row is not None
        extracted_rows.append(row)
    extracted = V2ExtractedPbf(
        V2PbfStem(tmp_path / f"{stem}.osm.pbf", stem, "region"),
        tuple(extracted_rows),
        0.0,
    )

    class CountingIndex:
        is_ready = True

        def __init__(self) -> None:
            self.calls = 0

        def by_titles(self, keys: tuple[tuple[str, str], ...]):
            self.calls += 1
            return {key: () for key in keys}

    class WikipediaClient:
        def fetch_article(self, _language: str, _site: str, title: str, **_kwargs: object):
            return FetchResult("ok", _article(title))

    class SectionClient:
        def parse_html(self, _project: str, _language: str, _revision_id: int) -> str:
            return "<p>text</p>"

    index = CountingIndex()
    merge_v2_region(
        root,
        extracted,
        index=index,  # type: ignore[arg-type]
        wikipedia_client=WikipediaClient(),  # type: ignore[arg-type]
        section_client=SectionClient(),
        direct_workers=1,
        wait_for_index=False,
    )

    assert index.calls == 2


def test_merge_rechecks_only_titles_missing_from_initial_v1_lookup(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    stem = "region-latest"
    extracted_rows = []
    for position, title in enumerate(("Existing", "Missing"), start=1):
        row = candidate_to_v2_row(
            (
                "way",
                position,
                {"wikipedia": f"en:{title}"},
                '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}',
            ),
            source_pbf_stem=stem,
            region="region",
            source_pbf=f"{stem}.osm.pbf",
            extracted_at="2026-01-01T00:00:00Z",
        )
        assert row is not None
        extracted_rows.append(row)
    extracted = V2ExtractedPbf(
        V2PbfStem(tmp_path / f"{stem}.osm.pbf", stem, "region"),
        tuple(extracted_rows),
        0.0,
    )

    existing = _v1_row("Existing")

    class CountingIndex:
        is_ready = True

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, str], ...]] = []

        def by_titles(self, keys: tuple[tuple[str, str], ...]):
            self.calls.append(keys)
            matches = {
                ("en", "existing"): (existing,),
            }
            return {
                (key[0].casefold(), key[1].replace("_", " ").casefold()): matches.get(
                    (key[0].casefold(), key[1].casefold()), ()
                )
                for key in keys
            }

    class WikipediaClient:
        def fetch_article(self, _language: str, _site: str, title: str, **_kwargs: object):
            return FetchResult("ok", _article(title, page_id=2))

    class SectionClient:
        def parse_html(self, _project: str, _language: str, _revision_id: int) -> str:
            return "<p>text</p>"

    index = CountingIndex()
    merge_v2_region(
        root,
        extracted,
        index=index,  # type: ignore[arg-type]
        wikipedia_client=WikipediaClient(),  # type: ignore[arg-type]
        section_client=SectionClient(),
        direct_workers=1,
        wait_for_index=False,
    )

    assert index.calls == [
        (("en", "Existing"), ("en", "Missing")),
        (("en", "Missing"),),
    ]


def test_merge_fetches_sections_before_waiting_for_final_index(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    stem = "region-latest"
    direct = candidate_to_v2_row(
        (
            "way",
            1,
            {"wikipedia": "en:Speculative page"},
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
    sections_started = threading.Event()

    class InFlightIndex:
        is_ready = False

        def by_title(self, _language: str, _title: str) -> tuple[dict[str, object], ...]:
            return ()

        def wait_until_ready(self) -> None:
            raise AssertionError("speculative merge must not wait for the index")

    class WikipediaClient:
        def fetch_article(self, *_args: object, **_kwargs: object) -> FetchResult:
            return FetchResult("ok", _article("Speculative page"))

    class SectionClient:
        def parse_html(self, _project: str, _language: str, _revision_id: int) -> str:
            sections_started.set()
            return "<p>speculative section</p>"

    merge_v2_region(
        root,
        extracted,
        index=InFlightIndex(),  # type: ignore[arg-type]
        wikipedia_client=WikipediaClient(),  # type: ignore[arg-type]
        section_client=SectionClient(),
        wait_for_index=False,
    )
    assert sections_started.is_set()
    documents = pq.read_table(
        root.processed_v2 / "wikipedia/documents" / f"{stem}.parquet"
    ).to_pylist()
    assert len(documents) == 1
    assert documents[0]["title"] == "Speculative page"
    assert load_v2_manifest(root.processed_v2)[stem]["v1_index_reconciled"] is False


def test_reconcile_v2_region_discards_speculative_duplicate_after_index_scan(
    tmp_path: Path,
) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    stem = "region-latest"
    _write(
        root.processed / "wikipedia/documents" / f"{stem}.parquet",
        wikipedia_document_schema(),
        [_v1_row(title="Speculative page")],
    )
    direct = candidate_to_v2_row(
        (
            "way",
            1,
            {"wikipedia": "en:Speculative page"},
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

    class PartialIndex:
        is_ready = False

        def by_title(self, _language: str, _title: str) -> tuple[dict[str, object], ...]:
            return ()

        def wait_until_ready(self) -> None:
            raise AssertionError("provisional merge must not wait")

    class WikipediaClient:
        def fetch_article(self, *_args: object, **_kwargs: object) -> FetchResult:
            return FetchResult("ok", _article("Speculative page", page_id=10))

    class SectionClient:
        def parse_html(self, _project: str, _language: str, _revision_id: int) -> str:
            return "<p>speculative section</p>"

    merge_v2_region(
        root,
        extracted,
        index=PartialIndex(),  # type: ignore[arg-type]
        wikipedia_client=WikipediaClient(),  # type: ignore[arg-type]
        section_client=SectionClient(),
        wait_for_index=False,
    )
    reconcile_v2_region(
        root,
        stem,
        index=build_v1_reuse_index(root.processed),
        section_client=SectionClient(),
    )
    documents = pq.read_table(
        root.processed_v2 / "wikipedia/documents" / f"{stem}.parquet"
    ).to_pylist()
    assert [row["document_id"] for row in documents] == ["Q42:wikipedia:en:1:2"]
    assert load_v2_manifest(root.processed_v2)[stem]["v1_index_reconciled"] is True
