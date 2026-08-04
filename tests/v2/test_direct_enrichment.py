from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.enrichment.wikipedia.models import FetchResult, WikipediaArticle
from osm_polygon_wikidata_only.enrichment.wikipedia.transport import InMemoryWikipediaClient
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.v2.config import V2_CACHE_CONTRACT_VERSION
from osm_polygon_wikidata_only.v2.direct_enrichment import enrich_wikipedia_refs
from osm_polygon_wikidata_only.v2.v1_index import build_v1_reuse_index
from osm_polygon_wikidata_only.v2.wikipedia_tags import WikipediaTagRef


def _article(title: str, page_id: int = 10) -> WikipediaArticle:
    return WikipediaArticle(
        language="en",
        site="enwiki",
        title=title,
        page_id=page_id,
        revision_id=20,
        revision_timestamp="2020-01-01T00:00:00Z",
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        lead_text="lead",
        extract="extract",
        full_text="full text",
        full_text_format="plain_text",
        thumbnail_url="",
        thumbnail_width=None,
        thumbnail_height=None,
        categories=[],
        license="CC BY-SA",
        attribution="",
        source_api="mediawiki_action_api",
        retrieved_at="2020-01-01T00:00:00Z",
    )


def _v1_row(title: str = "Existing", page_id: int = 1) -> dict[str, object]:
    row: dict[str, object] = {
        "document_id": "Q42:wikipedia:en:1:2",
        "article_id": "Q42:en:1:2",
        "wikidata": "Q42",
        "project": "wikipedia",
        "language": "en",
        "site": "enwiki",
        "title": title,
        "url": "https://en.wikipedia.org/wiki/Existing",
        "page_id": page_id,
        "revision_id": 2,
        "revision_timestamp": "2020-01-01T00:00:00Z",
        "retrieved_at": "2020-01-01T00:00:00Z",
        "wikidata_label": "",
        "wikidata_description": "",
        "wikidata_aliases": "[]",
        "lead_text": "",
        "extract": "",
        "full_text": "existing",
        "full_text_format": "plain_text",
        "article_length_chars": 8,
        "article_length_words": 1,
        "article_length_tokens_estimate": 2,
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": "[]",
        "license": "CC BY-SA",
        "attribution": "",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": "hash",
    }
    row["title"] = title
    row["page_id"] = page_id
    return row


def _write_v1(root: Path) -> None:
    path = root / "wikipedia" / "documents" / "region.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_v1_row()], schema=wikipedia_document_schema()), path)


def test_matching_v1_page_is_reused_without_fetch(tmp_path: Path) -> None:
    _write_v1(tmp_path)
    index = build_v1_reuse_index(tmp_path)
    result = enrich_wikipedia_refs(
        "polygon-1",
        (WikipediaTagRef("en", "Existing", "wikipedia", "en:Existing"),),
        index=index,
        wikipedia_client=InMemoryWikipediaClient({}),
    )
    assert len(result.documents) == 1
    assert result.documents[0]["document_id"] == "Q42:wikipedia:en:1:2"
    assert result.links[0]["link_sources"] == '["osm_wikipedia_tag"]'


def test_missing_page_is_fetched_and_retained_without_qid() -> None:
    article = _article("New page")
    client = InMemoryWikipediaClient({("enwiki", "New page"): FetchResult("ok", article)})
    result = enrich_wikipedia_refs(
        "polygon-1",
        (WikipediaTagRef("en", "New page", "wikipedia", "en:New page"),),
        index=build_v1_reuse_index(Path("/does-not-exist")),
        wikipedia_client=client,
    )
    assert result.documents[0]["wikidata"] is None
    assert result.documents[0]["project"] == "wikipedia"
    assert result.links[0]["link_sources"] == '["osm_wikipedia_tag"]'


def test_not_found_is_recorded_without_crashing() -> None:
    result = enrich_wikipedia_refs(
        "polygon-1",
        (WikipediaTagRef("en", "Missing", "wikipedia", "en:Missing"),),
        index=build_v1_reuse_index(Path("/does-not-exist")),
        wikipedia_client=InMemoryWikipediaClient({}),
    )
    assert result.documents == ()
    assert result.statuses[0].status == "article_not_found"


def test_successful_direct_fetch_is_cached_for_the_next_run(tmp_path: Path) -> None:
    class CountingClient(InMemoryWikipediaClient):
        calls = 0

        def fetch_article(self, *args: object, **kwargs: object) -> FetchResult:
            self.calls += 1
            return super().fetch_article(*args, **kwargs)  # type: ignore[arg-type]

    client = CountingClient({("enwiki", "New page"): FetchResult("ok", _article("New page"))})
    ref = WikipediaTagRef("en", "New page", "wikipedia", "en:New page")
    cache = JsonFileCache(tmp_path / "cache", contract_version=V2_CACHE_CONTRACT_VERSION)
    index = build_v1_reuse_index(tmp_path / "v1")
    enrich_wikipedia_refs("polygon-1", (ref,), index=index, wikipedia_client=client, cache=cache)
    enrich_wikipedia_refs("polygon-1", (ref,), index=index, wikipedia_client=client, cache=cache)
    assert client.calls == 1


def test_direct_enrichment_waits_for_background_index_before_fetching_missing_title() -> None:
    class BackgroundIndex:
        is_ready = False

        def __init__(self) -> None:
            self.waited = False

        def by_title(self, language: str, title: str) -> tuple[dict[str, object], ...]:
            if self.waited:
                return (
                    {
                        "document_id": "Q42:wikipedia:en:1:2",
                        "wikidata": "Q42",
                        "language": language,
                        "title": title,
                        "page_id": 1,
                        "revision_id": 2,
                    },
                )
            return ()

        def wait_until_ready(self) -> None:
            self.waited = True
            self.is_ready = True

    class FailingClient:
        def fetch_article(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("a missing title was fetched before index completion")

    index = BackgroundIndex()
    result = enrich_wikipedia_refs(
        "region:relation:1",
        (WikipediaTagRef("en", "Douglas Adams", "wikipedia:en", "Douglas Adams"),),
        index=index,  # type: ignore[arg-type]
        wikipedia_client=FailingClient(),  # type: ignore[arg-type]
    )
    assert result.statuses[0].status == "reused_v1"
    assert result.statuses[0].reused_v1
