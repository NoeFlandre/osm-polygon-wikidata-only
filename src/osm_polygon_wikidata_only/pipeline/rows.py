"""Pure construction of enriched polygon, article, and link rows."""

from __future__ import annotations

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.ids import article_id, content_hash
from osm_polygon_wikidata_only.domain.models import Article, Polygon, PolygonArticleLink
from osm_polygon_wikidata_only.enrichment.article_linker import LinkSummary, fetch_qids
from osm_polygon_wikidata_only.enrichment.text_cleaning import count_words, estimate_tokens
from osm_polygon_wikidata_only.enrichment.wikidata_client import WikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia_client import WikipediaArticle, WikipediaClient
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps


def enrich_polygon(
    polygon: Polygon,
    *,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    settings: Settings,
    summaries: dict[str, LinkSummary],
) -> Polygon:
    """Return a polygon with coverage fields derived from its complete summary."""
    summary = summaries.get(polygon.wikidata)
    if summary is None:
        summary = fetch_qids(
            [polygon.wikidata],
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            languages=settings.languages,
            fetch_full_text=settings.fetch_full_text,
            max_articles_per_qid=settings.max_articles_per_qid,
            batch_size=settings.enrichment_batch_size,
            site_workers=settings.enrichment_site_workers,
        )[0]
        summaries[polygon.wikidata] = summary
    languages = sorted({article.language for article in summary.articles})
    return Polygon(
        polygon_id=polygon.polygon_id,
        region=polygon.region,
        source_pbf=polygon.source_pbf,
        osm_type=polygon.osm_type,
        osm_id=polygon.osm_id,
        wikidata=polygon.wikidata,
        name=polygon.name,
        tags=polygon.tags,
        tag_keys=polygon.tag_keys,
        tag_count=polygon.tag_count,
        osm_primary_tag=polygon.osm_primary_tag,
        centroid=polygon.centroid,
        lat=polygon.lat,
        lon=polygon.lon,
        bbox=polygon.bbox,
        geometry=polygon.geometry,
        area_m2=polygon.area_m2,
        area_km2=polygon.area_km2,
        area_bucket=polygon.area_bucket,
        has_name=polygon.has_name,
        has_wikidata=polygon.has_wikidata,
        has_wikipedia=bool(summary.articles),
        wikipedia_language_count=len(languages),
        wikipedia_languages=json_dumps(languages),
        wikipedia_article_count=len(summary.articles),
        has_english_wikipedia="en" in languages,
        has_french_wikipedia="fr" in languages,
        text_available=any(article.full_text for article in summary.articles),
        best_language=summary.best_language(),
        extraction_version=polygon.extraction_version,
        extracted_at=polygon.extracted_at,
    )


def build_articles_and_links(
    polygons: list[Polygon], summaries: dict[str, LinkSummary]
) -> tuple[list[Article], list[PolygonArticleLink]]:
    """Build deterministic deduplicated articles and polygon-article links."""
    articles_by_id: dict[str, Article] = {}
    links: list[PolygonArticleLink] = []
    for polygon in polygons:
        summary = summaries.get(polygon.wikidata)
        if summary is None or not summary.articles:
            continue
        best = summary.best_language()
        for article in summary.articles:
            identifier = article_id(
                polygon.wikidata, article.language, article.page_id, article.revision_id
            )
            if identifier not in articles_by_id:
                articles_by_id[identifier] = article_row(
                    identifier, polygon.wikidata, article, summary
                )
            links.append(
                PolygonArticleLink(
                    polygon_id=polygon.polygon_id,
                    article_id=identifier,
                    wikidata=polygon.wikidata,
                    language=article.language,
                    source_pbf=polygon.source_pbf,
                    region=polygon.region,
                    osm_type=polygon.osm_type,
                    osm_id=polygon.osm_id,
                    page_id=article.page_id,
                    revision_id=article.revision_id,
                    is_best_language=article.language == best,
                )
            )
    return list(articles_by_id.values()), links


def article_row(
    identifier: str, qid: str, article: WikipediaArticle, summary: LinkSummary
) -> Article:
    """Build immutable derived metadata for one unique article revision."""
    entity = summary.entity
    label = entity.labels.get(article.language) if entity else ""
    description = entity.descriptions.get(article.language) if entity else ""
    aliases = entity.aliases.get(article.language) if entity else None
    if entity is not None:
        label = label or entity.labels.get("en", "")
        description = description or entity.descriptions.get("en", "")
        aliases = aliases or entity.aliases.get("en", [])
    return Article(
        article_id=identifier,
        wikidata=qid,
        language=article.language,
        site=article.site,
        title=article.title,
        url=article.url,
        page_id=article.page_id,
        revision_id=article.revision_id,
        revision_timestamp=article.revision_timestamp,
        retrieved_at=article.retrieved_at,
        wikidata_label=str(label or ""),
        wikidata_description=str(description or ""),
        wikidata_aliases=json_dumps(aliases or []),
        lead_text=article.lead_text,
        extract=article.extract,
        full_text=article.full_text,
        full_text_format=article.full_text_format,
        article_length_chars=len(article.full_text),
        article_length_words=count_words(article.full_text),
        article_length_tokens_estimate=estimate_tokens(article.full_text),
        thumbnail_url=article.thumbnail_url,
        thumbnail_width=article.thumbnail_width,
        thumbnail_height=article.thumbnail_height,
        categories=json_dumps(article.categories),
        license=article.license,
        attribution=article.attribution,
        source_api=article.source_api,
        fetch_status="ok",
        fetch_error="",
        content_hash=content_hash(article.full_text),
    )


__all__ = ["article_row", "build_articles_and_links", "enrich_polygon"]
