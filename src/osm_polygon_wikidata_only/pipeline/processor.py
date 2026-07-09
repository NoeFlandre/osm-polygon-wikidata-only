"""Single-PBF processor.

This is the per-PBF orchestrator that ties together the PBF reader,
the polygon extractor, the enrichment clients, the parquet writers,
and the manifest updater. The CLI calls :func:`process_pbf` once per
input file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only import __version__
from osm_polygon_wikidata_only.config.paths import (
    PROCESSED_ARTICLES,
    PROCESSED_LINKS,
    PROCESSED_POLYGONS,
    DataRoot,
)
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.ids import article_id
from osm_polygon_wikidata_only.domain.models import Article, Polygon, PolygonArticleLink
from osm_polygon_wikidata_only.enrichment.article_linker import LinkSummary, fetch_qids
from osm_polygon_wikidata_only.enrichment.text_cleaning import count_words, estimate_tokens
from osm_polygon_wikidata_only.enrichment.wikidata_client import WikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia_client import WikipediaClient
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.manifest import (
    manifest_path,
    upsert_entry,
)
from osm_polygon_wikidata_only.io.parquet import (
    write_articles,
    write_polygon_articles,
    write_polygons,
)
from osm_polygon_wikidata_only.io.pbf_reader import (  # noqa: F401  (kept for re-export)
    PBFReader,
    region_from_filename,
)
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .extractor import candidate_to_polygon, polygon_to_dict
from .stats import StreamingStats

LOGGER = logging.getLogger(__name__)


@dataclass
class PbfStem:
    """A parsed PBF filename.

    ``stem`` is the part of the filename before ``.osm.pbf``
    (e.g. ``monaco-latest``). ``region`` is the part before
    ``-latest.osm.pbf`` (e.g. ``monaco``). The remote parquet paths
    use ``stem`` so the layout is stable across reruns.
    """

    path: Path
    stem: str
    region: str

    @classmethod
    def from_path(cls, path: Path) -> PbfStem:
        name = path.name
        stem = name[: -len(".osm.pbf")] if name.endswith(".osm.pbf") else path.stem
        # Region is the stem without the trailing "-latest" (or the
        # whole stem if there's no "-latest" suffix).
        region = stem[: -len("-latest")] if stem.endswith("-latest") else stem
        return cls(path=path, stem=stem, region=region)


@dataclass
class ProcessResult:
    """What :func:`process_pbf` returns: row counts + manifest entry."""

    polygons_path: Path
    articles_path: Path
    polygon_articles_path: Path
    manifest_path: Path
    polygon_count: int
    article_count: int
    link_count: int
    manifest_entry: dict[str, Any]


def _local_path(processed_dir: Path, subdir: str, stem: str) -> Path:
    return processed_dir / subdir / f"{stem}.parquet"


def _remote_path(subdir: str, stem: str) -> str:
    return f"{subdir}/{stem}.parquet"


def _enrich_polygon(
    polygon: Polygon,
    *,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    settings: Settings,
    summaries: dict[str, LinkSummary],
) -> Polygon:
    """Apply Wikidata/Wikipedia enrichment to a polygon.

    Looks up the cached :class:`LinkSummary` for the polygon's
    Wikidata QID. If missing, fetches it now. The summary's articles
    are converted to per-language best-language decisions on the
    polygon row.
    """
    qid = polygon.wikidata
    summary = summaries.get(qid)
    if summary is None:
        summary_list = fetch_qids(
            [qid],
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            languages=settings.languages,
            fetch_full_text=settings.fetch_full_text,
            max_articles_per_qid=settings.max_articles_per_qid,
        )
        summary = summary_list[0]
        summaries[qid] = summary

    langs = sorted({a.language for a in summary.articles})
    best = summary.best_language()
    has_text = bool(summary.articles and any(a.full_text for a in summary.articles))
    has_en = "en" in langs
    has_fr = "fr" in langs

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
        wikipedia_language_count=len(langs),
        wikipedia_languages=json_dumps(langs),
        wikipedia_article_count=len(summary.articles),
        has_english_wikipedia=has_en,
        has_french_wikipedia=has_fr,
        text_available=has_text,
        best_language=best,
        extraction_version=polygon.extraction_version,
        extracted_at=polygon.extracted_at,
    )


def _build_articles_and_links(
    polygons: list[Polygon],
    summaries: dict[str, LinkSummary],
) -> tuple[list[Article], list[PolygonArticleLink]]:
    """Build per-PBF article rows and polygon-article links.

    Articles are deduplicated by ``article_id`` (one row per unique
    (wikidata, language, page_id, revision_id) tuple). Links are
    produced for every (polygon, article) pair.
    """
    articles_by_id: dict[str, Article] = {}
    links: list[PolygonArticleLink] = []

    for polygon in polygons:
        summary = summaries.get(polygon.wikidata)
        if summary is None or not summary.articles:
            continue
        best = summary.best_language()
        for art in summary.articles:
            aid = article_id(polygon.wikidata, art.language, art.page_id, art.revision_id)
            if aid not in articles_by_id:
                articles_by_id[aid] = _article_row(aid, polygon.wikidata, art, summary)

            links.append(
                PolygonArticleLink(
                    polygon_id=polygon.polygon_id,
                    article_id=aid,
                    wikidata=polygon.wikidata,
                    language=art.language,
                    source_pbf=polygon.source_pbf,
                    region=polygon.region,
                    osm_type=polygon.osm_type,
                    osm_id=polygon.osm_id,
                    page_id=art.page_id,
                    revision_id=art.revision_id,
                    is_best_language=(art.language == best),
                )
            )
    return list(articles_by_id.values()), links


def _article_row(aid: str, qid: str, art: Any, summary: LinkSummary) -> Article:
    """Build expensive article metadata exactly once per deduplicated row."""
    entity = summary.entity
    label = entity.labels.get(art.language) if entity else ""
    description = entity.descriptions.get(art.language) if entity else ""
    aliases = entity.aliases.get(art.language) if entity else None
    if entity is not None:
        label = label or entity.labels.get("en", "")
        description = description or entity.descriptions.get("en", "")
        aliases = aliases or entity.aliases.get("en", [])
    return Article(
        article_id=aid,
        wikidata=qid,
        language=art.language,
        site=art.site,
        title=art.title,
        url=art.url,
        page_id=art.page_id,
        revision_id=art.revision_id,
        revision_timestamp=art.revision_timestamp,
        retrieved_at=art.retrieved_at,
        wikidata_label=str(label or ""),
        wikidata_description=str(description or ""),
        wikidata_aliases=json_dumps(aliases or []),
        lead_text=art.lead_text,
        extract=art.extract,
        full_text=art.full_text,
        full_text_format=art.full_text_format,
        article_length_chars=len(art.full_text),
        article_length_words=count_words(art.full_text),
        article_length_tokens_estimate=estimate_tokens(art.full_text),
        thumbnail_url=art.thumbnail_url,
        thumbnail_width=art.thumbnail_width,
        thumbnail_height=art.thumbnail_height,
        categories=json_dumps(art.categories),
        license=art.license,
        attribution=art.attribution,
        source_api=art.source_api,
        fetch_status="ok",
        fetch_error="",
        content_hash=content_hash(art.full_text),
    )


def content_hash(text: str) -> str:
    from osm_polygon_wikidata_only.domain.ids import content_hash as _hash

    return _hash(text)


def process_pbf(
    pbf_path: Path,
    *,
    data_root: DataRoot,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    settings: Settings,
    cache: JsonFileCache | None = None,
) -> ProcessResult:
    """Process one PBF end-to-end.

    Steps:

    1. Stream polygonal candidates from the PBF.
    2. Convert to :class:`Polygon` rows.
    3. Enrich each polygon with Wikidata + Wikipedia data.
    4. Deduplicate articles; build polygon-article links.
    5. Write the three parquet files.
    6. Upsert the manifest entry.
    """
    stem = PbfStem.from_path(pbf_path)
    extracted_at = utc_now_iso()
    LOGGER.info("Processing %s (region=%s)", pbf_path.name, stem.region)

    # Step 1-2: extract polygons.
    polygons: list[Polygon] = []
    # Late import via the module so tests can monkeypatch
    # ``osm_polygon_wikidata_only.io.pbf_reader.PBFReader``.
    import osm_polygon_wikidata_only.io.pbf_reader as _pbf_reader_mod

    reader = _pbf_reader_mod.PBFReader(pbf_path)

    def add_candidate(candidate: object) -> None:
        if settings.limit is not None and len(polygons) >= settings.limit:
            return
        polygon = candidate_to_polygon(
            candidate,  # type: ignore[arg-type]
            source_pbf_stem=stem.stem,
            region=stem.region,
            source_pbf=pbf_path.name,
            extracted_at=extracted_at,
        )
        if polygon is not None:
            polygons.append(polygon)

    stream_candidates = getattr(reader, "iter_polygon_candidates", None)
    if callable(stream_candidates):
        stream_candidates(add_candidate)
    else:
        for candidate in reader.collect_polygon_candidates():
            add_candidate(candidate)
    LOGGER.info("Extracted %d polygons from %s", len(polygons), pbf_path.name)

    # Step 3-4: enrich.
    summaries: dict[str, LinkSummary] = {}
    unique_qids = sorted({p.wikidata for p in polygons if p.wikidata})
    if unique_qids:
        # Fetch unique QIDs once, then reuse the summaries for each polygon.
        unique_summaries = fetch_qids(
            unique_qids,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            languages=settings.languages,
            fetch_full_text=settings.fetch_full_text,
            max_articles_per_qid=settings.max_articles_per_qid,
        )
        for s in unique_summaries:
            summaries[s.qid] = s
    LOGGER.info(
        "Fetched summaries for %d unique QIDs (%d QIDs total)",
        len(summaries),
        len(unique_qids),
    )

    enriched: list[Polygon] = [
        _enrich_polygon(
            p,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            settings=settings,
            summaries=summaries,
        )
        for p in polygons
    ]

    articles, links = _build_articles_and_links(enriched, summaries)
    LOGGER.info(
        "Built %d unique articles and %d polygon-article links",
        len(articles),
        len(links),
    )

    # Step 5: write parquet.
    polygons_path = _local_path(data_root.processed, PROCESSED_POLYGONS, stem.stem)
    articles_path = _local_path(data_root.processed, PROCESSED_ARTICLES, stem.stem)
    links_path = _local_path(data_root.processed, PROCESSED_LINKS, stem.stem)

    write_polygons(polygons_path, [polygon_to_dict(p) for p in enriched])
    write_articles(articles_path, [_article_to_dict(a) for a in articles])
    write_polygon_articles(links_path, [_link_to_dict(link) for link in links])

    # Step 6: manifest.
    stats = StreamingStats()
    for p in enriched:
        stats.add_polygon(p)
    for a in articles:
        stats.add_article(a)
    for link in links:
        stats.add_link(link)
    final_stats = stats.finalize()

    mpath = manifest_path(data_root.processed_manifests)
    entry = upsert_entry(
        mpath,
        source_pbf=pbf_path.name,
        region=stem.region,
        polygons_path=_remote_path(PROCESSED_POLYGONS, stem.stem),
        articles_path=_remote_path(PROCESSED_ARTICLES, stem.stem),
        polygon_articles_path=_remote_path(PROCESSED_LINKS, stem.stem),
        stats=final_stats,
        extraction_version=__version__,
    )

    return ProcessResult(
        polygons_path=polygons_path,
        articles_path=articles_path,
        polygon_articles_path=links_path,
        manifest_path=mpath,
        polygon_count=len(enriched),
        article_count=len(articles),
        link_count=len(links),
        manifest_entry=entry,
    )


def _article_to_dict(a: Article) -> dict[str, Any]:
    return dict(a.__dict__)


def _link_to_dict(link: PolygonArticleLink) -> dict[str, Any]:
    return dict(link.__dict__)


__all__ = ["PbfStem", "ProcessResult", "process_pbf"]
