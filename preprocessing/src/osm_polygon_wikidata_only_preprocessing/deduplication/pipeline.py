from pathlib import Path

from .articles import DeduplicationStats, deduplicate_article_files


def deduplicate_region(
    raw_root: Path,
    processed_root: Path,
    region_stem: str,
) -> DeduplicationStats:
    if not region_stem or Path(region_stem).name != region_stem:
        raise ValueError("region_stem must be a filename stem")

    filename = f"{region_stem}.parquet"
    return deduplicate_article_files(
        raw_root / "articles" / filename,
        raw_root / "polygon_articles" / filename,
        processed_root / "articles" / filename,
        processed_root / "article_entities" / filename,
        processed_root / "polygon_articles" / filename,
    )
