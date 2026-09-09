CANONICAL_ARTICLE_ID_SQL = "concat(site, ':', page_id, ':', revision_id)"


def canonical_article_id(site: str, page_id: int, revision_id: int) -> str:
    return f"{site}:{page_id}:{revision_id}"
