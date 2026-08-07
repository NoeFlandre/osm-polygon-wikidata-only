def test_v1_reader_keeps_the_wikidata_only_default(tmp_path) -> None:
    from osm_polygon_wikidata_only.io.pbf_reader import PBFReader

    path = tmp_path / "fixture.osm.pbf"
    path.touch()
    reader = PBFReader(path)
    assert reader.include_wikipedia_tagged is False


def test_reader_exposes_an_explicit_wikipedia_tag_opt_in(tmp_path) -> None:
    from osm_polygon_wikidata_only.io.pbf_reader import PBFReader

    path = tmp_path / "fixture.osm.pbf"
    path.touch()
    reader = PBFReader(path, include_wikipedia_tagged=True)
    assert reader.include_wikipedia_tagged is True


def test_relevant_tag_filter_matches_last_duplicate_tag_semantics() -> None:
    from osm_polygon_wikidata_only.io.pbf_reader import _PolygonHandler

    class Tag:
        def __init__(self, key: str, value: str) -> None:
            self.k = key
            self.v = value

    def tags(*items: tuple[str, str]) -> list[Tag]:
        return [Tag(key, value) for key, value in items]

    assert _PolygonHandler._has_relevant_tag(
        tags(("name", "ignored"), ("wikidata", "Q42")), include_wikipedia_tagged=False
    )
    assert not _PolygonHandler._has_relevant_tag(
        tags(("wikidata", "Q42"), ("wikidata", "")), include_wikipedia_tagged=False
    )
    assert _PolygonHandler._has_relevant_tag(
        tags(("wikipedia:fr", "Titre")), include_wikipedia_tagged=True
    )
    assert not _PolygonHandler._has_relevant_tag(
        tags(("wikipedia:fr", "Titre"), ("wikipedia:fr", "")),
        include_wikipedia_tagged=True,
    )
