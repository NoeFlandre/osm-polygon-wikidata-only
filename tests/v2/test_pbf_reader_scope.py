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
