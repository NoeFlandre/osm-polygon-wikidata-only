"""Property-based invariants for identifiers, QIDs, Wikipedia tags and site keys."""

from __future__ import annotations

import re
from urllib.parse import quote

from hypothesis import given
from hypothesis import strategies as st

from osm_polygon_wikidata_only.domain.ids import article_id, content_hash, polygon_id
from osm_polygon_wikidata_only.domain.wikidata_qids import is_valid_qid, qids_from_osm_tag
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import language_from_site
from osm_polygon_wikidata_only.v2.wikipedia_tags import parse_wikipedia_tags

_qid_numbers = st.integers(min_value=1, max_value=10**12)
_qids = _qid_numbers.map(lambda n: f"Q{n}")
_padding = st.text(alphabet=" \t", max_size=3)

# Components never contain the ``:`` separator, as for real PBF stems,
# OSM types, QIDs and language codes.
_stems = st.from_regex(r"[a-z0-9][a-z0-9._-]{0,30}", fullmatch=True)
_osm_types = st.sampled_from(["node", "way", "relation"])
_ids = st.integers(min_value=0, max_value=2**63 - 1)

_subtag = st.from_regex(r"[a-z]{2,8}", fullmatch=True)
_languages = st.builds(
    lambda primary, subtags: "-".join([primary, *subtags]),
    st.from_regex(r"[a-z]{2,3}", fullmatch=True),
    st.lists(_subtag, max_size=2),
)
_title_chars = st.characters(
    whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" -'(),."
)
_titles = st.text(alphabet=_title_chars, min_size=1, max_size=30).map(str.strip).filter(bool)


@given(n=_qid_numbers)
def test_positive_q_numbers_are_valid(n: int) -> None:
    assert is_valid_qid(f"Q{n}")
    assert not is_valid_qid(f"Q0{n}")
    assert not is_valid_qid(f"q{n}")


@given(qids=st.lists(_qids, min_size=1, max_size=6), pads=st.lists(_padding, min_size=12))
def test_qids_from_osm_tag_ignores_padding_and_dedupes(qids: list[str], pads: list[str]) -> None:
    padded = ";".join(f"{pads[2 * i]}{qid}{pads[2 * i + 1]}" for i, qid in enumerate(qids))
    result = qids_from_osm_tag(padded)
    assert result == qids_from_osm_tag(";".join(qids))
    assert result == tuple(dict.fromkeys(qids))
    assert len(set(result)) == len(result)
    assert all(is_valid_qid(qid) for qid in result)


@given(value=st.text(max_size=40))
def test_qids_from_osm_tag_only_returns_valid_qids(value: str) -> None:
    assert all(is_valid_qid(qid) for qid in qids_from_osm_tag(value))


@given(a=st.tuples(_stems, _osm_types, _ids), b=st.tuples(_stems, _osm_types, _ids))
def test_polygon_id_is_deterministic_and_injective(
    a: tuple[str, str, int], b: tuple[str, str, int]
) -> None:
    assert polygon_id(*a) == polygon_id(*a)
    assert re.fullmatch(r"[^:]+:(node|way|relation):\d+", polygon_id(*a))
    assert (polygon_id(*a) == polygon_id(*b)) == (a == b)


_article_inputs = st.tuples(_qids, _languages, _ids, _ids)


@given(a=_article_inputs, b=_article_inputs)
def test_article_id_is_deterministic_and_injective(
    a: tuple[str, str, int, int], b: tuple[str, str, int, int]
) -> None:
    assert article_id(*a) == article_id(*a)
    assert re.fullmatch(r"Q\d+:[a-z-]+:\d+:\d+", article_id(*a))
    assert (article_id(*a) == article_id(*b)) == (a == b)


@given(a=st.text(), b=st.text())
def test_content_hash_is_deterministic_hex_and_injective(a: str, b: str) -> None:
    assert content_hash(a) == content_hash(a)
    assert re.fullmatch(r"[0-9a-f]{64}", content_hash(a))
    assert (content_hash(a) == content_hash(b)) == (a == b)


@given(language=_languages, title=_titles)
def test_wikipedia_prefix_and_url_forms_normalise_identically(language: str, title: str) -> None:
    url = f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='')}"
    prefixed, prefixed_rejected = parse_wikipedia_tags({"wikipedia": f"{language}:{title}"})
    from_url, url_rejected = parse_wikipedia_tags({"wikipedia": url})
    assert prefixed_rejected == url_rejected == ()
    assert [(r.language, r.title) for r in prefixed] == [(r.language, r.title) for r in from_url]
    assert [(r.language, r.title) for r in prefixed] == [(language, title)]


@given(
    tags=st.dictionaries(
        st.one_of(st.just("wikipedia"), _languages.map(lambda lang: f"wikipedia:{lang}")),
        st.lists(
            st.one_of(
                st.builds(lambda lang, t: f"{lang}:{t}", _languages, _titles),
                _titles,
            ),
            min_size=1,
            max_size=3,
        ).map(";".join),
        max_size=4,
    )
)
def test_parse_wikipedia_tags_is_idempotent_on_normalised_output(tags: dict[str, str]) -> None:
    refs, _ = parse_wikipedia_tags(tags)
    normalised = {"wikipedia": ";".join(f"{ref.language}:{ref.title}" for ref in refs)}
    again, rejected = parse_wikipedia_tags(normalised) if refs else ((), ())
    assert rejected == ()
    assert sorted({(r.language, r.title) for r in again}) == sorted(
        {(r.language, r.title) for r in refs}
    )


@given(language=_languages)
def test_language_from_site_inverts_site_keys(language: str) -> None:
    assert language_from_site(f"{language.replace('-', '_')}wiki") == language
    assert language_from_site(f"{language}wiki") == language
