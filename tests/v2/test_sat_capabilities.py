"""Independent offline contracts for the shared SaT capability reference."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import pytest

from osm_polygon_wikidata_only.v2.sat_capabilities import (
    REFERENCE_SHA256,
    REFERENCE_VERSION,
    load_supported_languages,
    parse_supported_languages,
)

EXPECTED_CODES = (
    "af",
    "am",
    "ar",
    "az",
    "be",
    "bg",
    "bn",
    "ca",
    "ceb",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "no",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "ta",
    "te",
    "tg",
    "th",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
)


def test_reference_keeps_the_original_85_codes() -> None:
    assert load_supported_languages() == EXPECTED_CODES
    assert REFERENCE_VERSION == "sat-3l-sm-v1"


def test_packaged_reference_has_valid_provenance_and_digests() -> None:
    content = files("osm_polygon_wikidata_only").joinpath("v2/sat-capabilities.json").read_bytes()
    assert hashlib.sha256(content).hexdigest() == REFERENCE_SHA256
    reference = json.loads(content)
    digest = reference.pop("digest")
    canonical = json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()
    assert digest == "sha256:" + hashlib.sha256(canonical).hexdigest()
    assert reference["reference_version"] == REFERENCE_VERSION
    assert reference["model_id"] == "segment-any-text/sat-3l-sm"
    assert reference["model_revision"] == "137da054051ad9f1eac42025f758db4ac9f22535"
    assert reference["authority"] == "NoeFlandre/osm-polygon-description-tag"
    assert reference["schema_version"] == 1
    assert reference["supported_languages"] == list(EXPECTED_CODES)
    assert parse_supported_languages(content) == EXPECTED_CODES


@pytest.mark.parametrize("content", [b"", b"{}", b"invalid", b"[]"])
def test_reference_loader_fails_closed_for_unpinned_bytes(content: bytes) -> None:
    with pytest.raises(ValueError, match="SaT capability reference digest mismatch"):
        parse_supported_languages(content)


def test_a_valid_json_reference_with_one_changed_code_is_rejected() -> None:
    content = files("osm_polygon_wikidata_only").joinpath("v2/sat-capabilities.json").read_bytes()
    changed = content.replace(b'"en"', b'"xx"')
    with pytest.raises(ValueError, match="SaT capability reference digest mismatch"):
        parse_supported_languages(changed)


def test_wikidata_exact_article_language_policy_and_revision_are_unchanged() -> None:
    from osm_polygon_wikidata_only.v2.sat import DEFAULT_SAT_MODEL_REVISION
    from osm_polygon_wikidata_only.v2.sentence_logic import (
        SAT_SUPPORTED_LANGUAGES,
        is_sat_supported_language,
    )

    assert SAT_SUPPORTED_LANGUAGES == EXPECTED_CODES
    assert DEFAULT_SAT_MODEL_REVISION == "137da05"
    for language in ("eng", "eng_Latn", "en-US", "EN", "ckb", "khk", "pbt", "ydd", "yue"):
        assert not is_sat_supported_language(language)
    assert is_sat_supported_language("en")
