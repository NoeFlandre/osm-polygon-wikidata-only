"""Identity assertions for the :mod:`enrichment.wikimedia` facade.

Every public name re-exported from ``enrichment.wikimedia`` must be
the *same object* as the canonical definition in
:mod:`enrichment.wikimedia_auth` or :mod:`enrichment.wikimedia.transport`.
This guards against accidental shadow re-definitions that would
silently split the public surface from the implementation site.
"""

from __future__ import annotations


def test_facade_re_exports_canonical_session() -> None:
    from osm_polygon_wikidata_only.enrichment.wikimedia import (
        WikimediaHttpSession as FacadeHttpSession,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia import (
        WikimediaSession as FacadeSession,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        WikimediaHttpSession as AuthHttpSession,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        WikimediaSession as AuthSession,
    )

    assert FacadeSession is AuthSession
    assert FacadeHttpSession is AuthHttpSession


def test_facade_re_exports_credentials_and_snapshot_types() -> None:
    from osm_polygon_wikidata_only.enrichment.wikimedia import (
        WikimediaAuthenticationError as FacadeAuthError,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia import (
        WikimediaAuthSnapshot as FacadeSnapshot,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia import (
        WikimediaConfigurationError as FacadeConfigError,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia import (
        WikimediaCredentials as FacadeCredentials,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        WikimediaAuthenticationError as AuthAuthError,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        WikimediaAuthSnapshot as AuthSnapshot,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        WikimediaConfigurationError as AuthConfigError,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        WikimediaCredentials as AuthCredentials,
    )

    assert FacadeSnapshot is AuthSnapshot
    assert FacadeAuthError is AuthAuthError
    assert FacadeConfigError is AuthConfigError
    assert FacadeCredentials is AuthCredentials


def test_facade_re_exports_credential_loader_and_constants() -> None:
    from osm_polygon_wikidata_only.enrichment.wikimedia import (
        WIKIMEDIA_BOT_PASSWORD,
        WIKIMEDIA_BOT_USERNAME,
        WIKIMEDIA_MAX_IN_FLIGHT,
        WIKIMEDIA_REQUESTS_PER_MINUTE,
        load_wikimedia_credentials,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        WIKIMEDIA_BOT_PASSWORD as auth_password,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        WIKIMEDIA_BOT_USERNAME as auth_username,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        WIKIMEDIA_MAX_IN_FLIGHT as auth_max_in_flight,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        WIKIMEDIA_REQUESTS_PER_MINUTE as auth_requests_per_minute,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
        load_wikimedia_credentials as auth_loader,
    )

    assert WIKIMEDIA_BOT_PASSWORD is auth_password
    assert WIKIMEDIA_BOT_USERNAME is auth_username
    assert WIKIMEDIA_MAX_IN_FLIGHT is auth_max_in_flight
    assert WIKIMEDIA_REQUESTS_PER_MINUTE is auth_requests_per_minute
    assert load_wikimedia_credentials is auth_loader


def test_facade_re_exports_transport_helper() -> None:
    from osm_polygon_wikidata_only.enrichment.wikimedia import (
        read_wikimedia_json as FacadeHelper,
    )
    from osm_polygon_wikidata_only.enrichment.wikimedia.transport import (
        read_wikimedia_json as transport_helper,
    )

    assert FacadeHelper is transport_helper


def test_facade_does_not_expose_internal_transport_symbols() -> None:
    import osm_polygon_wikidata_only.enrichment.wikimedia as facade

    for forbidden in (
        "NonObjectJsonError",
        "_NonObjectJsonError",
        "ThrottleCallback",
        "THROTTLE_STATUS_CODES",
    ):
        assert forbidden not in facade.__all__
        assert not hasattr(facade, forbidden), f"facade leaked internal symbol: {forbidden}"
