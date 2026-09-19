"""Bounded architecture checks for the Grid5000 controller split."""

from osm_polygon_wikidata_only.grid5000 import (
    sentence_controller,
    sentence_controller_policy,
    sentence_publication,
    sentence_transport,
)


def test_controller_facade_preserves_split_api_boundaries() -> None:
    assert sentence_controller.ControllerLimits is sentence_controller_policy.ControllerLimits
    assert sentence_controller.ControllerRunError is sentence_controller_policy.ControllerRunError
    assert sentence_controller.Grid5000Transport is sentence_transport.Grid5000Transport
    assert sentence_controller.HubPublisher is sentence_publication.HubPublisher


def test_implementation_owners_are_focused_modules() -> None:
    assert (
        sentence_controller.SubprocessGrid5000Transport.__mro__[1]
        is sentence_transport.SubprocessGrid5000Transport
    )
    assert (
        sentence_controller.HfHubSentencePublisher.__mro__[1]
        is sentence_publication.HfHubSentencePublisher
    )

