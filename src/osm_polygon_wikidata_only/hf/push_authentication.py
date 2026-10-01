"""Hugging Face credential preflight for commands that publish."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import NoReturn

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf.uploader import (
    UploadError,
    resolve_hf_token,
    verify_hf_token,
    verify_repo_authorization,
)

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")

Failure = Callable[[str], NoReturn]


def authenticate_push_targets(settings: Settings, repo_ids: Iterable[str], fail: Failure) -> None:
    """Validate the token and write access for every target repository.

    ``fail`` reports a user-facing message and must not return.
    """
    for repo_id in repo_ids:
        target = replace(settings, repo_id=repo_id)
        _require_token(target, fail)
        _verify_access(target, fail)


def _require_token(settings: Settings, fail: Failure) -> None:
    if resolve_hf_token(settings.hf_token):
        return
    explicit = bool(settings.hf_token)
    if explicit or os.environ.get("HF_TOKEN"):
        source = "--hf-token" if explicit else "HF_TOKEN"
        fail(
            f"--push: {source} is set but Hugging Face rejected it as invalid. "
            "Generate a fresh write token at https://huggingface.co/settings/tokens "
            "and replace the current value."
        )
    fail(
        "--push requires a Hugging Face write token: pass --hf-token, "
        "set HF_TOKEN, or run `huggingface-cli login`."
    )


def _verify_access(settings: Settings, fail: Failure) -> None:
    LOGGER.info("Connecting to Hugging Face using bounded IPv4 transport (connect timeout: 10s)")
    try:
        username = verify_hf_token(settings.hf_token)
        verify_repo_authorization(settings.hf_token, settings.repo_id)
    except UploadError as error:
        fail(str(error))
    LOGGER.info("Authenticated to Hugging Face as %s (target: %s)", username, settings.repo_id)
