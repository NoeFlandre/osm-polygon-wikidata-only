"""Lightweight constants shared by configuration, CLI, and runtime modules."""

DEFAULT_REPO_ID = "NoeFlandre/osm-polygon-wikidata-only"
V2_REPO_ID = "NoeFlandre/osm-polygon-wikidata-and-wikipedia"

# Wikimedia requires a User-Agent identifying the project and a contact.
# This is overridable via env var so deployments can set their own.
DEFAULT_USER_AGENT = (
    "osm-polygon-wikidata-only/0.1.0 (https://github.com/NoeFlandre/osm-polygon-wikidata-only) "
    "datasets-pipeline"
)

DEFAULT_MAX_STEMS = 4
DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024
DEFAULT_WALLTIME = "0:30"
DEFAULT_GRID5000_SITE = "grenoble"
DEFAULT_GRID5000_QUEUE = "besteffort"
DEFAULT_GRID5000_GPU_MODEL = "A40"
DEFAULT_BATCH_SIZE = 256
DEFAULT_INFERENCE_BATCH_SIZE = 16

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_GRID5000_GPU_MODEL",
    "DEFAULT_GRID5000_QUEUE",
    "DEFAULT_GRID5000_SITE",
    "DEFAULT_INFERENCE_BATCH_SIZE",
    "DEFAULT_MAX_INPUT_BYTES",
    "DEFAULT_MAX_STEMS",
    "DEFAULT_REPO_ID",
    "DEFAULT_USER_AGENT",
    "DEFAULT_WALLTIME",
    "V2_REPO_ID",
]
