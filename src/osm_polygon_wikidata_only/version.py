"""Single source of the package version, read from the installed metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    # ``pyproject.toml`` is the only place the version is written down.
    PACKAGE_VERSION = version("osm-polygon-wikidata-only")
except PackageNotFoundError:  # pragma: no cover - only without installed metadata
    PACKAGE_VERSION = "0.0.0+unknown"
