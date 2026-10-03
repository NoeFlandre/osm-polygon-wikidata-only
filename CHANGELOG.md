# Changelog

This file records all notable changes to the project.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Only `pyproject.toml` defines the version. When you push a matching `vX.Y.Z` tag, the workflow builds the sdist and the wheel and attaches them to a GitHub Release.

## [Unreleased]

- Read SaT model capabilities from a versioned, digest-checked offline reference.
  Preserve sentence-routing policy, runtime model pins, and historical fingerprints.

### Changed

- `pyproject.toml` is the only source of the package version. `__version__` and the default Wikimedia User-Agent come from the installed metadata.
- Added a release workflow that a tag starts, and this changelog.

## [0.1.0] - 2026-09-26

### Added

- Initial release. It extracts polygons only from OSM PBF files and joins them with Wikidata facts and multilingual Wikipedia and Wikivoyage text. It publishes the result as a multi-table Hugging Face dataset. The manifests record the code version as `extraction_version`.

[Unreleased]: https://github.com/NoeFlandre/osm-polygon-wikidata-only/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NoeFlandre/osm-polygon-wikidata-only/releases/tag/v0.1.0
