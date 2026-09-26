# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The version is defined only in `pyproject.toml`; pushing a matching `vX.Y.Z` tag
builds the sdist and wheel and attaches them to a GitHub Release.

## [Unreleased]

### Changed

- The package version is single-sourced from `pyproject.toml`: `__version__`
  and the default Wikimedia User-Agent are derived from the installed metadata.
- Added a tag-triggered release workflow and this changelog.

## [0.1.0] - 2026-09-26

### Added

- Initial release: polygon-only OSM PBF extraction joined with Wikidata facts
  and multilingual Wikipedia/Wikivoyage text, published as a multi-table
  Hugging Face dataset. Manifests record the code version as `extraction_version`.

[Unreleased]: https://github.com/NoeFlandre/osm-polygon-wikidata-only/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NoeFlandre/osm-polygon-wikidata-only/releases/tag/v0.1.0
