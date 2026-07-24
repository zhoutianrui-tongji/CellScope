# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/en/1.0.0/) and this project adheres to Semantic Versioning.

## [Unreleased]
### Changed
- Renamed the user-facing project and CLI to CellScoper; `cellscope` remains a compatibility alias.
- Replaced the output columns `cluster_in_cell` and `cell_sub` with `metaspot_in_cell` and `metaspot`.
- Made spatially constrained Ward clustering with an adaptive unique-XY-site target the canonical MetaSpot segmenter.

## [1.0.0] - 2025-12-02
### Added
- Initial public release of CellScoper.
- Modular pipeline (M1–M8), YAML configuration, resumable checkpoints.
- Clear module completion/cache messages; first-run suffix suppression.
- Intermediate save notifications and configurable intermediate directory.
- RAP-X annotation with RNALocate/APEX mapping and diagnostics.
- Comprehensive configuration reference under `docs/configuration.md`.

### Fixed
- CLI bin script shebang correctness.

### Changed
- README rewritten in English with concise quickstart.
