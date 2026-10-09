# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/en/1.0.0/) and this project adheres to Semantic Versioning.

## [Unreleased]
### Fixed
- Batched RAP-X, hybrid MetaSpot, provenance, and transcript annotation column
  writeback to avoid pandas DataFrame fragmentation and repeated
  `PerformanceWarning` messages on annotation-heavy runs.
- Preserved nucleus-contour vertex order and explicitly closed polygons so
  boundary-derived nuclear membership is not corrupted by row sorting.
- Prevented changed input files from reusing stale raw-input checkpoints.
- Removed `overlaps_nucleus`-based cell selection when polygon geometry is
  configured, avoiding benchmark population leakage.
- Made malformed explicit YAML configurations fail visibly.

### Added
- Native CSV/Parquet input loading and bounded previews for all three tables.
- Packaged default configuration and annotation references for wheel installs.
- Added Cell-Aware Deep Multiplex Graph Infomax (CA-DMGI) with separate
  cross-cell expression and
  within-cell spatial encoders, within-cell feature corruption, a bounded
  node-wise consensus gate, branch embeddings, and full training provenance.
- Added sparse stability-selected consensus Leiden for MetaDomain discovery,
  including multi-resolution/multi-seed auditing, cross-cell recurrence
  constraints, and edge-level co-association without a dense consensus matrix.

### Changed
- Reduced CA-DMGI full-batch cost by fusing positive/corrupted sparse
  propagation, auto-tuning CPU threads, using compact CSR indices, exposing an
  explicit CPU/CUDA selector with fail-fast CUDA validation, and skipping the
  unused standard-DGI graph conversion on the multiplex path.
- Made CA-DMGI plus consensus Leiden the packaged unsupervised
  clustering path; the previous single-graph DGI and MiniBatchKMeans paths
  remain available for reproducible ablations.
- Increased fragmentation-aware OMM sensitivity: pooled cell evidence still
  requires at least 6 marker transcripts from 3 genes plus enrichment and a
  binomial gate, while a focal one-marker anchor can be localized only with
  OMM-over-ER/Cytosol separation and majority DGI support. The bounded per-cell
  cap is now 5 evidence-qualified loci.
- Replaced the default two-layer plain DGI-GCN encoder with a configurable
  two-block initial-residual encoder and changed the packaged MetaSpot DGI
  output dimension from 64 to 32. The historical plain encoder remains
  available through `module8.dgi_encoder: plain`.
- Added DGI architecture, depth, dropout, residual, normalization, optimizer,
  and loss-history provenance; architecture changes now invalidate M8 caches.
- Renamed the user-facing project and CLI to CellScoper; `cellscope` remains a compatibility alias.
- Replaced the output columns `cluster_in_cell` and `cell_sub` with `metaspot_in_cell` and `metaspot`.
- Made spatially constrained Ward clustering with an adaptive unique-XY-site target the canonical MetaSpot segmenter.
- Restored the 200,000-node full-batch MetaSpot-DGI safety guard and documented
  `metadomain` as the large-dataset path.
- Limited warning suppression to targeted third-party compatibility notices.

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
