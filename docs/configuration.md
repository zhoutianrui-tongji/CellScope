# CellScoper Configuration Reference (params.yaml)

This document describes the configuration options read from `config/params.yaml`. All fields are optional unless stated otherwise. CLI flags control only high-level workflow; module hyperparameters are configured here.

- File path: `config/params.yaml`
- Format: YAML (UTF‑8)
- Scope: IO behavior, resume policy, module parameters (M1–M8), annotation, and UI/runtime hints

## Loading and precedence
- The pipeline loads `params.yaml` at runtime. If `--config` (JSON) is provided via CLI, it is merged on top (deprecated).
- CLI flags override only a few booleans (resume, save intermediates, sample preview) and resume policy; module hyperparameters come from YAML.

## io
- `final_format` (`csv|parquet`): Final table format. Default: `csv`.
- `h5ad_compression` (`lzf|gzip|None`): Compression for `.h5ad` intermediates and outputs. Default: `lzf`.
- `lite_intermediate` (bool): Reduce intermediate artifacts written. Default: `false`.
- `intermediate_dirname` (str): Name of the intermediate directory under output dir. Default: `intermediate`.
- `resume_from` (`load_inputs|preprocess|build_anndata|annotate|write_outputs`|null): Force starting stage. Default: unset.
- `resume_policy` (`auto|minimal|force`): Default resume behavior (CLI can override). Default: `auto`.
- `write_intermediate_csv` (bool): Write CSV intermediates when applicable. Default: `true`.
- `write_intermediate_parquet` (bool): Write Parquet intermediates when applicable. Default: `false`.
- `write_csv_copy` (bool): Always keep a CSV copy of the final table if feasible. Default: `true`.

## annotation
- `enable` (bool): Enable RAP-X style subcellular annotation (requires `scanpy`). Default: `true`.
- `two_stage` (bool): Optional two-stage enrichment. Default: `false`.
- `spatial_channel` (bool): Include spatial channel in annotation scoring. Default: `false`.
- `guard_mode` (`adaptive|fixed`): In `adaptive` mode, subtype support is calibrated against label-specific random marker sets with matched set size and weight profile. `fixed` restores the legacy absolute margin/q/support thresholds. Default: `adaptive`.
- `subtype_fdr` (float): Label-wise empirical FDR cutoff used by adaptive Nuclear Pore, Lamina, and OMM guards. Default: `0.05`.
- `min_effective_markers` (int): Minimum measured markers required for a subtype to be testable. Labels below this coverage are reported as not testable and fall back to their broad compartment. Default: `10`.
- `null_permutations` (int): Number of matched random marker sets per label. Empirical p-values use the add-one rule. Default: `500`.
- `null_seed` (int): Random seed for empirical-null calibration. Default: `0`.
- `omm_metaspot_refinement` (bool): Refine OMM versus ER/Cytosol at MetaSpot resolution after the domain-level RAP-X call. Default: `true`.
- `omm_metaspot_apply_to_primary` (bool): Apply the MetaSpot OMM refinement to the primary `subcellular_annotation` output. Default: `true`.
- `omm_metaspot_ambiguous_policy` (`competitor|explicit`): With `competitor`, ambiguous OMM/ER MetaSpots fall back to their strongest ER/Cytosol alternative so the primary output stays in the seven-class label space; ambiguity remains in the OMM audit columns. `explicit` retains the legacy `ER-OMM shared` prediction class. Default: `competitor`.
- `single_stage_compartment_score_weight` (float): Weight of the bounded nuclear/cytoplasmic consistency score added before the single-stage ranking. All labels remain in one competition. Default: `6.0`; set to `0` to disable.
- `single_stage_compartment_score_tau` (float): Smoothness of the signed-geometry score around the nuclear boundary. Default: `0.2`.
- `single_stage_compartment_guard` (bool): Post-hoc rescue for a supported same-side candidate. Default: `true`.
- `single_stage_compartment_min_abs_r`, `single_stage_compartment_max_score_gap`, `single_stage_compartment_min_z`, `single_stage_compartment_max_q`, `single_stage_compartment_min_channel_support`: boundary dead zone shared by joint scoring and thresholds for the legacy guard. Defaults: `0.05`, `8.5`, `1.0`, `0.35`, `1`.
- `lamina_guard` (bool): Collapse weak or non-peripheral Lamina calls to `lamina_fallback_label`. Default: `true`.
- `lamina_min_margin` (float): Minimum Lamina top-score margin against the runner-up in `guard_mode: fixed`. Default: `1.0`.
- `lamina_non_nuclear_top2_min_margin` (float): Stricter margin when Lamina competes with a cytoplasmic label. Default: `3.0`.
- `lamina_max_q` (float), `lamina_min_channel_support` (int): Statistical support required to retain Lamina in `guard_mode: fixed`. Defaults: `0.01`, `2`.
- `lamina_boundary_quantile` (float): Quantile of group-level `d_nuc_norm_mean` among nuclear-side groups used as the maximum Lamina distance from the nuclear envelope. Smaller values are stricter. Default: `0.5`.
- `lamina_fallback_label` (str): Broad label used for guarded Lamina calls. Default: `Nucleus`.
- `leiden_resolution` (float): Leiden resolution for cluster finding. Default: `1.0`.
- `hvg_n_top_genes` (int): HVG selection count. Default: `2000`.
- `pca_n_comps` (int): PCA components. Default: `25`.
- `neighbors_k` (int): k for neighbor graph (annotation step). Default: `15`.
- `umap_min_dist` (float), `umap_spread` (float): UMAP parameters. Default: `0.3`, `1.0`.
- `umap_prefix` (str): UMAP artifact base name. Default: `rapx_umap`.
- `enrichment_panels_name` (str), `quality_summary_name` (str): Output names for diagnostic plots. Default: `rapx_enrichment_panels`, `rapx_quality_summary`.
- `compute_umap` (bool), `save_umap_plot` (bool), `save_plots` (bool): UMAP/plot toggles. Default: `true`, `true`, `true`.
- `cluster_rep` (str): Preferred representation for Leiden/UMAP: `X_dgi_gcn|X_dgi_sage|X_expr|X_pca|auto`. Default: `X_dgi_gcn`.

## module1 (Geometry features)
- `knn_k` (int): Neighborhood size for geometry context. Default: `32`.
- `include_directional` (bool): Add directional features (sin/cos). Default: `true`.
- `winsor_perc_low`/`winsor_perc_high` (int, 0–100): Winsorization percentiles. Default: `1`, `99`.

## module2 (Composition embeddings)
- `top_g` (int): Vocabulary size (top genes). Default: `4096`.
- `bow_knn_k` (int): Neighborhood size for BoW. Default: `32`.
- `max_rows_per_cell_for_svd` (int): Per-cell TF rows cap for SVD sample. Default: `500`.
- `max_svd_rows` (int): Global TF rows cap for SVD sample. Default: `2_000_000`.
- `svd_n_components` (int): Latent dimensionality. Default: `24`.
- `bow_weight_mode` (`gaussian|uniform`): Neighbor weighting. Default: `gaussian`.
- `use_tfidf` (bool), `tf_norm` (bool): SVD weighting and TF normalization. Default: `true`, `true`.
- `parallel_chunk_cells` (int): Grouping factor for parallel TF blocks. Default: `128`.
- `stream_local_embeddings` (bool): Compute transcript-level neighborhood expression inside Module 3 one cell at a time instead of storing dense embedding columns for all transcripts. Default: `true`.

## module3 (Per-cell MetaSpot segmentation)
- `hdbscan_min_cluster_size` (int): Minimum subcluster size. Default: `20`.
- `hdbscan_allow_single_cluster` (bool): Allow single-cluster HDBSCAN. Default: `true`.
- `hdbscan_min_samples_frac` (float): Fraction of min_cluster_size for `min_samples`. Default: `0.5`.
- `kmeans_min_k`/`kmeans_max_k` (int): Bounds for KMeans fallback. Default: `2`, `6`.
- `merge_min_size` (int): Merge clusters smaller than this threshold. Default: `25`.
- `enforce_spatial_connectivity` (bool): Build every final `metaspot` as a connected XY region using spatially constrained Ward clustering.
- `target_sites_per_metaspot` (int|null): Target number of unique XY sites per MetaSpot. When set, this determines the per-cell K and bypasses HDBSCAN as the primary segmenter. Default: `100`.
- `min_metaspots_per_cell` (int): Lower bound for the per-cell K. Default: `2`; seqFISH profile: `20`.
- `max_metaspots_per_cell` (int|null): Upper bound for the per-cell K. Default: `100`.
- `spatial_connectivity_k` (int): Maximum nearby XY neighbors considered when constructing the local spatial graph. Default: `8`.
- `spatial_radius_quantile` (float): Quantile of per-site nearest-neighbor distances used to estimate native point spacing. Default: `0.9`.
- `spatial_radius_factor` (float): Multiplier applied to the estimated point spacing for the graph radius. Default: `1.5`.
- `spatial_smoothing_iterations` (int): Local majority-smoothing passes used only if constrained Ward clustering cannot run. Default: `1`.
- `spatial_smoothing_min_fraction` (float): Minimum weighted neighbor fraction required to change a site's preliminary label in that fallback. Default: `0.6`.
- `spatial_min_component_size` (int): Connected components smaller than this transcript count are merged across their strongest shared boundary. Default: `20`.
- `spatial_compactness` (float): Weight of standardized XY coordinates in the spatially constrained Ward clustering. Larger values make MetaSpots more compact; `0` uses features only while still restricting merges to spatial neighbors. Default: `1.0`.
- `hdbscan_*`, `target_sub_size`, and `kmeans_*`: Legacy fallback controls used when `target_sites_per_metaspot` is null or spatial connectivity is disabled.

## module5 (Meta-domain clustering)
- `block_size` (int): Chunk size for processing. Default: `50000`.
- `w_gc`/`w_expr`/`w_sp` (float): Feature weights (geometry, expression, spatial). Default: `0.5/0.5/0.0`.
- `use_gc`/`use_expr`/`use_spatial_feats`/`use_decell`/`use_cosine` (bool): Feature toggles. Defaults: `true/true/true/true/true` (spatial feats default may vary by build).
- `speed_preset` (`ultra|fast|balanced`): Internal speed/accuracy presets. Default: `ultra`.
- `enable_cap` (bool): Per-cell cap enabled. Default: `true`.
- K estimation:
  - `estimate_k` (`auto|fixed`): Strategy. Default: `auto`.
  - `fixed_k` (int): Used when `estimate_k=fixed`. Default: `1024`.
  - `target_sub_per_cluster` (int), `target_cluster_per_cell` (float): Targets for automatic K.
  - `K_min`/`K_max` (int): Hard bounds for K.
- Nucleus/cytoplasm split:
  - `split_by_nucleus` (bool), `nucleus_threshold` (float).
- Per-cell cap:
  - `cap_ratio_base` (float), `cap_min_keep` (int), `cap_margin_tau` (float), `cap_max_frac_soft` (float).

## module6 (Aggregation to adata2)
- `h5ad_compression_override` (bool): Respect IO compression override for `adata2`. Default: `true`.
- `save_shape_summary` (bool): Save `module6_adata2_shape.*`. Default: `true`.

## module7 (Graph construction and expression latent)
- `scvi_epochs` (int), `scvi_n_latent` (int), `scvi_lr` (float), `scvi_batch_size` (int): SCVI training params.
- `rep_priority` (list[str]): Representation preference for annotation: `X_dgi_gcn`, `X_dgi_sage`, `X_expr`, `X_pca`.
- `neighbors.k_expr`/`neighbors.k_geom` (int): k for expression/geometry graphs. Default: `15/15`.
- `alpha_expr`/`alpha_geom` (float): Fusion weights. Default: `0.5/0.5`.
- `same_cell_beta` (float): Bias for same-cell edges. Default: `0.5`.
- `topk_prune` (int): Row-wise top-k pruning of edges. Default: `32`.
- `edge_norm` (`rw|gcn|none`): Edge normalization. Default: `rw`.
- `feature_scale` (`standard|none`): Feature scaling mode. Default: `standard`.

## module8 (DGI embeddings)
- `dgi_epochs` (int): Training epochs. Default: `150`.
- `enable_dgi_sage` (bool): Train SAGE variant in addition to GCN. Default: `true`.
- `dgi_hidden`/`dgi_out` (int): Hidden and output dimensions. Default: `32/32`.
- `dgi_lr` (float), `dgi_weight_decay` (float), `dgi_clip_grad` (float): Optimizer settings.

## runtime (parallelism)
- `n_jobs` (int): Preferred worker count for parallel sections (honored unless `CELLSCOPE_N_JOBS` env is set).
- `joblib_backend` (`threading|loky`): Backend for joblib blocks (env `CELLSCOPE_JOBLIB_BACKEND` still wins).
- `blas_threads` (int): Caps BLAS/OpenMP thread pools; applied at startup and optionally via `threadpoolctl`.
- `use_threadpoolctl` (bool): If true and `blas_threads` is set, also call `threadpoolctl.threadpool_limits`.
- `neighbors_n_jobs` (int): Reserved for neighbor graph overrides (falls back to `n_jobs`).

## ui
- `show_sample` (bool): Show top-5 preview of inputs at start. Default: `true`.
- `progress_style` (`rich|plain`): Output style hint for CLI progress.

## Environment variables
- `CELLSCOPE_N_JOBS` (int): Override parallel job count for selected blocks.
- `CELLSCOPE_CLUSTER_CHUNK` (int): Chunk size for clustering batches (default ~32–128).

## Examples

Minimal (CSV output, defaults):
```yaml
io:
  final_format: csv
annotation:
  enable: true
```

Parquet output + strong pruning + faster DGI:
```yaml
io:
  final_format: parquet
  h5ad_compression: lzf
  lite_intermediate: true
module7:
  topk_prune: 16
module8:
  dgi_epochs: 60
  enable_dgi_sage: false
```

Resume controls in YAML (can also use CLI):
```yaml
io:
  resume_policy: minimal
  resume_from: build_anndata
```

## Change impact and resume
- The pipeline fingerprints module-relevant parts of the YAML. With `--resume-policy minimal`, it reruns from the earliest affected stage.
- First full run prints `Module X · Done` (no suffix). Subsequent runs show `(cache|recomputed)`.

## Troubleshooting
- Changes not taking effect: ensure `resume_policy` is `minimal` or use `--resume-policy minimal` and/or remove the output directory.
- Missing dependencies (e.g., `scanpy`/`scvi-tools`): the pipeline will use fallback paths (e.g., SVD/NN); adjust epochs or disable annotation if needed.
- Large runs: consider lowering `module8.dgi_epochs` and enabling `lite_intermediate`.
