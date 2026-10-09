# CellScoper Configuration Reference (params.yaml)

This document describes the configuration options read from `config/params.yaml`. All fields are optional unless stated otherwise. CLI flags control only high-level workflow; module hyperparameters are configured here.

- File path: `config/params.yaml`
- Format: YAML (UTF‑8)
- Scope: IO behavior, resume policy, module parameters (M1–M8), annotation, and UI/runtime hints

## Loading and precedence
- `--config` accepts YAML and is recursively merged over the packaged default.
- The merged configuration plus CLI overrides is atomically written to
  `effective_params.yaml`; all pipeline stages read that exact file.
- Explicit CLI/Python arguments take precedence over YAML. After a successful
  non-dry run, the same configuration is committed as `params_used.yaml`.

## io
- `final_format` (`csv|parquet`): Final table format. The shipped production
  configuration uses `parquet`; enable `write_csv_copy` only when a CSV is
  explicitly needed.
- `h5ad_compression` (`lzf|gzip|None`): Compression for `.h5ad` intermediates and outputs. Default: `lzf`.
- `lite_intermediate` (bool): Reduce intermediate artifacts written. Default: `true`.
- `intermediate_dirname` (str): Name of the intermediate directory under output dir. Default: `intermediate_results`.
- `resume_from` (`load_inputs|preprocess|build_anndata|annotate|write_outputs`|null): Force starting stage. Default: unset.
- `resume_policy` (`auto|minimal|force`): Default resume behavior (CLI can override). Default: `minimal`.
- `write_intermediate_csv` (bool): Write CSV intermediates when applicable. Default: `false`.
- `write_intermediate_parquet` (bool): Write Parquet intermediates when applicable. Default: `false`.
- `write_csv_copy` (bool): Also keep a CSV copy of the final table if feasible.
  The shipped production configuration defaults to `false` to avoid duplicating
  a gigabyte-scale table; set it to `true` when CSV is explicitly required.
- `write_diagnostic_tables` (bool): Write compact audit tables independently
  of large intermediate tables. Default: `false`.
- `write_diagnostic_csv`, `write_diagnostic_parquet` (bool): Formats for
  compact audit tables. Defaults: `false`, `false`.
- `broadcast_metaspot_diagnostics_to_transcripts` (bool): Duplicate wide
  MetaSpot audit columns onto every transcript. Default: `false`; MetaSpot
  diagnostics remain available in `adata1.obs`.
- `write_partitioned_dataset` (bool): Also write `final_data_dataset/` beside
  the canonical final table for streaming runs. Default: `false`.
- `drop_embeddings` (bool): Remove transcript embedding-like columns and write
  a compact final `adata1.h5ad`. The compact file retains the canonical
  `X_dgi_gcn_metaspot` representation needed for DGI provenance, but omits
  auxiliary embeddings and graph matrices. Default: `false`.
- `provenance_hash_inputs` (bool): Include exact SHA256 hashes of inputs and
  bundled annotation references in run manifests and resume fingerprints.
  Default: `true`.

## annotation

The packaged YAML exposes workflow controls and the production MetaSpot
evidence gates. Lower-level statistical and legacy guard settings retain
stable internal defaults and remain accepted as expert overrides.

- `enable` (bool): Enable RAP-X annotation. Default: `true`.
- `method` (str): Annotation implementation. Currently `rapx`.
- `unit` (`metaspot|metadomain|leiden`): Annotation decision unit. The packaged
  YAML uses `metaspot`; the backward-compatible internal default remains
  `metadomain`.
  - `metaspot`: run RAP-X on MetaDomains only to establish a broad
    nuclear/cytoplasmic prior, then combine pooled marker enrichment with a
    leave-one-out, same-cell/same-prior neighborhood consensus from
    `X_dgi_gcn_metaspot`. Fine labels compete only inside that compartment.
    Insufficient evidence falls back to `Nucleus` or `Cytosol`, keeping
    `subcellular_annotation` strictly within the established seven classes;
    uncertainty remains explicit in `subcellular_fine_state`,
    `subcellular_fine_supported`, and the fallback-reason columns.
  - `metadomain`: make and transfer the complete RAP-X label at MetaDomain
    resolution.
  - `leiden`: make one RAP-X decision per annotation Leiden cluster.
- `decision_mode` (`independent|legacy`): Label decision protocol. Default:
  `independent`.
- `two_stage` (bool): Optional routed two-stage enrichment. Default: `false`.
- `spatial_channel` (bool): Include the RAP-X spatial channel. Default: `true`.
- `metaspot_compartment_correction` (bool): Select the native-MetaSpot
  compartment-consistent track as production output. The local side uses
  `r_p50`; corrected and uncorrected tracks remain available. Default: `true`.
- `metaspot_pool_rep` (str), `metaspot_pool_k` (int): Native MetaSpot DGI
  representation and same-cell/same-compartment neighborhood size used by
  hybrid MetaSpot annotation. Defaults: `X_dgi_gcn_metaspot`, `8`.
- `metaspot_dgi_consensus_weight` (float): weight used in the audit-only fused
  score and the DGI evidence branch. It does not select the focal local
  candidate and does not enter the strict marker/count/enrichment/q-value
  gate. DGI can support an explicitly configured hard gate, exploratory
  rescue, cell-coverage rescue, and OMM-coherence rescue; a MetaSpot never
  votes for itself. Packaged default: `0.6`.
- `metaspot_dgi_seed_min_marker_transcripts` (int),
  `metaspot_dgi_seed_min_marker_genes` (int),
  `metaspot_dgi_seed_min_log2_enrichment` (float), and
  `metaspot_dgi_min_informative_neighbors` (int): define which unpooled
  MetaSpot marker calls may seed the independent DGI consensus and how many
  such neighbors are required. Packaged defaults: `1`, `1`, `0.05`, `2`.
- `metaspot_dgi_exploratory` (bool): allow a candidate that passes all marker
  amount/enrichment/separation gates, but not the BH threshold, to enter a
  separately tagged exploratory tier when leave-one-out DGI agreement is
  strong. Strict calls remain FDR-controlled. Default: `true`.
- `metaspot_dgi_exploratory_min_support` (float) and
  `metaspot_dgi_exploratory_min_margin` (float): minimum neighbor-vote share
  and lead over the next label for that exploratory tier. Packaged defaults:
  `0.55`, `0.05`.
- `metaspot_cell_coverage_rescue` (bool): recover a missing Nucleolus, Lamina,
  Nuclear Pore, or ER label only when the same cell has coherent multi-gene
  evidence in the matching broad compartment and one focal MetaSpot also
  passes its local marker and leave-one-out DGI gates. This is a bounded
  discovery rescue, not a per-cell label quota. Labels are evaluated against
  their own evidence arrays, so a valid secondary fine structure is not lost
  merely because another marker label had the largest local enrichment.
  Packaged default: `true`.
- `metaspot_cell_coverage_min_marker_transcripts` (int),
  `metaspot_cell_coverage_min_marker_genes` (int),
  `metaspot_cell_coverage_min_log2_enrichment` (float), and
  `metaspot_cell_coverage_max_binomial_p` (float): cell-level presence gates
  for that rescue. The packaged discovery profile uses two distinct marker
  genes plus enrichment above `-0.5`; its binomial p value is retained for
  audit rather than used as a significance veto because low-depth cells make
  that test underpowered. Packaged defaults: `2`, `2`, `-0.5`, and `1.0`.
- `metaspot_cell_coverage_min_local_marker_transcripts` (int),
  `metaspot_cell_coverage_min_local_marker_genes` (int), and
  `metaspot_cell_coverage_min_local_log2_enrichment` (float): focal MetaSpot
  anchor gates for label-wise coverage rescue. The cell-level multi-gene gate
  remains separate. Packaged defaults: `2`, `2`, and `0.0`.
- `metaspot_cell_coverage_min_dgi_support` (float),
  `metaspot_cell_coverage_min_dgi_margin` (float), and
  `metaspot_cell_coverage_max_per_label` (int): local DGI placement gates and
  the maximum number of rescued MetaSpots per missing label per cell. Because
  DGI seeds use the winning local label, the label-wise secondary-structure
  branch requires an informative neighborhood but treats its label vote as
  audit-only; the two-gene focal marker gate remains mandatory. Packaged
  defaults: `0.0`, `-1.0`, and `1`.
- `metaspot_diminishing_return_selection` (bool) and
  `metaspot_diminishing_return_labels` (list): apply one shared post-selection
  guard to the configured fine structures after all strict, local, cell-level,
  hierarchical, and OMM branches converge. The strongest first locus is always
  retained and strict loci are never pruned. Packaged defaults: `false` and all
  five fine structures; the shipped discovery profile enables the guard.
- `metaspot_diminishing_return_log_penalty` (float) and
  `metaspot_diminishing_return_burden_penalty` (float), and
  `metaspot_diminishing_return_min_dgi_support` (float): raise the evidence
  required for each additional same-cell, same-structure locus according to
  `log1p(number already retained)` and the already assigned fraction of the
  matching nuclear or cytoplasmic transcript pool while keeping the repeat
  guard independent of the more permissive coverage-rescue DGI threshold.
  Packaged defaults: `0.10`, `0.50`, and `0.50`.
- `metaspot_hierarchical_cell_rescue` (bool): final Tier-B branch for a cell
  that still has no supported fine structure. It emits at most one fine label
  per callable cell and requires label-specific cell-level marker presence, a
  measured marker in the focal MetaSpot, and leave-one-out DGI localization.
  This is not a per-cell quota. Packaged default: `true`.
- `metaspot_hierarchical_min_cell_transcripts` (int) and
  `metaspot_hierarchical_min_metaspots` (int): define whether a cell has enough
  observations for Tier-B fine-structure calling. Cells below either threshold
  retain their broad Nucleus/Cytosol fallback. Packaged defaults: `200` and
  `20`.
- `metaspot_hierarchical_min_presence_score` (float),
  `metaspot_hierarchical_min_localization_score` (float), and
  `metaspot_hierarchical_min_joint_score` (float): thresholds for the bounded
  `[0, 1]` evidence scores. Presence combines cell-level marker counts, genes,
  enrichment, and binomial evidence; localization combines focal markers with
  DGI support; joint is their geometric mean. These values are audit scores,
  not calibrated probabilities. Packaged defaults: `0.55`, `0.55`, `0.60`.
- `metaspot_hierarchical_min_local_marker_transcripts` (int),
  `metaspot_hierarchical_min_local_marker_genes` (int),
  `metaspot_hierarchical_min_cell_marker_transcripts` (int), and
  `metaspot_hierarchical_min_cell_marker_genes` (int): non-negotiable measured
  marker floors for Tier-B emission. Packaged defaults are all `1`.
- `metaspot_hierarchical_min_dgi_support` (float) and
  `metaspot_hierarchical_min_dgi_margin` (float): non-negotiable DGI floors.
  The slightly negative margin permits a marker-coherent secondary structure
  to survive winner-takes-all competition, while still rejecting strong DGI
  conflict. Packaged defaults: `0.35` and `-0.25`.
  The output audit field `subcellular_metaspot_hierarchical_cell_status`
  records `not_callable`, `already_has_fine_structure`,
  `no_anchored_candidate`, `score_below_threshold`, or `selected`; the
  corresponding best anchored label and evidence scores are retained in
  `subcellular_metaspot_hierarchical_best_*` columns.
- `metaspot_require_dgi_support` (bool), `metaspot_min_dgi_support` (float),
  and `metaspot_min_dgi_margin` (float): optionally make DGI agreement a hard
  support gate instead of only a scored evidence branch. Defaults: `false`,
  `0.0`, `0.0`.
- `metaspot_min_marker_transcripts` (int), `metaspot_min_marker_genes` (int),
  `metaspot_min_log2_enrichment` (float), `metaspot_min_score_gap` (float):
  minimum local evidence before a subtype can replace its broad prior.
  Packaged defaults: `1`, `1`, `0.05`, `0.03`.
- `metaspot_subtype_fdr` (float), `metaspot_exploratory_max_q` (float): strict
  and exploratory label-wise BH thresholds. Packaged defaults: `0.15`, `0.70`.
- `metaspot_background_prior_strength` (float): pseudocount strength used when
  comparing pooled marker proportions with compartment-wide background.
  Packaged default: `1.0`.
- `metaspot_label_thresholds` (mapping): exact per-label overrides for
  `min_marker_transcripts`, `min_marker_genes`, `min_log2_enrichment`,
  `min_score_gap`, `subtype_fdr`, `exploratory_max_q`, `min_dgi_support`, and
  `min_dgi_margin`. Unknown labels or fields are rejected rather than ignored.
- `metaspot_require_omm_confirmation` (bool): when the independent OMM model
  is identifiable, require its domain/local posterior audit to confirm a local
  OMM candidate. Default: `true`.
- `metaspot_omm_unidentifiable_policy` (`unresolved|strict_local`): behavior
  when independent OMM confirmation is unavailable. `unresolved` keeps strict
  local OMM candidates in audit columns and falls back to `Cytosol`;
  `strict_local` permits strict marker-only OMM calls in the refinement track.
  Packaged default: `strict_local`; the packaged `omm_policy: audit` still
  prevents that refinement track from replacing the primary public label.
- `metaspot_omm_cell_coherent_rescue` (bool): use pooled, multi-gene OMM
  evidence at the cell level to localize a bounded number of OMM MetaSpots.
  Each emitted locus must still pass the configured local marker amount,
  enrichment, OMM-over-ER/Cytosol separation, and leave-one-out DGI gates.
  Packaged default: `true`.
- `metaspot_omm_cell_min_marker_transcripts` (int),
  `metaspot_omm_cell_min_marker_genes` (int),
  `metaspot_omm_cell_min_log2_enrichment` (float), and
  `metaspot_omm_cell_max_binomial_p` (float): pooled cell-level OMM presence
  gates. Packaged defaults: `6`, `3`, `0.25`, and `1.0`; the low-depth
  binomial p-value is therefore retained for audit rather than used as a veto
  after the marker, local-separation, and DGI gates pass.
- `metaspot_omm_cell_exploratory_max_binomial_p` (float): relaxed cell-level
  p-value used only for the OMM sensitivity/audit tier. The local marker,
  OMM-over-ER/Cytosol, and DGI gates remain identical, and these rows never
  overwrite `subcellular_annotation`. Packaged default: `1.0`.
- `metaspot_omm_local_min_marker_transcripts` (int) and
  `metaspot_omm_local_min_marker_genes` (int): minimum focal evidence used to
  place OMM within an eligible cell. Packaged defaults are `1` and `1` because
  MetaSpots are small; this local anchor cannot emit OMM without the cell-level
  multi-gene, local enrichment/separation, and DGI gates.
- `metaspot_omm_local_min_dgi_support` (float) and
  `metaspot_omm_local_min_dgi_margin` (float): minimum leave-one-out neighbor
  agreement used by the cell-coherent OMM localization path. Packaged
  defaults: `0.50`, `0.0`.
- `metaspot_omm_cell_max_per_cell` (int): maximum number of already
  evidence-qualified OMM loci selected in one cell. It is a cap, not a quota;
  cells without coherent OMM evidence still receive no OMM call. Packaged
  default: `5`.
- `omm_policy` (`strict|balanced|audit|off`): Public OMM selector.
  - `strict`: require a confirmed OMM MetaDomain and local posterior at or
    above the internal high threshold.
  - `balanced`: also allow a locally high posterior to resolve an ER/OMM-shared
    MetaDomain to OMM. Unresolved shared evidence remains in audit columns and
    falls back to its ER/Cytosol competitor in the primary seven-class output.
  - `audit`: compute OMM refinement columns without replacing the primary
    annotation.
  - `off`: skip MetaSpot OMM refinement.
  Packaged default: `audit`.
- `cluster_rep` (str): Preferred annotation representation:
  `X_dgi_gcn|X_dgi_sage|X_expr|X_pca|auto`. Default: `X_dgi_gcn`.
- `save_plots` (bool): Write annotation diagnostics. Default: `true`.

### Annotation expert overrides

Existing detailed keys remain backward compatible, including
`independent_*`, `marker_*`, `guard_mode`, `subtype_fdr`,
`min_effective_markers`, `null_*`, `apex_omm_*`,
`omm_metaspot_posterior_*`, `omm_metaspot_min_informative_transcripts`,
`omm_metaspot_shrinkage_count`, clustering/UMAP settings, and legacy
`single_stage_*`, `nuclear_pore_*`, `lamina_*`, and `omm_guard` settings.

A legacy configuration without `omm_policy` keeps its former behavior:
`omm_metaspot_refinement: false` maps to `off`;
`omm_metaspot_apply_to_primary: false` maps to `audit`; otherwise it maps to
`strict`. When `omm_policy` is explicitly present, it is authoritative.

## module1 (Geometry features)
- `knn_k` (int): Neighborhood size for geometry context. Default: `32`.
- `include_directional` (bool): Add directional features (sin/cos). Default: `true`.
- `use_input_nucleus_flags` (bool): If `true`, reuse transcript-level
  `overlaps_nucleus`; if `false`, recompute nuclear membership from the
  supplied nucleus polygon. Benchmark configurations should normally use
  `false` to avoid treating a simulated truth column as a model input.
  Packaged default: `false`.
- `winsor_perc_low`/`winsor_perc_high` (int, 0–100): Winsorization percentiles. Default: `1`, `99`.

## module2 (Composition embeddings)
- `top_g` (int): Vocabulary size (top genes). Default: `1024`.
- `bow_knn_k` (int): Neighborhood size for BoW. Default: `12`.
- `max_rows_per_cell_for_svd` (int): Per-cell TF rows cap for SVD sample. Default: `256`.
- `max_svd_rows` (int): Global TF rows cap for SVD sample. Default: `100_000`.
- `svd_n_components` (int): Latent dimensionality. Default: `24`.
- `bow_weight_mode` (`gaussian|uniform`): Neighbor weighting. Default: `gaussian`.
- `use_tfidf` (bool), `tf_norm` (bool): SVD weighting and TF normalization. Default: `true`, `true`.
- `parallel_chunk_cells` (int): Grouping factor for parallel TF blocks. Default: `8`.
- `stream_local_embeddings` (bool): Compute transcript-level neighborhood expression inside Module 3 one cell at a time instead of storing dense embedding columns for all transcripts. Default: `true`.

## module3 (Per-cell MetaSpot segmentation)
- `parallel_chunk_cells` (int): Number of cells dispatched at a time through
  Module 3's persistent worker pool. Default: `256`; this controls scheduling
  overhead and does not change the requested MetaSpot count.
- `parallel_n_jobs` (int): Module 3 worker count, bounded by `runtime.n_jobs`
  and the process CPU affinity. Set `0` to use the conservative eight-worker
  automatic cap.
  Nested BLAS/OpenMP pools are limited to one thread during this block.
  Default: `12`.
- `hdbscan_min_cluster_size` (int): Minimum subcluster size. Default: `30`.
- `hdbscan_allow_single_cluster` (bool): Allow single-cluster HDBSCAN. Default: `true`.
- `hdbscan_min_samples_frac` (float): Fraction of min_cluster_size for `min_samples`. Default: `0.2`.
- `kmeans_min_k`/`kmeans_max_k` (int): Bounds for KMeans fallback. Default: `2`, `50`.
- `merge_min_size` (int): Merge clusters smaller than this threshold. Default: `20`.
- `enforce_spatial_connectivity` (bool): Build every final `metaspot` as a connected XY region using spatially constrained Ward clustering.
- `partition_by_nucleus` (bool): Allocate the cell-level MetaSpot budget
  across nucleus and cytoplasm before clustering. When enabled, a MetaSpot
  cannot cross the nuclear boundary. Default: `false`.
- `target_sites_per_metaspot` (int|null): Target number of unique XY sites per MetaSpot. When set, this determines the requested per-cell K before connected-component merging and bypasses HDBSCAN as the primary segmenter. Packaged default: `20`.
- `min_metaspots_per_cell` (int): Lower bound for the requested per-cell K. Packaged default: `10`.
- `max_metaspots_per_cell` (int|null): Upper bound for the per-cell K. Default: `100`.
- `spatial_connectivity_k` (int): Maximum nearby XY neighbors considered when constructing the local spatial graph. Default: `8`.
- `spatial_radius_quantile` (float): Quantile of per-site nearest-neighbor distances used to estimate native point spacing. Default: `0.9`.
- `spatial_radius_factor` (float): Multiplier applied to the estimated point spacing for the graph radius. Default: `1.5`.
- `spatial_smoothing_iterations` (int): Local majority-smoothing passes used only if constrained Ward clustering cannot run. Default: `1`.
- `spatial_smoothing_min_fraction` (float): Minimum weighted neighbor fraction required to change a site's preliminary label in that fallback. Default: `0.6`.
- `spatial_min_component_size` (int): Connected components smaller than this transcript count are merged across their strongest shared boundary. Default: `5`.
- Module 3 writes `module3_metaspot_segmentation_audit.csv` when compact
  diagnostics are enabled. It reports actual/requested MetaSpot counts, size
  distributions, and nuclear-mixing rates for every cell.

- `spatial_compactness` (float): Weight of standardized XY coordinates in the spatially constrained Ward clustering. Larger values make MetaSpots more compact; `0` uses features only while still restricting merges to spatial neighbors. Default: `1.0`.
- `hdbscan_*`, `target_sub_size`, and `kmeans_*`: Legacy fallback controls used when `target_sites_per_metaspot` is null or spatial connectivity is disabled.

## Fast exploratory overlay

`config/params.fast.yaml` is a partial configuration merged over the packaged
defaults by the CLI:

```bash
cellscoper ... --config config/params.fast.yaml
```

It opts into PyNNDescent for the exact-scale expression graph, caps CA-DMGI at
120 epochs, uses two consensus seeds, and disables UMAP/diagnostic figures.
This is intended for iteration and can change graph edges, embeddings, and
MetaDomains. The default `params.yaml` remains the production profile.

## module5 (Meta-domain clustering)
- `block_size` (int): Chunk size for processing. Default: `50000`.
- `clustering_method` (`consensus_leiden|minibatch_kmeans`): MetaDomain
  clustering backend. The packaged default is `consensus_leiden`: it builds a
  mutual-kNN graph in `X_dgi_gcn_metaspot`, runs every configured resolution
  and random seed, chooses the best stable candidate subject to size and
  cross-cell recurrence constraints, and consolidates assignments using
  sparse edge-level co-association. `minibatch_kmeans` is retained for legacy
  reproduction and ablation.
- `consensus_neighbors` (int): Neighbors used for the mutual-kNN graph.
  Default: `20`.
- `consensus_resolutions` (list[float]) and `consensus_seeds` (list[int]):
  resolution grid and independent Leiden seeds. Packaged defaults:
  `[1, 2, 4, 6, 8]` and `[17, 42, 89]`.
- `consensus_min_average_size`/`consensus_max_average_size` (float): feasible
  average MetaDomain-size interval. Packaged defaults: `100/800` MetaSpots.
- `consensus_min_recurrence_cells` (int),
  `consensus_max_small_cluster_fraction` (float), and
  `consensus_max_nonrecurrent_fraction` (float): reject resolutions dominated
  by tiny or single-cell-specific clusters. Defaults: `2`, `0.10`, `0.10`.
- `consensus_edge_agreement` (float): fraction of seed runs that must place
  both endpoints of a mutual-kNN edge in the same cluster before that edge is
  retained in the consensus graph. Default: `2/3`.
- `consensus_min_stability` (float) and `consensus_stability_tolerance`
  (float): require a cross-seed ARI of at least `0.55`, then retain the finest
  feasible candidate no more than `0.10` below the best feasible ARI. This
  avoids choosing a biologically uninformative coarse partition for a tiny
  numerical stability advantage.
- Consensus evidence is stored in
  `adata1.uns["module5_metadomain_clustering"]["consensus_by_block"]` and, when
  diagnostics are enabled, in `metadomain_consensus_leiden_diagnostics.png`
  plus `module5_consensus_leiden_diagnostics.*`.
- `w_gc`/`w_expr`/`w_sp` (float): Feature weights (geometry, expression, spatial). Packaged default: `0.4/0.4/0.2`.
- `use_gc`/`use_expr`/`use_spatial_feats`/`use_decell`/`use_cosine` (bool): Feature toggles. Defaults: `true/true/true/true/true` (spatial feats default may vary by build).
- `speed_preset` (`ultra|fast|balanced`): Internal speed/accuracy presets. Default: `ultra`.
- `enable_cap` (bool): Per-cell cap enabled. Default: `true`.
- `min_domain_size` (int): Clusters smaller than this are merged into the nearest sufficiently sized center after final assignment. Default: `5`.
- Legacy MiniBatchKMeans K estimation (used only when
  `clustering_method: minibatch_kmeans`):
  - `estimate_k` (`auto|sampled_knee|legacy_auto|fixed`): Strategy. `auto` and `sampled_knee` are aliases for bounded, cell-stratified MiniBatchKMeans search. `legacy_auto` restores the historical count-only heuristic. Default: `auto`.
  - `fixed_k` (int): Used when `estimate_k=fixed`. Default: `1024`.
  - `auto_k_min`/`auto_k_max` (int): Hard candidate bounds for sampled-knee selection. Defaults: `20/8000`.
  - `auto_k_min_average_size`/`auto_k_max_average_size` (int): Candidate K values are constrained so the expected average MetaDomain contains between these many MetaSpots. Defaults: `20/500`.
  - `auto_k_n_candidates` (int): Number of logarithmically spaced K candidates. Default: `8`.
  - `auto_k_train_sample`/`auto_k_validation_sample` (int): Maximum cell-stratified train and held-out validation sizes. Defaults: `50000/3000`.
  - `auto_k_sample_seed` (int), `auto_k_seeds` (list[int]): Reproducible sampling seed and MiniBatchKMeans stability seeds. Defaults: `2026` and `[42, 43]`.
  - `auto_k_stability_min` (float): Minimum held-out adjusted Rand agreement between seeds. If the distortion-knee candidate fails, selection backs off to a smaller feasible K. Default: `0.4`.
  - `auto_k_max_small_cluster_fraction` (float): Maximum sampled/full-evaluation point fraction assigned to clusters expected to fall below `min_domain_size`. Default: `0.01`.
  - `auto_k_min_train_per_cluster` (int): Candidate K cannot exceed the training sample size divided by this value. Default: `10`.
  - `auto_k_batch_size`/`auto_k_max_iter` (int): Search-only MiniBatchKMeans controls. Defaults: `2048/30`.
  - `auto_k_full_eval_max` (int): Candidate occupancy is evaluated on all MetaSpots up to this size and on the bounded sample above it. Default: `100000`.
  - `target_sub_per_cluster`, `target_cluster_per_cell`, and `K_min`: Legacy controls used only by `estimate_k=legacy_auto`. `K_max` remains a backward-compatible upper bound when `auto_k_max` is omitted.
  - The selected K and all candidate distortion/stability/occupancy diagnostics are written to `adata1.uns["module5_metadomain_clustering"]["k_selection"]`.
  - When intermediate figures are enabled, Module 5 writes `metadomain_k_selection_diagnostics.png` plus `module5_k_selection_diagnostics.*` with the plotted candidate-level evidence.
- Nucleus/cytoplasm split:
  - `split_by_nucleus` (bool), `nucleus_threshold` (float).
- Per-cell cap:
  - `cap_ratio_base` (float), `cap_min_keep` (int), `cap_margin_tau` (float), `cap_max_frac_soft` (float).

## module6 (Aggregation to adata2)
- `h5ad_compression_override` (bool): Respect IO compression override for `adata2`. Default: `true`.
- `save_shape_summary` (bool): Save `module6_adata2_shape.*`. Default: `true`.

After annotation, enabled intermediate figures also include
`metadomain_annotation_quality.png`. This quantitative diagnostic reports
MetaDomain sizes, leave-one-out within-domain
embedding coherence, nuclear/cytoplasmic purity, domain-to-final label
transfer, and MetaSpot-weighted label composition. Its source tables are
saved as `metadomain_quality_by_domain.*`,
`metadomain_annotation_transfer.*`, and
`metadomain_label_composition.*`.

The final MetaSpot-resolution result is written to
`metaspot_annotation_result.png`. Its UMAP panels compare inherited
MetaDomain evidence with `subcellular_annotation` (also mirrored in
`subcellular_fine_annotation`); the lower panels show
label transfer and exact decision provenance. The UMAP is diagnostic context,
not evidence of correctness. Exact values are saved in
`metaspot_annotation_summary.*`, `metaspot_fine_annotation_summary.*`,
`metaspot_fine_decision_provenance.*`, `metaspot_annotation_updates.*`, and
`metaspot_annotation_umap_coordinates.*` when diagnostic table writing is
enabled. Transcript-level seven-class coverage is written once as
`final_annotation_proportions.*`; the compatibility alias
`subcellular_fine_annotation` does not create a duplicate summary. Hybrid
MetaSpot mode additionally writes
`metaspot_local_annotation_scores.*`, including the broad parent, local
marker-only candidate, fused marker/DGI candidate, DGI support and margin,
pooled marker counts, enrichment, binomial p-value, label-wise BH q-value,
applied label-specific thresholds, final tier, OMM confirmation status,
decision provenance, and explicit fallback reason for every MetaSpot. For very
large datasets the UMAP display is capped at 50,000
MetaSpots using cell-and-label-stratified sampling; the summary calculations
always use all MetaSpots.

RAP-X confirmatory evidence is stored in `rapx_annotation_scores.*`,
`rapx_metadomain_evidence.*`, `rapx_candidate_to_strict.*`, and
`rapx_candidate_evidence_tier.*`. These are audit-only tables. The pipeline
does not emit a strict-call UMAP because strict `Unknown` denotes absence of
confirmatory FDR support rather than the final biological label. Each run also
writes `OUTPUT_GUIDE.md` at the output root to separate production files,
recommended result figures, method QC, audit tables, and checkpoints.

Every CLI run writes `run_manifest.json` with status, effective configuration
hash, input/reference identity, source-tree identity, runtime overrides, and
actual output paths. Annotation failures are fatal when annotation is enabled;
the pipeline will not label an unannotated or stale result as successful.

## module7 (Graph construction and expression latent)
- `graph_unit` is an internal compatibility field derived from
  `module8.dgi_unit`. New configurations should leave it as `auto`.
- `scvi_epochs` (int), `scvi_n_latent` (int), `scvi_lr` (float), `scvi_batch_size` (int): SCVI training params.
- `rep_priority` (list[str]): Representation preference for annotation: `X_dgi_gcn`, `X_dgi_sage`, `X_expr`, `X_pca`.
- `neighbors.k_expr`/`neighbors.k_geom` (int): k for expression/geometry graphs. Default: `15/15`.
- `alpha_expr`/`alpha_geom` (float): Fusion weights. Default: `0.5/0.5`.
- `same_cell_beta` (float): Bias for same-cell edges. Default: `0.5`.
- `topk_prune` (int): Row-wise top-k pruning of edges. Default: `32`.
- `edge_norm` (`rw|gcn|none`): Edge normalization. Default: `rw`.
- `feature_scale` (`standard|none`): Feature scaling mode. Default: `standard`.
- `knn_backend` (`auto|exact|pynndescent|faiss_hnsw|faiss_ivfpq`): kNN
  backend selected by the scale-aware execution plan. In the exact/full-batch
  profile, `auto` keeps the historical exact sklearn graph; an explicit
  `pynndescent` or `faiss_hnsw` opts into approximate graph membership while
  retaining full-batch DGI. `faiss_ivfpq` remains streaming-only.
- In MetaSpot mode, `neighbors.k_expr` and `neighbors.k_spatial` control two
  separately retained relation graphs:
  `obsp["metaspot_expr_connectivities"]` and
  `obsp["metaspot_spatial_connectivities"]`.
- `metaspot_expr_cross_cell_only` (bool): remove same-cell edges from the
  expression relation after expression-neighbor construction. The packaged
  CA-DMGI configuration sets this to `true`, so cross-cell reproducibility and
  within-cell spatial continuity cannot collapse into the same edge type.
- `metaspot_expr_n_components` (int): expression SVD/PCA dimension used for
  expression-neighbor discovery. Packaged default: `32`.
- `metaspot_use_geometry_features` (bool): append robust MetaSpot geometry to
  node features. Relation membership still comes from expression or spatial
  adjacency; geometry is not used as an annotation label.

## module8 (DGI embeddings)
- `dgi_unit` (`metadomain|metaspot`): Single switch controlling the DGI
  observation unit and module order. Packaged default: `metaspot`.
  - `metadomain`: run M5/M6 first, then train DGI on `adata2`. This is the
    scalable alternative for large datasets.
  - `metaspot`: train DGI on `adata1`, use
    `X_dgi_gcn_metaspot` for M5 clustering, and pool it into `adata2`.
    `module7.max_full_batch_nodes` is a hard safety limit; the current
    implementation does not silently fall back or pretend that an oversized
    MetaSpot graph completed. Its packaged default is `200000`; raising it
    only disables the guard and does not reduce full-batch memory cost.
  `annotation.unit: metaspot` requires `dgi_unit: metaspot`; incompatible
  configurations fail during loading before a costly pipeline run begins.
- `dgi_framework` (`cell_aware_dmgi|standard_dgi`): training objective. The
  packaged default, `cell_aware_dmgi`, uses two relation-specific residual GCN
  encoders: the cross-cell expression graph and the within-cell spatial graph.
  Each branch optimizes a local Infomax discriminator against features
  permuted within the same owning cell. Thus no real MetaSpot is declared a
  biological negative. A bounded node-wise gate combines the two embeddings
  into the public compatibility key `X_dgi_gcn_metaspot`; branch embeddings
  are retained as `X_dmgi_expression_metaspot` and
  `X_dmgi_spatial_metaspot`. `standard_dgi` retains the historical single-graph
  implementation for ablation and old configurations.
- `dmgi_expression_graph_key`/`dmgi_spatial_graph_key` (str): explicit
  relation-graph keys. Defaults: `metaspot_expr_connectivities` and
  `metaspot_spatial_connectivities`.
- `dmgi_expression_weight`/`dmgi_spatial_weight` (float): relation-specific
  Infomax weights. Both default to `1.0`.
- `dmgi_consensus_weight` (float): alignment weight between the gated
  consensus and both relation embeddings. Default: `0.1`.
- `dmgi_gate_floor` (float): lower bound for either relation's node-wise gate;
  `0.1` constrains expression and spatial contributions to `[0.1, 0.9]` and
  prevents silent branch collapse.
- `dmgi_fused_corruption_pass` (bool): propagate the positive and within-cell
  corrupted views together so each relation/layer scans its sparse edges once
  instead of twice. Default: `true`. With dropout disabled this is numerically
  identical to the separate-pass implementation; with dropout enabled it
  preserves the same objective and independent dropout sampling.
- `dmgi_early_stopping`, `dmgi_min_epochs`, `dmgi_patience`, and
  `dmgi_min_delta`: EMA-based stopping controls. The packaged configuration
  uses `true`, `40`, `15`, and `1e-4`; `dgi_epochs: 200` is the maximum rather
  than a mandatory run length.
- `dgi_epochs` (int): Maximum training epochs. Packaged default: `200`.
- `enable_dgi_sage` (bool): Train SAGE variant in addition to GCN. Default: `false`.
- `dgi_encoder` (`residual|plain`): Encoder architecture. `residual` uses an
  input projection, initial-feature injection, residual graph-convolution
  blocks, and optional LayerNorm. `plain` preserves the historical sequential
  GCN. Default: `residual`.
- `dgi_layers` (int): Number of graph message-passing layers. This is also the
  graph receptive-field depth; increasing it is not equivalent to adding only
  feature-transform capacity. Default: `2`.
- `dgi_hidden`/`dgi_out` (int): Hidden and output dimensions. Default:
  `64/32`. To reproduce embeddings produced before the residual-encoder
  update, set `dgi_encoder: plain`, `dgi_layers: 2`, `dgi_dropout: 0.0`, and
  `dgi_out: 64` together.
- `dgi_dropout` (float): Dropout used by the encoder. Default: `0.1`.
- `dgi_residual_alpha` (float): Fraction of the initial projected features
  injected into every residual graph block. Default: `0.1`.
- `dgi_layer_norm` (bool): Apply LayerNorm after residual graph blocks.
  Default: `true`.
- `dgi_device` (`auto|cpu|cuda`): execution device. `auto` selects CUDA only
  when the installed PyTorch build and driver expose an available device.
  Explicit `cuda` fails fast instead of silently running on CPU.
- `dgi_cpu_threads` (int): PyTorch intra-op threads used by DGI on CPU. `0`
  auto-selects up to 32 CPUs from the process affinity; positive values are
  explicit overrides. This is independent from preprocessing `blas_threads`.
- `dgi_sparse_layout` (`auto|csr|coo`): sparse relation-graph storage for
  CA-DMGI. `auto` uses CSR with compact native SciPy indices; `coo` is a
  compatibility fallback.
- `dgi_tf32` (bool): allow TF32 dense matrix multiplication on supported CUDA
  hardware. Sparse graph values and model outputs remain float32.
- `dgi_corruption_mode` (`auto|global|same_cell`): Negative-view feature
  corruption. In MetaSpot mode, `auto` uses a non-self permutation within each
  `cell_id`, so the discriminator must learn subcellular organization rather
  than relying on easy between-cell expression differences. In MetaDomain
  mode, `auto` uses the historical global permutation. This switch belongs to
  the `standard_dgi` fallback; CA-DMGI always records same-cell corruption.
- `dgi_local_loss_weight` (float): Weight of an unsupervised local edge
  contrastive loss added to standard DGI. Positive pairs are graph edges;
  negative pairs are non-edges from the same cell when `cell_id` is available.
  Packaged default: `0.0`, because a same-cell non-edge may still represent the
  same biological structure. Set it explicitly only for the legacy ablation.
- `dgi_local_temperature` (float): Cosine-logit temperature for the local
  contrastive term. Default: `0.2`.
- `dgi_local_samples` (int): Maximum positive edges sampled once per training
  run for the local term. Default: `65536`.
- `dgi_local_graph_key` (`auto` or an `obsp` key): With `auto`, prefer
  `metaspot_spatial_connectivities` and otherwise use the main DGI graph.
  Default: `auto`.
- `dgi_cpu_backend` (`auto|torch_csr|pyg`): CPU GCN execution backend. `auto`
  uses the equivalent pre-normalized PyTorch CSR SpMM path on CPU and retains
  PyG on GPU for `standard_dgi`; `pyg` restores the legacy COO scatter
  implementation. CA-DMGI uses `dgi_sparse_layout` instead.
- `dgi_lr` (float), `dgi_weight_decay` (float), `dgi_clip_grad` (float): Optimizer settings.

The packaged configuration deliberately keeps two propagation layers. CA-DMGI
is fully unsupervised: it uses expression, geometry, cell ownership and spatial
adjacency, but never final subcellular labels. Model selection should therefore
use cross-seed/cross-cell clustering stability, representation ablations and
downstream evidence support—not visual UMAP separation or training loss alone.
`module8_ca_dmgi_training.png` reports both relation losses and the gate
trajectory; exact histories, graph keys, corruption semantics and gate metrics
are stored under `adata1.uns["dgi_embeddings"]` and exported to
`module8_ca_dmgi_training_history.*`/`module8_ca_dmgi_summary.*`. To reproduce
the historical single-graph objective, set `dgi_framework: standard_dgi`,
`dgi_corruption_mode: global`, and `dgi_local_loss_weight: 0.0`.

`module5.cluster_rep`, `module6.pool_reps`, and `module7.graph_unit` are
materialized automatically in `effective_params.yaml`. Legacy configurations
that only set `module7.graph_unit` remain supported.

## runtime (parallelism)
- `n_jobs` (int): Preferred worker count for parallel sections (honored unless `CELLSCOPE_N_JOBS` env is set).
- `joblib_backend` (`threading|loky`): Backend for joblib blocks (env `CELLSCOPE_JOBLIB_BACKEND` still wins).
- `blas_threads` (int): Caps BLAS/OpenMP thread pools; applied at startup and optionally via `threadpoolctl`. CellScoper forces this to `1` when a multi-worker `threading` backend is active, because OpenBLAS is not safe under nested application threading.
- `numba_threads` (int): Import-time Numba/PyNNDescent thread ceiling. With a multi-worker `threading` backend, CellScoper caps this so `n_jobs * numba_threads` does not exceed the bundled OpenBLAS thread-metadata limit.
- `scanpy_n_jobs` (int): Scanpy neighbor-graph workers, capped at the resolved `numba_threads` value. `SCANPY_N_JOBS` remains an environment override but is subject to the same cap.
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
  omm_policy: balanced
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
  dgi_unit: metadomain
  dgi_epochs: 60
  enable_dgi_sage: false
```

MetaSpot-first DGI for a dataset below the configured full-batch node limit:
```yaml
module8:
  dgi_unit: metaspot
module7:
  max_full_batch_nodes: 200000
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
- Large runs: use `module8.dgi_unit: metadomain` when the expected number of
  MetaSpots exceeds `module7.max_full_batch_nodes`. Lowering epochs does not
  remove the full adjacency/feature memory requirement. MetaSpot-DGI above the
  limit requires a sampled or mini-batch graph implementation.
