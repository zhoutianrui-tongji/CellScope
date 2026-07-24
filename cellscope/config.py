import os
import yaml
import json
import hashlib
from typing import Any, Dict, Optional, Tuple


def _workspace_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _params_path() -> str:
    root = _workspace_root()
    return os.path.join(root, 'config', 'params.yaml')


def get_default_params_path() -> str:
    return _params_path()


def load_params_yaml(path: Optional[str] = None) -> Dict[str, Any]:
    """Load params YAML.

    If ``path`` is provided, load from that file; otherwise fall back to the
    ``CELLSCOPE_CONFIG`` env var (if set), then repo default ``config/params.yaml``.
    """
    if path:
        p = path
    else:
        env_p = os.environ.get("CELLSCOPE_CONFIG", "").strip()
        p = env_p if env_p else _params_path()
    try:
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


# Map config keys to pipeline stages for minimal re-run decisions.
# Only names that appear in `pipeline.stage_order` are relevant here.
_BUILD_STAGE_KEYS: Tuple[str, ...] = (
    # Module 1
    'module1.include_directional', 'module1.winsor_perc_low', 'module1.winsor_perc_high',
    # Module 2
    'module2.top_g','module2.bow_knn_k','module2.max_rows_per_cell_for_svd','module2.max_svd_rows','module2.svd_n_components','module2.bow_weight_mode','module2.use_tfidf','module2.tf_norm','module2.idf_mode','module2.parallel_chunk_cells','module2.svd_sample_cells','module2.source_from_point_df','module2.stream_local_embeddings',
    # Module 3
    'module3.hdbscan_min_cluster_size','module3.hdbscan_allow_single_cluster','module3.hdbscan_min_samples_frac','module3.target_sub_size','module3.n_feat_hdb','module3.kmeans_min_k','module3.kmeans_max_k','module3.use_k_cap','module3.merge_min_size','module3.hdbscan_max_points','module3.kmeans_train_size','module3.kmeans_batch_size','module3.small_backend_cells','module3.sequential_cells','module3.enforce_spatial_connectivity','module3.spatial_connectivity_k','module3.spatial_radius_quantile','module3.spatial_radius_factor','module3.spatial_smoothing_iterations','module3.spatial_smoothing_min_fraction','module3.spatial_min_component_size','module3.spatial_compactness','module3.target_sites_per_metaspot','module3.min_metaspots_per_cell','module3.max_metaspots_per_cell',
    # Module 5
    'module5.block_size',
    'module5.w_gc','module5.w_expr','module5.w_sp','module5.use_gc','module5.use_expr',
    'module5.speed_preset','module5.use_spatial_feats','module5.use_decell','module5.use_cosine','module5.enable_cap',
    'module5.estimate_k','module5.fixed_k',
    'module5.target_sub_per_cluster','module5.target_cluster_per_cell','module5.K_min','module5.K_max',
    'module5.split_by_nucleus','module5.nucleus_threshold',
    'module5.cap_ratio_base','module5.cap_min_keep','module5.cap_margin_tau','module5.cap_max_frac_soft',
    'module5.train_sample_per_block','module5.assign_chunk_size','module5.mbk_batch_size','module5.mbk_max_iter',
    # Module 6
    'module6.h5ad_compression_override','module6.save_shape_summary','module6.preserve_raw_counts',
    # Module 7
    'module7.scvi_epochs','module7.scvi_n_latent','module7.scvi_lr','module7.scvi_batch_size',
    'module7.neighbors','module7.neighbors.k_expr','module7.neighbors.k_geom',
    'module7.alpha_expr','module7.alpha_geom','module7.same_cell_beta','module7.topk_prune','module7.edge_norm','module7.feature_scale',
    # Module 8
    'module8.dgi_epochs','module8.enable_dgi_sage','module8.dgi_hidden','module8.dgi_out','module8.dgi_lr','module8.dgi_weight_decay','module8.dgi_clip_grad','module8.amp',
)

STAGE_PARAM_MAP: Dict[str, Tuple[str, ...]] = {
    'load_inputs': (
        'io.final_format', 'io.h5ad_compression', 'io.lite_intermediate', 'io.resume_from',
    ),
    'preprocess': (),
    'build_anndata': _BUILD_STAGE_KEYS,
    'annotate': (
        'annotation.enable','annotation.method','annotation.unit','annotation.decision_mode','annotation.independent_mixed_score_delta','annotation.independent_unknown_label','annotation.independent_geometry_min_abs_r','annotation.independent_assignment_mode','annotation.independent_exploratory_max_q','annotation.independent_exploratory_min_channel_support','annotation.independent_exploratory_min_score_gap','annotation.independent_subtype_parent_margin','annotation.independent_omm_competitor_margin','annotation.two_stage','annotation.spatial_channel','annotation.low_conf_policy','annotation.low_conf_relaxed_z','annotation.low_conf_relaxed_q','annotation.marker_set_target','annotation.marker_min_weight','annotation.marker_min_margin','annotation.gold_column_normalize','annotation.min_channel_support','annotation.enforce_geometry_consistency','annotation.geometry_quantile','annotation.geometry_channel_weight','annotation.guard_mode','annotation.subtype_fdr','annotation.min_effective_markers','annotation.null_permutations','annotation.null_seed','annotation.single_stage_compartment_score_weight','annotation.single_stage_compartment_score_tau','annotation.single_stage_compartment_guard','annotation.single_stage_compartment_min_abs_r','annotation.single_stage_compartment_max_score_gap','annotation.single_stage_compartment_min_z','annotation.single_stage_compartment_max_q','annotation.single_stage_compartment_min_channel_support','annotation.nuclear_pore_guard','annotation.nuclear_pore_min_margin','annotation.nuclear_pore_non_nuclear_top2_min_margin','annotation.nuclear_pore_max_q','annotation.nuclear_pore_min_channel_support','annotation.nuclear_pore_fallback_label','annotation.lamina_guard','annotation.lamina_min_margin','annotation.lamina_non_nuclear_top2_min_margin','annotation.lamina_max_q','annotation.lamina_min_channel_support','annotation.lamina_boundary_quantile','annotation.lamina_fallback_label','annotation.omm_guard','annotation.omm_min_margin','annotation.omm_max_q','annotation.omm_min_channel_support','annotation.omm_fallback_label','annotation.apex_omm_specificity_filter','annotation.apex_omm_min_log2fc','annotation.apex_omm_min_delta','annotation.omm_metaspot_refinement','annotation.omm_metaspot_apply_to_primary','annotation.omm_metaspot_ambiguous_policy','annotation.omm_metaspot_min_informative_transcripts','annotation.omm_metaspot_shrinkage_count','annotation.omm_metaspot_posterior_high','annotation.omm_metaspot_posterior_low','annotation.omm_refinement_schema_version','annotation.rnalocate_augment_er','annotation.rnalocate_species','annotation.use_manual_markers','annotation.hvg_n_top_genes','annotation.pca_n_comps','annotation.neighbors_k','annotation.umap_min_dist','annotation.umap_spread','annotation.umap_neighbors_key','annotation.leiden_resolution','annotation.compute_umap','annotation.save_umap_plot','annotation.save_plots','annotation.fast','annotation.cluster_rep','module7.rep_priority','module7.neighbors',
    ),
    'write_outputs': (
        'io.final_format','io.write_csv_copy',
    ),
}



# --- Module-level param maps for fine-grained resume within build_anndata ---
MODULE_PARAM_MAP: Dict[str, Tuple[str, ...]] = {
    # m1..m8 correspond to pipeline internal modules inside build_anndata
    'm1': (
        'module1.include_directional', 'module1.winsor_perc_low', 'module1.winsor_perc_high',
    ),
    'm2': (
        'module2.top_g','module2.bow_knn_k','module2.max_rows_per_cell_for_svd','module2.max_svd_rows','module2.svd_n_components','module2.bow_weight_mode','module2.use_tfidf','module2.tf_norm','module2.idf_mode','module2.parallel_chunk_cells','module2.svd_sample_cells','module2.source_from_point_df','module2.stream_local_embeddings',
    ),
    'm3': (
        'module3.hdbscan_min_cluster_size','module3.hdbscan_allow_single_cluster','module3.hdbscan_min_samples_frac','module3.target_sub_size','module3.n_feat_hdb','module3.kmeans_min_k','module3.kmeans_max_k','module3.use_k_cap','module3.merge_min_size','module3.hdbscan_max_points','module3.kmeans_train_size','module3.kmeans_batch_size','module3.small_backend_cells','module3.sequential_cells','module3.enforce_spatial_connectivity','module3.spatial_connectivity_k','module3.spatial_radius_quantile','module3.spatial_radius_factor','module3.spatial_smoothing_iterations','module3.spatial_smoothing_min_fraction','module3.spatial_min_component_size','module3.spatial_compactness','module3.target_sites_per_metaspot','module3.min_metaspots_per_cell','module3.max_metaspots_per_cell',
    ),
    # m4 builds adata1 from point features; no direct params, but depends on m1..m3
    'm5': (
        'module5.block_size','module5.w_gc','module5.w_expr','module5.w_sp','module5.use_gc','module5.use_expr','module5.speed_preset','module5.use_spatial_feats','module5.use_decell','module5.use_cosine','module5.enable_cap','module5.estimate_k','module5.fixed_k','module5.target_sub_per_cluster','module5.target_cluster_per_cell','module5.K_min','module5.K_max','module5.split_by_nucleus','module5.nucleus_threshold','module5.cap_ratio_base','module5.cap_min_keep','module5.cap_margin_tau','module5.cap_max_frac_soft','module5.train_sample_per_block','module5.assign_chunk_size','module5.mbk_batch_size','module5.mbk_max_iter',
    ),
    'm6': (
        'module6.h5ad_compression_override','module6.save_shape_summary','module6.preserve_raw_counts',
    ),
    # m7/m8 are build-time graph and DGI embedding steps
    'm7': (
        'module7.scvi_epochs','module7.scvi_n_latent','module7.scvi_lr','module7.scvi_batch_size','module7.neighbors','module7.neighbors.k_expr','module7.neighbors.k_geom','module7.alpha_expr','module7.alpha_geom','module7.same_cell_beta','module7.topk_prune','module7.edge_norm','module7.feature_scale',
    ),
    'm8': (
        'module8.dgi_epochs','module8.enable_dgi_sage','module8.dgi_hidden','module8.dgi_out','module8.dgi_lr','module8.dgi_weight_decay','module8.dgi_clip_grad','module8.amp',
    ),
}


def _get_by_path(cfg: Dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def stage_config_subset(cfg: Dict[str, Any], stage: str) -> Dict[str, Any]:
    keys = STAGE_PARAM_MAP.get(stage, ())
    sub: Dict[str, Any] = {}
    for k in keys:
        sub[k] = _get_by_path(cfg, k)
    return sub


def fingerprint_stage(cfg: Dict[str, Any], stage: str) -> str:
    sub = stage_config_subset(cfg, stage)
    try:
        payload = json.dumps(sub, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except Exception:
        payload = str(sub)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def module_config_subset(cfg: Dict[str, Any], module: str) -> Dict[str, Any]:
    keys = MODULE_PARAM_MAP.get(module, ())
    sub: Dict[str, Any] = {}
    for k in keys:
        sub[k] = _get_by_path(cfg, k)
    return sub


def fingerprint_module(cfg: Dict[str, Any], module: str) -> str:
    sub = module_config_subset(cfg, module)
    try:
        payload = json.dumps(sub, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except Exception:
        payload = str(sub)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def fingerprint_modules_set(cfg: Dict[str, Any]) -> Dict[str, str]:
    mods = ('m1','m2','m3','m5','m6','m7','m8')
    return {m: fingerprint_module(cfg, m) for m in mods}
