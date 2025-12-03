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


def load_params_yaml() -> Dict[str, Any]:
    """Load params YAML from `config/params.yaml`."""
    p = _params_path()
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
    'module1.knn_k', 'module1.include_directional', 'module1.winsor_perc_low', 'module1.winsor_perc_high',
    # Module 2
    'module2.top_g','module2.bow_knn_k','module2.max_rows_per_cell_for_svd','module2.max_svd_rows','module2.svd_n_components','module2.bow_weight_mode','module2.use_tfidf','module2.tf_norm','module2.parallel_chunk_cells',
    # Module 3
    'module3.hdbscan_min_cluster_size','module3.hdbscan_allow_single_cluster','module3.hdbscan_min_samples_frac','module3.target_sub_size','module3.n_feat_hdb','module3.kmeans_min_k','module3.kmeans_max_k','module3.use_k_cap','module3.merge_min_size',
    # Module 5
    'module5.block_size',
    'module5.w_gc','module5.w_expr','module5.w_sp','module5.use_gc','module5.use_expr',
    'module5.speed_preset','module5.use_spatial_feats','module5.use_decell','module5.use_cosine','module5.enable_cap',
    'module5.estimate_k','module5.fixed_k',
    'module5.target_sub_per_cluster','module5.target_cluster_per_cell','module5.K_min','module5.K_max',
    'module5.split_by_nucleus','module5.nucleus_threshold',
    'module5.cap_ratio_base','module5.cap_min_keep','module5.cap_margin_tau','module5.cap_max_frac_soft',
    # Module 6
    'module6.h5ad_compression_override','module6.save_shape_summary',
    # Module 7
    'module7.scvi_epochs','module7.scvi_n_latent','module7.scvi_lr','module7.scvi_batch_size',
    'module7.rep_priority','module7.neighbors','module7.neighbors.k_expr','module7.neighbors.k_geom',
    'module7.alpha_expr','module7.alpha_geom','module7.same_cell_beta','module7.topk_prune','module7.edge_norm','module7.feature_scale',
    # Module 8
    'module8.dgi_epochs','module8.enable_dgi_sage','module8.dgi_hidden','module8.dgi_out','module8.dgi_lr','module8.dgi_weight_decay','module8.dgi_clip_grad',
)

STAGE_PARAM_MAP: Dict[str, Tuple[str, ...]] = {
    'load_inputs': (
        'io.final_format', 'io.h5ad_compression', 'io.lite_intermediate', 'io.resume_from',
    ),
    'preprocess': (
        'ui.progress_style',
    ),
    'build_anndata': _BUILD_STAGE_KEYS,
    'annotate': (
        'annotation.enable','annotation.two_stage','annotation.spatial_channel','annotation.hvg_n_top_genes','annotation.pca_n_comps','annotation.neighbors_k','annotation.umap_min_dist','annotation.umap_spread','annotation.umap_neighbors_key','annotation.leiden_resolution','annotation.compute_umap','annotation.save_umap_plot','annotation.cluster_rep','module7.rep_priority','module7.neighbors',
    ),
    'write_outputs': (
        'io.final_format','io.write_csv_copy',
    ),
}



# --- Module-level param maps for fine-grained resume within build_anndata ---
MODULE_PARAM_MAP: Dict[str, Tuple[str, ...]] = {
    # m1..m8 correspond to pipeline internal modules inside build_anndata
    'm1': (
        'module1.knn_k', 'module1.include_directional', 'module1.winsor_perc_low', 'module1.winsor_perc_high',
    ),
    'm2': (
        'module2.top_g','module2.bow_knn_k','module2.max_rows_per_cell_for_svd','module2.max_svd_rows','module2.svd_n_components','module2.bow_weight_mode','module2.use_tfidf','module2.tf_norm','module2.parallel_chunk_cells',
    ),
    'm3': (
        'module3.hdbscan_min_cluster_size','module3.hdbscan_allow_single_cluster','module3.hdbscan_min_samples_frac','module3.target_sub_size','module3.n_feat_hdb','module3.kmeans_min_k','module3.kmeans_max_k','module3.use_k_cap','module3.merge_min_size',
    ),
    # m4 builds adata1 from point features; no direct params, but depends on m1..m3
    'm5': (
        'module5.block_size','module5.w_gc','module5.w_expr','module5.w_sp','module5.use_gc','module5.use_expr','module5.speed_preset','module5.use_spatial_feats','module5.use_decell','module5.use_cosine','module5.enable_cap','module5.estimate_k','module5.fixed_k','module5.target_sub_per_cluster','module5.target_cluster_per_cell','module5.K_min','module5.K_max','module5.split_by_nucleus','module5.nucleus_threshold','module5.cap_ratio_base','module5.cap_min_keep','module5.cap_margin_tau','module5.cap_max_frac_soft',
    ),
    'm6': (
        'module6.h5ad_compression_override','module6.save_shape_summary',
    ),
    # m7/m8 are build-time graph and DGI embedding steps
    'm7': (
        'module7.scvi_epochs','module7.scvi_n_latent','module7.scvi_lr','module7.scvi_batch_size','module7.rep_priority','module7.neighbors','module7.neighbors.k_expr','module7.neighbors.k_geom','module7.alpha_expr','module7.alpha_geom','module7.same_cell_beta','module7.topk_prune','module7.edge_norm','module7.feature_scale',
    ),
    'm8': (
        'module8.dgi_epochs','module8.enable_dgi_sage','module8.dgi_hidden','module8.dgi_out','module8.dgi_lr','module8.dgi_weight_decay','module8.dgi_clip_grad',
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

