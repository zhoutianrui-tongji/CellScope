import os
import copy
import yaml
import json
import hashlib
from typing import Any, Dict, Mapping, Optional, Tuple

from .resource_paths import default_config_path


BUILD_PIPELINE_SCHEMA_VERSION = 8
PREPROCESS_SCHEMA_VERSION = 2
ANNOTATION_SCHEMA_VERSION = 15
MODULE_SCHEMA_VERSIONS: Dict[str, int] = {
    'm1': 2,
    'm2': 1,
    'm3': 3,
    'm5': 6,
    'm6': 3,
    'm7': 6,
    'm8': 7,
}

METASPOT_DGI_REP = "X_dgi_gcn_metaspot"
_DGI_UNIT_ALIASES = {
    "metadomain": "metadomain",
    "meta-domain": "metadomain",
    "domain": "metadomain",
    "metaspot": "metaspot",
    "meta-spot": "metaspot",
    "spot": "metaspot",
}

_ANNOTATION_INTERNAL_DEFAULTS: Dict[str, Any] = {
    "enable": True,
    "method": "rapx",
    "unit": "metadomain",
    "decision_mode": "independent",
    "independent_mixed_score_delta": 0.75,
    "independent_unknown_label": "Unknown",
    "independent_geometry_min_abs_r": 0.05,
    "independent_domain_compartment_feasibility": True,
    "independent_assignment_mode": "hierarchical",
    "independent_exploratory_max_q": 0.30,
    "independent_exploratory_min_channel_support": 2,
    "independent_exploratory_min_score_gap": 0.50,
    "independent_subtype_parent_margin": 0.0,
    "independent_omm_competitor_margin": 0.0,
    "two_stage": False,
    "spatial_channel": True,
    "low_conf_policy": "supported",
    "low_conf_relaxed_z": 1.25,
    "low_conf_relaxed_q": 0.30,
    "marker_set_target": 600,
    "marker_min_weight": 0.05,
    "marker_min_margin": 0.05,
    "gold_column_normalize": False,
    "min_channel_support": 2,
    "enforce_geometry_consistency": False,
    "geometry_quantile": 0.4,
    "geometry_channel_weight": 0.0,
    "guard_mode": "adaptive",
    "subtype_fdr": 0.05,
    "min_effective_markers": 10,
    "null_permutations": 500,
    "null_seed": 0,
    "single_stage_compartment_score_weight": 0.0,
    "single_stage_compartment_score_tau": 0.2,
    "single_stage_compartment_guard": False,
    "single_stage_compartment_min_abs_r": 0.05,
    "single_stage_compartment_max_score_gap": 8.5,
    "single_stage_compartment_min_z": 1.0,
    "single_stage_compartment_max_q": 0.35,
    "single_stage_compartment_min_channel_support": 1,
    "metaspot_compartment_correction": True,
    "metaspot_pool_rep": "X_dgi_gcn_metaspot",
    "metaspot_pool_k": 8,
    "metaspot_dgi_consensus_weight": 0.75,
    "metaspot_dgi_seed_min_marker_transcripts": 3,
    "metaspot_dgi_seed_min_marker_genes": 2,
    "metaspot_dgi_seed_min_log2_enrichment": 0.5,
    "metaspot_dgi_min_informative_neighbors": 2,
    "metaspot_dgi_exploratory": True,
    "metaspot_dgi_exploratory_min_support": 0.75,
    "metaspot_dgi_exploratory_min_margin": 0.25,
    "metaspot_cell_coverage_rescue": False,
    "metaspot_cell_coverage_labels": [
        "Nucleolus",
        "Lamina",
        "Nuclear Pore",
        "ER",
    ],
    "metaspot_cell_coverage_min_marker_transcripts": 4,
    "metaspot_cell_coverage_min_marker_genes": 3,
    "metaspot_cell_coverage_min_log2_enrichment": 0.0,
    "metaspot_cell_coverage_max_binomial_p": 0.50,
    "metaspot_cell_coverage_min_local_marker_transcripts": 1,
    "metaspot_cell_coverage_min_local_marker_genes": 1,
    "metaspot_cell_coverage_min_local_log2_enrichment": 0.0,
    "metaspot_cell_coverage_min_dgi_support": 0.50,
    "metaspot_cell_coverage_min_dgi_margin": 0.0,
    "metaspot_cell_coverage_max_per_label": 1,
    "metaspot_diminishing_return_selection": False,
    "metaspot_diminishing_return_labels": [
        "Nucleolus",
        "Lamina",
        "Nuclear Pore",
        "ER",
        "OMM",
    ],
    "metaspot_diminishing_return_log_penalty": 0.10,
    "metaspot_diminishing_return_burden_penalty": 0.50,
    "metaspot_diminishing_return_min_dgi_support": 0.50,
    "metaspot_hierarchical_cell_rescue": False,
    "metaspot_hierarchical_min_cell_transcripts": 200,
    "metaspot_hierarchical_min_metaspots": 20,
    "metaspot_hierarchical_min_presence_score": 0.55,
    "metaspot_hierarchical_min_localization_score": 0.55,
    "metaspot_hierarchical_min_joint_score": 0.60,
    "metaspot_hierarchical_min_local_marker_transcripts": 1,
    "metaspot_hierarchical_min_local_marker_genes": 1,
    "metaspot_hierarchical_min_cell_marker_transcripts": 1,
    "metaspot_hierarchical_min_cell_marker_genes": 1,
    "metaspot_hierarchical_min_dgi_support": 0.35,
    "metaspot_hierarchical_min_dgi_margin": -0.25,
    "metaspot_omm_cell_coherent_rescue": False,
    "metaspot_omm_cell_min_marker_transcripts": 8,
    "metaspot_omm_cell_min_marker_genes": 4,
    "metaspot_omm_cell_min_log2_enrichment": 0.50,
    "metaspot_omm_cell_max_binomial_p": 0.05,
    "metaspot_omm_cell_exploratory_max_binomial_p": 0.20,
    "metaspot_omm_local_min_marker_transcripts": 2,
    "metaspot_omm_local_min_marker_genes": 2,
    "metaspot_omm_local_min_log2_enrichment": 0.50,
    "metaspot_omm_local_min_score_gap": 0.15,
    "metaspot_omm_local_min_dgi_support": 0.60,
    "metaspot_omm_local_min_dgi_margin": 0.0,
    "metaspot_omm_cell_max_per_cell": 3,
    "metaspot_result_umap_max_points": 10_000,
    "metaspot_require_dgi_support": False,
    "metaspot_min_dgi_support": 0.0,
    "metaspot_min_dgi_margin": 0.0,
    "metaspot_min_marker_transcripts": 3,
    "metaspot_min_marker_genes": 2,
    "metaspot_min_log2_enrichment": 0.5,
    "metaspot_min_score_gap": 0.25,
    "metaspot_subtype_fdr": 0.05,
    "metaspot_exploratory_max_q": 0.30,
    "metaspot_background_prior_strength": 2.0,
    "metaspot_label_thresholds": {},
    "metaspot_require_omm_confirmation": True,
    "metaspot_omm_unidentifiable_policy": "unresolved",
    "nuclear_pore_guard": False,
    "nuclear_pore_min_margin": 0.75,
    "nuclear_pore_non_nuclear_top2_min_margin": 5.0,
    "nuclear_pore_max_q": 0.10,
    "nuclear_pore_min_channel_support": 3,
    "nuclear_pore_fallback_label": "Nucleus",
    "lamina_guard": False,
    "lamina_min_margin": 1.0,
    "lamina_non_nuclear_top2_min_margin": 3.0,
    "lamina_max_q": 0.05,
    "lamina_min_channel_support": 2,
    "lamina_boundary_quantile": 0.5,
    "lamina_fallback_label": "Nucleus",
    "omm_guard": False,
    "omm_min_margin": 2.0,
    "omm_max_q": 0.01,
    "omm_min_channel_support": 3,
    "omm_fallback_label": "Cytosol",
    "apex_omm_specificity_filter": True,
    "apex_omm_min_log2fc": 0.75,
    "apex_omm_min_delta": 0.50,
    "omm_metaspot_refinement": True,
    "omm_metaspot_apply_to_primary": True,
    "omm_metaspot_ambiguous_policy": "competitor",
    "omm_metaspot_min_informative_transcripts": 5,
    "omm_metaspot_shrinkage_count": 20.0,
    "omm_metaspot_posterior_high": 0.80,
    "omm_metaspot_posterior_low": 0.20,
    "rnalocate_augment_er": False,
    "rnalocate_species": ["Homo sapiens", "Mus musculus"],
    "use_manual_markers": False,
    "fast": False,
    "leiden_resolution": 5.0,
    "hvg_n_top_genes": 2000,
    "pca_n_comps": 32,
    "neighbors_k": 20,
    "umap_min_dist": 0.3,
    "umap_spread": 1.0,
    "compute_umap": True,
    "save_umap_plot": True,
    "save_plots": True,
    "cluster_rep": "X_dgi_gcn",
    "require_cluster_rep": False,
}

OMM_POLICIES = frozenset({"strict", "balanced", "audit", "off"})
LEGACY_OMM_REFINEMENT_KEYS = frozenset(
    {
        "omm_metaspot_refinement",
        "omm_metaspot_apply_to_primary",
        "omm_metaspot_ambiguous_policy",
        "omm_metaspot_min_informative_transcripts",
        "omm_metaspot_shrinkage_count",
        "omm_metaspot_posterior_high",
        "omm_metaspot_posterior_low",
    }
)


def resolve_annotation_configuration(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Expand the compact public annotation config into runtime defaults.

    ``annotation.omm_policy`` is the public OMM selector. Existing detailed
    keys remain accepted as expert overrides. A legacy config without the new
    selector retains its former strict/audit/off execution semantics.
    """
    if not isinstance(config, Mapping):
        raw: Mapping[str, Any] = {}
    elif "annotation" in config:
        section = config.get("annotation", {}) or {}
        if not isinstance(section, Mapping):
            raise ValueError("annotation must be a mapping")
        raw = section
    elif any(
        key in config
        for key in (
            "io",
            "module1",
            "module2",
            "module3",
            "module5",
            "module6",
            "module7",
            "module8",
            "runtime",
            "run",
        )
    ):
        raw = {}
    else:
        raw = config

    resolved = copy.deepcopy(_ANNOTATION_INTERNAL_DEFAULTS)
    resolved.update(copy.deepcopy(dict(raw)))

    configured_policy = raw.get("omm_policy")
    if configured_policy is None and any(
        key in raw for key in LEGACY_OMM_REFINEMENT_KEYS
    ):
        if not bool(raw.get("omm_metaspot_refinement", True)):
            policy = "off"
        elif not bool(raw.get("omm_metaspot_apply_to_primary", True)):
            policy = "audit"
        else:
            policy = "strict"
    else:
        policy = str(configured_policy or "balanced").strip().lower()
    if policy not in OMM_POLICIES:
        raise ValueError(
            "annotation.omm_policy must be one of "
            "{'strict', 'balanced', 'audit', 'off'}"
        )

    resolved["omm_policy"] = policy
    resolved["omm_metaspot_refinement"] = policy != "off"
    resolved["omm_metaspot_apply_to_primary"] = policy in {
        "strict",
        "balanced",
    }
    return resolved


def _configured_dgi_selector(config: Mapping[str, Any]) -> Any:
    """Return the new DGI selector, falling back to the legacy graph key."""
    module8 = config.get("module8", {}) or {}
    module7 = config.get("module7", {}) or {}
    new_value = (
        module8.get("dgi_unit")
        if isinstance(module8, Mapping)
        else None
    )
    if str(new_value or "").strip().lower() not in {"", "auto"}:
        return new_value
    legacy_value = (
        module7.get("graph_unit")
        if isinstance(module7, Mapping)
        else None
    )
    if str(legacy_value or "").strip().lower() not in {"", "auto"}:
        return legacy_value
    return None


def dgi_unit_from_config(
    config: Mapping[str, Any],
    *,
    default: str = "metadomain",
) -> str:
    """Resolve ``metaspot`` versus ``metadomain`` DGI execution.

    ``module8.dgi_unit`` is the public selector. ``module7.graph_unit`` remains
    a backward-compatible fallback for configurations created before the
    single-switch interface was introduced.
    """
    raw = _configured_dgi_selector(config)
    if raw is None:
        raw = default
    normalized = str(raw).strip().lower().replace("_", "-")
    unit = _DGI_UNIT_ALIASES.get(normalized)
    if unit is None:
        raise ValueError(
            "module8.dgi_unit must be 'metaspot' or 'metadomain' "
            f"(legacy module7.graph_unit is also accepted), got {raw!r}"
        )
    return unit


def _pool_rep_list(value: Any) -> list[str]:
    if value is None or str(value).strip().lower() == "auto":
        return []
    if isinstance(value, str):
        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def resolve_dgi_configuration(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a canonical config in which one DGI selector drives M5–M8.

    When neither the new selector nor the legacy key is present, the mapping is
    returned without materializing defaults. This is important for partial
    user YAML files that are merged with the repository defaults by the CLI.
    """
    if not isinstance(config, Mapping):
        return {}
    resolved = copy.deepcopy(dict(config))
    if _configured_dgi_selector(resolved) is None:
        return resolved

    unit = dgi_unit_from_config(resolved)
    annotation = resolved.get("annotation", {}) or {}
    annotation_unit = (
        str(annotation.get("unit", "")).strip().lower().replace("_", "-")
        if isinstance(annotation, Mapping)
        else ""
    )
    if annotation_unit in {"metaspot", "meta-spot", "spot"} and (
        unit != "metaspot"
    ):
        raise ValueError(
            "annotation.unit='metaspot' requires "
            "module8.dgi_unit='metaspot' so X_dgi_gcn_metaspot is learned "
            "before final local annotation"
        )
    module5 = resolved.setdefault("module5", {})
    module6 = resolved.setdefault("module6", {})
    module7 = resolved.setdefault("module7", {})
    module8 = resolved.setdefault("module8", {})
    if not all(
        isinstance(section, dict)
        for section in (module5, module6, module7, module8)
    ):
        raise ValueError("module5, module6, module7, and module8 must be mappings")

    clustering_aliases = {
        "minibatch_kmeans": "minibatch_kmeans",
        "minibatch-kmeans": "minibatch_kmeans",
        "kmeans": "minibatch_kmeans",
        "legacy": "minibatch_kmeans",
        "consensus_leiden": "consensus_leiden",
        "consensus-leiden": "consensus_leiden",
        "stability_leiden": "consensus_leiden",
        "stability-leiden": "consensus_leiden",
        "leiden": "consensus_leiden",
    }
    raw_clustering_method = str(
        module5.get("clustering_method", "minibatch_kmeans")
    ).strip().lower()
    if raw_clustering_method not in clustering_aliases:
        raise ValueError(
            "module5.clustering_method must be one of "
            "{'minibatch_kmeans', 'consensus_leiden'}"
        )
    module5["clustering_method"] = clustering_aliases[
        raw_clustering_method
    ]
    if module5["clustering_method"] == "consensus_leiden":
        resolutions = module5.get(
            "consensus_resolutions", [1.0, 2.0, 4.0, 6.0, 8.0]
        )
        seeds = module5.get("consensus_seeds", [17, 42, 89])
        if not isinstance(resolutions, (list, tuple)) or not resolutions:
            raise ValueError(
                "module5.consensus_resolutions must be a non-empty list"
            )
        if not isinstance(seeds, (list, tuple)) or len(seeds) < 2:
            raise ValueError(
                "module5.consensus_seeds must contain at least two seeds"
            )
        try:
            resolutions = sorted({float(value) for value in resolutions})
            seeds = [int(value) for value in seeds]
            consensus_neighbors = int(
                module5.get("consensus_neighbors", 20)
            )
            consensus_iterations = int(
                module5.get("consensus_leiden_iterations", 2)
            )
            consensus_min_average_size = float(
                module5.get("consensus_min_average_size", 100.0)
            )
            consensus_max_average_size = float(
                module5.get("consensus_max_average_size", 800.0)
            )
            consensus_min_recurrence_cells = int(
                module5.get("consensus_min_recurrence_cells", 2)
            )
            consensus_max_small_fraction = float(
                module5.get("consensus_max_small_cluster_fraction", 0.10)
            )
            consensus_max_nonrecurrent_fraction = float(
                module5.get("consensus_max_nonrecurrent_fraction", 0.10)
            )
            consensus_edge_agreement = float(
                module5.get("consensus_edge_agreement", 2.0 / 3.0)
            )
            consensus_min_stability = float(
                module5.get("consensus_min_stability", 0.55)
            )
            consensus_stability_tolerance = float(
                module5.get("consensus_stability_tolerance", 0.10)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid module5 consensus Leiden configuration: {exc}"
            ) from exc
        if min(resolutions) <= 0.0:
            raise ValueError(
                "module5.consensus_resolutions must contain positive values"
            )
        if consensus_neighbors < 2 or consensus_iterations == 0:
            raise ValueError(
                "module5 consensus neighbors/iterations are invalid"
            )
        if (
            consensus_min_average_size <= 0.0
            or consensus_max_average_size < consensus_min_average_size
        ):
            raise ValueError(
                "module5 consensus average-size bounds are invalid"
            )
        if consensus_min_recurrence_cells < 1:
            raise ValueError(
                "module5.consensus_min_recurrence_cells must be >= 1"
            )
        if not 0.0 <= consensus_max_small_fraction <= 1.0:
            raise ValueError(
                "module5.consensus_max_small_cluster_fraction must be in [0, 1]"
            )
        if not 0.0 <= consensus_max_nonrecurrent_fraction <= 1.0:
            raise ValueError(
                "module5.consensus_max_nonrecurrent_fraction must be in [0, 1]"
            )
        if not 0.0 < consensus_edge_agreement <= 1.0:
            raise ValueError(
                "module5.consensus_edge_agreement must be in (0, 1]"
            )
        if not -1.0 <= consensus_min_stability <= 1.0:
            raise ValueError(
                "module5.consensus_min_stability must be in [-1, 1]"
            )
        if not 0.0 <= consensus_stability_tolerance <= 2.0:
            raise ValueError(
                "module5.consensus_stability_tolerance must be in [0, 2]"
            )
        module5["consensus_resolutions"] = resolutions
        module5["consensus_seeds"] = seeds
        module5["consensus_neighbors"] = consensus_neighbors
        module5["consensus_leiden_iterations"] = consensus_iterations
        module5[
            "consensus_min_average_size"
        ] = consensus_min_average_size
        module5[
            "consensus_max_average_size"
        ] = consensus_max_average_size
        module5[
            "consensus_min_recurrence_cells"
        ] = consensus_min_recurrence_cells
        module5[
            "consensus_max_small_cluster_fraction"
        ] = consensus_max_small_fraction
        module5[
            "consensus_max_nonrecurrent_fraction"
        ] = consensus_max_nonrecurrent_fraction
        module5["consensus_edge_agreement"] = consensus_edge_agreement
        module5["consensus_min_stability"] = consensus_min_stability
        module5[
            "consensus_stability_tolerance"
        ] = consensus_stability_tolerance

    large_scale_aliases = {
        "auto": "auto",
        "consensus_leiden": "consensus_leiden",
        "consensus-leiden": "consensus_leiden",
        "exact": "consensus_leiden",
        "hierarchical_prototype": "hierarchical_prototype",
        "hierarchical-prototype": "hierarchical_prototype",
        "hierarchical": "hierarchical_prototype",
        "prototype": "hierarchical_prototype",
    }
    raw_large_scale_method = str(
        module5.get("large_scale_method", "auto")
    ).strip().lower()
    if raw_large_scale_method not in large_scale_aliases:
        raise ValueError(
            "module5.large_scale_method must be one of "
            "{'auto','consensus_leiden','hierarchical_prototype'}"
        )
    module5["large_scale_method"] = large_scale_aliases[
        raw_large_scale_method
    ]
    for key, minimum in (
        ("hierarchical_block_nodes", 1_000),
        ("hierarchical_local_target_size", 5),
        ("hierarchical_global_target_size", 5),
        ("hierarchical_max_local_clusters", 1),
        ("hierarchical_min_global_clusters", 1),
        ("hierarchical_max_global_clusters", 1),
        ("hierarchical_batch_size", 256),
        ("hierarchical_max_iter", 5),
    ):
        value = int(module5.get(key, minimum))
        if value < minimum:
            raise ValueError(f"module5.{key} must be >= {minimum}")
        module5[key] = value
    if (
        module5["hierarchical_max_global_clusters"]
        < module5["hierarchical_min_global_clusters"]
    ):
        raise ValueError(
            "module5 hierarchical global cluster bounds are invalid"
        )

    encoder_aliases = {
        "plain": "plain",
        "gcn": "plain",
        "residual": "residual",
        "residual_gcn": "residual",
        "residual-gcn": "residual",
    }
    raw_encoder = str(
        module8.get("dgi_encoder", "residual")
    ).strip().lower()
    if raw_encoder not in encoder_aliases:
        raise ValueError(
            "module8.dgi_encoder must be 'plain' or 'residual', "
            f"got {raw_encoder!r}"
        )
    try:
        dgi_layers = int(module8.get("dgi_layers", 2))
        dgi_hidden = int(module8.get("dgi_hidden", 64))
        dgi_out = int(module8.get("dgi_out", 32))
        dgi_dropout = float(module8.get("dgi_dropout", 0.1))
        residual_alpha = float(module8.get("dgi_residual_alpha", 0.1))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid module8 DGI architecture configuration: {exc}"
        ) from exc
    if dgi_layers < 1:
        raise ValueError("module8.dgi_layers must be >= 1")
    if dgi_hidden < 1 or dgi_out < 1:
        raise ValueError("module8.dgi_hidden and module8.dgi_out must be >= 1")
    if not 0.0 <= dgi_dropout < 1.0:
        raise ValueError("module8.dgi_dropout must be in [0, 1)")
    if not 0.0 <= residual_alpha <= 1.0:
        raise ValueError("module8.dgi_residual_alpha must be in [0, 1]")
    layer_norm = module8.get("dgi_layer_norm", True)
    if not isinstance(layer_norm, bool):
        raise ValueError("module8.dgi_layer_norm must be true or false")
    dgi_device = str(module8.get("dgi_device", "auto")).strip().lower()
    if dgi_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "module8.dgi_device must be one of {'auto', 'cpu', 'cuda'}"
        )
    try:
        dgi_cpu_threads = int(module8.get("dgi_cpu_threads", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "module8.dgi_cpu_threads must be an integer >= 0"
        ) from exc
    if dgi_cpu_threads < 0:
        raise ValueError("module8.dgi_cpu_threads must be >= 0")
    dgi_sparse_layout = str(
        module8.get("dgi_sparse_layout", "auto")
    ).strip().lower()
    if dgi_sparse_layout not in {"auto", "csr", "coo"}:
        raise ValueError(
            "module8.dgi_sparse_layout must be one of {'auto', 'csr', 'coo'}"
        )
    dmgi_fused_corruption_pass = module8.get(
        "dmgi_fused_corruption_pass", True
    )
    if not isinstance(dmgi_fused_corruption_pass, bool):
        raise ValueError(
            "module8.dmgi_fused_corruption_pass must be true or false"
        )
    dgi_tf32 = module8.get("dgi_tf32", True)
    if not isinstance(dgi_tf32, bool):
        raise ValueError("module8.dgi_tf32 must be true or false")
    training_mode = str(
        module8.get("training_mode", "auto")
    ).strip().lower().replace("-", "_")
    if training_mode not in {"auto", "full_batch", "neighbor_sampled"}:
        raise ValueError(
            "module8.training_mode must be one of "
            "{'auto','full_batch','neighbor_sampled'}"
        )
    module8["training_mode"] = training_mode
    for key, minimum in (
        ("sampled_dgi_epochs", 1),
        ("sampled_batch_size", 1),
        ("sampled_nodes_per_epoch", 1),
        ("sampled_inference_batch_size", 1),
        ("sampled_validation_nodes", 1),
        ("sampled_num_workers", 0),
    ):
        value = int(module8.get(key, minimum))
        if value < minimum:
            raise ValueError(f"module8.{key} must be >= {minimum}")
        module8[key] = value
    for key in ("sampled_expression_fanout", "sampled_spatial_fanout"):
        fanout = module8.get(key, [1] * dgi_layers)
        if not isinstance(fanout, (list, tuple)) or not fanout:
            raise ValueError(f"module8.{key} must be a non-empty list")
        values = [int(value) for value in fanout]
        if any(value < 1 for value in values):
            raise ValueError(f"module8.{key} values must be >= 1")
        module8[key] = values

    corruption_aliases = {
        "auto": "auto",
        "global": "global",
        "global_shuffle": "global",
        "global-shuffle": "global",
        "same_cell": "same_cell",
        "same-cell": "same_cell",
        "within_cell": "same_cell",
        "within-cell": "same_cell",
    }
    raw_corruption = str(
        module8.get("dgi_corruption_mode", "auto")
    ).strip().lower()
    if raw_corruption not in corruption_aliases:
        raise ValueError(
            "module8.dgi_corruption_mode must be one of "
            "{'auto', 'global', 'same_cell'}, "
            f"got {raw_corruption!r}"
        )
    try:
        local_loss_weight = float(
            module8.get("dgi_local_loss_weight", 0.2)
        )
        local_temperature = float(
            module8.get("dgi_local_temperature", 0.2)
        )
        local_samples = int(module8.get("dgi_local_samples", 65536))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid module8 local DGI objective configuration: {exc}"
        ) from exc
    local_graph_key = str(
        module8.get("dgi_local_graph_key", "auto")
    ).strip()
    cpu_backend_aliases = {
        "auto": "auto",
        "pyg": "pyg",
        "coo": "pyg",
        "torch_csr": "torch_csr",
        "torch-csr": "torch_csr",
        "csr": "torch_csr",
    }
    raw_cpu_backend = str(
        module8.get("dgi_cpu_backend", "auto")
    ).strip().lower()
    if raw_cpu_backend not in cpu_backend_aliases:
        raise ValueError(
            "module8.dgi_cpu_backend must be one of "
            "{'auto', 'pyg', 'torch_csr'}"
        )
    if local_loss_weight < 0.0:
        raise ValueError("module8.dgi_local_loss_weight must be >= 0")
    if local_temperature <= 0.0:
        raise ValueError("module8.dgi_local_temperature must be > 0")
    if local_samples < 0:
        raise ValueError("module8.dgi_local_samples must be >= 0")
    if not local_graph_key:
        raise ValueError("module8.dgi_local_graph_key must not be empty")

    module8["dgi_encoder"] = encoder_aliases[raw_encoder]
    framework_aliases = {
        "standard_dgi": "standard_dgi",
        "standard-dgi": "standard_dgi",
        "dgi": "standard_dgi",
        "cell_aware_dmgi": "cell_aware_dmgi",
        "cell-aware-dmgi": "cell_aware_dmgi",
        "ca_dmgi": "cell_aware_dmgi",
        "ca-dmgi": "cell_aware_dmgi",
        "ch_dmgi": "cell_aware_dmgi",  # legacy compatibility only
        "ch-dmgi": "cell_aware_dmgi",  # legacy compatibility only
        "dmgi": "cell_aware_dmgi",
    }
    raw_framework = str(
        module8.get("dgi_framework", "standard_dgi")
    ).strip().lower()
    if raw_framework not in framework_aliases:
        raise ValueError(
            "module8.dgi_framework must be one of "
            "{'standard_dgi', 'cell_aware_dmgi'}"
        )
    framework = framework_aliases[raw_framework]
    if framework == "cell_aware_dmgi" and unit != "metaspot":
        raise ValueError(
            "module8.dgi_framework='cell_aware_dmgi' requires "
            "module8.dgi_unit='metaspot'"
        )
    try:
        dmgi_expression_weight = float(
            module8.get("dmgi_expression_weight", 1.0)
        )
        dmgi_spatial_weight = float(
            module8.get("dmgi_spatial_weight", 1.0)
        )
        dmgi_consensus_weight = float(
            module8.get("dmgi_consensus_weight", 0.1)
        )
        dmgi_gate_floor = float(module8.get("dmgi_gate_floor", 0.1))
        dmgi_min_epochs = int(module8.get("dmgi_min_epochs", 40))
        dmgi_patience = int(module8.get("dmgi_patience", 15))
        dmgi_min_delta = float(module8.get("dmgi_min_delta", 1e-4))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid module8 CA-DMGI configuration: {exc}") from exc
    dmgi_early_stopping = module8.get("dmgi_early_stopping", True)
    if not isinstance(dmgi_early_stopping, bool):
        raise ValueError("module8.dmgi_early_stopping must be true or false")
    dmgi_expression_graph_key = str(
        module8.get(
            "dmgi_expression_graph_key",
            "metaspot_expr_connectivities",
        )
    ).strip()
    dmgi_spatial_graph_key = str(
        module8.get(
            "dmgi_spatial_graph_key",
            "metaspot_spatial_connectivities",
        )
    ).strip()
    if min(dmgi_expression_weight, dmgi_spatial_weight) <= 0.0:
        raise ValueError("module8 CA-DMGI relation weights must be > 0")
    if dmgi_consensus_weight < 0.0:
        raise ValueError("module8.dmgi_consensus_weight must be >= 0")
    if not 0.0 <= dmgi_gate_floor < 0.5:
        raise ValueError("module8.dmgi_gate_floor must be in [0, 0.5)")
    configured_epochs = int(module8.get("dgi_epochs", 100))
    if dmgi_min_epochs < 1 or dmgi_min_epochs > configured_epochs:
        raise ValueError(
            "module8.dmgi_min_epochs must be within [1, dgi_epochs]"
        )
    if dmgi_patience < 1 or dmgi_min_delta < 0.0:
        raise ValueError(
            "module8.dmgi_patience must be >= 1 and dmgi_min_delta >= 0"
        )
    if not dmgi_expression_graph_key or not dmgi_spatial_graph_key:
        raise ValueError("module8 CA-DMGI relation graph keys must not be empty")
    module8["dgi_framework"] = framework
    module8["dmgi_expression_weight"] = dmgi_expression_weight
    module8["dmgi_spatial_weight"] = dmgi_spatial_weight
    module8["dmgi_consensus_weight"] = dmgi_consensus_weight
    module8["dmgi_gate_floor"] = dmgi_gate_floor
    module8["dmgi_early_stopping"] = dmgi_early_stopping
    module8["dmgi_min_epochs"] = dmgi_min_epochs
    module8["dmgi_patience"] = dmgi_patience
    module8["dmgi_min_delta"] = dmgi_min_delta
    module8["dmgi_expression_graph_key"] = dmgi_expression_graph_key
    module8["dmgi_spatial_graph_key"] = dmgi_spatial_graph_key
    module8["dgi_layers"] = dgi_layers
    module8["dgi_hidden"] = dgi_hidden
    module8["dgi_out"] = dgi_out
    module8["dgi_dropout"] = dgi_dropout
    module8["dgi_residual_alpha"] = residual_alpha
    module8["dgi_layer_norm"] = layer_norm
    module8["dgi_device"] = dgi_device
    module8["dgi_cpu_threads"] = dgi_cpu_threads
    module8["dgi_sparse_layout"] = dgi_sparse_layout
    module8["dmgi_fused_corruption_pass"] = dmgi_fused_corruption_pass
    module8["dgi_tf32"] = dgi_tf32
    module8["dgi_corruption_mode"] = corruption_aliases[raw_corruption]
    module8["dgi_local_loss_weight"] = local_loss_weight
    module8["dgi_local_temperature"] = local_temperature
    module8["dgi_local_samples"] = local_samples
    module8["dgi_local_graph_key"] = local_graph_key
    module8["dgi_cpu_backend"] = cpu_backend_aliases[raw_cpu_backend]

    # Materialize internal compatibility fields in effective_params.yaml so
    # provenance always records the actual execution path.
    module8["dgi_unit"] = unit
    module7["graph_unit"] = unit
    pool_reps = _pool_rep_list(module6.get("pool_reps"))

    if unit == "metaspot":
        module5["cluster_rep"] = METASPOT_DGI_REP
        if METASPOT_DGI_REP not in pool_reps:
            pool_reps.append(METASPOT_DGI_REP)
    else:
        configured_rep = str(module5.get("cluster_rep", "")).strip()
        if configured_rep.lower() in {
            "",
            "auto",
            METASPOT_DGI_REP.lower(),
        }:
            module5["cluster_rep"] = "legacy_features"
        pool_reps = [
            rep for rep in pool_reps
            if rep != METASPOT_DGI_REP
        ]
    module6["pool_reps"] = pool_reps
    return resolved


def _workspace_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _params_path() -> str:
    return default_config_path()


def get_default_params_path() -> str:
    return _params_path()


def load_params_yaml(
    path: Optional[str] = None,
    *,
    resolve_config: bool = True,
) -> Dict[str, Any]:
    """Load params YAML.

    If ``path`` is provided, load from that file; otherwise fall back to the
    ``CELLSCOPE_CONFIG`` env var (if set), then repo default ``config/params.yaml``.
    ``resolve_config=False`` returns the raw mapping for CLI overlay merging.
    """
    env_p = os.environ.get("CELLSCOPE_CONFIG", "").strip()
    explicit = bool(path or env_p)
    if path:
        p = path
    else:
        p = env_p if env_p else _params_path()
    try:
        if not os.path.isfile(p):
            if explicit:
                raise ValueError(f"configuration file does not exist: {p}")
            return {}
        with open(p, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
    except ValueError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        if explicit:
            raise ValueError(f"could not parse configuration {p}: {exc}") from exc
        return {}
    if not isinstance(loaded, Mapping):
        if explicit:
            raise ValueError(
                f"configuration root must be a mapping, got {type(loaded).__name__}"
            )
        return {}
    loaded_dict = copy.deepcopy(dict(loaded))
    return (
        resolve_dgi_configuration(loaded_dict)
        if resolve_config
        else loaded_dict
    )


# Map config keys to pipeline stages for minimal re-run decisions.
# Only names that appear in `pipeline.stage_order` are relevant here.
_BUILD_STAGE_KEYS: Tuple[str, ...] = (
    # Scale-aware execution
    'execution.profile','execution.exact_max_metaspots','execution.sampled_max_metaspots','execution.memory_budget_gb','execution.scratch_dir','execution.partition_rows','execution.fail_on_backend_fallback',
    # Module 1
    'module1.include_directional','module1.use_input_nucleus_flags','module1.winsor_perc_low','module1.winsor_perc_high',
    # Module 2
    'module2.top_g','module2.bow_knn_k','module2.max_rows_per_cell_for_svd','module2.max_svd_rows','module2.svd_n_components','module2.bow_weight_mode','module2.use_tfidf','module2.tf_norm','module2.idf_mode','module2.parallel_chunk_cells','module2.svd_sample_cells','module2.source_from_point_df','module2.stream_local_embeddings',
    # Module 3
    'module3.hdbscan_min_cluster_size','module3.hdbscan_allow_single_cluster','module3.hdbscan_min_samples_frac','module3.target_sub_size','module3.n_feat_hdb','module3.kmeans_min_k','module3.kmeans_max_k','module3.use_k_cap','module3.merge_min_size','module3.hdbscan_max_points','module3.kmeans_train_size','module3.kmeans_batch_size','module3.small_backend_cells','module3.sequential_cells','module3.enforce_spatial_connectivity','module3.partition_by_nucleus','module3.spatial_connectivity_k','module3.spatial_radius_quantile','module3.spatial_radius_factor','module3.spatial_smoothing_iterations','module3.spatial_smoothing_min_fraction','module3.spatial_min_component_size','module3.spatial_compactness','module3.target_sites_per_metaspot','module3.min_metaspots_per_cell','module3.max_metaspots_per_cell',
    # Module 5
    'module5.block_size','module5.clustering_method','module5.large_scale_method','module5.hierarchical_block_nodes','module5.hierarchical_local_target_size','module5.hierarchical_global_target_size','module5.hierarchical_max_local_clusters','module5.hierarchical_min_global_clusters','module5.hierarchical_max_global_clusters','module5.hierarchical_batch_size','module5.hierarchical_max_iter',
    'module5.consensus_neighbors','module5.consensus_resolutions','module5.consensus_seeds','module5.consensus_leiden_iterations',
    'module5.consensus_min_average_size','module5.consensus_max_average_size','module5.consensus_min_recurrence_cells',
    'module5.consensus_max_small_cluster_fraction','module5.consensus_max_nonrecurrent_fraction','module5.consensus_edge_agreement','module5.consensus_min_stability','module5.consensus_stability_tolerance',
    'module5.w_gc','module5.w_expr','module5.w_sp','module5.use_gc','module5.use_expr',
    'module5.speed_preset','module5.use_spatial_feats','module5.use_decell','module5.use_cosine','module5.enable_cap',
    'module5.estimate_k','module5.fixed_k',
    'module5.auto_k_min','module5.auto_k_max',
    'module5.auto_k_min_average_size','module5.auto_k_max_average_size',
    'module5.auto_k_n_candidates','module5.auto_k_train_sample',
    'module5.auto_k_validation_sample','module5.auto_k_sample_seed',
    'module5.auto_k_seeds','module5.auto_k_stability_min',
    'module5.auto_k_max_small_cluster_fraction',
    'module5.auto_k_min_train_per_cluster','module5.auto_k_batch_size',
    'module5.auto_k_max_iter','module5.auto_k_full_eval_max',
    'module5.target_sub_per_cluster','module5.target_cluster_per_cell','module5.K_min','module5.K_max',
    'module5.split_by_nucleus','module5.nucleus_threshold',
    'module5.cap_ratio_base','module5.cap_min_keep','module5.cap_margin_tau','module5.cap_max_frac_soft',
    'module5.train_sample_per_block','module5.assign_chunk_size','module5.mbk_batch_size','module5.mbk_max_iter',
    'module5.cluster_rep','module5.cluster_rep_decell','module5.min_domain_size','module5.random_seed',
    # Module 6
    'module6.pool_reps','module6.pool_normalize',
    # Module 7
    'module7.scvi_epochs','module7.scvi_n_latent','module7.scvi_lr','module7.scvi_batch_size',
    'module7.graph_unit','module7.neighbors','module7.neighbors.k_expr','module7.neighbors.k_geom','module7.neighbors.k_spatial',
    'module7.alpha_expr','module7.alpha_geom','module7.same_cell_beta','module7.topk_prune','module7.edge_norm','module7.feature_scale',
    'module7.metaspot_expr_n_components','module7.metaspot_alpha_expr','module7.metaspot_alpha_spatial','module7.metaspot_edge_norm','module7.metaspot_use_geometry_features','module7.metaspot_expr_cross_cell_only','module7.max_full_batch_nodes','module7.knn_backend','module7.ann_candidate_multiplier','module7.ann_query_chunk_nodes','module7.ann_train_sample','module7.faiss_hnsw_m','module7.faiss_nprobe','module7.faiss_pq_subvectors','module7.faiss_pq_bits','module7.faiss_use_gpu','module7.faiss_gpu_device',
    # Module 8
    'module8.dgi_unit','module8.dgi_framework','module8.dgi_epochs','module8.enable_dgi_sage','module8.dgi_encoder','module8.dgi_layers','module8.dgi_hidden','module8.dgi_out','module8.dgi_dropout','module8.dgi_residual_alpha','module8.dgi_layer_norm','module8.dgi_device','module8.dgi_cpu_threads','module8.dgi_sparse_layout','module8.dgi_tf32','module8.dgi_corruption_mode','module8.dgi_local_loss_weight','module8.dgi_local_temperature','module8.dgi_local_samples','module8.dgi_local_graph_key','module8.dgi_cpu_backend','module8.dmgi_expression_graph_key','module8.dmgi_spatial_graph_key','module8.dmgi_expression_weight','module8.dmgi_spatial_weight','module8.dmgi_consensus_weight','module8.dmgi_gate_floor','module8.dmgi_fused_corruption_pass','module8.dmgi_early_stopping','module8.dmgi_min_epochs','module8.dmgi_patience','module8.dmgi_min_delta','module8.dgi_lr','module8.dgi_weight_decay','module8.dgi_clip_grad','module8.amp','module8.random_seed','module8.training_mode','module8.sampled_dgi_epochs','module8.sampled_batch_size','module8.sampled_expression_fanout','module8.sampled_spatial_fanout','module8.sampled_nodes_per_epoch','module8.sampled_inference_batch_size','module8.sampled_validation_nodes','module8.sampled_num_workers',
)

STAGE_PARAM_MAP: Dict[str, Tuple[str, ...]] = {
    'load_inputs': (),
    'preprocess': ('module1.use_input_nucleus_flags',),
    'build_anndata': _BUILD_STAGE_KEYS + (
        'io.drop_embeddings',
        'io.final_keep_columns',
    ),
    'annotate': (
        'annotation.enable','annotation.method','annotation.unit','annotation.decision_mode','annotation.independent_mixed_score_delta','annotation.independent_unknown_label','annotation.independent_geometry_min_abs_r','annotation.independent_domain_compartment_feasibility','annotation.independent_assignment_mode','annotation.independent_exploratory_max_q','annotation.independent_exploratory_min_channel_support','annotation.independent_exploratory_min_score_gap','annotation.independent_subtype_parent_margin','annotation.independent_omm_competitor_margin','annotation.two_stage','annotation.spatial_channel','annotation.low_conf_policy','annotation.low_conf_relaxed_z','annotation.low_conf_relaxed_q','annotation.marker_set_target','annotation.marker_min_weight','annotation.marker_min_margin','annotation.gold_column_normalize','annotation.min_channel_support','annotation.enforce_geometry_consistency','annotation.geometry_quantile','annotation.geometry_channel_weight','annotation.guard_mode','annotation.subtype_fdr','annotation.min_effective_markers','annotation.null_permutations','annotation.null_seed','annotation.single_stage_compartment_score_weight','annotation.single_stage_compartment_score_tau','annotation.single_stage_compartment_guard','annotation.single_stage_compartment_min_abs_r','annotation.single_stage_compartment_max_score_gap','annotation.single_stage_compartment_min_z','annotation.single_stage_compartment_max_q','annotation.single_stage_compartment_min_channel_support','annotation.metaspot_compartment_correction','annotation.metaspot_pool_rep','annotation.metaspot_pool_k','annotation.metaspot_dgi_consensus_weight','annotation.metaspot_dgi_seed_min_marker_transcripts','annotation.metaspot_dgi_seed_min_marker_genes','annotation.metaspot_dgi_seed_min_log2_enrichment','annotation.metaspot_dgi_min_informative_neighbors','annotation.metaspot_dgi_exploratory','annotation.metaspot_dgi_exploratory_min_support','annotation.metaspot_dgi_exploratory_min_margin','annotation.metaspot_cell_coverage_rescue','annotation.metaspot_cell_coverage_labels','annotation.metaspot_cell_coverage_min_marker_transcripts','annotation.metaspot_cell_coverage_min_marker_genes','annotation.metaspot_cell_coverage_min_log2_enrichment','annotation.metaspot_cell_coverage_max_binomial_p','annotation.metaspot_cell_coverage_min_local_marker_transcripts','annotation.metaspot_cell_coverage_min_local_marker_genes','annotation.metaspot_cell_coverage_min_local_log2_enrichment','annotation.metaspot_cell_coverage_min_dgi_support','annotation.metaspot_cell_coverage_min_dgi_margin','annotation.metaspot_cell_coverage_max_per_label','annotation.metaspot_multistructure_first_locus','annotation.metaspot_multistructure_first_locus_min_presence','annotation.metaspot_multistructure_first_locus_min_localization','annotation.metaspot_multistructure_first_locus_min_evidence','annotation.metaspot_multistructure_first_locus_max_projected_burden','annotation.metaspot_fine_min_spatial_coherence','annotation.metaspot_diminishing_return_selection','annotation.metaspot_diminishing_return_labels','annotation.metaspot_diminishing_return_log_penalty','annotation.metaspot_diminishing_return_burden_penalty','annotation.metaspot_diminishing_return_min_dgi_support','annotation.metaspot_hierarchical_cell_rescue','annotation.metaspot_hierarchical_min_cell_transcripts','annotation.metaspot_hierarchical_min_metaspots','annotation.metaspot_hierarchical_min_presence_score','annotation.metaspot_hierarchical_min_localization_score','annotation.metaspot_hierarchical_min_joint_score','annotation.metaspot_hierarchical_min_local_marker_transcripts','annotation.metaspot_hierarchical_min_local_marker_genes','annotation.metaspot_hierarchical_min_cell_marker_transcripts','annotation.metaspot_hierarchical_min_cell_marker_genes','annotation.metaspot_hierarchical_min_dgi_support','annotation.metaspot_hierarchical_min_dgi_margin','annotation.metaspot_omm_cell_coherent_rescue','annotation.metaspot_omm_cell_min_marker_transcripts','annotation.metaspot_omm_cell_min_marker_genes','annotation.metaspot_omm_cell_min_log2_enrichment','annotation.metaspot_omm_cell_max_binomial_p','annotation.metaspot_omm_cell_exploratory_max_binomial_p','annotation.metaspot_omm_local_min_marker_transcripts','annotation.metaspot_omm_local_min_marker_genes','annotation.metaspot_omm_local_min_log2_enrichment','annotation.metaspot_omm_local_min_score_gap','annotation.metaspot_omm_local_min_dgi_support','annotation.metaspot_omm_local_min_dgi_margin','annotation.metaspot_omm_cell_max_per_cell','annotation.metaspot_result_umap_max_points','annotation.metaspot_require_dgi_support','annotation.metaspot_min_dgi_support','annotation.metaspot_min_dgi_margin','annotation.metaspot_min_marker_transcripts','annotation.metaspot_min_marker_genes','annotation.metaspot_min_log2_enrichment','annotation.metaspot_min_score_gap','annotation.metaspot_subtype_fdr','annotation.metaspot_exploratory_max_q','annotation.metaspot_background_prior_strength','annotation.metaspot_label_thresholds','annotation.metaspot_unresolved_nuclear_label','annotation.metaspot_unresolved_cytoplasmic_label','annotation.metaspot_require_omm_confirmation','annotation.metaspot_omm_unidentifiable_policy','annotation.nuclear_pore_guard','annotation.nuclear_pore_min_margin','annotation.nuclear_pore_non_nuclear_top2_min_margin','annotation.nuclear_pore_max_q','annotation.nuclear_pore_min_channel_support','annotation.nuclear_pore_fallback_label','annotation.lamina_guard','annotation.lamina_min_margin','annotation.lamina_non_nuclear_top2_min_margin','annotation.lamina_max_q','annotation.lamina_min_channel_support','annotation.lamina_boundary_quantile','annotation.lamina_fallback_label','annotation.omm_guard','annotation.omm_min_margin','annotation.omm_max_q','annotation.omm_min_channel_support','annotation.omm_fallback_label','annotation.apex_omm_specificity_filter','annotation.apex_omm_min_log2fc','annotation.apex_omm_min_delta','annotation.omm_policy','annotation.omm_metaspot_refinement','annotation.omm_metaspot_apply_to_primary','annotation.omm_metaspot_ambiguous_policy','annotation.omm_metaspot_min_informative_transcripts','annotation.omm_metaspot_shrinkage_count','annotation.omm_metaspot_posterior_high','annotation.omm_metaspot_posterior_low','annotation.rnalocate_augment_er','annotation.rnalocate_species','annotation.use_manual_markers','annotation.hvg_n_top_genes','annotation.pca_n_comps','annotation.neighbors_k','annotation.umap_min_dist','annotation.umap_spread','annotation.umap_neighbors_key','annotation.leiden_resolution','annotation.compute_umap','annotation.save_umap_plot','annotation.save_plots','annotation.fast','annotation.cluster_rep','annotation.require_cluster_rep','module7.rep_priority','module7.neighbors','io.write_diagnostic_tables','io.write_diagnostic_csv','io.write_diagnostic_parquet','io.broadcast_metaspot_diagnostics_to_transcripts',
    ),
    'write_outputs': (
        'io.final_format','io.h5ad_compression','io.write_csv_copy','io.write_diagnostic_tables',
        'io.write_diagnostic_csv','io.write_diagnostic_parquet',
        'io.write_partitioned_dataset','io.drop_embeddings','io.final_keep_columns',
    ),
}



# --- Module-level param maps for fine-grained resume within build_anndata ---
MODULE_PARAM_MAP: Dict[str, Tuple[str, ...]] = {
    # m1..m8 correspond to pipeline internal modules inside build_anndata
    'm1': (
        'module1.include_directional','module1.use_input_nucleus_flags','module1.winsor_perc_low','module1.winsor_perc_high',
    ),
    'm2': (
        'module2.top_g','module2.bow_knn_k','module2.max_rows_per_cell_for_svd','module2.max_svd_rows','module2.svd_n_components','module2.bow_weight_mode','module2.use_tfidf','module2.tf_norm','module2.idf_mode','module2.parallel_chunk_cells','module2.svd_sample_cells','module2.source_from_point_df','module2.stream_local_embeddings',
    ),
    'm3': (
        'module3.hdbscan_min_cluster_size','module3.hdbscan_allow_single_cluster','module3.hdbscan_min_samples_frac','module3.target_sub_size','module3.n_feat_hdb','module3.kmeans_min_k','module3.kmeans_max_k','module3.use_k_cap','module3.merge_min_size','module3.hdbscan_max_points','module3.kmeans_train_size','module3.kmeans_batch_size','module3.small_backend_cells','module3.sequential_cells','module3.enforce_spatial_connectivity','module3.partition_by_nucleus','module3.spatial_connectivity_k','module3.spatial_radius_quantile','module3.spatial_radius_factor','module3.spatial_smoothing_iterations','module3.spatial_smoothing_min_fraction','module3.spatial_min_component_size','module3.spatial_compactness','module3.target_sites_per_metaspot','module3.min_metaspots_per_cell','module3.max_metaspots_per_cell',
    ),
    # m4 builds adata1 from point features; no direct params, but depends on m1..m3
    'm5': (
        'module5.block_size','module5.clustering_method','module5.large_scale_method','module5.hierarchical_block_nodes','module5.hierarchical_local_target_size','module5.hierarchical_global_target_size','module5.hierarchical_max_local_clusters','module5.hierarchical_min_global_clusters','module5.hierarchical_max_global_clusters','module5.hierarchical_batch_size','module5.hierarchical_max_iter','module5.consensus_neighbors','module5.consensus_resolutions','module5.consensus_seeds','module5.consensus_leiden_iterations','module5.consensus_min_average_size','module5.consensus_max_average_size','module5.consensus_min_recurrence_cells','module5.consensus_max_small_cluster_fraction','module5.consensus_max_nonrecurrent_fraction','module5.consensus_edge_agreement','module5.consensus_min_stability','module5.consensus_stability_tolerance','module5.w_gc','module5.w_expr','module5.w_sp','module5.use_gc','module5.use_expr','module5.speed_preset','module5.use_spatial_feats','module5.use_decell','module5.use_cosine','module5.enable_cap','module5.estimate_k','module5.fixed_k','module5.auto_k_min','module5.auto_k_max','module5.auto_k_min_average_size','module5.auto_k_max_average_size','module5.auto_k_n_candidates','module5.auto_k_train_sample','module5.auto_k_validation_sample','module5.auto_k_sample_seed','module5.auto_k_seeds','module5.auto_k_stability_min','module5.auto_k_max_small_cluster_fraction','module5.auto_k_min_train_per_cluster','module5.auto_k_batch_size','module5.auto_k_max_iter','module5.auto_k_full_eval_max','module5.target_sub_per_cluster','module5.target_cluster_per_cell','module5.K_min','module5.K_max','module5.split_by_nucleus','module5.nucleus_threshold','module5.cap_ratio_base','module5.cap_min_keep','module5.cap_margin_tau','module5.cap_max_frac_soft','module5.train_sample_per_block','module5.assign_chunk_size','module5.mbk_batch_size','module5.mbk_max_iter','module5.cluster_rep','module5.cluster_rep_decell','module5.min_domain_size','module5.random_seed','module7.graph_unit','module8.dgi_unit',
    ),
    'm6': (
        'module6.pool_reps','module6.pool_normalize','module8.dgi_unit',
    ),
    # m7/m8 are build-time graph and DGI embedding steps
    'm7': (
        'execution.profile','execution.exact_max_metaspots','execution.sampled_max_metaspots','module7.graph_unit','module8.dgi_unit','module7.scvi_epochs','module7.scvi_n_latent','module7.scvi_lr','module7.scvi_batch_size','module7.neighbors','module7.neighbors.k_expr','module7.neighbors.k_geom','module7.neighbors.k_spatial','module7.alpha_expr','module7.alpha_geom','module7.same_cell_beta','module7.topk_prune','module7.edge_norm','module7.feature_scale','module7.metaspot_expr_n_components','module7.metaspot_alpha_expr','module7.metaspot_alpha_spatial','module7.metaspot_edge_norm','module7.metaspot_use_geometry_features','module7.metaspot_expr_cross_cell_only','module7.max_full_batch_nodes','module7.knn_backend','module7.ann_candidate_multiplier','module7.ann_query_chunk_nodes','module7.ann_train_sample','module7.faiss_hnsw_m','module7.faiss_nprobe','module7.faiss_pq_subvectors','module7.faiss_pq_bits','module7.faiss_use_gpu','module7.faiss_gpu_device',
    ),
    'm8': (
        'execution.profile','execution.exact_max_metaspots','execution.sampled_max_metaspots','module8.dgi_unit','module8.dgi_framework','module8.dgi_epochs','module8.enable_dgi_sage','module8.dgi_encoder','module8.dgi_layers','module8.dgi_hidden','module8.dgi_out','module8.dgi_dropout','module8.dgi_residual_alpha','module8.dgi_layer_norm','module8.dgi_device','module8.dgi_cpu_threads','module8.dgi_sparse_layout','module8.dgi_tf32','module8.dgi_corruption_mode','module8.dgi_local_loss_weight','module8.dgi_local_temperature','module8.dgi_local_samples','module8.dgi_local_graph_key','module8.dgi_cpu_backend','module8.dmgi_expression_graph_key','module8.dmgi_spatial_graph_key','module8.dmgi_expression_weight','module8.dmgi_spatial_weight','module8.dmgi_consensus_weight','module8.dmgi_gate_floor','module8.dmgi_fused_corruption_pass','module8.dmgi_early_stopping','module8.dmgi_min_epochs','module8.dmgi_patience','module8.dmgi_min_delta','module8.dgi_lr','module8.dgi_weight_decay','module8.dgi_clip_grad','module8.amp','module8.random_seed','module8.training_mode','module8.sampled_dgi_epochs','module8.sampled_batch_size','module8.sampled_expression_fanout','module8.sampled_spatial_fanout','module8.sampled_nodes_per_epoch','module8.sampled_inference_batch_size','module8.sampled_validation_nodes','module8.sampled_num_workers',
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
    if stage == 'build_anndata':
        sub['__build_pipeline_schema_version__'] = BUILD_PIPELINE_SCHEMA_VERSION
    elif stage == 'preprocess':
        sub['__preprocess_schema_version__'] = PREPROCESS_SCHEMA_VERSION
    elif stage == 'annotate':
        sub['__annotation_schema_version__'] = ANNOTATION_SCHEMA_VERSION
    return sub


def fingerprint_stage(
    cfg: Dict[str, Any],
    stage: str,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    sub = stage_config_subset(cfg, stage)
    if context:
        sub["__runtime_context__"] = dict(context)
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
    sub['__module_schema_version__'] = MODULE_SCHEMA_VERSIONS.get(module, 1)
    return sub


def fingerprint_module(
    cfg: Dict[str, Any],
    module: str,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    sub = module_config_subset(cfg, module)
    if context:
        sub["__runtime_context__"] = dict(context)
    try:
        payload = json.dumps(sub, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except Exception:
        payload = str(sub)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def fingerprint_modules_set(
    cfg: Dict[str, Any],
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    mods = ('m1','m2','m3','m5','m6','m7','m8')
    return {
        m: fingerprint_module(cfg, m, context=context)
        for m in mods
    }
