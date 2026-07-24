"""MetaSpot-level refinement of OMM evidence.

This module deliberately solves only one local question:

    does a cytoplasmic MetaSpot resemble an OMM-specific expression program
    more than its ER/Cytosol alternatives?

It does not run a sparse seven-way annotation model and it does not overwrite
the production annotation.  A deterministic two-component Gaussian mixture is
fit to a shrunken OMM-vs-alternative contrast.  The returned probability is a
posterior under that reference-conditioned mixture model, not a probability
that a transcript physically resides on the outer mitochondrial membrane.

The model fails closed.  If marker coverage, mixture separation, or agreement
with independent MetaDomain gates is inadequate, posterior values are left
missing and no soft prevalence is reported.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture


OMM_REFINEMENT_COLUMNS: Tuple[str, ...] = (
    "omm_local_contrast",
    "omm_local_contrast_shrunk",
    "omm_informative_count",
    "omm_reference_posterior",
    "omm_domain_gate",
    "omm_refined_state",
    "omm_ambiguous",
    "omm_confirmed_soft_weight",
    "omm_confirmed_or_shared_soft_weight",
)


def build_omm_contrast_marker_weights(
    gold: pd.DataFrame,
    *,
    gene_col: str = "Common_Gene",
    omm_col: str = "OMM",
    alternative_cols: Sequence[str] = ("ER", "Cytosol"),
    min_region_weight: float = 0.75,
    min_margin: float = 0.50,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """Build disjoint OMM-specific and alternative marker weights.

    The input may contain normalized reference weights or raw enrichment
    values.  A gene is OMM-specific only when OMM is both sufficiently strong
    and exceeds the strongest ER/Cytosol alternative by ``min_margin``.
    Alternative markers are selected by the symmetric rule.

    Returned weights are positive margins.  They are normalized separately
    after matching to an AnnData gene panel by :func:`refine_omm_metaspots`.
    """

    if not isinstance(gold, pd.DataFrame):
        raise TypeError("gold must be a pandas DataFrame")
    if gene_col not in gold.columns:
        if gold.index.name == gene_col:
            table = gold.reset_index()
        else:
            raise KeyError(f"gold is missing gene column {gene_col!r}")
    else:
        table = gold.copy()

    required = [omm_col, *alternative_cols]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise KeyError(f"gold is missing required columns: {missing}")

    genes = table[gene_col].astype(str).str.strip().str.upper()
    omm = pd.to_numeric(table[omm_col], errors="coerce")
    alternatives = table[list(alternative_cols)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    alternative = alternatives.max(axis=1, skipna=True)
    omm_margin = omm - alternative
    alternative_margin = alternative - omm

    positive_mask = (
        genes.ne("")
        & omm.ge(float(min_region_weight))
        & omm_margin.ge(float(min_margin))
    )
    negative_mask = (
        genes.ne("")
        & alternative.ge(float(min_region_weight))
        & alternative_margin.ge(float(min_margin))
    )

    positive = pd.Series(
        omm_margin.loc[positive_mask].to_numpy(dtype=float),
        index=genes.loc[positive_mask],
    )
    negative = pd.Series(
        alternative_margin.loc[negative_mask].to_numpy(dtype=float),
        index=genes.loc[negative_mask],
    )
    positive = positive.groupby(level=0, sort=False).max()
    negative = negative.groupby(level=0, sort=False).max()

    overlap = positive.index.intersection(negative.index)
    if len(overlap):
        positive = positive.drop(overlap)
        negative = negative.drop(overlap)

    positive_dict = {
        str(gene): float(weight)
        for gene, weight in positive.items()
        if np.isfinite(weight) and weight > 0
    }
    negative_dict = {
        str(gene): float(weight)
        for gene, weight in negative.items()
        if np.isfinite(weight) and weight > 0
    }
    metadata = {
        "gene_column": str(gene_col),
        "omm_column": str(omm_col),
        "alternative_columns": [str(value) for value in alternative_cols],
        "min_region_weight": float(min_region_weight),
        "min_margin": float(min_margin),
        "n_positive_reference_genes": len(positive_dict),
        "n_alternative_reference_genes": len(negative_dict),
        "n_overlapping_genes_removed": int(len(overlap)),
    }
    return positive_dict, negative_dict, metadata


def _canonical_marker_weights(
    weights: Mapping[str, float],
) -> Dict[str, float]:
    canonical: Dict[str, float] = {}
    for gene, value in weights.items():
        name = str(gene).strip().upper()
        if not name:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric) or numeric <= 0:
            continue
        canonical[name] = max(canonical.get(name, 0.0), numeric)
    return canonical


def _marker_hash(weights: Mapping[str, float]) -> str:
    payload = "\n".join(
        f"{gene}\t{float(weight):.12g}"
        for gene, weight in sorted(weights.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _panel_weight_vector(
    var_names: pd.Index,
    weights: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Map case-insensitive gene weights to the assay without double counting."""

    canonical = _canonical_marker_weights(weights)
    names = pd.Index(var_names.astype(str).str.strip().str.upper())
    positions: Dict[str, list[int]] = {}
    for position, gene in enumerate(names):
        if gene in canonical:
            positions.setdefault(str(gene), []).append(position)

    vector = np.zeros(len(names), dtype=np.float64)
    for gene, indices in positions.items():
        # Split a gene's reference weight over case-colliding assay features.
        per_feature = canonical[gene] / max(len(indices), 1)
        vector[np.asarray(indices, dtype=np.int64)] = per_feature
    total = float(vector.sum())
    if total > 0:
        vector /= total
    measured = np.flatnonzero(vector > 0)
    return vector, measured, len(positions)


def _normalize_state_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    token = str(value).strip().lower()
    token = token.replace("_", " ").replace("–", "-").replace("—", "-")
    token = re.sub(r"\s+", " ", token)
    return token


def _canonical_domain_gates(
    values: pd.Series,
    *,
    confirmed_states: Sequence[str],
    shared_states: Sequence[str],
    rejected_states: Sequence[str],
) -> pd.Series:
    confirmed = {_normalize_state_token(value) for value in confirmed_states}
    shared = {_normalize_state_token(value) for value in shared_states}
    rejected = {_normalize_state_token(value) for value in rejected_states}

    def convert(value: Any) -> str:
        token = _normalize_state_token(value)
        if token in confirmed:
            return "confirmed"
        if token in shared:
            return "shared"
        if token in rejected:
            return "rejected"
        return "unknown"

    return values.map(convert).astype("object")


def _matrix_is_count_like(matrix: Any, tolerance: float = 1e-4) -> bool:
    if sp.issparse(matrix):
        values = np.asarray(matrix.data)
    else:
        values = np.asarray(matrix).ravel()
    if values.size == 0:
        return True
    if not np.isfinite(values).all() or np.any(values < -tolerance):
        return False
    return bool(
        np.max(np.abs(values - np.rint(values)), initial=0.0)
        <= float(tolerance)
    )


def _as_csr_counts(adata: ad.AnnData, layer: Optional[str]) -> sp.csr_matrix:
    matrix = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(matrix):
        return matrix.tocsr().astype(np.float64, copy=False)
    return sp.csr_matrix(np.asarray(matrix, dtype=np.float64))


def _resolve_domain_gates(
    adata: ad.AnnData,
    domain_states: Optional[Mapping[str, Any]],
    *,
    domain_col: str,
    domain_state_col: str,
    confirmed_states: Sequence[str],
    shared_states: Sequence[str],
    rejected_states: Sequence[str],
) -> pd.Series:
    if domain_states is None:
        if domain_state_col not in adata.obs.columns:
            raw = pd.Series("", index=adata.obs_names, dtype="object")
        else:
            raw = adata.obs[domain_state_col].astype("object")
    else:
        if domain_col not in adata.obs.columns:
            raise KeyError(
                f"adata1.obs is missing domain column {domain_col!r}"
            )
        mapping = pd.Series(dict(domain_states), dtype="object")
        mapping.index = mapping.index.astype(str)
        raw = adata.obs[domain_col].astype(str).map(mapping)

    return _canonical_domain_gates(
        raw,
        confirmed_states=confirmed_states,
        shared_states=shared_states,
        rejected_states=rejected_states,
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _cell_bootstrap_interval(
    values: np.ndarray,
    denominators: np.ndarray,
    cell_values: pd.Series,
    *,
    iterations: int,
    random_state: int,
) -> Optional[Dict[str, float]]:
    """Return a deterministic cell-resampling interval for a soft fraction."""
    cells = pd.Series(cell_values, dtype="object").astype(str)
    codes, unique_cells = pd.factorize(cells, sort=True)
    n_cells = int(len(unique_cells))
    if n_cells < 2 or int(iterations) < 1:
        return None
    numerators = np.bincount(
        codes,
        weights=np.asarray(values, dtype=float)
        * np.asarray(denominators, dtype=float),
        minlength=n_cells,
    )
    cell_denominators = np.bincount(
        codes,
        weights=np.asarray(denominators, dtype=float),
        minlength=n_cells,
    )
    rng = np.random.default_rng(int(random_state))
    estimates = np.empty(int(iterations), dtype=float)
    for iteration in range(int(iterations)):
        sampled = rng.integers(0, n_cells, size=n_cells)
        denominator = float(cell_denominators[sampled].sum())
        estimates[iteration] = (
            float(numerators[sampled].sum()) / denominator
            if denominator > 0
            else np.nan
        )
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return None
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "lower_95": float(lower),
        "upper_95": float(upper),
        "iterations": int(iterations),
        "resampling_unit": "cell",
        "n_cells": n_cells,
    }


def refine_omm_metaspots(
    adata1: ad.AnnData,
    final_df: pd.DataFrame,
    positive_marker_weights: Mapping[str, float],
    alternative_marker_weights: Mapping[str, float],
    *,
    domain_states: Optional[Mapping[str, Any]] = None,
    domain_col: str = "meta-domain",
    domain_state_col: str = "subcell_omm_domain_state",
    metaspot_col: str = "metaspot",
    cell_col: str = "cell_id",
    geometry_col: str = "r_mean",
    geometry_min_r: float = -0.05,
    count_layer: Optional[str] = None,
    min_measured_markers: int = 20,
    min_informative_transcripts: int = 5,
    shrinkage_count: float = 20.0,
    contrast_pseudocount: float = 0.5,
    min_eligible_spots: int = 200,
    min_cells: int = 5,
    min_component_spots: float = 50.0,
    min_component_separation: float = 1.0,
    min_bic_improvement: float = 10.0,
    min_anchor_spots: int = 20,
    min_anchor_cells: int = 3,
    min_anchor_auc: float = 0.70,
    min_anchor_odds_ratio: float = 2.0,
    max_anchor_fisher_p: float = 0.05,
    posterior_high: float = 0.80,
    posterior_low: float = 0.20,
    confirmed_domain_states: Sequence[str] = (
        "confirmed",
        "confirmed OMM",
        "OMM-confirmed",
        "strict OMM",
        "OMM",
    ),
    shared_domain_states: Sequence[str] = (
        "shared",
        "ER-OMM shared",
        "OMM-ER shared",
        "ambiguous",
    ),
    rejected_domain_states: Sequence[str] = (
        "rejected",
        "OMM-rejected",
        "non-OMM",
        "Cytosol",
        "ER",
    ),
    random_state: int = 0,
    bootstrap_iterations: int = 500,
    gmm_n_init: int = 10,
    gmm_reg_covar: float = 1e-4,
    copy_adata: bool = False,
) -> Tuple[ad.AnnData, pd.DataFrame, Dict[str, Any]]:
    """Refine OMM evidence at MetaSpot resolution.

    Parameters
    ----------
    adata1
        MetaSpot-by-gene AnnData with raw count-like values in ``X`` or
        ``count_layer``.
    final_df
        Transcript table containing ``metaspot_col``.  Results are mapped by
        this key, never by MetaDomain.
    positive_marker_weights, alternative_marker_weights
        Disjoint OMM-specific and ER/Cytosol-alternative marker weights.
    domain_states
        Mapping from MetaDomain identifier to a confirmed/shared/rejected
        state.  When omitted, ``adata1.obs[domain_state_col]`` is used.

    Returns
    -------
    updated_adata1, updated_final_df, metadata
        The production annotation is never overwritten.  New columns listed
        in :data:`OMM_REFINEMENT_COLUMNS` are added to both outputs.

    Notes
    -----
    ``omm_reference_posterior`` is conditional on an identifiable two-Gaussian
    model of the OMM contrast.  It is intentionally left as NaN when the model
    cannot be distinguished from a single component or fails alignment with
    MetaDomain anchors.
    """

    if not isinstance(adata1, ad.AnnData):
        raise TypeError("adata1 must be an AnnData object")
    if not isinstance(final_df, pd.DataFrame):
        raise TypeError("final_df must be a pandas DataFrame")
    if metaspot_col not in final_df.columns:
        raise KeyError(f"final_df is missing {metaspot_col!r}")
    if not 0.0 < float(posterior_high) < 1.0:
        raise ValueError("posterior_high must lie strictly between zero and one")
    if not 0.0 < float(posterior_low) < 1.0:
        raise ValueError("posterior_low must lie strictly between zero and one")
    if float(posterior_low) >= float(posterior_high):
        raise ValueError("posterior_low must be smaller than posterior_high")
    if float(shrinkage_count) < 0:
        raise ValueError("shrinkage_count must be non-negative")
    if float(contrast_pseudocount) <= 0:
        raise ValueError("contrast_pseudocount must be positive")
    if int(bootstrap_iterations) < 0:
        raise ValueError("bootstrap_iterations must be non-negative")

    output_adata = adata1.copy() if copy_adata else adata1
    output_final = final_df.copy()
    n_spots = output_adata.n_obs

    if metaspot_col in output_adata.obs.columns:
        if output_adata.obs[metaspot_col].isna().any():
            raise ValueError(
                "MetaSpot identifiers in adata1 contain missing values"
            )
        spot_ids = pd.Index(
            output_adata.obs[metaspot_col].astype(str),
            name=metaspot_col,
        )
    else:
        spot_ids = pd.Index(
            output_adata.obs_names.astype(str),
            name=metaspot_col,
        )
    if not spot_ids.is_unique:
        raise ValueError("MetaSpot identifiers in adata1 must be unique")

    if output_final[metaspot_col].isna().any():
        raise ValueError("MetaSpot identifiers in final_df contain missing values")
    final_spot_ids = output_final[metaspot_col].astype(str)
    missing_spots = pd.Index(pd.unique(final_spot_ids)).difference(spot_ids)
    if len(missing_spots):
        raise ValueError(
            "final_df contains MetaSpots absent from adata1; "
            f"examples={missing_spots[:5].tolist()}"
        )

    domain_gate = _resolve_domain_gates(
        output_adata,
        domain_states,
        domain_col=domain_col,
        domain_state_col=domain_state_col,
        confirmed_states=confirmed_domain_states,
        shared_states=shared_domain_states,
        rejected_states=rejected_domain_states,
    )
    domain_gate.index = output_adata.obs_names
    gate_values = domain_gate.to_numpy(dtype=object)

    raw_contrast = np.full(n_spots, np.nan, dtype=np.float64)
    shrunk_contrast = np.full(n_spots, np.nan, dtype=np.float64)
    informative_count = np.zeros(n_spots, dtype=np.int64)
    posterior = np.full(n_spots, np.nan, dtype=np.float64)
    refined_state = np.full(n_spots, "Unidentifiable", dtype=object)
    ambiguous = np.zeros(n_spots, dtype=bool)
    confirmed_soft = np.full(n_spots, np.nan, dtype=np.float64)
    possible_soft = np.full(n_spots, np.nan, dtype=np.float64)

    positive_canonical = _canonical_marker_weights(positive_marker_weights)
    alternative_canonical = _canonical_marker_weights(
        alternative_marker_weights
    )
    overlapping_markers = set(positive_canonical).intersection(
        alternative_canonical
    )
    for gene in overlapping_markers:
        positive_canonical.pop(gene, None)
        alternative_canonical.pop(gene, None)

    positive_vector, positive_idx, n_positive_measured = _panel_weight_vector(
        output_adata.var_names,
        positive_canonical,
    )
    alternative_vector, alternative_idx, n_alternative_measured = (
        _panel_weight_vector(
            output_adata.var_names,
            alternative_canonical,
        )
    )

    checks: Dict[str, Any] = {}
    failure_reasons: list[str] = []
    warnings: list[str] = []

    def fail(reason: str) -> None:
        if reason not in failure_reasons:
            failure_reasons.append(reason)

    try:
        matrix_source = (
            output_adata.layers[count_layer]
            if count_layer is not None
            else output_adata.X
        )
    except KeyError:
        matrix_source = None
        fail("missing_count_layer")

    count_like = (
        matrix_source is not None and _matrix_is_count_like(matrix_source)
    )
    checks["count_like_matrix"] = bool(count_like)
    if not count_like:
        fail("matrix_not_count_like")

    checks["n_positive_measured_markers"] = int(n_positive_measured)
    checks["n_alternative_measured_markers"] = int(n_alternative_measured)
    checks["min_measured_markers"] = int(min_measured_markers)
    if n_positive_measured < int(min_measured_markers):
        fail("insufficient_positive_markers")
    if n_alternative_measured < int(min_measured_markers):
        fail("insufficient_alternative_markers")

    if geometry_col in output_adata.obs.columns:
        geometry = pd.to_numeric(
            output_adata.obs[geometry_col],
            errors="coerce",
        ).to_numpy(dtype=float)
        geometry_feasible = (
            np.isfinite(geometry) & (geometry >= float(geometry_min_r))
        )
    else:
        geometry_feasible = np.ones(n_spots, dtype=bool)
        warnings.append("geometry_column_missing_all_spots_treated_as_feasible")

    known_cytoplasmic_gate = np.isin(
        gate_values,
        ["confirmed", "shared", "rejected"],
    )

    total_counts = np.zeros(n_spots, dtype=np.float64)
    eligible = np.zeros(n_spots, dtype=bool)
    model_info: Dict[str, Any] = {}
    anchor_info: Dict[str, Any] = {}

    if not failure_reasons:
        counts = _as_csr_counts(output_adata, count_layer)
        total_counts = np.asarray(counts.sum(axis=1)).ravel()
        positive_signal = np.asarray(counts @ positive_vector).ravel()
        alternative_signal = np.asarray(counts @ alternative_vector).ravel()
        informative_idx = np.union1d(positive_idx, alternative_idx)
        informative_count = np.asarray(
            counts[:, informative_idx].sum(axis=1)
        ).ravel().astype(np.int64, copy=False)

        pseudocount = float(contrast_pseudocount)
        raw_contrast = np.log(
            (positive_signal + pseudocount)
            / (alternative_signal + pseudocount)
        )
        preliminary = (
            geometry_feasible
            & known_cytoplasmic_gate
            & (informative_count >= int(min_informative_transcripts))
            & np.isfinite(raw_contrast)
        )
        if preliminary.any():
            center = float(np.median(raw_contrast[preliminary]))
        else:
            center = 0.0
        reliability = informative_count.astype(np.float64) / (
            informative_count.astype(np.float64)
            + float(shrinkage_count)
            + 1e-12
        )
        shrunk_contrast = center + reliability * (raw_contrast - center)
        eligible = preliminary & np.isfinite(shrunk_contrast)

        n_eligible = int(eligible.sum())
        checks["n_eligible_spots"] = n_eligible
        checks["min_eligible_spots"] = int(min_eligible_spots)
        if n_eligible < int(min_eligible_spots):
            fail("insufficient_eligible_spots")

        if cell_col in output_adata.obs.columns:
            eligible_cells = int(
                output_adata.obs.loc[eligible, cell_col].astype(str).nunique()
            )
        else:
            eligible_cells = 0
            warnings.append("cell_column_missing")
        checks["n_eligible_cells"] = eligible_cells
        checks["min_cells"] = int(min_cells)
        if eligible_cells < int(min_cells):
            fail("insufficient_cells")

        unique_scores = int(np.unique(shrunk_contrast[eligible]).size)
        checks["n_unique_eligible_scores"] = unique_scores
        if unique_scores < 2:
            fail("single_valued_contrast")

    candidate_posterior: Optional[np.ndarray] = None
    if not failure_reasons:
        values = shrunk_contrast[eligible].reshape(-1, 1)
        try:
            one_component = GaussianMixture(
                n_components=1,
                covariance_type="full",
                n_init=max(1, int(gmm_n_init)),
                random_state=int(random_state),
                reg_covar=float(gmm_reg_covar),
                max_iter=500,
            ).fit(values)
            two_component = GaussianMixture(
                n_components=2,
                covariance_type="full",
                n_init=max(1, int(gmm_n_init)),
                random_state=int(random_state),
                reg_covar=float(gmm_reg_covar),
                max_iter=500,
            ).fit(values)
        except Exception as exc:  # pragma: no cover - defensive numerical path
            fail(f"gmm_fit_failed:{type(exc).__name__}")
        else:
            means = two_component.means_.reshape(-1)
            variances = two_component.covariances_.reshape(-1)
            high_component = int(np.argmax(means))
            low_component = 1 - high_component
            local_posterior = two_component.predict_proba(values)[
                :,
                high_component,
            ]
            candidate_posterior = np.full(n_spots, np.nan, dtype=np.float64)
            candidate_posterior[eligible] = local_posterior

            pooled_sd = math.sqrt(
                max(
                    (
                        float(variances[high_component])
                        + float(variances[low_component])
                    )
                    / 2.0,
                    1e-12,
                )
            )
            separation = abs(
                float(means[high_component])
                - float(means[low_component])
            ) / pooled_sd
            bic_one = float(one_component.bic(values))
            bic_two = float(two_component.bic(values))
            bic_improvement = bic_one - bic_two
            effective_high = float(np.sum(local_posterior))
            effective_low = float(np.sum(1.0 - local_posterior))

            model_info = {
                "converged": bool(two_component.converged_),
                "high_component_mean": float(means[high_component]),
                "low_component_mean": float(means[low_component]),
                "high_component_sd": math.sqrt(
                    max(float(variances[high_component]), 0.0)
                ),
                "low_component_sd": math.sqrt(
                    max(float(variances[low_component]), 0.0)
                ),
                "high_component_weight": float(
                    two_component.weights_[high_component]
                ),
                "low_component_weight": float(
                    two_component.weights_[low_component]
                ),
                "effective_high_spots": effective_high,
                "effective_low_spots": effective_low,
                "component_separation": float(separation),
                "bic_one_component": bic_one,
                "bic_two_component": bic_two,
                "bic_improvement": float(bic_improvement),
            }
            if not two_component.converged_:
                fail("two_component_gmm_not_converged")
            if separation < float(min_component_separation):
                fail("components_not_separated")
            if bic_improvement < float(min_bic_improvement):
                fail("two_components_not_preferred_by_bic")
            if min(effective_high, effective_low) < float(
                min_component_spots
            ):
                fail("insufficient_effective_component_size")

    if candidate_posterior is not None:
        confirmed_anchor = eligible & (gate_values == "confirmed")
        rejected_anchor = eligible & (gate_values == "rejected")
        n_confirmed_anchor = int(confirmed_anchor.sum())
        n_rejected_anchor = int(rejected_anchor.sum())
        if cell_col in output_adata.obs.columns:
            n_confirmed_cells = int(
                output_adata.obs.loc[confirmed_anchor, cell_col]
                .astype(str)
                .nunique()
            )
            n_rejected_cells = int(
                output_adata.obs.loc[rejected_anchor, cell_col]
                .astype(str)
                .nunique()
            )
        else:
            n_confirmed_cells = 0
            n_rejected_cells = 0

        anchor_info.update(
            {
                "n_confirmed_anchor_spots": n_confirmed_anchor,
                "n_rejected_anchor_spots": n_rejected_anchor,
                "n_confirmed_anchor_cells": n_confirmed_cells,
                "n_rejected_anchor_cells": n_rejected_cells,
            }
        )
        if (
            n_confirmed_anchor < int(min_anchor_spots)
            or n_rejected_anchor < int(min_anchor_spots)
        ):
            fail("insufficient_anchor_spots")
        if (
            n_confirmed_cells < int(min_anchor_cells)
            or n_rejected_cells < int(min_anchor_cells)
        ):
            fail("insufficient_anchor_cells")

        if n_confirmed_anchor and n_rejected_anchor:
            labels = np.concatenate(
                [
                    np.ones(n_confirmed_anchor, dtype=np.int8),
                    np.zeros(n_rejected_anchor, dtype=np.int8),
                ]
            )
            scores = np.concatenate(
                [
                    candidate_posterior[confirmed_anchor],
                    candidate_posterior[rejected_anchor],
                ]
            )
            anchor_auc = float(roc_auc_score(labels, scores))
            confirmed_high = int(
                np.sum(
                    candidate_posterior[confirmed_anchor]
                    >= float(posterior_high)
                )
            )
            rejected_high = int(
                np.sum(
                    candidate_posterior[rejected_anchor]
                    >= float(posterior_high)
                )
            )
            table = np.asarray(
                [
                    [
                        confirmed_high,
                        n_confirmed_anchor - confirmed_high,
                    ],
                    [
                        rejected_high,
                        n_rejected_anchor - rejected_high,
                    ],
                ],
                dtype=np.int64,
            )
            odds_ratio, fisher_p = fisher_exact(table)
            odds_ratio_value = float(odds_ratio)
            fisher_p_value = float(fisher_p)
            anchor_info.update(
                {
                    "auc": anchor_auc,
                    "high_posterior_fisher_table": table.tolist(),
                    "high_posterior_odds_ratio": odds_ratio_value,
                    "high_posterior_fisher_p": fisher_p_value,
                }
            )
            odds_supported = (
                odds_ratio_value >= float(min_anchor_odds_ratio)
                and fisher_p_value <= float(max_anchor_fisher_p)
            )
            auc_supported = anchor_auc >= float(min_anchor_auc)
            anchor_info["validation_passed"] = bool(
                odds_supported or auc_supported
            )
            if not (odds_supported or auc_supported):
                fail("domain_anchor_validation_failed")
        else:
            anchor_info["validation_passed"] = False
            fail("domain_anchor_validation_failed")

    identifiable = candidate_posterior is not None and not failure_reasons
    if identifiable:
        posterior = candidate_posterior
        finite_posterior = np.isfinite(posterior)

        refined_state[:] = "DomainStateMissing"
        refined_state[~geometry_feasible] = "GeometryInfeasible"
        refined_state[
            geometry_feasible
            & known_cytoplasmic_gate
            & (informative_count < int(min_informative_transcripts))
        ] = "InsufficientEvidence"

        confirmed = finite_posterior & (gate_values == "confirmed")
        shared = finite_posterior & (gate_values == "shared")
        rejected = finite_posterior & (gate_values == "rejected")
        high = posterior >= float(posterior_high)
        low = posterior <= float(posterior_low)
        middle = finite_posterior & ~(high | low)

        refined_state[confirmed & high] = "OMM-supported"
        refined_state[confirmed & middle] = "ER-OMM shared"
        refined_state[confirmed & low] = "OMM-not-supported"
        refined_state[shared & ~low] = "ER-OMM shared"
        refined_state[shared & low] = "OMM-not-supported"
        refined_state[rejected] = "OMM-rejected-by-domain"
        ambiguous = refined_state == "ER-OMM shared"

        confirmed_soft[:] = 0.0
        possible_soft[:] = 0.0
        confirmed_soft[confirmed] = posterior[confirmed]
        possible_soft[confirmed | shared] = posterior[confirmed | shared]
        # A confirmed/shared domain with inadequate local information remains
        # missing rather than being silently counted as zero evidence.
        target_gate = np.isin(gate_values, ["confirmed", "shared"])
        missing_target = target_gate & ~finite_posterior
        confirmed_soft[
            missing_target & (gate_values == "confirmed")
        ] = np.nan
        possible_soft[missing_target] = np.nan

    output_adata.obs["omm_local_contrast"] = raw_contrast.astype(np.float32)
    output_adata.obs["omm_local_contrast_shrunk"] = (
        shrunk_contrast.astype(np.float32)
    )
    output_adata.obs["omm_informative_count"] = informative_count.astype(
        np.int32
    )
    output_adata.obs["omm_reference_posterior"] = posterior.astype(np.float32)
    output_adata.obs["omm_domain_gate"] = gate_values
    output_adata.obs["omm_refined_state"] = refined_state
    output_adata.obs["omm_ambiguous"] = ambiguous.astype(bool)
    output_adata.obs["omm_confirmed_soft_weight"] = confirmed_soft.astype(
        np.float32
    )
    output_adata.obs[
        "omm_confirmed_or_shared_soft_weight"
    ] = possible_soft.astype(np.float32)

    for column in OMM_REFINEMENT_COLUMNS:
        mapping = pd.Series(
            output_adata.obs[column].to_numpy(),
            index=spot_ids,
        )
        output_final[column] = final_spot_ids.map(mapping).to_numpy()

    if identifiable:
        confirmed_values = np.nan_to_num(confirmed_soft, nan=0.0)
        possible_values = np.nan_to_num(possible_soft, nan=0.0)
        total_transcripts = float(total_counts.sum())
        prevalence = {
            "metaspot_soft_confirmed": float(
                confirmed_values.sum() / max(n_spots, 1)
            ),
            "metaspot_soft_confirmed_or_shared": float(
                possible_values.sum() / max(n_spots, 1)
            ),
            "transcript_weighted_soft_confirmed": (
                float(
                    np.dot(total_counts, confirmed_values)
                    / total_transcripts
                )
                if total_transcripts > 0
                else None
            ),
            "transcript_weighted_soft_confirmed_or_shared": (
                float(
                    np.dot(total_counts, possible_values)
                    / total_transcripts
                )
                if total_transcripts > 0
                else None
            ),
            "metaspot_posterior_coverage": float(
                np.isfinite(posterior).mean()
            ),
            "transcript_posterior_coverage": (
                float(
                    total_counts[np.isfinite(posterior)].sum()
                    / total_transcripts
                )
                if total_transcripts > 0
                else None
            ),
        }
        if cell_col in output_adata.obs.columns:
            cell_values = output_adata.obs[cell_col]
            prevalence_ci = {
                "metaspot_soft_confirmed": _cell_bootstrap_interval(
                    confirmed_values,
                    np.ones(n_spots, dtype=float),
                    cell_values,
                    iterations=bootstrap_iterations,
                    random_state=random_state + 101,
                ),
                "metaspot_soft_confirmed_or_shared": (
                    _cell_bootstrap_interval(
                        possible_values,
                        np.ones(n_spots, dtype=float),
                        cell_values,
                        iterations=bootstrap_iterations,
                        random_state=random_state + 102,
                    )
                ),
                "transcript_weighted_soft_confirmed": (
                    _cell_bootstrap_interval(
                        confirmed_values,
                        total_counts,
                        cell_values,
                        iterations=bootstrap_iterations,
                        random_state=random_state + 103,
                    )
                ),
                "transcript_weighted_soft_confirmed_or_shared": (
                    _cell_bootstrap_interval(
                        possible_values,
                        total_counts,
                        cell_values,
                        iterations=bootstrap_iterations,
                        random_state=random_state + 104,
                    )
                ),
            }
        else:
            prevalence_ci = {}
    else:
        prevalence = {
            "metaspot_soft_confirmed": None,
            "metaspot_soft_confirmed_or_shared": None,
            "transcript_weighted_soft_confirmed": None,
            "transcript_weighted_soft_confirmed_or_shared": None,
            "metaspot_posterior_coverage": 0.0,
            "transcript_posterior_coverage": 0.0,
        }
        prevalence_ci = {}

    state_counts = (
        pd.Series(refined_state, dtype="object")
        .value_counts(dropna=False)
        .to_dict()
    )
    gate_counts = (
        pd.Series(gate_values, dtype="object")
        .value_counts(dropna=False)
        .to_dict()
    )
    metadata: Dict[str, Any] = {
        "method": "deterministic_two_component_gmm",
        "identifiable": bool(identifiable),
        "failure_reasons": list(failure_reasons),
        "warnings": list(warnings),
        "posterior_interpretation": (
            "Reference-conditioned posterior for the high OMM-specific "
            "contrast component. It is not a physical OMM localization "
            "probability and is not a calibrated biological class posterior."
        ),
        "domain_gate_interpretation": (
            "The local model may refine confirmed/shared MetaDomains but "
            "cannot create an OMM call in a rejected or unknown domain."
        ),
        "n_metaspots": int(n_spots),
        "n_transcripts_from_count_matrix": int(round(float(total_counts.sum()))),
        "marker_reference": {
            "positive_input_genes": len(positive_canonical),
            "alternative_input_genes": len(alternative_canonical),
            "positive_measured_genes": int(n_positive_measured),
            "alternative_measured_genes": int(n_alternative_measured),
            "overlapping_genes_removed": int(len(overlapping_markers)),
            "positive_sha256": _marker_hash(positive_canonical),
            "alternative_sha256": _marker_hash(alternative_canonical),
        },
        "parameters": {
            "geometry_min_r": float(geometry_min_r),
            "min_measured_markers": int(min_measured_markers),
            "min_informative_transcripts": int(
                min_informative_transcripts
            ),
            "shrinkage_count": float(shrinkage_count),
            "contrast_pseudocount": float(contrast_pseudocount),
            "min_eligible_spots": int(min_eligible_spots),
            "min_cells": int(min_cells),
            "min_component_spots": float(min_component_spots),
            "min_component_separation": float(
                min_component_separation
            ),
            "min_bic_improvement": float(min_bic_improvement),
            "min_anchor_spots": int(min_anchor_spots),
            "min_anchor_cells": int(min_anchor_cells),
            "min_anchor_auc": float(min_anchor_auc),
            "min_anchor_odds_ratio": float(min_anchor_odds_ratio),
            "max_anchor_fisher_p": float(max_anchor_fisher_p),
            "posterior_high": float(posterior_high),
            "posterior_low": float(posterior_low),
            "random_state": int(random_state),
            "bootstrap_iterations": int(bootstrap_iterations),
            "gmm_n_init": int(gmm_n_init),
            "gmm_reg_covar": float(gmm_reg_covar),
        },
        "checks": checks,
        "model": model_info,
        "anchor_validation": anchor_info,
        "domain_gate_counts": {
            str(key): int(value) for key, value in gate_counts.items()
        },
        "refined_state_counts": {
            str(key): int(value) for key, value in state_counts.items()
        },
        "soft_prevalence": prevalence,
        "soft_prevalence_cell_bootstrap_95ci": prevalence_ci,
    }
    return output_adata, output_final, metadata


__all__ = [
    "OMM_REFINEMENT_COLUMNS",
    "build_omm_contrast_marker_weights",
    "refine_omm_metaspots",
]
