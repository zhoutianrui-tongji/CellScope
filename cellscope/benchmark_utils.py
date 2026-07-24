"""Shared, dependency-light helpers for the benchmark entry points."""

from __future__ import annotations

import hashlib
import os
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


_FINE_ANNOTATION_ALIASES = {
    "nucleus": "Nucleus",
    "nuclear": "Nucleus",
    "nucleolus": "Nucleolus",
    "lamina": "Lamina",
    "nuclearlamina": "Lamina",
    "nuclearpore": "Nuclear Pore",
    "nuclearenvelope": "Nuclear Pore",
    "cytosol": "Cytosol",
    "cytoplasm": "Cytosol",
    "cytoplasmic": "Cytosol",
    "er": "ER",
    "erlumen": "ER",
    "erm": "ER",
    "endoplasmicreticulum": "ER",
    "omm": "OMM",
    "outermitochondrialmembrane": "OMM",
    "mitochondria": "OMM",
    "mitochondrial": "OMM",
    "erommshared": "ER-OMM shared",
    "ommershared": "ER-OMM shared",
}

_NUCLEAR_FINE_CLASSES = {
    "Nucleus",
    "Nucleolus",
    "Lamina",
    "Nuclear Pore",
}

_CYTOPLASMIC_FINE_CLASSES = {
    "Cytosol",
    "ER",
    "OMM",
    "ER-OMM shared",
}

RAPX_BENCHMARK_TRACKS = (
    "production",
    "candidate",
    "exploratory",
    "strict",
)


def normalize_benchmark_metaspot_column(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize the historical ``cell_sub`` key to ``metaspot``."""

    if "metaspot" in frame.columns:
        return frame
    if "cell_sub" not in frame.columns:
        raise KeyError(
            "Benchmark data must contain 'metaspot' or legacy 'cell_sub'."
        )
    return frame.rename(columns={"cell_sub": "metaspot"})


def _normalize_fine_annotation(value: object) -> str:
    if pd.isna(value):
        return "Unknown"

    compact = (
        str(value)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("/", "")
    )
    return _FINE_ANNOTATION_ALIASES.get(compact, "Unknown")


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator_array, denominator_array = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.zeros(numerator_array.shape, dtype=np.float64)
    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=denominator_array != 0,
    )
    return result


def _binary_metrics_from_counts(
    tn: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tp: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate sklearn-compatible binary metrics from confusion counts."""

    tn = np.asarray(tn, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.float64)

    support_cytoplasmic = tn + fp
    support_nuclear = fn + tp
    predicted_cytoplasmic = tn + fn
    predicted_nuclear = fp + tp
    total = support_cytoplasmic + support_nuclear

    cytoplasmic_precision = _safe_ratio(tn, predicted_cytoplasmic)
    cytoplasmic_recall = _safe_ratio(tn, support_cytoplasmic)
    cytoplasmic_f1 = _safe_ratio(2 * tn, 2 * tn + fp + fn)

    nuclear_precision = _safe_ratio(tp, predicted_nuclear)
    nuclear_recall = _safe_ratio(tp, support_nuclear)
    nuclear_f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)

    present_class_count = (
        (support_cytoplasmic > 0).astype(np.float64)
        + (support_nuclear > 0).astype(np.float64)
    )
    balanced_accuracy = _safe_ratio(
        cytoplasmic_recall * (support_cytoplasmic > 0)
        + nuclear_recall * (support_nuclear > 0),
        present_class_count,
    )

    accuracy = _safe_ratio(tn + tp, total)
    macro_f1 = (cytoplasmic_f1 + nuclear_f1) / 2
    weighted_f1 = _safe_ratio(
        cytoplasmic_f1 * support_cytoplasmic
        + nuclear_f1 * support_nuclear,
        total,
    )

    mcc_denominator = np.sqrt(
        predicted_nuclear
        * support_nuclear
        * support_cytoplasmic
        * predicted_cytoplasmic
    )
    mcc = _safe_ratio(tp * tn - fp * fn, mcc_denominator)

    expected_accuracy = _safe_ratio(
        support_cytoplasmic * predicted_cytoplasmic
        + support_nuclear * predicted_nuclear,
        total**2,
    )
    kappa_denominator = 1 - expected_accuracy
    cohen_kappa = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(
        accuracy - expected_accuracy,
        kappa_denominator,
        out=cohen_kappa,
        where=kappa_denominator != 0,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "MCC": mcc,
        "Cohen_kappa": cohen_kappa,
        "cytoplasmic_precision": cytoplasmic_precision,
        "cytoplasmic_recall": cytoplasmic_recall,
        "cytoplasmic_f1": cytoplasmic_f1,
        "nuclear_precision": nuclear_precision,
        "nuclear_recall": nuclear_recall,
        "nuclear_f1": nuclear_f1,
    }


def evaluate_binary_predictions_fast(
    df: pd.DataFrame,
    method_name: str,
    pred_col: str,
    *,
    true_col: str = "True_Class",
    cell_col: str = "cell_id",
    metaspot_col: str = "metaspot",
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    """Evaluate transcript predictions without Python-per-row metric calls.

    Fine-label normalization is performed once per distinct raw label. Cell
    metrics are then derived from one grouped confusion-count aggregation. The
    function intentionally returns only the summary, cell metrics, and global
    per-class metrics; it does not duplicate the transcript-level input frame.
    """

    required_columns = [true_col, cell_col, metaspot_col, pred_col]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(f"{method_name}: missing {missing_columns}")

    raw_prediction = pd.Categorical(df[pred_col])
    normalized_categories = np.asarray(
        [_normalize_fine_annotation(value) for value in raw_prediction.categories],
        dtype=object,
    )
    category_binary_codes = np.full(len(normalized_categories), -1, dtype=np.int8)
    category_binary_codes[
        np.isin(normalized_categories, list(_CYTOPLASMIC_FINE_CLASSES))
    ] = 0
    category_binary_codes[
        np.isin(normalized_categories, list(_NUCLEAR_FINE_CLASSES))
    ] = 1

    prediction_codes = np.full(len(df), -1, dtype=np.int8)
    non_missing_prediction = raw_prediction.codes >= 0
    prediction_codes[non_missing_prediction] = category_binary_codes[
        raw_prediction.codes[non_missing_prediction]
    ]

    true_values = df[true_col].to_numpy(copy=False)
    true_codes = np.full(len(df), -1, dtype=np.int8)
    true_codes[true_values == "Cytoplasmic"] = 0
    true_codes[true_values == "Nuclear"] = 1

    valid_truth = true_codes >= 0
    assigned = prediction_codes >= 0
    evaluated = valid_truth & assigned
    if not evaluated.any():
        raise ValueError(f"{method_name}: all predictions are Unknown")

    tn_mask = evaluated & (true_codes == 0) & (prediction_codes == 0)
    fp_mask = evaluated & (true_codes == 0) & (prediction_codes == 1)
    fn_mask = evaluated & (true_codes == 1) & (prediction_codes == 0)
    tp_mask = evaluated & (true_codes == 1) & (prediction_codes == 1)

    tn = int(tn_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    tp = int(tp_mask.sum())
    global_metrics = _binary_metrics_from_counts(tn, fp, fn, tp)

    n_total = int(valid_truth.sum())
    n_evaluated = int(evaluated.sum())
    summary: dict[str, float | int | str] = {
        "method": method_name,
        "n_transcripts_total": n_total,
        "n_transcripts_eval": n_evaluated,
        "n_cells": int(df.loc[evaluated, cell_col].nunique()),
        "n_metaspots": int(df.loc[evaluated, metaspot_col].nunique()),
        "coverage": float(n_evaluated / n_total),
        "unknown_fraction": float(1 - n_evaluated / n_total),
        "overall_accuracy": float((tn + tp) / n_total),
        **{
            name: float(np.asarray(value))
            for name, value in global_metrics.items()
        },
    }

    indicator_frame = pd.DataFrame(
        {
            "n_transcripts_total": valid_truth,
            "n_transcripts": evaluated,
            "tn": tn_mask,
            "fp": fp_mask,
            "fn": fn_mask,
            "tp": tp_mask,
        }
    )
    grouped = (
        indicator_frame
        .groupby(df[cell_col], sort=False, observed=True)
        .sum()
    )
    grouped = grouped.loc[grouped["n_transcripts"].gt(0)].copy()

    cell_metric_values = _binary_metrics_from_counts(
        grouped["tn"].to_numpy(),
        grouped["fp"].to_numpy(),
        grouped["fn"].to_numpy(),
        grouped["tp"].to_numpy(),
    )
    cell_metrics = pd.DataFrame(
        {
            "method": method_name,
            cell_col: grouped.index.to_numpy(),
            "n_transcripts_total": grouped["n_transcripts_total"].to_numpy(),
            "n_transcripts": grouped["n_transcripts"].to_numpy(),
            "coverage": (
                grouped["n_transcripts"] / grouped["n_transcripts_total"]
            ).to_numpy(),
            "overall_accuracy": (
                (grouped["tn"] + grouped["tp"])
                / grouped["n_transcripts_total"]
            ).to_numpy(),
            **cell_metric_values,
        }
    )

    per_class = pd.DataFrame(
        {
            "class": ["Cytoplasmic", "Nuclear"],
            "precision": [
                float(global_metrics["cytoplasmic_precision"]),
                float(global_metrics["nuclear_precision"]),
            ],
            "recall": [
                float(global_metrics["cytoplasmic_recall"]),
                float(global_metrics["nuclear_recall"]),
            ],
            "f1": [
                float(global_metrics["cytoplasmic_f1"]),
                float(global_metrics["nuclear_f1"]),
            ],
            "support": [tn + fp, fn + tp],
            "method": method_name,
        }
    )
    return summary, cell_metrics, per_class


def _sha256_file(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_frame(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(
        frame,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_active_rapx_gold_table(
    target_var_names: pd.Index,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the same RAP-X reference used by the active CellScope config."""

    from .config import get_default_params_path, load_params_yaml
    from .pipeline import (
        _configure_rnalocate_augmentation,
        _load_default_gold_tables,
        _load_manual_marker_data,
        build_gold_from_rnalocate_l1,
        harmonize_gold_gene_case,
        merge_gold,
        project_l1_to_seven,
    )

    config = load_params_yaml() or {}
    annotation = config.get("annotation", {}) or {}
    species_value = annotation.get(
        "rnalocate_species",
        ["Homo sapiens", "Mus musculus"],
    )
    if species_value is None:
        species = None
    elif isinstance(species_value, str):
        species = tuple(
            value.strip()
            for value in species_value.split(",")
            if value.strip()
        )
    else:
        species = tuple(str(value) for value in species_value)

    augment_er = bool(annotation.get("rnalocate_augment_er", False))
    use_manual_markers = bool(annotation.get("use_manual_markers", False))

    apex_filter = bool(
        annotation.get("apex_omm_specificity_filter", True)
    )
    apex_min_log2fc = float(
        annotation.get("apex_omm_min_log2fc", 0.75)
    )
    apex_min_delta = float(
        annotation.get("apex_omm_min_delta", 0.50)
    )
    gold_apex, rna_locate, apex_omm_audit = _load_default_gold_tables(
        apex_omm_specificity_filter=apex_filter,
        apex_omm_min_log2fc=apex_min_log2fc,
        apex_omm_min_delta=apex_min_delta,
        return_omm_audit=True,
    )
    l1 = build_gold_from_rnalocate_l1(
        rna_locate,
        species=species,
        rna_types=("mRNA",),
        agg_mode="prob_or",
        score_floor=0.6,
        use_pubmed_weight=True,
    )
    gold_rna = _configure_rnalocate_augmentation(
        project_l1_to_seven(
            l1,
            mode="strict",
            mito_non_omm=None,
        ),
        augment_er=augment_er,
    )
    gold = merge_gold(gold_apex, gold_rna, fuse="max")

    manual_used = False
    if use_manual_markers:
        manual = _load_manual_marker_data()
        if manual is not None:
            gold = merge_gold(gold, manual, fuse="max")
            manual_used = True

    gold = harmonize_gold_gene_case(
        gold,
        target_var_names=target_var_names,
    )
    gene_column = (
        "Common_Gene"
        if "Common_Gene" in gold.columns
        else str(gold.columns[0])
    )
    hash_frame = gold.copy()
    hash_frame[gene_column] = (
        hash_frame[gene_column].astype(str).str.strip().str.upper()
    )
    hash_frame = hash_frame.sort_values(gene_column).reset_index(drop=True)

    config_path = os.environ.get(
        "CELLSCOPE_CONFIG",
        get_default_params_path(),
    )
    provenance: dict[str, object] = {
        "config_path": os.path.abspath(config_path),
        "config_sha256": _sha256_file(config_path),
        "rnalocate_species": (
            list(species) if species is not None else None
        ),
        "rnalocate_augment_er": augment_er,
        "apex_source_species": "Homo sapiens",
        "apex_source_cell_context": "HEK293T",
        "gene_mapping_mode": "case-insensitive same-symbol intersection",
        "ortholog_mapping_performed": False,
        "cross_species_limitation": (
            "Mouse matches are conserved same-symbol matches, not validated "
            "one-to-one ortholog conversions."
        ),
        "apex_omm_specificity_filter": apex_filter,
        "apex_omm_min_log2fc": apex_min_log2fc,
        "apex_omm_min_delta": apex_min_delta,
        "apex_omm_basal_pass_rows": int(
            apex_omm_audit["omm_basal_pass"].sum()
        ),
        "apex_omm_specific_pass_rows": int(
            apex_omm_audit["omm_specific_pass"].sum()
        ),
        "apex_omm_shared_rows": int(
            apex_omm_audit["omm_shared"].sum()
        ),
        "manual_markers_requested": use_manual_markers,
        "manual_markers_used": manual_used,
        "gold_reference_sha256": _sha256_frame(hash_frame),
        "gold_reference_rows": int(len(gold)),
    }
    return gold, provenance


def run_rapx_annotation_tracks(
    adata_full,
    cluster_labels: pd.Series,
    gold_df: pd.DataFrame,
    *,
    use_two_stage: bool = False,
    use_spatial_channel: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the shared RAP-X head and retain all evidence-aware tracks."""

    from .pipeline import run_rapx

    adata_mode = adata_full.copy()
    labels = pd.Series(
        cluster_labels,
        index=cluster_labels.index,
        dtype="object",
    ).reindex(adata_mode.obs_names)
    if labels.isna().any():
        missing = labels.index[labels.isna()][:10].tolist()
        raise ValueError(
            "Cluster labels are missing for RAP-X observations: "
            f"{missing}"
        )
    adata_mode.obs["leiden"] = labels.astype(str).to_numpy()

    run_rapx(
        adata_mode,
        gold_df=gold_df,
        produced_label_col="subcellular_annotation",
        plot_umap=False,
        plot_diag_enrich=False,
        use_two_stage=use_two_stage,
        use_spatial_channel=use_spatial_channel,
    )

    required = {
        "production": "subcellular_annotation",
        "candidate": "subcellular_candidate",
        "exploratory": "subcellular_annotation_exploratory",
        "strict": "subcellular_annotation_strict",
        "tier": "subcellular_annotation_tier",
    }
    missing_columns = [
        source
        for source in required.values()
        if source not in adata_mode.obs.columns
    ]
    if missing_columns:
        raise RuntimeError(
            "The active RAP-X configuration did not emit the evidence-aware "
            f"benchmark columns: {missing_columns}. Use "
            "annotation.decision_mode=independent."
        )

    tracks = pd.DataFrame(index=adata_mode.obs_names)
    for target, source in required.items():
        tracks[target] = adata_mode.obs[source].astype(str).to_numpy()
    optional = {
        "primary_empirical_q": "subcell_primary_empirical_q",
        "evidence_score_gap": "subcell_evidence_score_gap",
        "n_supported_labels": "subcell_n_supported_labels",
    }
    for target, source in optional.items():
        if source in adata_mode.obs.columns:
            tracks[target] = pd.to_numeric(
                adata_mode.obs[source],
                errors="coerce",
            ).to_numpy()

    annotation_meta = dict(
        adata_mode.uns.get("subcellular_annotation_meta", {})
    )
    metadata: dict[str, object] = {
        "annotation_group_unit": "method_cluster",
        "n_annotation_groups": int(labels.nunique()),
        "two_stage": bool(use_two_stage),
        "spatial_channel": bool(use_spatial_channel),
        "track_columns": {
            track: required[track]
            for track in (*RAPX_BENCHMARK_TRACKS, "tier")
        },
        "rapx_params": annotation_meta.get("params", {}),
    }
    return tracks, metadata


def validate_benchmark_population(
    final_data: pd.DataFrame,
    obs_names: Sequence[object],
) -> dict[str, object]:
    """Fail fast on stale MetaSpot populations and freeze membership identity."""

    required = {
        "cell_id",
        "metaspot",
        "overlaps_nucleus",
    }
    missing = required - set(final_data.columns)
    if missing:
        raise ValueError(
            f"Benchmark final_data is missing columns: {sorted(missing)}"
        )

    metaspot = final_data["metaspot"].astype(str)
    observed = pd.Index(pd.unique(metaspot))
    expected = pd.Index(obs_names, dtype="object").astype(str)
    missing_ids = expected.difference(observed)
    extra_ids = observed.difference(expected)
    if len(missing_ids) or len(extra_ids):
        raise ValueError(
            "Benchmark MetaSpot population mismatch: "
            f"missing={missing_ids[:10].tolist()}, "
            f"extra={extra_ids[:10].tolist()}"
        )

    owners = (
        final_data.assign(_metaspot=metaspot)
        .groupby("_metaspot", observed=True)["cell_id"]
        .nunique()
    )
    conflicting = owners[owners.ne(1)]
    if len(conflicting):
        raise ValueError(
            "MetaSpots must belong to exactly one cell: "
            f"{conflicting.index[:10].tolist()}"
        )

    membership = pd.DataFrame(
        {
            "cell_id": final_data["cell_id"].astype(str),
            "metaspot": metaspot,
        }
    )
    return {
        "n_transcripts": int(len(final_data)),
        "n_cells": int(final_data["cell_id"].nunique()),
        "n_metaspots": int(len(observed)),
        "metaspot_membership_sha256": _sha256_frame(membership),
        "membership_checksum_semantics": (
            "sha256(pandas_hash(cell_id,metaspot) in transcript row order)"
        ),
    }


def evaluate_rapx_annotation_tracks(
    final_data: pd.DataFrame,
    tracks: pd.DataFrame,
    method_name: str,
) -> tuple[
    dict[str, dict[str, float | int | str]],
    dict[str, dict[str, dict[str, float | int | str]]],
    pd.DataFrame,
    dict[str, object],
]:
    """Evaluate production and all evidence-audit tracks on one population."""

    population = validate_benchmark_population(final_data, tracks.index)
    metaspot = final_data["metaspot"].astype(str)
    work = pd.DataFrame(
        {
            "cell_id": final_data["cell_id"].astype(str),
            "metaspot": metaspot,
            "True_Class": np.where(
                pd.to_numeric(
                    final_data["overlaps_nucleus"],
                    errors="coerce",
                ).fillna(0).astype(int).eq(1),
                "Nuclear",
                "Cytoplasmic",
            ),
        },
        index=final_data.index,
    )
    mapped = pd.DataFrame(index=final_data.index)
    summaries: dict[str, dict[str, float | int | str]] = {}
    per_class_output: dict[
        str,
        dict[str, dict[str, float | int | str]],
    ] = {}

    for track in RAPX_BENCHMARK_TRACKS:
        if track not in tracks.columns:
            raise KeyError(f"Missing RAP-X benchmark track: {track}")
        mapped_track = metaspot.map(tracks[track].astype(str))
        prediction_column = f"_prediction_{track}"
        work[prediction_column] = pd.Categorical(mapped_track)
        summary, _, per_class = evaluate_binary_predictions_fast(
            work,
            f"{method_name}[{track}]",
            prediction_column,
        )
        summaries[track] = summary
        per_class_output[track] = (
            per_class.set_index("class")[
                ["precision", "recall", "f1", "support"]
            ]
            .to_dict(orient="index")
        )
        mapped[track] = pd.Categorical(mapped_track)
        work.drop(columns=[prediction_column], inplace=True)

    mapped["tier"] = pd.Categorical(
        metaspot.map(tracks["tier"].astype(str)),
        categories=["Strict", "Exploratory", "Candidate"],
        ordered=True,
    )
    return summaries, per_class_output, mapped, population


def extract_rapx_tracks_from_final_data(
    final_data: pd.DataFrame,
) -> pd.DataFrame:
    """Recover one evidence-aware RAP-X call per MetaSpot from final output."""

    frame = normalize_benchmark_metaspot_column(final_data)
    sources = {
        "production": "subcellular_annotation",
        "candidate": "subcellular_candidate",
        "exploratory": "subcellular_annotation_exploratory",
        "strict": "subcellular_annotation_strict",
        "tier": "subcellular_annotation_tier",
    }
    missing = [
        source for source in sources.values() if source not in frame.columns
    ]
    if missing:
        raise KeyError(
            "CellScope output predates evidence-aware benchmark tracks; "
            f"missing {missing}. Rerun CellScope before benchmarking."
        )

    metaspot = frame["metaspot"].astype(str)
    tracks = pd.DataFrame()
    for target, source in sources.items():
        values = frame[source].astype(str)
        unique_count = (
            pd.DataFrame({"metaspot": metaspot, "value": values})
            .groupby("metaspot", observed=True)["value"]
            .nunique(dropna=False)
        )
        inconsistent = unique_count[unique_count.ne(1)]
        if len(inconsistent):
            raise ValueError(
                f"{source} is not constant within MetaSpot: "
                f"{inconsistent.index[:10].tolist()}"
            )
        tracks[target] = (
            pd.DataFrame({"metaspot": metaspot, "value": values})
            .drop_duplicates("metaspot")
            .set_index("metaspot")["value"]
        )
    return tracks


def evaluate_geometry_clustering_diagnostic(
    final_data: pd.DataFrame,
    cluster_labels: pd.Series,
) -> dict[str, float | int | str]:
    """Transcript-weighted geometry clustering metrics and oracle upper bound."""

    validate_benchmark_population(final_data, cluster_labels.index)
    metaspot = final_data["metaspot"].astype(str)
    clusters = metaspot.map(cluster_labels.astype(str))
    truth = np.where(
        pd.to_numeric(
            final_data["overlaps_nucleus"],
            errors="coerce",
        ).fillna(0).astype(int).eq(1),
        "Nuclear",
        "Cytoplasmic",
    )
    contingency = pd.crosstab(
        clusters,
        pd.Categorical(
            truth,
            categories=["Cytoplasmic", "Nuclear"],
        ),
    ).reindex(columns=["Cytoplasmic", "Nuclear"], fill_value=0)
    counts = contingency.to_numpy(dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("No transcripts are available for clustering metrics.")

    def entropy(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        values = values[values > 0]
        if not len(values):
            return 0.0
        probabilities = values / values.sum()
        return float(-(probabilities * np.log(probabilities)).sum())

    class_counts = counts.sum(axis=0)
    cluster_counts = counts.sum(axis=1)
    class_entropy = entropy(class_counts)
    cluster_entropy = entropy(cluster_counts)
    conditional_class = sum(
        cluster_counts[i] / total * entropy(counts[i, :])
        for i in range(len(cluster_counts))
        if cluster_counts[i] > 0
    )
    conditional_cluster = sum(
        class_counts[j] / total * entropy(counts[:, j])
        for j in range(len(class_counts))
        if class_counts[j] > 0
    )
    homogeneity = (
        1.0 if class_entropy == 0 else 1.0 - conditional_class / class_entropy
    )
    completeness = (
        1.0
        if cluster_entropy == 0
        else 1.0 - conditional_cluster / cluster_entropy
    )
    v_measure = (
        0.0
        if homogeneity + completeness == 0
        else 2.0
        * homogeneity
        * completeness
        / (homogeneity + completeness)
    )

    predicts_nuclear = counts[:, 1] > counts[:, 0]
    tn = counts[~predicts_nuclear, 0].sum()
    fn = counts[~predicts_nuclear, 1].sum()
    fp = counts[predicts_nuclear, 0].sum()
    tp = counts[predicts_nuclear, 1].sum()
    oracle = _binary_metrics_from_counts(tn, fp, fn, tp)

    return {
        "truth_source": "nucleus_overlap_geometry",
        "n_clusters": int(len(contingency)),
        "transcript_weighted_purity": float(counts.max(axis=1).sum() / total),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "oracle_majority_accuracy": float(oracle["accuracy"]),
        "oracle_majority_nuclear_f1": float(oracle["nuclear_f1"]),
        "oracle_note": (
            "Uses geometry truth to label each cluster; diagnostic upper bound, "
            "not a predictive annotation result."
        ),
    }


def build_cell_isolated_coordinates(
    final_data: pd.DataFrame,
    obs_names: Iterable[str],
    gap_factor: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, float | int]]:
    """Tile cells apart while preserving every within-cell displacement.

    SpaGCN accepts one coordinate array and otherwise treats all observations as
    one tissue section.  SeqFISH MetaSpots from different cells must not become
    spatial neighbours, so each cell is translated to a separate tile.  Only a
    translation is applied: distances and angles within a cell are unchanged.
    """

    required = {
        "metaspot",
        "cell_id",
        "x_location",
        "y_location",
    }
    missing = required - set(final_data.columns)
    if missing:
        raise ValueError(
            "Cannot isolate spatial coordinates; missing columns: "
            f"{sorted(missing)}"
        )

    frame = final_data.loc[:, sorted(required)].copy()
    frame["metaspot"] = frame["metaspot"].astype(str)
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame["x_location"] = pd.to_numeric(
        frame["x_location"],
        errors="coerce",
    )
    frame["y_location"] = pd.to_numeric(
        frame["y_location"],
        errors="coerce",
    )

    if frame[["x_location", "y_location"]].isna().any().any():
        raise ValueError("SpaGCN coordinates contain non-numeric values.")

    cell_counts = frame.groupby("metaspot", observed=True)["cell_id"].nunique()
    conflicting = cell_counts[cell_counts.ne(1)]
    if not conflicting.empty:
        raise ValueError(
            "Each MetaSpot must belong to exactly one cell; conflicting IDs: "
            f"{conflicting.index[:10].tolist()}"
        )

    node_meta = frame.groupby("metaspot", observed=True).agg(
        cell_id=("cell_id", "first"),
        x=("x_location", "mean"),
        y=("y_location", "mean"),
    )

    ordered_index = pd.Index(obs_names, dtype="object").astype(str)
    missing_metaspots = ordered_index.difference(node_meta.index)
    if len(missing_metaspots):
        raise ValueError(
            "MetaSpots in AnnData are missing from final_data: "
            f"{missing_metaspots[:10].tolist()}"
        )

    node_meta = node_meta.loc[ordered_index]
    original = node_meta[["x", "y"]].astype(float)
    groups = node_meta["cell_id"].astype(str)

    centers = original.groupby(groups, observed=True).transform("mean")
    local = original - centers

    spans = original.groupby(groups, observed=True).agg(
        x_span=("x", lambda values: float(values.max() - values.min())),
        y_span=("y", lambda values: float(values.max() - values.min())),
    )
    maximum_span = max(
        1.0,
        float(spans[["x_span", "y_span"]].to_numpy().max()),
    )
    spacing = maximum_span * max(float(gap_factor), 2.0)

    cell_ids = sorted(groups.unique().tolist())
    grid_columns = max(1, int(math.ceil(math.sqrt(len(cell_ids)))))
    offsets = {
        cell_id: np.array(
            [
                (position % grid_columns) * spacing,
                (position // grid_columns) * spacing,
            ],
            dtype=float,
        )
        for position, cell_id in enumerate(cell_ids)
    }

    isolated_values = local.to_numpy(dtype=float).copy()
    for row_index, cell_id in enumerate(groups):
        isolated_values[row_index] += offsets[cell_id]

    isolated = pd.DataFrame(
        isolated_values,
        index=ordered_index,
        columns=["x", "y"],
    )

    stats: dict[str, float | int] = {
        "n_spatial_groups": int(len(cell_ids)),
        "spatial_tile_spacing": float(spacing),
        "spatial_gap_factor": float(max(float(gap_factor), 2.0)),
    }
    return original, isolated, groups, stats


def restore_bento_transcript_population(
    input_transcripts: pd.DataFrame,
    annotated_transcripts: pd.DataFrame,
    annotation_columns: Iterable[str],
    *,
    missing_reason: str,
    id_column: str = "bento_input_row_id",
) -> pd.DataFrame:
    """Left-join Bento scores onto the controlling input population."""

    if id_column not in input_transcripts.columns:
        raise KeyError(f"Input transcripts are missing {id_column!r}.")
    if input_transcripts[id_column].duplicated().any():
        raise ValueError(f"Input {id_column!r} values must be unique.")

    requested_columns = list(dict.fromkeys(annotation_columns))
    if annotated_transcripts.empty:
        annotations = pd.DataFrame({id_column: pd.Series(dtype=input_transcripts[id_column].dtype)})
    else:
        if id_column not in annotated_transcripts.columns:
            raise KeyError(f"Bento output is missing {id_column!r}.")
        if annotated_transcripts[id_column].duplicated().any():
            raise ValueError(f"Bento output {id_column!r} values must be unique.")
        available = [
            column
            for column in requested_columns
            if column in annotated_transcripts.columns
        ]
        annotations = annotated_transcripts[[id_column, *available]].copy()

    annotations["_bento_row_survived"] = True

    base = input_transcripts.drop(
        columns=[
            column
            for column in [*requested_columns, "bento_prediction_available", "bento_missing_reason"]
            if column in input_transcripts.columns
        ],
        errors="ignore",
    )
    output = base.merge(
        annotations,
        on=id_column,
        how="left",
        sort=False,
        validate="one_to_one",
    )

    for column in requested_columns:
        if column not in output.columns:
            output[column] = pd.NA

    survived = (
        output.pop("_bento_row_survived")
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    if "bento_annotation" not in output.columns:
        output["bento_annotation"] = pd.NA

    annotation_text = output["bento_annotation"].astype("string")
    prediction_available = annotation_text.notna() & annotation_text.ne("Unknown")
    output["bento_prediction_available"] = prediction_available.astype(bool)

    reason = pd.Series("", index=output.index, dtype="object")
    reason.loc[~survived] = missing_reason
    reason.loc[survived & ~prediction_available] = "no_fazal_score"
    if "_bento_valid_gene" in output.columns:
        invalid_gene = ~(
            output["_bento_valid_gene"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        reason.loc[invalid_gene] = "invalid_gene_symbol"

    output["bento_missing_reason"] = reason
    output["bento_annotation"] = annotation_text.fillna("Unknown").astype(str)

    if len(output) != len(input_transcripts):
        raise RuntimeError("Bento population restoration changed the input row count.")
    return output


def analytic_permutation_wsum_zscore(
    expression,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute the exact-moment z-score for feature-permuted weighted sums.

    Bento's Fazal2019 enrichment uses decoupler WSUM and estimates the null
    mean and standard deviation by repeatedly permuting feature weights.  For
    a uniformly random permutation, those two moments have a closed form.  The
    result therefore targets the same normalized weighted-sum statistic while
    avoiding a large ``samples x sources x permutations`` Monte Carlo tensor.

    Parameters
    ----------
    expression
        Samples by features matrix. Dense NumPy arrays and SciPy sparse
        matrices are supported.
    weights
        Features by gene-set/source weight matrix.
    """

    if sparse.issparse(expression):
        matrix = expression.tocsr().astype(np.float64, copy=False)
    else:
        matrix = np.asarray(expression, dtype=np.float64)

    weight_matrix = np.asarray(weights, dtype=np.float64)

    if matrix.ndim != 2 or weight_matrix.ndim != 2:
        raise ValueError("expression and weights must both be two-dimensional.")
    if matrix.shape[1] != weight_matrix.shape[0]:
        raise ValueError(
            "Feature dimensions do not match: "
            f"expression={matrix.shape[1]}, weights={weight_matrix.shape[0]}."
        )
    if matrix.shape[1] == 0:
        raise ValueError("At least one feature is required.")
    if not np.isfinite(weight_matrix).all():
        raise ValueError("weights contains non-finite values.")
    if sparse.issparse(matrix):
        if not np.isfinite(matrix.data).all():
            raise ValueError("expression contains non-finite values.")
    elif not np.isfinite(matrix).all():
        raise ValueError("expression contains non-finite values.")

    n_features = matrix.shape[1]
    estimate = np.asarray(matrix @ weight_matrix, dtype=np.float64)
    expression_sum = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    weight_sum = weight_matrix.sum(axis=0)
    null_mean = expression_sum[:, None] * weight_sum[None, :] / n_features

    if sparse.issparse(matrix):
        expression_square_sum = np.asarray(
            matrix.multiply(matrix).sum(axis=1),
            dtype=np.float64,
        ).reshape(-1)
    else:
        expression_square_sum = np.einsum("ij,ij->i", matrix, matrix)

    expression_centered_ss = np.maximum(
        expression_square_sum - expression_sum**2 / n_features,
        0.0,
    )
    weight_square_sum = np.einsum(
        "ij,ij->j",
        weight_matrix,
        weight_matrix,
    )
    weight_centered_ss = np.maximum(
        weight_square_sum - weight_sum**2 / n_features,
        0.0,
    )

    if n_features == 1:
        null_variance = np.zeros_like(estimate)
    else:
        null_variance = (
            expression_centered_ss[:, None]
            * weight_centered_ss[None, :]
            / (n_features - 1)
        )

    null_sd = np.sqrt(np.maximum(null_variance, 0.0))
    centered_estimate = estimate - null_mean
    scores = np.zeros_like(centered_estimate)
    np.divide(
        centered_estimate,
        null_sd,
        out=scores,
        where=null_sd > 0,
    )
    return scores
