#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

from cellscope.pipeline import (
    _configure_rnalocate_augmentation,
    _load_default_gold_tables,
    _load_manual_marker_data,
    _prepare_metaspot_marker_sets,
    _refine_metaspot_annotations_with_local_dgi,
    build_gold_from_rnalocate_l1,
    harmonize_gold_gene_case,
    merge_gold,
    project_l1_to_seven,
)


FINE_STRUCTURES = ("Nucleolus", "Lamina", "Nuclear Pore", "ER", "OMM")
ALL_LABELS = (
    "Nucleus",
    "Nucleolus",
    "Lamina",
    "Nuclear Pore",
    "Cytosol",
    "ER",
    "OMM",
)
COMPARTMENTS = {
    "Nuclear": ("Nucleus", "Nucleolus", "Lamina", "Nuclear Pore"),
    "Cytoplasmic": ("Cytosol", "ER", "OMM"),
}
STRUCTURE_COMPARTMENT = {
    label: compartment
    for compartment, labels in COMPARTMENTS.items()
    for label in labels
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit five fine structures in an existing CellScoper run."
    )
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(data: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _cell_max(
    values: np.ndarray,
    mask: np.ndarray,
    cell_codes: np.ndarray,
    cells: pd.Index,
) -> np.ndarray:
    selected = np.flatnonzero(mask & np.isfinite(values))
    if selected.size == 0:
        return np.full(len(cells), np.nan)
    grouped = pd.Series(values[selected], index=cell_codes[selected]).groupby(
        level=0, sort=False
    ).max()
    return grouped.reindex(np.arange(len(cells))).to_numpy(dtype=float)


def _cell_any(mask: np.ndarray, cell_codes: np.ndarray, n_cells: int) -> np.ndarray:
    return np.bincount(
        cell_codes[np.asarray(mask, dtype=bool)], minlength=n_cells
    ).astype(bool)


def _cell_count(mask: np.ndarray, cell_codes: np.ndarray, n_cells: int) -> np.ndarray:
    return np.bincount(
        cell_codes[np.asarray(mask, dtype=bool)], minlength=n_cells
    ).astype(int)


def _build_cell_structure_atlas(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atlas = detail.copy()
    atlas["status"] = np.select(
        [
            atlas["strict_detected"].to_numpy(bool),
            atlas["final_detected"].to_numpy(bool),
            atlas["detectable"].to_numpy(bool),
        ],
        [
            "Detected_strict",
            "Detected_supported",
            "Detectable_not_localized",
        ],
        default="Insufficient_evidence",
    )
    atlas["confidence"] = np.where(
        atlas["final_detected"].to_numpy(bool),
        atlas["soft_support_score"].to_numpy(float),
        np.nan,
    )
    atlas["loci"] = atlas["n_final_loci"].to_numpy(int)
    atlas["burden_total_cell"] = atlas["final_transcript_fraction"].to_numpy(float)
    atlas["burden_compartment"] = atlas["compartment_fraction"].to_numpy(float)

    grouped = atlas.groupby("cell_id", sort=False)
    cell_summary = grouped.agg(
        cell_total_transcripts=("cell_total_transcripts", "first"),
        n_fine_marker_evidence=("marker_evidence", "sum"),
        n_fine_detectable=("detectable", "sum"),
        n_fine_detected=("final_detected", "sum"),
    )
    detected_when_detectable = (
        atlas["detectable"].to_numpy(bool)
        & atlas["final_detected"].to_numpy(bool)
    )
    atlas["detected_when_detectable"] = detected_when_detectable
    cell_summary["n_detectable_detected"] = grouped[
        "detected_when_detectable"
    ].sum()
    cell_summary["detectable_coverage"] = np.divide(
        cell_summary["n_detectable_detected"],
        cell_summary["n_fine_detectable"],
        out=np.full(len(cell_summary), np.nan),
        where=cell_summary["n_fine_detectable"].to_numpy() > 0,
    )
    cell_summary.reset_index(inplace=True)
    atlas = atlas.merge(cell_summary, on="cell_id", how="left", validate="many_to_one")

    cell_summary["transcript_depth"] = pd.cut(
        cell_summary["cell_total_transcripts"],
        bins=[-np.inf, 249, 499, 999, np.inf],
        labels=["<250", "250-499", "500-999", ">=1000"],
        ordered=True,
    )
    depth_summary = (
        cell_summary.groupby("transcript_depth", observed=False)
        .agg(
            cells=("cell_id", "size"),
            median_transcripts=("cell_total_transcripts", "median"),
            mean_fine_marker_evidence=("n_fine_marker_evidence", "mean"),
            mean_fine_detectable=("n_fine_detectable", "mean"),
            mean_fine_detected=("n_fine_detected", "mean"),
            mean_detectable_coverage=("detectable_coverage", "mean"),
            median_detectable_coverage=("detectable_coverage", "median"),
        )
        .reset_index()
    )
    return atlas, cell_summary, depth_summary


def _build_gold(annotation: dict, var_names: pd.Index) -> pd.DataFrame:
    gold_apex, rna_locate = _load_default_gold_tables(
        apex_omm_specificity_filter=bool(
            annotation.get("apex_omm_specificity_filter", True)
        ),
        apex_omm_min_log2fc=float(annotation.get("apex_omm_min_log2fc", 0.75)),
        apex_omm_min_delta=float(annotation.get("apex_omm_min_delta", 0.50)),
    )
    raw_species = annotation.get(
        "rnalocate_species", ("Homo sapiens", "Mus musculus")
    )
    if raw_species is None:
        species = None
    elif isinstance(raw_species, str):
        species = tuple(value.strip() for value in raw_species.split(",") if value.strip())
    else:
        species = tuple(str(value) for value in raw_species)
    rna_gold = project_l1_to_seven(
        build_gold_from_rnalocate_l1(
            rna_locate,
            species=species,
            agg_mode="prob_or",
            score_floor=0.6,
            use_pubmed_weight=True,
        ),
        mode="strict",
        mito_non_omm=None,
    )
    gold = merge_gold(
        gold_apex,
        _configure_rnalocate_augmentation(
            rna_gold,
            augment_er=bool(annotation.get("rnalocate_augment_er", False)),
        ),
        fuse="max",
    )
    if bool(annotation.get("use_manual_markers", False)):
        manual = _load_manual_marker_data()
        if manual is not None:
            gold = merge_gold(gold, manual, fuse="max")
    return harmonize_gold_gene_case(gold, target_var_names=var_names)


def _cell_marker_evidence(
    adata: ad.AnnData,
    marker_sets: dict[str, list[str]],
    broad: np.ndarray,
    cell_codes: np.ndarray,
    n_cells: int,
    prior_strength: float,
) -> dict[str, dict[str, np.ndarray]]:
    var_position = {
        str(gene).upper(): position
        for position, gene in enumerate(adata.var_names.astype(str))
    }
    matrix = adata.layers.get("counts", adata.X)
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    matrix = matrix.tocsr()
    result: dict[str, dict[str, np.ndarray]] = {}
    for compartment, compartment_labels in COMPARTMENTS.items():
        rows = np.flatnonzero(broad == compartment)
        membership = sp.csr_matrix(
            (
                np.ones(rows.size, dtype=float),
                (cell_codes[rows], rows),
            ),
            shape=(n_cells, adata.n_obs),
        )
        label_matrices: dict[str, sp.csr_matrix] = {}
        total_by_cell = np.zeros(n_cells, dtype=float)
        total_global = 0.0
        global_by_label: dict[str, float] = {}
        for label in compartment_labels:
            positions = [var_position[gene] for gene in marker_sets[label]]
            label_matrix = (membership @ matrix[:, positions]).tocsr()
            label_matrices[label] = label_matrix
            counts = np.asarray(label_matrix.sum(axis=1)).ravel()
            total_by_cell += counts
            global_by_label[label] = float(counts.sum())
            total_global += global_by_label[label]
        alpha = 0.5
        denominator = total_global + alpha * len(compartment_labels)
        for label in compartment_labels:
            label_matrix = label_matrices[label]
            counts = np.asarray(label_matrix.sum(axis=1)).ravel()
            genes = np.asarray(label_matrix.getnnz(axis=1), dtype=int)
            background = (global_by_label[label] + alpha) / denominator
            fraction = (counts + prior_strength * background) / (
                total_by_cell + prior_strength
            )
            enrichment = np.log2(np.maximum(fraction, 1e-12) / background)
            p_value = np.full(n_cells, np.nan)
            tested = total_by_cell > 0
            p_value[tested] = binom.sf(
                np.rint(counts[tested]).astype(int) - 1,
                np.rint(total_by_cell[tested]).astype(int),
                background,
            )
            q_value = np.full(n_cells, np.nan)
            finite = np.isfinite(p_value)
            if finite.any():
                q_value[finite] = multipletests(p_value[finite], method="fdr_bh")[1]
            result[label] = {
                "transcripts": counts,
                "genes": genes,
                "enrichment": enrichment,
                "p_value": p_value,
                "q_value": q_value,
                "background_probability": np.full(n_cells, background),
            }
    return result


def main() -> None:
    output_dir = _parse_args().output_dir.resolve()
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    config = yaml.safe_load((output_dir / "effective_params.yaml").read_text())
    annotation = config["annotation"]
    adata = ad.read_h5ad(output_dir / "adata1.h5ad")
    original_final = adata.obs["subcellular_annotation"].astype(str).to_numpy()
    gold = _build_gold(annotation, adata.var_names)
    marker_sets = _prepare_metaspot_marker_sets(gold, adata.var_names, annotation)
    refined, score_table, metadata = _refine_metaspot_annotations_with_local_dgi(
        adata,
        gold,
        annotation,
        build_score_table=True,
    )
    if score_table is None:
        raise RuntimeError("Full MetaSpot score table was not produced")
    recomputed_final = refined.obs["subcellular_annotation"].astype(str).to_numpy()
    mismatch_count = int(np.count_nonzero(original_final != recomputed_final))
    if mismatch_count:
        raise RuntimeError(
            f"Audit recomputation differs from persisted labels at {mismatch_count} MetaSpots"
        )

    cells = pd.Index(pd.unique(refined.obs["cell_id"].astype(str)), name="cell_id")
    cell_lookup = {cell: index for index, cell in enumerate(cells)}
    cell_codes = refined.obs["cell_id"].astype(str).map(cell_lookup).to_numpy(dtype=int)
    n_cells = len(cells)
    broad = refined.obs["subcellular_broad_compartment"].astype(str).to_numpy()
    count_matrix = refined.layers.get("counts", refined.X)
    spot_transcripts = np.asarray(count_matrix.sum(axis=1)).ravel().astype(float)
    cell_transcripts = np.bincount(
        cell_codes, weights=spot_transcripts, minlength=n_cells
    )
    if int(cell_transcripts.sum()) != int(manifest["outputs"]["counts"]["n_transcripts"]):
        raise RuntimeError("adata1 transcript totals do not match run_manifest.json")

    marker_evidence = _cell_marker_evidence(
        refined,
        marker_sets,
        broad,
        cell_codes,
        n_cells,
        float(annotation.get("metaspot_background_prior_strength", 1.0)),
    )
    thresholds = metadata["label_thresholds"]
    local_enrichment = {
        label: score_table[f"log2_enrichment__{label.lower().replace(' ', '_')}"]
        .to_numpy(dtype=float)
        for label in ALL_LABELS
    }
    dgi_support = {
        label: score_table[f"dgi_support__{label.lower().replace(' ', '_')}"]
        .to_numpy(dtype=float)
        for label in ALL_LABELS
    }
    structure_presence = np.asarray(
        refined.obsm["X_subcellular_structure_presence"], dtype=float
    )
    structure_localization = np.asarray(
        refined.obsm["X_subcellular_structure_localization"], dtype=float
    )
    structure_evidence = np.asarray(
        refined.obsm["X_subcellular_structure_evidence"], dtype=float
    )
    structure_spatial_coherence = np.asarray(
        refined.obsm["X_subcellular_structure_spatial_coherence"], dtype=float
    )
    spatial_thresholds = annotation.get(
        "metaspot_fine_min_spatial_coherence", 0.5
    )
    structure_index = {
        label: index for index, label in enumerate(FINE_STRUCTURES)
    }
    dgi_informative = score_table["dgi_informative_neighbors"].to_numpy(dtype=int)
    records = []
    final_labels = refined.obs["subcellular_annotation"].astype(str).to_numpy()
    strict_labels = refined.obs["subcellular_annotation_strict"].astype(str).to_numpy()

    for structure in FINE_STRUCTURES:
        evidence_idx = structure_index[structure]
        compartment = STRUCTURE_COMPARTMENT[structure]
        compatible_labels = COMPARTMENTS[compartment]
        other_labels = [label for label in compatible_labels if label != structure]
        suffix = structure.lower().replace(" ", "_")
        counts = score_table[f"marker_transcripts__{suffix}"].to_numpy(dtype=float)
        genes = score_table[f"marker_genes__{suffix}"].to_numpy(dtype=int)
        enrichment = local_enrichment[structure]
        q_value = score_table[f"bh_q__{suffix}"].to_numpy(dtype=float)
        alternative_enrichment = np.max(
            np.column_stack([local_enrichment[label] for label in other_labels]), axis=1
        )
        specificity_margin = np.full(refined.n_obs, np.nan)
        finite_specificity = np.isfinite(enrichment) & np.isfinite(
            alternative_enrichment
        )
        specificity_margin[finite_specificity] = (
            enrichment[finite_specificity]
            - alternative_enrichment[finite_specificity]
        )
        support = dgi_support[structure]
        alternative_support = np.max(
            np.column_stack([dgi_support[label] for label in other_labels]), axis=1
        )
        dgi_margin = support - alternative_support
        domain_support = pd.to_numeric(
            refined.obs[f"subcell_evidence_score__{structure}"], errors="coerce"
        ).to_numpy(dtype=float)
        domain_affinity = pd.to_numeric(
            refined.obs[f"subcell_relative_evidence__{structure}"], errors="coerce"
        ).to_numpy(dtype=float)
        rapx_calibrated_score = pd.to_numeric(
            refined.obs[f"subcell_subtype_calibrated_score__{structure}"],
            errors="coerce",
        ).to_numpy(dtype=float)
        domain_geometry = refined.obs[
            f"subcell_geometry_feasible__{structure}"
        ].astype(bool).to_numpy()
        geometry = (broad == compartment) & domain_geometry
        local_threshold = thresholds[structure]
        spatial_threshold = float(
            spatial_thresholds.get(structure, 0.5)
            if isinstance(spatial_thresholds, dict)
            else spatial_thresholds
        )
        spatial_coherence = structure_spatial_coherence[:, evidence_idx]
        local_candidate = (
            geometry
            & (counts >= int(local_threshold["min_marker_transcripts"]))
            & (genes >= int(local_threshold["min_marker_genes"]))
            & np.isfinite(enrichment)
            & (enrichment >= float(local_threshold["min_log2_enrichment"]))
            & np.isfinite(q_value)
            & np.isfinite(spatial_coherence)
            & (spatial_coherence >= spatial_threshold)
        )
        dgi_min_support = float(
            annotation.get("metaspot_cell_coverage_min_dgi_support", 0.5)
        )
        dgi_min_margin = float(
            annotation.get("metaspot_cell_coverage_min_dgi_margin", 0.0)
        )
        localizable_spot = (
            local_candidate
            & (dgi_informative >= int(annotation.get("metaspot_dgi_min_informative_neighbors", 2)))
            & np.isfinite(support)
            & np.isfinite(dgi_margin)
            & (support >= dgi_min_support)
            & (dgi_margin >= dgi_min_margin)
        )
        if structure == "OMM":
            localizable_spot &= specificity_margin >= float(
                annotation.get("metaspot_omm_local_min_score_gap", 0.15)
            )

        cell_marker = marker_evidence[structure]
        if structure == "OMM":
            detectable = (
                (cell_marker["transcripts"] >= int(annotation["metaspot_omm_cell_min_marker_transcripts"]))
                & (cell_marker["genes"] >= int(annotation["metaspot_omm_cell_min_marker_genes"]))
                & (cell_marker["enrichment"] >= float(annotation["metaspot_omm_cell_min_log2_enrichment"]))
                & (cell_marker["p_value"] <= float(annotation["metaspot_omm_cell_max_binomial_p"]))
            )
        else:
            detectable = (
                (cell_marker["transcripts"] >= int(annotation["metaspot_cell_coverage_min_marker_transcripts"]))
                & (cell_marker["genes"] >= int(annotation["metaspot_cell_coverage_min_marker_genes"]))
                & (cell_marker["enrichment"] >= float(annotation["metaspot_cell_coverage_min_log2_enrichment"]))
                & (cell_marker["p_value"] <= float(annotation["metaspot_cell_coverage_max_binomial_p"]))
            )

        final_spot = final_labels == structure
        strict_spot = strict_labels == structure
        compartment_transcripts = np.bincount(
            cell_codes[broad == compartment],
            weights=spot_transcripts[broad == compartment],
            minlength=n_cells,
        )
        final_transcripts = np.bincount(
            cell_codes[final_spot],
            weights=spot_transcripts[final_spot],
            minlength=n_cells,
        )
        final_detected = final_transcripts > 0
        strict_detected = _cell_any(strict_spot, cell_codes, n_cells)
        localizable = _cell_any(localizable_spot, cell_codes, n_cells)
        audit_status = np.select(
            [
                strict_detected,
                final_detected,
                detectable & localizable,
                detectable,
                localizable,
            ],
            [
                "Detected_strict",
                "Detected_supported",
                "Cell_and_local_evidence_not_selected",
                "Detectable_not_localized",
                "Local_evidence_without_cell_gate",
            ],
            default="Insufficient_evidence",
        )
        records.append(
            pd.DataFrame(
                {
                    "cell_id": cells,
                    "structure": structure,
                    "marker_transcripts": cell_marker["transcripts"],
                    "marker_genes": cell_marker["genes"],
                    "cell_log2_enrichment": cell_marker["enrichment"],
                    "cell_binomial_p": cell_marker["p_value"],
                    "cell_bh_q": cell_marker["q_value"],
                    "marker_evidence": (cell_marker["transcripts"] > 0) & (cell_marker["genes"] > 0),
                    "detectable": detectable,
                    "n_candidate_metaspots": _cell_count(local_candidate, cell_codes, n_cells),
                    "n_localizable_metaspots": _cell_count(localizable_spot, cell_codes, n_cells),
                    "localizable": localizable,
                    "best_dgi_support": _cell_max(support, local_candidate, cell_codes, cells),
                    "best_dgi_margin": _cell_max(dgi_margin, local_candidate, cell_codes, cells),
                    "best_metadomain_support": _cell_max(domain_support, geometry, cell_codes, cells),
                    "best_metadomain_affinity": _cell_max(domain_affinity, geometry, cell_codes, cells),
                    "best_rapx_calibrated_score": _cell_max(rapx_calibrated_score, geometry, cell_codes, cells),
                    "best_local_log2_enrichment": _cell_max(enrichment, geometry, cell_codes, cells),
                    "best_specificity_margin": _cell_max(specificity_margin, geometry, cell_codes, cells),
                    "geometry_feasible": _cell_any(geometry, cell_codes, n_cells),
                    "n_geometry_feasible_metaspots": _cell_count(geometry, cell_codes, n_cells),
                    "cell_total_transcripts": cell_transcripts,
                    "spatial_coherence": _cell_max(
                        spatial_coherence,
                        np.ones(refined.n_obs, dtype=bool),
                        cell_codes,
                        cells,
                    ),
                    "min_spatial_coherence": np.full(
                        n_cells, spatial_threshold
                    ),
                    "presence_score": _cell_max(
                        structure_presence[:, evidence_idx],
                        np.ones(refined.n_obs, dtype=bool),
                        cell_codes,
                        cells,
                    ),
                    "localization_score": _cell_max(
                        structure_localization[:, evidence_idx],
                        np.ones(refined.n_obs, dtype=bool),
                        cell_codes,
                        cells,
                    ),
                    "soft_support_score": _cell_max(
                        structure_evidence[:, evidence_idx],
                        np.ones(refined.n_obs, dtype=bool),
                        cell_codes,
                        cells,
                    ),
                    "final_detected": final_detected,
                    "strict_detected": strict_detected,
                    "n_final_loci": _cell_count(final_spot, cell_codes, n_cells),
                    "final_transcript_fraction": np.divide(
                        final_transcripts,
                        cell_transcripts,
                        out=np.zeros(n_cells),
                        where=cell_transcripts > 0,
                    ),
                    "compartment_fraction": np.divide(
                        final_transcripts,
                        compartment_transcripts,
                        out=np.zeros(n_cells),
                        where=compartment_transcripts > 0,
                    ),
                    "audit_status": audit_status,
                }
            )
        )

    detail = pd.concat(records, ignore_index=True)
    if len(detail) != n_cells * len(FINE_STRUCTURES):
        raise RuntimeError("Cell × structure audit matrix is incomplete")
    atlas, cell_summary, depth_summary = _build_cell_structure_atlas(detail)
    expected_statuses = {
        "Detected_strict",
        "Detected_supported",
        "Detectable_not_localized",
        "Insufficient_evidence",
    }
    if atlas[["cell_id", "structure"]].duplicated().any():
        raise RuntimeError("Cell × structure atlas contains duplicate rows")
    if not set(atlas["status"]).issubset(expected_statuses):
        raise RuntimeError("Cell × structure atlas contains an unknown status")
    if not atlas["soft_support_score"].between(0.0, 1.0).all():
        raise RuntimeError("Cell × structure support scores must be in [0, 1]")
    summary_rows = []
    for structure, frame in detail.groupby("structure", sort=False):
        detected = frame["final_detected"]
        summary_rows.append(
            {
                "structure": structure,
                "cells": len(frame),
                "marker_evidence_cells": int(frame["marker_evidence"].sum()),
                "detectable_cells": int(frame["detectable"].sum()),
                "localizable_cells": int(frame["localizable"].sum()),
                "detectable_and_localizable_cells": int((frame["detectable"] & frame["localizable"]).sum()),
                "final_detected_cells": int(detected.sum()),
                "final_without_detectable_cells": int((detected & ~frame["detectable"]).sum()),
                "detectable_not_final_cells": int((frame["detectable"] & ~detected).sum()),
                "localizable_not_final_cells": int((frame["localizable"] & ~detected).sum()),
                "strict_detected_cells": int(frame["strict_detected"].sum()),
                "median_final_loci_detected": float(frame.loc[detected, "n_final_loci"].median()) if detected.any() else 0.0,
                "median_final_transcript_fraction_detected": float(frame.loc[detected, "final_transcript_fraction"].median()) if detected.any() else 0.0,
                "median_compartment_fraction_detected": float(frame.loc[detected, "compartment_fraction"].median()) if detected.any() else 0.0,
                "p95_compartment_fraction_detected": float(frame.loc[detected, "compartment_fraction"].quantile(0.95)) if detected.any() else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    intermediate = output_dir / "intermediate_results"
    intermediate.mkdir(exist_ok=True)
    detail_path = intermediate / "phase1_cell_structure_evidence.csv"
    summary_path = intermediate / "phase1_structure_summary.csv"
    metadata_path = intermediate / "phase1_structure_audit.json"
    atlas_path = intermediate / "cell_structure_atlas.csv"
    cell_summary_path = intermediate / "cell_structure_summary.csv"
    depth_summary_path = intermediate / "cell_structure_depth_summary.csv"
    _atomic_csv(detail, detail_path)
    _atomic_csv(summary, summary_path)
    _atomic_csv(atlas, atlas_path)
    _atomic_csv(cell_summary, cell_summary_path)
    _atomic_csv(depth_summary, depth_summary_path)
    _atomic_json(
        {
            "schema_version": 3,
            "run_finished_at": manifest["finished_at"],
            "run_status": manifest["status"],
            "python_source_sha256": manifest["code"]["python_source_sha256"],
            "effective_params_sha256": manifest["configuration"]["effective_sha256"],
            "n_cells": n_cells,
            "n_metaspots": refined.n_obs,
            "n_cell_structure_rows": len(detail),
            "recomputed_final_label_mismatches": mismatch_count,
            "fine_structures": list(FINE_STRUCTURES),
            "marker_genes_per_structure": {
                label: len(marker_sets[label]) for label in FINE_STRUCTURES
            },
            "definitions": {
                "marker_evidence": "At least one exclusive measured marker transcript and gene in the matching broad compartment.",
                "detectable": "Passes the current structure-specific cell-level marker gates from effective_params.yaml; this is assay detectability, not biological organelle presence.",
                "candidate_metaspot": "Geometry-feasible MetaSpot passing the structure's focal marker count, gene, enrichment, and finite-q gates before DGI.",
                "localizable": "At least one candidate MetaSpot passes leave-one-out same-cell DGI support and margin; OMM additionally passes its local specificity gap.",
                "spatial_coherence": "Total-variation distance between structure-marker and all-transcript MetaSpot distributions within the cell parent compartment.",
                "best_metadomain_support": "Maximum mapped RAP-X MetaDomain evidence score among geometry-feasible MetaSpots.",
                "final_detected": "At least one persisted final MetaSpot label equals the structure.",
                "compartment_fraction": "Fraction of geometry-corrected nuclear or cytoplasmic MetaSpot transcripts assigned to the structure.",
                "soft_support_score": "Bounded composite of cell presence, focal marker, MetaDomain affinity, same-cell DGI, geometry, competitor separation, and RAP-X reference evidence; it is not a calibrated probability.",
                "detectable_coverage": "Detected structures among structures that pass the current cell-level detectability gate; cells with no detectable structures are NA.",
            },
            "status_definitions": {
                "Detected_strict": "Persisted final locus with strict statistical support.",
                "Detected_supported": "Persisted final locus with supported multi-source evidence.",
                "Detectable_not_localized": "Cell-level detectability gate passes but no final locus is emitted.",
                "Insufficient_evidence": "The current assay does not pass the cell-level detectability gate; this does not imply biological absence.",
            },
            "soft_support_score_components": {
                "presence": "marker amount, marker diversity, pooled enrichment, and binomial evidence",
                "localization": "focal marker, MetaDomain affinity, same-cell DGI, geometry, competitor separation, and RAP-X calibrated evidence",
                "combination": "geometric mean of bounded presence and localization scores",
            },
            "outputs": {
                "detail": str(detail_path),
                "summary": str(summary_path),
                "atlas": str(atlas_path),
                "cell_summary": str(cell_summary_path),
                "depth_summary": str(depth_summary_path),
            },
        },
        metadata_path,
    )
    print(summary.to_string(index=False))
    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {atlas_path}")
    print(f"wrote {cell_summary_path}")
    print(f"wrote {depth_summary_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
