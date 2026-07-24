import numpy as np
import pandas as pd
from scipy import sparse

from cellscope.benchmark_utils import (
    analytic_permutation_wsum_zscore,
    build_active_rapx_gold_table,
    build_cell_isolated_coordinates,
    evaluate_binary_predictions_fast,
    evaluate_geometry_clustering_diagnostic,
    evaluate_rapx_annotation_tracks,
    extract_rapx_tracks_from_final_data,
    normalize_benchmark_metaspot_column,
    restore_bento_transcript_population,
    validate_benchmark_population,
)


def test_active_rapx_reference_records_cross_species_limitations():
    _, provenance = build_active_rapx_gold_table(
        pd.Index(["ACTB", "GAPDH", "MALAT1"])
    )

    assert provenance["apex_source_species"] == "Homo sapiens"
    assert provenance["apex_source_cell_context"] == "HEK293T"
    assert provenance["gene_mapping_mode"] == (
        "case-insensitive same-symbol intersection"
    )
    assert provenance["ortholog_mapping_performed"] is False
    assert "not validated" in provenance["cross_species_limitation"]


def test_cell_isolation_preserves_within_cell_displacements():
    final_data = pd.DataFrame(
        {
            "metaspot": ["a0", "a1", "b0", "b1"],
            "cell_id": ["a", "a", "b", "b"],
            "x_location": [0.0, 3.0, 0.0, 3.0],
            "y_location": [0.0, 4.0, 0.0, 4.0],
        }
    )

    original, isolated, groups, stats = build_cell_isolated_coordinates(
        final_data,
        ["a0", "a1", "b0", "b1"],
    )

    np.testing.assert_allclose(
        original.loc["a1"] - original.loc["a0"],
        isolated.loc["a1"] - isolated.loc["a0"],
    )
    np.testing.assert_allclose(
        original.loc["b1"] - original.loc["b0"],
        isolated.loc["b1"] - isolated.loc["b0"],
    )
    assert groups.to_dict() == {"a0": "a", "a1": "a", "b0": "b", "b1": "b"}
    assert stats["n_spatial_groups"] == 2
    assert np.linalg.norm(isolated.loc["a0"] - isolated.loc["b0"]) > 5.0


def test_legacy_cell_sub_is_normalized_to_metaspot():
    frame = pd.DataFrame({"cell_sub": ["c:0", "c:1"]})

    normalized = normalize_benchmark_metaspot_column(frame)

    assert normalized["metaspot"].tolist() == ["c:0", "c:1"]
    assert "cell_sub" not in normalized.columns


def test_bento_population_restoration_keeps_missing_rows_as_unknown():
    inputs = pd.DataFrame(
        {
            "bento_input_row_id": [0, 1, 2],
            "feature_name": ["A", "B", ""],
            "_bento_valid_gene": [True, True, False],
        }
    )
    annotations = pd.DataFrame(
        {
            "bento_input_row_id": [0],
            "bento_annotation": ["Nucleus"],
            "bento_max_score": [2.0],
        }
    )

    output = restore_bento_transcript_population(
        inputs,
        annotations,
        ["bento_annotation", "bento_max_score"],
        missing_reason="dropped_by_bento_prep_or_raster",
    )

    assert len(output) == 3
    assert output["bento_annotation"].tolist() == ["Nucleus", "Unknown", "Unknown"]
    assert output["bento_prediction_available"].tolist() == [True, False, False]
    assert output["bento_missing_reason"].tolist() == [
        "",
        "dropped_by_bento_prep_or_raster",
        "invalid_gene_symbol",
    ]


def test_analytic_wsum_matches_exact_permutation_moments():
    expression = np.array(
        [
            [0.5, -1.0, 2.0, 0.25],
            [1.5, 0.0, -0.5, 3.0],
        ]
    )
    weights = np.array(
        [
            [1.0, -0.25],
            [-0.5, 0.75],
            [0.25, 1.5],
            [2.0, -1.0],
        ]
    )

    from itertools import permutations

    null_values = np.stack(
        [
            expression @ weights[np.asarray(order)]
            for order in permutations(range(expression.shape[1]))
        ],
        axis=-1,
    )
    observed = expression @ weights
    expected = (
        observed - null_values.mean(axis=-1)
    ) / null_values.std(axis=-1, ddof=0)

    actual = analytic_permutation_wsum_zscore(expression, weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_analytic_wsum_dense_and_sparse_inputs_match():
    expression = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 2.0, 1.0],
        ]
    )
    weights = np.array(
        [
            [0.5, 1.0],
            [-0.25, 0.0],
            [1.5, -1.0],
        ]
    )

    dense = analytic_permutation_wsum_zscore(expression, weights)
    sparse_result = analytic_permutation_wsum_zscore(
        sparse.csr_matrix(expression),
        weights,
    )
    np.testing.assert_allclose(dense, sparse_result)


def test_fast_binary_benchmark_matches_sklearn_reference():
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    frame = pd.DataFrame(
        {
            "cell_id": ["a"] * 6 + ["b"] * 5 + ["c"],
            "metaspot": [f"m{i}" for i in range(12)],
            "True_Class": [
                "Cytoplasmic", "Cytoplasmic", "Nuclear", "Nuclear", "Nuclear", "Unknown",
                "Nuclear", "Nuclear", "Nuclear", "Cytoplasmic", "Cytoplasmic", "Cytoplasmic",
            ],
            "prediction": [
                "cytosol", "Nucleus", "Nucleolus", "Unknown", "ER Lumen", "Nucleus",
                "Nuclear Pore", "OMM", "Nucleus", "mitochondrial", None, "not-a-class",
            ],
        }
    )

    summary, cell_metrics, per_class = evaluate_binary_predictions_fast(
        frame,
        "test",
        "prediction",
    )

    normalized = frame["prediction"].map(
        {
            "cytosol": "Cytoplasmic",
            "Nucleus": "Nuclear",
            "Nucleolus": "Nuclear",
            "ER Lumen": "Cytoplasmic",
            "Nuclear Pore": "Nuclear",
            "OMM": "Cytoplasmic",
            "mitochondrial": "Cytoplasmic",
        }
    ).fillna("Unknown")
    valid_truth = frame["True_Class"].isin(["Cytoplasmic", "Nuclear"])
    assigned = normalized.isin(["Cytoplasmic", "Nuclear"])
    evaluation = frame.loc[valid_truth & assigned].copy()
    y_true = evaluation["True_Class"]
    y_pred = normalized.loc[evaluation.index]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["Cytoplasmic", "Nuclear"],
        zero_division=0,
    )

    assert summary["n_transcripts_total"] == int(valid_truth.sum())
    assert summary["n_transcripts_eval"] == len(evaluation)
    assert summary["accuracy"] == accuracy_score(y_true, y_pred)
    assert summary["balanced_accuracy"] == balanced_accuracy_score(y_true, y_pred)
    assert summary["macro_f1"] == f1_score(
        y_true,
        y_pred,
        labels=["Cytoplasmic", "Nuclear"],
        average="macro",
        zero_division=0,
    )
    assert summary["weighted_f1"] == f1_score(
        y_true,
        y_pred,
        labels=["Cytoplasmic", "Nuclear"],
        average="weighted",
        zero_division=0,
    )
    assert summary["MCC"] == matthews_corrcoef(y_true, y_pred)
    assert summary["Cohen_kappa"] == cohen_kappa_score(y_true, y_pred)
    np.testing.assert_allclose(per_class["precision"], precision)
    np.testing.assert_allclose(per_class["recall"], recall)
    np.testing.assert_allclose(per_class["f1"], f1)
    np.testing.assert_array_equal(per_class["support"], support)

    for _, row in cell_metrics.iterrows():
        cell_mask = frame["cell_id"].eq(row["cell_id"]) & valid_truth & assigned
        cell_true = frame.loc[cell_mask, "True_Class"]
        cell_pred = normalized.loc[cell_mask]
        assert row["accuracy"] == accuracy_score(cell_true, cell_pred)
        assert row["balanced_accuracy"] == balanced_accuracy_score(
            cell_true,
            cell_pred,
        )
        assert row["macro_f1"] == f1_score(
            cell_true,
            cell_pred,
            labels=["Cytoplasmic", "Nuclear"],
            average="macro",
            zero_division=0,
        )
        assert row["MCC"] == matthews_corrcoef(cell_true, cell_pred)


def test_rapx_tracks_report_coverage_and_overall_accuracy_separately():
    final_data = pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c2", "c2"],
            "metaspot": ["m1", "m1", "m2", "m2"],
            "overlaps_nucleus": [0, 0, 1, 0],
        }
    )
    tracks = pd.DataFrame(
        {
            "production": ["Cytosol", "Nucleus"],
            "candidate": ["Cytosol", "Nucleus"],
            "exploratory": ["Cytosol", "Nucleus"],
            "strict": ["Cytosol", "Unknown"],
            "tier": ["Strict", "Candidate"],
        },
        index=["m1", "m2"],
    )

    summaries, per_class, mapped, population = (
        evaluate_rapx_annotation_tracks(
            final_data,
            tracks,
            "method",
        )
    )

    assert summaries["candidate"]["coverage"] == 1.0
    assert summaries["candidate"]["accuracy"] == 0.75
    assert summaries["candidate"]["overall_accuracy"] == 0.75
    assert summaries["strict"]["coverage"] == 0.5
    assert summaries["strict"]["accuracy"] == 1.0
    assert summaries["strict"]["overall_accuracy"] == 0.5
    assert per_class["strict"]["Cytoplasmic"]["support"] == 2
    assert mapped["tier"].astype(str).tolist() == [
        "Strict",
        "Strict",
        "Candidate",
        "Candidate",
    ]
    assert population["n_metaspots"] == 2
    assert len(population["metaspot_membership_sha256"]) == 64


def test_extract_rapx_tracks_requires_constant_metaspot_calls():
    final_data = pd.DataFrame(
        {
            "metaspot": ["m1", "m1", "m2"],
            "subcellular_annotation": ["Cytosol", "Cytosol", "Nucleus"],
            "subcellular_candidate": ["ER", "ER", "Nucleus"],
            "subcellular_annotation_exploratory": [
                "ER",
                "ER",
                "Unknown",
            ],
            "subcellular_annotation_strict": [
                "Unknown",
                "Unknown",
                "Unknown",
            ],
            "subcellular_annotation_tier": [
                "Exploratory",
                "Exploratory",
                "Candidate",
            ],
        }
    )

    tracks = extract_rapx_tracks_from_final_data(final_data)

    assert tracks.loc["m1", "candidate"] == "ER"
    assert tracks.loc["m1", "production"] == "Cytosol"
    assert tracks.loc["m2", "tier"] == "Candidate"

    final_data.loc[1, "subcellular_candidate"] = "OMM"
    with np.testing.assert_raises(ValueError):
        extract_rapx_tracks_from_final_data(final_data)


def test_population_validation_rejects_stale_metaspot_ids():
    final_data = pd.DataFrame(
        {
            "cell_id": ["c1"],
            "metaspot": ["current"],
            "overlaps_nucleus": [0],
        }
    )

    with np.testing.assert_raises(ValueError):
        validate_benchmark_population(final_data, ["stale"])


def test_geometry_clustering_metrics_are_explicit_oracle_diagnostics():
    final_data = pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c2", "c2"],
            "metaspot": ["m1", "m1", "m2", "m2"],
            "overlaps_nucleus": [0, 0, 1, 0],
        }
    )
    clusters = pd.Series(["a", "b"], index=["m1", "m2"])

    result = evaluate_geometry_clustering_diagnostic(
        final_data,
        clusters,
    )

    assert result["truth_source"] == "nucleus_overlap_geometry"
    assert result["transcript_weighted_purity"] == 0.75
    assert result["oracle_majority_accuracy"] == 0.75
    assert "upper bound" in result["oracle_note"]
