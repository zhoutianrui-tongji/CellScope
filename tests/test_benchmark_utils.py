import numpy as np
import pandas as pd
import anndata as ad
import pytest
import yaml
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
    run_shared_metadomain_rapx_track,
    validate_benchmark_population,
    write_benchmark_h5ad,
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


@pytest.mark.parametrize("invalid_truth", [np.nan, "not-binary", 2])
def test_benchmark_population_rejects_invalid_geometry_truth(invalid_truth):
    final_data = pd.DataFrame(
        {
            "cell_id": ["c1", "c1"],
            "metaspot": ["m1", "m2"],
            "overlaps_nucleus": [1, invalid_truth],
        }
    )

    with pytest.raises(ValueError, match="only binary 0/1"):
        validate_benchmark_population(final_data, ["m1", "m2"])


def test_extract_tracks_prefers_explicit_domain_evidence_columns():
    final_data = pd.DataFrame(
        {
            "cell_id": ["c1", "c1"],
            "metaspot": ["m1", "m2"],
            "overlaps_nucleus": [1, 0],
            "subcellular_annotation": ["Nucleus", "Cytosol"],
            "subcellular_candidate_domain": ["Nucleus", "ER"],
            "subcellular_candidate": ["OMM", "OMM"],
            "subcellular_annotation_domain_exploratory": [
                "Nucleus",
                "ER",
            ],
            "subcellular_annotation_exploratory": ["OMM", "OMM"],
            "subcellular_annotation_domain_strict": [
                "Nucleus",
                "Unknown",
            ],
            "subcellular_annotation_strict": ["OMM", "OMM"],
            "subcellular_annotation_domain_tier": [
                "Strict",
                "Candidate",
            ],
            "subcellular_annotation_tier": ["Candidate", "Candidate"],
        }
    )

    tracks = extract_rapx_tracks_from_final_data(final_data)

    assert tracks["candidate"].tolist() == ["Nucleus", "ER"]
    assert tracks["exploratory"].tolist() == ["Nucleus", "ER"]
    assert tracks["strict"].tolist() == ["Nucleus", "Unknown"]
    assert tracks["tier"].tolist() == ["Strict", "Candidate"]


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


def test_benchmark_h5ad_writer_keeps_only_explicit_method_embedding(tmp_path):
    source = ad.AnnData(
        X=sparse.csr_matrix(np.eye(2, dtype=np.float32)),
        obs=pd.DataFrame(index=["m1", "m2"]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    source.obsm["X_method_metaspot"] = np.ones((2, 3), dtype=np.float32)
    source.obsm["X_dgi_gcn_metaspot"] = np.zeros((2, 4), dtype=np.float32)
    output_path = tmp_path / "adata_metaspot.h5ad"

    write_benchmark_h5ad(
        source,
        {
            "method_cluster": pd.Series(
                ["0", "1"],
                index=["m1", "m2"],
            ),
        },
        str(output_path),
        obsm_keys=["X_method_metaspot"],
        metadata={"observation_unit": "metaspot"},
    )

    output = ad.read_h5ad(output_path)
    assert list(output.obsm.keys()) == ["X_method_metaspot"]
    assert "X_dgi_gcn_metaspot" not in output.obsm
    assert output.uns["benchmark_output"]["observation_unit"] == "metaspot"
    assert output.uns["benchmark_output"][
        "excluded_inherited_representations"
    ] is True


def test_shared_metadomain_adapter_uses_named_method_representation(
    tmp_path,
    monkeypatch,
):
    obs_names = [f"m{i}" for i in range(8)]
    cells = ["c1"] * 4 + ["c2"] * 4
    obs = pd.DataFrame(
        {
            "cell_id": cells,
            "r_mean": np.linspace(-0.8, 0.8, 8),
            "r_std": np.full(8, 0.1),
            "r_p50": np.linspace(-0.8, 0.8, 8),
            "d_cell_norm_mean": np.full(8, 0.4),
            "d_nuc_norm_mean": np.full(8, 0.5),
            "elongation": np.full(8, 1.0),
            "spread": np.full(8, 0.3),
            "log_n": np.full(8, 1.2),
        },
        index=obs_names,
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            np.tile([[3, 1], [1, 3]], (4, 1)),
            dtype=np.float32,
        ),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    adata.obsm["X_method_metaspot"] = np.column_stack(
        [
            np.linspace(-2, 2, 8),
            np.zeros(8),
        ]
    ).astype(np.float32)
    adata.obsm["X_dgi_gcn_metaspot"] = np.full(
        (8, 3),
        99.0,
        dtype=np.float32,
    )
    final_data = pd.DataFrame(
        {
            "cell_id": cells,
            "metaspot": obs_names,
            "overlaps_nucleus": [0, 0, 0, 0, 1, 1, 1, 1],
            "r_signed": [-1, -1, -1, -1, 1, 1, 1, 1],
        }
    )

    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "module5": {
                    "estimate_k": "fixed",
                    "fixed_k": 2,
                    "min_domain_size": 1,
                    "enable_cap": False,
                    "split_by_nucleus": False,
                },
                "module6": {"pool_normalize": True},
                "module7": {"graph_unit": "metaspot"},
            }
        )
    )
    monkeypatch.setenv("CELLSCOPE_CONFIG", str(config_path))

    captured = {}

    def fake_annotate(
        adata1,
        adata2,
        final_df,
        **kwargs,
    ):
        if kwargs.get("annotation_unit_override") == "metaspot":
            captured["pool"] = np.asarray(
                adata1.obsm["X_dgi_gcn_metaspot"]
            ).copy()
            captured["boundary_columns"] = {
                column
                for column in ("r_mean", "r_std", "r_p50", "d_nuc_norm_mean")
                if column in adata1.obs.columns
            }
        adata2.obs["leiden"] = pd.Categorical(
            np.arange(adata2.n_obs).astype(str)
        )
        adata2.obs["leiden_embedding"] = adata2.obs["leiden"].copy()
        adata2.obs["annotation_group"] = pd.Categorical(
            adata2.obs_names.astype(str)
        )
        adata2.uns["annotation_unit"] = "metadomain"
        labels = pd.Series(
            np.where(
                np.arange(adata2.n_obs) % 2 == 0,
                "Cytosol",
                "Nucleus",
            ),
            index=adata2.obs_names.astype(str),
        )
        domain = adata1.obs["meta-domain"].astype(str)
        sources = {
            "subcellular_annotation": labels,
            "subcellular_candidate": labels,
            "subcellular_annotation_exploratory": labels,
            "subcellular_annotation_strict": labels,
        }
        for column, values in sources.items():
            adata2.obs[column] = values.reindex(
                adata2.obs_names.astype(str)
            ).to_numpy()
            adata1.obs[column] = domain.map(values).to_numpy()
        tiers = pd.Series(
            "Strict",
            index=adata2.obs_names.astype(str),
        )
        adata2.obs["subcellular_annotation_tier"] = tiers.to_numpy()
        adata1.obs["subcellular_annotation_tier"] = (
            domain.map(tiers).to_numpy()
        )
        # Simulate the downstream local geometry guard changing the MetaSpot
        # production labels while leaving raw MetaDomain evidence untouched.
        adata1.obs["subcellular_annotation"] = "Nucleus"
        return adata1, adata2, final_df

    monkeypatch.setattr(
        "cellscope.pipeline._maybe_cluster_and_annotate",
        fake_annotate,
    )
    result = run_shared_metadomain_rapx_track(
        adata,
        final_data,
        embedding_key="X_method_metaspot",
        metadomain_embedding_key="X_method_metadomain",
        method_name="Method",
        module5_config_override={"random_seed": 7},
    )

    assert "X_method_metadomain" in result["adata_metadomain"].obsm
    assert "X_dgi_gcn_metadomain" not in result["adata_metadomain"].obsm
    assert result["metadata"]["representation"][
        "uses_cellscope_dgi_embedding"
    ] is False
    assert result["metadata"]["n_metaspots"] == 8
    assert result["metadata"]["n_metadomains"] == 2
    assert list(result["tracks"].index) == obs_names
    assert result["tracks"].equals(result["raw_metadomain_tracks"])
    assert not result["tracks"]["production"].equals(
        result["geometry_guarded_tracks"]["production"]
    )
    assert result["metadata"]["primary_track_semantics"] == (
        "raw_metadomain_rapx_without_metaspot_geometry_correction"
    )
    assert result["metadata"][
        "n_production_labels_changed_by_geometry_guard"
    ] > 0
    assert result["adata_metaspot"].uns[
        "module5_metadomain_clustering"
    ]["random_seed"] == 7

    refined = run_shared_metadomain_rapx_track(
        adata,
        final_data,
        embedding_key="X_method_metaspot",
        metadomain_embedding_key="X_method_metadomain",
        method_name="Method",
        module5_config_override={"random_seed": 7},
        enable_metaspot_refinement=True,
    )

    np.testing.assert_array_equal(
        captured["pool"],
        adata.obsm["X_method_metaspot"],
    )
    assert captured["boundary_columns"] == set()
    assert refined["metadata"]["primary_track_semantics"] == (
        "shared_metaspot_refinement_without_boundary_geometry"
    )
    assert refined["shared_metaspot_refinement_tracks"] is not None
    assert refined[
        "shared_metaspot_refinement_geometry_guarded_tracks"
    ] is not None
    assert refined["metadata"][
        "n_production_labels_changed_by_geometry_guard"
    ] == 4

    gate = pd.Series(
        np.linspace(0.1, 0.9, 8),
        index=obs_names,
    )
    gated = run_shared_metadomain_rapx_track(
        adata,
        final_data,
        embedding_key="X_method_metaspot",
        metadomain_embedding_key="X_method_metadomain",
        method_name="Method",
        module5_config_override={"random_seed": 7},
        enable_metaspot_refinement=True,
        crossfitted_compartment_score=gate,
        crossfitted_score_source="toy_leave_one_group_out",
    )

    assert captured["boundary_columns"] == {"r_mean", "r_p50"}
    np.testing.assert_allclose(
        gated["adata_metaspot"].obs["r_p50"],
        0.5 - gate,
    )
    assert gated["metadata"]["primary_track_semantics"] == (
        "crossfitted_compartment_gated_shared_metaspot_refinement"
    )
    assert gated["metadata"]["representation"][
        "crossfitted_compartment_gate"
    ]["injected_after_module5_clustering"] is True


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
