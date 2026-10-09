import os
import tempfile
import unittest
import warnings
from unittest import mock
from contextlib import contextmanager

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import cellscope.pipeline as pipeline_module

from cellscope.pipeline import (
    _LocalCompositionModel,
    _coerce_bool_series,
    _ensure_obs_columns_initialized,
    _apply_low_confidence_policy,
    _apply_low_confidence_policy_by_candidate,
    _apply_independent_evidence_policy,
    _apply_independent_evidence_tracks,
    _choose_annotation_neighbors_key,
    _build_same_cell_metaspot_dgi_pool,
    _stable_group_layout,
    _high_confidence_cell_coverage,
    _unique_cell_counts_for_masks,
    _compute_metaspot_dgi_consensus,
    _compute_cell_structure_spatial_coherence,
    _configure_annotation_unit,
    _configure_rnalocate_augmentation,
    _prepare_apex_gold_reference,
    _refine_metaspot_annotations_with_local_dgi,
    _resolve_metaspot_label_thresholds,
    _resolve_metaspot_omm_confirmation,
    _apply_metaspot_diminishing_return_selection,
    _select_metaspot_cell_coverage_rescues,
    _select_metaspot_labelwise_cell_coverage_rescues,
    _select_metaspot_multistructure_first_loci,
    _select_metaspot_hierarchical_cell_rescues,
    _select_metaspot_omm_cell_coherent_rescues,
    _load_default_gold_tables,
    _compose_omm_metaspot_labels,
    _apply_local_compartment_guard,
    _metaspot_compartment_guard_values,
    _map_metaspot_labels_to_transcripts,
    _key_row_indexer,
    _take_obs_column_by_row_indexer,
    _sanitize_obs_for_h5ad,
    _demote_weak_top1,
    _guard_nuclear_pore_overcall,
    _guard_omm_overcall,
    _guard_lamina_overcall,
    _rapx_annotate_core,
    _choose_geometry_consistent_label,
    _compartment_geometry_scores,
    _empirical_marker_set_calibration,
    _nuclear_geometry_thresholds,
    _is_count_like_matrix,
    _subtype_geometry_scores,
    build_specific_marker_sets,
    robust_gene_stats,
    _refresh_subcellular_group_columns,
    _resume_build_outputs,
    _allocate_partition_target_counts,
    _enforce_spatially_connected_labels,
    _metaspot_segmentation_audit,
    _migrate_legacy_metaspot_columns,
    _select_top_variance_features,
    _save_adata,
    _stretch_signed_geometry,
    _target_metaspot_count,
    _validate_adata1_point_df_alignment,
    build_point_geometry_v2,
    build_composition_embeddings,
    build_adata1,
    build_adata2_from_adata1_and_pointdf,
    per_cell_clustering,
)
from cellscope.config import module_config_subset, stage_config_subset


@contextmanager
def _local_config():
    content = """
module2:
  top_g: 4
  bow_knn_k: 5
  max_rows_per_cell_for_svd: 40
  max_svd_rows: 160
  svd_n_components: 3
  svd_sample_cells: 4
  use_tfidf: true
  tf_norm: true
  stream_local_embeddings: true
runtime:
  n_jobs: 1
  joblib_backend: threading
"""
    old = os.environ.get("CELLSCOPE_CONFIG")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(content)
        path = handle.name
    os.environ["CELLSCOPE_CONFIG"] = path
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CELLSCOPE_CONFIG", None)
        else:
            os.environ["CELLSCOPE_CONFIG"] = old
        os.unlink(path)


def _synthetic_transcripts():
    rows = []
    for cell in range(4):
        for i in range(40):
            left = i < 20
            genes = ("LEFT", "RIGHT") if cell % 2 == 0 else ("A", "B")
            rows.append(
                {
                    "cell_id": str(cell),
                    "x_location": float(i),
                    "y_location": float(cell),
                    "feature_name": genes[0] if left else genes[1],
                    "r_signed": 0.0,
                    "d_cell_norm": 0.5,
                    "d_nuc_norm": 0.5,
                }
            )
    return pd.DataFrame(rows)


class LocalCompositionTests(unittest.TestCase):
    def test_count_like_matrix_rejects_normalized_proportions(self):
        self.assertTrue(
            _is_count_like_matrix(sp.csr_matrix([[0.0, 2.0], [3.0, 1.0]]))
        )
        self.assertFalse(
            _is_count_like_matrix(sp.csr_matrix([[0.25, 0.75]]))
        )

    def test_module6_preserves_raw_counts_separately_from_tf_mean(self):
        obs = pd.DataFrame(
            {
                "meta-domain": pd.Categorical(["A", "A", "B"]),
                "cell_id": ["c1", "c2", "c1"],
                "r_mean": [-0.5, -0.4, 0.6],
                "r_std": [0.1, 0.1, 0.2],
                "r_p50": [-0.5, -0.4, 0.6],
                "d_cell_norm_mean": [0.4, 0.4, 0.3],
                "d_nuc_norm_mean": [0.2, 0.2, 0.7],
                "elongation": [1.0, 1.1, 1.2],
                "spread": [0.3, 0.4, 0.5],
                "log_n": [1.0, 1.2, 1.1],
            },
            index=["m1", "m2", "m3"],
        )
        adata1 = ad.AnnData(
            X=sp.csr_matrix([[2, 0], [0, 3], [1, 1]], dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=["G1", "G2"]),
        )

        result = build_adata2_from_adata1_and_pointdf(
            adata1,
            pd.DataFrame(),
            module6_config_override={"pool_reps": []},
            graph_unit_override="metadomain",
        )

        np.testing.assert_allclose(
            result.layers["counts"].toarray(),
            np.array([[2, 3], [1, 1]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(result.X.sum(axis=1)).ravel(),
            np.ones(2),
        )
        self.assertTrue(_is_count_like_matrix(result.layers["counts"]))
        self.assertIn(
            "layers_counts",
            result.uns["matrix_semantics"],
        )

    def test_module6_accepts_lightweight_geometry_metadata(self):
        obs = pd.DataFrame(
            {
                "meta-domain": pd.Categorical(["A", "A", "B"]),
                "cell_id": ["c1", "c1", "c2"],
                "r_mean": [-0.5, -0.4, 0.6],
                "r_p50": [-0.5, -0.4, 0.6],
                "d_cell_norm_mean": [0.4, 0.4, 0.3],
                "d_nuc_norm_mean": [0.2, 0.2, 0.7],
            },
            index=["m1", "m2", "m3"],
        )
        adata1 = ad.AnnData(
            X=sp.csr_matrix([[2, 0], [0, 3], [1, 1]], dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=["G1", "G2"]),
        )

        result = build_adata2_from_adata1_and_pointdf(
            adata1,
            pd.DataFrame(),
            module6_config_override={"pool_reps": []},
            graph_unit_override="metadomain",
        )

        assert list(result.obs.columns) == [
            "n_metaspots",
            "r_mean",
            "r_p50",
            "d_cell_norm_mean",
            "d_nuc_norm_mean",
            "rep_cell_id",
        ]
        np.testing.assert_allclose(
            result.obs["r_mean"].to_numpy(),
            np.array([-0.45, 0.6]),
        )

    def test_explicit_annotation_rep_is_not_overridden_by_priority(self):
        adata = ad.AnnData(X=np.ones((3, 2), dtype=np.float32))
        graph = sp.eye(3, format="csr", dtype=np.float32)
        adata.obsp["pca_nbrs_connectivities"] = graph
        adata.obsp["gcn_nbrs_connectivities"] = graph

        selected = _choose_annotation_neighbors_key(
            adata,
            cluster_rep="X_pca",
            rep_priority=["X_dgi_gcn", "X_pca"],
        )

        self.assertEqual(selected, "pca_nbrs")

    def test_metadomain_annotation_unit_preserves_embedding_leiden(self):
        adata = ad.AnnData(
            X=np.ones((3, 2), dtype=np.float32),
            obs=pd.DataFrame(
                {"leiden": pd.Categorical(["0", "0", "1"])},
                index=["d1", "d2", "d3"],
            ),
        )

        result = _configure_annotation_unit(adata, "metadomain")

        self.assertEqual(result.obs["leiden"].astype(str).tolist(), ["d1", "d2", "d3"])
        self.assertEqual(result.obs["leiden_embedding"].tolist(), ["0", "0", "1"])
        self.assertEqual(result.uns["annotation_unit"], "metadomain")

    def test_metaspot_annotation_unit_uses_metadomain_prior_groups(self):
        adata = ad.AnnData(
            X=np.ones((3, 2), dtype=np.float32),
            obs=pd.DataFrame(
                {"leiden": pd.Categorical(["0", "0", "1"])},
                index=["d1", "d2", "d3"],
            ),
        )

        result = _configure_annotation_unit(adata, "metaspot")

        self.assertEqual(
            result.obs["leiden"].astype(str).tolist(),
            ["d1", "d2", "d3"],
        )
        self.assertEqual(result.obs["leiden_embedding"].tolist(), ["0", "0", "1"])
        self.assertEqual(result.uns["annotation_unit"], "metaspot")
        self.assertEqual(result.uns["annotation_prior_unit"], "metadomain")

    def test_metaspot_dgi_pool_never_crosses_cell_or_compartment(self):
        obs = pd.DataFrame(
            {
                "cell_id": ["c1", "c1", "c1", "c1", "c2", "c2"],
            },
            index=[f"s{i}" for i in range(6)],
        )
        adata = ad.AnnData(
            X=sp.csr_matrix(np.ones((6, 1), dtype=np.int32)),
            obs=obs,
            var=pd.DataFrame(index=["G"]),
        )
        adata.obsm["X_dgi_gcn_metaspot"] = np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [1.0, 0.0], [0.99, 0.01], [1.0, 0.0], [0.99, 0.01]],
            dtype=np.float32,
        )
        broad = pd.Series(
            ["Nuclear", "Nuclear", "Cytoplasmic", "Cytoplasmic", "Nuclear", "Nuclear"],
            index=obs.index,
        )

        pooling, pool_size = _build_same_cell_metaspot_dgi_pool(
            adata,
            broad,
            n_neighbors=2,
        )

        rows, cols = pooling.nonzero()
        for row, col in zip(rows, cols):
            self.assertEqual(obs.iloc[row]["cell_id"], obs.iloc[col]["cell_id"])
            self.assertEqual(broad.iloc[row], broad.iloc[col])
        self.assertTrue(np.all(pool_size == 2))
        self.assertTrue(np.all(pooling.diagonal() == 1))

    def test_stable_group_layout_indexes_each_group_once(self):
        group_codes = np.asarray([2, 0, 2, 1, 0, 2, 1], dtype=np.int64)

        member_order, offsets, sizes = _stable_group_layout(group_codes, 3)

        np.testing.assert_array_equal(sizes, [2, 2, 3])
        np.testing.assert_array_equal(offsets, [0, 2, 4, 7])
        np.testing.assert_array_equal(
            member_order[offsets[0] : offsets[1]],
            [1, 4],
        )
        np.testing.assert_array_equal(
            member_order[offsets[1] : offsets[2]],
            [3, 6],
        )
        np.testing.assert_array_equal(
            member_order[offsets[2] : offsets[3]],
            [0, 2, 5],
        )

    def test_metaspot_dgi_consensus_is_leave_one_out(self):
        labels = ["Nucleus", "Nucleolus", "Lamina"]
        allowed = {"Nuclear": labels}
        broad = np.asarray(["Nuclear"] * 4, dtype=object)
        raw_scores = np.asarray(
            [
                [0.0, 3.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        raw_counts = np.zeros_like(raw_scores)
        raw_genes = np.zeros_like(raw_scores, dtype=np.int32)
        for row, label_idx in enumerate([1, 1, 2, 0]):
            raw_counts[row, label_idx] = 2
            raw_genes[row, label_idx] = 1
        pooling = sp.csr_matrix(
            np.asarray(
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1, 0, 1, 1],
                ],
                dtype=np.float32,
            )
        )

        candidate, _, support, informative = (
            _compute_metaspot_dgi_consensus(
                pooling,
                raw_scores,
                raw_counts,
                raw_genes,
                broad,
                labels,
                allowed,
                seed_min_transcripts=1,
                seed_min_genes=1,
                seed_min_enrichment=0.0,
                min_informative_neighbors=1,
            )
        )

        self.assertEqual(candidate.tolist(), [1, 1, 2, 0])
        self.assertEqual(informative.tolist(), [1, 1, 0, 2])
        self.assertEqual(float(support[2].sum()), 0.0)
        self.assertAlmostEqual(float(support[3, 1]), 0.5)
        self.assertAlmostEqual(float(support[3, 2]), 0.5)

    def test_metaspot_label_thresholds_are_validated_per_label(self):
        labels = ["Nucleus", "Nucleolus", "Cytosol", "ER"]
        defaults = {
            "min_marker_transcripts": 3,
            "min_marker_genes": 2,
            "min_log2_enrichment": 0.5,
            "min_score_gap": 0.25,
            "subtype_fdr": 0.05,
            "exploratory_max_q": 0.30,
            "min_dgi_support": 0.0,
            "min_dgi_margin": 0.0,
        }

        arrays, resolved = _resolve_metaspot_label_thresholds(
            {
                "metaspot_label_thresholds": {
                    "nucleolus": {
                        "min_marker_transcripts": 5,
                        "min_dgi_support": 0.60,
                    }
                }
            },
            labels,
            defaults,
        )

        self.assertEqual(int(arrays["min_marker_transcripts"][1]), 5)
        self.assertEqual(int(arrays["min_marker_transcripts"][3]), 3)
        self.assertAlmostEqual(
            resolved["Nucleolus"]["min_dgi_support"],
            0.60,
        )
        with self.assertRaisesRegex(ValueError, "unknown label"):
            _resolve_metaspot_label_thresholds(
                {
                    "metaspot_label_thresholds": {
                        "Mitochondria": {"min_marker_genes": 2}
                    }
                },
                labels,
                defaults,
            )
        with self.assertRaisesRegex(ValueError, "Unknown MetaSpot"):
            _resolve_metaspot_label_thresholds(
                {
                    "metaspot_label_thresholds": {
                        "ER": {"unsupported_field": 1}
                    }
                },
                labels,
                defaults,
            )

    def test_cell_coverage_rescue_is_bounded_and_evidence_gated(self):
        cell_ids = np.asarray(
            ["c1", "c1", "c1", "c1", "c2", "c2", "c2"],
            dtype=object,
        )
        candidate = np.asarray(
            [
                "Nucleolus",
                "Nucleolus",
                "ER",
                "OMM",
                "Nucleolus",
                "Nucleolus",
                "ER",
            ],
            dtype=object,
        )
        already_supported = np.asarray(
            [False, False, False, False, True, False, False]
        )
        marker_base = np.asarray(
            [True, True, True, True, True, True, False]
        )
        dgi_available = np.ones(7, dtype=bool)
        support = np.asarray([0.60, 0.80, 0.70, 1.0, 0.90, 0.95, 1.0])
        margin = np.asarray([0.10, 0.20, 0.10, 1.0, 0.30, 0.40, 1.0])
        enrichment = np.asarray([2.0, 1.5, 1.0, 3.0, 2.0, 3.0, 4.0])
        score_gap = np.asarray([1.0, 1.0, 0.5, 2.0, 1.0, 2.0, 3.0])
        q_values = np.asarray([1.0, 1.0, 1.0, 0.01, 1.0, 1.0, 1.0])
        counts = np.asarray([2, 2, 3, 5, 2, 4, 5])
        cell_counts = np.asarray([1, 8, 10, 20, 8, 8, 1])
        cell_genes = np.asarray([1, 3, 3, 5, 3, 3, 1])
        cell_enrichment = np.asarray([-0.2, 0.4, 0.5, 1.0, 0.3, 0.3, 0.5])
        cell_p = np.asarray([0.9, 0.1, 0.05, 0.01, 0.1, 0.1, 0.01])

        rescued = _select_metaspot_cell_coverage_rescues(
            cell_ids,
            candidate,
            already_supported,
            marker_base,
            dgi_available,
            support,
            margin,
            enrichment,
            score_gap,
            q_values,
            counts,
            cell_counts,
            cell_genes,
            cell_enrichment,
            cell_p,
            labels=["Nucleolus", "ER"],
            min_cell_transcripts=3,
            min_cell_genes=2,
            min_cell_enrichment=0.0,
            max_cell_p=0.50,
            min_dgi_support=0.50,
            min_dgi_margin=0.0,
            max_per_cell_label=1,
        )

        # c1's first Nucleolus lacks cell-level multi-gene evidence, while the
        # second Nucleolus and ER pass cell, local, and DGI gates.
        self.assertEqual(np.flatnonzero(rescued).tolist(), [1, 2])
        # c2 already has Nucleolus, its ER fails marker evidence, and OMM was
        # deliberately not included in the coverage-rescue label list.
        self.assertFalse(rescued[3])
        self.assertFalse(rescued[5])
        self.assertFalse(rescued[6])

    def test_labelwise_cell_coverage_rescue_recovers_nonwinning_label(self):
        labels = ["Nucleus", "Nucleolus", "Lamina"]
        allowed = {"Nuclear": labels}
        cell_ids = np.asarray(
            ["recover", "recover", "existing", "existing", "weak"],
            dtype=object,
        )
        broad = np.asarray(["Nuclear"] * 5, dtype=object)
        candidate = np.asarray(
            ["Lamina", "Nucleus", "Nucleolus", "Nucleus", "Lamina"],
            dtype=object,
        )
        supported = np.asarray([False, False, True, False, False])
        counts = np.asarray(
            [[2, 1, 2], [3, 1, 0], [1, 2, 0], [2, 0, 1], [2, 1, 2]],
            dtype=float,
        )
        genes = (counts > 0).astype(int)
        enrichment = np.asarray(
            [
                [0.2, 0.4, 0.8],
                [0.3, 0.6, -0.2],
                [0.0, 1.0, -0.1],
                [0.2, -0.1, 0.5],
                [0.2, 0.6, 0.8],
            ]
        )
        q_values = np.full((5, 3), 0.9)
        dgi_support = np.asarray(
            [
                [0.0, 0.6, 0.4],
                [0.0, 0.7, 0.3],
                [0.0, 0.8, 0.2],
                [0.0, 0.3, 0.7],
                [0.0, 0.7, 0.3],
            ]
        )
        cell_counts = np.asarray(
            [[5, 4, 2], [5, 4, 2], [3, 3, 3], [3, 3, 3], [5, 2, 4]],
            dtype=float,
        )
        cell_genes = np.asarray(
            [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]]
        )

        rescued = _select_metaspot_labelwise_cell_coverage_rescues(
            cell_ids,
            broad,
            candidate,
            supported,
            counts,
            genes,
            enrichment,
            q_values,
            np.ones(5, dtype=bool),
            dgi_support,
            cell_counts,
            cell_genes,
            np.full((5, 3), 0.5),
            np.full((5, 3), 0.1),
            labels=["Nucleolus", "Lamina"],
            all_labels=labels,
            allowed=allowed,
            min_local_transcripts=np.ones(3, dtype=int),
            min_local_genes=np.ones(3, dtype=int),
            min_local_enrichment=np.zeros(3),
            min_cell_transcripts=3,
            min_cell_genes=2,
            min_cell_enrichment=0.0,
            max_cell_p=0.5,
            min_dgi_support=0.5,
            min_dgi_margin=0.0,
            max_per_cell_label=1,
        )

        # Nucleolus is not the primary candidate in the first cell, yet its
        # own evidence recovers row 1. The existing cell already has
        # Nucleolus but can independently recover its missing Lamina at row 3;
        # the weak cell fails the multi-gene presence gate.
        self.assertEqual(
            rescued.tolist(), ["", "Nucleolus", "", "Lamina", ""]
        )

    def test_diminishing_return_keeps_first_and_protects_strict_loci(self):
        labels = [
            "Nucleus",
            "Nucleolus",
            "Lamina",
            "Nuclear Pore",
            "Cytosol",
            "ER",
            "OMM",
        ]
        allowed = {
            "Nuclear": labels[:4],
            "Cytoplasmic": labels[4:],
        }
        enrichment = np.zeros((4, len(labels)), dtype=float)
        support = np.zeros_like(enrichment)
        enrichment[:3, labels.index("OMM")] = [0.9, 0.7, 0.6]
        support[:3, labels.index("OMM")] = 0.6
        enrichment[3, labels.index("Nucleolus")] = -0.5

        keep, diagnostics = _apply_metaspot_diminishing_return_selection(
            ["a", "a", "a", "b"],
            ["OMM", "OMM", "OMM", "Nucleolus"],
            [False, False, False, False],
            ["Cytoplasmic", "Cytoplasmic", "Cytoplasmic", "Nuclear"],
            enrichment,
            support,
            [10, 10, 80, 100],
            enabled_labels=["Nucleolus", "OMM"],
            all_labels=labels,
            allowed=allowed,
            min_local_enrichment=[0, 0, 0, 0, 0, 0, 0.5],
            dgi_weight=0.6,
            min_dgi_support=0.5,
            log_penalty=0.2,
            burden_penalty=1.0,
        )

        self.assertEqual(keep.tolist(), [True, True, False, True])
        self.assertEqual(
            diagnostics["first_locus"].tolist(),
            [True, False, False, True],
        )
        self.assertAlmostEqual(diagnostics["burden_before"][2], 0.2)

        strict_keep, _ = _apply_metaspot_diminishing_return_selection(
            ["a", "a", "a"],
            ["OMM", "OMM", "OMM"],
            [False, False, True],
            ["Cytoplasmic"] * 3,
            enrichment[:3],
            support[:3],
            [10, 10, 80],
            enabled_labels=["OMM"],
            all_labels=labels,
            allowed=allowed,
            min_local_enrichment=[0, 0, 0, 0, 0, 0, 0.5],
            dgi_weight=0.6,
            min_dgi_support=0.5,
            log_penalty=0.2,
            burden_penalty=1.0,
        )
        self.assertTrue(strict_keep[2])

    def test_multistructure_first_locus_is_missing_only_and_burden_guarded(self):
        structures = ["Nucleolus", "ER"]
        shape = (7, 2)
        presence = np.zeros(shape, dtype=float)
        localization = np.zeros(shape, dtype=float)
        evidence = np.zeros(shape, dtype=float)
        presence[[0, 1, 3, 5, 6], :] = 0.8
        localization[[0, 1, 3, 5, 6], :] = 0.8
        evidence[0, 0] = 0.9
        evidence[1, 1] = 0.8
        evidence[3, 0] = 0.9
        evidence[5, 0] = 0.9
        evidence[6, 0] = 0.95
        marker_counts = np.ones(shape, dtype=int)
        spatial_coherence = np.ones(shape, dtype=float)

        selected, diagnostics = _select_metaspot_multistructure_first_loci(
            ["a", "a", "a", "b", "b", "c", "c"],
            ["Nucleus", "Cytosol", "Nucleus", "Nucleus", "Nucleus", "Nucleolus", "Nucleus"],
            [False, False, False, False, False, True, False],
            presence,
            localization,
            evidence,
            spatial_coherence,
            np.ones(shape, dtype=bool),
            marker_counts,
            marker_counts,
            marker_counts,
            marker_counts,
            np.ones(shape, dtype=float),
            np.zeros(shape, dtype=float),
            [True] * 7,
            [10, 10, 80, 30, 70, 10, 90],
            structures=structures,
            min_presence_score=0.5,
            min_localization_score=0.5,
            min_spatial_coherence=[0.5, 0.5],
            min_evidence_score=[0.7, 0.7],
            min_local_marker_transcripts=[1, 1],
            min_local_marker_genes=[1, 1],
            min_cell_marker_transcripts=[1, 1],
            min_cell_marker_genes=[1, 1],
            min_cell_marker_enrichment=[0.0, 0.0],
            max_cell_marker_p=[1.0, 1.0],
            max_projected_burden=0.2,
        )

        self.assertEqual(
            selected.tolist(),
            ["Nucleolus", "ER", "", "", "", "", ""],
        )
        self.assertTrue(diagnostics["burden_rejected"][3])
        self.assertFalse(diagnostics["eligible"][6])
        self.assertAlmostEqual(diagnostics["projected_burden"][0], 0.1)

    def test_cell_structure_spatial_coherence_is_compartment_normalized(self):
        coherence = _compute_cell_structure_spatial_coherence(
            np.asarray([0, 0, 0, 0, 0]),
            ["Nuclear", "Nuclear", "Nuclear", "Nuclear", "Cytoplasmic"],
            np.asarray([[1, 4], [1, 0], [1, 0], [1, 0], [9, 9]]),
            [10, 10, 10, 10, 90],
            ["diffuse", "focused"],
            {"diffuse": "Nuclear", "focused": "Nuclear"},
        )

        np.testing.assert_allclose(coherence[:4, 0], 0.0)
        np.testing.assert_allclose(coherence[:4, 1], 0.75)
        np.testing.assert_allclose(coherence[4], [0.0, 0.75])

    def test_hierarchical_cell_rescue_requires_callability_marker_and_dgi(self):
        labels = ["Nucleus", "Nucleolus", "Lamina"]
        allowed = {"Nuclear": labels}
        cell_ids = np.asarray(
            [
                "good",
                "good",
                "existing",
                "existing",
                "shallow",
                "shallow",
                "no_dgi",
                "no_dgi",
            ],
            dtype=object,
        )
        broad = np.asarray(["Nuclear"] * 8, dtype=object)
        supported = np.asarray(
            [False, False, True, False, False, False, False, False]
        )
        local_counts = np.zeros((8, 3), dtype=float)
        local_genes = np.zeros((8, 3), dtype=float)
        local_enrichment = np.full((8, 3), -1.0, dtype=float)
        for row in (0, 4, 6):
            local_counts[row, 1] = 1.0
            local_genes[row, 1] = 1.0
            local_enrichment[row, 1] = 1.0

        dgi_support = np.tile([0.2, 0.7, 0.1], (8, 1))
        cell_counts = np.zeros((8, 3), dtype=float)
        cell_genes = np.zeros((8, 3), dtype=float)
        cell_enrichment = np.full((8, 3), -1.0, dtype=float)
        cell_p = np.ones((8, 3), dtype=float)
        for row in (0, 1, 4, 5, 6, 7):
            cell_counts[row, 1] = 2.0
            cell_genes[row, 1] = 2.0
            cell_enrichment[row, 1] = 1.0
            cell_p[row, 1] = 0.01

        rescue_labels, diagnostics = (
            _select_metaspot_hierarchical_cell_rescues(
                cell_ids,
                broad,
                supported,
                local_counts,
                local_genes,
                local_enrichment,
                np.asarray(
                    [True, True, True, True, True, True, False, False]
                ),
                dgi_support,
                cell_counts,
                cell_genes,
                cell_enrichment,
                cell_p,
                np.asarray([6, 6, 6, 6, 3, 3, 6, 6], dtype=float),
                labels=["Nucleolus", "Lamina"],
                all_labels=labels,
                allowed=allowed,
                min_cell_transcripts=10,
                min_metaspots=2,
                min_presence_score=0.60,
                min_localization_score=0.60,
                min_joint_score=0.65,
                min_local_marker_transcripts=1,
                min_local_marker_genes=1,
                min_cell_marker_transcripts=1,
                min_cell_marker_genes=1,
                min_dgi_support=0.35,
                min_dgi_margin=-0.25,
            )
        )

        self.assertEqual(
            rescue_labels.tolist(),
            ["Nucleolus", "", "", "", "", "", "", ""],
        )
        self.assertTrue(diagnostics["callable"][0])
        self.assertFalse(diagnostics["callable"][4])
        self.assertFalse(diagnostics["eligible_cell"][2])
        self.assertTrue(diagnostics["eligible_cell"][6])
        self.assertEqual(diagnostics["status"][0], "selected")
        self.assertEqual(
            diagnostics["status"][2], "already_has_fine_structure"
        )
        self.assertEqual(diagnostics["status"][4], "not_callable")
        self.assertEqual(
            diagnostics["status"][6], "no_anchored_candidate"
        )
        self.assertEqual(diagnostics["best_label"][0], "Nucleolus")
        self.assertGreater(diagnostics["joint_score"][0], 0.65)
        self.assertGreater(diagnostics["best_joint_score"][0], 0.65)
        self.assertTrue(np.isnan(diagnostics["joint_score"][6]))

    def test_omm_cell_coherent_rescue_requires_cell_and_local_evidence(self):
        cell_ids = np.asarray(
            ["supported", "supported", "good", "good", "good", "weak"],
            dtype=object,
        )
        candidate = np.asarray(["OMM"] * 6, dtype=object)
        already_supported = np.asarray([True, False, False, False, False, False])
        broad = np.asarray(["Cytoplasmic"] * 6, dtype=object)
        local_counts = np.asarray([8, 4, 3, 2, 3, 3])
        local_genes = np.asarray([4, 2, 2, 2, 2, 2])
        local_enrichment = np.asarray([2.0, 1.0, 0.8, 0.9, 0.7, 1.0])
        local_gap = np.asarray([1.0, 0.5, 0.4, 0.3, 0.2, 0.5])
        dgi_available = np.ones(6, dtype=bool)
        dgi_support = np.asarray([1.0, 0.9, 0.8, 0.9, 0.7, 0.9])
        dgi_margin = np.asarray([1.0, 0.4, 0.3, 0.4, -0.1, 0.4])
        cell_counts = np.asarray([20, 20, 12, 12, 12, 4])
        cell_genes = np.asarray([6, 6, 5, 5, 5, 2])
        cell_enrichment = np.asarray([1.0, 1.0, 0.8, 0.8, 0.8, 1.0])
        cell_p = np.asarray([0.001, 0.001, 0.01, 0.01, 0.01, 0.001])

        rescued = _select_metaspot_omm_cell_coherent_rescues(
            cell_ids,
            candidate,
            already_supported,
            broad,
            local_counts,
            local_genes,
            local_enrichment,
            local_gap,
            dgi_available,
            dgi_support,
            dgi_margin,
            cell_counts,
            cell_genes,
            cell_enrichment,
            cell_p,
            min_cell_transcripts=8,
            min_cell_genes=4,
            min_cell_enrichment=0.5,
            max_cell_p=0.05,
            min_local_transcripts=2,
            min_local_genes=2,
            min_local_enrichment=0.5,
            min_local_score_gap=0.15,
            min_dgi_support=0.6,
            min_dgi_margin=0.0,
            max_per_cell=2,
        )

        # A pre-existing OMM prevents more calls in that cell. The good cell
        # receives its two strongest local loci; the negative-margin locus and
        # the cell without aggregate multi-gene evidence remain unlabelled.
        self.assertEqual(np.flatnonzero(rescued).tolist(), [2, 3])

    def test_omm_single_marker_anchor_still_requires_cell_and_dgi_gates(self):
        rescued = _select_metaspot_omm_cell_coherent_rescues(
            np.asarray(["eligible", "weak_cell", "weak_dgi"], dtype=object),
            np.asarray(["OMM", "OMM", "OMM"], dtype=object),
            np.zeros(3, dtype=bool),
            np.asarray(["Cytoplasmic"] * 3, dtype=object),
            np.ones(3, dtype=int),
            np.ones(3, dtype=int),
            np.ones(3, dtype=float),
            np.full(3, 0.5),
            np.ones(3, dtype=bool),
            np.asarray([0.50, 0.90, 0.49]),
            np.asarray([0.00, 0.40, -0.01]),
            np.asarray([6, 5, 10]),
            np.asarray([3, 3, 4]),
            np.asarray([0.25, 1.0, 1.0]),
            np.asarray([0.10, 0.01, 0.01]),
            min_cell_transcripts=6,
            min_cell_genes=3,
            min_cell_enrichment=0.25,
            max_cell_p=0.10,
            min_local_transcripts=1,
            min_local_genes=1,
            min_local_enrichment=0.5,
            min_local_score_gap=0.15,
            min_dgi_support=0.50,
            min_dgi_margin=0.0,
            max_per_cell=10,
        )

        # A focal one-marker anchor is sufficient only in the cell that also
        # passes aggregate multi-gene evidence and both local DGI gates.
        self.assertEqual(np.flatnonzero(rescued).tolist(), [0])

    def test_hybrid_metaspot_annotation_restricts_subtypes_and_falls_back(self):
        labels = [
            "Nucleus",
            "Nucleolus",
            "Lamina",
            "Nuclear_Pore",
            "Cytosol",
            "ER",
            "OMM",
        ]
        genes = [f"{label}_G{rep}" for label in labels for rep in range(2)]
        gold = pd.DataFrame({"Common_Gene": genes})
        for label in labels:
            gold[label] = [
                1.0 if gene.startswith(f"{label}_") else 0.0
                for gene in genes
            ]

        X = np.zeros((10, len(genes)), dtype=np.int32)
        gene_pos = {gene: idx for idx, gene in enumerate(genes)}
        for row, label in {
            0: "Nucleolus",
            1: "Nucleolus",
            3: "Nucleus",
            4: "Lamina",
            5: "ER",
            6: "ER",
            8: "Cytosol",
            9: "OMM",
        }.items():
            source_label = label.replace(" ", "_")
            X[row, gene_pos[f"{source_label}_G0"]] = 3
            X[row, gene_pos[f"{source_label}_G1"]] = 2
        obs = pd.DataFrame(
            {
                "cell_id": ["c1"] * 5 + ["c2"] * 5,
                "metaspot": [f"s{i}" for i in range(10)],
                "subcellular_annotation": ["Nucleus"] * 5 + ["Cytosol"] * 5,
            },
            index=[f"s{i}" for i in range(10)],
        )
        adata = ad.AnnData(
            X=sp.csr_matrix(X),
            obs=obs,
            var=pd.DataFrame(index=genes),
        )
        adata.obsm["X_dgi_gcn_metaspot"] = np.eye(
            10,
            dtype=np.float32,
        )
        adata_template = adata.copy()
        cfg = {
            "gold_column_normalize": False,
            "marker_set_target": 20,
            "marker_min_weight": 0.05,
            "marker_min_margin": 0.05,
            "metaspot_pool_k": 1,
            "metaspot_min_marker_transcripts": 1,
            "metaspot_min_marker_genes": 1,
            "metaspot_min_log2_enrichment": 0.0,
            "metaspot_min_score_gap": 0.0,
            "metaspot_subtype_fdr": 1.0,
            "metaspot_exploratory_max_q": 1.0,
            "metaspot_background_prior_strength": 1.0,
            "metaspot_require_omm_confirmation": False,
        }

        result, scores, metadata = (
            _refine_metaspot_annotations_with_local_dgi(
                adata,
                gold,
                cfg,
            )
        )

        final = result.obs["subcellular_annotation"].astype(str)
        self.assertEqual(final.loc["s0"], "Nucleolus")
        self.assertEqual(final.loc["s5"], "ER")
        self.assertEqual(final.loc["s2"], "Nucleus")
        self.assertEqual(final.loc["s7"], "Cytosol")
        self.assertFalse(final.iloc[:5].isin(["Cytosol", "ER", "OMM"]).any())
        self.assertFalse(final.iloc[5:].isin(["Nucleus", "Nucleolus", "Lamina", "Nuclear Pore"]).any())
        self.assertEqual(len(scores), 10)
        self.assertTrue(metadata["same_cell_only"])
        self.assertTrue(metadata["same_broad_compartment_only"])
        self.assertTrue(metadata["dgi_consensus_excludes_self"])
        for key in (
            "X_subcellular_structure_presence",
            "X_subcellular_structure_localization",
            "X_subcellular_structure_evidence",
        ):
            self.assertEqual(result.obsm[key].shape, (10, 5))
            self.assertTrue(np.isfinite(result.obsm[key]).all())
            self.assertTrue((result.obsm[key] >= 0.0).all())
            self.assertTrue((result.obsm[key] <= 1.0).all())
        self.assertEqual(
            metadata["multi_structure_evidence"]["structures"],
            ["Nucleolus", "Lamina", "Nuclear Pore", "ER", "OMM"],
        )
        self.assertFalse(
            metadata["multi_structure_evidence"]["used_for_final_selection"]
        )
        np.testing.assert_allclose(
            scores["structure_evidence__nucleolus"].to_numpy(),
            result.obsm["X_subcellular_structure_evidence"][:, 0],
        )
        self.assertEqual(
            result.uns["cell_structure_evidence"]["presence_score"].shape,
            (2, 5),
        )
        self.assertEqual(
            str(result.obs.loc["s2", "subcellular_fine_annotation"]),
            "Nucleus",
        )
        self.assertEqual(
            str(result.obs.loc["s2", "subcellular_fine_state"]),
            "Broad-prior fallback",
        )
        self.assertEqual(
            str(result.obs.loc["s0", "subcellular_fine_annotation"]),
            "Nucleolus",
        )
        self.assertEqual(
            result.obs.loc["s2", "subcellular_metaspot_fallback_reason"],
            "broad_label_wins",
        )

        lean_result, lean_scores, lean_metadata = (
            _refine_metaspot_annotations_with_local_dgi(
                adata_template.copy(),
                gold,
                cfg,
                build_score_table=False,
            )
        )
        self.assertIsNone(lean_scores)
        self.assertEqual(
            lean_result.obs["subcellular_annotation"].astype(str).tolist(),
            result.obs["subcellular_annotation"].astype(str).tolist(),
        )
        self.assertEqual(
            lean_metadata["final_label_counts"],
            metadata["final_label_counts"],
        )
        self.assertNotIn(
            "subcellular_metaspot_fused_score",
            lean_result.obs.columns,
        )
        np.testing.assert_allclose(
            lean_result.obsm["X_subcellular_structure_evidence"],
            result.obsm["X_subcellular_structure_evidence"],
        )

        # A marker-qualified candidate can enter only the explicitly tagged
        # exploratory tier when leave-one-out DGI neighbors agree but its
        # label-wise BH q-value does not pass the configured cutoff.
        dgi_rescue = adata_template.copy()
        dgi_rescue.obsm["X_dgi_gcn_metaspot"][1] = (
            dgi_rescue.obsm["X_dgi_gcn_metaspot"][0]
            + np.asarray(
                [0.0, 0.001] + [0.0] * 8,
                dtype=np.float32,
            )
        )
        dgi_result, dgi_scores, dgi_metadata = (
            _refine_metaspot_annotations_with_local_dgi(
                dgi_rescue,
                gold,
                {
                    **cfg,
                    "metaspot_pool_k": 2,
                    "metaspot_subtype_fdr": 0.0,
                    "metaspot_exploratory_max_q": 0.0,
                    "metaspot_dgi_seed_min_marker_transcripts": 1,
                    "metaspot_dgi_seed_min_marker_genes": 1,
                    "metaspot_dgi_seed_min_log2_enrichment": 0.0,
                    "metaspot_dgi_min_informative_neighbors": 1,
                    "metaspot_dgi_exploratory_min_support": 0.75,
                    "metaspot_dgi_exploratory_min_margin": 0.25,
                },
            )
        )
        dgi_s0 = dgi_scores.set_index("metaspot").loc["s0"]
        self.assertGreater(float(dgi_s0["candidate_bh_q"]), 0.0)
        self.assertTrue(bool(dgi_s0["dgi_exploratory_supported"]))
        self.assertEqual(
            str(dgi_result.obs.loc["s0", "subcellular_fine_annotation"]),
            "Nucleolus",
        )
        self.assertEqual(
            str(dgi_result.obs.loc["s0", "subcellular_fine_state"]),
            "DGI-supported exploratory fine structure",
        )
        self.assertGreaterEqual(
            dgi_metadata["dgi_exploratory_supported_metaspots"],
            2,
        )

        # The final Tier-B branch can recover one structure in a callable cell
        # when q-value-based paths remain silent, but only through the same
        # measured-marker plus leave-one-out DGI decision path.
        hierarchical = adata_template.copy()
        hierarchical.obsm["X_dgi_gcn_metaspot"][1] = (
            hierarchical.obsm["X_dgi_gcn_metaspot"][0]
            + np.asarray(
                [0.0, 0.001] + [0.0] * 8,
                dtype=np.float32,
            )
        )
        hierarchical_result, hierarchical_scores, hierarchical_metadata = (
            _refine_metaspot_annotations_with_local_dgi(
                hierarchical,
                gold,
                {
                    **cfg,
                    "metaspot_pool_k": 2,
                    "metaspot_subtype_fdr": 0.0,
                    "metaspot_exploratory_max_q": 0.0,
                    "metaspot_dgi_exploratory": False,
                    "metaspot_dgi_seed_min_marker_transcripts": 1,
                    "metaspot_dgi_seed_min_marker_genes": 1,
                    "metaspot_dgi_seed_min_log2_enrichment": 0.0,
                    "metaspot_dgi_min_informative_neighbors": 1,
                    "metaspot_hierarchical_cell_rescue": True,
                    "metaspot_hierarchical_min_cell_transcripts": 1,
                    "metaspot_hierarchical_min_metaspots": 1,
                    "metaspot_hierarchical_min_presence_score": 0.50,
                    "metaspot_hierarchical_min_localization_score": 0.50,
                    "metaspot_hierarchical_min_joint_score": 0.50,
                    "metaspot_hierarchical_min_local_marker_transcripts": 1,
                    "metaspot_hierarchical_min_local_marker_genes": 1,
                    "metaspot_hierarchical_min_cell_marker_transcripts": 1,
                    "metaspot_hierarchical_min_cell_marker_genes": 1,
                    "metaspot_hierarchical_min_dgi_support": 0.35,
                    "metaspot_hierarchical_min_dgi_margin": -0.25,
                },
            )
        )
        hierarchical_rows = hierarchical_scores.loc[
            hierarchical_scores["hierarchical_cell_rescued"]
        ]
        self.assertEqual(len(hierarchical_rows), 2)
        self.assertEqual(hierarchical_rows["cell_id"].nunique(), 2)
        self.assertEqual(
            hierarchical_rows.groupby("cell_id").size().max(), 1
        )
        c1_row = hierarchical_rows.loc[
            hierarchical_rows["cell_id"] == "c1"
        ].iloc[0]
        self.assertEqual(
            c1_row["fine_annotation"], "Nucleolus"
        )
        hierarchical_metaspot = c1_row["metaspot"]
        self.assertEqual(
            str(
                hierarchical_result.obs.loc[
                    hierarchical_metaspot,
                    "subcellular_metaspot_decision_provenance",
                ]
            ),
            "callable_cell_presence_plus_local_marker_plus_dgi",
        )
        self.assertGreater(
            hierarchical_metadata["callable_cell_fine_coverage_after_hierarchical"],
            hierarchical_metadata["callable_cell_fine_coverage_before_hierarchical"],
        )

        # An unidentifiable independent OMM model is unresolved by default;
        # strict-local evidence remains visible in diagnostics.
        unidentifiable = adata_template.copy()
        unidentifiable.uns["omm_metaspot_refinement"] = {
            "identifiable": False,
        }
        required_cfg = {
            **cfg,
            "metaspot_require_omm_confirmation": True,
        }
        unidentifiable_result, unidentifiable_scores, metadata = (
            _refine_metaspot_annotations_with_local_dgi(
                unidentifiable,
                gold,
                required_cfg,
            )
        )
        self.assertEqual(
            str(unidentifiable_result.obs.loc["s9", "subcellular_annotation"]),
            "Cytosol",
        )
        self.assertEqual(
            str(
                unidentifiable_result.obs.loc[
                    "s9", "subcellular_fine_annotation"
                ]
            ),
            "Cytosol",
        )
        self.assertEqual(metadata["omm_confirmation_mode"], "unidentifiable")
        self.assertEqual(metadata["omm_strict_local_candidate_metaspots"], 1)
        self.assertEqual(metadata["omm_strict_local_fallback_metaspots"], 0)
        self.assertEqual(
            unidentifiable_result.obs.loc[
                "s9", "subcellular_metaspot_decision_provenance"
            ],
            "broad_prior_fallback",
        )
        self.assertEqual(
            unidentifiable_scores.set_index("metaspot").loc[
                "s9", "omm_confirmation_status"
            ],
            "unidentifiable_strict_local_unresolved",
        )

        # The legacy behavior remains an explicit expert override.
        strict_local = adata_template.copy()
        strict_local.uns["omm_metaspot_refinement"] = {
            "identifiable": False,
        }
        strict_local_result, _, strict_local_metadata = (
            _refine_metaspot_annotations_with_local_dgi(
                strict_local,
                gold,
                {
                    **required_cfg,
                    "metaspot_omm_unidentifiable_policy": "strict_local",
                },
            )
        )
        self.assertEqual(
            str(strict_local_result.obs.loc["s9", "subcellular_annotation"]),
            "OMM",
        )
        self.assertEqual(
            strict_local_metadata["omm_strict_local_fallback_metaspots"],
            1,
        )

        # When the same local call is exploratory but not strict, it remains
        # audit-only while confirmation is unidentifiable.
        s9_q = float(
            scores.set_index("metaspot").loc["s9", "candidate_bh_q"]
        )
        self.assertGreater(s9_q, 0.0)
        exploratory_only = adata_template.copy()
        exploratory_only.uns["omm_metaspot_refinement"] = {
            "identifiable": False,
        }
        exploratory_cfg = {
            **required_cfg,
            "metaspot_subtype_fdr": s9_q / 2.0,
        }
        exploratory_result, _, _ = (
            _refine_metaspot_annotations_with_local_dgi(
                exploratory_only,
                gold,
                exploratory_cfg,
            )
        )
        self.assertEqual(
            str(exploratory_result.obs.loc["s9", "subcellular_annotation"]),
            "Cytosol",
        )
        self.assertEqual(
            exploratory_result.obs.loc[
                "s9", "subcellular_metaspot_fallback_reason"
            ],
            "omm_confirmation_unidentifiable_unresolved",
        )

        # If the model is identifiable, the original confirmation gate stays
        # in force and an explicitly rejected OMM call still fails closed.
        identifiable = adata_template.copy()
        identifiable.uns["omm_metaspot_refinement"] = {
            "identifiable": True,
        }
        identifiable.obs["omm_domain_gate"] = "rejected"
        identifiable.obs["omm_refined_state"] = "OMM-rejected-by-domain"
        identifiable_result, _, metadata = (
            _refine_metaspot_annotations_with_local_dgi(
                identifiable,
                gold,
                required_cfg,
            )
        )
        self.assertEqual(metadata["omm_confirmation_mode"], "identifiable")
        self.assertEqual(
            str(identifiable_result.obs.loc["s9", "subcellular_annotation"]),
            "Cytosol",
        )
        self.assertEqual(
            identifiable_result.obs.loc[
                "s9", "subcellular_metaspot_fallback_reason"
            ],
            "omm_not_confirmed",
        )

    def test_metaspot_omm_confirmation_missing_outputs_are_unavailable(self):
        adata = ad.AnnData(np.zeros((2, 1), dtype=np.float32))
        adata.uns["omm_metaspot_refinement"] = {"identifiable": True}

        mode, confirmed = _resolve_metaspot_omm_confirmation(
            adata,
            require_confirmation=True,
            posterior_high=0.8,
        )

        self.assertEqual(mode, "unavailable")
        self.assertFalse(confirmed.any())

    def test_independent_evidence_policy_supports_mixed_and_unknown(self):
        index = pd.Index(["d1", "d2", "d3"], name="leiden")
        regions = ["Nucleus", "Cytosol", "ER", "OMM"]
        annot = pd.DataFrame(index=index)
        channel_values = {
            "Nucleus": [5.0, 5.0, 5.0],
            "Cytosol": [0.2, 0.2, 0.2],
            "ER": [4.0, 3.0, 2.0],
            "OMM": [1.0, 3.2, 1.0],
        }
        q_values = {
            "Nucleus": [0.001, 0.001, 0.5],
            "Cytosol": [0.5, 0.5, 0.5],
            "ER": [0.001, 0.001, 0.5],
            "OMM": [0.001, 0.001, 0.5],
        }
        for region in regions:
            for prefix in ("z_mwu", "z_proj", "z_hyp"):
                annot[f"{prefix}__{region}"] = channel_values[region]
            annot[f"subtype_empirical_q__{region}"] = q_values[region]
            annot[f"subtype_testable__{region}"] = True
            annot[f"effective_marker_count__{region}"] = 20

        result = _apply_independent_evidence_policy(
            annot,
            regions,
            group_r_mean=pd.Series([0.5, 0.5, 0.5], index=index),
            subtype_fdr=0.05,
            min_channel_support=2,
            mixed_score_delta=0.75,
        )

        self.assertEqual(result.loc["d1", "label"], "ER")
        self.assertEqual(result.loc["d1", "annotation_state"], "Resolved")
        self.assertNotIn("Nucleus", result.loc["d1", "supported_labels"])
        self.assertEqual(result.loc["d2", "annotation_state"], "Mixed")
        self.assertEqual(result.loc["d2", "primary_label"], "OMM")
        self.assertEqual(result.loc["d2", "secondary_label"], "ER")
        self.assertEqual(result.loc["d3", "label"], "Unknown")
        self.assertEqual(result.loc["d3", "annotation_state"], "Unknown")

    def test_independent_tracks_preserve_expression_only_counterfactual(self):
        index = pd.Index(["nuclear-domain"], name="leiden")
        regions = ["Nucleus", "Cytosol", "ER"]
        annot = pd.DataFrame(index=index)
        values = {
            "Nucleus": [2.0],
            "Cytosol": [0.5],
            "ER": [5.0],
        }
        for region in regions:
            for prefix in ("z_mwu", "z_proj", "z_hyp"):
                annot[f"{prefix}__{region}"] = values[region]
            annot[f"subtype_empirical_q__{region}"] = [0.001]
            annot[f"subtype_testable__{region}"] = True
            annot[f"effective_marker_count__{region}"] = [20]

        result = _apply_independent_evidence_tracks(
            annot,
            regions,
            group_r_mean=pd.Series([-0.8], index=index),
            assignment_mode="hierarchical",
            subtype_fdr=0.05,
            min_channel_support=2,
        )

        self.assertEqual(result.loc["nuclear-domain", "label"], "Nucleus")
        self.assertEqual(
            result.loc["nuclear-domain", "expression_only_label"],
            "ER",
        )
        self.assertEqual(
            result.loc[
                "nuclear-domain",
                "expression_only_candidate_label",
            ],
            "ER",
        )
        self.assertTrue(
            result.loc[
                "nuclear-domain",
                "domain_compartment_feasibility_applied",
            ]
        )
        self.assertFalse(
            result.loc["nuclear-domain", "geometry_feasible__ER"]
        )

    def test_independent_evidence_policy_tiered_keeps_strict_audit(self):
        index = pd.Index(["exploratory", "candidate"], name="leiden")
        regions = ["Cytosol", "ER", "OMM"]
        annot = pd.DataFrame(index=index)
        values = {
            "Cytosol": [0.5, 0.5],
            "ER": [3.0, 3.0],
            "OMM": [1.0, 1.0],
        }
        q_values = {
            "Cytosol": [0.90, 0.90],
            "ER": [0.20, 0.80],
            "OMM": [0.90, 0.90],
        }
        for region in regions:
            for prefix in ("z_mwu", "z_proj", "z_hyp"):
                annot[f"{prefix}__{region}"] = values[region]
            annot[f"subtype_empirical_q__{region}"] = q_values[region]
            annot[f"subtype_testable__{region}"] = True
            annot[f"effective_marker_count__{region}"] = 20

        result = _apply_independent_evidence_policy(
            annot,
            regions,
            subtype_fdr=0.05,
            min_channel_support=2,
            assignment_mode="tiered",
            exploratory_max_q=0.30,
            exploratory_min_channel_support=2,
            exploratory_min_score_gap=0.50,
        )

        self.assertEqual(result.loc["exploratory", "label"], "ER")
        self.assertEqual(
            result.loc["exploratory", "strict_label"],
            "Unknown",
        )
        self.assertEqual(
            result.loc["exploratory", "annotation_tier"],
            "Exploratory",
        )
        self.assertEqual(
            result.loc["candidate", "annotation_tier"],
            "Candidate",
        )
        self.assertEqual(
            result.loc["candidate", "exploratory_label"],
            "Unknown",
        )
        self.assertFalse(
            bool(
                result.loc[
                    "exploratory",
                    "subtype_empirical_supported",
                ]
            )
        )
        self.assertEqual(
            result.loc["exploratory", "primary_empirical_q"],
            0.20,
        )

    def test_independent_hierarchical_mode_falls_back_to_broad_parent(self):
        index = pd.Index(
            ["exploratory", "candidate"],
            name="leiden",
        )
        regions = ["Cytosol", "ER", "OMM"]
        annot = pd.DataFrame(index=index)
        values = {
            "Cytosol": [1.0, 1.0],
            "ER": [3.0, 3.0],
            "OMM": [0.5, 0.5],
        }
        q_values = {
            "Cytosol": [0.90, 0.90],
            "ER": [0.20, 0.80],
            "OMM": [0.90, 0.90],
        }
        for region in regions:
            for prefix in ("z_mwu", "z_proj", "z_hyp"):
                annot[f"{prefix}__{region}"] = values[region]
            annot[f"subtype_empirical_q__{region}"] = q_values[region]
            annot[f"subtype_testable__{region}"] = True
            annot[f"effective_marker_count__{region}"] = 20

        result = _apply_independent_evidence_policy(
            annot,
            regions,
            assignment_mode="hierarchical",
            subtype_fdr=0.05,
            exploratory_max_q=0.30,
            exploratory_min_channel_support=2,
            exploratory_min_score_gap=0.50,
        )

        self.assertEqual(result.loc["exploratory", "label"], "ER")
        self.assertEqual(
            result.loc["exploratory", "annotation_tier"],
            "Exploratory",
        )
        self.assertEqual(result.loc["candidate", "label"], "Cytosol")
        self.assertEqual(
            result.loc["candidate", "candidate_label"],
            "ER",
        )
        self.assertEqual(
            result.loc["candidate", "annotation_state"],
            "BroadFallback",
        )
        self.assertEqual(
            result.loc["candidate", "strict_label"],
            "Unknown",
        )

    def test_independent_choice_excludes_marker_size_biased_projection(self):
        index = pd.Index(["domain"], name="leiden")
        regions = ["Cytosol", "ER"]
        annot = pd.DataFrame(index=index)
        annot["z_mwu__Cytosol"] = [2.0]
        annot["z_proj__Cytosol"] = [2.0]
        annot["z_hyp__Cytosol"] = [2.0]
        annot["z_mwu__ER"] = [1.0]
        annot["z_proj__ER"] = [10.0]
        annot["z_hyp__ER"] = [1.0]
        for region in regions:
            annot[f"subtype_empirical_q__{region}"] = [0.80]
            annot[f"subtype_testable__{region}"] = [True]
            annot[f"effective_marker_count__{region}"] = [20]
        annot["subtype_calibrated_score__Cytosol"] = [2.5]
        annot["subtype_calibrated_score__ER"] = [0.5]

        result = _apply_independent_evidence_policy(
            annot,
            regions,
            assignment_mode="tiered",
        )

        self.assertEqual(result.loc["domain", "candidate_label"], "Cytosol")
        self.assertGreater(
            result.loc["domain", "raw_evidence_score__ER"],
            result.loc["domain", "raw_evidence_score__Cytosol"],
        )
        self.assertGreater(
            result.loc["domain", "evidence_score__Cytosol"],
            result.loc["domain", "evidence_score__ER"],
        )

    def test_independent_omm_requires_raw_er_cytosol_specificity(self):
        index = pd.Index(["shared", "specific"], name="leiden")
        regions = ["Cytosol", "ER", "OMM"]
        annot = pd.DataFrame(index=index)
        values = {
            "Cytosol": [1.0, 1.0],
            "ER": [5.0, 2.0],
            "OMM": [4.0, 4.0],
        }
        for region in regions:
            for prefix in ("z_mwu", "z_proj", "z_hyp"):
                annot[f"{prefix}__{region}"] = values[region]
            annot[f"subtype_empirical_q__{region}"] = [0.01, 0.01]
            annot[f"subtype_testable__{region}"] = True
            annot[f"effective_marker_count__{region}"] = 20

        result = _apply_independent_evidence_policy(
            annot,
            regions,
            subtype_fdr=0.05,
            assignment_mode="hierarchical",
        )

        self.assertFalse(result.loc["shared", "omm_specificity_pass"])
        self.assertEqual(
            result.loc["shared", "omm_domain_state"],
            "ER-OMM shared",
        )
        self.assertEqual(result.loc["shared", "label"], "ER")
        self.assertTrue(result.loc["specific", "omm_specificity_pass"])
        self.assertEqual(
            result.loc["specific", "omm_domain_state"],
            "Confirmed",
        )
        self.assertEqual(result.loc["specific", "label"], "OMM")

    def test_omm_metaspot_labels_require_both_domain_and_local_support(self):
        index = pd.Index(
            [
                "high",
                "middle",
                "low",
                "rejected",
                "rejected_omm",
                "shared_low",
            ]
        )
        result = _compose_omm_metaspot_labels(
            pd.Series(
                ["OMM", "OMM", "OMM", "ER", "OMM", "OMM"],
                index=index,
            ),
            pd.Series(
                [
                    "OMM-supported",
                    "ER-OMM shared",
                    "OMM-not-supported",
                    "OMM-rejected-by-domain",
                    "OMM-rejected-by-domain",
                    "OMM-not-supported",
                ],
                index=index,
            ),
            pd.Series(
                [
                    "confirmed",
                    "confirmed",
                    "confirmed",
                    "rejected",
                    "rejected",
                    "shared",
                ],
                index=index,
            ),
            pd.Series(
                ["ER", "ER", "ER", "Cytosol", "Cytosol", "ER"],
                index=index,
            ),
            identifiable=True,
            ambiguous_policy="competitor",
        )

        self.assertEqual(
            result.tolist(),
            [
                "OMM",
                "ER",
                "ER",
                "ER",
                "Cytosol",
                "ER",
            ],
        )

    def test_balanced_omm_policy_resolves_shared_domain_with_high_local_support(
        self,
    ):
        index = pd.Index(
            [
                "confirmed-high",
                "shared-high",
                "shared-middle",
                "rejected-high",
            ]
        )
        result = _compose_omm_metaspot_labels(
            pd.Series(["OMM", "ER", "ER", "OMM"], index=index),
            pd.Series(
                [
                    "OMM-supported",
                    "ER-OMM shared",
                    "ER-OMM shared",
                    "OMM-rejected-by-domain",
                ],
                index=index,
            ),
            pd.Series(
                ["confirmed", "shared", "shared", "rejected"],
                index=index,
            ),
            pd.Series(["ER", "ER", "Cytosol", "Cytosol"], index=index),
            pd.Series([0.90, 0.95, 0.60, 0.99], index=index),
            identifiable=True,
            ambiguous_policy="competitor",
            decision_policy="balanced",
            posterior_high=0.80,
        )

        self.assertEqual(
            result.tolist(),
            ["OMM", "OMM", "Cytosol", "Cytosol"],
        )

    def test_balanced_omm_policy_requires_local_posteriors(self):
        index = pd.Index(["shared"])
        with self.assertRaisesRegex(ValueError, "reference_posteriors"):
            _compose_omm_metaspot_labels(
                pd.Series(["ER"], index=index),
                pd.Series(["ER-OMM shared"], index=index),
                pd.Series(["shared"], index=index),
                identifiable=True,
                decision_policy="balanced",
            )

    def test_omm_metaspot_labels_fail_closed_when_model_is_unidentifiable(self):
        index = pd.Index(["confirmed", "rejected", "shared"])
        result = _compose_omm_metaspot_labels(
            pd.Series(["OMM", "OMM", "ER"], index=index),
            pd.Series(["Unidentifiable"] * 3, index=index),
            pd.Series(["confirmed", "rejected", "shared"], index=index),
            pd.Series(["ER", "Cytosol", "ER"], index=index),
            identifiable=False,
        )

        self.assertEqual(
            result.tolist(),
            ["ER-OMM shared", "Cytosol", "ER-OMM shared"],
        )

    def test_omm_metaspot_labels_competitor_policy_fails_closed(self):
        index = pd.Index(["confirmed", "rejected", "shared"])
        result = _compose_omm_metaspot_labels(
            pd.Series(["OMM", "OMM", "ER"], index=index),
            pd.Series(["Unidentifiable"] * 3, index=index),
            pd.Series(["confirmed", "rejected", "shared"], index=index),
            pd.Series(["ER", "Cytosol", "ER"], index=index),
            identifiable=False,
            ambiguous_policy="competitor",
        )

        self.assertEqual(
            result.tolist(),
            ["ER", "Cytosol", "ER"],
        )

    def test_omm_metaspot_labels_reject_unknown_ambiguity_policy(self):
        index = pd.Index(["spot"])
        with self.assertRaisesRegex(ValueError, "ambiguous_policy"):
            _compose_omm_metaspot_labels(
                pd.Series(["OMM"], index=index),
                pd.Series(["ER-OMM shared"], index=index),
                pd.Series(["confirmed"], index=index),
                pd.Series(["ER"], index=index),
                identifiable=True,
                ambiguous_policy="unsupported",
            )

    def test_omm_metaspot_competitor_policy_removes_residual_shared_label(self):
        index = pd.Index(["legacy"])
        result = _compose_omm_metaspot_labels(
            pd.Series(["ER-OMM shared"], index=index),
            pd.Series(["OMM-rejected-by-domain"], index=index),
            pd.Series(["rejected"], index=index),
            pd.Series(["ER"], index=index),
            identifiable=True,
            ambiguous_policy="competitor",
        )

        self.assertEqual(result.tolist(), ["ER"])

    def test_point_geometry_prefers_input_nucleus_flags_for_unordered_boundaries(self):
        transcripts = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "x_location": [0.25, 0.75, 0.25, 0.75],
                "y_location": [0.25, 0.25, 0.75, 0.75],
                "feature_name": ["A", "A", "B", "B"],
                "overlaps_nucleus": [1, 0, 1, 0],
            }
        )
        # Deliberately unordered raster-like boundary points.
        nucleus = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "vertex_x": [0.0, 1.0, 0.0, 1.0],
                "vertex_y": [0.0, 1.0, 1.0, 0.0],
            }
        )
        cell = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "vertex_x": [-1.0, 2.0, 2.0, -1.0],
                "vertex_y": [-1.0, -1.0, 2.0, 2.0],
            }
        )

        result = build_point_geometry_v2(
            transcripts,
            nucleus,
            cell,
            sampled_cells=["cell"],
            include_directional=False,
        )

        self.assertEqual(result["in_nucleus"].tolist(), [True, False, True, False])
        self.assertTrue((result.loc[result["in_nucleus"], "r_signed"] <= 0).all())
        self.assertTrue((result.loc[~result["in_nucleus"], "r_signed"] >= 0).all())

    def test_point_geometry_can_recompute_nucleus_flags_from_polygon(self):
        transcripts = pd.DataFrame(
            {
                "cell_id": ["cell", "cell"],
                "x_location": [0.5, 1.5],
                "y_location": [0.5, 0.5],
                "feature_name": ["A", "B"],
                # Deliberately wrong; benchmark mode must ignore this column.
                "overlaps_nucleus": [0, 1],
            }
        )
        nucleus = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "vertex_x": [0.0, 1.0, 1.0, 0.0],
                "vertex_y": [0.0, 0.0, 1.0, 1.0],
            }
        )
        cell = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "vertex_x": [-1.0, 2.0, 2.0, -1.0],
                "vertex_y": [-1.0, -1.0, 2.0, 2.0],
            }
        )

        result = build_point_geometry_v2(
            transcripts,
            nucleus,
            cell,
            sampled_cells=["cell"],
            include_directional=False,
            use_input_nucleus_flags=False,
        )

        self.assertEqual(result["in_nucleus"].tolist(), [True, False])

    def test_signed_geometry_stretch_preserves_boundary_sign(self):
        values = np.array([-4.0, -1.0, -0.1, 0.0, 0.1, 1.0, 4.0] * 3)
        stretched = _stretch_signed_geometry(values, cell_size=len(values))

        self.assertTrue((stretched[values < 0] <= 0).all())
        self.assertTrue((stretched[values > 0] >= 0).all())
        self.assertEqual(float(stretched[values == 0][0]), 0.0)

    def test_subtype_geometry_scores_are_directional(self):
        regions = [
            "Nucleus",
            "Nucleolus",
            "Lamina",
            "Nuclear Pore",
        ]
        boundary = _subtype_geometry_scores(
            regions,
            r_value=-0.8,
            center=-0.5,
            scale=0.1,
        )
        deep = _subtype_geometry_scores(
            regions,
            r_value=-0.2,
            center=-0.5,
            scale=0.1,
        )

        self.assertGreater(boundary[2], boundary[1])
        self.assertGreater(deep[1], deep[2])
        self.assertEqual(boundary[2], boundary[3])

    def test_compartment_geometry_scores_are_symmetric_and_bounded(self):
        regions = ["Nucleus", "Lamina", "Cytosol", "ER"]
        nuclear_side = _compartment_geometry_scores(
            regions,
            r_value=-0.6,
            tau=0.2,
            min_abs_r=0.05,
        )
        cytoplasmic_side = _compartment_geometry_scores(
            regions,
            r_value=0.6,
            tau=0.2,
            min_abs_r=0.05,
        )

        np.testing.assert_allclose(nuclear_side, -cytoplasmic_side)
        self.assertGreater(nuclear_side[0], 0.0)
        self.assertLess(nuclear_side[3], 0.0)
        self.assertTrue((np.abs(nuclear_side) <= 1.0).all())

    def test_compartment_geometry_scores_keep_boundary_as_soft_dead_zone(self):
        scores = _compartment_geometry_scores(
            ["Nucleus", "ER"],
            r_value=-0.01,
            tau=0.2,
            min_abs_r=0.05,
        )

        np.testing.assert_array_equal(scores, np.zeros(2, dtype=np.float32))

    def test_compartment_score_can_reverse_er_overcall_without_hard_gate(self):
        regions = ["Nucleus", "ER"]
        expression_scores = np.array([0.0, 10.0], dtype=np.float32)
        geometry_scores = _compartment_geometry_scores(
            regions,
            r_value=-0.6,
            tau=0.2,
            min_abs_r=0.05,
        )
        adjusted = expression_scores + 6.0 * geometry_scores

        self.assertEqual(regions[int(np.argmax(expression_scores))], "ER")
        self.assertEqual(regions[int(np.argmax(adjusted))], "Nucleus")

        near_boundary = _compartment_geometry_scores(
            regions,
            r_value=-0.06,
            tau=0.2,
            min_abs_r=0.05,
        )
        weakly_adjusted = expression_scores + 6.0 * near_boundary
        self.assertEqual(regions[int(np.argmax(weakly_adjusted))], "ER")

    def test_single_stage_compartment_guard_rescues_supported_candidate(self):
        regions = ["Nucleus", "Nucleolus", "Cytosol", "OMM"]
        z_scores = np.array([2.1, 3.2, 8.0, 0.5])
        q_scores = np.array([0.03, 0.01, 1e-8, 0.5])
        support = np.array([2, 2, 3, 1])

        idx, rescued, preferred, gap = _choose_geometry_consistent_label(
            regions,
            z_scores,
            q_scores,
            support,
            chosen_idx=2,
            r_value=-0.4,
            max_score_gap=6.0,
        )

        self.assertTrue(rescued)
        self.assertEqual(preferred, "Nuclear")
        self.assertEqual(regions[idx], "Nucleolus")
        self.assertAlmostEqual(gap, 4.8)

    def test_single_stage_compartment_guard_keeps_overwhelming_winner(self):
        regions = ["Nucleus", "Nucleolus", "Cytosol", "OMM"]
        z_scores = np.array([2.1, 3.2, 15.0, 0.5])
        q_scores = np.array([0.03, 0.01, 1e-12, 0.5])
        support = np.array([2, 2, 3, 1])

        idx, rescued, preferred, gap = _choose_geometry_consistent_label(
            regions,
            z_scores,
            q_scores,
            support,
            chosen_idx=2,
            r_value=-0.4,
            max_score_gap=6.0,
        )

        self.assertFalse(rescued)
        self.assertEqual(preferred, "Nuclear")
        self.assertEqual(regions[idx], "Cytosol")
        self.assertAlmostEqual(gap, 11.8)

    def test_single_stage_compartment_guard_ignores_boundary_geometry(self):
        regions = ["Nucleus", "Nucleolus", "Cytosol", "OMM"]
        z_scores = np.array([2.1, 3.2, 8.0, 0.5])
        q_scores = np.array([0.03, 0.01, 1e-8, 0.5])
        support = np.array([2, 2, 3, 1])

        idx, rescued, preferred, _ = _choose_geometry_consistent_label(
            regions,
            z_scores,
            q_scores,
            support,
            chosen_idx=2,
            r_value=-0.01,
            min_abs_r=0.05,
            max_score_gap=6.0,
        )

        self.assertFalse(rescued)
        self.assertEqual(preferred, "")
        self.assertEqual(regions[idx], "Cytosol")

    def test_local_compartment_guard_prevents_cross_boundary_labels(self):
        labels = pd.Series(["Lamina", "ER", "Nucleus", "Cytosol"])
        r_values = pd.Series([0.2, -0.1, -0.5, 0.6])

        result, conflict = _apply_local_compartment_guard(
            labels,
            r_values,
        )

        self.assertEqual(
            result.tolist(),
            ["Cytosol", "Nucleus", "Nucleus", "Cytosol"],
        )
        self.assertEqual(conflict.tolist(), [True, True, False, False])

    def test_metaspot_guard_uses_median_side_instead_of_mean_distance(self):
        obs = pd.DataFrame(
            {
                "r_mean": [-0.4, 0.3],
                "r_p50": [0.1, -0.2],
            },
            index=["spot-a", "spot-b"],
        )
        labels = pd.Series(
            ["ER", "Nucleus"],
            index=obs.index,
        )

        guard_values = _metaspot_compartment_guard_values(obs)
        self.assertIsNotNone(guard_values)
        result, conflict = _apply_local_compartment_guard(
            labels,
            guard_values,
        )

        self.assertEqual(result.tolist(), ["ER", "Nucleus"])
        self.assertEqual(conflict.tolist(), [False, False])

    def test_metaspot_guard_does_not_fall_back_to_mean_distance(self):
        obs = pd.DataFrame(
            {"r_mean": [-0.4]},
            index=["legacy-spot"],
        )

        self.assertIsNone(_metaspot_compartment_guard_values(obs))

    def test_primary_annotation_is_broadcast_from_local_metaspots(self):
        adata = ad.AnnData(np.zeros((2, 1), dtype=np.float32))
        adata.obs_names = ["spot-a", "spot-b"]
        adata.obs["metaspot"] = ["cell-1::0", "cell-1::1"]
        adata.obs["subcellular_annotation"] = pd.Categorical(
            ["Nucleus", "Cytosol"]
        )
        transcripts = pd.DataFrame(
            {
                "metaspot": [
                    "cell-1::1",
                    "cell-1::0",
                    "cell-1::1",
                ]
            }
        )

        mapped = _map_metaspot_labels_to_transcripts(
            adata,
            transcripts,
        )

        self.assertEqual(
            mapped.tolist(),
            ["Cytosol", "Nucleus", "Cytosol"],
        )

    def test_primary_annotation_mapping_rejects_missing_metaspots(self):
        adata = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
        adata.obs["metaspot"] = ["cell-1::0"]
        adata.obs["subcellular_annotation"] = ["Nucleus"]
        transcripts = pd.DataFrame({"metaspot": ["cell-1::missing"]})

        with self.assertRaisesRegex(
            ValueError,
            "every transcript",
        ):
            _map_metaspot_labels_to_transcripts(
                adata,
                transcripts,
            )

    def test_shared_row_indexer_preserves_categorical_codes(self):
        source_ids = pd.Index(["spot-a", "spot-b"])
        target_ids = pd.Series(["spot-b", "spot-a", "spot-b"])
        source = pd.Series(
            pd.Categorical(
                ["Nucleus", "ER"],
                categories=["Nucleus", "ER", "OMM"],
            ),
            index=source_ids,
            name="label",
        )

        row_indexer = _key_row_indexer(
            source_ids,
            target_ids,
            error_context="test",
        )
        taken = _take_obs_column_by_row_indexer(
            source,
            row_indexer,
            pd.RangeIndex(3),
        )

        self.assertEqual(row_indexer.dtype, np.int32)
        self.assertIsInstance(taken.dtype, pd.CategoricalDtype)
        self.assertEqual(taken.tolist(), ["ER", "Nucleus", "ER"])
        self.assertEqual(
            taken.cat.categories.tolist(),
            ["Nucleus", "ER", "OMM"],
        )
        numeric_target_indexer = _key_row_indexer(
            pd.Index(["1", "2"]),
            pd.Series([2, 1]),
            error_context="numeric fallback test",
        )
        np.testing.assert_array_equal(numeric_target_indexer, [1, 0])

    def test_integer_cell_coverage_matches_pair_deduplication(self):
        cell_ids = pd.Series(
            ["c1", "c1", "c2", "c2", "c3", "c3", None]
        )
        labels = pd.Categorical(
            ["Nucleus", "ER", "ER", "ER", "Nucleus", "ER", "ER"]
        )
        confidence = pd.Categorical(
            ["High", "High", "High", "Low", "High", "High", "High"]
        )

        total_cells, coverage = _high_confidence_cell_coverage(
            cell_ids,
            labels,
            confidence,
        )
        self.assertEqual(total_cells, 3)
        self.assertEqual(coverage.to_dict(), {"ER": 3, "Nucleus": 2})

        mask_a = np.asarray(labels == "ER")
        mask_b = np.asarray(confidence == "High")
        total_cells, counts = _unique_cell_counts_for_masks(
            cell_ids,
            [mask_a, mask_b, mask_a & mask_b],
        )
        self.assertEqual(total_cells, 3)
        self.assertEqual(counts, [3, 3, 3])

    def test_h5ad_sanitizer_keeps_compact_categoricals(self):
        adata = ad.AnnData(np.zeros((3, 1), dtype=np.float32))
        adata.obs["label"] = pd.Categorical(
            ["Nucleus", "ER", "Nucleus"]
        )
        adata.obs["scalar_text"] = pd.Series(
            ["a", "b", "c"],
            index=adata.obs_names,
            dtype=object,
        )
        adata.obs["sequence"] = pd.Series(
            [["a", "b"], ["c"], None],
            index=adata.obs_names,
            dtype=object,
        )

        _sanitize_obs_for_h5ad(adata)

        self.assertIsInstance(adata.obs["label"].dtype, pd.CategoricalDtype)
        self.assertEqual(adata.obs["scalar_text"].tolist(), ["a", "b", "c"])
        self.assertEqual(adata.obs["sequence"].tolist(), ["a,b", "c", ""])

    def test_spatial_false_disables_subtype_geometry_checks(self):
        adata = ad.AnnData(np.zeros((8, 40), dtype=np.float32))
        adata.var_names = [f"G{i}" for i in range(40)]
        adata.obs["leiden"] = pd.Categorical(
            ["0", "0", "1", "1", "2", "2", "3", "3"]
        )
        adata.obs["r_mean"] = [
            -0.9,
            -0.9,
            -0.7,
            -0.7,
            -0.3,
            -0.3,
            -0.1,
            -0.1,
        ]
        regions = ["Lamina", "Nucleus", "Nucleolus", "Nuclear Pore"]
        sets = {
            region: [f"G{i}" for i in range(j * 10, (j + 1) * 10)]
            for j, region in enumerate(regions)
        }

        result = _rapx_annotate_core(
            adata,
            sets,
            regions,
            use_spatial_channel=False,
            enforce_geometry_consistency=True,
        )

        self.assertTrue(result["geometry_consistent"].all())
        self.assertFalse(
            any(column.startswith("z_spatial__") for column in result.columns)
        )

    def test_weak_top1_demotion_updates_selected_candidate_state(self):
        annotation = pd.DataFrame(
            {
                "label": ["ER", "OMM"],
                "top1": ["ER", "ER"],
                "z1": [1.1, 1.1],
                "q1": [0.5, 0.5],
                "top2": ["OMM", "OMM"],
                "z2": [1.0, 1.0],
                "q2": [0.02, 0.02],
                "selected_z": [1.1, 1.0],
                "selected_q": [0.5, 0.02],
                "channel_support": [1, 1],
                "raw_combined_score": [5.0, 4.0],
                "low_conf": [False, False],
                "z_comb__OMM": [4.0, 4.0],
                "z_mwu__OMM": [2.0, 2.0],
                "z_proj__OMM": [1.0, 1.0],
                "z_hyp__OMM": [-0.5, -0.5],
            },
            index=["selected_top1", "already_top2"],
        )

        result = _demote_weak_top1(annotation, demote_labels=("ER",))

        self.assertEqual(result.loc["selected_top1", "label"], "OMM")
        self.assertEqual(result.loc["selected_top1", "core_candidate_label"], "ER")
        self.assertAlmostEqual(result.loc["selected_top1", "selected_z"], 1.0)
        self.assertAlmostEqual(result.loc["selected_top1", "selected_q"], 0.02)
        self.assertAlmostEqual(result.loc["selected_top1", "raw_combined_score"], 4.0)
        self.assertEqual(result.loc["selected_top1", "channel_support"], 2)
        self.assertTrue(result.loc["selected_top1", "weak_top1_demoted"])
        self.assertFalse(result.loc["already_top2", "weak_top1_demoted"])

    def test_strong_single_channel_evidence_is_medium_confidence(self):
        annotation = pd.DataFrame(
            {
                "label": ["ER"],
                "low_conf": [True],
                "selected_z": [4.15],
                "selected_q": [0.00005],
                "channel_support": [1],
                "geometry_consistent": [True],
            }
        )

        result = _apply_low_confidence_policy(
            annotation,
            compartment="Cytoplasmic",
        )

        self.assertEqual(result["confidence_level"].tolist(), ["Medium"])
        self.assertEqual(result["label"].tolist(), ["ER"])

    def test_nuclear_geometry_thresholds_use_correct_direction(self):
        boundary, deep = _nuclear_geometry_thresholds(
            np.array([-0.9, -0.7, -0.3, -0.1]),
            quantile=0.25,
        )

        self.assertLess(boundary, deep)
        self.assertAlmostEqual(boundary, -0.75)
        self.assertAlmostEqual(deep, -0.25)

    def test_lamina_guard_requires_expression_and_envelope_geometry(self):
        annotation = pd.DataFrame(
            {
                "label": ["Lamina", "Lamina", "Lamina", "Cytosol"],
                "top1": ["Lamina", "Lamina", "Lamina", "Cytosol"],
                "top2": ["Nucleus", "Nucleus", "Nucleus", "Lamina"],
                "z1": [5.0, 2.0, 6.0, 7.0],
                "z2": [1.0, 1.0, 1.0, 1.0],
                "q1": [0.001, 0.001, 0.001, 0.001],
                "channel_support": [3, 3, 3, 3],
                "low_conf": [False, False, False, False],
                "geometry_consistent": [False, False, True, True],
            },
            index=["0", "1", "2", "3"],
        )
        r_mean = pd.Series([-0.8, -0.7, -0.6, -0.5], index=annotation.index)
        d_nuc = pd.Series([0.05, 0.10, 0.40, 0.20], index=annotation.index)

        result = _guard_lamina_overcall(
            annotation,
            group_r_mean=r_mean,
            group_d_nuc_norm_mean=d_nuc,
            boundary_quantile=0.5,
        )

        self.assertEqual(
            result["label"].tolist(),
            ["Lamina", "Nucleus", "Nucleus", "Cytosol"],
        )
        self.assertEqual(
            result["lamina_guarded"].tolist(),
            [False, True, True, False],
        )
        self.assertEqual(result.loc["1", "candidate_label"], "Lamina")
        self.assertTrue(result.loc["0", "geometry_consistent"])
        self.assertFalse(result.loc["2", "lamina_geometry_consistent"])
        self.assertFalse(result.loc["2", "geometry_consistent"])
        self.assertAlmostEqual(
            result["lamina_boundary_threshold"].iloc[0],
            0.15,
        )

    def test_lamina_guard_uses_stricter_margin_against_cytoplasmic_label(self):
        annotation = pd.DataFrame(
            {
                "label": ["Lamina"],
                "top1": ["Lamina"],
                "top2": ["ER"],
                "z1": [5.0],
                "z2": [2.5],
                "q1": [0.001],
                "channel_support": [3],
                "low_conf": [False],
            },
            index=["0"],
        )

        result = _guard_lamina_overcall(
            annotation,
            group_r_mean=pd.Series([-0.8], index=annotation.index),
            group_d_nuc_norm_mean=pd.Series([0.05], index=annotation.index),
            min_margin=2.0,
            non_nuclear_top2_min_margin=3.0,
        )

        self.assertEqual(result.loc["0", "label"], "Nucleus")
        self.assertTrue(result.loc["0", "lamina_guarded"])

    def test_nuclear_pore_guard_only_handles_selected_candidates(self):
        annotation = pd.DataFrame(
            {
                "label": ["Nuclear Pore", "Nuclear Pore", "Lamina"],
                "top1": ["Nuclear Pore", "Nuclear Pore", "Nuclear Pore"],
                "top2": ["Nucleus", "Nucleus", "Lamina"],
                "z1": [4.0, 1.1, 1.1],
                "z2": [1.0, 1.0, 1.0],
                "q1": [0.001, 0.4, 0.4],
                "selected_q": [0.001, 0.4, 0.05],
                "channel_support": [3, 1, 2],
                "low_conf": [False, False, True],
                "z_mwu__Nuclear Pore": [2.0, 2.0, 2.0],
                "z_proj__Nuclear Pore": [1.0, -1.0, -1.0],
                "z_hyp__Nuclear Pore": [1.0, 1.0, 1.0],
            },
            index=["strong", "weak", "already_lamina"],
        )

        result = _guard_nuclear_pore_overcall(annotation)

        self.assertEqual(
            result["label"].tolist(),
            ["Nuclear Pore", "Nucleus", "Lamina"],
        )
        self.assertEqual(
            result["nuclear_pore_guarded"].tolist(),
            [False, True, False],
        )

    def test_empirical_marker_calibration_is_reproducible_and_label_specific(self):
        scores = np.zeros((4, 40), dtype=np.float32)
        scores[0, :10] = 8.0
        scores[1, 10:20] = 8.0
        scores[2, 20:30] = 2.0
        scores[3, 30:40] = 2.0
        regions = ["Lamina", "OMM", "Nucleolus"]
        region_indices = {
            "Lamina": np.arange(0, 10, dtype=np.int64),
            "OMM": np.arange(10, 20, dtype=np.int64),
            "Nucleolus": np.arange(20, 25, dtype=np.int64),
        }
        region_weights = {}
        for region, idx in region_indices.items():
            weights = np.zeros(40, dtype=np.float32)
            weights[idx] = 1.0 / np.sqrt(len(idx))
            region_weights[region] = weights

        first = _empirical_marker_set_calibration(
            scores,
            ["0", "1", "2", "3"],
            regions,
            region_indices,
            region_weights,
            expression_matrix=np.abs(scores),
            permutations=199,
            min_effective_markers=10,
            seed=7,
        )
        second = _empirical_marker_set_calibration(
            scores,
            ["0", "1", "2", "3"],
            regions,
            region_indices,
            region_weights,
            expression_matrix=np.abs(scores),
            permutations=199,
            min_effective_markers=10,
            seed=7,
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["subtype_testable__Lamina"].all())
        self.assertEqual(first["effective_marker_count__OMM"].iloc[0], 10)
        self.assertFalse(first["subtype_testable__Nucleolus"].any())
        self.assertTrue(first["subtype_empirical_q__Nucleolus"].isna().all())
        self.assertTrue(
            np.isfinite(
                first["subtype_calibrated_score__Lamina"]
            ).all()
        )
        self.assertTrue(
            first["subtype_calibrated_score__Nucleolus"].isna().all()
        )
        self.assertLessEqual(first.loc["0", "subtype_empirical_q__Lamina"], 0.05)
        self.assertLessEqual(first.loc["1", "subtype_empirical_q__OMM"], 0.05)
        self.assertGreater(
            first.loc["1", "subtype_empirical_p__Lamina"],
            first.loc["0", "subtype_empirical_p__Lamina"],
        )

    def test_adaptive_guards_use_empirical_fdr_not_fixed_margin(self):
        annotation = pd.DataFrame(
            {
                "label": ["Nuclear Pore", "Nuclear Pore"],
                "top1": ["Nuclear Pore", "Nuclear Pore"],
                "top2": ["Nucleus", "Nucleus"],
                "z1": [2.0, 20.0],
                "z2": [1.9, 1.0],
                "q1": [0.9, 0.001],
                "selected_q": [0.9, 0.001],
                "channel_support": [1, 3],
                "low_conf": [False, False],
                "subtype_empirical_q__Nuclear Pore": [0.01, 0.20],
                "subtype_testable__Nuclear Pore": [True, True],
                "effective_marker_count__Nuclear Pore": [20, 20],
            },
            index=["empirically_supported", "empirically_unsupported"],
        )

        result = _guard_nuclear_pore_overcall(
            annotation,
            guard_mode="adaptive",
            subtype_fdr=0.05,
            min_effective_markers=10,
            min_margin=100.0,
            max_q=0.0,
            min_channel_support=99,
        )

        self.assertEqual(
            result["label"].tolist(),
            ["Nuclear Pore", "Nucleus"],
        )
        self.assertEqual(
            result["nuclear_pore_guarded"].tolist(),
            [False, True],
        )

    def test_omm_guard_requires_subtype_specific_expression_support(self):
        annotation = pd.DataFrame(
            {
                "label": ["OMM", "OMM", "OMM", "ER"],
                "top1": ["OMM", "OMM", "ER", "ER"],
                "top2": ["ER", "ER", "OMM", "OMM"],
                "z1": [6.0, 7.0, 3.1, 5.0],
                "z2": [2.0, 2.0, 3.0, 1.0],
                "q1": [0.001, 0.001, 0.001, 0.001],
                "low_conf": [False, False, False, False],
                "z_mwu__OMM": [2.0, 2.0, 2.0, 2.0],
                "z_proj__OMM": [1.5, -0.5, 1.0, 1.0],
                "z_hyp__OMM": [1.0, 1.0, 1.0, 1.0],
                "z_comb__OMM": [6.0, 7.0, 3.0, 1.0],
                "z_comb__ER": [2.0, 2.0, 3.1, 5.0],
                "z_comb__Cytosol": [1.0, 1.0, 1.0, 0.5],
                "q_comb__OMM": [0.001, 0.001, 0.001, 0.1],
            },
            index=["strong", "discordant", "promoted", "non_omm"],
        )

        result = _guard_omm_overcall(annotation)

        self.assertEqual(
            result["label"].tolist(),
            ["OMM", "Cytosol", "Cytosol", "ER"],
        )
        self.assertEqual(
            result["omm_guarded"].tolist(),
            [False, True, True, False],
        )
        self.assertEqual(result.loc["discordant", "candidate_label"], "OMM")
        self.assertTrue(result.loc["discordant", "low_conf"])
        self.assertAlmostEqual(result.loc["strong", "omm_margin"], 4.0)
        self.assertAlmostEqual(result.loc["promoted", "omm_margin"], -0.1)

    def test_rnalocate_er_augmentation_is_disabled_by_default(self):
        gold = pd.DataFrame(
            {
                "Common_Gene": ["A", "B"],
                "ER": [0.8, 0.4],
                "Cytosol": [0.2, 0.7],
            }
        )

        result = _configure_rnalocate_augmentation(gold)

        self.assertEqual(result["ER"].tolist(), [0.0, 0.0])
        self.assertEqual(result["Cytosol"].tolist(), [0.2, 0.7])
        self.assertEqual(gold["ER"].tolist(), [0.8, 0.4])

    def test_rnalocate_er_augmentation_can_be_enabled(self):
        gold = pd.DataFrame({"Common_Gene": ["A"], "ER": [0.8]})

        result = _configure_rnalocate_augmentation(
            gold,
            augment_er=True,
        )

        self.assertEqual(result["ER"].tolist(), [0.8])

    def test_apex_omm_specificity_filter_keeps_only_raw_specific_rows(self):
        raw = pd.DataFrame(
            {
                "Common_Gene": ["EDGE", "SHARED", "LOW"],
                "Source": ["test"] * 3,
                "Nucleus_log2FC": [0.0, 0.0, 0.0],
                "Cytosol_log2FC": [0.25, 0.26, 0.10],
                "ERM_log2FC": [0.20, 0.30, 0.10],
                "OMM_log2FC": [0.75, 0.75, 0.74],
            }
        )

        filtered, audit = _prepare_apex_gold_reference(
            raw,
            omm_min_log2fc=0.75,
            omm_min_delta=0.50,
        )
        legacy, _ = _prepare_apex_gold_reference(
            raw,
            omm_specificity_filter=False,
            omm_min_log2fc=0.75,
            omm_min_delta=0.50,
        )

        filtered = filtered.set_index("Common_Gene")
        legacy = legacy.set_index("Common_Gene")
        audit = audit.set_index("Common_Gene")
        self.assertTrue(audit.loc["EDGE", "omm_specific_pass"])
        self.assertGreater(filtered.loc["EDGE", "OMM"], 0.0)
        self.assertTrue(audit.loc["SHARED", "omm_shared"])
        self.assertEqual(filtered.loc["SHARED", "OMM"], 0.0)
        self.assertFalse(audit.loc["LOW", "omm_shared"])
        self.assertEqual(filtered.loc["LOW", "OMM"], 0.0)
        pd.testing.assert_series_equal(
            filtered.loc["SHARED"].drop("OMM"),
            legacy.loc["SHARED"].drop("OMM"),
        )

    def test_bundled_apex_omm_filter_regression_counts(self):
        gold, _, audit = _load_default_gold_tables(
            return_omm_audit=True,
        )

        self.assertEqual(int(audit["omm_basal_pass"].sum()), 1158)
        self.assertEqual(int(audit["omm_specific_pass"].sum()), 258)
        self.assertEqual(int(audit["omm_shared"].sum()), 900)
        self.assertEqual(
            int(
                audit.loc[
                    audit["omm_specific_pass"],
                    "Common_Gene",
                ]
                .dropna()
                .astype(str)
                .nunique()
            ),
            254,
        )
        self.assertEqual(int(gold["OMM"].gt(0).sum()), 254)

    def test_specific_marker_sets_are_exclusive(self):
        gold = pd.DataFrame(
            {
                "Nucleus": [1.0, 0.8, 0.1],
                "Lamina": [0.9, 0.2, 1.0],
                "ER": [0.0, 0.0, 0.0],
            },
            index=["shared", "nucleus", "lamina"],
        )

        sets, weights, regions = build_specific_marker_sets(
            gold,
            target=10,
            min_weight=0.05,
            min_margin=0.05,
        )

        memberships = [
            region
            for region in regions
            for gene in sets[region]
            if gene == "shared"
        ]
        self.assertEqual(memberships, ["Nucleus"])
        self.assertEqual(sets["Lamina"], ["lamina"])
        self.assertGreater(weights["Nucleus"]["shared"], 0)

    def test_robust_gene_stats_does_not_explode_when_mad_is_zero(self):
        matrix = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        )

        result = robust_gene_stats(matrix)

        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[:, 0], np.zeros(3))
        self.assertLess(np.abs(result[:, 1]).max(), 10)

    def test_bool_parser_handles_h5ad_style_string_categories(self):
        values = pd.Series(
            pd.Categorical(["False", "True", None], categories=["False", "True"])
        )

        result = _coerce_bool_series(values, default=True)

        self.assertEqual(result.tolist(), [False, True, True])

    def test_default_policy_keeps_candidates_but_filters_final_label(self):
        annotation = pd.DataFrame(
            {
                "label": ["Lamina", "OMM"],
                "low_conf": [True, True],
                "z1": [0.58, 0.02],
                "q1": [0.75, 0.75],
                "channel_support": [2, 2],
                "geometry_consistent": [False, True],
            }
        )

        nuclear = _apply_low_confidence_policy(
            annotation.iloc[[0]],
            compartment="Nuclear",
        )
        cytoplasmic = _apply_low_confidence_policy(
            annotation.iloc[[1]],
            compartment="Cytoplasmic",
        )

        self.assertEqual(nuclear["label"].tolist(), ["Nucleus"])
        self.assertEqual(nuclear["candidate_label"].tolist(), ["Lamina"])
        self.assertEqual(
            nuclear["conservative_label"].tolist(),
            ["Nucleus"],
        )
        self.assertEqual(nuclear["confidence_level"].tolist(), ["Low"])
        self.assertEqual(cytoplasmic["label"].tolist(), ["Cytosol"])
        self.assertEqual(cytoplasmic["candidate_label"].tolist(), ["OMM"])
        self.assertEqual(
            cytoplasmic["conservative_label"].tolist(),
            ["Cytosol"],
        )

    def test_mixed_policy_application_preserves_new_confidence_columns(self):
        annotation = pd.DataFrame(
            {
                "label": ["Lamina", "OMM"],
                "low_conf": [True, True],
                "z1": [0.58, 0.02],
                "q1": [0.75, 0.75],
                "channel_support": [2, 2],
                "geometry_consistent": [False, True],
            },
            index=["0", "1"],
        )

        result = _apply_low_confidence_policy_by_candidate(annotation)

        self.assertIn("candidate_label", result.columns)
        self.assertIn("conservative_label", result.columns)
        self.assertIn("confidence_level", result.columns)
        self.assertEqual(result.loc["0", "label"], "Nucleus")
        self.assertEqual(result.loc["1", "label"], "Cytosol")

    def test_supported_policy_keeps_geometry_consistent_medium_subtype(self):
        annotation = pd.DataFrame(
            {
                "label": ["Nucleolus"],
                "low_conf": [True],
                "selected_z": [1.5],
                "selected_q": [0.2],
                "channel_support": [2],
                "geometry_consistent": [True],
            }
        )

        result = _apply_low_confidence_policy(
            annotation,
            compartment="Nuclear",
        )

        self.assertEqual(result["confidence_level"].tolist(), ["Medium"])
        self.assertEqual(result["label"].tolist(), ["Nucleolus"])

    def test_keep_low_confidence_policy_preserves_omm_candidate(self):
        annotation = pd.DataFrame(
            {
                "label": ["OMM"],
                "low_conf": [True],
                "z1": [0.016],
                "q1": [0.75],
                "channel_support": [1],
            }
        )

        result = _apply_low_confidence_policy(
            annotation,
            compartment="Cytoplasmic",
            policy="keep",
        )

        self.assertEqual(result["label"].tolist(), ["OMM"])
        self.assertEqual(result["candidate_label"].tolist(), ["OMM"])

    def test_adaptive_low_confidence_policy_preserves_supported_subtype(self):
        annotation = pd.DataFrame(
            {
                "label": ["Nucleolus", "Lamina"],
                "low_conf": [True, True],
                "z1": [1.4, 0.8],
                "q1": [0.2, 0.5],
                "channel_support": [2, 2],
            }
        )

        result = _apply_low_confidence_policy(
            annotation,
            compartment="Nuclear",
            policy="adaptive",
            relaxed_z=1.25,
            relaxed_q=0.30,
        )

        self.assertEqual(result["label"].tolist(), ["Nucleolus", "Nucleus"])
        self.assertEqual(
            result["candidate_label"].tolist(),
            ["Nucleolus", "Lamina"],
        )

    def test_confidence_uses_selected_label_score_after_top2_switch(self):
        annotation = pd.DataFrame(
            {
                "label": ["Nucleolus"],
                "low_conf": [True],
                "z1": [0.2],
                "q1": [0.9],
                "selected_z": [1.8],
                "selected_q": [0.1],
                "channel_support": [2],
                "geometry_consistent": [True],
            }
        )

        result = _apply_low_confidence_policy(
            annotation,
            compartment="Nuclear",
        )

        self.assertEqual(result["confidence_level"].tolist(), ["Medium"])
        self.assertEqual(result["label"].tolist(), ["Nucleolus"])

    def test_annotation_columns_can_be_reinitialized_from_categorical(self):
        adata = ad.AnnData(np.zeros((2, 1), dtype=np.float32))
        adata.obs["subcellular_annotation"] = pd.Categorical(["ER", "Nucleus"])

        _ensure_obs_columns_initialized(
            adata,
            "subcellular_annotation",
            ["top1"],
        )
        adata.obs.loc[:, "subcellular_annotation"] = ["Cytosol", "Nucleolus"]

        self.assertEqual(
            adata.obs["subcellular_annotation"].tolist(),
            ["Cytosol", "Nucleolus"],
        )

    def test_intermediate_anndata_save_is_atomic_and_readable(self):
        adata = ad.AnnData(np.zeros((2, 1), dtype=np.float32))
        adata.obs["mixed_object"] = pd.Series(
            ["ER", pd.NA],
            index=adata.obs_names,
            dtype="object",
        )
        original = adata.obs["mixed_object"].copy()

        with tempfile.TemporaryDirectory() as out_dir, _local_config():
            _save_adata(adata, out_dir, "annotation_checkpoint")
            path = os.path.join(
                out_dir,
                "intermediate",
                "annotation_checkpoint.h5ad",
            )
            self.assertTrue(os.path.exists(path))
            reloaded = ad.read_h5ad(path)

        self.assertEqual(reloaded.shape, adata.shape)
        self.assertIn("mixed_object", reloaded.obs.columns)
        pd.testing.assert_series_equal(
            adata.obs["mixed_object"],
            original,
        )

    def test_resume_build_uses_aligned_final_table_in_lite_mode(self):
        adata1 = ad.AnnData(
            np.zeros((2, 1), dtype=np.float32),
            obs=pd.DataFrame(
                {
                    "cell_id": ["c1", "c2"],
                    "metaspot": ["m1", "m2"],
                    "log_n": np.log1p([2, 1]),
                },
                index=["m1", "m2"],
            ),
        )
        adata2 = ad.AnnData(
            np.zeros((2, 1), dtype=np.float32),
            obs=pd.DataFrame(index=["d1", "d2"]),
        )
        final_data = pd.DataFrame(
            {
                "cell_id": ["c1", "c1", "c2"],
                "metaspot": ["m1", "m1", "m2"],
                "meta-domain": ["d1", "d1", "d2"],
                "subcellular_annotation": ["ER", "ER", "Nucleus"],
            }
        )

        with tempfile.TemporaryDirectory() as out_dir, _local_config():
            intermediate = os.path.join(out_dir, "intermediate")
            os.makedirs(intermediate, exist_ok=True)
            adata1.write_h5ad(
                os.path.join(intermediate, "module5_adata1.h5ad")
            )
            adata2.write_h5ad(
                os.path.join(intermediate, "module6_adata2.h5ad")
            )
            final_data.to_parquet(
                os.path.join(out_dir, "final_data.parquet"),
                index=False,
            )

            resumed = _resume_build_outputs(out_dir)

        self.assertIsNotNone(resumed)
        resumed_adata1, resumed_adata2, resumed_final = resumed
        self.assertEqual(resumed_adata1.n_obs, 2)
        self.assertEqual(resumed_adata2.n_obs, 2)
        self.assertEqual(len(resumed_final), 3)

    def test_object_annotation_is_not_overwritten_by_leiden_majority(self):
        adata = ad.AnnData(np.zeros((3, 1), dtype=np.float32))
        adata.obs["leiden"] = pd.Categorical(["0", "0", "0"])
        adata.obs["subcellular_annotation"] = pd.Categorical(
            ["ER", "ER", "Nucleolus"]
        )

        _refresh_subcellular_group_columns(adata)

        self.assertEqual(
            adata.obs["cluster-subcellular"].tolist(),
            ["0:ER", "0:ER", "0:Nucleolus"],
        )
        self.assertEqual(
            adata.obs["leiden-dominant-subcellular"].tolist(),
            ["ER", "ER", "ER"],
        )

    def test_n_feat_hdb_limits_features_by_local_variance(self):
        matrix = np.array(
            [
                [0.0, 0.0, 0.0, 4.0],
                [1.0, 0.1, 0.0, 3.0],
                [2.0, 0.2, 0.0, 2.0],
                [3.0, 0.3, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        selected = _select_top_variance_features(matrix, 2)

        self.assertEqual(selected.shape, (4, 2))
        np.testing.assert_array_equal(selected[:, 0], matrix[:, 0])
        np.testing.assert_array_equal(selected[:, 1], matrix[:, 3])

    def test_spatial_connectivity_splits_disconnected_islands(self):
        coordinates = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
            dtype=np.float32,
        )
        labels = np.zeros(4, dtype=np.int32)

        connected = _enforce_spatially_connected_labels(
            coordinates,
            labels,
            connectivity_k=2,
            radius_quantile=0.9,
            radius_factor=1.1,
            smoothing_iterations=0,
            min_component_size=1,
        )

        self.assertEqual(len(np.unique(connected)), 2)
        self.assertEqual(connected[0], connected[1])
        self.assertEqual(connected[2], connected[3])
        self.assertNotEqual(connected[0], connected[2])

    def test_spatial_ward_quiets_known_disconnected_graph_warning(self):
        coordinates = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
        features = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)

        with warnings.catch_warnings(record=True) as caught:
            connected = _enforce_spatially_connected_labels(
                coordinates,
                labels,
                feature_matrix=features,
                target_n_clusters=2,
                connectivity_k=1,
                radius_quantile=0.9,
                radius_factor=1.1,
                smoothing_iterations=0,
                min_component_size=1,
            )

        disconnected_notices = [
            warning
            for warning in caught
            if "number of connected components of the connectivity matrix"
            in str(warning.message)
        ]
        self.assertEqual(disconnected_notices, [])
        self.assertEqual(connected[0], connected[1])
        self.assertEqual(connected[2], connected[3])
        self.assertNotEqual(connected[0], connected[2])

    def test_spatial_ward_early_stop_matches_full_tree_membership(self):
        rng = np.random.default_rng(91)
        coordinates = np.round(
            rng.uniform(0.0, 50.0, size=(600, 2)),
            2,
        ).astype(np.float32)
        features = rng.normal(size=(600, 12)).astype(np.float32)
        labels = np.zeros(600, dtype=np.int32)
        real_clustering = pipeline_module.AgglomerativeClustering

        def _full_tree_clustering(*args, **kwargs):
            kwargs["compute_full_tree"] = True
            return real_clustering(*args, **kwargs)

        with mock.patch.object(
            pipeline_module,
            "AgglomerativeClustering",
            side_effect=_full_tree_clustering,
        ):
            full_tree = _enforce_spatially_connected_labels(
                coordinates,
                labels,
                feature_matrix=features,
                target_n_clusters=30,
                connectivity_k=8,
                min_component_size=5,
            )
        early_stop = _enforce_spatially_connected_labels(
            coordinates,
            labels,
            feature_matrix=features,
            target_n_clusters=30,
            connectivity_k=8,
            min_component_size=5,
        )

        np.testing.assert_array_equal(early_stop, full_tree)

    def test_spatial_connectivity_merges_small_component_to_neighbor(self):
        coordinates = np.column_stack(
            (np.arange(6, dtype=np.float32), np.zeros(6, dtype=np.float32))
        )
        labels = np.asarray([0, 0, 1, 0, 0, 0], dtype=np.int32)

        connected = _enforce_spatially_connected_labels(
            coordinates,
            labels,
            connectivity_k=2,
            radius_quantile=0.9,
            radius_factor=1.1,
            smoothing_iterations=0,
            min_component_size=2,
        )

        self.assertIn(connected[2], {connected[1], connected[3]})
        self.assertLessEqual(len(np.unique(connected)), 2)

    def test_target_metaspot_count_uses_unique_sites_and_bounds(self):
        coordinates = np.repeat(
            np.column_stack(
                (np.arange(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32))
            ),
            repeats=3,
            axis=0,
        )

        self.assertEqual(_target_metaspot_count(coordinates, 100, 2, 100), 10)
        self.assertEqual(_target_metaspot_count(coordinates, 500, 5, 100), 5)
        self.assertEqual(_target_metaspot_count(coordinates, 5, 2, 20), 20)

    def test_segmentation_audit_vectorizes_requested_metaspot_counts(self):
        frame = pd.DataFrame(
            {
                "cell_id": ["A"] * 5 + ["B"] * 3 + ["C"] * 8,
                "metaspot": (
                    ["A:0", "A:0", "A:0", "A:1", "A:1"]
                    + ["B:0"] * 3
                    + ["C:0"] * 4
                    + ["C:1"] * 4
                ),
                "x_location": [
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    np.nan,
                    np.nan,
                    np.inf,
                    np.nan,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                ],
                "y_location": np.zeros(16, dtype=np.float32),
                "in_nucleus": [
                    True,
                    False,
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                ],
            }
        )
        original_eq = pd.Series.eq

        def reject_full_cell_scan(series, other, *args, **kwargs):
            if series.name == "cell_id" and len(series) == len(frame):
                raise AssertionError("audit rescanned the full cell_id column")
            return original_eq(series, other, *args, **kwargs)

        with mock.patch.object(pd.Series, "eq", new=reject_full_cell_scan):
            audit = _metaspot_segmentation_audit(
                frame,
                target_sites_per_metaspot=2,
                min_metaspots_per_cell=2,
                max_metaspots_per_cell=3,
                partition_by_nucleus=True,
            )

        expected = {
            cell_id: _target_metaspot_count(
                group[["x_location", "y_location"]].to_numpy(
                    np.float32,
                    copy=False,
                ),
                target_sites_per_metaspot=2,
                min_metaspots_per_cell=2,
                max_metaspots_per_cell=3,
            )
            for cell_id, group in frame.groupby("cell_id", sort=False)
        }
        actual = audit.set_index("cell_id")["requested_metaspots"].to_dict()

        self.assertEqual(actual, expected)
        self.assertEqual(actual, {"A": 2, "B": 1, "C": 3})
        self.assertEqual(
            audit.set_index("cell_id")["n_unique_sites"].to_dict(),
            {"A": 4, "B": 2, "C": 8},
        )

    def test_partition_target_allocation_preserves_cell_budget(self):
        nucleus = np.column_stack(
            (
                np.arange(30, dtype=np.float32),
                np.zeros(30, dtype=np.float32),
            )
        )
        cytoplasm = np.column_stack(
            (
                np.arange(70, dtype=np.float32),
                np.ones(70, dtype=np.float32),
            )
        )

        allocated = _allocate_partition_target_counts(
            [nucleus, cytoplasm],
            target_sites_per_metaspot=10,
            min_metaspots_per_cell=1,
            max_metaspots_per_cell=8,
        )

        self.assertEqual(allocated, [2, 6])
        self.assertEqual(sum(allocated), 8)

    def test_module3_partition_prevents_nuclear_mixed_metaspots(self):
        n_per_side = 40
        frame = pd.DataFrame(
            {
                "cell_id": ["Cell1"] * (2 * n_per_side),
                "feature_name": ["GeneA", "GeneB"] * n_per_side,
                "x_location": np.r_[
                    np.linspace(0, 19, n_per_side),
                    np.linspace(20, 39, n_per_side),
                ].astype(np.float32),
                "y_location": np.zeros(2 * n_per_side, dtype=np.float32),
                "in_nucleus": np.r_[
                    np.ones(n_per_side, dtype=bool),
                    np.zeros(n_per_side, dtype=bool),
                ],
                "r_signed": np.r_[
                    -np.ones(n_per_side, dtype=np.float32),
                    np.ones(n_per_side, dtype=np.float32),
                ],
                "d_cell_norm": np.full(
                    2 * n_per_side,
                    0.5,
                    dtype=np.float32,
                ),
                "d_nuc_norm": np.full(
                    2 * n_per_side,
                    0.5,
                    dtype=np.float32,
                ),
            }
        )
        clustered = per_cell_clustering(
            frame,
            ["r_signed", "d_cell_norm", "d_nuc_norm"],
            hdbscan_min_cluster_size=5,
            sequential_cells=10,
            enforce_spatial_connectivity=True,
            partition_by_nucleus=True,
            spatial_connectivity_k=4,
            spatial_min_component_size=2,
            target_sites_per_metaspot=5,
            min_metaspots_per_cell=2,
            max_metaspots_per_cell=8,
        )
        mixed = (
            clustered.groupby("metaspot", observed=True)["in_nucleus"]
            .nunique()
            .gt(1)
        )
        self.assertFalse(mixed.any())
        self.assertLessEqual(clustered["metaspot"].nunique(), 8)

        audit = _metaspot_segmentation_audit(
            clustered,
            target_sites_per_metaspot=5,
            min_metaspots_per_cell=2,
            max_metaspots_per_cell=8,
            partition_by_nucleus=True,
        )
        self.assertEqual(
            int(audit["nucleus_mixed_metaspots"].sum()),
            0,
        )
        self.assertEqual(
            float(audit["nucleus_mixed_transcript_fraction"].max()),
            0.0,
        )

    def test_legacy_metaspot_columns_are_migrated(self):
        legacy = pd.DataFrame(
            {
                "cell_sub": ["cell:0"],
                "cluster_in_cell": [0],
            }
        )

        migrated = _migrate_legacy_metaspot_columns(legacy)

        self.assertEqual(migrated.columns.tolist(), ["metaspot", "metaspot_in_cell"])

    def test_module4_cache_alignment_rejects_stale_metaspot_numbering(self):
        point_df = pd.DataFrame(
            {
                "metaspot": ["cell:0"] * 3 + ["cell:1"] * 3,
                "feature_name": ["A", "A", "B", "A", "B", "B"],
                "r_signed": [-0.8, -0.7, -0.6, 0.5, 0.6, 0.7],
                "d_cell_norm": [0.2] * 6,
                "d_nuc_norm": [0.8] * 6,
                "x_location": np.arange(6, dtype=np.float32),
                "y_location": np.zeros(6, dtype=np.float32),
            }
        )
        cached = build_adata1(point_df)

        aligned, reason = _validate_adata1_point_df_alignment(cached, point_df)
        self.assertTrue(aligned, reason)

        renumbered = point_df.copy()
        renumbered.loc[0, "metaspot"] = "cell:1"
        aligned, reason = _validate_adata1_point_df_alignment(
            cached,
            renumbered,
        )
        self.assertFalse(aligned)
        self.assertIn("transcript counts differ", reason)

        changed_geometry = point_df.copy()
        changed_geometry.loc[
            changed_geometry["metaspot"].eq("cell:0"),
            "r_signed",
        ] = 0.9
        aligned, reason = _validate_adata1_point_df_alignment(
            cached,
            changed_geometry,
        )
        self.assertFalse(aligned)
        self.assertIn("r_mean differs", reason)

    def test_module4_vectorized_statistics_match_reference_formulas(self):
        point_df = pd.DataFrame(
            {
                "metaspot": ["cell:0"] * 5 + ["cell:1"] * 4,
                "feature_name": ["A", "B", "A", "C", "B"] * 1
                + ["A", "C", "C", "B"],
                "r_signed": [-0.8, np.nan, -0.2, 0.1, 0.5, 0.2, 0.4, 0.7, 0.9],
                "d_cell_norm": np.linspace(0.1, 0.9, 9),
                "d_nuc_norm": np.linspace(0.9, 0.1, 9),
                "x_location": [0.0, np.nan, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
                "y_location": [0.0, 1.0, 0.0, 2.0, 1.0, 5.0, 5.5, np.nan, 7.0],
            }
        )

        adata1 = build_adata1(point_df)

        for metaspot, group in point_df.groupby("metaspot", sort=True):
            r_values = group["r_signed"].to_numpy(np.float32)
            expected_quantiles = np.nanpercentile(
                r_values,
                [10, 50, 90],
            )
            np.testing.assert_allclose(
                adata1.obs.loc[
                    metaspot,
                    ["r_p10", "r_p50", "r_p90"],
                ].to_numpy(dtype=float),
                expected_quantiles,
                rtol=0.0,
                atol=1e-6,
            )
            self.assertAlmostEqual(
                float(adata1.obs.loc[metaspot, "r_mean"]),
                float(np.nanmean(r_values)),
                places=6,
            )
            self.assertAlmostEqual(
                float(adata1.obs.loc[metaspot, "r_std"]),
                float(np.nanstd(r_values)),
                places=6,
            )

            x_values = group["x_location"].to_numpy(np.float32)
            y_values = group["y_location"].to_numpy(np.float32)
            centered_x = x_values - np.nanmean(x_values)
            centered_y = y_values - np.nanmean(y_values)
            finite = np.isfinite(centered_x) & np.isfinite(centered_y)
            denominator = max(int(finite.sum()) - 1, 1)
            var_x = float(
                np.square(centered_x[finite]).sum() / denominator
            )
            var_y = float(
                np.square(centered_y[finite]).sum() / denominator
            )
            covariance = float(
                (centered_x[finite] * centered_y[finite]).sum()
                / denominator
            )
            trace = max(var_x + var_y, 0.0)
            discriminant = max(
                trace * trace
                - 4.0 * (var_x * var_y - covariance * covariance),
                0.0,
            )
            root = np.sqrt(discriminant)
            lambda_1 = max(0.5 * (trace + root), 1e-8)
            lambda_2 = max(0.5 * (trace - root), 1e-8)
            np.testing.assert_allclose(
                float(adata1.obs.loc[metaspot, "elongation"]),
                lambda_1 / lambda_2,
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                float(adata1.obs.loc[metaspot, "spread"]),
                np.sqrt(trace),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_resume_fingerprint_tracks_applied_tuning_parameters(self):
        cfg = {
            "module1": {"use_input_nucleus_flags": False},
            "module2": {"idf_mode": "log"},
            "module3": {
                "hdbscan_max_points": 99,
                "sequential_cells": 7,
                "enforce_spatial_connectivity": True,
                "partition_by_nucleus": True,
                "target_sites_per_metaspot": 100,
            },
            "module5": {"assign_chunk_size": 123, "mbk_batch_size": 17},
            "module6": {"pool_normalize": True},
            "module8": {"amp": False},
            "annotation": {
                "unit": "metadomain",
                "decision_mode": "independent",
                "gold_column_normalize": False,
                "guard_mode": "adaptive",
                "subtype_fdr": 0.05,
                "min_effective_markers": 10,
                "null_permutations": 500,
                "null_seed": 0,
                "apex_omm_specificity_filter": True,
                "apex_omm_min_log2fc": 0.75,
                "apex_omm_min_delta": 0.50,
                "independent_domain_compartment_feasibility": True,
                "independent_omm_competitor_margin": 0.0,
                "omm_metaspot_refinement": True,
                "omm_metaspot_ambiguous_policy": "competitor",
            },
        }

        self.assertEqual(module_config_subset(cfg, "m2")["module2.idf_mode"], "log")
        self.assertFalse(
            module_config_subset(cfg, "m1")[
                "module1.use_input_nucleus_flags"
            ]
        )
        self.assertEqual(
            module_config_subset(cfg, "m3")["module3.hdbscan_max_points"],
            99,
        )
        self.assertTrue(
            module_config_subset(cfg, "m3")[
                "module3.enforce_spatial_connectivity"
            ]
        )
        self.assertTrue(
            module_config_subset(cfg, "m3")[
                "module3.partition_by_nucleus"
            ]
        )
        self.assertEqual(
            module_config_subset(cfg, "m3")[
                "module3.target_sites_per_metaspot"
            ],
            100,
        )
        self.assertEqual(
            module_config_subset(cfg, "m5")["module5.assign_chunk_size"],
            123,
        )
        self.assertFalse(module_config_subset(cfg, "m8")["module8.amp"])
        self.assertTrue(
            module_config_subset(cfg, "m6")["module6.pool_normalize"]
        )
        annotate_cfg = stage_config_subset(cfg, "annotate")
        self.assertEqual(annotate_cfg["annotation.unit"], "metadomain")
        self.assertEqual(
            annotate_cfg["annotation.decision_mode"],
            "independent",
        )
        self.assertFalse(
            annotate_cfg["annotation.gold_column_normalize"]
        )
        self.assertEqual(annotate_cfg["annotation.guard_mode"], "adaptive")
        self.assertEqual(annotate_cfg["annotation.null_permutations"], 500)
        self.assertTrue(
            annotate_cfg["annotation.apex_omm_specificity_filter"]
        )
        self.assertEqual(
            annotate_cfg["annotation.apex_omm_min_delta"],
            0.50,
        )
        self.assertEqual(
            annotate_cfg["annotation.independent_omm_competitor_margin"],
            0.0,
        )
        self.assertTrue(
            annotate_cfg[
                "annotation.independent_domain_compartment_feasibility"
            ]
        )
        self.assertTrue(
            annotate_cfg["annotation.omm_metaspot_refinement"]
        )
        self.assertEqual(
            annotate_cfg["annotation.omm_metaspot_ambiguous_policy"],
            "competitor",
        )

    def test_streaming_model_preserves_local_expression_without_global_columns(self):
        frame = _synthetic_transcripts()
        with _local_config():
            output, _, model = build_composition_embeddings(
                frame.copy(),
                frame,
                frame["cell_id"].unique().tolist(),
            )

        self.assertIsInstance(model, _LocalCompositionModel)
        self.assertFalse(any(c.startswith("comp_emb") for c in output.columns))

        embedding, density = model.transform_cell(frame[frame["cell_id"] == "0"])
        separation = np.linalg.norm(
            embedding[:15].mean(axis=0) - embedding[25:].mean(axis=0)
        )
        self.assertGreater(separation, 0.1)
        self.assertTrue(np.isfinite(density).all())

    def test_module3_consumes_streaming_expression_model(self):
        frame = _synthetic_transcripts()
        real_import = __import__

        def reject_hdbscan(name, *args, **kwargs):
            if name == "hdbscan":
                raise AssertionError(
                    "canonical spatial clustering must not import hdbscan"
                )
            return real_import(name, *args, **kwargs)

        with _local_config(), mock.patch(
            "builtins.__import__",
            side_effect=reject_hdbscan,
        ):
            point_df, gmm_cols, model = build_composition_embeddings(
                frame.copy(),
                frame,
                frame["cell_id"].unique().tolist(),
            )
            clustered = per_cell_clustering(
                point_df,
                gmm_cols,
                composition_model=model,
                hdbscan_min_cluster_size=5,
                target_sub_size=10,
                kmeans_min_k=2,
                kmeans_max_k=4,
                use_k_cap=True,
                merge_min_size=5,
                hdbscan_max_points=1,
                sequential_cells=10,
                enforce_spatial_connectivity=True,
                spatial_min_component_size=2,
                spatial_compactness=1.0,
                target_sites_per_metaspot=10,
                min_metaspots_per_cell=2,
                max_metaspots_per_cell=4,
            )

        self.assertIn("metaspot_in_cell", clustered.columns)
        self.assertIn("metaspot", clustered.columns)
        self.assertNotIn("cluster_in_cell", clustered.columns)
        self.assertNotIn("cell_sub", clustered.columns)
        self.assertGreaterEqual(
            clustered.loc[
                clustered["cell_id"] == "0",
                "metaspot_in_cell",
            ].nunique(),
            2,
        )
        self.assertFalse(any(c.startswith("comp_emb") for c in clustered.columns))

        adata1 = build_adata1(clustered)
        self.assertEqual(adata1.n_obs, clustered["metaspot"].nunique())
        self.assertNotIn("cell_sub", adata1.obs.columns)

    def test_module3_worker_does_not_capture_full_point_table(self):
        frame = _synthetic_transcripts()
        captured = {}

        class CaptureParallel:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __call__(self, tasks):
                worker, args, _ = next(iter(tasks))
                captured["freevars"] = set(worker.__code__.co_freevars)
                cell_id, group = args[0]
                return [
                    (
                        cell_id,
                        pd.Series(
                            np.zeros(len(group), dtype=np.int32),
                            index=group.index,
                        ),
                    )
                ]

        with _local_config(), mock.patch(
            "joblib.Parallel",
            CaptureParallel,
        ):
            per_cell_clustering(
                frame,
                ["r_signed", "d_cell_norm", "d_nuc_norm"],
                sequential_cells=0,
                small_backend_cells=0,
                parallel_chunk_cells=16,
                parallel_n_jobs=2,
                enforce_spatial_connectivity=True,
                target_sites_per_metaspot=10,
                min_metaspots_per_cell=2,
                max_metaspots_per_cell=4,
            )

        self.assertNotIn("point_df", captured["freevars"])


if __name__ == "__main__":
    unittest.main()
