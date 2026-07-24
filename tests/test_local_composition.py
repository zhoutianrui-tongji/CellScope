import os
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from cellscope.pipeline import (
    _LocalCompositionModel,
    _coerce_bool_series,
    _ensure_obs_columns_initialized,
    _apply_low_confidence_policy,
    _apply_low_confidence_policy_by_candidate,
    _apply_independent_evidence_policy,
    _choose_annotation_neighbors_key,
    _configure_annotation_unit,
    _configure_rnalocate_augmentation,
    _prepare_apex_gold_reference,
    _load_default_gold_tables,
    _compose_omm_metaspot_labels,
    _apply_local_compartment_guard,
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
    _enforce_spatially_connected_labels,
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

    def test_resume_fingerprint_tracks_applied_tuning_parameters(self):
        cfg = {
            "module2": {"idf_mode": "log"},
            "module3": {
                "hdbscan_max_points": 99,
                "sequential_cells": 7,
                "enforce_spatial_connectivity": True,
                "target_sites_per_metaspot": 100,
            },
            "module5": {"assign_chunk_size": 123, "mbk_batch_size": 17},
            "module6": {"preserve_raw_counts": True},
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
                "independent_omm_competitor_margin": 0.0,
                "omm_metaspot_refinement": True,
                "omm_metaspot_ambiguous_policy": "competitor",
            },
        }

        self.assertEqual(module_config_subset(cfg, "m2")["module2.idf_mode"], "log")
        self.assertEqual(
            module_config_subset(cfg, "m3")["module3.hdbscan_max_points"],
            99,
        )
        self.assertTrue(
            module_config_subset(cfg, "m3")[
                "module3.enforce_spatial_connectivity"
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
            module_config_subset(cfg, "m6")[
                "module6.preserve_raw_counts"
            ]
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
        with _local_config():
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


if __name__ == "__main__":
    unittest.main()
