import os
import sys
import tempfile
import unittest
import warnings
from unittest import mock
from contextlib import contextmanager

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from cellscope.config import (
    dgi_unit_from_config,
    fingerprint_module,
    fingerprint_stage,
    resolve_dgi_configuration,
)
from cellscope.pipeline import (
    METADOMAIN_DGI_KEY,
    METASPOT_DMGI_EXPRESSION_KEY,
    METASPOT_DMGI_SPATIAL_KEY,
    METASPOT_DGI_KEY,
    METASPOT_GRAPH_KEY,
    _build_final_annotation_proportions,
    _build_groupwise_corruption_plan,
    _build_ground_truth_nuclear_cytoplasmic_proportions,
    _build_module_execution_order,
    _build_metadomain_annotation_quality_tables,
    _build_metaspot_annotation_result_tables,
    _attach_annotation_provenance_columns,
    _choose_annotation_neighbors_key,
    _metadomain_feature_matrix,
    _remove_deprecated_output_artifacts,
    _remove_inactive_method_artifacts,
    _save_rapx_metadomain_evidence_audit_tables,
    _save_transcript_annotation_proportion_diagnostics,
    _draw_groupwise_corruption_index,
    _sample_local_contrastive_pairs,
    _stratified_metaspot_sample,
    _write_output_guide,
    build_adata1,
    build_adata2_from_adata1_and_pointdf,
    build_metaspot_graph,
    run_module8_dgi,
    run_module5_simple,
    run_rapx,
)
from cellscope.consensus_clustering import run_stability_consensus_leiden
from cellscope.multiplex_dgi import train_cell_aware_dmgi
from cellscope.sampled_multiplex_dgi import train_sampled_cell_aware_dmgi


@contextmanager
def _config_file(config):
    previous = os.environ.get("CELLSCOPE_CONFIG")
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".yaml",
        delete=False,
    ) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    os.environ["CELLSCOPE_CONFIG"] = path
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CELLSCOPE_CONFIG", None)
        else:
            os.environ["CELLSCOPE_CONFIG"] = previous
        os.unlink(path)


def _point_table():
    rows = []
    genes = ["A", "B", "C", "D"]
    for cell_index, cell in enumerate(["c1", "c2"]):
        for spot_index in range(3):
            metaspot = f"{cell}:{spot_index}"
            for transcript_index in range(3):
                rows.append(
                    {
                        "metaspot": metaspot,
                        "cell_id": cell,
                        "feature_name": genes[
                            (spot_index + transcript_index + cell_index)
                            % len(genes)
                        ],
                        "x_location": float(
                            100 * cell_index + 3 * spot_index + transcript_index
                        ),
                        "y_location": float(cell_index),
                        "r_signed": float(spot_index - 1),
                        "d_cell_norm": 0.2 + 0.1 * spot_index,
                        "d_nuc_norm": 0.7 - 0.1 * spot_index,
                    }
                )
    return pd.DataFrame(rows)


def _domain_obs():
    return pd.DataFrame(
        {
            "meta-domain": pd.Categorical(["d1", "d1", "d2"]),
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


class MetaSpotDGIPipelineTests(unittest.TestCase):
    def test_ca_dmgi_public_name_and_framework_aliases(self):
        aliases = ("ca_dmgi", "ca-dmgi", "ch" + "_dmgi", "ch" + "-dmgi")
        fingerprints = set()
        for alias in aliases:
            with self.subTest(alias=alias):
                resolved = resolve_dgi_configuration(
                    {
                        "module8": {
                            "dgi_unit": "metaspot",
                            "dgi_framework": alias,
                        }
                    }
                )
                self.assertEqual(
                    resolved["module8"]["dgi_framework"],
                    "cell_aware_dmgi",
                )
                fingerprints.add(fingerprint_module(resolved, "m8"))
        self.assertEqual(len(fingerprints), 1)

    def test_public_files_do_not_reintroduce_legacy_dmgi_name(self):
        project_root = os.path.dirname(os.path.dirname(__file__))
        forbidden = ("ch" + "-dmgi", "ch" + "_dmgi")
        checked_suffixes = {".py", ".md", ".yaml", ".yml"}
        offenders = []
        for relative in ("cellscope", "config", "docs", "README.md", "CHANGELOG.md"):
            target = os.path.join(project_root, relative)
            paths = [target]
            if os.path.isdir(target):
                paths = []
                for directory, dirs, names in os.walk(target):
                    dirs[:] = [name for name in dirs if name != "__pycache__"]
                    paths.extend(os.path.join(directory, name) for name in names)
            for path in paths:
                if os.path.splitext(path)[1].lower() not in checked_suffixes:
                    continue
                with open(path, encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        lowered = line.lower()
                        if (
                            any(token in lowered for token in forbidden)
                            and "legacy compatibility only" not in lowered
                        ):
                            offenders.append(f"{path}:{line_number}")
        self.assertEqual(offenders, [])

    def test_single_dgi_unit_switch_resolves_metaspot_dependencies(self):
        resolved = resolve_dgi_configuration(
            {
                "module5": {"cluster_rep": "auto"},
                "module6": {"pool_reps": "auto"},
                "module7": {"graph_unit": "auto"},
                "module8": {"dgi_unit": "metaspot"},
            }
        )

        self.assertEqual(dgi_unit_from_config(resolved), "metaspot")
        self.assertEqual(
            resolved["module5"]["cluster_rep"],
            METASPOT_DGI_KEY,
        )
        self.assertEqual(
            resolved["module6"]["pool_reps"],
            [METASPOT_DGI_KEY],
        )
        self.assertEqual(resolved["module7"]["graph_unit"], "metaspot")
        self.assertEqual(resolved["module8"]["dgi_encoder"], "residual")
        self.assertEqual(resolved["module8"]["dgi_framework"], "standard_dgi")
        self.assertEqual(
            resolved["module5"]["clustering_method"],
            "minibatch_kmeans",
        )
        self.assertEqual(resolved["module8"]["dgi_layers"], 2)
        self.assertEqual(resolved["module8"]["dgi_hidden"], 64)
        self.assertEqual(resolved["module8"]["dgi_out"], 32)
        self.assertAlmostEqual(resolved["module8"]["dgi_dropout"], 0.1)
        self.assertAlmostEqual(
            resolved["module8"]["dgi_residual_alpha"],
            0.1,
        )
        self.assertTrue(resolved["module8"]["dgi_layer_norm"])
        self.assertEqual(resolved["module8"]["dgi_device"], "auto")
        self.assertEqual(resolved["module8"]["dgi_cpu_threads"], 0)
        self.assertEqual(resolved["module8"]["dgi_sparse_layout"], "auto")
        self.assertTrue(resolved["module8"]["dgi_tf32"])
        self.assertTrue(
            resolved["module8"]["dmgi_fused_corruption_pass"]
        )
        self.assertEqual(
            resolved["module8"]["dgi_corruption_mode"],
            "auto",
        )
        self.assertAlmostEqual(
            resolved["module8"]["dgi_local_loss_weight"],
            0.2,
        )
        self.assertAlmostEqual(
            resolved["module8"]["dgi_local_temperature"],
            0.2,
        )
        self.assertEqual(resolved["module8"]["dgi_local_samples"], 65536)
        self.assertEqual(
            resolved["module8"]["dgi_local_graph_key"],
            "auto",
        )
        self.assertEqual(
            resolved["module8"]["dgi_cpu_backend"],
            "auto",
        )
        self.assertEqual(
            _build_module_execution_order(resolved),
            ["m1", "m2", "m3", "m4", "m7", "m8", "m5", "m6"],
        )

    def test_single_dgi_unit_switch_resolves_metadomain_dependencies(self):
        resolved = resolve_dgi_configuration(
            {
                "module5": {"cluster_rep": METASPOT_DGI_KEY},
                "module6": {"pool_reps": [METASPOT_DGI_KEY]},
                "module7": {"graph_unit": "metaspot"},
                "module8": {"dgi_unit": "metadomain"},
            }
        )

        self.assertEqual(dgi_unit_from_config(resolved), "metadomain")
        self.assertEqual(
            resolved["module5"]["cluster_rep"],
            "legacy_features",
        )
        self.assertEqual(resolved["module6"]["pool_reps"], [])
        self.assertEqual(resolved["module7"]["graph_unit"], "metadomain")
        self.assertEqual(
            _build_module_execution_order(resolved),
            ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"],
        )

    def test_legacy_graph_unit_remains_supported(self):
        resolved = resolve_dgi_configuration(
            {"module7": {"graph_unit": "metaspot"}}
        )
        self.assertEqual(resolved["module8"]["dgi_unit"], "metaspot")
        self.assertEqual(
            resolved["module5"]["cluster_rep"],
            METASPOT_DGI_KEY,
        )

    def test_invalid_dgi_unit_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "module8.dgi_unit"):
            resolve_dgi_configuration(
                {"module8": {"dgi_unit": "transcript"}}
            )

    def test_metaspot_annotation_rejects_metadomain_dgi_early(self):
        with self.assertRaisesRegex(
            ValueError,
            "annotation.unit='metaspot'.*module8.dgi_unit='metaspot'",
        ):
            resolve_dgi_configuration(
                {
                    "annotation": {"unit": "metaspot"},
                    "module8": {"dgi_unit": "metadomain"},
                }
            )

    def test_invalid_dgi_encoder_configuration_fails_closed(self):
        invalid_cases = (
            ({"dgi_encoder": "deep_plain"}, "dgi_encoder"),
            ({"dgi_layers": 0}, "dgi_layers"),
            ({"dgi_out": 0}, "dgi_hidden and module8.dgi_out"),
            ({"dgi_dropout": 1.0}, "dgi_dropout"),
            ({"dgi_residual_alpha": -0.1}, "dgi_residual_alpha"),
            ({"dgi_layer_norm": "yes"}, "dgi_layer_norm"),
            ({"dgi_device": "mps"}, "dgi_device"),
            ({"dgi_cpu_threads": -1}, "dgi_cpu_threads"),
            ({"dgi_sparse_layout": "dense"}, "dgi_sparse_layout"),
            ({"dgi_tf32": "yes"}, "dgi_tf32"),
            (
                {"dmgi_fused_corruption_pass": "yes"},
                "dmgi_fused_corruption_pass",
            ),
            ({"dgi_corruption_mode": "label_shuffle"}, "dgi_corruption_mode"),
            ({"dgi_local_loss_weight": -0.1}, "dgi_local_loss_weight"),
            ({"dgi_local_temperature": 0.0}, "dgi_local_temperature"),
            ({"dgi_local_samples": -1}, "dgi_local_samples"),
            ({"dgi_local_graph_key": ""}, "dgi_local_graph_key"),
            ({"dgi_cpu_backend": "dense"}, "dgi_cpu_backend"),
            ({"dgi_framework": "supervised_gnn"}, "dgi_framework"),
            (
                {
                    "dgi_framework": "cell_aware_dmgi",
                    "dmgi_gate_floor": 0.5,
                },
                "dmgi_gate_floor",
            ),
            (
                {
                    "dgi_framework": "cell_aware_dmgi",
                    "dgi_epochs": 10,
                    "dmgi_min_epochs": 11,
                },
                "dmgi_min_epochs",
            ),
        )
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_dgi_configuration(
                        {
                            "module8": {
                                "dgi_unit": "metaspot",
                                **overrides,
                            }
                        }
                    )

    def test_invalid_consensus_leiden_configuration_fails_closed(self):
        invalid_cases = (
            ({"consensus_seeds": [1]}, "consensus_seeds"),
            ({"consensus_resolutions": [0.0, 1.0]}, "consensus_resolutions"),
            ({"consensus_neighbors": 1}, "neighbors/iterations"),
            ({"consensus_edge_agreement": 0.0}, "consensus_edge_agreement"),
            ({"consensus_min_stability": 1.1}, "consensus_min_stability"),
            (
                {"consensus_stability_tolerance": -0.1},
                "consensus_stability_tolerance",
            ),
        )
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_dgi_configuration(
                        {
                            "module5": {
                                "clustering_method": "consensus_leiden",
                                **overrides,
                            },
                            "module8": {"dgi_unit": "metaspot"},
                        }
                    )

    def test_same_cell_corruption_never_crosses_groups_or_self(self):
        groups = np.asarray(["c1", "c1", "c1", "c2", "c2", "c3"])
        plan = _build_groupwise_corruption_plan(groups)
        permutation = _draw_groupwise_corruption_index(
            plan,
            np.random.default_rng(11),
        )

        np.testing.assert_array_equal(groups[permutation], groups)
        nonsingleton = np.asarray([True, True, True, True, True, False])
        self.assertTrue(np.all(permutation[nonsingleton] != np.arange(6)[nonsingleton]))
        self.assertEqual(permutation[5], 5)

    def test_dmgi_fused_sparse_pass_matches_separate_pass_without_dropout(self):
        rng = np.random.default_rng(7)
        n_nodes = 24
        features = rng.normal(size=(n_nodes, 7)).astype(np.float32)
        rows = np.arange(n_nodes)
        expression = sp.coo_matrix(
            (
                np.ones(n_nodes, dtype=np.float32),
                (rows, (rows + 1) % n_nodes),
            ),
            shape=(n_nodes, n_nodes),
        ).tocsr()
        spatial = sp.coo_matrix(
            (
                np.ones(n_nodes, dtype=np.float32),
                (rows, (rows + 3) % n_nodes),
            ),
            shape=(n_nodes, n_nodes),
        ).tocsr()
        cell_ids = np.repeat(np.arange(6), 4).astype(str)
        base_config = {
            "dgi_epochs": 3,
            "dgi_hidden": 8,
            "dgi_out": 5,
            "dgi_layers": 2,
            "dgi_dropout": 0.0,
            "dmgi_early_stopping": False,
            "dmgi_min_epochs": 1,
            "dgi_device": "cpu",
            "dgi_cpu_threads": 1,
            "dgi_sparse_layout": "csr",
            "random_seed": 13,
        }
        separate = train_cell_aware_dmgi(
            features,
            expression,
            spatial,
            cell_ids,
            config={
                **base_config,
                "dmgi_fused_corruption_pass": False,
            },
        )
        fused = train_cell_aware_dmgi(
            features,
            expression,
            spatial,
            cell_ids,
            config={
                **base_config,
                "dmgi_fused_corruption_pass": True,
            },
        )

        for key in (
            "embedding",
            "expression_embedding",
            "spatial_embedding",
            "expression_gate",
        ):
            np.testing.assert_allclose(fused[key], separate[key], atol=1e-6)
        np.testing.assert_allclose(
            fused["history"]["loss"],
            separate["history"]["loss"],
            atol=1e-6,
        )
        self.assertEqual(fused["runtime"]["device"], "cpu")
        self.assertEqual(fused["runtime"]["cpu_threads"], 1)
        self.assertEqual(fused["runtime"]["sparse_layout"], "csr")
        self.assertTrue(fused["runtime"]["fused_corruption_pass"])

    def test_sampled_dmgi_honors_cpu_threads_and_records_timings(self):
        import torch

        rng = np.random.default_rng(17)
        n_nodes = 16
        rows = np.arange(n_nodes)
        features = rng.normal(size=(n_nodes, 6)).astype(np.float32)
        expression = sp.csr_matrix(
            (
                np.ones(n_nodes, dtype=np.float32),
                (rows, (rows + 1) % n_nodes),
            ),
            shape=(n_nodes, n_nodes),
        )
        spatial = sp.csr_matrix(
            (
                np.ones(n_nodes, dtype=np.float32),
                (rows, (rows + 2) % n_nodes),
            ),
            shape=(n_nodes, n_nodes),
        )
        previous_threads = torch.get_num_threads()
        try:
            result = train_sampled_cell_aware_dmgi(
                features,
                expression,
                spatial,
                np.repeat(np.arange(4), 4).astype(str),
                config={
                    "dgi_epochs": 1,
                    "dgi_hidden": 8,
                    "dgi_out": 4,
                    "dgi_layers": 1,
                    "dgi_dropout": 0.0,
                    "dgi_device": "cpu",
                    "dgi_cpu_threads": 2,
                    "dmgi_early_stopping": False,
                    "dmgi_min_epochs": 1,
                    "dmgi_patience": 1,
                    "sampled_batch_size": 8,
                    "sampled_nodes_per_epoch": 14,
                    "sampled_inference_batch_size": 8,
                    "sampled_validation_nodes": 2,
                    "sampled_num_workers": 0,
                    "amp": False,
                    "random_seed": 19,
                },
            )
        finally:
            torch.set_num_threads(previous_threads)

        self.assertEqual(result["runtime"]["cpu_threads"], 2)
        self.assertEqual(result["runtime"]["sampled_num_workers"], 0)
        self.assertEqual(result["history"]["validation_loss"].shape, (1,))
        self.assertTrue(np.isfinite(result["embedding"]).all())
        timings = result["runtime"]["timing_seconds"]
        for key in (
            "train_loader",
            "train_compute",
            "validation_loader",
            "validation_compute",
            "inference_loader",
            "inference_compute",
            "total",
        ):
            self.assertGreaterEqual(timings[key], 0.0)

    def test_local_contrastive_pairs_use_edges_and_same_cell_nonedges(self):
        graph = sp.csr_matrix(
            (
                np.ones(4, dtype=np.float32),
                (
                    np.asarray([0, 1, 3, 4]),
                    np.asarray([1, 0, 4, 3]),
                ),
            ),
            shape=(6, 6),
        )
        groups = np.asarray(["c1", "c1", "c1", "c2", "c2", "c2"])
        source, positive, negative = _sample_local_contrastive_pairs(
            graph,
            groups,
            n_samples=10,
            seed=5,
        )

        self.assertEqual(source.size, 2)
        self.assertTrue(np.all(graph[source, positive].A1 > 0))
        self.assertTrue(np.all(graph[source, negative].A1 == 0))
        np.testing.assert_array_equal(groups[source], groups[negative])
        self.assertTrue(np.all(source != negative))

    def test_metaspot_sampling_covers_distinct_cells_when_cells_exceed_sample(self):
        cells = np.repeat(
            np.asarray([f"c{i}" for i in range(100)]),
            2,
        )
        selected = _stratified_metaspot_sample(
            cells,
            20,
            seed=17,
        )
        self.assertEqual(selected.size, 20)
        self.assertEqual(np.unique(selected).size, 20)
        self.assertEqual(np.unique(cells[selected]).size, 20)

    def test_metaspot_spatial_edges_are_strictly_within_cell(self):
        points = _point_table()
        adata1 = build_adata1(points)
        result = build_metaspot_graph(
            adata1,
            points,
            k_expr=2,
            k_spatial=1,
            expr_n_components=2,
            use_geometry_features=False,
        )

        spatial = result.obsp["metaspot_spatial_connectivities"].tocoo()
        cells = result.obs["cell_id"].astype(str).to_numpy()
        self.assertGreater(spatial.nnz, 0)
        self.assertTrue(np.all(cells[spatial.row] == cells[spatial.col]))
        self.assertEqual(result.obsp[METASPOT_GRAPH_KEY].shape, (6, 6))
        self.assertEqual(
            result.uns["metaspot_graph"]["feature_keys"],
            ["X_expr_metaspot_scaled"],
        )
        self.assertFalse(
            result.uns["metaspot_graph"]["uses_boundary_geometry_features"]
        )

    def test_metaspot_expression_edges_can_be_strictly_cross_cell(self):
        points = _point_table()
        result = build_metaspot_graph(
            build_adata1(points),
            points,
            k_expr=5,
            k_spatial=1,
            expr_n_components=2,
            use_geometry_features=False,
            expression_cross_cell_only=True,
        )

        expression = result.obsp[
            "metaspot_expr_connectivities"
        ].tocoo()
        cells = result.obs["cell_id"].astype(str).to_numpy()
        self.assertGreater(expression.nnz, 0)
        self.assertTrue(
            np.all(cells[expression.row] != cells[expression.col])
        )
        self.assertTrue(
            result.uns["metaspot_graph"][
                "expression_edges_cross_cell_only"
            ]
        )
        graph_metadata = result.uns["metaspot_graph"]["ann_metadata"]
        self.assertEqual(
            graph_metadata["nodes_below_available_degree"],
            0,
        )
        self.assertEqual(graph_metadata["isolated_fraction"], 0.0)

    def test_metaspot_graph_full_batch_guard_fails_closed(self):
        points = _point_table()
        with self.assertRaisesRegex(ValueError, "above module7.max_full_batch_nodes"):
            build_metaspot_graph(
                build_adata1(points),
                points,
                max_full_batch_nodes=2,
            )

    def test_module_order_places_graph_and_dgi_before_metadomain(self):
        self.assertEqual(
            _build_module_execution_order(
                {"module7": {"graph_unit": "metaspot"}}
            ),
            ["m1", "m2", "m3", "m4", "m7", "m8", "m5", "m6"],
        )
        self.assertEqual(
            _build_module_execution_order(
                {"module7": {"graph_unit": "metadomain"}}
            ),
            ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"],
        )

    def test_metadomain_feature_selector_requires_explicit_dgi_rep(self):
        adata1 = ad.AnnData(
            X=sp.csr_matrix(np.eye(4, dtype=np.float32)),
            obs=pd.DataFrame(
                {"cell_id": ["c1", "c1", "c2", "c2"]},
                index=["m1", "m2", "m3", "m4"],
            ),
        )
        adata1.obsm[METASPOT_DGI_KEY] = np.array(
            [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        )
        matrix, cells, unique_cells, codes = _metadomain_feature_matrix(
            adata1,
            pd.DataFrame(),
            cluster_rep=METASPOT_DGI_KEY,
            cluster_rep_decell=False,
            speed_preset="ultra",
            use_spatial_feats=False,
            use_decell=False,
            use_cosine=True,
            w_gc=0.0,
            w_expr=0.0,
            w_sp=0.0,
            use_gc=False,
            use_expr=False,
        )
        self.assertEqual(matrix.shape, (4, 2))
        self.assertEqual(cells.tolist(), ["c1", "c1", "c2", "c2"])
        self.assertEqual(unique_cells.tolist(), ["c1", "c2"])
        self.assertEqual(codes.tolist(), [0, 0, 1, 1])
        self.assertLess(float(matrix[0] @ matrix[2]), 0.0)

        with self.assertRaisesRegex(KeyError, "refusing to fall back"):
            _metadomain_feature_matrix(
                adata1,
                pd.DataFrame(),
                cluster_rep="X_missing",
                cluster_rep_decell=False,
                speed_preset="ultra",
                use_spatial_feats=False,
                use_decell=False,
                use_cosine=True,
                w_gc=0.0,
                w_expr=0.0,
                w_sp=0.0,
                use_gc=False,
                use_expr=False,
            )

    def test_module5_sampled_knee_selects_and_records_data_driven_k(self):
        rng = np.random.default_rng(7)
        centers = np.asarray(
            [[-2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        embedding = np.vstack(
            [
                center + rng.normal(0.0, 0.18, size=(40, 3))
                for center in centers
            ]
        ).astype(np.float32)
        adata1 = ad.AnnData(
            X=sp.csr_matrix(np.ones((120, 2), dtype=np.float32)),
            obs=pd.DataFrame(
                {
                    "cell_id": [f"c{i % 12}" for i in range(120)],
                    "r_p50": np.linspace(-1.0, 1.0, 120),
                },
                index=[f"m{i}" for i in range(120)],
            ),
            var=pd.DataFrame(index=["G1", "G2"]),
        )
        adata1.obsm["X_test_metaspot"] = embedding

        result, diagnostics = run_module5_simple(
            adata1,
            pd.DataFrame(),
            module5_config_override={
                "clustering_method": "minibatch_kmeans",
                "cluster_rep": "X_test_metaspot",
                "estimate_k": "auto",
                "auto_k_min": 2,
                "auto_k_max": 8,
                "auto_k_min_average_size": 10,
                "auto_k_max_average_size": 60,
                "auto_k_n_candidates": 4,
                "auto_k_train_sample": 80,
                "auto_k_validation_sample": 30,
                "auto_k_seeds": [3, 4],
                "auto_k_stability_min": 0.0,
                "auto_k_max_small_cluster_fraction": 1.0,
                "auto_k_min_train_per_cluster": 5,
                "auto_k_full_eval_max": 1000,
                "min_domain_size": 2,
                "enable_cap": False,
                "split_by_nucleus": False,
            },
        )

        self.assertEqual(int(diagnostics.loc[0, "K_total"]), 3)
        self.assertEqual(
            diagnostics.loc[0, "selection_method"],
            "sampled_knee",
        )
        self.assertEqual(result.obs["meta-domain"].nunique(), 3)
        k_selection = result.uns[
            "module5_metadomain_clustering"
        ]["k_selection"]
        self.assertEqual(k_selection["selected_k"], 3)
        self.assertEqual(k_selection["knee_k"], 3)
        np.testing.assert_array_equal(
            k_selection["candidate_k"],
            np.asarray([2, 3, 5, 8], dtype=np.int32),
        )
        self.assertEqual(k_selection["distortion"].shape, (4,))
        self.assertEqual(k_selection["stability_ari"].shape, (4,))

    def test_consensus_leiden_selects_stable_resolution_without_fixed_k(self):
        rng = np.random.default_rng(31)
        cells = np.asarray([f"c{i % 10}" for i in range(90)])
        centers = np.asarray(
            [[-4.0, 0.0], [0.0, 4.0], [4.0, 0.0]],
            dtype=np.float32,
        )
        values = np.vstack(
            [
                center + rng.normal(0.0, 0.20, size=(30, 2))
                for center in centers
            ]
        ).astype(np.float32)
        config = {
            "consensus_neighbors": 8,
            "consensus_resolutions": [0.2, 0.5, 1.0],
            "consensus_seeds": [3, 5, 7],
            "consensus_leiden_iterations": 2,
            "consensus_min_average_size": 10,
            "consensus_max_average_size": 40,
            "consensus_min_recurrence_cells": 2,
            "consensus_max_small_cluster_fraction": 1.0,
            "consensus_max_nonrecurrent_fraction": 1.0,
            "consensus_edge_agreement": 2.0 / 3.0,
            "min_domain_size": 2,
            "random_seed": 19,
        }

        labels, metadata = run_stability_consensus_leiden(
            values,
            cells,
            config=config,
            n_jobs=1,
        )

        self.assertEqual(labels.shape, (90,))
        self.assertGreaterEqual(np.unique(labels).size, 3)
        self.assertEqual(metadata["method"], "stability_consensus_leiden")
        self.assertIn(
            metadata["selected_resolution"],
            config["consensus_resolutions"],
        )
        self.assertEqual(metadata["candidate_resolution"].shape, (3,))
        self.assertEqual(metadata["candidate_stability_ari"].shape, (3,))
        self.assertTrue(
            np.all(metadata["candidate_stability_ari"] >= -1.0)
        )
        self.assertGreater(metadata["mutual_graph_nnz"], 0)
        self.assertGreater(metadata["consensus_graph_nnz"], 0)
        self.assertEqual(
            metadata["selection_reason"],
            "finest_within_stability_tolerance",
        )

    def test_metadomain_quality_tables_report_geometry_and_label_transfer(self):
        adata1 = ad.AnnData(
            X=sp.csr_matrix(np.ones((6, 2), dtype=np.float32)),
            obs=pd.DataFrame(
                {
                    "meta-domain": ["d1", "d1", "d1", "d2", "d2", "d2"],
                    "cell_id": ["c1", "c2", "c3", "c1", "c2", "c3"],
                    "r_mean": [-0.8, -0.6, 0.2, 0.2, 0.4, 0.6],
                    "subcellular_annotation_domain": [
                        "Nucleus",
                        "Nucleus",
                        "Nucleus",
                        "ER",
                        "ER",
                        "ER",
                    ],
                    "subcellular_annotation": [
                        "Nucleus",
                        "Nucleus",
                        "Cytosol",
                        "Cytosol",
                        "ER",
                        "ER",
                    ],
                },
                index=[f"m{i}" for i in range(6)],
            ),
            var=pd.DataFrame(index=["G1", "G2"]),
        )
        adata1.obsm["X_test_metaspot"] = np.asarray(
            [
                [-2.0, 0.0],
                [-1.8, 0.1],
                [-1.6, -0.1],
                [1.6, 0.0],
                [1.8, 0.1],
                [2.0, -0.1],
            ],
            dtype=np.float32,
        )
        adata1.uns["module5_metadomain_clustering"] = {
            "cluster_rep": "X_test_metaspot",
        }

        tables = _build_metadomain_annotation_quality_tables(adata1)
        domains = tables["domains"]
        self.assertEqual(domains.loc["d1", "n_metaspots"], 3)
        self.assertAlmostEqual(
            domains.loc["d1", "geometry_purity"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            domains.loc["d1", "domain_final_broad_agreement"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            domains.loc["d2", "domain_final_broad_agreement"],
            1.0,
        )
        self.assertTrue(
            np.isfinite(
                domains["embedding_loo_cosine_median"]
            ).all()
        )
        self.assertTrue(domains["representation_available"].all())
        transfer = tables["transfer_counts"]
        self.assertEqual(transfer.loc["Nucleus", "Cytosol"], 1)
        self.assertEqual(transfer.loc["ER", "ER"], 2)

    def test_metadomain_quality_marks_missing_representation_unavailable(self):
        adata1 = ad.AnnData(
            X=sp.csr_matrix(np.ones((2, 1), dtype=np.float32)),
            obs=pd.DataFrame(
                {
                    "meta-domain": ["d1", "d1"],
                    "cell_id": ["c1", "c2"],
                    "r_mean": [-0.5, 0.5],
                    "subcellular_annotation_domain": [
                        "Nucleus",
                        "Nucleus",
                    ],
                    "subcellular_annotation": [
                        "Nucleus",
                        "Cytosol",
                    ],
                },
                index=["m1", "m2"],
            ),
            var=pd.DataFrame(index=["G1"]),
        )
        adata1.uns["module5_metadomain_clustering"] = {
            "cluster_rep": "legacy_features",
        }

        domains = _build_metadomain_annotation_quality_tables(
            adata1
        )["domains"]

        self.assertFalse(domains["representation_available"].any())
        self.assertTrue(
            domains["embedding_loo_cosine_median"].isna().all()
        )

    def test_metaspot_result_tables_separate_domain_and_local_updates(self):
        adata1 = ad.AnnData(
            X=sp.csr_matrix(np.ones((6, 2), dtype=np.float32)),
            obs=pd.DataFrame(
                {
                    "cell_id": ["c1", "c1", "c2", "c2", "c3", "c3"],
                    "subcellular_annotation_domain": [
                        "Nucleus",
                        "Nucleus",
                        "ER",
                        "ER",
                        "OMM",
                        "Cytosol",
                    ],
                    "subcellular_annotation": [
                        "Nucleus",
                        "Cytosol",
                        "ER",
                        "OMM",
                        "ER",
                        "Cytosol",
                    ],
                    "subcellular_annotation_tier": [
                        "Strict",
                        "Strict",
                        "Exploratory",
                        "Exploratory",
                        "Candidate",
                        "Strict",
                    ],
                },
                index=[f"m{i}" for i in range(6)],
            ),
            var=pd.DataFrame(index=["G1", "G2"]),
        )

        tables = _build_metaspot_annotation_result_tables(adata1)
        updates = tables["spots"]["annotation_update"].value_counts()
        self.assertEqual(updates["Unchanged"], 3)
        self.assertEqual(updates["Subtype/local refinement"], 2)
        self.assertEqual(updates["Broad-compartment correction"], 1)
        self.assertIn(
            "domain_evidence_tier",
            tables["spots"].columns,
        )
        self.assertNotIn(
            "annotation_tier",
            tables["spots"].columns,
        )
        self.assertAlmostEqual(
            tables["summary"]["changed_fraction"],
            0.5,
        )
        er = tables["label_summary"].set_index("label").loc["ER"]
        self.assertEqual(er["domain_evidence_n_metaspots"], 2)
        self.assertEqual(er["final_n_metaspots"], 2)
        self.assertEqual(er["changed_to_label_n_metaspots"], 1)
        self.assertEqual(er["final_n_cells"], 2)
        transfer = tables["transfer_counts"]
        self.assertEqual(transfer.loc["Nucleus", "Nucleus"], 1)
        self.assertEqual(transfer.loc["Nucleus", "Cytosol"], 1)
        self.assertAlmostEqual(
            tables["transfer_fraction"].loc["Nucleus", "Cytosol"],
            0.5,
        )
        provenance = tables["provenance_counts"]
        self.assertEqual(provenance.loc["Cytosol", "Unchanged"], 1)
        self.assertEqual(
            provenance.loc[
                "Cytosol",
                "Broad-compartment correction",
            ],
            1,
        )
        self.assertAlmostEqual(
            tables["provenance_fraction"].loc[
                "Cytosol",
                "Broad-compartment correction",
            ],
            0.5,
        )

    def test_transcript_proportion_tables_include_zero_share_cells(self):
        final_df = pd.DataFrame(
            {
                "cell_id": ["c1", "c1", "c1", "c1", "c2", "c2"],
                "overlaps_nucleus": [1, 1, 0, 0, True, True],
                "subcellular_annotation": pd.Categorical(
                    [
                        "Nucleus",
                        "Nucleus",
                        "ER",
                        "Cytosol",
                        "OMM",
                        "OMM",
                    ]
                ),
            }
        )

        truth = _build_ground_truth_nuclear_cytoplasmic_proportions(
            final_df
        )
        final = _build_final_annotation_proportions(final_df)

        self.assertEqual(
            truth["proportions"].columns.tolist(),
            ["Nucleus", "Cytosol"],
        )
        self.assertAlmostEqual(
            truth["proportions"].loc["c1", "Nucleus"],
            50.0,
        )
        self.assertAlmostEqual(
            truth["proportions"].loc["c2", "Cytosol"],
            0.0,
        )
        self.assertAlmostEqual(
            final["proportions"].loc["c1", "OMM"],
            0.0,
        )
        self.assertAlmostEqual(
            final["proportions"].loc["c2", "OMM"],
            100.0,
        )
        self.assertIn("Nucleolus", final["proportions"].columns)
        self.assertTrue(
            final["proportions"]["Nucleolus"].eq(0.0).all()
        )
        np.testing.assert_allclose(
            truth["proportions"].sum(axis=1).to_numpy(),
            100.0,
        )
        np.testing.assert_allclose(
            final["proportions"].sum(axis=1).to_numpy(),
            100.0,
        )

        truth_summary = truth["summary"].set_index("label")
        final_summary = final["summary"].set_index("label")
        self.assertAlmostEqual(
            truth_summary.loc[
                "Nucleus",
                "global_transcript_percentage",
            ],
            4 / 6 * 100,
        )
        self.assertAlmostEqual(
            final_summary.loc["OMM", "cell_detection_percentage"],
            50.0,
        )

    def test_transcript_proportion_diagnostics_write_both_plots_and_tables(self):
        final_df = pd.DataFrame(
            {
                "cell_id": ["c1", "c1", "c2", "c2"],
                "overlaps_nucleus": [1, 0, 1, 1],
                "subcellular_annotation": [
                    "Nucleus",
                    "Cytosol",
                    "ER",
                    "OMM",
                ],
                "subcellular_fine_annotation": [
                    "Nucleus",
                    "Cytosol",
                    "ER",
                    "OMM",
                ],
            }
        )
        config = {
            "io": {
                "intermediate_dirname": "intermediate_results",
                "write_intermediate_figures": True,
                "write_diagnostic_tables": True,
                "write_diagnostic_csv": True,
                "write_diagnostic_parquet": False,
            }
        }
        with tempfile.TemporaryDirectory() as out_dir, _config_file(config):
            results = _save_transcript_annotation_proportion_diagnostics(
                final_df,
                out_dir,
            )
            intermediate = os.path.join(out_dir, "intermediate_results")
            for name in (
                "ground_truth_nuclear_cytoplasmic_proportions",
                "final_annotation_proportions",
            ):
                self.assertIn(name, results)
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(intermediate, f"{name}.png")
                    )
                )
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(intermediate, f"{name}.csv")
                    )
                )
            self.assertNotIn("fine_annotation_proportions", results)
            self.assertFalse(
                os.path.exists(
                    os.path.join(
                        intermediate,
                        "fine_annotation_proportions.png",
                    )
                )
            )

    def test_rapx_strict_evidence_is_table_only(self):
        adata2 = ad.AnnData(
            X=np.zeros((4, 1), dtype=np.float32),
            obs=pd.DataFrame(
                {
                    "subcell_core_candidate_label": [
                        "Cytosol",
                        "ER",
                        "OMM",
                        "Nucleus",
                    ],
                    "subcellular_annotation_strict": [
                        "Unknown",
                        "ER",
                        "Unknown",
                        "Unknown",
                    ],
                    "subcellular_annotation_tier": [
                        "Candidate",
                        "Strict",
                        "Exploratory",
                        "Candidate",
                    ],
                },
                index=["d1", "d2", "d3", "d4"],
            ),
        )
        config = {
            "io": {
                "intermediate_dirname": "intermediate_results",
                "write_intermediate_figures": True,
                "write_diagnostic_tables": True,
                "write_diagnostic_csv": True,
                "write_diagnostic_parquet": False,
            }
        }
        with tempfile.TemporaryDirectory() as out_dir, _config_file(config):
            tables = _save_rapx_metadomain_evidence_audit_tables(
                adata2,
                out_dir,
            )
            intermediate = os.path.join(out_dir, "intermediate_results")
            self.assertIsNotNone(tables)
            for name in (
                "rapx_metadomain_evidence",
                "rapx_candidate_to_strict",
                "rapx_candidate_evidence_tier",
            ):
                self.assertTrue(
                    os.path.isfile(os.path.join(intermediate, f"{name}.csv"))
                )
            self.assertFalse(
                os.path.exists(
                    os.path.join(
                        intermediate,
                        "rapx_umap_cluster_annotation.png",
                    )
                )
            )

    def test_output_cleanup_is_exact_and_output_guide_separates_roles(self):
        config = {
            "io": {"intermediate_dirname": "intermediate_results"}
        }
        with tempfile.TemporaryDirectory() as out_dir, _config_file(config):
            intermediate = os.path.join(out_dir, "intermediate_results")
            os.makedirs(intermediate, exist_ok=True)
            deprecated = (
                "rapx_umap_cluster_annotation.png",
                "annotation_umap_leiden.png",
                "fine_annotation_proportions.csv",
            )
            for name in deprecated:
                with open(os.path.join(intermediate, name), "wb") as handle:
                    handle.write(b"deprecated")
            retained = os.path.join(intermediate, "rapx_annotation_scores.csv")
            with open(retained, "wb") as handle:
                handle.write(b"audit")

            removed = _remove_deprecated_output_artifacts(out_dir)
            self.assertEqual(len(removed), len(deprecated))
            self.assertTrue(os.path.isfile(retained))
            self.assertTrue(
                all(
                    not os.path.exists(os.path.join(intermediate, name))
                    for name in deprecated
                )
            )

            inactive = (
                "metadomain_k_selection_diagnostics.png",
                "module5_k_selection_diagnostics.csv",
                "module8_dgi_GCN_curves.png",
            )
            for name in inactive:
                with open(os.path.join(intermediate, name), "wb") as handle:
                    handle.write(b"inactive")
            active = os.path.join(
                intermediate,
                "module8_ca_dmgi_training.png",
            )
            with open(active, "wb") as handle:
                handle.write(b"active")
            legacy = os.path.join(
                intermediate,
                "module8_ch_dmgi_training.png",  # legacy compatibility only
            )
            with open(legacy, "wb") as handle:
                handle.write(b"legacy")
            removed = _remove_inactive_method_artifacts(
                out_dir,
                {
                    "module5": {"clustering_method": "consensus_leiden"},
                    "module8": {"dgi_framework": "cell_aware_dmgi"},
                },
            )
            self.assertEqual(len(removed), len(inactive) + 1)
            self.assertTrue(os.path.isfile(active))
            self.assertFalse(os.path.exists(legacy))

            guide = _write_output_guide(
                out_dir,
                counts={
                    "n_transcripts": 100,
                    "n_cells": 5,
                    "n_metaspots": 20,
                    "n_metadomains": 4,
                },
                annotation_enabled=True,
            )
            with open(guide, encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn("Primary data products", text)
            self.assertIn("Recommended result figures", text)
            self.assertIn("Audit-only annotation tables", text)
            self.assertIn("not the final public annotation", text)
            self.assertIn("module8_ca_dmgi_training_history", text)
            self.assertIn("module8_ca_dmgi_summary", text)
            self.assertNotIn("module7_degree_hist", text)

            csv_guide = _write_output_guide(
                out_dir,
                counts={
                    "n_transcripts": 100,
                    "n_cells": 5,
                    "n_metaspots": 20,
                    "n_metadomains": 4,
                },
                annotation_enabled=False,
                primary_data_product="final_data.csv",
            )
            with open(csv_guide, encoding="utf-8") as handle:
                csv_text = handle.read()
            self.assertIn(
                "`final_data.csv`: transcript-level production result",
                csv_text,
            )

    def test_annotation_provenance_columns_separate_domain_and_local_labels(self):
        frame = pd.DataFrame(
            {
                "subcellular_annotation_domain": [
                    "Nucleus",
                    "ER",
                    "OMM",
                ],
                "subcellular_annotation": [
                    "Nucleus",
                    "Cytosol",
                    "Nucleus",
                ],
                "subcellular_annotation_domain_strict": [
                    "Nucleus",
                    "Unknown",
                    "Nucleus",
                ],
            }
        )

        # Reproduce the heavily fragmented annotation tables emitted by older
        # versions without allowing the setup warning to obscure the behavior
        # under test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            for column_index in range(110):
                frame[f"legacy_audit_{column_index}"] = column_index

        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            frame = _attach_annotation_provenance_columns(frame)

        assert frame["subcellular_annotation_provenance"].astype(str).tolist() == [
            "MetaDomain RAP-X",
            "MetaSpot subtype/local refinement",
            "MetaSpot broad-compartment correction",
        ]
        assert frame["final_label_supported_by_domain_evidence"].tolist() == [
            True,
            False,
            True,
        ]

    def test_rapx_bulk_writeback_avoids_dataframe_fragmentation(self):
        regions = [
            "Nucleus",
            "Nucleolus",
            "Lamina",
            "Nuclear Pore",
            "Cytosol",
            "ER",
            "OMM",
        ]
        genes = [f"G{gene_index:02d}" for gene_index in range(35)]
        gold = pd.DataFrame({"Common_Gene": genes})
        for region_index, region in enumerate(regions):
            gold[region] = 0.0
            start = region_index * 5
            gold.loc[start:start + 4, region] = 1.0

        expression = np.ones((14, len(genes)), dtype=np.float32)
        for obs_index in range(expression.shape[0]):
            start = (obs_index % len(regions)) * 5
            expression[obs_index, start:start + 5] = 20.0
        obs = pd.DataFrame(
            {
                "leiden": [str(index % len(regions)) for index in range(14)],
                "r_mean": [-1.0] * 8 + [1.0] * 6,
            },
            index=[f"metadomain_{index}" for index in range(14)],
        )

        # run_rapx imports Scanpy for its optional plots. The plots are disabled
        # here, so a stub keeps this writeback regression independent of Numba's
        # machine-specific cache configuration.
        with mock.patch.dict(sys.modules, {"scanpy": mock.MagicMock()}):
            for use_two_stage in (False, True):
                adata = ad.AnnData(
                    X=expression.copy(),
                    obs=obs.copy(),
                    var=pd.DataFrame(index=genes),
                )
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "error",
                        pd.errors.PerformanceWarning,
                    )
                    run_rapx(
                        adata,
                        gold_df=gold,
                        plot_umap=False,
                        plot_diag_enrich=False,
                        use_two_stage=use_two_stage,
                        use_spatial_channel=False,
                        set_target_size=5,
                        min_effective_markers=1,
                        null_permutations=8,
                    )
                self.assertIn("subcellular_annotation", adata.obs.columns)
                self.assertIn("subcellular_confidence", adata.obs.columns)

    def test_module6_pools_normalized_metaspot_embeddings(self):
        adata1 = ad.AnnData(
            X=sp.csr_matrix(
                [[2, 0], [0, 3], [1, 1]],
                dtype=np.float32,
            ),
            obs=_domain_obs(),
            var=pd.DataFrame(index=["G1", "G2"]),
        )
        source = np.array(
            [[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        adata1.obsm[METASPOT_DGI_KEY] = source

        config = {
            "module6": {
                "pool_reps": [METASPOT_DGI_KEY],
                "pool_normalize": True,
            },
            "module7": {"graph_unit": "metaspot"},
        }
        with _config_file(config):
            result = build_adata2_from_adata1_and_pointdf(
                adata1,
                pd.DataFrame(),
            )

        normalized = source / np.linalg.norm(source, axis=1, keepdims=True)
        expected_d1 = normalized[:2].mean(axis=0)
        expected_d1 /= np.linalg.norm(expected_d1)
        expected_d2 = normalized[2]
        expected = np.vstack([expected_d1, expected_d2])
        np.testing.assert_allclose(
            result.obsm[METADOMAIN_DGI_KEY],
            expected,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.obsm["X_dgi_gcn"],
            expected,
            atol=1e-6,
        )
        self.assertEqual(result.obs["n_metaspots"].tolist(), [2, 1])
        self.assertEqual(
            result.uns["embedding_pooling"]["representations"][
                METADOMAIN_DGI_KEY
            ]["source_unit"],
            "metaspot",
        )

    def test_benchmark_overrides_pool_named_method_embedding_without_dgi_alias(self):
        adata1 = ad.AnnData(
            X=sp.csr_matrix(
                [[2, 0], [0, 3], [1, 1], [2, 1]],
                dtype=np.float32,
            ),
            obs=pd.DataFrame(
                {
                    "cell_id": ["c1", "c1", "c2", "c2"],
                    "r_mean": [-0.5, -0.4, 0.5, 0.6],
                    "r_std": [0.1, 0.1, 0.2, 0.2],
                    "r_p50": [-0.5, -0.4, 0.5, 0.6],
                    "d_cell_norm_mean": [0.4, 0.5, 0.4, 0.5],
                    "d_nuc_norm_mean": [0.2, 0.3, 0.6, 0.7],
                    "elongation": [1.0, 1.1, 1.2, 1.3],
                    "spread": [0.3, 0.4, 0.5, 0.6],
                    "log_n": [1.0, 1.1, 1.2, 1.3],
                },
                index=["m1", "m2", "m3", "m4"],
            ),
            var=pd.DataFrame(index=["G1", "G2"]),
        )
        source_key = "X_method_metaspot"
        target_key = "X_method_metadomain"
        adata1.obsm[source_key] = np.array(
            [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        )

        clustered, _ = run_module5_simple(
            adata1,
            pd.DataFrame(),
            module5_config_override={
                "clustering_method": "minibatch_kmeans",
                "large_scale_method": "minibatch_kmeans",
                "cluster_rep": source_key,
                "estimate_k": "fixed",
                "fixed_k": 2,
                "min_domain_size": 1,
                "enable_cap": False,
                "split_by_nucleus": False,
            },
        )
        result = build_adata2_from_adata1_and_pointdf(
            clustered,
            pd.DataFrame(),
            module6_config_override={
                "pool_reps": [source_key],
                "pool_normalize": True,
            },
            graph_unit_override="benchmark_method_embedding",
            pool_output_key_map={source_key: target_key},
        )

        self.assertIn(target_key, result.obsm)
        self.assertNotIn(METADOMAIN_DGI_KEY, result.obsm)
        self.assertNotIn("X_dgi_gcn", result.obsm)
        provenance = result.uns["embedding_pooling"]["representations"][
            target_key
        ]
        self.assertEqual(provenance["source_key"], source_key)
        self.assertEqual(provenance["target_unit"], "metadomain")

    def test_dgi_writes_native_metaspot_provenance(self):
        points = _point_table()
        adata1 = build_metaspot_graph(
            build_adata1(points),
            points,
            k_expr=2,
            k_spatial=1,
            expr_n_components=2,
        )
        config = {
            "module8": {
                "dgi_framework": "standard_dgi",
                "dgi_epochs": 1,
                "dgi_hidden": 4,
                "dgi_out": 3,
                "enable_dgi_sage": False,
                "amp": False,
                "random_seed": 7,
            }
        }
        with tempfile.TemporaryDirectory() as output_dir, _config_file(config):
            _, result = run_module8_dgi(
                adata1,
                adata1,
                out_dir=output_dir,
                # This test audits the standard-DGI compatibility path. The
                # relation-specific CA-DMGI path has a separate test below.
                graph_key=METASPOT_GRAPH_KEY,
                feature_keys=("X_expr_metaspot_scaled",),
                output_key=METASPOT_DGI_KEY,
                broadcast_to_adata1=False,
                compatibility_alias=True,
            )

        self.assertEqual(result.obsm[METASPOT_DGI_KEY].shape, (6, 3))
        self.assertTrue(np.isfinite(result.obsm[METASPOT_DGI_KEY]).all())
        np.testing.assert_array_equal(
            result.obsm[METASPOT_DGI_KEY],
            result.obsm["X_dgi_gcn"],
        )
        provenance = result.uns["dgi_embeddings"][METASPOT_DGI_KEY]
        self.assertEqual(provenance["node_unit"], "metaspot")
        self.assertEqual(provenance["graph_key"], METASPOT_GRAPH_KEY)
        self.assertEqual(provenance["random_seed"], 7)
        self.assertEqual(provenance["encoder"], "residual")
        self.assertEqual(provenance["message_passing_layers"], 2)
        self.assertEqual(provenance["hidden_dim"], 4)
        self.assertAlmostEqual(provenance["dropout"], 0.1)
        self.assertAlmostEqual(provenance["residual_alpha"], 0.1)
        self.assertTrue(provenance["layer_norm"])
        self.assertEqual(provenance["corruption_mode_requested"], "auto")
        self.assertEqual(provenance["corruption_mode_effective"], "same_cell")
        self.assertEqual(provenance["corruption_group_key"], "cell_id")
        self.assertEqual(provenance["corruption_group_count"], 2)
        self.assertEqual(
            provenance["local_graph_key"],
            "metaspot_spatial_connectivities",
        )
        self.assertAlmostEqual(provenance["local_loss_weight"], 0.2)
        self.assertGreater(provenance["local_pairs_sampled"], 0)
        self.assertEqual(provenance["gcn_backend_requested"], "auto")
        self.assertIn(
            provenance["gcn_backend_effective"],
            {"torch_csr", "pyg"},
        )
        self.assertTrue(
            np.isfinite(provenance["final_local_positive_cosine"])
        )
        self.assertTrue(
            np.isfinite(provenance["final_local_negative_cosine"])
        )
        self.assertAlmostEqual(
            provenance["final_local_cosine_margin"],
            provenance["final_local_positive_cosine"]
            - provenance["final_local_negative_cosine"],
        )
        self.assertEqual(len(provenance["training_loss"]), 1)
        self.assertEqual(len(provenance["training_dgi_loss"]), 1)
        self.assertEqual(len(provenance["training_local_loss"]), 1)
        self.assertTrue(np.isfinite(provenance["final_training_loss"]))
        self.assertTrue(np.isfinite(provenance["final_dgi_loss"]))
        self.assertTrue(np.isfinite(provenance["final_local_loss"]))
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "dgi.h5ad")
            result.write_h5ad(checkpoint_path)
            restored = ad.read_h5ad(checkpoint_path)
        restored_provenance = restored.uns["dgi_embeddings"][
            METASPOT_DGI_KEY
        ]
        self.assertEqual(restored_provenance["encoder"], "residual")
        np.testing.assert_allclose(
            restored_provenance["training_loss"],
            provenance["training_loss"],
        )
        np.testing.assert_allclose(
            restored_provenance["training_local_loss"],
            provenance["training_local_loss"],
        )

    def test_cell_aware_dmgi_writes_relation_embeddings_and_audit(self):
        points = _point_table()
        adata1 = build_metaspot_graph(
            build_adata1(points),
            points,
            k_expr=5,
            k_spatial=1,
            expr_n_components=2,
            expression_cross_cell_only=True,
        )
        config = {
            "io": {
                "intermediate_dirname": "intermediate_results",
                "write_diagnostic_tables": True,
                "write_diagnostic_csv": True,
                "write_diagnostic_parquet": False,
            },
            "module8": {
                "dgi_framework": "cell_aware_dmgi",
                "dgi_epochs": 3,
                "dgi_hidden": 4,
                "dgi_out": 3,
                "dgi_layers": 2,
                "dgi_dropout": 0.0,
                "dgi_residual_alpha": 0.1,
                "dgi_layer_norm": True,
                "dgi_device": "cpu",
                "dgi_cpu_threads": 1,
                "dgi_sparse_layout": "csr",
                "dmgi_fused_corruption_pass": True,
                "dgi_lr": 0.001,
                "dgi_weight_decay": 0.0,
                "dgi_clip_grad": 1.0,
                "dmgi_expression_graph_key": (
                    "metaspot_expr_connectivities"
                ),
                "dmgi_spatial_graph_key": (
                    "metaspot_spatial_connectivities"
                ),
                "dmgi_expression_weight": 1.0,
                "dmgi_spatial_weight": 1.0,
                "dmgi_consensus_weight": 0.1,
                "dmgi_gate_floor": 0.1,
                "dmgi_early_stopping": False,
                "dmgi_min_epochs": 1,
                "dmgi_patience": 1,
                "dmgi_min_delta": 0.0,
                "enable_dgi_sage": False,
                "amp": False,
                "random_seed": 13,
            }
        }
        with tempfile.TemporaryDirectory() as output_dir, _config_file(config):
            with self.assertLogs("cellscope", level="INFO") as captured_logs:
                _, result = run_module8_dgi(
                    adata1,
                    adata1,
                    out_dir=output_dir,
                    graph_key=METASPOT_GRAPH_KEY,
                    feature_keys=("X_expr_metaspot_scaled",),
                    output_key=METASPOT_DGI_KEY,
                    broadcast_to_adata1=False,
                    compatibility_alias=True,
                )
            self.assertTrue(
                any("[CA-DMGI]" in message for message in captured_logs.output)
            )
            intermediate = os.path.join(
                output_dir,
                "intermediate_results",
            )
            history_path = os.path.join(
                intermediate,
                "module8_ca_dmgi_training_history.csv",
            )
            summary_path = os.path.join(
                intermediate,
                "module8_ca_dmgi_summary.csv",
            )
            figure_path = os.path.join(
                intermediate,
                "module8_ca_dmgi_training.png",
            )
            self.assertTrue(os.path.isfile(history_path))
            self.assertTrue(os.path.isfile(summary_path))
            self.assertTrue(os.path.isfile(figure_path))
            self.assertFalse(
                os.path.exists(
                    os.path.join(
                        intermediate,
                        "module8_ch_dmgi_training.png",  # legacy compatibility only
                    )
                )
            )
            history_table = pd.read_csv(history_path)
            summary_table = pd.read_csv(summary_path)
            self.assertEqual(len(history_table), 3)
            self.assertEqual(summary_table.loc[0, "epochs_run"], 3)
            self.assertEqual(summary_table.loc[0, "device"], "cpu")
            self.assertEqual(summary_table.loc[0, "cpu_threads"], 1)
            self.assertEqual(summary_table.loc[0, "sparse_layout"], "csr")
            self.assertTrue(
                bool(summary_table.loc[0, "fused_corruption_pass"])
            )
            self.assertEqual(
                summary_table.loc[0, "negative_semantics"],
                "corrupted_features_not_real_nodes",
            )

        self.assertEqual(result.obsm[METASPOT_DGI_KEY].shape, (6, 3))
        self.assertEqual(
            result.obsm[METASPOT_DMGI_EXPRESSION_KEY].shape,
            (6, 3),
        )
        self.assertEqual(
            result.obsm[METASPOT_DMGI_SPATIAL_KEY].shape,
            (6, 3),
        )
        expression_gate = result.obs[
            "dmgi_expression_gate"
        ].to_numpy(float)
        spatial_gate = result.obs["dmgi_spatial_gate"].to_numpy(float)
        self.assertTrue(np.all(expression_gate >= 0.1))
        self.assertTrue(np.all(expression_gate <= 0.9))
        np.testing.assert_allclose(expression_gate + spatial_gate, 1.0)
        provenance = result.uns["dgi_embeddings"][METASPOT_DGI_KEY]
        self.assertEqual(provenance["framework"], "cell_aware_dmgi")
        self.assertEqual(
            provenance["objective"],
            "relation_specific_local_infomax",
        )
        self.assertEqual(
            provenance["negative_semantics"],
            "corrupted_features_not_real_nodes",
        )
        self.assertEqual(provenance["corruption_mode_effective"], "same_cell")
        self.assertEqual(provenance["epochs"], 3)
        self.assertEqual(provenance["device"], "cpu")
        self.assertEqual(provenance["cpu_threads"], 1)
        self.assertEqual(provenance["sparse_layout"], "csr")
        self.assertTrue(provenance["fused_corruption_pass"])
        self.assertEqual(len(provenance["training_loss"]), 3)
        self.assertEqual(
            len(provenance["training_expression_infomax_loss"]),
            3,
        )
        self.assertEqual(
            len(provenance["training_spatial_infomax_loss"]),
            3,
        )
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "dmgi.h5ad")
            result.write_h5ad(checkpoint_path)
            restored = ad.read_h5ad(checkpoint_path)
        self.assertEqual(
            restored.uns["dgi_embeddings"][METASPOT_DGI_KEY]["framework"],
            "cell_aware_dmgi",
        )

    def test_dgi_plain_encoder_and_depth_remain_configurable(self):
        points = _point_table()
        adata1 = build_metaspot_graph(
            build_adata1(points),
            points,
            k_expr=2,
            k_spatial=1,
            expr_n_components=2,
        )
        config = {
            "module8": {
                "dgi_epochs": 1,
                "dgi_encoder": "plain",
                "dgi_layers": 3,
                "dgi_hidden": 5,
                "dgi_out": 2,
                "dgi_dropout": 0.0,
                "dgi_residual_alpha": 0.0,
                "dgi_layer_norm": False,
                "dgi_corruption_mode": "global",
                "dgi_local_loss_weight": 0.0,
                "enable_dgi_sage": False,
                "amp": False,
                "random_seed": 11,
            }
        }
        with tempfile.TemporaryDirectory() as output_dir, _config_file(config):
            _, result = run_module8_dgi(
                adata1,
                adata1,
                out_dir=output_dir,
                graph_key=METASPOT_GRAPH_KEY,
                feature_keys=("X_expr_metaspot_scaled",),
                output_key=METASPOT_DGI_KEY,
                broadcast_to_adata1=False,
            )

        self.assertEqual(result.obsm[METASPOT_DGI_KEY].shape, (6, 2))
        provenance = result.uns["dgi_embeddings"][METASPOT_DGI_KEY]
        self.assertEqual(provenance["encoder"], "plain")
        self.assertEqual(provenance["message_passing_layers"], 3)
        self.assertEqual(provenance["hidden_dim"], 5)
        self.assertEqual(provenance["embedding_dim"], 2)
        self.assertEqual(provenance["dropout"], 0.0)
        self.assertFalse(provenance["layer_norm"])
        self.assertEqual(provenance["corruption_mode_effective"], "global")
        self.assertEqual(provenance["local_pairs_sampled"], 0)
        self.assertEqual(provenance["final_local_loss"], 0.0)

    def test_annotation_selects_aggregated_metadomain_rep(self):
        adata2 = ad.AnnData(X=np.ones((3, 2), dtype=np.float32))
        graph = sp.eye(3, format="csr", dtype=np.float32)
        key = "dgi_gcn_metadomain_nbrs"
        adata2.obsp[f"{key}_connectivities"] = graph
        selected = _choose_annotation_neighbors_key(
            adata2,
            cluster_rep=METADOMAIN_DGI_KEY,
            rep_priority=["X_pca"],
            require_cluster_rep=True,
        )
        self.assertEqual(selected, key)

    def test_new_parameters_change_build_and_module_fingerprints(self):
        base = {
            "module5": {"cluster_rep": "legacy_features"},
            "module7": {
                "graph_unit": "metaspot",
                "neighbors": {"k_spatial": 6},
            },
        }
        changed = {
            "module5": {"cluster_rep": METASPOT_DGI_KEY},
            "module7": {
                "graph_unit": "metaspot",
                "neighbors": {"k_spatial": 8},
            },
        }
        self.assertNotEqual(
            fingerprint_module(base, "m5"),
            fingerprint_module(changed, "m5"),
        )
        self.assertNotEqual(
            fingerprint_module(base, "m7"),
            fingerprint_module(changed, "m7"),
        )
        self.assertNotEqual(
            fingerprint_stage(base, "build_anndata"),
            fingerprint_stage(changed, "build_anndata"),
        )

        auto_base = {
            "module5": {
                "cluster_rep": METASPOT_DGI_KEY,
                "estimate_k": "auto",
                "auto_k_train_sample": 50000,
            }
        }
        auto_changed = {
            "module5": {
                "cluster_rep": METASPOT_DGI_KEY,
                "estimate_k": "auto",
                "auto_k_train_sample": 60000,
            }
        }
        self.assertNotEqual(
            fingerprint_module(auto_base, "m5"),
            fingerprint_module(auto_changed, "m5"),
        )
        self.assertNotEqual(
            fingerprint_stage(auto_base, "build_anndata"),
            fingerprint_stage(auto_changed, "build_anndata"),
        )

        metadomain = resolve_dgi_configuration(
            {"module8": {"dgi_unit": "metadomain"}}
        )
        metaspot = resolve_dgi_configuration(
            {"module8": {"dgi_unit": "metaspot"}}
        )
        self.assertNotEqual(
            fingerprint_stage(metadomain, "build_anndata"),
            fingerprint_stage(metaspot, "build_anndata"),
        )
        for module in ("m5", "m6", "m7", "m8"):
            self.assertNotEqual(
                fingerprint_module(metadomain, module),
                fingerprint_module(metaspot, module),
            )

        architecture_base = {
            "module8": {
                "dgi_encoder": "residual",
                "dgi_layers": 2,
                "dgi_hidden": 64,
                "dgi_out": 32,
                "dgi_dropout": 0.1,
                "dgi_residual_alpha": 0.1,
                "dgi_layer_norm": True,
                "dgi_corruption_mode": "auto",
                "dgi_local_loss_weight": 0.2,
                "dgi_local_temperature": 0.2,
                "dgi_local_samples": 65536,
                "dgi_local_graph_key": "auto",
                "dgi_cpu_backend": "auto",
            }
        }
        architecture_changes = {
            "dgi_encoder": "plain",
            "dgi_layers": 3,
            "dgi_hidden": 128,
            "dgi_out": 64,
            "dgi_dropout": 0.2,
            "dgi_residual_alpha": 0.2,
            "dgi_layer_norm": False,
            "dgi_corruption_mode": "global",
            "dgi_local_loss_weight": 0.0,
            "dgi_local_temperature": 0.5,
            "dgi_local_samples": 1024,
            "dgi_local_graph_key": METASPOT_GRAPH_KEY,
            "dgi_cpu_backend": "pyg",
        }
        for key, value in architecture_changes.items():
            changed_architecture = {
                "module8": {
                    **architecture_base["module8"],
                    key: value,
                }
            }
            self.assertNotEqual(
                fingerprint_module(architecture_base, "m8"),
                fingerprint_module(changed_architecture, "m8"),
                msg=key,
            )
            self.assertNotEqual(
                fingerprint_stage(architecture_base, "build_anndata"),
                fingerprint_stage(changed_architecture, "build_anndata"),
                msg=key,
            )


if __name__ == "__main__":
    unittest.main()
