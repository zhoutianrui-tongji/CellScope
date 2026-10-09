import numpy as np
import pandas as pd
import pytest

from cellscope.hierarchical_metadomain import (
    run_hierarchical_prototype_clustering,
)
from cellscope.scalable_execution import resolve_execution_plan
from cellscope.scalable_graph import build_knn_connectivity
from cellscope.pipeline import (
    _atomic_partitioned_parquet_write,
    _load_module3_membership_checkpoint,
    _save_module3_membership_checkpoint,
)


def _config():
    return {
        "execution": {
            "profile": "auto",
            "exact_max_metaspots": 100,
            "sampled_max_metaspots": 1_000,
            "memory_budget_gb": 8,
            "fail_on_backend_fallback": True,
        },
        "module5": {"large_scale_method": "auto"},
        "module7": {"knn_backend": "auto"},
        "module8": {"training_mode": "auto"},
    }


@pytest.mark.parametrize(
    ("nodes", "profile", "backend", "training", "clustering"),
    [
        (50, "exact", "exact", "full_batch", "consensus_leiden"),
        (500, "sampled", "faiss_hnsw", "neighbor_sampled", "hierarchical_prototype"),
        (5_000, "streaming", "faiss_ivfpq", "neighbor_sampled", "hierarchical_prototype"),
    ],
)
def test_execution_plan_routes_by_realized_metaspot_count(
    nodes, profile, backend, training, clustering
):
    plan = resolve_execution_plan(
        _config(),
        n_metaspots=nodes,
        capability_overrides={
            "faiss": True,
            "pynndescent": True,
            "torch_geometric": True,
            "pyg_lib": True,
            "torch_sparse": False,
        },
    )
    assert plan.effective_profile == profile
    assert plan.knn_backend == backend
    assert plan.dgi_training_mode == training
    assert plan.metadomain_method == clustering


def test_large_profile_refuses_full_batch_dgi():
    config = _config()
    config["module8"]["training_mode"] = "full_batch"
    with pytest.raises(ValueError, match="cannot use full-batch DGI"):
        resolve_execution_plan(
            config,
            n_metaspots=500,
            capability_overrides={
                "faiss": True,
                "pynndescent": True,
                "torch_geometric": True,
                "pyg_lib": True,
                "torch_sparse": False,
            },
        )


def test_exact_profile_allows_explicit_approximate_knn_with_full_batch_dgi():
    config = _config()
    config["module7"]["knn_backend"] = "pynndescent"
    plan = resolve_execution_plan(
        config,
        n_metaspots=50,
        capability_overrides={
            "faiss": False,
            "pynndescent": True,
            "torch_geometric": True,
            "pyg_lib": False,
            "torch_sparse": False,
        },
    )

    assert plan.effective_profile == "exact"
    assert plan.knn_backend == "pynndescent"
    assert plan.dgi_training_mode == "full_batch"


def test_fixed_k_routes_to_minibatch_kmeans():
    config = _config()
    config["module5"].update({"estimate_k": "fixed", "fixed_k": 3})

    plan = resolve_execution_plan(
        config,
        n_metaspots=50,
        capability_overrides={
            "faiss": True,
            "pynndescent": True,
            "torch_geometric": True,
            "pyg_lib": True,
            "torch_sparse": False,
        },
    )

    assert plan.metadomain_method == "minibatch_kmeans"


def test_streaming_profile_refuses_non_faiss_fallback():
    with pytest.raises(RuntimeError, match="requires FAISS IVFPQ"):
        resolve_execution_plan(
            _config(),
            n_metaspots=5_000,
            capability_overrides={
                "faiss": False,
                "pynndescent": True,
                "torch_geometric": True,
                "pyg_lib": True,
                "torch_sparse": False,
            },
        )


def test_chunked_knn_filters_same_group_before_graph_materialization():
    values = np.asarray(
        [[0.0], [0.01], [1.0], [1.01], [2.0], [2.01]],
        dtype=np.float32,
    )
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    graph, metadata = build_knn_connectivity(
        values,
        n_neighbors=2,
        backend="exact",
        excluded_groups=groups,
        candidate_multiplier=3,
        query_chunk_size=2,
    )
    coo = graph.tocoo()
    assert graph.shape == (6, 6)
    assert graph.nnz > 0
    assert np.all(groups[coo.row] != groups[coo.col])
    assert metadata["excluded_same_group"] is True


def test_constrained_exact_knn_finds_neighbors_beyond_same_cell_candidates():
    rng = np.random.default_rng(0)
    values = np.r_[
        rng.normal(0.0, 0.01, size=(100, 4)),
        rng.normal(10.0, 0.01, size=(100, 4)),
    ].astype(np.float32)
    groups = np.asarray(["a"] * 100 + ["b"] * 100)

    graph, metadata = build_knn_connectivity(
        values,
        n_neighbors=15,
        backend="exact",
        excluded_groups=groups,
        candidate_multiplier=4,
    )

    coo = graph.tocoo()
    assert graph.nnz > 0
    assert np.all(groups[coo.row] != groups[coo.col])
    assert metadata["query_k"] >= 115
    assert metadata["nodes_below_available_degree"] == 0
    assert metadata["directed_degree_min"] == 15
    assert metadata["isolated_fraction"] == 0.0


def test_faiss_hnsw_parallel_query_preserves_fixed_index_graph():
    faiss = pytest.importorskip("faiss")
    rng = np.random.default_rng(23)
    values = rng.normal(size=(512, 8)).astype(np.float32)
    previous_threads = faiss.omp_get_max_threads()
    try:
        graph_one, metadata_one = build_knn_connectivity(
            values,
            n_neighbors=5,
            backend="faiss_hnsw",
            candidate_multiplier=2,
            query_chunk_size=128,
            n_jobs=1,
        )
        graph_four, metadata_four = build_knn_connectivity(
            values,
            n_neighbors=5,
            backend="faiss_hnsw",
            candidate_multiplier=2,
            query_chunk_size=128,
            n_jobs=4,
        )
    finally:
        faiss.omp_set_num_threads(previous_threads)

    np.testing.assert_array_equal(graph_four.indptr, graph_one.indptr)
    np.testing.assert_array_equal(graph_four.indices, graph_one.indices)
    assert metadata_one["faiss_cpu_threads"] == 1
    assert metadata_four["faiss_cpu_threads"] == 4
    assert metadata_four["timing_seconds"]["faiss_query"] >= 0.0
    assert faiss.omp_get_max_threads() == previous_threads


def test_hierarchical_prototypes_preserve_all_memberships_and_sides():
    rng = np.random.default_rng(7)
    values = np.r_[
        rng.normal(0.0, 0.1, size=(120, 4)),
        rng.normal(4.0, 0.1, size=(120, 4)),
    ].astype(np.float32)
    cells = np.repeat(np.arange(24), 10)
    sides = np.where(np.arange(240) < 120, "N", "C")
    labels, metadata, diagnostics = run_hierarchical_prototype_clustering(
        values,
        cells,
        side_labels=sides,
        config={
            "hierarchical_block_nodes": 1_000,
            "hierarchical_local_target_size": 20,
            "hierarchical_global_target_size": 60,
            "random_seed": 7,
        },
    )
    assert labels.shape == (240,)
    assert np.all(labels >= 0)
    assert metadata["prototype_weight_sum"] == 240
    assert metadata["n_local_prototypes"] < 240
    assert set(metadata["global_by_side"]) == {"N", "C"}
    assert diagnostics["n_metaspots"].sum() == 240


def test_partitioned_parquet_keeps_each_cell_in_one_part(tmp_path):
    frame = pd.DataFrame(
        {
            "transcript_id": np.arange(17),
            "cell_id": np.repeat(["a", "b", "c", "d"], [5, 4, 3, 5]),
        }
    )
    destination = tmp_path / "final_data_dataset"
    manifest = _atomic_partitioned_parquet_write(
        frame,
        str(destination),
        target_rows_per_part=5,
    )
    assert manifest["rows"] == len(frame)
    assert (destination / "_SUCCESS").is_file()
    memberships = {}
    for part in destination.glob("part-*.parquet"):
        for cell in pd.read_parquet(part)["cell_id"].unique():
            memberships.setdefault(cell, set()).add(part.name)
    assert all(len(parts) == 1 for parts in memberships.values())


def test_compact_module3_membership_restores_only_when_rows_align(tmp_path):
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "a", "b", "b"],
            "feature_name": ["G1", "G2", "G1", "G2"],
            "metaspot_in_cell": np.asarray([0, 1, 0, 1], dtype=np.int32),
            "metaspot": ["a:0", "a:1", "b:0", "b:1"],
        }
    )
    fingerprints = {"m1": "one", "m2": "two", "m3": "three"}
    _save_module3_membership_checkpoint(
        frame,
        str(tmp_path),
        module_fingerprints=fingerprints,
    )
    upstream = frame.drop(columns=["metaspot_in_cell", "metaspot"])
    restored = _load_module3_membership_checkpoint(
        upstream,
        str(tmp_path),
        module_fingerprints=fingerprints,
    )
    assert restored is not None
    assert restored["metaspot"].astype(str).tolist() == frame["metaspot"].tolist()
    misaligned = upstream.iloc[[0, 2, 1, 3]].reset_index(drop=True)
    assert (
        _load_module3_membership_checkpoint(
            misaligned,
            str(tmp_path),
            module_fingerprints=fingerprints,
        )
        is None
    )
