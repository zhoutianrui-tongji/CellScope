"""Stability-selected sparse consensus Leiden clustering."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph


def build_mutual_knn_graph(
    values: np.ndarray,
    *,
    n_neighbors: int,
    n_jobs: int = 1,
) -> sp.csr_matrix:
    """Build a mutual-kNN graph and reconnect only isolated observations."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("Consensus Leiden requires at least two observations")
    if not np.isfinite(matrix).all():
        raise ValueError("Consensus Leiden representation contains non-finite values")
    k = min(max(2, int(n_neighbors)), matrix.shape[0] - 1)
    directed = kneighbors_graph(
        matrix,
        n_neighbors=k,
        mode="distance",
        include_self=False,
        # The brute-force distance kernel is vectorized; joblib's chunk
        # scheduler is slower here and can multiply BLAS pools. Keep the
        # exact mutual-kNN graph deterministic and single-scheduled.
        n_jobs=1,
    ).tocsr()
    # Convert distance to a bounded local similarity before taking reciprocity.
    positive_distances = directed.data[directed.data > 0]
    scale = (
        float(np.median(positive_distances))
        if positive_distances.size
        else 1.0
    )
    directed.data = np.exp(
        -directed.data / max(scale, 1e-8)
    ).astype(np.float32)
    mutual = directed.minimum(directed.T).tocsr()
    mutual.eliminate_zeros()

    degree = np.asarray(mutual.getnnz(axis=1)).ravel()
    isolated = np.flatnonzero(degree == 0)
    if isolated.size:
        # A strict mutual graph can isolate rare structures. Add only the
        # strongest directed edge for each isolated node, symmetrically.
        rows = []
        cols = []
        data = []
        for row in isolated:
            start, stop = directed.indptr[row], directed.indptr[row + 1]
            if start == stop:
                continue
            local = int(start + np.argmax(directed.data[start:stop]))
            rows.extend((int(row), int(directed.indices[local])))
            cols.extend((int(directed.indices[local]), int(row)))
            data.extend((float(directed.data[local]),) * 2)
        if rows:
            repair = sp.csr_matrix(
                (np.asarray(data, dtype=np.float32), (rows, cols)),
                shape=mutual.shape,
            )
            mutual = mutual.maximum(repair).tocsr()
    mutual.setdiag(0)
    mutual.eliminate_zeros()
    mutual.sort_indices()
    if mutual.nnz == 0:
        raise ValueError("Consensus Leiden mutual-kNN graph contains no edges")
    return mutual


def _igraph_from_sparse(adjacency: sp.csr_matrix):
    import igraph as ig

    graph = adjacency.maximum(adjacency.T).tocsr().astype(np.float64)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return ig.Graph.Weighted_Adjacency(
        graph,
        mode="undirected",
        attr="weight",
        loops=False,
    )


def _leiden_labels(
    graph,
    *,
    resolution: float,
    seed: int,
    n_iterations: int,
) -> np.ndarray:
    import leidenalg

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        n_iterations=int(n_iterations),
        seed=int(seed),
    )
    return np.asarray(partition.membership, dtype=np.int32)


def _partition_diagnostics(
    labels: np.ndarray,
    cell_codes: np.ndarray,
    *,
    min_domain_size: int,
    min_recurrence_cells: int,
) -> Dict[str, float]:
    cluster_ids, inverse, counts = np.unique(
        labels,
        return_inverse=True,
        return_counts=True,
    )
    small = counts < max(1, int(min_domain_size))
    small_fraction = float(np.mean(small[inverse])) if labels.size else 0.0
    cells = np.asarray(cell_codes, dtype=np.int64).reshape(-1)
    if cells.size != labels.size or (cells.size and cells.min() < 0):
        raise ValueError("Partition diagnostic cell codes are not aligned")
    n_cells = int(cells.max()) + 1 if cells.size else 0
    if cluster_ids.size and n_cells:
        # Count unique (cluster, cell) pairs once. The former per-cluster
        # boolean scans were O(n_clusters * n_metaspots) for every seed and
        # resolution.
        pair_codes = inverse.astype(np.int64) * np.int64(n_cells) + cells
        unique_pairs = np.unique(pair_codes)
        recurrence = np.bincount(
            unique_pairs // np.int64(n_cells),
            minlength=cluster_ids.size,
        ).astype(np.int32, copy=False)
    else:
        recurrence = np.zeros(cluster_ids.size, dtype=np.int32)
    nonrecurrent = recurrence < max(1, int(min_recurrence_cells))
    nonrecurrent_fraction = (
        float(np.mean(nonrecurrent[inverse])) if labels.size else 0.0
    )
    return {
        "n_clusters": float(cluster_ids.size),
        "average_size": float(labels.size / max(1, cluster_ids.size)),
        "small_cluster_fraction": small_fraction,
        "nonrecurrent_fraction": nonrecurrent_fraction,
        "median_recurrence_cells": float(np.median(recurrence)),
    }


def run_stability_consensus_leiden(
    values: np.ndarray,
    cell_ids: Sequence[Any],
    *,
    config: Mapping[str, Any],
    n_jobs: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select a stable Leiden resolution and consolidate seed assignments.

    Co-association is evaluated only on sparse mutual-kNN edges, avoiding the
    quadratic dense consensus matrix used by classical consensus clustering.
    """
    matrix = np.asarray(values, dtype=np.float32)
    cells = np.asarray(cell_ids).astype(str)
    if cells.size != matrix.shape[0]:
        raise ValueError("Consensus Leiden cell_ids are not aligned")
    _, cell_codes = np.unique(cells, return_inverse=True)
    cell_codes = cell_codes.astype(np.int64, copy=False)
    resolutions_raw = config.get(
        "consensus_resolutions",
        [1.0, 2.0, 4.0, 6.0, 8.0],
    )
    seeds_raw = config.get("consensus_seeds", [17, 42, 89])
    if isinstance(resolutions_raw, (int, float)):
        resolutions_raw = [resolutions_raw]
    if isinstance(seeds_raw, (int, float)):
        seeds_raw = [seeds_raw]
    resolutions = sorted({float(value) for value in resolutions_raw})
    seeds = [int(value) for value in seeds_raw]
    if not resolutions or min(resolutions) <= 0.0:
        raise ValueError("module5.consensus_resolutions must be positive")
    if len(seeds) < 2:
        raise ValueError("module5.consensus_seeds requires at least two seeds")
    n_neighbors = int(config.get("consensus_neighbors", 20))
    n_iterations = int(config.get("consensus_leiden_iterations", 2))
    min_average_size = float(
        config.get("consensus_min_average_size", 100.0)
    )
    max_average_size = float(
        config.get("consensus_max_average_size", 800.0)
    )
    min_domain_size = int(config.get("min_domain_size", 20))
    min_recurrence_cells = int(
        config.get("consensus_min_recurrence_cells", 2)
    )
    max_small_fraction = float(
        config.get("consensus_max_small_cluster_fraction", 0.10)
    )
    max_nonrecurrent_fraction = float(
        config.get("consensus_max_nonrecurrent_fraction", 0.10)
    )
    agreement_threshold = float(
        config.get("consensus_edge_agreement", 2.0 / 3.0)
    )
    min_stability = float(config.get("consensus_min_stability", 0.55))
    stability_tolerance = float(
        config.get("consensus_stability_tolerance", 0.10)
    )
    final_seed = int(config.get("random_seed", 42))
    if n_neighbors < 2 or n_iterations == 0:
        raise ValueError("Consensus Leiden neighbors/iterations are invalid")
    if not 0.0 < agreement_threshold <= 1.0:
        raise ValueError("module5.consensus_edge_agreement must be in (0, 1]")
    if not -1.0 <= min_stability <= 1.0:
        raise ValueError("module5.consensus_min_stability must be in [-1, 1]")
    if not 0.0 <= stability_tolerance <= 2.0:
        raise ValueError(
            "module5.consensus_stability_tolerance must be in [0, 2]"
        )
    if not 0.0 <= max_small_fraction <= 1.0:
        raise ValueError(
            "module5.consensus_max_small_cluster_fraction must be in [0, 1]"
        )
    if not 0.0 <= max_nonrecurrent_fraction <= 1.0:
        raise ValueError(
            "module5.consensus_max_nonrecurrent_fraction must be in [0, 1]"
        )
    if min_average_size <= 0 or max_average_size < min_average_size:
        raise ValueError("Consensus Leiden average-size bounds are invalid")

    adjacency = build_mutual_knn_graph(
        matrix,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )
    graph = _igraph_from_sparse(adjacency)
    candidates = []
    runs_by_resolution: Dict[float, list[np.ndarray]] = {}
    for resolution in resolutions:
        runs = [
            _leiden_labels(
                graph,
                resolution=resolution,
                seed=seed,
                n_iterations=n_iterations,
            )
            for seed in seeds
        ]
        runs_by_resolution[resolution] = runs
        pairwise_stability = []
        stability_sums = np.zeros(len(runs), dtype=np.float64)
        for first, second in combinations(range(len(runs)), 2):
            value = float(adjusted_rand_score(runs[first], runs[second]))
            pairwise_stability.append(value)
            stability_sums[first] += value
            stability_sums[second] += value
        medoid_index = int(np.argmax(stability_sums))
        diagnostics = _partition_diagnostics(
            runs[medoid_index],
            cell_codes,
            min_domain_size=min_domain_size,
            min_recurrence_cells=min_recurrence_cells,
        )
        stability = float(np.mean(pairwise_stability))
        feasible = bool(
            min_average_size
            <= diagnostics["average_size"]
            <= max_average_size
            and diagnostics["small_cluster_fraction"]
            <= max_small_fraction
            and diagnostics["nonrecurrent_fraction"]
            <= max_nonrecurrent_fraction
        )
        score = (
            stability
            - diagnostics["small_cluster_fraction"]
            - diagnostics["nonrecurrent_fraction"]
        )
        candidates.append(
            {
                "resolution": float(resolution),
                "stability_ari": stability,
                "feasible": feasible,
                "selection_score": float(score),
                **diagnostics,
            }
        )

    feasible_candidates = [item for item in candidates if item["feasible"]]
    selected_reason = "best_penalized_stability_no_feasible_candidate"
    stability_floor = float("nan")
    if feasible_candidates:
        best_feasible_stability = max(
            item["stability_ari"] for item in feasible_candidates
        )
        stability_floor = max(
            min_stability,
            best_feasible_stability - stability_tolerance,
        )
        stable_fine_candidates = [
            item
            for item in feasible_candidates
            if item["stability_ari"] >= stability_floor
        ]
        if stable_fine_candidates:
            # A clustering tool should not systematically choose a trivial
            # coarse partition merely because its ARI is a few thousandths
            # higher. Among candidates operationally close to the best stable
            # solution, retain the one with the most reproducible structure.
            selected = max(
                stable_fine_candidates,
                key=lambda item: (
                    item["n_clusters"],
                    item["stability_ari"],
                    item["resolution"],
                ),
            )
            selected_reason = "finest_within_stability_tolerance"
        else:
            selected = max(
                feasible_candidates,
                key=lambda item: (
                    item["selection_score"],
                    item["stability_ari"],
                ),
            )
            selected_reason = "best_feasible_below_stability_floor"
    else:
        selected = max(
            candidates,
            key=lambda item: (
                item["selection_score"],
                item["stability_ari"],
                item["resolution"],
            ),
        )
    selected_resolution = float(selected["resolution"])
    selected_runs = runs_by_resolution[selected_resolution]

    upper = sp.triu(adjacency, k=1, format="coo")
    agreement = np.zeros(upper.nnz, dtype=np.float32)
    for labels in selected_runs:
        agreement += (labels[upper.row] == labels[upper.col]).astype(np.float32)
    agreement /= float(len(selected_runs))
    keep = agreement >= agreement_threshold
    consensus = sp.csr_matrix(
        (
            upper.data[keep] * agreement[keep],
            (upper.row[keep], upper.col[keep]),
        ),
        shape=adjacency.shape,
        dtype=np.float32,
    )
    consensus = (consensus + consensus.T).tocsr()
    consensus.eliminate_zeros()
    if consensus.nnz == 0:
        consensus = adjacency
    consensus_graph = _igraph_from_sparse(consensus)
    labels = _leiden_labels(
        consensus_graph,
        resolution=selected_resolution,
        seed=final_seed,
        n_iterations=n_iterations,
    )
    final_diagnostics = _partition_diagnostics(
        labels,
        cell_codes,
        min_domain_size=min_domain_size,
        min_recurrence_cells=min_recurrence_cells,
    )
    metadata: Dict[str, Any] = {
        "method": "stability_consensus_leiden",
        "selected_resolution": selected_resolution,
        "selection_reason": selected_reason,
        "minimum_stability": float(min_stability),
        "stability_tolerance": float(stability_tolerance),
        "selection_stability_floor": float(stability_floor),
        "n_neighbors": int(n_neighbors),
        "seeds": np.asarray(seeds, dtype=np.int32),
        "edge_agreement_threshold": float(agreement_threshold),
        "mutual_graph_nnz": int(adjacency.nnz),
        "consensus_graph_nnz": int(consensus.nnz),
        "candidate_resolution": np.asarray(
            [item["resolution"] for item in candidates], dtype=np.float32
        ),
        "candidate_stability_ari": np.asarray(
            [item["stability_ari"] for item in candidates], dtype=np.float32
        ),
        "candidate_n_clusters": np.asarray(
            [item["n_clusters"] for item in candidates], dtype=np.int32
        ),
        "candidate_average_size": np.asarray(
            [item["average_size"] for item in candidates], dtype=np.float32
        ),
        "candidate_small_cluster_fraction": np.asarray(
            [item["small_cluster_fraction"] for item in candidates],
            dtype=np.float32,
        ),
        "candidate_nonrecurrent_fraction": np.asarray(
            [item["nonrecurrent_fraction"] for item in candidates],
            dtype=np.float32,
        ),
        "candidate_feasible": np.asarray(
            [item["feasible"] for item in candidates], dtype=bool
        ),
        "final_n_clusters": int(final_diagnostics["n_clusters"]),
        "final_average_size": float(final_diagnostics["average_size"]),
        "final_small_cluster_fraction": float(
            final_diagnostics["small_cluster_fraction"]
        ),
        "final_nonrecurrent_fraction": float(
            final_diagnostics["nonrecurrent_fraction"]
        ),
        "final_median_recurrence_cells": float(
            final_diagnostics["median_recurrence_cells"]
        ),
    }
    return labels.astype(np.int32, copy=False), metadata
