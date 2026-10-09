"""Bounded-memory hierarchical MetaDomain clustering for large cohorts."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def _cell_aligned_blocks(
    cell_ids: np.ndarray,
    target_nodes: int,
) -> list[np.ndarray]:
    """Return node blocks without splitting a cell across two blocks."""
    codes, _ = pd.factorize(cell_ids, sort=False)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    ends = np.r_[starts[1:], order.size]
    blocks: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    pending_size = 0
    for start, end in zip(starts, ends):
        indices = order[start:end]
        if pending and pending_size + indices.size > target_nodes:
            blocks.append(np.concatenate(pending))
            pending = []
            pending_size = 0
        pending.append(indices)
        pending_size += int(indices.size)
    if pending:
        blocks.append(np.concatenate(pending))
    return blocks


def _fit_predict(
    values: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    batch_size: int,
    max_iter: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_clusters <= 1 or values.shape[0] <= 1:
        return (
            np.zeros(values.shape[0], dtype=np.int32),
            np.mean(values, axis=0, keepdims=True).astype(np.float32),
        )
    model = MiniBatchKMeans(
        n_clusters=min(int(n_clusters), int(values.shape[0])),
        random_state=int(seed),
        batch_size=max(int(batch_size), int(n_clusters)),
        max_iter=max(1, int(max_iter)),
        n_init=1,
        reassignment_ratio=0.01,
    )
    try:
        model.fit(values, sample_weight=sample_weight)
    except TypeError:
        model.fit(values)
    labels = model.predict(values).astype(np.int32, copy=False)
    return labels, np.asarray(model.cluster_centers_, dtype=np.float32)


def run_hierarchical_prototype_clustering(
    features: np.ndarray,
    cell_ids: Sequence[Any],
    *,
    side_labels: Optional[Sequence[Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]:
    """Cluster local cell-aligned blocks, then cluster their prototypes.

    The number of objects presented to the global clustering step is bounded
    by approximately ``n_nodes / local_target_size``.  Every node keeps a
    direct local-prototype membership, so the global result maps back without
    another all-pairs graph or dense co-association matrix.
    """
    options = dict(config or {})
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("Hierarchical MetaDomain features must be two-dimensional")
    groups = np.asarray(cell_ids, dtype=object).reshape(-1)
    if groups.size != values.shape[0]:
        raise ValueError("Hierarchical MetaDomain cell IDs are not aligned")
    sides = (
        np.full(values.shape[0], "A", dtype=object)
        if side_labels is None
        else np.asarray(side_labels, dtype=object).reshape(-1)
    )
    if sides.size != values.shape[0]:
        raise ValueError("Hierarchical MetaDomain side labels are not aligned")

    block_nodes = max(1_000, int(options.get("hierarchical_block_nodes", 100_000)))
    local_target = max(5, int(options.get("hierarchical_local_target_size", 100)))
    global_target = max(5, int(options.get("hierarchical_global_target_size", 400)))
    max_local_clusters = max(
        1, int(options.get("hierarchical_max_local_clusters", 2_000))
    )
    min_global_clusters = max(
        1, int(options.get("hierarchical_min_global_clusters", 2))
    )
    max_global_clusters = max(
        min_global_clusters,
        int(options.get("hierarchical_max_global_clusters", 8_000)),
    )
    batch_size = max(256, int(options.get("hierarchical_batch_size", 8_192)))
    max_iter = max(5, int(options.get("hierarchical_max_iter", 60)))
    seed = int(options.get("random_seed", 42))

    local_membership = np.full(values.shape[0], -1, dtype=np.int64)
    prototypes: list[np.ndarray] = []
    prototype_weights: list[int] = []
    prototype_sides: list[str] = []
    prototype_blocks: list[int] = []
    diagnostics: list[Dict[str, Any]] = []
    prototype_offset = 0

    blocks = _cell_aligned_blocks(groups, block_nodes)
    for block_index, block_indices in enumerate(blocks):
        for side in pd.unique(sides[block_indices]):
            indices = block_indices[sides[block_indices] == side]
            if indices.size == 0:
                continue
            local_k = min(
                max_local_clusters,
                max(1, int(math.ceil(indices.size / local_target))),
            )
            local_labels, centers = _fit_predict(
                values[indices],
                local_k,
                seed=seed + block_index * 97 + len(prototypes),
                batch_size=batch_size,
                max_iter=max_iter,
            )
            present = np.unique(local_labels)
            remap = {int(label): pos for pos, label in enumerate(present)}
            dense = np.asarray([remap[int(x)] for x in local_labels], dtype=np.int64)
            local_membership[indices] = dense + prototype_offset
            for label in present:
                mask = local_labels == label
                # Recompute the observed mean: MiniBatchKMeans centers can be
                # biased by streaming updates and are not exact prototypes.
                prototypes.append(np.mean(values[indices[mask]], axis=0))
                prototype_weights.append(int(mask.sum()))
                prototype_sides.append(str(side))
                prototype_blocks.append(int(block_index))
            diagnostics.append(
                {
                    "block": int(block_index),
                    "side": str(side),
                    "n_metaspots": int(indices.size),
                    "n_local_prototypes": int(present.size),
                }
            )
            prototype_offset += int(present.size)

    if np.any(local_membership < 0):
        raise RuntimeError("Hierarchical MetaDomain left nodes without a prototype")
    prototype_matrix = np.asarray(prototypes, dtype=np.float32)
    weights = np.asarray(prototype_weights, dtype=np.float64)
    prototype_sides_array = np.asarray(prototype_sides, dtype=object)
    prototype_global = np.full(prototype_matrix.shape[0], -1, dtype=np.int32)
    global_offset = 0
    global_by_side: Dict[str, Any] = {}
    for side in pd.unique(prototype_sides_array):
        proto_indices = np.flatnonzero(prototype_sides_array == side)
        side_node_count = int(weights[proto_indices].sum())
        global_k = min(
            int(proto_indices.size),
            max_global_clusters,
            max(min_global_clusters, int(math.ceil(side_node_count / global_target))),
        )
        if proto_indices.size == 1:
            global_k = 1
        labels, _ = _fit_predict(
            prototype_matrix[proto_indices],
            global_k,
            seed=seed + 100_000 + global_offset,
            batch_size=batch_size,
            max_iter=max_iter,
            sample_weight=weights[proto_indices],
        )
        present = np.unique(labels)
        remap = {int(label): pos for pos, label in enumerate(present)}
        dense = np.asarray([remap[int(x)] for x in labels], dtype=np.int32)
        prototype_global[proto_indices] = dense + global_offset
        global_by_side[str(side)] = {
            "n_metaspots": side_node_count,
            "n_local_prototypes": int(proto_indices.size),
            "requested_global_clusters": int(global_k),
            "realized_global_clusters": int(present.size),
        }
        global_offset += int(present.size)

    labels = prototype_global[local_membership]
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "clustering_method": "hierarchical_prototype",
        "n_metaspots": int(values.shape[0]),
        "n_cells": int(pd.unique(groups).size),
        "n_blocks": int(len(blocks)),
        "n_local_prototypes": int(prototype_matrix.shape[0]),
        "n_metadomains": int(np.unique(labels).size),
        "block_nodes": int(block_nodes),
        "local_target_size": int(local_target),
        "global_target_size": int(global_target),
        "global_by_side": global_by_side,
        "prototype_weight_sum": int(weights.sum()),
        "prototype_block_count": int(np.unique(prototype_blocks).size),
    }
    return labels.astype(np.int32, copy=False), metadata, pd.DataFrame(diagnostics)
