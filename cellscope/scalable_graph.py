"""Bounded-memory exact and approximate kNN graph construction."""

from __future__ import annotations

import math
import os
import tempfile
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


def _filter_candidates(
    candidate_indices: np.ndarray,
    *,
    row_ids: np.ndarray,
    n_neighbors: int,
    excluded_groups: Optional[np.ndarray],
) -> np.ndarray:
    """Keep the first k valid candidates per row without a Python row loop."""
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    rows = np.asarray(row_ids, dtype=np.int64)
    selected = np.full((rows.size, n_neighbors), -1, dtype=np.int64)
    counts = np.zeros(rows.size, dtype=np.int32)
    local_rows = np.arange(rows.size, dtype=np.int64)
    for column in range(candidates.shape[1]):
        candidate = candidates[:, column]
        valid = (candidate >= 0) & (candidate != rows) & (counts < n_neighbors)
        if excluded_groups is not None:
            safe_candidate = np.clip(candidate, 0, excluded_groups.size - 1)
            valid &= excluded_groups[safe_candidate] != excluded_groups[rows]
        accepting = local_rows[valid]
        if accepting.size:
            selected[accepting, counts[accepting]] = candidate[valid]
            counts[accepting] += 1
        if np.all(counts >= n_neighbors):
            break
    return selected


def _selected_to_graph(selected: np.ndarray, n_nodes: int) -> sp.csr_matrix:
    valid = selected >= 0
    row = np.repeat(
        np.arange(selected.shape[0], dtype=np.int64),
        selected.shape[1],
    )[valid.ravel()]
    col = selected.ravel()[valid.ravel()]
    graph = sp.csr_matrix(
        (np.ones(row.size, dtype=np.float32), (row, col)),
        shape=(n_nodes, n_nodes),
    )
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _query_exact(
    matrix: np.ndarray,
    query_k: int,
    query_chunk_size: int,
    n_jobs: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(
        n_neighbors=query_k,
        metric="euclidean",
        algorithm="auto",
        n_jobs=max(1, int(n_jobs)),
    ).fit(matrix)
    for start in range(0, matrix.shape[0], query_chunk_size):
        stop = min(matrix.shape[0], start + query_chunk_size)
        indices = model.kneighbors(
            matrix[start:stop],
            n_neighbors=query_k,
            return_distance=False,
        )
        yield start, np.asarray(indices, dtype=np.int64)


def _query_pynndescent(
    matrix: np.ndarray,
    query_k: int,
    query_chunk_size: int,
    n_jobs: int,
    random_seed: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    # Numba may already have been imported by Scanpy/anndata before this
    # backend is selected. In that case setting NUMBA_CACHE_DIR alone is too
    # late; update the live Numba config as well so read-only site-packages do
    # not make PyNNDescent fail during decorator initialization.
    cache_dir = os.environ.get(
        "NUMBA_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "cellscope-cache", "numba"),
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("NUMBA_CACHE_DIR", cache_dir)
        import numba

        if not getattr(numba.config, "CACHE_DIR", ""):
            numba.config.CACHE_DIR = cache_dir
    except Exception:
        # PyNNDescent will provide the actionable import/cache error below if
        # the host genuinely cannot create a writable cache directory.
        pass
    from pynndescent import NNDescent

    index = NNDescent(
        matrix,
        n_neighbors=min(
            matrix.shape[0] - 1,
            max(15, min(query_k, 64)),
        ),
        metric="euclidean",
        random_state=int(random_seed),
        n_jobs=max(1, int(n_jobs)),
        low_memory=True,
        compressed=False,
        verbose=False,
    )
    index.prepare()
    for start in range(0, matrix.shape[0], query_chunk_size):
        stop = min(matrix.shape[0], start + query_chunk_size)
        indices, _ = index.query(matrix[start:stop], k=query_k)
        yield start, np.asarray(indices, dtype=np.int64)


def _pq_subvectors(dimension: int, requested: int) -> int:
    upper = min(max(1, int(requested)), dimension)
    for value in range(upper, 0, -1):
        if dimension % value == 0:
            return value
    return 1


def _query_faiss(
    matrix: np.ndarray,
    *,
    backend: str,
    query_k: int,
    query_chunk_size: int,
    train_sample: int,
    hnsw_m: int,
    nprobe: int,
    pq_subvectors: int,
    pq_bits: int,
    random_seed: int,
    use_gpu: bool,
    gpu_device: int,
    n_jobs: int,
    timing_seconds: Dict[str, float],
) -> Iterable[Tuple[int, np.ndarray]]:
    import faiss

    values = np.ascontiguousarray(matrix, dtype=np.float32)
    dimension = int(values.shape[1])
    previous_threads = int(faiss.omp_get_max_threads())
    query_threads = max(1, int(n_jobs))
    # HNSW construction is deliberately single-threaded: FAISS parallel add is
    # faster but changes the approximate graph. Read-only search is deterministic
    # for a fixed index and can safely use the configured runtime worker count.
    faiss.omp_set_num_threads(1)
    try:
        build_started = time.perf_counter()
        if backend == "faiss_hnsw":
            index = faiss.IndexHNSWFlat(dimension, max(4, int(hnsw_m)))
            index.hnsw.efConstruction = max(40, 2 * query_k)
            index.hnsw.efSearch = max(64, 2 * query_k)
        else:
            nlist = int(
                max(64, min(65536, round(4.0 * math.sqrt(values.shape[0]))))
            )
            nlist = min(nlist, max(1, values.shape[0] // 40))
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(
                quantizer,
                dimension,
                max(1, nlist),
                _pq_subvectors(dimension, pq_subvectors),
                max(4, int(pq_bits)),
            )
            sample_n = min(values.shape[0], max(10_000, int(train_sample)))
            rng = np.random.default_rng(int(random_seed))
            sample = (
                values
                if sample_n == values.shape[0]
                else values[rng.choice(values.shape[0], sample_n, replace=False)]
            )
            index.train(np.ascontiguousarray(sample, dtype=np.float32))
            index.nprobe = min(max(1, int(nprobe)), max(1, nlist))
        gpu_resources = None
        if use_gpu:
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(
                gpu_resources,
                int(gpu_device),
                index,
            )
        index.add(values)
        timing_seconds["faiss_index_build"] = (
            time.perf_counter() - build_started
        )
        if not use_gpu:
            faiss.omp_set_num_threads(query_threads)
        query_seconds = 0.0
        for start in range(0, values.shape[0], query_chunk_size):
            stop = min(values.shape[0], start + query_chunk_size)
            query_started = time.perf_counter()
            _, indices = index.search(values[start:stop], query_k)
            query_seconds += time.perf_counter() - query_started
            yield start, np.asarray(indices, dtype=np.int64)
        timing_seconds["faiss_query"] = query_seconds
    finally:
        faiss.omp_set_num_threads(previous_threads)


def build_knn_connectivity(
    values: np.ndarray,
    *,
    n_neighbors: int,
    backend: str = "exact",
    excluded_groups: Optional[Sequence[Any]] = None,
    candidate_multiplier: int = 4,
    query_chunk_size: int = 100_000,
    train_sample: int = 1_000_000,
    hnsw_m: int = 32,
    nprobe: int = 32,
    pq_subvectors: int = 8,
    pq_bits: int = 8,
    faiss_use_gpu: Any = "auto",
    faiss_gpu_device: int = 0,
    random_seed: int = 42,
    n_jobs: int = 1,
) -> Tuple[sp.csr_matrix, Dict[str, Any]]:
    """Build a symmetric connectivity graph using a selected kNN backend."""
    function_started = time.perf_counter()
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("kNN values must have shape (n>=2, d>=1)")
    if not np.isfinite(matrix).all():
        raise ValueError("kNN values contain non-finite values")
    n_nodes = int(matrix.shape[0])
    k = min(max(1, int(n_neighbors)), n_nodes - 1)
    multiplier = max(1, int(candidate_multiplier))
    normalized_backend = str(backend).strip().lower().replace("-", "_")
    query_k = min(
        n_nodes,
        k + 1 if normalized_backend == "exact" else k * multiplier + 1,
    )
    groups = None
    available_degree = np.full(n_nodes, k, dtype=np.int64)
    if excluded_groups is not None:
        groups = np.asarray(excluded_groups).astype(str)
        if groups.size != n_nodes:
            raise ValueError("excluded_groups are not aligned to kNN values")
        _, group_codes = np.unique(groups, return_inverse=True)
        group_sizes = np.bincount(group_codes)
        available_degree = np.minimum(
            k,
            n_nodes - group_sizes[group_codes],
        )
        # At most ``group_size`` returned candidates can belong to the query
        # node's excluded group (including the node itself). Asking for the
        # largest group plus k therefore guarantees k valid candidates where
        # k are globally available, independent of how tightly cells cluster.
        query_k = min(
            n_nodes,
            max(query_k, int(group_sizes.max()) + k),
        )
    requested_chunk_size = max(1, int(query_chunk_size))
    # Bound the temporary candidate matrix even when one cell contains many
    # MetaSpots and the cross-cell exclusion requires a wide query.
    chunk_size = min(
        requested_chunk_size,
        max(1, 5_000_000 // max(1, query_k)),
    )
    faiss_gpu_available = False
    faiss_gpu_effective = False
    timing_seconds: Dict[str, float] = {}
    if normalized_backend == "exact":
        queried = _query_exact(matrix, query_k, chunk_size, n_jobs)
    elif normalized_backend == "pynndescent":
        queried = _query_pynndescent(
            matrix,
            query_k,
            chunk_size,
            n_jobs,
            random_seed,
        )
    elif normalized_backend in {"faiss_hnsw", "faiss_ivfpq"}:
        import faiss

        faiss_gpu_available = bool(
            hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "get_num_gpus")
            and int(faiss.get_num_gpus()) > int(faiss_gpu_device)
        )
        requested_gpu = str(faiss_use_gpu).strip().lower()
        if requested_gpu not in {"auto", "true", "false", "1", "0"}:
            raise ValueError("faiss_use_gpu must be auto, true, or false")
        wants_gpu = requested_gpu in {"auto", "true", "1"}
        # FAISS GPU does not implement the HNSW index. IVFPQ is the large
        # streaming path and can move training/add/query to one CUDA device.
        faiss_gpu_effective = bool(
            normalized_backend == "faiss_ivfpq"
            and wants_gpu
            and faiss_gpu_available
        )
        if requested_gpu in {"true", "1"} and not faiss_gpu_effective:
            raise RuntimeError(
                "FAISS GPU was required but is unavailable or incompatible "
                f"with backend {normalized_backend!r}"
            )
        queried = _query_faiss(
            matrix,
            backend=normalized_backend,
            query_k=query_k,
            query_chunk_size=chunk_size,
            train_sample=train_sample,
            hnsw_m=hnsw_m,
            nprobe=nprobe,
            pq_subvectors=pq_subvectors,
            pq_bits=pq_bits,
            random_seed=random_seed,
            use_gpu=faiss_gpu_effective,
            gpu_device=faiss_gpu_device,
            n_jobs=n_jobs,
            timing_seconds=timing_seconds,
        )
    else:
        raise ValueError(f"Unsupported kNN backend: {backend!r}")

    selected = np.full((n_nodes, k), -1, dtype=np.int64)
    filter_seconds = 0.0
    for start, candidates in queried:
        stop = start + candidates.shape[0]
        filter_started = time.perf_counter()
        selected[start:stop] = _filter_candidates(
            candidates,
            row_ids=np.arange(start, stop, dtype=np.int64),
            n_neighbors=k,
            excluded_groups=groups,
        )
        filter_seconds += time.perf_counter() - filter_started
    timing_seconds["candidate_filter"] = filter_seconds
    materialize_started = time.perf_counter()
    graph = _selected_to_graph(selected, n_nodes)
    timing_seconds["graph_materialize"] = (
        time.perf_counter() - materialize_started
    )
    timing_seconds["total"] = time.perf_counter() - function_started
    directed_degree = np.sum(selected >= 0, axis=1)
    below_available = directed_degree < available_degree
    metadata = {
        "backend": normalized_backend,
        "n_nodes": n_nodes,
        "n_neighbors_requested": int(n_neighbors),
        "n_neighbors_effective": k,
        "candidate_multiplier": multiplier,
        "query_k": query_k,
        "query_chunk_size_requested": requested_chunk_size,
        "query_chunk_size_effective": chunk_size,
        "excluded_same_group": groups is not None,
        "nodes_below_requested_degree": int(np.sum(directed_degree < k)),
        "nodes_below_available_degree": int(np.sum(below_available)),
        "directed_degree_min": int(directed_degree.min()),
        "directed_degree_p05": float(np.quantile(directed_degree, 0.05)),
        "directed_degree_median": float(np.median(directed_degree)),
        "directed_degree_max": int(directed_degree.max()),
        "available_degree_min": int(available_degree.min()),
        "isolated_nodes": int(np.sum(directed_degree == 0)),
        "isolated_fraction": float(np.mean(directed_degree == 0)),
        "symmetric_nnz": int(graph.nnz),
        "faiss_gpu_available": bool(faiss_gpu_available),
        "faiss_gpu_effective": bool(faiss_gpu_effective),
        "faiss_gpu_device": int(faiss_gpu_device),
        "faiss_cpu_threads": int(
            0
            if faiss_gpu_effective or not normalized_backend.startswith("faiss_")
            else max(1, int(n_jobs))
        ),
        "timing_seconds": {
            key: float(value) for key, value in timing_seconds.items()
        },
    }
    return graph, metadata
