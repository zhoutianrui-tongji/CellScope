"""Resolve an explicit, provenance-friendly execution plan by realized scale."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: int
    requested_profile: str
    effective_profile: str
    n_metaspots: int
    n_transcripts: int
    exact_max_metaspots: int
    sampled_max_metaspots: int
    memory_budget_gb: float
    partition_rows: int
    knn_backend: str
    dgi_training_mode: str
    metadomain_method: str
    partitioned_annotation: bool
    full_diagnostics: bool
    faiss_available: bool
    pynndescent_available: bool
    torch_geometric_available: bool
    neighbor_sampler_available: bool
    estimated_feature_gb: float
    estimated_graph_gb: float
    estimated_embedding_gb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve_execution_plan(
    config: Mapping[str, Any],
    *,
    n_metaspots: int,
    n_transcripts: int = 0,
    capability_overrides: Optional[Mapping[str, bool]] = None,
) -> ExecutionPlan:
    """Resolve exact/sampled/streaming execution without silent fallback."""
    if n_metaspots < 1:
        raise ValueError("Execution planning requires at least one MetaSpot")
    root = dict(config or {})
    options = root.get("execution", {}) or {}
    if not isinstance(options, Mapping):
        raise ValueError("execution must be a mapping")

    requested_raw = str(options.get("profile", "auto")).strip().lower()
    aliases = {
        "auto": "auto",
        "exact": "exact",
        "full_batch": "exact",
        "full-batch": "exact",
        "sampled": "sampled",
        "neighbor_sampled": "sampled",
        "neighbor-sampled": "sampled",
        "streaming": "streaming",
        "large": "streaming",
    }
    if requested_raw not in aliases:
        raise ValueError(
            "execution.profile must be one of "
            "{'auto','exact','sampled','streaming'}"
        )
    requested = aliases[requested_raw]
    exact_max = int(options.get("exact_max_metaspots", 200_000))
    sampled_max = int(options.get("sampled_max_metaspots", 2_000_000))
    memory_budget = float(options.get("memory_budget_gb", 64.0))
    partition_rows = int(options.get("partition_rows", 1_000_000))
    fail_on_fallback = bool(options.get("fail_on_backend_fallback", True))
    if (
        exact_max < 1
        or sampled_max < exact_max
        or memory_budget <= 0
        or partition_rows < 1
    ):
        raise ValueError(
            "execution thresholds must satisfy 1 <= exact_max_metaspots <= "
            "sampled_max_metaspots, memory_budget_gb > 0, and "
            "partition_rows >= 1"
        )

    if requested == "auto":
        if n_metaspots <= exact_max:
            effective = "exact"
        elif n_metaspots <= sampled_max:
            effective = "sampled"
        else:
            effective = "streaming"
    else:
        effective = requested
    if effective == "exact" and n_metaspots > exact_max:
        raise ValueError(
            f"execution.profile='exact' received {n_metaspots:,} MetaSpots, "
            f"above execution.exact_max_metaspots={exact_max:,}"
        )

    capabilities = {
        "faiss": _module_available("faiss"),
        "pynndescent": _module_available("pynndescent"),
        "torch_geometric": _module_available("torch_geometric"),
        "pyg_lib": _module_available("pyg_lib"),
        "torch_sparse": _module_available("torch_sparse"),
    }
    if capability_overrides:
        capabilities.update(
            {str(key): bool(value) for key, value in capability_overrides.items()}
        )

    module7 = root.get("module7", {}) or {}
    requested_backend = str(module7.get("knn_backend", "auto")).strip().lower()
    if requested_backend == "auto":
        if effective == "exact":
            backend = "exact"
        elif capabilities["faiss"]:
            backend = "faiss_ivfpq" if effective == "streaming" else "faiss_hnsw"
        elif effective == "sampled" and capabilities["pynndescent"]:
            backend = "pynndescent"
        else:
            backend = "unavailable"
    else:
        backend = requested_backend.replace("-", "_")
    valid_backends = {"exact", "pynndescent", "faiss_hnsw", "faiss_ivfpq"}
    if backend == "unavailable":
        if effective == "streaming":
            raise RuntimeError(
                "execution.profile='streaming' requires FAISS IVFPQ, but "
                "the faiss module is unavailable"
            )
        raise RuntimeError(
            "No bounded large-scale kNN backend is installed; install "
            "FAISS or PyNNDescent"
        )
    if backend not in valid_backends:
        raise ValueError(
            "module7.knn_backend must be one of "
            "{'auto','exact','pynndescent','faiss_hnsw','faiss_ivfpq'}"
        )
    backend_available = (
        backend == "exact"
        or (backend == "pynndescent" and capabilities["pynndescent"])
        or (backend.startswith("faiss_") and capabilities["faiss"])
    )
    if not backend_available:
        message = f"Requested large-scale kNN backend {backend!r} is unavailable"
        if fail_on_fallback:
            raise RuntimeError(message)
        if capabilities["pynndescent"]:
            backend = "pynndescent"
        else:
            raise RuntimeError(message + "; no bounded fallback is installed")
    if effective == "exact" and backend == "faiss_ivfpq":
        raise ValueError(
            "execution.profile='exact' cannot use the streaming-only "
            "FAISS IVFPQ backend"
        )
    if effective != "exact" and backend == "exact":
        raise ValueError(
            "Large sampled/streaming profiles cannot use exact kNN"
        )
    if effective == "streaming" and backend != "faiss_ivfpq":
        raise RuntimeError(
            "execution.profile='streaming' requires FAISS IVFPQ; "
            "PyNNDescent/exact kNN are not accepted for this scale"
        )

    module8 = root.get("module8", {}) or {}
    requested_training_mode = str(
        module8.get("training_mode", "auto")
    ).strip().lower().replace("-", "_")
    if requested_training_mode not in {
        "auto",
        "full_batch",
        "neighbor_sampled",
    }:
        raise ValueError(
            "module8.training_mode must be one of "
            "{'auto','full_batch','neighbor_sampled'}"
        )
    training_mode = (
        "full_batch" if effective == "exact" else "neighbor_sampled"
    ) if requested_training_mode == "auto" else requested_training_mode
    if effective != "exact" and training_mode == "full_batch":
        raise ValueError(
            "Large sampled/streaming profiles cannot use full-batch DGI; "
            "set module8.training_mode='auto' or 'neighbor_sampled'"
        )
    neighbor_sampler_available = bool(
        capabilities["torch_geometric"]
        and (capabilities["pyg_lib"] or capabilities["torch_sparse"])
    )
    if training_mode == "neighbor_sampled" and not neighbor_sampler_available:
        raise RuntimeError(
            "Sampled MetaSpot-DGI requires torch_geometric plus pyg-lib or "
            "torch-sparse; refusing full-batch fallback"
        )

    n = float(n_metaspots)
    feature_gb = n * 32.0 * 4.0 / (1024.0 ** 3)
    graph_gb = n * 21.0 * 8.0 * 2.0 / (1024.0 ** 3)
    embedding_gb = n * 32.0 * 4.0 * 3.0 / (1024.0 ** 3)

    module5 = root.get("module5", {}) or {}
    requested_metadomain = str(
        module5.get("large_scale_method", "auto")
    ).strip().lower().replace("-", "_")
    if requested_metadomain in {"hierarchical", "prototype"}:
        requested_metadomain = "hierarchical_prototype"
    if requested_metadomain in {"exact"}:
        requested_metadomain = "consensus_leiden"
    if requested_metadomain not in {
        "auto",
        "consensus_leiden",
        "hierarchical_prototype",
    }:
        raise ValueError(
            "module5.large_scale_method must be one of "
            "{'auto','consensus_leiden','hierarchical_prototype'}"
        )
    estimate_k = str(module5.get("estimate_k", "auto")).strip().lower()
    if estimate_k == "fixed":
        if requested_metadomain != "auto":
            raise ValueError(
                "module5.estimate_k='fixed' requires "
                "module5.large_scale_method='auto'"
            )
        metadomain_method = "minibatch_kmeans"
    else:
        metadomain_method = (
            "consensus_leiden"
            if effective == "exact"
            else "hierarchical_prototype"
        ) if requested_metadomain == "auto" else requested_metadomain
    if effective != "exact" and metadomain_method == "consensus_leiden":
        raise ValueError(
            "Large sampled/streaming profiles cannot run repeated global "
            "consensus Leiden; use module5.large_scale_method='auto' or "
            "'hierarchical_prototype'"
        )

    return ExecutionPlan(
        schema_version=1,
        requested_profile=requested,
        effective_profile=effective,
        n_metaspots=int(n_metaspots),
        n_transcripts=int(n_transcripts),
        exact_max_metaspots=exact_max,
        sampled_max_metaspots=sampled_max,
        memory_budget_gb=memory_budget,
        partition_rows=partition_rows,
        knn_backend=backend,
        dgi_training_mode=training_mode,
        metadomain_method=metadomain_method,
        partitioned_annotation=effective == "streaming",
        full_diagnostics=effective != "streaming",
        faiss_available=capabilities["faiss"],
        pynndescent_available=capabilities["pynndescent"],
        torch_geometric_available=capabilities["torch_geometric"],
        neighbor_sampler_available=neighbor_sampler_available,
        estimated_feature_gb=float(feature_gb),
        estimated_graph_gb=float(graph_gb),
        estimated_embedding_gb=float(embedding_gb),
    )
