"""Cell-Aware Deep Multiplex Graph Infomax (CA-DMGI) training utilities.

This module intentionally contains no AnnData or pipeline I/O.  It trains one
relation-specific residual GCN for the cross-cell expression graph and one for
the within-cell spatial graph, then learns a bounded node-wise gate that forms
the consensus MetaSpot representation used by downstream clustering.
"""

from __future__ import annotations

import copy
import math
import os
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp


def resolve_dgi_device(torch_module: Any, requested: Any = "auto") -> Any:
    """Resolve an explicit DGI device request without silent CUDA fallback."""
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("module8.dgi_device must be 'auto', 'cpu', or 'cuda'")
    if normalized == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError(
                "module8.dgi_device='cuda' was requested, but this PyTorch "
                "build/runtime has no available CUDA device"
            )
        return torch_module.device("cuda")
    if normalized == "cpu":
        return torch_module.device("cpu")
    return torch_module.device(
        "cuda" if torch_module.cuda.is_available() else "cpu"
    )


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def _configure_dgi_cpu_threads(
    torch_module: Any,
    device: Any,
    requested: int,
) -> int:
    """Apply the shared full-batch/sampled CA-DMGI CPU thread policy."""
    effective = int(torch_module.get_num_threads())
    if device.type == "cpu":
        effective = int(requested) if int(requested) > 0 else min(
            32,
            _available_cpu_count(),
        )
        torch_module.set_num_threads(effective)
    return effective


def _validated_graph(
    adjacency: sp.spmatrix,
    n_nodes: int,
    *,
    name: str,
) -> sp.csr_matrix:
    graph = adjacency.tocsr().astype(np.float32, copy=True)
    if graph.shape != (n_nodes, n_nodes):
        raise ValueError(
            f"{name} must have shape {(n_nodes, n_nodes)}, got {graph.shape}"
        )
    if graph.data.size and not np.isfinite(graph.data).all():
        raise ValueError(f"{name} contains non-finite edge weights")
    graph.data = np.maximum(graph.data, 0.0)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    if graph.nnz == 0:
        raise ValueError(f"{name} contains no usable edges")
    return graph


def _gcn_normalize(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    from scipy.sparse import diags, eye

    graph = adjacency + eye(
        adjacency.shape[0], dtype=np.float32, format="csr"
    )
    graph.sum_duplicates()
    degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    degree[degree <= 0] = 1.0
    scale = (1.0 / np.sqrt(degree)).astype(np.float32)
    normalized = (diags(scale) @ graph @ diags(scale)).tocsr()
    normalized.eliminate_zeros()
    normalized.sort_indices()
    return normalized.astype(np.float32, copy=False)


def _context_normalize(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize neighbors without self leakage in local summaries."""
    from scipy.sparse import diags

    graph = adjacency.copy().tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    isolated = degree <= 0
    if isolated.any():
        graph = graph.tolil(copy=False)
        indices = np.flatnonzero(isolated)
        graph[indices, indices] = 1.0
        graph = graph.tocsr()
        degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    normalized = (diags(1.0 / np.maximum(degree, 1e-12)) @ graph).tocsr()
    normalized.eliminate_zeros()
    normalized.sort_indices()
    return normalized.astype(np.float32, copy=False)


def _groupwise_plan(group_values: Sequence[Any]) -> Dict[str, np.ndarray]:
    values = np.asarray(group_values, dtype=object).reshape(-1).copy()
    missing = np.asarray(
        [value is None or (isinstance(value, float) and np.isnan(value))
         for value in values],
        dtype=bool,
    )
    for index in np.flatnonzero(missing):
        values[index] = f"__cellscope_missing_group_{int(index)}"
    lookup: Dict[Any, int] = {}
    codes = np.empty(values.size, dtype=np.int64)
    for index, value in enumerate(values):
        if value not in lookup:
            lookup[value] = len(lookup)
        codes[index] = lookup[value]
    order = np.argsort(codes, kind="stable").astype(np.int64, copy=False)
    position = np.empty(order.size, dtype=np.int64)
    position[order] = np.arange(order.size, dtype=np.int64)
    sizes = np.bincount(codes, minlength=len(lookup)).astype(np.int64)
    starts = np.zeros(sizes.size, dtype=np.int64)
    if sizes.size > 1:
        starts[1:] = np.cumsum(sizes[:-1], dtype=np.int64)
    return {
        "codes": codes,
        "order": order,
        "position": position,
        "starts": starts,
        "sizes": sizes,
        "missing_count": np.asarray([int(missing.sum())], dtype=np.int64),
    }


def train_cell_aware_dmgi(
    features: np.ndarray,
    expression_adjacency: sp.spmatrix,
    spatial_adjacency: sp.spmatrix,
    cell_ids: Sequence[Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, Any]:
    """Train cell-aware multiplex local Infomax without annotation labels.

    Real MetaSpots are never used as pairwise negatives.  Negative examples
    come from feature corruption within the same owning cell, while relation-
    specific local summaries are computed from the uncorrupted expression or
    spatial neighborhood.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    options = dict(config or {})
    epochs = int(options.get("dgi_epochs", 100))
    hidden_dim = int(options.get("dgi_hidden", 64))
    output_dim = int(options.get("dgi_out", 32))
    n_layers = int(options.get("dgi_layers", 2))
    dropout = float(options.get("dgi_dropout", 0.1))
    residual_alpha = float(options.get("dgi_residual_alpha", 0.1))
    layer_norm = bool(options.get("dgi_layer_norm", True))
    learning_rate = float(options.get("dgi_lr", 1e-3))
    weight_decay = float(options.get("dgi_weight_decay", 1e-4))
    gradient_clip = float(options.get("dgi_clip_grad", 1.0))
    expression_weight = float(options.get("dmgi_expression_weight", 1.0))
    spatial_weight = float(options.get("dmgi_spatial_weight", 1.0))
    consensus_weight = float(options.get("dmgi_consensus_weight", 0.1))
    gate_floor = float(options.get("dmgi_gate_floor", 0.1))
    ablation_mode = str(options.get("dmgi_ablation_mode", "full")).strip().lower()
    if ablation_mode not in {"full", "expression_only", "spatial_only", "no_consensus"}:
        raise ValueError("dmgi_ablation_mode must be full, expression_only, spatial_only, or no_consensus")
    if ablation_mode == "expression_only":
        spatial_weight, consensus_weight = 0.0, 0.0
    elif ablation_mode == "spatial_only":
        expression_weight, consensus_weight = 0.0, 0.0
    elif ablation_mode == "no_consensus":
        consensus_weight = 0.0
    early_stopping = bool(options.get("dmgi_early_stopping", True))
    min_epochs = int(options.get("dmgi_min_epochs", min(40, epochs)))
    patience = int(options.get("dmgi_patience", 15))
    min_delta = float(options.get("dmgi_min_delta", 1e-4))
    requested_device = str(options.get("dgi_device", "auto")).strip().lower()
    cpu_threads_requested = int(options.get("dgi_cpu_threads", 0))
    sparse_layout_requested = str(
        options.get("dgi_sparse_layout", "auto")
    ).strip().lower()
    fused_corruption_pass = bool(
        options.get("dmgi_fused_corruption_pass", True)
    )
    tf32_requested = bool(options.get("dgi_tf32", True))
    seed = int(options.get("random_seed", 42))

    if epochs < 1 or hidden_dim < 1 or output_dim < 1 or n_layers < 1:
        raise ValueError("CA-DMGI epochs, dimensions, and layers must be >= 1")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("CA-DMGI dropout must be in [0, 1)")
    if not 0.0 <= residual_alpha <= 1.0:
        raise ValueError("CA-DMGI residual alpha must be in [0, 1]")
    if learning_rate <= 0.0 or weight_decay < 0.0 or gradient_clip <= 0.0:
        raise ValueError("CA-DMGI optimizer settings are invalid")
    if expression_weight < 0.0 or spatial_weight < 0.0 or expression_weight + spatial_weight <= 0.0:
        raise ValueError("CA-DMGI relation weights must be non-negative and not both zero")
    if consensus_weight < 0.0:
        raise ValueError("CA-DMGI consensus weight must be >= 0")
    if not 0.0 <= gate_floor < 0.5:
        raise ValueError("CA-DMGI gate floor must be in [0, 0.5)")
    if min_epochs < 1 or min_epochs > epochs:
        raise ValueError("CA-DMGI min_epochs must be within [1, dgi_epochs]")
    if patience < 1 or min_delta < 0.0:
        raise ValueError("CA-DMGI patience must be >= 1 and min_delta >= 0")
    if cpu_threads_requested < 0:
        raise ValueError("module8.dgi_cpu_threads must be >= 0")
    if sparse_layout_requested not in {"auto", "csr", "coo"}:
        raise ValueError(
            "module8.dgi_sparse_layout must be 'auto', 'csr', or 'coo'"
        )

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("CA-DMGI features must be a non-empty 2D matrix")
    if not np.isfinite(values).all():
        raise ValueError("CA-DMGI features contain non-finite values")
    n_nodes = int(values.shape[0])
    groups = np.asarray(cell_ids, dtype=object).reshape(-1)
    if groups.size != n_nodes:
        raise ValueError("CA-DMGI cell_ids are not aligned to features")

    expression_graph = _validated_graph(
        expression_adjacency,
        n_nodes,
        name="expression_adjacency",
    )
    spatial_graph = _validated_graph(
        spatial_adjacency,
        n_nodes,
        name="spatial_adjacency",
    )
    expression_gcn = _gcn_normalize(expression_graph)
    spatial_gcn = _gcn_normalize(spatial_graph)
    expression_context = _context_normalize(expression_graph)
    spatial_context = _context_normalize(spatial_graph)

    device = resolve_dgi_device(torch, requested_device)
    cpu_threads_effective = _configure_dgi_cpu_threads(
        torch,
        device,
        cpu_threads_requested,
    )
    tf32_enabled = False
    if device.type == "cuda" and hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = tf32_requested
        tf32_enabled = bool(torch.backends.cuda.matmul.allow_tf32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    sparse_layout = (
        "csr" if sparse_layout_requested == "auto" else sparse_layout_requested
    )

    def _to_sparse(matrix: sp.csr_matrix):
        if sparse_layout == "csr":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Sparse CSR tensor support is in beta state.*",
                    category=UserWarning,
                )
                return torch.sparse_csr_tensor(
                    torch.from_numpy(matrix.indptr),
                    torch.from_numpy(matrix.indices),
                    torch.from_numpy(
                        matrix.data.astype(np.float32, copy=False)
                    ),
                    size=matrix.shape,
                    dtype=torch.float32,
                    device=device,
                )
        coo = matrix.tocoo()
        indices = torch.from_numpy(
            np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
        ).to(device)
        data = torch.from_numpy(coo.data.astype(np.float32, copy=False)).to(
            device
        )
        return torch.sparse_coo_tensor(
            indices,
            data,
            size=matrix.shape,
            dtype=torch.float32,
            device=device,
        ).coalesce()

    expression_gcn_t = _to_sparse(expression_gcn)
    spatial_gcn_t = _to_sparse(spatial_gcn)
    expression_context_t = _to_sparse(expression_context)
    spatial_context_t = _to_sparse(spatial_context)
    x = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    x = F.normalize(x, p=2, dim=1)

    plan = _groupwise_plan(groups)
    plan_tensors = {
        key: torch.as_tensor(plan[key], dtype=torch.long, device=device)
        for key in ("codes", "order", "position", "starts", "sizes")
    }

    def _corruption_index():
        codes = plan_tensors["codes"]
        order = plan_tensors["order"]
        position = plan_tensors["position"]
        starts = plan_tensors["starts"]
        sizes = plan_tensors["sizes"]
        shifts = torch.zeros_like(sizes)
        nonsingleton = sizes > 1
        if bool(nonsingleton.any()):
            random_values = torch.rand(
                int(nonsingleton.sum().item()), device=device
            )
            shifts[nonsingleton] = (
                torch.floor(
                    random_values
                    * (sizes[nonsingleton] - 1).to(torch.float32)
                ).to(torch.long)
                + 1
            )
        within = position - starts[codes]
        return order[starts[codes] + (within + shifts[codes]) % sizes[codes]]

    class _SparseGraphConv(nn.Module):
        def __init__(self, input_dim: int, result_dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(result_dim, input_dim))
            self.bias = nn.Parameter(torch.zeros(result_dim))
            nn.init.xavier_uniform_(self.weight)

        def forward(self, node_values, adjacency):
            transformed = F.linear(node_values, self.weight, bias=None)
            return torch.sparse.mm(adjacency, transformed) + self.bias

        def forward_pair(self, positive, negative, adjacency):
            """Propagate positive/corrupted states with one sparse edge scan."""
            positive_transformed = F.linear(
                positive, self.weight, bias=None
            )
            negative_transformed = F.linear(
                negative, self.weight, bias=None
            )
            paired = torch.cat(
                (positive_transformed, negative_transformed), dim=1
            )
            propagated = torch.sparse.mm(adjacency, paired)
            positive_result, negative_result = propagated.split(
                self.bias.numel(), dim=1
            )
            return (
                positive_result + self.bias,
                negative_result + self.bias,
            )

    class _ResidualEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Linear(values.shape[1], hidden_dim)
            self.input_activation = nn.PReLU()
            self.convs = nn.ModuleList(
                _SparseGraphConv(hidden_dim, hidden_dim)
                for _ in range(n_layers)
            )
            self.activations = nn.ModuleList(
                nn.PReLU() for _ in range(n_layers)
            )
            self.norms = nn.ModuleList(
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
                for _ in range(n_layers)
            )
            self.dropout = nn.Dropout(dropout)
            self.output_projection = nn.Linear(hidden_dim, output_dim)

        def forward(self, node_values, adjacency):
            initial = self.input_activation(self.input_projection(node_values))
            encoded = self.dropout(initial)
            for conv, activation, norm in zip(
                self.convs,
                self.activations,
                self.norms,
            ):
                previous = encoded
                message = activation(conv(encoded, adjacency))
                message = self.dropout(message)
                message = (
                    (1.0 - residual_alpha) * message
                    + residual_alpha * initial
                )
                encoded = norm(previous + message)
            return self.output_projection(encoded)

        def forward_pair(self, positive_values, negative_values, adjacency):
            positive_initial = self.input_activation(
                self.input_projection(positive_values)
            )
            negative_initial = self.input_activation(
                self.input_projection(negative_values)
            )
            positive_encoded = self.dropout(positive_initial)
            negative_encoded = self.dropout(negative_initial)
            for conv, activation, norm in zip(
                self.convs,
                self.activations,
                self.norms,
            ):
                positive_previous = positive_encoded
                negative_previous = negative_encoded
                positive_message, negative_message = conv.forward_pair(
                    positive_encoded,
                    negative_encoded,
                    adjacency,
                )
                positive_message = self.dropout(
                    activation(positive_message)
                )
                negative_message = self.dropout(
                    activation(negative_message)
                )
                positive_message = (
                    (1.0 - residual_alpha) * positive_message
                    + residual_alpha * positive_initial
                )
                negative_message = (
                    (1.0 - residual_alpha) * negative_message
                    + residual_alpha * negative_initial
                )
                positive_encoded = norm(
                    positive_previous + positive_message
                )
                negative_encoded = norm(
                    negative_previous + negative_message
                )
            return (
                self.output_projection(positive_encoded),
                self.output_projection(negative_encoded),
            )

    class _CellAwareDMGI(nn.Module):
        def __init__(self):
            super().__init__()
            self.expression_encoder = _ResidualEncoder()
            self.spatial_encoder = _ResidualEncoder()
            self.expression_discriminator = nn.Parameter(
                torch.empty(output_dim, output_dim)
            )
            self.spatial_discriminator = nn.Parameter(
                torch.empty(output_dim, output_dim)
            )
            nn.init.xavier_uniform_(self.expression_discriminator)
            nn.init.xavier_uniform_(self.spatial_discriminator)
            self.gate = nn.Linear(2 * output_dim, 1)

        def encode(self, node_values):
            expression = self.expression_encoder(
                node_values, expression_gcn_t
            )
            spatial = self.spatial_encoder(node_values, spatial_gcn_t)
            return expression, spatial

        def encode_pair(self, positive_values, negative_values):
            expression, corrupt_expression = (
                self.expression_encoder.forward_pair(
                    positive_values,
                    negative_values,
                    expression_gcn_t,
                )
            )
            spatial, corrupt_spatial = self.spatial_encoder.forward_pair(
                positive_values,
                negative_values,
                spatial_gcn_t,
            )
            return expression, spatial, corrupt_expression, corrupt_spatial

        def fuse(self, expression, spatial):
            raw_gate = torch.sigmoid(
                self.gate(torch.cat([expression, spatial], dim=1))
            )
            expression_gate = (
                gate_floor + (1.0 - 2.0 * gate_floor) * raw_gate
            )
            consensus = (
                expression_gate * expression
                + (1.0 - expression_gate) * spatial
            )
            return consensus, expression_gate

        @staticmethod
        def infomax_loss(positive, negative, context, discriminator):
            projected = positive @ discriminator
            corrupted_projected = negative @ discriminator
            scale = math.sqrt(max(1, positive.shape[1]))
            positive_logits = (projected * context).sum(dim=1) / scale
            negative_logits = (
                corrupted_projected * context
            ).sum(dim=1) / scale
            return (
                F.binary_cross_entropy_with_logits(
                    positive_logits,
                    torch.ones_like(positive_logits),
                )
                + F.binary_cross_entropy_with_logits(
                    negative_logits,
                    torch.zeros_like(negative_logits),
                )
            )

    model = _CellAwareDMGI().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: Dict[str, list[float]] = {
        "loss": [],
        "expression_infomax_loss": [],
        "spatial_infomax_loss": [],
        "consensus_loss": [],
        "loss_ema": [],
        "expression_gate_mean": [],
    }
    ema: Optional[float] = None
    best_ema = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, Any]] = None
    stale_epochs = 0

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        permutation = _corruption_index()
        corrupted_x = x[permutation]
        if fused_corruption_pass:
            (
                expression,
                spatial,
                corrupt_expression,
                corrupt_spatial,
            ) = model.encode_pair(x, corrupted_x)
        else:
            expression, spatial = model.encode(x)
            corrupt_expression, corrupt_spatial = model.encode(corrupted_x)
        expression_summary = torch.sigmoid(
            torch.sparse.mm(expression_context_t, expression)
        )
        spatial_summary = torch.sigmoid(
            torch.sparse.mm(spatial_context_t, spatial)
        )
        expression_loss = model.infomax_loss(
            expression,
            corrupt_expression,
            expression_summary,
            model.expression_discriminator,
        )
        spatial_loss = model.infomax_loss(
            spatial,
            corrupt_spatial,
            spatial_summary,
            model.spatial_discriminator,
        )
        consensus, expression_gate = model.fuse(expression, spatial)
        normalized_consensus = F.normalize(consensus, p=2, dim=1)
        consensus_loss = 0.5 * (
            (
                1.0
                - (
                    F.normalize(expression, p=2, dim=1)
                    * normalized_consensus
                ).sum(dim=1)
            ).mean()
            + (
                1.0
                - (
                    F.normalize(spatial, p=2, dim=1)
                    * normalized_consensus
                ).sum(dim=1)
            ).mean()
        )
        loss = (
            expression_weight * expression_loss
            + spatial_weight * spatial_loss
            + consensus_weight * consensus_loss
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_value = float(loss.detach())
        expression_value = float(expression_loss.detach())
        spatial_value = float(spatial_loss.detach())
        consensus_value = float(consensus_loss.detach())
        gate_value = float(expression_gate.detach().mean())
        ema = loss_value if ema is None else 0.9 * ema + 0.1 * loss_value
        history["loss"].append(loss_value)
        history["expression_infomax_loss"].append(expression_value)
        history["spatial_infomax_loss"].append(spatial_value)
        history["consensus_loss"].append(consensus_value)
        history["loss_ema"].append(float(ema))
        history["expression_gate_mean"].append(gate_value)

        if ema < best_ema - min_delta:
            best_ema = float(ema)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress_callback is not None:
            progress_callback(epoch, epochs, loss_value)
        if (
            early_stopping
            and epoch >= min_epochs
            and stale_epochs >= patience
        ):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        expression, spatial = model.encode(x)
        consensus, expression_gate = model.fuse(expression, spatial)
        expression_normalized = F.normalize(expression, p=2, dim=1)
        spatial_normalized = F.normalize(spatial, p=2, dim=1)
        relation_cosine = (
            expression_normalized * spatial_normalized
        ).sum(dim=1)
        expression_context_embedding = F.normalize(
            torch.sparse.mm(expression_context_t, expression_normalized),
            p=2,
            dim=1,
        )
        spatial_context_embedding = F.normalize(
            torch.sparse.mm(spatial_context_t, spatial_normalized),
            p=2,
            dim=1,
        )
        expression_context_cosine = (
            expression_normalized * expression_context_embedding
        ).sum(dim=1)
        spatial_context_cosine = (
            spatial_normalized * spatial_context_embedding
        ).sum(dim=1)

    array_history = {
        key: np.asarray(values_history, dtype=np.float32)
        for key, values_history in history.items()
    }
    gate_array = expression_gate.detach().cpu().numpy().reshape(-1).astype(
        np.float32,
        copy=False,
    )
    if ablation_mode == "expression_only":
        output_embedding = expression
    elif ablation_mode == "spatial_only":
        output_embedding = spatial
    elif ablation_mode == "no_consensus":
        output_embedding = 0.5 * (F.normalize(expression, p=2, dim=1) + F.normalize(spatial, p=2, dim=1))
    else:
        output_embedding = consensus
    return {
        "embedding": output_embedding.detach().cpu().numpy().astype(
            np.float32, copy=False
        ),
        "expression_embedding": expression.detach().cpu().numpy().astype(
            np.float32, copy=False
        ),
        "spatial_embedding": spatial.detach().cpu().numpy().astype(
            np.float32, copy=False
        ),
        "expression_gate": gate_array,
        "history": array_history,
        "metrics": {
            "best_epoch": int(best_epoch),
            "epochs_run": int(len(array_history["loss"])),
            "early_stopped": bool(len(array_history["loss"]) < epochs),
            "final_expression_gate_mean": float(np.mean(gate_array)),
            "final_expression_gate_std": float(np.std(gate_array)),
            "final_relation_cosine": float(
                relation_cosine.mean().detach().cpu()
            ),
            "final_expression_context_cosine": float(
                expression_context_cosine.mean().detach().cpu()
            ),
            "final_spatial_context_cosine": float(
                spatial_context_cosine.mean().detach().cpu()
            ),
            "ablation_mode": ablation_mode,
        },
        "runtime": {
            "device": str(device),
            "device_requested": requested_device,
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else ""
            ),
            "cpu_threads": int(cpu_threads_effective),
            "sparse_layout": sparse_layout,
            "sparse_index_dtype": str(expression_gcn_t.crow_indices().dtype)
            if sparse_layout == "csr"
            else str(expression_gcn_t.indices().dtype),
            "fused_corruption_pass": bool(fused_corruption_pass),
            "tf32_enabled": bool(tf32_enabled),
            "expression_edges_nnz": int(expression_graph.nnz),
            "spatial_edges_nnz": int(spatial_graph.nnz),
            "corruption_group_count": int(plan["sizes"].size),
            "corruption_singleton_group_count": int(
                np.sum(plan["sizes"] == 1)
            ),
            "missing_group_count": int(plan["missing_count"][0]),
        },
    }
