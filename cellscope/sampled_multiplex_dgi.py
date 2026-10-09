"""Neighbor-sampled CA-DMGI for large MetaSpot graphs."""

from __future__ import annotations

import copy
import math
import os
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .multiplex_dgi import (
    _configure_dgi_cpu_threads,
    _context_normalize,
    _groupwise_plan,
    _validated_graph,
    resolve_dgi_device,
)


def _fanout(value: Any, layers: int, default: Sequence[int]) -> list[int]:
    if isinstance(value, (int, float)):
        values = [int(value)] * layers
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [int(item) for item in value]
    else:
        values = [int(item) for item in default]
    if not values:
        raise ValueError("Neighbor fanout must not be empty")
    if len(values) < layers:
        values.extend([values[-1]] * (layers - len(values)))
    values = values[:layers]
    if any(item < 1 for item in values):
        raise ValueError("Neighbor fanout values must be >= 1")
    return values


def _same_group_permutation(
    plan: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    codes = np.asarray(plan["codes"], dtype=np.int64)
    order = np.asarray(plan["order"], dtype=np.int64)
    position = np.asarray(plan["position"], dtype=np.int64)
    starts = np.asarray(plan["starts"], dtype=np.int64)
    sizes = np.asarray(plan["sizes"], dtype=np.int64)
    shifts = np.zeros(sizes.size, dtype=np.int64)
    eligible = np.flatnonzero(sizes > 1)
    if eligible.size:
        shifts[eligible] = np.asarray(
            [rng.integers(1, sizes[index]) for index in eligible],
            dtype=np.int64,
        )
    within = position - starts[codes]
    return order[starts[codes] + (within + shifts[codes]) % sizes[codes]]


def _edge_index(matrix: sp.csr_matrix):
    import torch

    coo = matrix.tocoo(copy=False)
    return torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    )


def train_sampled_cell_aware_dmgi(
    features: np.ndarray,
    expression_adjacency: sp.spmatrix,
    spatial_adjacency: sp.spmatrix,
    cell_ids: Sequence[Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, Any]:
    """Train one global CA-DMGI model from neighbor-sampled subgraphs."""
    function_started = time.perf_counter()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import scatter

    options = dict(config or {})
    epochs = int(options.get("dgi_epochs", 100))
    hidden_dim = int(options.get("dgi_hidden", 64))
    output_dim = int(options.get("dgi_out", 32))
    layers = int(options.get("dgi_layers", 2))
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
    early_stopping = bool(options.get("dmgi_early_stopping", True))
    min_epochs = int(options.get("dmgi_min_epochs", min(20, epochs)))
    patience = int(options.get("dmgi_patience", 8))
    min_delta = float(options.get("dmgi_min_delta", 1e-4))
    batch_size = int(options.get("sampled_batch_size", 4096))
    nodes_per_epoch = int(options.get("sampled_nodes_per_epoch", 500_000))
    inference_batch_size = int(
        options.get("sampled_inference_batch_size", 8192)
    )
    validation_nodes = int(options.get("sampled_validation_nodes", 50_000))
    num_workers = int(options.get("sampled_num_workers", 0))
    requested_device = str(options.get("dgi_device", "auto"))
    cpu_threads_requested = int(options.get("dgi_cpu_threads", 0))
    amp_requested = bool(options.get("amp", True))
    seed = int(options.get("random_seed", 42))
    checkpoint_dir = str(options.get("sampled_checkpoint_dir", "")).strip()
    expression_fanout = _fanout(
        options.get("sampled_expression_fanout"),
        layers,
        [15, 10],
    )
    spatial_fanout = _fanout(
        options.get("sampled_spatial_fanout"),
        layers,
        [6, 6],
    )
    if min(epochs, hidden_dim, output_dim, layers, batch_size) < 1:
        raise ValueError("Sampled CA-DMGI dimensions and batch size must be >= 1")
    if min_epochs < 1 or min_epochs > epochs or patience < 1:
        raise ValueError("Invalid sampled CA-DMGI early-stopping settings")
    if cpu_threads_requested < 0:
        raise ValueError("module8.dgi_cpu_threads must be >= 0")

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Sampled CA-DMGI features must have shape (n>=2,d>=1)")
    if not np.isfinite(values).all():
        raise ValueError("Sampled CA-DMGI features contain non-finite values")
    values = values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True),
        1e-12,
    )
    n_nodes = int(values.shape[0])
    groups = np.asarray(cell_ids, dtype=object).reshape(-1)
    if groups.size != n_nodes:
        raise ValueError("Sampled CA-DMGI cell IDs are not aligned")
    expression_graph = _validated_graph(
        expression_adjacency, n_nodes, name="expression_adjacency"
    )
    spatial_graph = _validated_graph(
        spatial_adjacency, n_nodes, name="spatial_adjacency"
    )

    expression_relation = ("metaspot", "expression", "metaspot")
    spatial_relation = ("metaspot", "spatial", "metaspot")
    data = HeteroData()
    data["metaspot"].num_nodes = n_nodes
    data["metaspot"].node_id = torch.arange(n_nodes, dtype=torch.long)
    data[expression_relation].edge_index = _edge_index(expression_graph)
    data[spatial_relation].edge_index = _edge_index(spatial_graph)
    num_neighbors = {
        expression_relation: expression_fanout,
        spatial_relation: spatial_fanout,
    }

    device = resolve_dgi_device(torch, requested_device)
    cpu_threads_effective = _configure_dgi_cpu_threads(
        torch,
        device,
        cpu_threads_requested,
    )
    amp_enabled = bool(amp_requested and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(
            options.get("dgi_tf32", True)
        )
    torch.manual_seed(seed)
    # Keep immutable features on the training device to avoid per-batch NumPy
    # gathers and host-to-device feature copies.
    feature_tensor = torch.from_numpy(values).to(device)
    rng = np.random.default_rng(seed)
    plan = _groupwise_plan(groups)

    class _RelationEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Linear(values.shape[1], hidden_dim)
            self.input_activation = nn.PReLU()
            self.convs = nn.ModuleList(
                GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
                for _ in range(layers)
            )
            self.activations = nn.ModuleList(nn.PReLU() for _ in range(layers))
            self.norms = nn.ModuleList(
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
                for _ in range(layers)
            )
            self.dropout = nn.Dropout(dropout)
            self.output_projection = nn.Linear(hidden_dim, output_dim)

        def forward(self, node_values, relation_edges):
            initial = self.input_activation(self.input_projection(node_values))
            encoded = self.dropout(initial)
            for conv, activation, norm in zip(
                self.convs, self.activations, self.norms
            ):
                previous = encoded
                message = self.dropout(activation(conv(encoded, relation_edges)))
                message = (
                    (1.0 - residual_alpha) * message
                    + residual_alpha * initial
                )
                encoded = norm(previous + message)
            return self.output_projection(encoded)

    class _SampledDMGI(nn.Module):
        def __init__(self):
            super().__init__()
            self.expression_encoder = _RelationEncoder()
            self.spatial_encoder = _RelationEncoder()
            self.expression_discriminator = nn.Parameter(
                torch.empty(output_dim, output_dim)
            )
            self.spatial_discriminator = nn.Parameter(
                torch.empty(output_dim, output_dim)
            )
            nn.init.xavier_uniform_(self.expression_discriminator)
            nn.init.xavier_uniform_(self.spatial_discriminator)
            self.gate = nn.Linear(2 * output_dim, 1)

        def encode(self, node_values, expression_edges, spatial_edges):
            return (
                self.expression_encoder(node_values, expression_edges),
                self.spatial_encoder(node_values, spatial_edges),
            )

        def fuse(self, expression, spatial):
            raw = torch.sigmoid(self.gate(torch.cat((expression, spatial), dim=1)))
            gate = gate_floor + (1.0 - 2.0 * gate_floor) * raw
            return gate * expression + (1.0 - gate) * spatial, gate

    def _context(embedding, edges):
        source, target = edges
        if source.numel() == 0:
            return torch.sigmoid(embedding)
        aggregated = scatter(
            embedding[source], target, dim=0, dim_size=embedding.shape[0], reduce="mean"
        )
        degree = scatter(
            torch.ones_like(target, dtype=embedding.dtype),
            target,
            dim=0,
            dim_size=embedding.shape[0],
            reduce="sum",
        )
        isolated = degree <= 0
        if bool(isolated.any()):
            aggregated[isolated] = embedding[isolated]
        return torch.sigmoid(aggregated)

    def _infomax(positive, negative, context, discriminator, seed_count):
        positive = positive[:seed_count]
        negative = negative[:seed_count]
        context = context[:seed_count]
        scale = math.sqrt(max(1, positive.shape[1]))
        positive_logits = ((positive @ discriminator) * context).sum(dim=1) / scale
        negative_logits = ((negative @ discriminator) * context).sum(dim=1) / scale
        return F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits)
        ) + F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits)
        )

    model = _SampledDMGI().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # torch<2.4 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    all_nodes = np.arange(n_nodes, dtype=np.int64)
    validation_count = min(max(1, validation_nodes), max(1, n_nodes // 10))
    validation_ids = np.sort(
        rng.choice(n_nodes, size=validation_count, replace=False)
    ).astype(np.int64)
    train_pool = np.setdiff1d(all_nodes, validation_ids, assume_unique=True)
    history: Dict[str, list[float]] = {
        "loss": [],
        "validation_loss": [],
        "expression_infomax_loss": [],
        "spatial_infomax_loss": [],
        "consensus_loss": [],
        "loss_ema": [],
        "expression_gate_mean": [],
    }
    best_state: Optional[Dict[str, Any]] = None
    best_epoch = 0
    best_monitor = float("inf")
    stale_epochs = 0
    ema: Optional[float] = None
    start_epoch = 0
    checkpoint_path = (
        os.path.join(checkpoint_dir, "sampled_dmgi_latest.pt")
        if checkpoint_dir
        else ""
    )
    model_signature = {
        "n_nodes": n_nodes,
        "n_features": int(values.shape[1]),
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "layers": layers,
        "expression_fanout": expression_fanout,
        "spatial_fanout": spatial_fanout,
        "random_seed": seed,
    }
    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=device,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=device)
            if checkpoint.get("model_signature") == model_signature:
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                start_epoch = min(int(checkpoint.get("epoch", 0)), epochs)
                best_epoch = int(checkpoint.get("best_epoch", 0))
                best_monitor = float(checkpoint.get("best_monitor", float("inf")))
                best_state = checkpoint.get("best_state")
                restored_history = checkpoint.get("history", {})
                history = {
                    key: [
                        float(value)
                        for value in restored_history.get(
                            key,
                            [float("nan")] * start_epoch,
                        )
                    ][:start_epoch]
                    for key in history
                }
                ema = history["loss_ema"][-1] if history["loss_ema"] else None
                stale_epochs = int(checkpoint.get("stale_epochs", 0))
                validation_ids = np.asarray(
                    checkpoint.get("validation_ids", validation_ids),
                    dtype=np.int64,
                )
                train_pool = np.setdiff1d(
                    all_nodes, validation_ids, assume_unique=True
                )
                if "numpy_rng_state" in checkpoint:
                    rng.bit_generator.state = checkpoint["numpy_rng_state"]
                if "torch_rng_state" in checkpoint:
                    torch.set_rng_state(checkpoint["torch_rng_state"])
        except Exception:
            # A truncated or obsolete checkpoint must not make the whole run
            # unrecoverable; its signature is also persisted for audit.
            start_epoch = 0

    timing_seconds = {
        "setup": 0.0,
        "train_loader": 0.0,
        "train_compute": 0.0,
        "validation_loader": 0.0,
        "validation_compute": 0.0,
        "checkpoint": 0.0,
        "inference_loader": 0.0,
        "inference_compute": 0.0,
        "postprocess": 0.0,
        "total": 0.0,
    }
    epoch_seconds: list[float] = []

    def _loader(input_ids: np.ndarray, *, shuffle: bool, size: int):
        return NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=("metaspot", torch.from_numpy(input_ids)),
            batch_size=max(1, int(size)),
            shuffle=shuffle,
            num_workers=max(0, num_workers),
            pin_memory=device.type == "cuda",
            persistent_workers=bool(num_workers > 0),
        )

    def _timed_batches(loader, key: str):
        iterator = iter(loader)
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                timing_seconds[key] += time.perf_counter() - started
                return
            timing_seconds[key] += time.perf_counter() - started
            yield batch

    def _batch_loss(batch, corruption: torch.Tensor):
        node_store = batch["metaspot"]
        n_id = node_store.n_id.detach().to(device, non_blocking=True)
        positive_x = feature_tensor[n_id]
        negative_x = feature_tensor[corruption[n_id]]
        expression_edges = batch[expression_relation].edge_index.to(
            device, non_blocking=True
        )
        spatial_edges = batch[spatial_relation].edge_index.to(
            device, non_blocking=True
        )
        seed_count = int(node_store.batch_size)
        expression, spatial = model.encode(
            positive_x, expression_edges, spatial_edges
        )
        corrupt_expression, corrupt_spatial = model.encode(
            negative_x, expression_edges, spatial_edges
        )
        expression_loss = _infomax(
            expression,
            corrupt_expression,
            _context(expression, expression_edges),
            model.expression_discriminator,
            seed_count,
        )
        spatial_loss = _infomax(
            spatial,
            corrupt_spatial,
            _context(spatial, spatial_edges),
            model.spatial_discriminator,
            seed_count,
        )
        consensus, gate = model.fuse(expression, spatial)
        normalized_consensus = F.normalize(consensus[:seed_count], p=2, dim=1)
        consensus_loss = 0.5 * (
            1.0
            - (
                F.normalize(expression[:seed_count], p=2, dim=1)
                * normalized_consensus
            ).sum(dim=1).mean()
            + 1.0
            - (
                F.normalize(spatial[:seed_count], p=2, dim=1)
                * normalized_consensus
            ).sum(dim=1).mean()
        )
        total = (
            expression_weight * expression_loss
            + spatial_weight * spatial_loss
            + consensus_weight * consensus_loss
        )
        return total, expression_loss, spatial_loss, consensus_loss, gate[:seed_count]

    # Training and validation node sets stay fixed across epochs. Reusing their
    # loaders avoids rebuilding sampler state and worker processes.
    timing_seconds["setup"] = time.perf_counter() - function_started
    training_started = time.perf_counter()
    static_train_loader = None
    validation_loader = None
    if start_epoch < epochs:
        full_epoch = min(max(1, nodes_per_epoch), train_pool.size)
        if full_epoch == train_pool.size:
            static_train_loader = _loader(
                train_pool, shuffle=True, size=batch_size
            )
        validation_loader = _loader(
            validation_ids, shuffle=False, size=inference_batch_size
        )

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_started = time.perf_counter()
        take = min(max(1, nodes_per_epoch), train_pool.size)
        epoch_ids = (
            train_pool
            if take == train_pool.size
            else rng.choice(train_pool, size=take, replace=False)
        ).astype(np.int64, copy=False)
        corruption = _same_group_permutation(plan, rng)
        running = np.zeros(5, dtype=np.float64)
        batches = 0
        corruption_tensor = torch.from_numpy(corruption).to(
            device, non_blocking=True
        )
        model.train()
        train_loader = (
            static_train_loader
            if static_train_loader is not None
            else _loader(epoch_ids, shuffle=True, size=batch_size)
        )
        for batch in _timed_batches(train_loader, "train_loader"):
            compute_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                total, expression_loss, spatial_loss, consensus_loss, gate = (
                    _batch_loss(batch, corruption_tensor)
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            running += np.asarray(
                [
                    float(total.detach()),
                    float(expression_loss.detach()),
                    float(spatial_loss.detach()),
                    float(consensus_loss.detach()),
                    float(gate.detach().mean()),
                ]
            )
            batches += 1
            timing_seconds["train_compute"] += (
                time.perf_counter() - compute_started
            )
        running /= max(1, batches)

        model.eval()
        validation_running = 0.0
        validation_batches = 0
        validation_corruption = _same_group_permutation(
            plan, np.random.default_rng(seed + 10_000 + epoch)
        )
        with torch.no_grad():
            validation_corruption_tensor = torch.from_numpy(
                validation_corruption
            ).to(device, non_blocking=True)
            for batch in _timed_batches(
                validation_loader,
                "validation_loader",
            ):
                compute_started = time.perf_counter()
                total, _, _, _, _ = _batch_loss(
                    batch, validation_corruption_tensor
                )
                validation_running += float(total.detach())
                validation_batches += 1
                timing_seconds["validation_compute"] += (
                    time.perf_counter() - compute_started
                )
        monitor = validation_running / max(1, validation_batches)
        ema = monitor if ema is None else 0.9 * ema + 0.1 * monitor
        for key, value in zip(
            (
                "loss",
                "expression_infomax_loss",
                "spatial_infomax_loss",
                "consensus_loss",
                "expression_gate_mean",
            ),
            running,
        ):
            history[key].append(float(value))
        history["validation_loss"].append(float(monitor))
        history["loss_ema"].append(float(ema))

        if ema < best_monitor - min_delta:
            best_monitor = float(ema)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if checkpoint_dir:
            checkpoint_started = time.perf_counter()
            os.makedirs(checkpoint_dir, exist_ok=True)
            temporary = os.path.join(checkpoint_dir, ".sampled_dmgi_latest.tmp.pt")
            final = os.path.join(checkpoint_dir, "sampled_dmgi_latest.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_epoch": best_epoch,
                    "best_monitor": best_monitor,
                    "best_state": best_state,
                    "stale_epochs": stale_epochs,
                    "history": history,
                    "random_seed": seed,
                    "validation_ids": validation_ids,
                    "model_signature": model_signature,
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                },
                temporary,
            )
            os.replace(temporary, final)
            timing_seconds["checkpoint"] += (
                time.perf_counter() - checkpoint_started
            )
        epoch_seconds.append(time.perf_counter() - epoch_started)
        if progress_callback is not None:
            progress_callback(epoch, epochs, float(running[0]))
        if early_stopping and epoch >= min_epochs and stale_epochs >= patience:
            break

    training_seconds = time.perf_counter() - training_started
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    expression_result = np.empty((n_nodes, output_dim), dtype=np.float32)
    spatial_result = np.empty((n_nodes, output_dim), dtype=np.float32)
    consensus_result = np.empty((n_nodes, output_dim), dtype=np.float32)
    gate_result = np.empty(n_nodes, dtype=np.float32)
    inference_started = time.perf_counter()
    with torch.no_grad():
        for batch in _timed_batches(
            _loader(all_nodes, shuffle=False, size=inference_batch_size),
            "inference_loader",
        ):
            compute_started = time.perf_counter()
            node_store = batch["metaspot"]
            n_id = node_store.n_id.detach().cpu().numpy().astype(np.int64, copy=False)
            seed_count = int(node_store.batch_size)
            seed_ids = n_id[:seed_count]
            expression, spatial = model.encode(
                feature_tensor[
                    node_store.n_id.detach().to(device, non_blocking=True)
                ],
                batch[expression_relation].edge_index.to(device, non_blocking=True),
                batch[spatial_relation].edge_index.to(device, non_blocking=True),
            )
            consensus, gate = model.fuse(expression, spatial)
            expression_result[seed_ids] = expression[:seed_count].cpu().numpy()
            spatial_result[seed_ids] = spatial[:seed_count].cpu().numpy()
            consensus_result[seed_ids] = consensus[:seed_count].cpu().numpy()
            gate_result[seed_ids] = gate[:seed_count, 0].cpu().numpy()
            timing_seconds["inference_compute"] += (
                time.perf_counter() - compute_started
            )

    inference_seconds = time.perf_counter() - inference_started
    postprocess_started = time.perf_counter()
    expression_normalized = expression_result / np.maximum(
        np.linalg.norm(expression_result, axis=1, keepdims=True), 1e-12
    )
    spatial_normalized = spatial_result / np.maximum(
        np.linalg.norm(spatial_result, axis=1, keepdims=True), 1e-12
    )
    relation_cosine = np.sum(expression_normalized * spatial_normalized, axis=1)
    expression_context = _context_normalize(expression_graph) @ expression_normalized
    spatial_context = _context_normalize(spatial_graph) @ spatial_normalized
    expression_context /= np.maximum(
        np.linalg.norm(expression_context, axis=1, keepdims=True), 1e-12
    )
    spatial_context /= np.maximum(
        np.linalg.norm(spatial_context, axis=1, keepdims=True), 1e-12
    )
    array_history = {
        key: np.asarray(value, dtype=np.float32) for key, value in history.items()
    }
    timing_seconds["postprocess"] = time.perf_counter() - postprocess_started
    timing_seconds["training"] = training_seconds
    timing_seconds["inference"] = inference_seconds
    timing_seconds["total"] = time.perf_counter() - function_started
    return {
        "embedding": consensus_result,
        "expression_embedding": expression_result,
        "spatial_embedding": spatial_result,
        "expression_gate": gate_result,
        "history": array_history,
        "metrics": {
            "best_epoch": int(best_epoch),
            "epochs_run": int(len(array_history["loss"])),
            "early_stopped": bool(len(array_history["loss"]) < epochs),
            "final_expression_gate_mean": float(np.mean(gate_result)),
            "final_expression_gate_std": float(np.std(gate_result)),
            "final_relation_cosine": float(np.mean(relation_cosine)),
            "final_expression_context_cosine": float(
                np.mean(np.sum(expression_normalized * expression_context, axis=1))
            ),
            "final_spatial_context_cosine": float(
                np.mean(np.sum(spatial_normalized * spatial_context, axis=1))
            ),
        },
        "runtime": {
            "device": str(device),
            "device_requested": requested_device,
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else ""
            ),
            "cpu_threads": int(cpu_threads_effective),
            "sparse_layout": "pyg_neighbor_loader",
            "sparse_index_dtype": "torch.int64",
            "fused_corruption_pass": False,
            "tf32_enabled": bool(
                device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32
            ),
            "expression_edges_nnz": int(expression_graph.nnz),
            "spatial_edges_nnz": int(spatial_graph.nnz),
            "corruption_group_count": int(plan["sizes"].size),
            "corruption_singleton_group_count": int(np.sum(plan["sizes"] == 1)),
            "missing_group_count": int(plan["missing_count"][0]),
            "training_mode": "neighbor_sampled",
            "sampled_batch_size": batch_size,
            "sampled_nodes_per_epoch": min(nodes_per_epoch, train_pool.size),
            "sampled_num_workers": int(num_workers),
            "expression_fanout": expression_fanout,
            "spatial_fanout": spatial_fanout,
            "validation_nodes": validation_count,
            "amp_enabled": amp_enabled,
            "checkpoint_path": checkpoint_path,
            "resumed_from_epoch": int(start_epoch),
            "timing_seconds": {
                key: float(value) for key, value in timing_seconds.items()
            },
            "epoch_seconds": [float(value) for value in epoch_seconds],
        },
    }
