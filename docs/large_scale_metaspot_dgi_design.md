# Large-scale MetaSpot-DGI execution design

## 1. Objective and non-negotiable invariants

The large-scale path must preserve MetaSpot-DGI as the learned representation
used by MetaDomain construction and final MetaSpot-local annotation.  It must
change execution and storage, not silently replace the method with
MetaDomain-DGI.

The following invariants are mandatory:

1. `X_dgi_gcn_metaspot` remains the canonical representation consumed by
   Module 5 and by MetaSpot-local annotation evidence.
2. The expression and spatial relations remain separate throughout Module 8.
3. Expression edges can remain cross-cell-only and spatial edges remain
   within-cell-only.
4. Negative examples remain feature corruptions within the same owning cell;
   real MetaSpots are never declared biological negatives.
5. The learned expression/spatial consensus gate is retained.
6. The public `subcellular_annotation` vocabulary and decision semantics do
   not change.
7. Approximate or sampled execution is recorded explicitly in provenance.  No
   oversized run may silently fall back to another node unit or algorithm.

## 2. Scale routing

Routing is based on the realized number of MetaSpots, graph edges, and
transcripts, not only the number of cells.

| Profile | Realized MetaSpots | Execution path |
|---|---:|---|
| `exact` | <= 200,000 | Current exact graph, full-batch CA-DMGI, consensus Leiden |
| `sampled` | 200,001 to 2,000,000 | ANN graph, sampled CA-DMGI, hierarchical MetaDomain |
| `streaming` | > 2,000,000 | Sharded Module 3, disk-backed ANN graph, sampled CA-DMGI, hierarchical MetaDomain and annotation |

`auto` selects one of these profiles after Module 3 has committed its manifest.
The resolved decision is written to `execution_plan.json` and becomes part of
the resume fingerprint.  A user can request a profile explicitly, but an
explicit `exact` request above its memory guard fails before graph construction.

Initial default guards:

```yaml
execution:
  profile: auto
  exact_max_metaspots: 200000
  sampled_max_metaspots: 2000000
  memory_budget_gb: 64
  scratch_dir: null
  partition_rows: 1000000
  fail_on_backend_fallback: true
```

The current `module7.max_full_batch_nodes` value is not a scale strategy.  In
the new path it remains only the final guard for `profile: exact`.

## 3. Disk-backed data contracts

Large runs must not use one in-memory transcript DataFrame as the contract
between every stage.  Each stage commits immutable, partitioned artifacts and
an atomic manifest.

### 3.1 Stable identifiers

- `transcript_row_id`: stable `uint64` assigned during input normalization.
- `cell_id`: normalized source cell identifier.
- `partition_id`: source FOV when available; otherwise a deterministic hash
  bucket of `cell_id`.
- `metaspot_id`: `{partition_id}|{cell_id}|{local_metaspot_id}`.
- `local_domain_id`: `{partition_id}|{side}|{local_cluster_id}`.

A cell is never split across Module 3 partitions.  Ordered ID checksums and
schema versions are stored at every boundary.

### 3.2 Required artifacts

```text
intermediate_results/
  execution_plan.json
  module3_membership/
    part-*.parquet
    manifest.json
  module4_metaspots/
    obs.parquet
    counts.zarr
    manifest.json
  module7_graph/
    expression.{indptr,indices,data}.npy
    spatial.{indptr,indices,data}.npy
    node_features.zarr
    manifest.json
  module8_embeddings/
    consensus.zarr
    expression.zarr
    spatial.zarr
    gate.zarr
    checkpoints/
    manifest.json
  module5_local_domains.parquet
  module5_global_prototypes.parquet
  module5_metaspot_to_domain.parquet
  annotation_parts/
    part-*.parquet
    manifest.json
```

Manifests are written to a temporary name and atomically renamed only after
all parts and checksums are complete.  Resume reuses only completed partitions
whose source fingerprint and ordered population checksum match.

## 4. Module 3: cell/FOV sharding

### 4.1 Execution

1. Convert CSV input once to a partitioned Parquet dataset.  Native Parquet
   input skips this conversion.
2. Partition by source FOV when present.  Otherwise assign whole cells to
   deterministic hash buckets sized for the configured memory budget.
3. Run the existing per-cell MetaSpot segmentation inside one partition.
4. Write only the compact membership and required local geometry columns to
   `module3_membership/part-*.parquet`.
5. Incrementally aggregate sparse gene counts and MetaSpot summaries into
   disk-backed Module 4 arrays.
6. Commit per-partition audit rows, then reduce them into the global audit.

### 4.2 Memory and resume behavior

Peak Module 3 memory becomes proportional to the largest partition rather than
all transcripts.  A failed partition is rerun independently.  Compact
membership is mandatory even when `lite_intermediate: true`; otherwise a late
resume cannot reconstruct transcript-to-MetaSpot alignment safely.

Suggested defaults:

```yaml
module3:
  execution: streaming
  partition_target_transcripts: 2000000
  partition_target_cells: 5000
  parquet_row_group_size: 250000
  workers: 4
```

## 5. Module 7: approximate, chunked multiplex graph

### 5.1 Expression relation

1. Fit TF-IDF/SVD on a deterministic training sample.
2. Transform all MetaSpots in chunks and store the 32-dimensional expression
   representation in Zarr or a memory-mapped array.
3. Use FAISS IVF-PQ for `streaming`, FAISS HNSW or PyNNDescent for `sampled`,
   and exact sklearn kNN by default for `exact`. An explicit approximate
   backend is also allowed at exact scale when fast iteration is preferred;
   it keeps full-batch DGI but changes graph membership.
4. Retrieve more candidates than required, remove self and same-cell neighbors
   when `metaspot_expr_cross_cell_only: true`, then retain the best `k_expr`.
5. Write edge blocks incrementally and consolidate them into disk-backed CSR.

Recommended FAISS starting point for millions of MetaSpots:

```yaml
module7:
  knn_backend: auto          # exact | pynndescent | faiss_hnsw | faiss_ivfpq
  ann_train_sample: 1000000
  ann_candidate_multiplier: 4
  faiss_nlist: auto          # approximately 4 * sqrt(N), bounded
  faiss_nprobe: 32
  faiss_pq_subvectors: 8
  faiss_pq_bits: 8
  graph_write_chunk_nodes: 100000
```

FAISS is an explicit optional dependency.  With
`fail_on_backend_fallback: true`, its absence fails during preflight instead of
starting a multi-day exact-kNN job.

### 5.2 Spatial relation

Spatial neighbors remain exact because each search is restricted to one cell.
Cells are factorized and sorted once; the implementation must never scan the
full node array once per cell.  Edge blocks are appended by cell/FOV partition.

### 5.3 Provenance and quality audit

Record backend, index parameters, training sample checksum, candidate recall on
a fixed exact-kNN audit subset, cross-cell filtering rate, isolated-node count,
degree quantiles, and relation-specific edge counts.

The minimum default ANN recall target is `recall@k >= 0.90` on the audit
subset.  Falling below the target is a failed graph build, not a warning.

## 6. Module 8: neighbor-sampled CA-DMGI

Module 8 is the critical change.  The model parameters and objective remain
global, but each optimizer step operates on a sampled computation graph.

### 6.1 Loader and model

- Represent expression and spatial edges as two PyG relation types over the
  same MetaSpot node population.
- Use `NeighborLoader` with relation-specific fanouts.  `Cluster-GCN` is an
  optional alternative when repeated neighbor sampling dominates runtime.
- Keep two residual encoders, two local Infomax discriminators, and the learned
  node-wise consensus gate.
- Compute loss only on seed nodes.  Sampled neighbors provide message-passing
  context but are not counted again as independent training observations.
- Construct corrupted features from a deterministic same-cell permutation.
  The corruption map is global and indexed by sampled node IDs.

Recommended initial settings:

```yaml
module8:
  training_mode: auto        # full_batch | neighbor_sampled
  sampler: neighbor
  batch_size: 4096
  expression_fanout: [15, 10]
  spatial_fanout: [6, 6]
  sampled_nodes_per_epoch: 500000
  inference_batch_size: 8192
  mixed_precision: bf16
  gradient_accumulation_steps: 1
  dgi_epochs: 100
  dmgi_min_epochs: 20
  dmgi_patience: 8
  validation_nodes: 50000
```

`dgi_epochs` in sampled mode means passes over
`sampled_nodes_per_epoch`, not full-graph epochs.  Early stopping uses a fixed,
held-out seed-node set and an EMA of validation loss, so batch noise cannot
prematurely stop training.

### 6.2 Inference and checkpoints

After training, run layer-wise or neighbor-sampled inference over every node,
writing expression, spatial, consensus, and gate arrays by row slice.  Do not
materialize all three embeddings on GPU or duplicate them in memory.

Checkpoint after every epoch:

- model and optimizer states;
- RNG and sampler state;
- best validation state;
- completed epoch and embedding inference ranges;
- ordered node checksum and graph-manifest checksum.

CUDA is preferred but not required.  CPU sampled training remains bounded but
will be slower.  GPU preflight reports free memory and selects batch size by a
short dry-run; out-of-memory retries may reduce batch size only before the
first committed epoch.

The NeighborLoader preflight checks the compiled sampler as well as the Python
package: `torch_geometric` alone is insufficient; either `pyg-lib` or
`torch-sparse` must also import successfully.

## 7. Module 5: hierarchical MetaDomain construction

The large path must not build one exact global kNN graph and execute all 15
Leiden candidates on millions of MetaSpots.

### 7.1 Local domains

1. Split nuclear and cytoplasmic MetaSpots using the existing robust geometry
   rule.
2. Form FOV/cell blocks of approximately 50,000 to 100,000 MetaSpots.
3. Cluster each block in `X_dgi_gcn_metaspot` with one calibrated local Leiden
   resolution or MiniBatchKMeans.
4. Persist local assignments immediately.

### 7.2 Global prototypes

For each local domain, compute a weighted prototype containing:

- mean and variance of the consensus DGI embedding;
- pooled expression profile;
- robust geometry summaries;
- MetaSpot and cell counts;
- nuclear/cytoplasmic side;
- source partition and local-domain checksum.

Run stability-selected consensus Leiden on the much smaller prototype graph.
Prototype weights enter occupancy diagnostics and global centroids.  Propagate
the global prototype label back to every MetaSpot through the persisted local
mapping, then apply the existing minimum-size/recurrence audit.

Suggested defaults:

```yaml
module5:
  large_scale_method: hierarchical_prototype
  local_block_nodes: 75000
  local_target_domain_size: 300
  local_method: minibatch_kmeans
  global_method: consensus_leiden
  global_max_prototypes: 100000
```

The global MetaDomain describes a recurring expression/spatial state; it may
contain disconnected local pieces from different FOVs.  This is intentional
and must not be confused with a single contiguous tissue region.

## 8. Streaming annotation and output

RAP-X operates on the global MetaDomain/prototype table, which is small.  Its
evidence is joined back to local domains and MetaSpots by persisted mapping.
MetaSpot-local refinement is then executed one cell/FOV partition at a time.

Each partition writes final transcript labels directly to
`annotation_parts/part-*.parquet`.  Global summaries are reduced from compact
partition summaries; a full transcript DataFrame is never reconstructed.

Large-profile output policy:

```yaml
annotation:
  write_mode: partitioned
  partition_cells: 5000
  diagnostics_sample_metaspots: 50000
io:
  final_format: parquet_dataset
  write_intermediate_csv: false
  write_intermediate_figures: false
  write_full_diagnostic_tables: false
```

Compact audit tables and a deterministic sampled UMAP may be enabled.  Full
CSV, full UMAP, and full per-MetaSpot diagnostic CSV outputs are rejected in
`streaming` mode because they defeat the memory and I/O design.

## 9. Complexity and resource envelope

Let `N` be MetaSpots, `T` transcripts, `k` retained graph neighbors, `d` DGI
dimension, and `P` local prototypes.

| Stage | Time | Peak memory |
|---|---|---|
| Module 3 | `O(T)` partitioned | largest transcript partition |
| Module 7 ANN | approximately `O(N log N + Nk)` | ANN index + one node chunk |
| Module 8 sampled | `O(E_sampled * epochs)` | one sampled computation graph |
| Module 5 hierarchy | `O(N * local_work + P log P)` | one local block + prototype graph |
| Annotation | `O(T)` partitioned | one annotation partition |

For 2 to 8 million MetaSpots with 32-dimensional features, plan for 64 to
128 GB host RAM, fast local scratch storage, and a 16 to 40 GB CUDA GPU.  The
pipeline must estimate graph/embedding disk and memory requirements during
preflight and refuse a run whose configured budget is insufficient.

## 10. Validation gates

Large-scale support is complete only after all gates pass:

1. **Exact-path regression:** existing 2k output remains on `exact`; public
   labels and ordered memberships are unchanged.
2. **Approximation audit:** ANN `recall@k >= 0.90` on a fixed exact subset.
3. **Representation parity:** sampled versus full-batch DGI on the 2k dataset
   reports neighbor-overlap, Procrustes/cosine agreement, and downstream
   MetaDomain ARI.
4. **Annotation parity:** broad-compartment agreement >= 0.98 and seven-class
   agreement >= 0.95 unless a reviewed scientific difference is documented.
5. **Scale test:** synthetic 1M- and 5M-node graphs remain below configured
   host/GPU memory ceilings.
6. **Resume test:** terminate and resume each sharded stage without duplicate
   rows, missing ranges, or checksum drift.
7. **No silent fallback:** missing FAISS/PyG/CUDA capability is reported in
   preflight and provenance.

## 11. Implementation order

1. Introduce execution-plan resolution, manifests, and a mandatory compact
   Module 3 membership checkpoint.
2. Replace the quadratic within-cell grouping scan and add disk-backed graph
   contracts.
3. Add FAISS/PyNNDescent expression graph construction with recall audit.
4. Add neighbor-sampled CA-DMGI plus batched inference and epoch checkpoints.
5. Add hierarchical prototype MetaDomain clustering.
6. Add partitioned annotation/final output.
7. Run 2k parity, then 1M synthetic, then a staged 100k-cell dataset.

Until steps 1 through 6 and their validation gates are complete, increasing
`max_full_batch_nodes` must not be presented as large-scale support.

## 12. Implemented execution contract

The production code now implements the following scale-sensitive behavior:

- `execution_plan.json` is resolved from the realized MetaSpot count and is
  stored in `adata1.uns`; graph/DGI caches are rejected when their effective
  profile or backend differs.
- Module 3 no longer retains every per-cell result object until the end and
  always commits compact transcript-to-MetaSpot membership Parquet.
- Module 7 supports chunked exact queries, PyNNDescent, FAISS HNSW, and FAISS
  IVF-PQ. Streaming refuses exact/PyNNDescent fallbacks. A GPU-enabled FAISS
  build moves IVF-PQ add/query to the configured CUDA device.
- Module 8 uses relation-specific NeighborLoader mini-batches, held-out EMA
  early stopping, mixed precision on CUDA, resumable epoch checkpoints, and
  batched all-node inference while retaining the canonical
  `X_dgi_gcn_metaspot` output.
- Module 5 uses cell-aligned local prototypes and weighted global prototype
  clustering for sampled/streaming runs, split by the existing robust nuclear
  versus cytoplasmic geometry rule.
- Streaming runs additionally emit `final_data_dataset/part-*.parquet` with a
  manifest and `_SUCCESS`; all transcripts from one cell remain in one part.

The current implementation still keeps the normalized transcript table in
memory for RAP-X scoring. Partitioned output and compact checkpoints reduce
serialization and resume cost, but they are not a claim that every annotation
calculation is fully out-of-core. Peak-memory validation on the intended
100k-cell dataset remains a release gate.
