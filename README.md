# CellScoper

Unsupervised subcellular clustering and annotation for high-resolution Spatial Transcriptomics

CellScoper is a computational pipeline for cell identification and subcellular annotation on high-resolution spatial transcriptomics data. It supports sequencing-based technologies (e.g., Stereo-seq, Seq-Scope) and imaging-based technologies (e.g., seqFISH+, Xenium). The repository provides a CLI and a configurable, modular pipeline with resume and lightweight checkpoints.

![Pipeline overview](./src/CellScope_Framework.jpg)

The figure is a conceptual overview retained for continuity. The current
execution path is M1 geometry → M2 local composition → M3 spatially connected
MetaSpots → M4 `adata1`. With the packaged `module8.dgi_unit: metaspot`, M7/M8
then learn `X_dgi_gcn_metaspot` before M5 MetaDomain clustering and M6 pooling.
The default Cell-Aware Deep Multiplex Graph Infomax (CA-DMGI) keeps cross-cell
expression similarity and within-cell spatial continuity as separate graph
relations, learns a bounded per-MetaSpot consensus gate, and supplies the
embedding to stability-selected consensus Leiden. RAP-X then combines broad
MetaDomain evidence, focal MetaSpot
markers, and explicit same-cell DGI support/rescue branches. No subcellular
gold label is used to train the representation or clustering. With
`dgi_unit: metadomain`, M5/M6 run first and the legacy single-graph DGI path is
trained on `adata2` instead.

## System Requirements
**Operating System**: Linux (tested on Rocky Linux 9.4)

**Core Python Dependencies**:
```
scanpy
anndata
pandas
numpy
scikit-learn
scipy
shapely
pyarrow
hdbscan
matplotlib
seaborn
umap-learn
python-igraph
leidenalg
PyYAML
```
See `requirements.txt` for the full list.

## Install CellScoper
### Recommended (Conda environment)
```bash
git clone https://github.com/zhoutianrui-tongji/CellScope.git
cd CellScope
conda create -n cellscope-env python=3.9
conda activate cellscope-env
pip install -r requirements.txt
pip install -e .
```

## Documentation
Configuration and usage are documented in this `README.md` and the `docs/` folder. The main configuration is `config/params.yaml` (I/O, module parameters, annotation settings).

## Usage
```bash
cellscoper --help
usage: cellscoper [OPTIONS]

Options:
  --spatial PATH               Input transcripts CSV/Parquet
  --cell-boundaries PATH       Cell boundaries CSV/Parquet
  --nucleus-boundaries PATH    Nucleus boundaries CSV/Parquet
  --out-dir PATH               Output directory
  --config PATH                YAML config (default: config/params.yaml)
  --internal-progress          Show internal progress
  --no-resume                  Disable resume
  --no-save-intermediate       Do not persist intermediate artifacts
  --version                    Print version
```
Example:
```bash
cellscoper \
  --spatial test_data/test_transcripts_df.csv \
  --cell-boundaries test_data/test_cell_boundaries_df.csv \
  --nucleus-boundaries test_data/test_nucleus_boundaries_df.csv \
  --out-dir test_out
```

## Inputs

CellScoper expects three primary tabular inputs (CSV or Parquet), with required columns as below.

1) Transcripts table (`--spatial`)
- Required columns: `cell_id`, `x_location`, `y_location`, `feature_name`
- Meaning:
	- `cell_id`: the cell identifier each transcript belongs to
	- `x_location`/`y_location`: transcript coordinates in image/pixel space
	- `feature_name`: gene or feature name of the transcript (e.g. `ACTB`)
	- `overlaps_nucleus`: optional when `module1.use_input_nucleus_flags: false`;
	  otherwise it is required and accepts boolean `0/1` or `true/false`.
		- In polygon mode (`false`, the packaged default), this column is neither
		  used to select cells nor used to infer nuclear membership.
- Example (CSV):
	```csv
	cell_id,x_location,y_location,feature_name,overlaps_nucleus
	C001,102.3,55.8,ACTB,0
	C001,98.1,60.2,GAPDH,1
	C002,210.0,120.5,TUBB,0
	```

2) Cell boundaries (`--cell-boundaries`)
- Required columns: `cell_id`, `vertex_x`, `vertex_y`
- Meaning:
	- Each row is one polygon vertex of a cell boundary; vertices with the same `cell_id` are grouped to form the polygon.
	- Vertex ordering should trace the boundary (clockwise or counter‑clockwise) for each `cell_id`.
- Example (CSV):
	```csv
	cell_id,vertex_x,vertex_y
	C001,90.0,40.0
	C001,130.0,40.0
	C001,130.0,80.0
	C001,90.0,80.0
	C002,200.0,100.0
	C002,240.0,100.0
	C002,240.0,140.0
	C002,200.0,140.0
	```

3) Nucleus boundaries (`--nucleus-boundaries`)
- Schema identical to cell boundaries: `cell_id`, `vertex_x`, `vertex_y`
- Vertices with the same `cell_id` define the nucleus polygon of that cell.
- Example (CSV):
	```csv
	cell_id,vertex_x,vertex_y
	C001,100.0,50.0
	C001,120.0,50.0
	C001,120.0,70.0
	C001,100.0,70.0
	```

Notes:
- Column names are fixed as above; validation reports missing required columns.
- CSV/Parquet are auto‑detected; top‑5 sampling previews print unless `--no-show-sample`.
- Coordinates should be in the same reference frame for transcripts and boundaries.

## Scale-aware execution

`execution.profile: auto` routes from the realized MetaSpot count: up to
200,000 MetaSpots uses the regression-compatible exact/full-batch path by
default;
200,001–2,000,000 uses ANN plus neighbor-sampled CA-DMGI; larger runs require
FAISS IVF-PQ and use hierarchical prototype MetaDomains. The resolved choice
is recorded in `intermediate_results/execution_plan.json` and in `adata1.uns`.
Large runs never silently fall back to exact kNN or full-batch DGI.

The sampled DGI path requires `torch-geometric` plus `pyg-lib` or
`torch-sparse`. PyNNDescent is accepted for the sampled profile. Streaming
runs require FAISS; install the CPU extra with `pip install -e
'.[large-scale]'`, or install a CUDA-compatible FAISS build to accelerate
IVF-PQ add/query on GPU. `module7.faiss_use_gpu: true` makes CUDA mandatory;
the default `auto` uses it when available.

For iteration runs, `--config config/params.fast.yaml` merges an explicit
exploratory speed overlay over the packaged defaults. It keeps full-batch DGI
for exact-scale data, uses PyNNDescent for the global expression graph, trains
CA-DMGI for at most 120 epochs, evaluates two consensus seeds, and skips
diagnostic figures/UMAP. These choices can change graph edges, embeddings, and
MetaDomains; use the default `params.yaml` for production results.

Streaming output can additionally include `final_data_dataset/`, a cell-stable
partitioned Parquet dataset with `manifest.json` and `_SUCCESS`, when
`io.write_partitioned_dataset: true`. See
`docs/large_scale_metaspot_dgi_design.md` for the execution contract and the
remaining 100k-cell validation gates.

## Outputs

- `adata1.h5ad`: final MetaSpot-level object. The user-facing fine result
  remains `obs["subcellular_annotation"]`. Explicit audit tracks include
  `subcellular_annotation_domain`,
  `subcellular_annotation_metaspot_uncorrected`, and
  `subcellular_annotation_metaspot_corrected`. With `annotation.unit:
  metaspot`, additional audit columns record the MetaDomain broad parent,
  same-cell DGI-neighborhood candidate, strict/exploratory local calls,
  marker evidence, q-value, and fallback reason. The alias
  `subcellular_fine_annotation` carries the same seven-class result;
  unsupported MetaSpots fall back to `Nucleus`/`Cytosol`, with uncertainty
  retained in `subcellular_fine_state` and support/reason columns. When
  annotation plots are enabled,
  `obsm["X_umap_metaspot"]` stores the full MetaSpot UMAP for datasets within
  the plotting limit. With `io.drop_embeddings: true`, the final file instead
  retains only the canonical `X_dgi_gcn_metaspot` representation plus
  annotation/audit columns; auxiliary embeddings and graph matrices are
  omitted to keep large outputs portable.
- `adata2.h5ad`: internal MetaDomain evidence object (after M6–M8).
- `final_data.csv|parquet`: transcript-level results including the canonical
  `metaspot_in_cell`, `metaspot`, `meta-domain`, broad-compatible annotation,
  and evidence-qualified fine-annotation columns.
- `effective_params.yaml`, `params_used.yaml`, `run_manifest.json`: exact
  effective configuration and run/input/reference/code provenance.
- `OUTPUT_GUIDE.md`: generated index separating production results,
  publication-facing figures, method QC, audit-only tables, and checkpoints.
- `intermediate_results/metaspot_annotation_result.png`: four-panel MetaSpot result
  view comparing inherited MetaDomain evidence, evidence-qualified fine
  annotation, label transfer, and exact decision provenance. Companion summary
  and plotted-coordinate tables are saved with the `metaspot_annotation_*`
  and `metaspot_fine_*` prefixes.
- `intermediate_results/final_annotation_proportions.png`: the single canonical
  per-cell composition summary. The compatibility alias
  `subcellular_fine_annotation` does not generate a duplicate figure.
- `intermediate_results/module8_ca_dmgi_training.png`: CA-DMGI relation-specific
  expression/spatial Infomax losses and consensus-gate trajectory; it is model
  QC, not an annotation-accuracy figure. Exact epoch histories and final graph/
  gate metrics are stored in `module8_ca_dmgi_training_history.*` and
  `module8_ca_dmgi_summary.*`.
- `intermediate_results/metadomain_consensus_leiden_diagnostics.png`: candidate
  resolution stability, granularity, small-cluster fraction and cross-cell
  recurrence. Exact candidate values are saved in the companion table.
- RAP-X confirmatory-FDR results are retained as `rapx_*` audit tables only.
  A strict-call UMAP is deliberately not generated because strict `Unknown`
  means “not confirmatory-FDR supported”, not the final public annotation.

The historical `cellscope` command remains available as a compatibility alias. Old checkpoints containing `cluster_in_cell` or `cell_sub` are migrated to `metaspot_in_cell` and `metaspot` when loaded.
