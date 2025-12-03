# CellScope

Resumable pipeline for spatial transcriptomics with clear progress, YAML-driven configuration, and optional subcellular annotation.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE) 
![Python](https://img.shields.io/badge/python-3.9%2B-blue) 
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)

CellScope is a computational pipeline for high-resolution spatial transcriptomics. It calculates the geometric and compositional characteristics of each transcript, aggregates transcripts within the cell, aggregates meta-domains, constructs fused expression/geometric graphs, and learns graph embeddings (DGI). Then map the meta-domains annotations to the subcellular compartments through the reference information. This pipeline is modular, recoverable and can be configured via YAML.

## Change Log

See [CHANGELOG.md](../CHANGELOG.md).

### Operating System
- Linux (tested on recent distros).

### Python Dependencies
- Core dependencies are listed in `CellScope/requirements.txt`.
- Heavy/optional components include PyTorch + torch-geometric (for DGI) and scvi-tools (for expression latent); fallbacks are handled automatically when possible.

## Install CellScope

We recommend installing in a fresh conda environment:

```bash
git clone https://github.com/your-org/CellScope.git
cd CellScope
conda create -n cellscope python=3.9 -y
conda activate cellscope
pip install -r CellScope/requirements.txt
pip install ./CellScope
```

Verify the installation:

```bash
cellscope --help
cellscope --version
```

Typical install time on a server is about 10 minutes (longer with GPU packages).

## Documentation

Full configuration reference: `docs/configuration.md`.

## Usage

```bash
cellscope --help
```

Typical run on CSV inputs:

```bash
python CellScope/bin/cellscope \
	--spatial path/to/transcripts.csv \
	--cell-boundaries path/to/cell_boundaries.csv \
	--nucleus-boundaries path/to/nucleus_boundaries.csv \
	--out-dir out_dir \
	--resume-policy minimal \
	--internal-progress
```

Key flags:
- `--no-resume`: disable resume/checkpoints (default is resume enabled).
- `--no-save-intermediate`: skip writing intermediate artifacts/checkpoints.
- `--no-show-sample`: do not print top-5 input previews.
- `--internal-progress/--no-internal-progress`: toggle step-level prints and timings.
- `--resume-policy {auto|minimal|force}`:
	- `auto`: continue from the last completed stage;
	- `minimal` (recommended): rerun from the earliest stage affected by config fingerprints;
	- `force`: start from scratch.
- `--dry-run-diff`: show which stage(s) would rerun based on config changes; do not execute.

Most module parameters are configured via `CellScope/config/params.yaml`.

## Modules (M1–M8)

- M1 · Geometry features: distances to cell/nucleus, orientation, r‑signature, etc.
- M2 · Composition embeddings: neighborhood BoW + TF‑IDF + SVD.
- M3 · Per‑cell clustering: HDBSCAN with KMeans fallback; outputs `cluster_in_cell` and `cell_sub`.
- M4 · Per‑cell feature table (adata1): aggregate M1/M2 metrics into `adata1.obs`.
- M5 · Meta‑domain clustering: write `adata1.obs['cell_cluster']`.
- M6 · Aggregate to adata2: meta‑domain‐level AnnData.
- M7 · Graph construction: fused expression/geometry graphs, row top‑k pruning, edge normalization.
- M8 · Graph embeddings (DGI): train GCN (optional SAGE) and backfill embeddings into `adata1/adata2`.

Annotation (RAP‑X, enabled by default): normalize/log1p `adata2`, run HVGs + PCA + neighbors + Leiden, and map labels using RNALocate/APEX priors. Diagnostic plots are saved to the intermediate directory.

## Visibility and Resume

- First run: completion lines print as `Module X · Done` (no `(recomputed)`).
- Subsequent runs or cache hits: completion lines include `(cache|recomputed)`.
- Loading cache prints: `Resume: load Module X cache`.
- When writing intermediates: prints `Saving intermediate: ...` and `Saved intermediate: ...`.
- State file: `out_dir/pipeline_state.json`.
- Intermediates: `out_dir/<intermediate_dirname>/` (default `intermediate`).

## Inputs

CellScope expects three primary tabular inputs (CSV or Parquet), with required columns as below.

1) Transcripts table (`--spatial`)
- Required columns: `cell_id`, `x_location`, `y_location`, `feature_name`, `overlaps_nucleus`
- Meaning:
	- `cell_id`: the cell identifier each transcript belongs to
	- `x_location`/`y_location`: transcript coordinates in image/pixel space
	- `feature_name`: gene or feature name of the transcript (e.g. `ACTB`)
	- `overlaps_nucleus`: whether the transcript lies within the nucleus (boolean `0/1` or `true/false`)
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
- Column names are currently fixed as above; validation will error if any are missing.
- CSV/Parquet are auto‑detected; top‑5 sampling previews print unless `--no-show-sample`.
- Coordinates should be in the same reference frame for transcripts and boundaries.

## Outputs

- `adata1.h5ad`: per‑cell/metaspot table (after M4–M5).
- `adata2.h5ad`: meta‑domain table (after M6–M8).
- `final_data.csv|parquet`: transcript‑level results including clustering and annotation.
- Intermediates include `preprocess_*.{csv,parquet}`, `module1_point_features.*`, `module2_point_df.*`, `module3_point_df.*`, `module8_dgi_*_curves.png`, `module5_domain_transcript_stats.*`, `module6_adata2_shape.*`, etc.

Re-running the same command will reuse caches. Use `--dry-run-diff` to preview rerun stages.

## Citation

Please see [CITATION.cff](../CITATION.cff).

## License

This project is licensed under the MIT License — see [LICENSE](../LICENSE).