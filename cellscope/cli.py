import sys
import os
try:
    import warnings as _warnings
    _warnings.simplefilter("ignore")
    _warnings.filterwarnings("ignore", category=FutureWarning)
    _warnings.filterwarnings("ignore", category=UserWarning)
    _warnings.filterwarnings("ignore", category=RuntimeWarning)
    _warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Silence noisy modules early
    _warnings.filterwarnings("ignore", category=FutureWarning, module=r"anndata.*")
    _warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn.utils.deprecation")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
except Exception:
    pass
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
import pandas as pd

from .pipeline import run_pipeline
from .config import load_params_yaml
from . import __version__


console = Console()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="cellscope", message="%(prog)s %(version)s")
@click.option("--spatial", required=True, type=click.Path(exists=True), help="Spatial Omics CSV file.")
@click.option("--cell-boundaries", required=True, type=click.Path(exists=True), help="Cell boundaries CSV file.")
@click.option("--nucleus-boundaries", required=True, type=click.Path(exists=True), help="Nucleus boundaries CSV file.")
@click.option("--out-dir", required=True, type=click.Path(), help="Output directory.")
@click.option("--gold-data", type=click.Path(exists=True), help="Optional gold table CSV/Parquet to use for annotation (overrides default).")
@click.option("--no-resume", is_flag=True, help="Disable resume from checkpoints.")
@click.option("--no-save-intermediate", is_flag=True, help="Do not save intermediate checkpoints.")
# Default: show samples; provide a flag to disable
@click.option("--no-show-sample", is_flag=True, help="Do not show sample rows after loading inputs.")
@click.option("--config", type=click.Path(exists=True), help="Optional JSON config file overriding defaults.")
@click.option("--internal-progress/--no-internal-progress", default=False, show_default=True, help="Show internal step prints and per-step timing.")
@click.option("--resume-policy", type=click.Choice(["auto","minimal","force"], case_sensitive=False), default="auto", show_default=True, help="Resume behavior: auto (continue from last completed), minimal (diff-based), force (start from scratch).")
@click.option("--dry-run-diff", is_flag=True, help="Only show which stage would rerun based on config changes; do not execute.")
# 'resume_from' is configured via YAML (not via CLI)

# Module parameters are read from the unified YAML config (no per-module CLI overrides)
# matmul precision is auto-detected on GPU; option removed for simplicity
def main(spatial, cell_boundaries, nucleus_boundaries, out_dir, gold_data, no_resume, no_save_intermediate, no_show_sample, config, internal_progress, resume_policy, dry_run_diff):
    """CellScope CLI: run pipeline and write outputs."""
    os.makedirs(out_dir, exist_ok=True)

    # Load defaults/config and apply CLI overrides
    # Load unified YAML config (params.yaml); JSON override deprecated
    cfg = load_params_yaml()
    # If a JSON path was passed via --config, try merging on top (deprecated)
    if config:
        try:
            import json as _json
            with open(config, 'r', encoding='utf-8') as f:
                override = _json.load(f) or {}
            def _deep_update(base, upd):
                for k, v in upd.items():
                    if isinstance(v, dict) and isinstance(base.get(k), dict):
                        _deep_update(base[k], v)
                    else:
                        base[k] = v
            _deep_update(cfg, override)
        except Exception:
            pass
    # Compression, formats, and module parameters are controlled by the YAML config
    if no_show_sample:
        cfg["show_sample"] = False

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        t = progress.add_task("Starting pipeline", total=None)
        try:
            # Show input samples by default unless disabled via config/flag
            if cfg.get("show_sample", True):
                # Read only a small preview to avoid long startup delays on large files
                s_df = pd.read_csv(spatial, nrows=5)
                c_df = pd.read_csv(cell_boundaries, nrows=5)
                n_df = pd.read_csv(nucleus_boundaries, nrows=5)
                def show_table(df, title):
                    table = Table(title=title, show_lines=False)
                    for col in df.columns:
                        table.add_column(str(col))
                    for _, row in df.iterrows():
                        table.add_row(*[str(row[c]) for c in df.columns])
                    console.print(table)
                show_table(s_df, "Spatial transcripts sample (top 5)")
                show_table(c_df, "Cell boundaries sample (top 5)")
                show_table(n_df, "Nucleus boundaries sample (top 5)")

            def _cb(desc: str):
                try:
                    progress.update(t, description=desc)
                except Exception:
                    pass
                # Print sticky lines for key milestones (more obvious in console)
                try:
                    if (
                        desc.startswith("Resume: load Module") or
                        (desc.startswith("Module ") and ("· Done" in desc))
                    ):
                        console.print(desc)
                except Exception:
                    pass
            # Read module parameters only from unified YAML config (no CLI overrides)
            _scvi_epochs_cfg = int(cfg.get("module7", {}).get("scvi_epochs", 100))
            _dgi_epochs_cfg = int(cfg.get("module8", {}).get("dgi_epochs", 100))
            _enable_annotation = bool(cfg.get("annotation", {}).get("enable", True))

            outputs = run_pipeline(
                spatial_path=spatial,
                cell_boundaries_path=cell_boundaries,
                nucleus_boundaries_path=nucleus_boundaries,
                gold_data=gold_data,
                out_dir=out_dir,
                resume=not no_resume,
                save_intermediate=not no_save_intermediate,
                lite_intermediate=bool(cfg.get("io", {}).get("lite_intermediate", False)),
                compression=cfg.get("io", {}).get("h5ad_compression", "lzf"),
                resume_from=cfg.get("io", {}).get("resume_from"),
                show_internal_progress=internal_progress,
                enable_annotation=_enable_annotation,
                final_format=("parquet" if str(cfg.get("io", {}).get("final_format", "csv")).lower()=="parquet" else "csv"),
                scvi_epochs=_scvi_epochs_cfg,
                dgi_epochs=_dgi_epochs_cfg,
                enable_dgi_sage=bool(cfg.get("module8", {}).get("enable_dgi_sage", False)),
                resume_policy=str(resume_policy).lower(),
                dry_run_diff=bool(dry_run_diff),
                progress_callback=_cb,
            )
            progress.update(t, description="Finished")
        except Exception as e:
            progress.update(t, description="Error")
            console.print(f"[red]Error: {e}")
            sys.exit(1)

    console.print("[green]Done.")
    # Summarize outputs (filter None) and enrich with shapes/stats
    try:
        import anndata as _ad
        import pandas as _pd
    except Exception:
        _ad = None; _pd = None

    adata1_path = outputs.get('adata1')
    adata2_path = outputs.get('adata2')
    final_csv_path = outputs.get('final_csv')
    final_parquet_path = outputs.get('final_parquet')

    # Prefer parquet if available else csv
    final_table_path = final_parquet_path or final_csv_path

    meta_info = []
    if _ad and adata1_path and os.path.exists(adata1_path):
        try:
            _a1 = _ad.read_h5ad(adata1_path)
            n_metaspots = _a1.n_obs
            n_genes = _a1.n_vars
            n_domains = int(_a1.obs['cell_cluster'].nunique()) if 'cell_cluster' in _a1.obs.columns else None
            meta_info.append(f"- adata1: {adata1_path} (metaspots={n_metaspots}, genes={n_genes}{', meta_domains='+str(n_domains) if n_domains is not None else ''})")
        except Exception:
            meta_info.append(f"- adata1: {adata1_path}")
    if _ad and adata2_path and os.path.exists(adata2_path):
        try:
            _a2 = _ad.read_h5ad(adata2_path)
            n_clusters = _a2.n_obs
            genes2 = _a2.n_vars
            meta_info.append(f"- adata2: {adata2_path} (clusters={n_clusters}, genes={genes2})")
        except Exception:
            meta_info.append(f"- adata2: {adata2_path}")
    if final_table_path and os.path.exists(final_table_path):
        try:
            if _pd:
                # Efficient row count without full load for CSV
                if final_table_path.endswith('.csv'):
                    with open(final_table_path, 'r', encoding='utf-8') as f:
                        # subtract header line
                        rows = sum(1 for _ in f) - 1
                else:
                    df_tmp = _pd.read_parquet(final_table_path, columns=[c for c in _pd.read_parquet(final_table_path).columns[:1]])
                    # Fallback to full read if minimal slice fails
                    rows = len(df_tmp)
                meta_info.append(f"- final_data: {final_table_path} (rows={rows})")
            else:
                meta_info.append(f"- final_data: {final_table_path}")
        except Exception:
            meta_info.append(f"- final_data: {final_table_path}")
    # Print only existing artifacts; hide None entries
    for line in meta_info:
        console.print(line)


if __name__ == "__main__":
    main()
