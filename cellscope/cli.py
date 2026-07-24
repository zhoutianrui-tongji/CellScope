import sys
import os
import copy
import functools
import re
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
import yaml

from .pipeline import run_pipeline
from .config import load_params_yaml, get_default_params_path
from . import __version__
from .run_lock import OutputDirectoryLockError, OutputDirectoryRunLock


console = Console()


def _safe_load_yaml(path: str) -> dict:
    """Directly load YAML dict from path; return {} on any failure."""
    if not path or not os.path.exists(path):
        console.print(f"[red]YAML load skipped (not exists): {path}[/red]") if path else None
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        console.print(f"[cyan]YAML exists:[/cyan] {path} (bytes={len(content)})")
        preview = "\n".join(content.splitlines()[:5])
        console.print(f"[cyan]YAML head:[/cyan]\n{preview}")
        data = yaml.safe_load(content) or {}
        if not isinstance(data, dict):
            console.print(f"[red]YAML parsed but not dict, type={type(data)}[/red]")
            return {}
        return data
    except Exception as e:
        console.print(f"[yellow]YAML load error at {path}: {e}; try sanitize and reload[/yellow]")
        try:
            sanitized = content.replace("\t", "  ")
            sanitized = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", sanitized)
            sanitized = re.sub(r":([^\s\[{])", r": \1", sanitized)
            data = yaml.safe_load(sanitized) or {}
            if isinstance(data, dict) and data:
                console.print(f"[green]YAML sanitized and loaded successfully[/green]")
                return data
            console.print("[red]YAML sanitize reload parsed but empty or non-dict[/red]")
            return {}
        except Exception as e2:
            console.print(f"[red]YAML sanitize reload failed: {e2}[/red]")
            return {}


def _load_params_with_fallback(path: str, allow_fallback: bool = False):
    """Load YAML; optionally fall back to repo config/params.yaml if empty or missing."""
    path_norm = os.path.normpath(os.path.abspath(path)) if path else None
    cfg = load_params_yaml(path=path_norm) if path_norm else {}
    if not cfg:
        cfg = _safe_load_yaml(path_norm)
    if cfg:
        return cfg, path_norm
    if allow_fallback:
        fallback = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "config", "params.yaml"))
        if (not path_norm) or (fallback != path_norm):
            cfg = load_params_yaml(path=fallback) or _safe_load_yaml(fallback)
            if cfg:
                return cfg, fallback
    if path_norm:
        console.print(f"[red]Warning: params.yaml could not be loaded or is empty at {path_norm}[/red]")
    return {}, path_norm


def _merge_with_defaults(default_cfg: dict, user_cfg: dict) -> dict:
    """Return merged config where missing/None in user_cfg fall back to default_cfg."""
    d = copy.deepcopy(default_cfg if isinstance(default_cfg, dict) else {})
    if not isinstance(user_cfg, dict):
        return d
    for k, v_user in user_cfg.items():
        v_def = d.get(k)
        if isinstance(v_user, dict) and isinstance(v_def, dict):
            d[k] = _merge_with_defaults(v_def, v_user)
        else:
            if v_user is not None:
                d[k] = v_user
    return d


def _with_output_directory_lock(func):
    """Serialize CLI runs that target the same output directory."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        out_dir = kwargs.get("out_dir")
        if not out_dir:
            raise click.ClickException("Missing required --out-dir for run locking.")
        try:
            with OutputDirectoryRunLock(out_dir):
                return func(*args, **kwargs)
        except OutputDirectoryLockError as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapped


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="cellscoper", message="%(prog)s %(version)s")
@click.option("--spatial", required=True, type=click.Path(exists=True), help="Spatial Omics CSV file.")
@click.option("--cell-boundaries", required=True, type=click.Path(exists=True), help="Cell boundaries CSV file.")
@click.option("--nucleus-boundaries", required=True, type=click.Path(exists=True), help="Nucleus boundaries CSV file.")
@click.option("--out-dir", required=True, type=click.Path(), help="Output directory.")
@click.option("--gold-data", type=click.Path(exists=True), help="Optional gold table CSV/Parquet to use for annotation (overrides default).")
@click.option("--no-resume", is_flag=True, help="Disable resume from checkpoints.")
@click.option("--no-save-intermediate", is_flag=True, help="Do not save intermediate checkpoints.")
# Default: show samples; provide a flag to disable
@click.option("--no-show-sample", is_flag=True, help="Do not show sample rows after loading inputs.")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True),
    help="Path to params.yaml (replaces built-in default).",
)
@click.option("--internal-progress/--no-internal-progress", default=False, show_default=True, help="Show internal step prints and per-step timing.")
@click.option("--resume-policy", type=click.Choice(["auto","minimal","force"], case_sensitive=False), default="auto", show_default=True, help="Resume behavior: auto (continue from last completed), minimal (diff-based), force (start from scratch).")
@click.option("--dry-run-diff", is_flag=True, help="Only show which stage would rerun based on config changes; do not execute.")
# 'resume_from' is configured via YAML (not via CLI)

# Module parameters are read from the unified YAML config (no per-module CLI overrides)
# matmul precision is auto-detected on GPU; option removed for simplicity
@_with_output_directory_lock
def main(spatial, cell_boundaries, nucleus_boundaries, out_dir, gold_data, no_resume, no_save_intermediate, no_show_sample, config, internal_progress, resume_policy, dry_run_diff):
    """CellScoper CLI: run pipeline and write outputs."""
    os.makedirs(out_dir, exist_ok=True)

    # Load unified YAML config (params.yaml); allow overriding with --config
    cfg_path = config or os.environ.get("CELLSCOPE_CONFIG") or None
    default_cfg, default_used_path = _load_params_with_fallback(get_default_params_path(), allow_fallback=True)
    if not default_cfg:
        console.print("[red]Warning: default params.yaml could not be loaded or is empty[/red]")
    user_cfg, user_used_path = _load_params_with_fallback(cfg_path, allow_fallback=False) if cfg_path else ({}, None)
    cfg = _merge_with_defaults(default_cfg, user_cfg)
    # 新增调试输出，确认配置读取与合并结果
    console.print(f"[cyan]default_cfg loaded from:[/cyan] {default_used_path} (empty={not bool(default_cfg)})")
    console.print(f"[cyan]user_cfg loaded from:[/cyan] {user_used_path or 'N/A'} (empty={not bool(user_cfg)})")
    console.print(f"[cyan]merged cfg keys:[/cyan] {list(cfg.keys())}")
    if user_used_path:
        console.print(f"[cyan]Loaded params from:[/cyan] {user_used_path} (merged with defaults)")
    else:
        console.print(f"[cyan]Loaded params from default:[/cyan] {default_used_path}")
    # Ensure downstream pipeline internals load the same effective config file.
    if user_used_path:
        os.environ["CELLSCOPE_CONFIG"] = os.path.abspath(user_used_path)
    elif default_used_path:
        os.environ.setdefault("CELLSCOPE_CONFIG", os.path.abspath(default_used_path))
    try:
        m3 = cfg.get("module3", {})
        target_sites = m3.get("target_sites_per_metaspot")
        try:
            target_sites_enabled = float(target_sites) > 0
        except (TypeError, ValueError):
            target_sites_enabled = False
        spatial_mode = (
            bool(m3.get("enforce_spatial_connectivity", False))
            and target_sites_enabled
        )
        if spatial_mode:
            active_keys = [
                "target_sites_per_metaspot",
                "min_metaspots_per_cell",
                "max_metaspots_per_cell",
                "spatial_connectivity_k",
                "spatial_radius_quantile",
                "spatial_radius_factor",
                "spatial_min_component_size",
                "spatial_compactness",
                "n_feat_hdb",
            ]
            preview = {"segmentation_mode": "spatial_ward"}
        else:
            active_keys = [
                "hdbscan_min_cluster_size",
                "hdbscan_min_samples_frac",
                "target_sub_size",
                "kmeans_min_k",
                "kmeans_max_k",
                "use_k_cap",
                "merge_min_size",
                "hdbscan_max_points",
                "kmeans_train_size",
                "kmeans_batch_size",
                "n_feat_hdb",
            ]
            preview = {"segmentation_mode": "legacy_hdbscan_kmeans"}
        preview.update({key: m3.get(key) for key in active_keys})
        console.print("[cyan]module3 active config:[/cyan] " + str(preview))
    except Exception:
        pass
    # Persist the effective config for reproducibility
    try:
        with open(os.path.join(out_dir, "params_used.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
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

            _run_kwargs = dict(
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
            console.print("[cyan]run_pipeline kwargs preview:[/cyan]")
            for k, v in _run_kwargs.items():
                console.print(f"  - {k}: {v}")

            outputs = run_pipeline(**_run_kwargs)
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
