import os
import copy
import functools
import json
import subprocess
try:
    import warnings as _warnings
    # Keep only known third-party compatibility notices quiet. Runtime and
    # scientific warnings remain visible to users and logs.
    _warnings.filterwarnings("ignore", category=FutureWarning, module=r"anndata.*")
    _warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn.utils.deprecation")
except Exception:
    pass
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
import yaml

from .io import read_table_preview
from .config import (
    LEGACY_OMM_REFINEMENT_KEYS,
    get_default_params_path,
    load_params_yaml,
    resolve_dgi_configuration,
)
from . import __version__
from .provenance import (
    atomic_write_json,
    atomic_write_yaml,
    build_input_provenance,
    bundled_reference_provenance,
    file_provenance,
    git_provenance,
    sha256_mapping,
    source_tree_provenance,
    utc_now,
)
from .run_lock import OutputDirectoryLockError, OutputDirectoryRunLock


console = Console()


def _safe_load_yaml(path: str) -> dict:
    """Load a YAML mapping for a missing-default fallback only."""
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
    except Exception as exc:
        console.print(f"[red]YAML load error at {path}: {exc}[/red]")
        return {}


def _load_params_with_fallback(
    path: str,
    allow_fallback: bool = False,
    *,
    resolve_config: bool = True,
):
    """Load YAML; optionally fall back to repo config/params.yaml if empty or missing."""
    path_norm = os.path.normpath(os.path.abspath(path)) if path else None
    try:
        cfg = (
            load_params_yaml(
                path=path_norm,
                resolve_config=resolve_config,
            )
            if path_norm
            else {}
        )
    except ValueError as exc:
        raise click.ClickException(
            f"Invalid configuration at {path_norm}: {exc}"
        ) from exc
    if not cfg:
        cfg = _safe_load_yaml(path_norm)
    if cfg:
        return cfg, path_norm
    if allow_fallback:
        fallback = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "config", "params.yaml"))
        if (not path_norm) or (fallback != path_norm):
            cfg = load_params_yaml(
                path=fallback,
                resolve_config=resolve_config,
            ) or _safe_load_yaml(fallback)
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


def _preserve_legacy_omm_policy(
    merged_cfg: dict,
    user_cfg: dict,
) -> dict:
    """Keep pre-policy user YAML from inheriting the new balanced default."""
    user_annotation = (
        user_cfg.get("annotation", {})
        if isinstance(user_cfg, dict)
        else {}
    )
    if (
        isinstance(user_annotation, dict)
        and "omm_policy" not in user_annotation
        and any(
            key in user_annotation
            for key in LEGACY_OMM_REFINEMENT_KEYS
        )
    ):
        merged_cfg.setdefault("annotation", {}).pop("omm_policy", None)
    return merged_cfg


def _with_output_directory_lock(func):
    """Serialize CLI runs that target the same output directory."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        out_dir = kwargs.get("out_dir")
        if not out_dir:
            raise click.ClickException("Missing required --out-dir for run locking.")
        lock = OutputDirectoryRunLock(out_dir)
        try:
            with lock:
                try:
                    return func(*args, **kwargs)
                finally:
                    try:
                        subprocess.run(
                            ["chmod", "-R", "g+rX", lock.out_dir],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except (OSError, subprocess.CalledProcessError) as exc:
                        console.print(
                            "[yellow]Could not make outputs group-readable: "
                            f"{exc}[/yellow]"
                        )
        except OutputDirectoryLockError as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapped


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="cellscoper", message="%(prog)s %(version)s")
@click.option("--spatial", required=True, type=click.Path(exists=True), help="Spatial Omics CSV/Parquet file.")
@click.option("--cell-boundaries", required=True, type=click.Path(exists=True), help="Cell boundaries CSV/Parquet file.")
@click.option("--nucleus-boundaries", required=True, type=click.Path(exists=True), help="Nucleus boundaries CSV/Parquet file.")
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
@click.option(
    "--resume-policy",
    type=click.Choice(["auto", "minimal", "force"], case_sensitive=False),
    default=None,
    help=(
        "Resume behavior. If omitted, use io.resume_policy from the effective "
        "configuration (default: minimal)."
    ),
)
@click.option(
    "--resume-from",
    type=click.Choice(
        [
            "load_inputs",
            "preprocess",
            "build_anndata",
            "annotate",
            "write_outputs",
        ],
        case_sensitive=False,
    ),
    default=None,
    help=(
        "Explicit stage to rerun from. Overrides io.resume_from for this "
        "invocation; use 'annotate' to reuse upstream checkpoints."
    ),
)
@click.option("--dry-run-diff", is_flag=True, help="Only show which stage would rerun based on config changes; do not execute.")

# Module parameters are read from the unified YAML config (no per-module CLI overrides)
# matmul precision is auto-detected on GPU; option removed for simplicity
@_with_output_directory_lock
def main(spatial, cell_boundaries, nucleus_boundaries, out_dir, gold_data, no_resume, no_save_intermediate, no_show_sample, config, internal_progress, resume_policy, resume_from, dry_run_diff):
    """CellScoper CLI: run pipeline and write outputs."""
    os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.realpath(os.path.abspath(out_dir))

    # Load unified YAML config (params.yaml); allow overriding with --config
    cfg_path = config or os.environ.get("CELLSCOPE_CONFIG") or None
    default_cfg, default_used_path = _load_params_with_fallback(
        get_default_params_path(),
        allow_fallback=True,
        resolve_config=False,
    )
    if not default_cfg:
        console.print("[red]Warning: default params.yaml could not be loaded or is empty[/red]")
    user_cfg, user_used_path = (
        _load_params_with_fallback(
            cfg_path,
            allow_fallback=False,
            resolve_config=False,
        )
        if cfg_path
        else ({}, None)
    )
    try:
        cfg = resolve_dgi_configuration(
            _preserve_legacy_omm_policy(
                _merge_with_defaults(default_cfg, user_cfg),
                user_cfg,
            )
        )
    except ValueError as exc:
        raise click.ClickException(
            f"Invalid effective configuration: {exc}"
        ) from exc
    if no_show_sample:
        cfg["show_sample"] = False
    io_cfg = cfg.setdefault("io", {})
    if no_save_intermediate:
        io_cfg["write_intermediate"] = False
    if resume_policy is not None:
        io_cfg["resume_policy"] = str(resume_policy).lower()
    if resume_from is not None:
        io_cfg["resume_from"] = str(resume_from).lower()
    effective_resume_policy = str(
        io_cfg.get("resume_policy", "minimal")
    ).lower()
    if effective_resume_policy not in {"auto", "minimal", "force"}:
        raise click.ClickException(
            "io.resume_policy must be one of: auto, minimal, force"
        )
    cfg["run"] = {
        "resume": not no_resume,
        "resume_policy": effective_resume_policy,
        "resume_from": io_cfg.get("resume_from"),
        "save_intermediate": bool(
            io_cfg.get("write_intermediate", True)
        ),
        "show_sample": bool(cfg.get("show_sample", True)),
        "dry_run_diff": bool(dry_run_diff),
    }

    effective_config_path = os.path.join(out_dir, "effective_params.yaml")
    params_used_path = os.path.join(out_dir, "params_used.yaml")
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    previous_invocation = None
    # A SIGKILL/OOM/segmentation fault cannot execute the normal error handler,
    # so an older manifest may remain marked ``running``. The output-directory
    # lock is held by this invocation at this point, which proves no previous
    # writer is active; preserve that stale status as diagnostic history before
    # replacing the manifest for the new run.
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                previous_manifest = json.load(handle)
            if previous_manifest.get("status") == "running":
                previous_manifest["status"] = "interrupted"
                previous_manifest["interrupted_at"] = utc_now()
                previous_manifest["interrupted_reason"] = (
                    "Previous invocation did not finalize its manifest; "
                    "check cellscoper.error.log and kernel logs."
                )
                atomic_write_json(manifest_path, previous_manifest)
                previous_invocation = {
                    key: previous_manifest.get(key)
                    for key in (
                        "status",
                        "started_at",
                        "interrupted_at",
                        "interrupted_reason",
                    )
                }
        except Exception:
            # A malformed/stale manifest must not prevent a new run from
            # starting; the new manifest below will replace it atomically.
            pass
    try:
        atomic_write_yaml(effective_config_path, cfg)
    except Exception as exc:
        raise click.ClickException(
            f"Could not write effective configuration: {exc}"
        ) from exc

    previous_config_env = os.environ.get("CELLSCOPE_CONFIG")
    os.environ["CELLSCOPE_CONFIG"] = effective_config_path
    # Numeric thread pools initialize while importing the pipeline, so defer
    # that import until the effective configuration is available.
    from .pipeline import run_pipeline

    config_sha256 = sha256_mapping(cfg)
    console.print(
        "[cyan]Configuration[/cyan] "
        f"default={default_used_path} "
        f"user={user_used_path or 'none'} "
        f"effective={effective_config_path} "
        f"sha256={config_sha256[:12]}"
    )
    console.print(
        "[cyan]Runtime[/cyan] "
        f"resume={not no_resume} "
        f"resume_policy={effective_resume_policy} "
        f"resume_from={io_cfg.get('resume_from') or 'auto'} "
        f"write_intermediate={io_cfg.get('write_intermediate', True)} "
        f"dry_run={bool(dry_run_diff)}"
    )

    hash_inputs = bool(io_cfg.get("provenance_hash_inputs", True))
    console.print(
        "[cyan]Provenance[/cyan] "
        f"fingerprinting inputs (sha256={hash_inputs})"
    )
    try:
        input_provenance = build_input_provenance(
            spatial_path=spatial,
            cell_boundaries_path=cell_boundaries,
            nucleus_boundaries_path=nucleus_boundaries,
            gold_data=gold_data,
            hash_inputs=hash_inputs,
        )
        package_dir = os.path.dirname(__file__)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "started_at": utc_now(),
            "previous_invocation": previous_invocation,
            "cellscoper_version": __version__,
            "configuration": {
                "default_path": default_used_path,
                "user_path": user_used_path,
                "effective_path": effective_config_path,
                "effective_sha256": config_sha256,
            },
            "invocation": dict(cfg["run"]),
            "inputs": input_provenance,
            "references": bundled_reference_provenance(
                package_dir,
                hash_content=hash_inputs,
            ),
            "code": {
                **git_provenance(package_dir),
                **source_tree_provenance(package_dir),
            },
        }
        atomic_write_json(manifest_path, manifest)
    except Exception as exc:
        if previous_config_env is None:
            os.environ.pop("CELLSCOPE_CONFIG", None)
        else:
            os.environ["CELLSCOPE_CONFIG"] = previous_config_env
        raise click.ClickException(
            f"Could not build run provenance: {exc}"
        ) from exc
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
    except Exception as exc:
        console.print(
            f"[yellow]Could not summarize Module 3 settings: {exc}[/yellow]"
        )

    outputs = {}
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            t = progress.add_task("Starting pipeline", total=None)
            # Show input samples by default unless disabled via config/flag
            if cfg.get("show_sample", True):
                # Read only a small preview to avoid long startup delays on large files
                s_df = read_table_preview(spatial, nrows=5)
                c_df = read_table_preview(cell_boundaries, nrows=5)
                n_df = read_table_preview(nucleus_boundaries, nrows=5)
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
                save_intermediate=bool(
                    io_cfg.get("write_intermediate", True)
                ),
                lite_intermediate=bool(cfg.get("io", {}).get("lite_intermediate", False)),
                compression=cfg.get("io", {}).get("h5ad_compression", "lzf"),
                resume_from=cfg.get("io", {}).get("resume_from"),
                show_internal_progress=internal_progress,
                enable_annotation=_enable_annotation,
                final_format=("parquet" if str(cfg.get("io", {}).get("final_format", "csv")).lower()=="parquet" else "csv"),
                scvi_epochs=_scvi_epochs_cfg,
                dgi_epochs=_dgi_epochs_cfg,
                enable_dgi_sage=bool(cfg.get("module8", {}).get("enable_dgi_sage", False)),
                resume_policy=effective_resume_policy,
                dry_run_diff=bool(dry_run_diff),
                progress_callback=_cb,
                input_provenance=input_provenance,
            )
            console.print(
                "[cyan]Pipeline[/cyan] "
                f"annotation={_enable_annotation} "
                f"final_format={_run_kwargs['final_format']} "
                f"config={effective_config_path}"
            )

            outputs = run_pipeline(**_run_kwargs)
            progress.update(t, description="Finished")
        if dry_run_diff:
            manifest["status"] = "dry_run_completed"
        else:
            atomic_write_yaml(params_used_path, cfg)
            manifest["status"] = "completed"
        manifest["finished_at"] = utc_now()
        manifest["outputs"] = {
            key: (
                file_provenance(value, hash_content=False)
                if isinstance(value, str) and os.path.isfile(value)
                else value
            )
            for key, value in outputs.items()
        }
        atomic_write_json(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = utc_now()
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            atomic_write_json(manifest_path, manifest)
        except Exception as manifest_exc:
            console.print(
                "[yellow]Failed to update run manifest after error: "
                f"{manifest_exc}[/yellow]"
            )
        console.print(
            f"[red]Pipeline failed ({type(exc).__name__}): {exc}[/red]"
        )
        console.print(
            f"[yellow]See {os.path.join(out_dir, 'cellscoper.error.log')} "
            f"and {manifest_path}[/yellow]"
        )
        raise click.ClickException(str(exc)) from exc
    finally:
        if previous_config_env is None:
            os.environ.pop("CELLSCOPE_CONFIG", None)
        else:
            os.environ["CELLSCOPE_CONFIG"] = previous_config_env

    if dry_run_diff:
        console.print(
            f"[green]Dry-run complete.[/green] Manifest: {manifest_path}"
        )
        return
    console.print(
        "[green]Done.[/green] "
        f"params={params_used_path} manifest={manifest_path}"
    )
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
                    try:
                        import pyarrow.parquet as _pq
                        rows = int(
                            _pq.ParquetFile(final_table_path).metadata.num_rows
                        )
                    except Exception:
                        rows = len(_pd.read_parquet(final_table_path))
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
