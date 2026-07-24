from typing import Optional, Literal, Tuple, List, Callable, cast, Dict, Any, Union, Sequence
import os
import re
import math
import logging as _logging
from collections import defaultdict, Counter

# Optional early config load so runtime.blas_threads / runtime.n_jobs in params.yaml
# can cap BLAS/OMP threads before importing heavy numeric libs.
try:
    from .config import load_params_yaml as _load_params_yaml_threads
except Exception:  # pragma: no cover - defensive for import edge cases
    _load_params_yaml_threads = None

def _coerce_pos_int(v: object) -> Optional[int]:
    try:
        iv = int(v)  # type: ignore[arg-type]
        return iv if iv > 0 else None
    except Exception:
        return None

from tqdm.auto import tqdm
from scipy.stats import hypergeom, norm, rankdata, fisher_exact
from scipy.special import erfc
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# Configure BLAS/OMP threads before importing numpy/scipy.
# Priority: params.yaml runtime.blas_threads / runtime.n_jobs > env > fallback heuristic.
_runtime_cfg_threads = {}
try:
    if _load_params_yaml_threads:
        _runtime_cfg_threads = (_load_params_yaml_threads() or {}).get("runtime", {}) or {}
except Exception:
    _runtime_cfg_threads = {}

_cpu = os.cpu_count() or 8
_fallback_threads = max(1, min(32, _cpu // 2 or 1))
_yaml_threads = _coerce_pos_int(_runtime_cfg_threads.get("blas_threads")) or _coerce_pos_int(_runtime_cfg_threads.get("n_jobs"))
_threads_default = str(_yaml_threads or _fallback_threads)
_THREAD_ENV_DEFAULTS = {
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", _threads_default),
    "OMP_NUM_THREADS":      os.environ.get("OMP_NUM_THREADS", _threads_default),
    "MKL_NUM_THREADS":      os.environ.get("MKL_NUM_THREADS", _threads_default),
    "NUMEXPR_MAX_THREADS":  os.environ.get("NUMEXPR_MAX_THREADS", _threads_default),
    "BLIS_NUM_THREADS":     os.environ.get("BLIS_NUM_THREADS", _threads_default),
    "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", _threads_default),
    "ACCELERATE_NUM_THREADS": os.environ.get("ACCELERATE_NUM_THREADS", _threads_default),
}
for _k, _v in _THREAD_ENV_DEFAULTS.items():
    os.environ.setdefault(_k, str(_v))
del _THREAD_ENV_DEFAULTS, _k, _v, _cpu, _threads_default, _fallback_threads, _yaml_threads, _runtime_cfg_threads

import pandas as pd
import numpy as np
import anndata as ad
from sklearn.neighbors import KDTree
def _kdtree_query(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build KDTree and query with tuned params; returns (indices, distances).
    Points must be float32 array of shape (n, 2). k will be clamped to [1, n].
    """
    n = points.shape[0]
    if n == 0:
        return None, None
    k = max(1, min(int(k), n))
    leaf = max(20, min(64, n // 10 if n >= 100 else 20))
    # breadth_first/sort_results improve determinism; avoid dualtree for small n
    kdt = KDTree(points, leaf_size=leaf)
    try:
        dist, ind = kdt.query(points, k=k, return_distance=True, breadth_first=True, sort_results=True)
    except TypeError:
        # Fallback for older sklearn without options
        dist, ind = kdt.query(points, k=k)
    ind = np.asarray(ind)
    if ind.dtype != np.int64:
        ind = ind.astype(np.int64, copy=False)
    return ind, dist
from matplotlib.path import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin
import scipy.sparse as sp
import gc


METASPOT_COL = "metaspot"
METASPOT_LOCAL_COL = "metaspot_in_cell"
_LEGACY_METASPOT_COL = "cell_sub"
_LEGACY_METASPOT_LOCAL_COL = "cluster_in_cell"


def _migrate_legacy_metaspot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize historical MetaSpot columns when loading old checkpoints."""
    rename = {}
    if METASPOT_COL not in df.columns and _LEGACY_METASPOT_COL in df.columns:
        rename[_LEGACY_METASPOT_COL] = METASPOT_COL
    if (
        METASPOT_LOCAL_COL not in df.columns
        and _LEGACY_METASPOT_LOCAL_COL in df.columns
    ):
        rename[_LEGACY_METASPOT_LOCAL_COL] = METASPOT_LOCAL_COL
    if rename:
        df = df.rename(columns=rename)
    stale = [
        column
        for column in (_LEGACY_METASPOT_COL, _LEGACY_METASPOT_LOCAL_COL)
        if column in df.columns
    ]
    if stale:
        df = df.drop(columns=stale)
    return df


def _migrate_legacy_metaspot_obs(adata: ad.AnnData) -> ad.AnnData:
    """Normalize historical MetaSpot columns in an AnnData checkpoint."""
    rename = {}
    if METASPOT_COL not in adata.obs.columns and _LEGACY_METASPOT_COL in adata.obs.columns:
        rename[_LEGACY_METASPOT_COL] = METASPOT_COL
    if (
        METASPOT_LOCAL_COL not in adata.obs.columns
        and _LEGACY_METASPOT_LOCAL_COL in adata.obs.columns
    ):
        rename[_LEGACY_METASPOT_LOCAL_COL] = METASPOT_LOCAL_COL
    if rename:
        adata.obs = adata.obs.rename(columns=rename)
    stale = [
        column
        for column in (_LEGACY_METASPOT_COL, _LEGACY_METASPOT_LOCAL_COL)
        if column in adata.obs.columns
    ]
    if stale:
        adata.obs = adata.obs.drop(columns=stale)
    return adata


class _LocalCompositionModel:
    """Compact local-expression transformer used one cell at a time."""

    def __init__(
        self,
        gene2idx: Dict[str, int],
        transformer: Any,
        idf_vec: np.ndarray,
        n_components: int,
        knn_k: int,
        weight_mode: str,
        tf_norm: bool,
        use_tfidf: bool,
    ):
        self.gene2idx = gene2idx
        self.transformer = transformer
        self.idf_vec = np.asarray(idf_vec, dtype=np.float32)
        self.n_components = int(n_components)
        self.knn_k = int(knn_k)
        self.weight_mode = str(weight_mode)
        self.tf_norm = bool(tf_norm)
        self.use_tfidf = bool(use_tfidf)

    @property
    def feature_names(self) -> List[str]:
        return [f"comp_emb{i}" for i in range(self.n_components)]

    def transform_cell(self, dfc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(dfc)
        if n == 0:
            return (
                np.zeros((0, self.n_components), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        points = dfc[["x_location", "y_location"]].to_numpy(np.float32, copy=False)
        ind, dist = _kdtree_query(points, k=min(self.knn_k, n))
        gene_idx = (
            dfc["feature_name"]
            .astype(str)
            .map(self.gene2idx)
            .fillna(-1)
            .to_numpy(dtype=np.int64, copy=False)
        )
        k = ind.shape[1]

        if self.weight_mode == "gaussian":
            radius = dist[:, -1].astype(np.float32) + 1e-6
            weights = np.exp(-((dist / radius[:, None]) ** 2)).astype(np.float32)
        elif self.weight_mode == "inv":
            weights = (1.0 / (dist + 1e-3)).astype(np.float32)
        else:
            weights = np.ones_like(dist, dtype=np.float32)

        rows = np.repeat(np.arange(n, dtype=np.int64), k)
        neighbors = ind.ravel().astype(np.int64, copy=False)
        cols = gene_idx[neighbors]
        values = weights.ravel()
        valid = (cols >= 0) & (cols < len(self.gene2idx))
        bow = sp.coo_matrix(
            (values[valid], (rows[valid], cols[valid])),
            shape=(n, len(self.gene2idx)),
            dtype=np.float32,
        ).tocsr()

        if self.tf_norm:
            row_sum = np.asarray(bow.sum(axis=1)).ravel().astype(np.float32)
            row_sum[row_sum == 0] = 1.0
            bow = bow.multiply(1.0 / row_sum[:, None]).tocsr()
        if self.use_tfidf:
            bow = bow.multiply(self.idf_vec[np.newaxis, :]).tocsr()

        z = self.transformer.transform(bow)
        if sp.issparse(z):
            z = z.toarray()
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        if z.shape[1] < self.n_components:
            z = np.pad(z, ((0, 0), (0, self.n_components - z.shape[1])))
        elif z.shape[1] > self.n_components:
            z = z[:, :self.n_components]

        radius = dist[:, -1].astype(np.float32) + 1e-3
        density = (k / (np.pi * radius * radius)).astype(np.float32)
        return z, density


# Set Scanpy parallel jobs if available (non-intrusive performance tweak)
try:
    import scanpy as sc
    sc.settings.n_jobs = max(1, int(os.environ.get("SCANPY_N_JOBS", "4")))
except Exception:
    pass

# Lightweight helper to downcast numeric dtypes to reduce memory and improve speed
def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for c in df.columns:
            col = df[c]
            if pd.api.types.is_float_dtype(col):
                df[c] = col.astype(np.float32)
            elif pd.api.types.is_integer_dtype(col):
                df[c] = col.astype(np.int32)
    except Exception:
        pass
    return df
def _progress_iter(iterator, desc: str = "", total: Optional[int] = None, show: bool = False):
    """Terminal-friendly progress iterator with stage messages and ETA.
    - TTY: Rich single-line bar with ETA; prints start/done summaries.
    - Non-TTY: milestone prints at 0/25/50/75/100% with ETA + done summary.
    """
    if not show:
        return iterator
    import time as _time
    _desc = desc or "Working"
    # Try infer total if not provided
    if total is None:
        try:
            total = len(iterator)  # may fail for generators/groupby
        except Exception:
            total = None
    # Attempt Rich progress (best single-line UX)
    try:
        from rich.console import Console as _Console
        from rich.progress import (
            Progress as _Progress,
            SpinnerColumn as _Spinner,
            BarColumn as _Bar,
            TextColumn as _Text,
            TimeElapsedColumn as _TimeElapsed,
            TimeRemainingColumn as _TimeRemain,
            TaskProgressColumn as _TaskProg,
            MofNCompleteColumn as _MofN,
        )
        _con = _Console(stderr=True)
        if _con.is_terminal:  # TTY only
            start_t = _time.perf_counter()
            _con.print(f">> Start: {_desc}")
            if total is None:
                columns = [
                    _Spinner(),
                    _Text(" {task.description}"),
                    _Text("  elapsed:"), _TimeElapsed(),
                ]
                prog = _Progress(*columns, transient=True, console=_con)
                def _gen_rich_spin():
                    task = prog.add_task(_desc, total=None)
                    with prog:
                        c = 0
                        for item in iterator:
                            c += 1
                            prog.advance(task, 1)
                            yield item
                    elapsed = _time.perf_counter() - start_t
                    rate = f"{c/elapsed:,.1f}/s" if elapsed > 0 else "n/a"
                    _con.print(f"<< Done: {_desc} in {elapsed:0.2f}s (items={c}, rate={rate})")
                return _gen_rich_spin()
            else:
                columns = [
                    _Text("{task.description}: "),
                    _Bar(),
                    _TaskProg(),
                    _Text("  "), _MofN(),
                    _Text("  elapsed:"), _TimeElapsed(),
                    _Text("  ETA:"), _TimeRemain(),
                ]
                prog = _Progress(*columns, transient=True, console=_con)
                def _gen_rich_bar():
                    task = prog.add_task(_desc, total=total)
                    with prog:
                        c = 0
                        for item in iterator:
                            c += 1
                            prog.update(task, completed=c)
                            yield item
                    elapsed = _time.perf_counter() - start_t
                    rate = f"{c/elapsed:,.1f}/s" if elapsed > 0 else "n/a"
                    _con.print(f"<< Done: {_desc} in {elapsed:0.2f}s ({c}/{total}, rate={rate})")
                return _gen_rich_bar()
    except Exception:
        pass
    # Fallback: milestone-only prints with ETA
    milestones = [0, 25, 50, 75, 100] if (isinstance(total, int) and total >= 20) else [0]
    def _eta_str(done: int, tot: int, elapsed: float) -> str:
        if tot and done > 0:
            frac = done / tot
            remain = max((elapsed / frac) - elapsed, 0.0)
            # format mm:ss
            mm = int(remain // 60); ss = int(remain % 60)
            return f"ETA {mm:02d}:{ss:02d}"
        return "ETA --:--"
    def _gen_milestone():
        start = _time.perf_counter(); cnt = 0
        if total:
            print(f">> Start: {_desc} (0/{total})")
            print(f"{_desc}: 0% (0/{total}) {_eta_str(0, total, 0.0)}")
            if milestones and milestones[0] == 0:
                milestones.pop(0)
        else:
            print(f">> Start: {_desc}")
        for item in iterator:
            cnt += 1
            if total:
                perc = int(cnt * 100 / max(total, 1))
                while len(milestones) and perc >= milestones[0]:
                    m = milestones.pop(0)
                    eta = _eta_str(cnt, total, _time.perf_counter() - start)
                    print(f"{_desc}: {m}% ({cnt}/{total}) {eta}")
            yield item
        elapsed = _time.perf_counter() - start
        rate = f"{(cnt/elapsed):,.1f}/s" if elapsed > 0 else "n/a"
        if total:
            print(f"<< Done: {_desc} in {elapsed:0.2f}s ({cnt}/{total}, rate={rate})")
        else:
            print(f"<< Done: {_desc} in {elapsed:0.2f}s (items={cnt}, rate={rate})")
    return _gen_milestone()
from .io import (
    read_spatial_transcripts,
    read_boundaries,
    ensure_dir,
    save_checkpoint,
    load_checkpoint,
)
from .logging_utils import setup_logger
from .config import load_params_yaml, fingerprint_stage, fingerprint_modules_set
import runpy


def load_inputs(spatial_path: str, cell_boundaries_path: str, nucleus_boundaries_path: str, out_dir: str, resume: bool, save_intermediate: bool):
    cp_spatial = _checkpoint_path(out_dir, "spatial")
    cp_cell = _checkpoint_path(out_dir, "cell_boundaries")
    cp_nuc = _checkpoint_path(out_dir, "nucleus_boundaries")
    cache_inputs = bool(resume or save_intermediate)

    if resume and os.path.exists(cp_spatial):
        spatial_df = load_checkpoint(cp_spatial)
    else:
        spatial_df = read_spatial_transcripts(spatial_path)
        if cache_inputs:
            save_checkpoint(spatial_df, cp_spatial)

    if resume and os.path.exists(cp_cell):
        cell_df = load_checkpoint(cp_cell)
    else:
        cell_df = read_boundaries(cell_boundaries_path)
        if cache_inputs:
            save_checkpoint(cell_df, cp_cell)

    if resume and os.path.exists(cp_nuc):
        nucleus_df = load_checkpoint(cp_nuc)
    else:
        nucleus_df = read_boundaries(nucleus_boundaries_path)
        if cache_inputs:
            save_checkpoint(nucleus_df, cp_nuc)

    return spatial_df, cell_df, nucleus_df


def _load_df_checkpoint_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return load_checkpoint(path)
    except Exception:
        return None


def _resume_raw_inputs(out_dir: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    cp_spatial = _checkpoint_path(out_dir, "spatial")
    cp_cell = _checkpoint_path(out_dir, "cell_boundaries")
    cp_nuc = _checkpoint_path(out_dir, "nucleus_boundaries")
    spatial_df = _load_df_checkpoint_if_exists(cp_spatial)
    cell_df = _load_df_checkpoint_if_exists(cp_cell)
    nuc_df = _load_df_checkpoint_if_exists(cp_nuc)
    if spatial_df is None or cell_df is None or nuc_df is None:
        return None
    return spatial_df, cell_df, nuc_df


def _resume_preprocessed_inputs(out_dir: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    cp_spatial = _checkpoint_path(out_dir, "spatial_preproc")
    cp_cell = _checkpoint_path(out_dir, "cell_preproc")
    cp_nuc = _checkpoint_path(out_dir, "nucleus_preproc")
    spatial_df = _load_df_checkpoint_if_exists(cp_spatial)
    cell_df = _load_df_checkpoint_if_exists(cp_cell)
    nuc_df = _load_df_checkpoint_if_exists(cp_nuc)
    if spatial_df is None or cell_df is None or nuc_df is None:
        return None
    return spatial_df, cell_df, nuc_df


def _load_dataframe_checkpoint(base_dir: str, stem: str) -> Optional[pd.DataFrame]:
    csv_path = os.path.join(base_dir, f"{stem}.csv")
    pq_path = os.path.join(base_dir, f"{stem}.parquet")
    try:
        if os.path.exists(pq_path):
            return _migrate_legacy_metaspot_columns(pd.read_parquet(pq_path))
        if os.path.exists(csv_path):
            return _migrate_legacy_metaspot_columns(pd.read_csv(csv_path))
    except Exception:
        return None
    return None


def _load_adata_checkpoint(path: str) -> Optional[ad.AnnData]:
    if not os.path.exists(path):
        return None
    try:
        return _migrate_legacy_metaspot_obs(ad.read_h5ad(path))
    except Exception:
        return None


def _resume_build_outputs(out_dir: str) -> Optional[Tuple[ad.AnnData, Optional[ad.AnnData], Optional[pd.DataFrame]]]:
    base = _intermediate_dir(out_dir)
    a1_path = os.path.join(base, "module5_adata1.h5ad")
    adata1 = _load_adata_checkpoint(a1_path)
    if adata1 is None:
        adata1 = _load_adata_checkpoint(os.path.join(out_dir, "adata1.h5ad"))
    if adata1 is None:
        return None
    adata2 = _load_adata_checkpoint(os.path.join(base, "module6_adata2.h5ad"))
    if adata2 is None:
        adata2 = _load_adata_checkpoint(os.path.join(out_dir, "adata2.h5ad"))
    final_df = _load_dataframe_checkpoint(base, "module5_final_data")
    used_final_output = False
    if final_df is None:
        # Lite-intermediate mode may intentionally omit the multi-million-row
        # build table even though the final Parquet/CSV contains the same
        # upstream membership columns. Reuse that final table for reannotation
        # only after strict MetaSpot ownership/count alignment checks.
        final_df = _load_dataframe_checkpoint(out_dir, "final_data")
        used_final_output = final_df is not None
    if final_df is None:
        return adata1, adata2, None

    if used_final_output:
        if not {"metaspot", "cell_id"}.issubset(final_df.columns):
            return adata1, adata2, None
        expected_ids = pd.Index(adata1.obs_names.astype(str))
        observed_ids = pd.Index(
            pd.unique(final_df["metaspot"].astype(str))
        )
        if (
            len(expected_ids.difference(observed_ids))
            or len(observed_ids.difference(expected_ids))
        ):
            return adata1, adata2, None
        ownership_frame = pd.DataFrame(
            {
                "metaspot": final_df["metaspot"].astype(str),
                "cell_id": final_df["cell_id"].astype(str),
            }
        )
        owner_groups = ownership_frame.groupby(
            "metaspot",
            observed=True,
        )["cell_id"]
        if owner_groups.nunique().ne(1).any():
            return adata1, adata2, None
        owners = owner_groups.first()
        expected_owners = pd.Series(
            adata1.obs["cell_id"].astype(str).to_numpy(),
            index=expected_ids,
        )
        if not owners.reindex(expected_ids).equals(expected_owners):
            return adata1, adata2, None
        if "log_n" in adata1.obs.columns:
            transcript_counts = (
                final_df["metaspot"].astype(str).value_counts()
            ).reindex(expected_ids)
            expected_log_n = pd.to_numeric(
                adata1.obs["log_n"],
                errors="coerce",
            ).to_numpy(dtype=float)
            observed_log_n = np.log1p(
                transcript_counts.to_numpy(dtype=float)
            )
            if not np.allclose(
                expected_log_n,
                observed_log_n,
                rtol=1e-5,
                atol=1e-5,
                equal_nan=False,
            ):
                return adata1, adata2, None
        if (
            adata2 is not None
            and "meta-domain" in final_df.columns
        ):
            expected_domains = pd.Index(adata2.obs_names.astype(str))
            observed_domains = pd.Index(
                pd.unique(final_df["meta-domain"].astype(str))
            )
            if (
                len(expected_domains.difference(observed_domains))
                or len(observed_domains.difference(expected_domains))
            ):
                return adata1, adata2, None
        _logging.getLogger("cellscope").info(
            "Resume: using aligned final_data as the build-table checkpoint "
            "(lite-intermediate mode)"
        )
    return adata1, adata2, final_df


def _resume_annotation_outputs(out_dir: str) -> Optional[Tuple[Optional[ad.AnnData], Optional[pd.DataFrame]]]:
    base = _intermediate_dir(out_dir)
    final_df = _load_dataframe_checkpoint(base, "annotated_final_csv")
    if final_df is None:
        return None
    ann_adata2 = _load_adata_checkpoint(os.path.join(base, "annotated_adata2.h5ad"))
    return ann_adata2, final_df


def preprocess(spatial_df: pd.DataFrame, cell_df: pd.DataFrame, nucleus_df: pd.DataFrame, show_internal_progress: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Keep only cells where `overlaps_nucleus` has both 0 and 1; enforce consistent filtering across the three input tables.
    """
    if "overlaps_nucleus" not in spatial_df.columns:
        raise ValueError(
            "spatial_df is missing overlaps_nucleus; clear the spatial checkpoint and reload from the CSV"
        )
    unique_counts = (
        spatial_df.groupby("cell_id")["overlaps_nucleus"].nunique().reset_index(name="n_unique")
    )
    valid_cells = unique_counts.query("n_unique == 2")["cell_id"]

    spatial_df = spatial_df.query("cell_id in @valid_cells").copy()
    cell_df = cell_df.query("cell_id in @valid_cells").copy()
    nucleus_df = nucleus_df.query("cell_id in @valid_cells").copy()

    return spatial_df, cell_df, nucleus_df


def _stretch_signed_geometry(values: np.ndarray, cell_size: int) -> np.ndarray:
    """Robustly scale signed geometry while keeping the nuclear boundary at zero."""
    x = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(x)
    if not finite.any():
        return x

    if cell_size >= 16:
        negative = finite & (x < 0)
        positive = finite & (x > 0)
        if negative.any():
            neg_scale = float(np.nanpercentile(np.abs(x[negative]), 95.0))
            if np.isfinite(neg_scale) and neg_scale > 1e-6:
                x[negative] /= neg_scale
        if positive.any():
            pos_scale = float(np.nanpercentile(x[positive], 95.0))
            if np.isfinite(pos_scale) and pos_scale > 1e-6:
                x[positive] /= pos_scale

    x = np.clip(x, -2, 2)
    adaptive_band = 0.05 * (1.0 + 0.5 * np.exp(-cell_size / 1000))
    x[np.abs(x) < adaptive_band] = 0.0
    return x.astype(np.float32, copy=False)


def build_point_geometry_v2(
    transcripts_df: pd.DataFrame,
    nucleus_boundaries_df: pd.DataFrame,
    cell_boundaries_df: pd.DataFrame,
    sampled_cells: List,
    include_directional: bool = True,
    winsor_perc=(1, 99),
    show_internal_progress: bool = False,
) -> pd.DataFrame:
    DTYPE = np.float32

    def _group_chunks(df: pd.DataFrame, chunk_size: int):
        buf = []
        for pair in df.groupby('cell_id', sort=False):
            buf.append(pair)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def _winsorize_inplace(df: pd.DataFrame, cols, perc=(1, 99)) -> None:
        lo_p, hi_p = perc
        for c in cols:
            x = df[c].to_numpy()
            lo, hi = np.nanpercentile(x, [lo_p, hi_p])
            df[c] = np.clip(x, lo, hi).astype(DTYPE, copy=False)

    def _build_kdtrees(boundary_df: pd.DataFrame, cells, desc="KDTree") -> dict:
        kd = {}
        sub = boundary_df[boundary_df['cell_id'].isin(cells)]
        it = sub.groupby('cell_id', sort=False)
        iterator = _progress_iter(it, desc=f'Building {desc}', total=it.ngroups, show=show_internal_progress)
        for cid, g in iterator:
            coords = g[['vertex_x', 'vertex_y']].to_numpy()
            kd[cid] = KDTree(coords) if len(coords) else None
        return kd

    def _compute_min_dist(df_points: pd.DataFrame, kdtrees: dict, label: str) -> pd.Series:
        parts = []
        it = df_points.groupby('cell_id', sort=False)
        iterator = _progress_iter(it, desc='Compute '+label, total=it.ngroups, show=show_internal_progress)
        for cid, g in iterator:
            kt = kdtrees.get(cid, None)
            if kt is None:
                parts.append(pd.Series(np.nan, index=g.index))
            else:
                q = g[['x_location', 'y_location']].to_numpy()
                d, _ = kt.query(q, k=1, return_distance=True)
                parts.append(pd.Series(d[:, 0].astype(DTYPE), index=g.index))
        return pd.concat(parts)

    def _build_nucleus_paths(nucleus_df: pd.DataFrame, cells) -> dict:
        paths = {}
        sub = nucleus_df[nucleus_df['cell_id'].isin(cells)]
        it = sub.groupby('cell_id', sort=False)
        iterator = _progress_iter(it, desc='Build nucleus Path', total=it.ngroups, show=show_internal_progress)
        for cid, g in iterator:
            pts = g[['vertex_x', 'vertex_y']].to_numpy()
            paths[cid] = Path(pts, closed=True) if len(pts) >= 3 else None
        return paths

    # Avoid copying the full transcript table. We keep the original rows and
    # build geometry columns in chunks so Module 1 can run without doubling RAM.
    df = transcripts_df[transcripts_df['cell_id'].isin(sampled_cells)]
    cell_kd = _build_kdtrees(cell_boundaries_df, sampled_cells, desc="Cell KDTree")
    nuc_kd = _build_kdtrees(nucleus_boundaries_df, sampled_cells, desc="Nucleus KDTree")
    use_input_nucleus_flags = "overlaps_nucleus" in df.columns
    nuc_paths = (
        {}
        if use_input_nucleus_flags
        else _build_nucleus_paths(nucleus_boundaries_df, sampled_cells)
    )

    # Process grouped rows in chunks to avoid materializing several full-size
    # temporary Series at once.
    d_cell_parts = []
    d_nuc_parts = []
    flags = []
    group_iter = _group_chunks(df, chunk_size=128)
    total_groups = int(df['cell_id'].nunique(dropna=False)) if len(df) else 0
    iterator = _progress_iter(group_iter, desc='Point geometry chunks', total=max(1, (total_groups + 127) // 128), show=show_internal_progress)
    for chunk in iterator:
        for cid, g in chunk:
            kt = cell_kd.get(cid, None)
            if kt is None:
                d_cell_parts.append(pd.Series(np.nan, index=g.index))
            else:
                q = g[['x_location', 'y_location']].to_numpy()
                d, _ = kt.query(q, k=1, return_distance=True)
                d_cell_parts.append(pd.Series(d[:, 0].astype(DTYPE), index=g.index))

            kt = nuc_kd.get(cid, None)
            if kt is None:
                d_nuc_parts.append(pd.Series(np.nan, index=g.index))
            else:
                q = g[['x_location', 'y_location']].to_numpy()
                d, _ = kt.query(q, k=1, return_distance=True)
                d_nuc_parts.append(pd.Series(d[:, 0].astype(DTYPE), index=g.index))

            if use_input_nucleus_flags:
                input_flags = (
                    pd.to_numeric(g["overlaps_nucleus"], errors="coerce")
                    .fillna(0)
                    .ne(0)
                )
                flags.append(pd.Series(input_flags.to_numpy(), index=g.index))
            else:
                p = nuc_paths.get(cid, None)
                if p is None:
                    flags.append(pd.Series(False, index=g.index))
                else:
                    pts = g[['x_location', 'y_location']].to_numpy()
                    flags.append(pd.Series(p.contains_points(pts), index=g.index))

    df['d_cell'] = pd.concat(d_cell_parts)
    df['d_nuc'] = pd.concat(d_nuc_parts)
    df['in_nucleus'] = pd.concat(flags)

    d_cell = df['d_cell'].to_numpy(dtype=DTYPE, copy=False)
    d_nuc = df['d_nuc'].to_numpy(dtype=DTYPE, copy=False)
    denom = (d_cell + d_nuc).astype(DTYPE, copy=False)
    denom[denom <= 0] = np.nan

    inside = df['in_nucleus'].to_numpy()
    df['r_signed'] = np.where(inside, -d_cell/denom, d_nuc/denom).astype(DTYPE, copy=False)
    df['d_cell_norm'] = (d_cell/denom).astype(DTYPE, copy=False)
    df['d_nuc_norm'] = (d_nuc/denom).astype(DTYPE, copy=False)

    if include_directional:
        nuc_cent = (nucleus_boundaries_df[nucleus_boundaries_df['cell_id'].isin(sampled_cells)]
                    .groupby('cell_id', sort=False)
                    .agg(nuc_x=('vertex_x', 'mean'), nuc_y=('vertex_y', 'mean')))
        df = df.join(nuc_cent, on='cell_id')
        nan_c = df['nuc_x'].isna() | df['nuc_y'].isna()
        if nan_c.any():
            fb = (df.loc[nan_c]
                  .groupby('cell_id')[['x_location', 'y_location']]
                  .mean()
                  .rename(columns={'x_location': 'nuc_x', 'y_location': 'nuc_y'}))
            df.update(fb, overwrite=False)
        dx = (df['x_location'] - df['nuc_x']).to_numpy(dtype=DTYPE, copy=False)
        dy = (df['y_location'] - df['nuc_y']).to_numpy(dtype=DTYPE, copy=False)
        norm = np.sqrt(dx**2 + dy**2)
        norm[norm <= 0] = np.nan
        df['cos_theta'] = (dx / norm).astype(DTYPE, copy=False)
        df['sin_theta'] = (dy / norm).astype(DTYPE, copy=False)
        st = df['sin_theta'].to_numpy(dtype=DTYPE, copy=False)
        ct = df['cos_theta'].to_numpy(dtype=DTYPE, copy=False)
        df['sin2_theta'] = (2.0 * st * ct).astype(DTYPE, copy=False)
        df['cos2_theta'] = (ct*ct - st*st).astype(DTYPE, copy=False)

    # Nuclear radius estimation and per-cell stretching
    nuc_cent_xy = (nucleus_boundaries_df.groupby('cell_id', sort=False)
                   .agg(_nx=('vertex_x', 'mean'), _ny=('vertex_y', 'mean')))
    nuc_bd = (nucleus_boundaries_df
              .join(nuc_cent_xy, on='cell_id')
              .assign(_r=lambda t: np.sqrt((t['vertex_x']-t['_nx'])**2 + (t['vertex_y']-t['_ny'])**2)))
    rad_df = nuc_bd.groupby('cell_id', sort=False)['_r'].median().rename('_nuc_r')
    df = df.join(rad_df, on='cell_id')

    _nuc_r = df['_nuc_r'].to_numpy(dtype=DTYPE, copy=False)
    _nuc_r = np.where((~np.isfinite(_nuc_r)) | (_nuc_r <= 0), np.nan, _nuc_r)
    rs_norm = df['r_signed'].to_numpy(dtype=DTYPE, copy=False) / np.nan_to_num(_nuc_r, nan=1.0)

    df['_rs_tmp'] = rs_norm.astype(DTYPE, copy=False)
    cell_sizes = df.groupby('cell_id').size()
    df['_cell_size'] = df['cell_id'].map(cell_sizes)
    r_parts = []
    group_iter = _group_chunks(df, chunk_size=128)
    iterator = _progress_iter(group_iter, desc='Stretch r_signed chunks', total=max(1, (total_groups + 127) // 128), show=show_internal_progress)
    for chunk in iterator:
        for cid, g in chunk:
            r_parts.append(pd.Series(_stretch_signed_geometry(g['_rs_tmp'].values, len(g)), index=g.index))
    df['r_signed'] = pd.concat(r_parts).astype(DTYPE)
    df.drop(columns=['_rs_tmp', '_nuc_r', '_cell_size'], errors='ignore', inplace=True)

    _winsorize_inplace(df, ['d_cell', 'd_nuc', 'r_signed', 'd_cell_norm', 'd_nuc_norm'], winsor_perc)
    return df


def build_composition_embeddings(point_df: pd.DataFrame, transcripts_df: pd.DataFrame, sampled_cell_ids: List, show_internal_progress: bool = False) -> Tuple[pd.DataFrame, List[str], Optional[Any]]:
    RANDOM_STATE = 42
    # Build vocabulary
    # Keep only the columns Module 2 actually needs; carrying the full point table
    # through this stage is a major memory multiplier.
    # Avoid huge memory spikes when sampled_cell_ids is a full row-level list.
    use_full_source = False
    try:
        use_full_source = (
            sampled_cell_ids is None
            or len(sampled_cell_ids) >= transcripts_df['cell_id'].nunique(dropna=False)
        )
    except Exception:
        use_full_source = False
    if use_full_source:
        sub_tr_all = transcripts_df
    else:
        sampled_ids = pd.unique(np.asarray(sampled_cell_ids))
        sub_tr_all = transcripts_df[transcripts_df['cell_id'].isin(sampled_ids)]
    gene_counts = sub_tr_all['feature_name'].astype(str).value_counts()
    # defaults; will be overridden by config when available
    TOP_G = 384
    N_COMP = 24
    K_NEI = 32
    MAX_ROWS_PER_CELL_FOR_SVD = 600
    MAX_SVD_ROWS = 120_000
    BOW_WEIGHT_MODE = 'gaussian'  # gaussian | uniform
    USE_TFIDF = True
    TF_NORM = True
    IDF_MODE = 'log1p'  # 'log1p' | 'log' | 'none'
    STREAM_LOCAL = True
    PARALLEL_CHUNK_CELLS = 128
    try:
        _cfg = load_params_yaml()
        _m2 = _cfg.get('module2', {}) if _cfg else {}
        TOP_G = int(_m2.get('top_g', TOP_G))
        N_COMP = int(_m2.get('svd_n_components', N_COMP))
        K_NEI = int(_m2.get('bow_knn_k', K_NEI))
        MAX_ROWS_PER_CELL_FOR_SVD = int(_m2.get('max_rows_per_cell_for_svd', MAX_ROWS_PER_CELL_FOR_SVD))
        MAX_SVD_ROWS = int(_m2.get('max_svd_rows', MAX_SVD_ROWS))
        BOW_WEIGHT_MODE = str(_m2.get('bow_weight_mode', BOW_WEIGHT_MODE)).lower()
        USE_TFIDF = bool(_m2.get('use_tfidf', USE_TFIDF))
        TF_NORM = bool(_m2.get('tf_norm', TF_NORM))
        IDF_MODE = str(_m2.get('idf_mode', IDF_MODE)).lower()
        STREAM_LOCAL = bool(_m2.get('stream_local_embeddings', STREAM_LOCAL))
        PARALLEL_CHUNK_CELLS = max(1, int(_m2.get('parallel_chunk_cells', PARALLEL_CHUNK_CELLS)))
    except Exception:
        pass
    vocab_genes = gene_counts.head(TOP_G).index.astype(str).tolist()
    gene2idx = {g: i for i, g in enumerate(vocab_genes)}

    # For real runs, the old per-transcript KNN + BoW path still produces large
    # temporary matrices. A cell-level summary embedding is much cheaper and is
    # sufficient for downstream cell/subcell clustering because the geometry
    # features already stay transcript-level.
    force_cell_summary = False
    try:
        force_cell_summary = os.environ.get('CELLSCOPE_CELL_SUMMARY_ONLY', '').lower() in ('1', 'true', 'yes')
    except Exception:
        force_cell_summary = False

    if force_cell_summary:
        cell_ids = []
        row_idx = []
        col_idx = []
        values = []
        local_density = []
        df_vec = np.zeros(len(vocab_genes), dtype=np.float32)

        group_iter = _progress_iter(
            transcripts_df.groupby('cell_id', sort=False),
            desc='Cell composition summary',
            total=int(transcripts_df['cell_id'].nunique(dropna=False)) if len(transcripts_df) else 0,
            show=show_internal_progress,
        )
        for cid, dfc in group_iter:
            feat = dfc['feature_name'].astype(str).map(gene2idx).fillna(-1).to_numpy(dtype=np.int32, copy=False)
            feat = feat[feat >= 0]
            if feat.size == 0:
                continue
            counts = np.bincount(feat, minlength=len(vocab_genes)).astype(np.float32, copy=False)
            nz = np.flatnonzero(counts)
            r = len(cell_ids)
            cell_ids.append(cid)
            row_idx.extend([r] * len(nz))
            col_idx.extend(nz.tolist())
            values.extend(counts[nz].tolist())
            df_vec[nz] += 1.0
            local_density.append(float(np.log1p(len(dfc))))

        if len(cell_ids) == 0:
            raise RuntimeError('Cell summary composition samples empty.')

        cell_bow = sp.csr_matrix(
            (np.asarray(values, dtype=np.float32), (np.asarray(row_idx, dtype=np.int64), np.asarray(col_idx, dtype=np.int64))),
            shape=(len(cell_ids), len(vocab_genes)),
            dtype=np.float32,
        )

        N_rows = float(cell_bow.shape[0])
        if USE_TFIDF and N_rows > 0:
            if IDF_MODE == 'none':
                idf_vec = np.ones_like(df_vec, dtype=np.float32)
            elif IDF_MODE == 'log':
                idf_vec = np.log(np.maximum(N_rows / np.maximum(df_vec, 1.0), 1.0)).astype(np.float32)
            else:
                idf_vec = np.log1p(N_rows / (1.0 + df_vec)).astype(np.float32)
            cell_bow = cell_bow.multiply(idf_vec[np.newaxis, :]).tocsr()

        gc.collect()
        if os.environ.get('CELLSCOPE_ULTRALOW', '').lower() in ('1', 'true', 'yes') or cell_bow.shape[0] > 50000:
            from sklearn.random_projection import SparseRandomProjection
            emb_model = SparseRandomProjection(n_components=N_COMP, random_state=RANDOM_STATE)
            emb_model.fit(cell_bow)
            transformer = emb_model
        else:
            svd_local = TruncatedSVD(n_components=N_COMP, random_state=RANDOM_STATE)
            svd_local.fit(cell_bow)
            transformer = svd_local

        # Transform in batches to avoid materializing the full dense cell embedding
        # matrix at once. The cell-level output stays small enough to concatenate.
        batch_size = max(1024, min(8192, int(np.ceil(cell_bow.shape[0] / 32))))
        cell_emb_parts = []
        for start in range(0, cell_bow.shape[0], batch_size):
            end = min(start + batch_size, cell_bow.shape[0])
            comp_part = transformer.transform(cell_bow[start:end])
            if sp.issparse(comp_part):
                comp_part = comp_part.toarray()
            comp_part = np.asarray(comp_part, dtype=np.float32)
            if comp_part.ndim == 1:
                comp_part = comp_part.reshape(-1, 1)
            if comp_part.shape[1] == 0:
                comp_part = np.zeros((comp_part.shape[0], 1), dtype=np.float32)
            comp_cols = [f'comp_emb{i}' for i in range(comp_part.shape[1])]
            part_index = pd.Index(cell_ids[start:end], name='cell_id')
            part_df = pd.DataFrame(comp_part, index=part_index, columns=comp_cols)
            part_df['local_density'] = np.asarray(local_density[start:end], dtype=np.float32)
            cell_emb_parts.append(part_df)

        cell_emb = pd.concat(cell_emb_parts, axis=0) if len(cell_emb_parts) > 1 else cell_emb_parts[0]
        comp_cols = [c for c in cell_emb.columns if str(c).startswith('comp_emb')]

        geom_cols = ['r_signed', 'd_cell_norm', 'd_nuc_norm'] + (
            ['sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta'] if 'sin_theta' in point_df.columns else []
        )
        gmm_cols = geom_cols + comp_cols + ['local_density']
        del cell_bow, df_vec, values, row_idx, col_idx, local_density
        gc.collect()
        return point_df, gmm_cols, cell_emb

    def _build_knn_per_cell(dfc: pd.DataFrame):
        pts = dfc[['x_location', 'y_location']].to_numpy(np.float32, copy=False)
        if len(pts) == 0:
            return None, None
        ind, dist = _kdtree_query(pts, k=min(K_NEI, len(dfc)))
        return ind, dist

    def _nei_bow_for_indices(
        dfc: pd.DataFrame,
        ind_mat: np.ndarray,
        dist_mat: np.ndarray = None,
        weight_mode: str = 'gaussian',
    ) -> sp.csr_matrix:
        """Vectorized neighborhood BoW with optional distance weights (sparse CSR).
        weight_mode: 'gaussian' | 'inv' | 'uniform'
        """
        G = len(vocab_genes)
        n = len(dfc)
        if n == 0 or G == 0:
            return sp.csr_matrix((n, G), dtype=np.float32)

        gene_idx = (
            dfc['feature_name']
               .astype(str)
               .map(gene2idx).fillna(-1)
               .astype(np.int64, copy=False)
               .to_numpy(copy=False)
        )

        ind = np.asarray(ind_mat)
        if ind.ndim == 1:
            ind = ind[None, :]
        ind = ind.astype(np.int64, copy=False)
        k = ind.shape[1]

        if dist_mat is not None:
            D = np.asarray(dist_mat, dtype=np.float32)
            if D.ndim == 1:
                D = D[None, :]
            D = np.clip(D, 0.0, np.inf)
        else:
            D = None

        # Build weight matrix W aligned with ind's shape
        if D is not None:
            if weight_mode == 'gaussian':
                rk = D[:, min(k - 1, D.shape[1] - 1)].astype(np.float32) + 1e-6
                scale = rk[:, None]
                W = np.exp(- (D / scale) ** 2).astype(np.float32, copy=False)
            elif weight_mode == 'inv':
                W = (1.0 / (D + 1e-3)).astype(np.float32, copy=False)
            else:
                W = np.ones_like(D, dtype=np.float32)
        else:
            W = np.ones_like(ind, dtype=np.float32)

        # Flatten and filter invalid neighbor rows/genes
        row_idx = np.repeat(np.arange(n, dtype=np.int64), k)
        nei_flat = ind.ravel()
        w_flat = W.ravel()

        valid = (nei_flat >= 0) & (nei_flat < n)
        if not valid.any():
            return sp.csr_matrix((n, G), dtype=np.float32)
        nei_flat = nei_flat[valid]
        row_idx = row_idx[valid]
        w_flat = w_flat[valid]

        gi_flat = gene_idx[nei_flat]
        gene_valid = (gi_flat >= 0) & (gi_flat < G)
        if not gene_valid.any():
            return sp.csr_matrix((n, G), dtype=np.float32)

        gi_flat = gi_flat[gene_valid]
        row_idx = row_idx[gene_valid]
        w_flat = w_flat[gene_valid]

        # Build sparse BoW via COO then convert to CSR
        bow = sp.coo_matrix((w_flat.astype(np.float32, copy=False), (row_idx, gi_flat)), shape=(n, G))
        return bow.tocsr()
    # Sample to build SVD training set
    rng = np.random.default_rng(RANDOM_STATE)
    from joblib import Parallel, delayed
    # speed-oriented parallel settings (bounded without env vars)
    # Keep module 2 conservative on memory: use at most 2 workers for large runs.
    N_JOBS = max(1, min(2, _preferred_n_jobs()))
    _backend = _joblib_backend()
    unique_cells = pd.unique(sub_tr_all['cell_id'])
    sample_cell_n = len(unique_cells)
    try:
        _cfg = load_params_yaml() or {}
        _m2 = _cfg.get('module2', {}) if _cfg else {}
        svd_sample_cells = _m2.get('svd_sample_cells')
        if isinstance(svd_sample_cells, (int, float)) and int(svd_sample_cells) > 0:
            sample_cell_n = min(sample_cell_n, int(svd_sample_cells))
    except Exception:
        pass
    # Extra safety caps if explicit ultralow env set or very constrained workers
    try:
        if os.environ.get('CELLSCOPE_ULTRALOW', '').lower() in ('1', 'true', 'yes'):
            sample_cell_n = min(sample_cell_n, 128)
            MAX_SVD_ROWS = min(MAX_SVD_ROWS, 20000)
    except Exception:
        pass
    # Keep complete cells so the fitted basis sees real spatial neighborhoods.
    # The row budget bounds both the number of cells and retained SVD documents.
    max_cells_by_rows = max(1, MAX_SVD_ROWS // max(1, MAX_ROWS_PER_CELL_FOR_SVD))
    sample_cell_n = max(1, min(int(sample_cell_n), len(unique_cells), max_cells_by_rows))
    sample_cells = rng.choice(unique_cells, size=sample_cell_n, replace=False)

    def _iter_group_chunks(df: pd.DataFrame, chunk_size: int):
        buf = []
        for pair in df.groupby('cell_id', sort=False):
            buf.append(pair)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    sub_tr_sample = sub_tr_all[sub_tr_all['cell_id'].isin(sample_cells)]
    n_groups_sample = int(sub_tr_sample['cell_id'].nunique(dropna=False)) if len(sub_tr_sample) else 0

    def _svd_tf_block(args):
        cid, dfc = args
        ind, dist = _build_knn_per_cell(dfc)
        if ind is None:
            return None
        bow = _nei_bow_for_indices(dfc, ind, dist, weight_mode=BOW_WEIGHT_MODE)
        if TF_NORM:
            row_sum = np.asarray(bow.sum(1)).ravel().astype(np.float32)
            row_sum[row_sum == 0] = 1.0
            tf = bow.multiply(1.0 / row_sum[:, None]).tocsr()
        else:
            tf = bow.tocsr()
        if tf.shape[0] > MAX_ROWS_PER_CELL_FOR_SVD:
            idx_loc = rng.choice(tf.shape[0], size=MAX_ROWS_PER_CELL_FOR_SVD, replace=False)
            tf = tf[idx_loc]
        return tf

    bow_samples = []
    total_rows = 0
    # accumulate document frequency per gene for IDF without storing all blocks
    df_vec = np.zeros(len(vocab_genes), dtype=np.float32)
    use_rp = bool(STREAM_LOCAL)
    sample_chunk_size = PARALLEL_CHUNK_CELLS
    n_sample_chunks = (n_groups_sample + sample_chunk_size - 1) // sample_chunk_size if n_groups_sample else 0
    iterator = _progress_iter(
        _iter_group_chunks(sub_tr_sample, sample_chunk_size),
        desc='SVD sample TF (parallel)',
        total=n_sample_chunks,
        show=show_internal_progress,
    )
    for chunk in iterator:
        out = Parallel(n_jobs=N_JOBS, backend=_backend, verbose=0)(
            delayed(_svd_tf_block)(pair) for pair in chunk
        )
        for tf in out:
            if tf is None:
                continue
            # update df counts and optionally keep block
            df_vec += np.asarray(tf.getnnz(axis=0), dtype=np.float32)
            if total_rows < MAX_SVD_ROWS and not STREAM_LOCAL:
                bow_samples.append(tf)
            total_rows += tf.shape[0]
            if total_rows >= MAX_SVD_ROWS:
                break
        if total_rows >= MAX_SVD_ROWS:
            break

    if len(bow_samples) == 0 and not STREAM_LOCAL:
        raise RuntimeError('SVD samples empty.')

    # Decide whether to use TruncatedSVD or fall back to SparseRandomProjection
    # to avoid vstack of many sparse blocks which can peak memory.
    if total_rows > MAX_SVD_ROWS or os.environ.get('CELLSCOPE_ULTRALOW', '').lower() in ('1', 'true', 'yes'):
        use_rp = True

    if not use_rp:
        B_sample = sp.vstack(bow_samples, format='csr', dtype=np.float32)
        # Optional TF-IDF reweighting for SVD (sparse)
        if USE_TFIDF and B_sample.shape[0] > 0:
            df_vec = np.asarray(B_sample.getnnz(axis=0), dtype=np.float32)
            N_rows = float(B_sample.shape[0])
            if IDF_MODE == 'none':
                idf_vec = np.ones_like(df_vec, dtype=np.float32)
            elif IDF_MODE == 'log':
                idf_vec = np.log(np.maximum(N_rows / np.maximum(df_vec, 1.0), 1.0)).astype(np.float32)
            else:  # 'log1p'
                idf_vec = np.log1p(N_rows / (1.0 + df_vec)).astype(np.float32)
            B_sample = B_sample.multiply(idf_vec[np.newaxis, :]).tocsr()
        # Fit SVD on the sampled (sparse) block. GC before fit to reduce peaks.
        gc.collect()
        svd_local = TruncatedSVD(n_components=N_COMP, random_state=RANDOM_STATE)
        svd_local.fit(B_sample)
    else:
        # Low-memory fallback: sparse random projection
        from sklearn.random_projection import SparseRandomProjection
        N_rows = float(max(1, total_rows))
        if IDF_MODE == 'none':
            idf_vec = np.ones_like(df_vec, dtype=np.float32)
        elif IDF_MODE == 'log':
            idf_vec = np.log(np.maximum(N_rows / np.maximum(df_vec, 1.0), 1.0)).astype(np.float32)
        else:
            idf_vec = np.log1p(N_rows / (1.0 + df_vec)).astype(np.float32)
        rp_components = max(1, min(N_COMP, len(vocab_genes)))
        rp = SparseRandomProjection(n_components=rp_components, random_state=RANDOM_STATE)
        rp.fit(np.zeros((1, len(vocab_genes)), dtype=np.float32))
        svd_local = rp

    if STREAM_LOCAL:
        model_idf = (
            idf_vec
            if USE_TFIDF
            else np.ones(len(vocab_genes), dtype=np.float32)
        )
        local_model = _LocalCompositionModel(
            gene2idx=gene2idx,
            transformer=svd_local,
            idf_vec=model_idf,
            n_components=rp_components if use_rp else N_COMP,
            knn_k=K_NEI,
            weight_mode=BOW_WEIGHT_MODE,
            tf_norm=TF_NORM,
            use_tfidf=USE_TFIDF,
        )
        geom_cols = ['r_signed', 'd_cell_norm', 'd_nuc_norm'] + (
            ['sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta'] if 'sin_theta' in point_df.columns else []
        )
        gmm_cols = geom_cols + local_model.feature_names + ['local_density']
        del bow_samples
        if 'B_sample' in locals():
            del B_sample
        gc.collect()
        return point_df, gmm_cols, local_model

    # Embed all rows (full pass)
    # Avoid copying the full table to reduce peak memory; rely on index positions.
    sub_tr = sub_tr_all
    emb_dtype = np.float32
    try:
        # For very large runs, float16 greatly reduces memory and is sufficient
        # because downstream clustering re-casts to float32 before fitting.
        if len(sub_tr) >= 5_000_000:
            emb_dtype = np.float16
    except Exception:
        pass
    n_groups_full = int(sub_tr['cell_id'].nunique(dropna=False)) if len(sub_tr) else 0

    def _emb_block(args):
        cid, dfc = args
        ind, dist = _build_knn_per_cell(dfc)
        if ind is None:
            return None
        bow = _nei_bow_for_indices(dfc, ind, dist, weight_mode=BOW_WEIGHT_MODE)
        if TF_NORM:
            row_sum = np.asarray(bow.sum(1)).ravel().astype(np.float32)
            row_sum[row_sum == 0] = 1.0
            tf = bow.multiply(1.0 / row_sum[:, None]).tocsr()
        else:
            tf = bow.tocsr()
        if USE_TFIDF:
            # Use same IDF as fitted on the sample
            tfidf = tf.multiply(idf_vec[np.newaxis, :]).tocsr()
            Z = svd_local.transform(tfidf).astype(np.float32, copy=False)
        else:
            Z = svd_local.transform(tf).astype(np.float32, copy=False)
        if sp.issparse(Z):
            Z = Z.toarray()
        k_used = min(K_NEI, len(dfc))
        if dist is not None and dist.shape[1] >= k_used:
            rk = dist[:, k_used - 1].astype(np.float32) + 1e-3
            dens = (k_used) / (np.pi * rk * rk)
        else:
            dens = np.full(len(dfc), k_used, np.float32)
        # Use the dataframe index positions as row ids to avoid adding columns/copies
        rows = dfc.index.to_numpy(dtype=np.int64, copy=False)
        return rows, Z, dens

    chunk_size2 = PARALLEL_CHUNK_CELLS
    n_chunks_full = (n_groups_full + chunk_size2 - 1) // chunk_size2 if n_groups_full else 0
    iterator2 = _progress_iter(
        _iter_group_chunks(sub_tr, chunk_size2),
        desc='Per-cell KNN & BoW (parallel)',
        total=n_chunks_full,
        show=show_internal_progress,
    )
    comp_cols = None
    for chunk in iterator2:
        try:
            out = Parallel(
                n_jobs=N_JOBS,
                backend=_backend,
                verbose=0,
                pre_dispatch=f"2*{N_JOBS}",
                batch_size="auto",
                # Avoid memmap on small arrays to reduce overhead
                max_nbytes=None,
            )(
                delayed(_emb_block)(pair) for pair in chunk
            )
        except Exception as e:
            print(f"[WARN] Parallel KNN/BoW chunk failed, fallback sequential: {e}")
            out = [ _emb_block(pair) for pair in chunk ]
        for item in out:
            if item is None:
                continue
            rows, Z, dens = item
            Z = np.asarray(Z, dtype=emb_dtype)
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
            if Z.shape[1] == 0:
                Z = np.zeros((Z.shape[0], 1), dtype=emb_dtype)
            if comp_cols is None:
                comp_cols = [f'comp_emb{i}' for i in range(Z.shape[1])]
            elif Z.shape[1] != len(comp_cols):
                if Z.shape[1] < len(comp_cols):
                    pad = np.zeros((Z.shape[0], len(comp_cols) - Z.shape[1]), dtype=emb_dtype)
                    Z = np.concatenate([Z, pad], axis=1)
                else:
                    Z = Z[:, :len(comp_cols)]
            point_df.loc[rows, comp_cols] = Z
            point_df.loc[rows, 'local_density'] = np.asarray(dens, dtype=emb_dtype)

    if comp_cols is None:
        comp_cols = [f'comp_emb{i}' for i in range(N_COMP)]

    # Do not materialize gmm_feat_* dense columns here: this can double memory on
    # ultra-large tables. Per-cell MetaSpot segmentation consumes gmm_cols and performs
    # per-cell standardization internally.
    geom_cols = ['r_signed', 'd_cell_norm', 'd_nuc_norm'] + (
        ['sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta'] if 'sin_theta' in point_df.columns else []
    )
    gmm_cols = geom_cols + comp_cols + ['local_density']
    del B_sample, bow_samples
    gc.collect()
    return point_df, gmm_cols, None


def _safe_standardize(X: np.ndarray):
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0
    mu = X.mean(0, dtype=np.float64)
    sd = X.std(0, dtype=np.float64)
    good = sd > 0
    if not np.any(good):
        return np.zeros_like(X, np.float32), good
    Xg = X[:, good].astype(np.float32, copy=False)
    Xz = (Xg - mu[good].astype(np.float32)) / sd[good].astype(np.float32)
    return Xz, good


def _select_top_variance_features(X: np.ndarray, n_features: int) -> np.ndarray:
    """Select per-cell features without creating a dataset-wide dense matrix."""
    X = np.asarray(X, dtype=np.float32)
    limit = int(n_features)
    if limit <= 0 or X.ndim != 2 or X.shape[1] <= limit:
        return X
    finite = np.where(np.isfinite(X), X, 0.0)
    variances = np.var(finite, axis=0, dtype=np.float64)
    order = np.argsort(-variances, kind="stable")[:limit]
    return X[:, order]


def _target_metaspot_count(
    coordinates: np.ndarray,
    target_sites_per_metaspot: int,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
) -> int:
    """Choose a per-cell MetaSpot count from the number of unique XY sites."""
    xy = np.asarray(coordinates, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_points, 2)")
    finite_xy = xy[np.isfinite(xy).all(axis=1)]
    if len(finite_xy) == 0:
        return 1

    n_sites = int(np.unique(finite_xy, axis=0).shape[0])
    target_sites = max(1, int(target_sites_per_metaspot))
    target_count = int(math.ceil(n_sites / float(target_sites)))
    lower_bound = max(1, int(min_metaspots_per_cell))
    target_count = max(lower_bound, target_count)
    if isinstance(max_metaspots_per_cell, (int, float)):
        upper_bound = max(lower_bound, int(max_metaspots_per_cell))
        target_count = min(target_count, upper_bound)
    return max(1, min(target_count, n_sites))


def _enforce_spatially_connected_labels(
    coordinates: np.ndarray,
    labels: np.ndarray,
    feature_matrix: Optional[np.ndarray] = None,
    target_n_clusters: Optional[int] = None,
    spatial_compactness: float = 1.0,
    connectivity_k: int = 8,
    radius_quantile: float = 0.9,
    radius_factor: float = 1.5,
    smoothing_iterations: int = 1,
    smoothing_min_fraction: float = 0.6,
    min_component_size: int = 20,
) -> np.ndarray:
    """Turn feature-space labels into spatially connected 2D components.

    Duplicate transcript coordinates are collapsed to one spatial site, labels
    are re-clustered with an XY-constrained Ward model when features are given,
    and undersized components are merged across a shared boundary. Without a
    feature matrix, the preliminary labels are smoothed and split directly.
    Component sizes are measured in transcripts, not unique XY sites.
    """
    xy = np.asarray(coordinates, dtype=np.float32)
    point_labels = np.asarray(labels, dtype=np.int32)
    n_points = len(point_labels)
    if n_points == 0:
        return point_labels.copy()
    if xy.ndim != 2 or xy.shape != (n_points, 2):
        raise ValueError("coordinates must have shape (n_points, 2)")
    features = None
    if feature_matrix is not None:
        features = np.asarray(feature_matrix, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != n_points:
            raise ValueError("feature_matrix must have n_points rows")

    finite = np.isfinite(xy).all(axis=1)
    if not finite.all():
        # Module 3 already drops invalid XY rows; keep this helper defensive.
        out = point_labels.copy()
        if finite.any():
            out[finite] = _enforce_spatially_connected_labels(
                xy[finite],
                point_labels[finite],
                feature_matrix=features[finite] if features is not None else None,
                target_n_clusters=target_n_clusters,
                spatial_compactness=spatial_compactness,
                connectivity_k=connectivity_k,
                radius_quantile=radius_quantile,
                radius_factor=radius_factor,
                smoothing_iterations=smoothing_iterations,
                smoothing_min_fraction=smoothing_min_fraction,
                min_component_size=min_component_size,
            )
        return out

    unique_xy, inverse, site_weights = np.unique(
        xy,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    n_sites = len(unique_xy)
    if n_sites <= 1:
        return np.zeros(n_points, dtype=np.int32)

    # Pick a deterministic majority label at coordinates containing multiple
    # transcripts. Most imaging datasets already agree exactly at each site.
    site_labels = np.empty(n_sites, dtype=np.int32)
    order = np.argsort(inverse, kind="stable")
    sorted_sites = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_sites)) + 1]
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        site = int(sorted_sites[start])
        values, counts = np.unique(point_labels[order[start:stop]], return_counts=True)
        site_labels[site] = int(values[np.argmax(counts)])

    query_k = max(2, min(n_sites, int(connectivity_k) + 1))
    neighbor_idx, neighbor_dist = _kdtree_query(unique_xy, query_k)
    if neighbor_idx is None or neighbor_dist is None:
        return point_labels.copy()

    positive_dist = np.where(neighbor_dist > 1e-8, neighbor_dist, np.inf)
    nearest = positive_dist.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return point_labels.copy()

    q = float(np.clip(radius_quantile, 0.0, 1.0))
    base_spacing = float(np.quantile(nearest, q))
    radius = base_spacing * max(float(radius_factor), 1e-6)
    if not np.isfinite(radius) or radius <= 0:
        return point_labels.copy()

    local_edges = set()
    knn_edges = set()
    for i in range(n_sites):
        for j, distance in zip(neighbor_idx[i], neighbor_dist[i]):
            j = int(j)
            if i == j or not np.isfinite(distance) or distance <= 0:
                continue
            a, b = (i, j) if i < j else (j, i)
            knn_edges.add((a, b))
            if distance <= radius:
                local_edges.add((a, b))

    # Feature-aware Ward clustering needs one connected neighborhood graph to
    # preserve its requested cluster count. Direct label splitting keeps the
    # stricter native-spacing radius so genuinely separated islands stay split.
    edges = knn_edges if features is not None else local_edges
    if not edges:
        return point_labels.copy()

    edge_list = sorted(edges)
    adjacency = [[] for _ in range(n_sites)]
    for a, b in edge_list:
        adjacency[a].append(b)
        adjacency[b].append(a)

    used_constrained_clustering = False
    if features is not None and n_sites > 1:
        n_features = features.shape[1]
        site_features = np.zeros((n_sites, n_features), dtype=np.float32)
        np.add.at(site_features, inverse, features)
        site_features /= site_weights[:, None].astype(np.float32)
        site_features, _ = _safe_standardize(site_features)
        xy_features, _ = _safe_standardize(unique_xy)
        combined_features = np.hstack(
            [
                site_features,
                xy_features * max(float(spatial_compactness), 0.0),
            ]
        ).astype(np.float32, copy=False)

        requested_clusters = (
            int(target_n_clusters)
            if isinstance(target_n_clusters, (int, float))
            else int(np.unique(point_labels).size)
        )
        requested_clusters = max(1, min(requested_clusters, n_sites))
        if requested_clusters > 1:
            rows = np.fromiter(
                (endpoint for edge in edge_list for endpoint in (edge[0], edge[1])),
                dtype=np.int32,
            )
            cols = np.fromiter(
                (endpoint for edge in edge_list for endpoint in (edge[1], edge[0])),
                dtype=np.int32,
            )
            connectivity = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(n_sites, n_sites),
            )
            try:
                site_labels = AgglomerativeClustering(
                    n_clusters=requested_clusters,
                    linkage="ward",
                    connectivity=connectivity,
                ).fit_predict(combined_features).astype(np.int32)
                used_constrained_clustering = True
            except Exception as exc:
                _logging.getLogger("cellscope").warning(
                    "Spatially constrained clustering failed; using label "
                    "smoothing fallback: %s",
                    exc,
                )

    if not used_constrained_clustering:
        smooth_fraction = float(np.clip(smoothing_min_fraction, 0.5, 1.0))
        for _ in range(max(0, int(smoothing_iterations))):
            updated = site_labels.copy()
            changed = False
            for i, neighbors in enumerate(adjacency):
                if not neighbors:
                    continue
                votes: Dict[int, int] = defaultdict(int)
                votes[int(site_labels[i])] += int(site_weights[i])
                for j in neighbors:
                    votes[int(site_labels[j])] += int(site_weights[j])
                current = int(site_labels[i])
                winner = max(
                    votes,
                    key=lambda label: (votes[label], label == current, -int(label)),
                )
                total = sum(votes.values())
                if (
                    winner != current
                    and total > 0
                    and votes[winner] / total >= smooth_fraction
                ):
                    updated[i] = int(winner)
                    changed = True
            site_labels = updated
            if not changed:
                break

    parent = np.arange(n_sites, dtype=np.int32)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return int(node)

    def union(a: int, b: int) -> bool:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return False
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        return True

    # First split every original label into its XY-connected components.
    for a, b in edge_list:
        if site_labels[a] == site_labels[b]:
            union(a, b)

    min_size = max(1, int(min_component_size))
    for _ in range(32):
        roots = np.asarray([find(i) for i in range(n_sites)], dtype=np.int32)
        component_size: Dict[int, int] = defaultdict(int)
        for site, root in enumerate(roots):
            component_size[int(root)] += int(site_weights[site])

        small_roots = {
            root for root, size in component_size.items() if size < min_size
        }
        if not small_roots:
            break

        boundary_contacts: Dict[int, Counter] = defaultdict(Counter)
        for a, b in edge_list:
            root_a, root_b = find(a), find(b)
            if root_a == root_b:
                continue
            boundary_contacts[root_a][root_b] += 1
            boundary_contacts[root_b][root_a] += 1

        changed = False
        for root in sorted(small_roots, key=lambda r: (component_size[r], r)):
            root = find(root)
            if component_size.get(root, 0) >= min_size:
                continue
            candidates = {
                find(candidate): contacts
                for candidate, contacts in boundary_contacts.get(root, {}).items()
                if find(candidate) != root
            }
            if not candidates:
                continue
            target = max(
                candidates,
                key=lambda candidate: (
                    candidates[candidate],
                    component_size.get(candidate, 0),
                    -int(candidate),
                ),
            )
            changed |= union(root, target)
        if not changed:
            break

    roots = np.asarray([find(i) for i in range(n_sites)], dtype=np.int32)
    root_order = sorted(np.unique(roots), key=lambda root: int(np.flatnonzero(roots == root)[0]))
    root_to_label = {int(root): label for label, root in enumerate(root_order)}
    connected_site_labels = np.asarray(
        [root_to_label[int(root)] for root in roots],
        dtype=np.int32,
    )
    return connected_site_labels[inverse]


def per_cell_clustering(
    point_df: pd.DataFrame,
    gmm_cols: List[str],
    composition_model: Optional[_LocalCompositionModel] = None,
    show_internal_progress: bool = False,
    hdbscan_min_cluster_size: int = 6,
    hdbscan_allow_single_cluster: bool = True,
    hdbscan_min_samples_frac: float = 0.3,
    target_sub_size: int = 12,
    n_feat_hdb: int = 24,
    kmeans_min_k: int = 2,
    kmeans_max_k: int = 999,
    use_k_cap: bool = False,
    merge_min_size: Optional[int] = None,
    hdbscan_max_points: int = 512,
    kmeans_train_size: int = 1024,
    kmeans_batch_size: int = 2048,
    small_backend_cells: int = 12000,
    sequential_cells: int = 256,
    enforce_spatial_connectivity: bool = False,
    spatial_connectivity_k: int = 8,
    spatial_radius_quantile: float = 0.9,
    spatial_radius_factor: float = 1.5,
    spatial_smoothing_iterations: int = 1,
    spatial_smoothing_min_fraction: float = 0.6,
    spatial_min_component_size: Optional[int] = None,
    spatial_compactness: float = 1.0,
    target_sites_per_metaspot: Optional[int] = None,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
    ) -> pd.DataFrame:
    import hdbscan  # lazy import

    MIN_SUB_PTS = int(hdbscan_min_cluster_size)
    # 使用更小的目标簇大小以促进细分，且不与最小簇大小绑定
    TARGET_SUB_SIZE = max(8, int(target_sub_size))
    K_CAP_BASE = 999
    MIN_SAMPLES_FRAC = float(hdbscan_min_samples_frac)
    SPATIAL_MIN_COMPONENT_SIZE = (
        int(spatial_min_component_size)
        if isinstance(spatial_min_component_size, (int, float))
        else int(merge_min_size if merge_min_size is not None else MIN_SUB_PTS)
    )
    TARGET_SITES_PER_METASPOT = (
        int(target_sites_per_metaspot)
        if isinstance(target_sites_per_metaspot, (int, float))
        and int(target_sites_per_metaspot) > 0
        else None
    )

    def _finalize_spatial_labels(
        g: pd.DataFrame,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int32)
        if not bool(enforce_spatial_connectivity) or len(labels) <= 1:
            return labels
        coordinates = g[["x_location", "y_location"]].to_numpy(
            np.float32,
            copy=False,
        )
        target_count = int(np.unique(labels).size)
        if TARGET_SITES_PER_METASPOT is not None:
            target_count = _target_metaspot_count(
                coordinates,
                target_sites_per_metaspot=TARGET_SITES_PER_METASPOT,
                min_metaspots_per_cell=min_metaspots_per_cell,
                max_metaspots_per_cell=max_metaspots_per_cell,
            )
        return _enforce_spatially_connected_labels(
            coordinates,
            labels,
            feature_matrix=features,
            target_n_clusters=target_count,
            spatial_compactness=spatial_compactness,
            connectivity_k=spatial_connectivity_k,
            radius_quantile=spatial_radius_quantile,
            radius_factor=spatial_radius_factor,
            smoothing_iterations=spatial_smoothing_iterations,
            smoothing_min_fraction=spatial_smoothing_min_fraction,
            min_component_size=SPATIAL_MIN_COMPONENT_SIZE,
        )

    # Select features with the highest variance. If per-row composition columns
    # are not materialized in `point_df` (to avoid large memory), a
    # precomputed per-cell composition DataFrame may be present in the
    # environment and will be used when building per-transcript features.
    # We build a modest sample by combining per-row geometry and per-cell
    # composition (looked up) to compute variances without expanding global
    # comp arrays.
    geom_cols = [c for c in gmm_cols if c in ('x_location', 'y_location', 'r_signed', 'd_cell_norm', 'd_nuc_norm', 'sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta', 'local_density')]
    comp_cols = [c for c in gmm_cols if c not in geom_cols]

    def _decide_K_min(n: int) -> int:
        # 更积极的最低簇数下界
        if n >= 6 * MIN_SUB_PTS:
            return 4
        elif n >= 4 * MIN_SUB_PTS:
            return 3
        elif n >= 3 * MIN_SUB_PTS:
            return 2
        else:
            return 1

    def _decide_K_for_cell(n: int) -> int:
        if n < MIN_SUB_PTS * 1.2:
            return 1
        k_by_size = max(1, n // MIN_SUB_PTS)
        k_target = max(1, int(np.ceil(n / float(TARGET_SUB_SIZE))))
        k_cap = min(K_CAP_BASE, k_by_size)
        K = min(k_target, k_cap)
        K = max(_decide_K_min(n), K)
        if n >= 8 * MIN_SUB_PTS:
            K = max(K, _decide_K_min(n) + 1)
        # Optional clamp by configured bounds
        if bool(use_k_cap):
            K = max(int(kmeans_min_k), min(int(kmeans_max_k), int(K)))
        return int(max(1, K))

    def _merge_small_clusters(labels: np.ndarray, X: np.ndarray, min_size: int) -> np.ndarray:
        y = labels.astype(np.int32, copy=True)
        while True:
            labs, cnts = np.unique(y, return_counts=True)
            tiny = labs[cnts < min_size]
            if len(tiny) == 0:
                break
            big = labs[cnts >= min_size]
            if len(big) == 0:
                big = np.array([labs[0]], dtype=labs.dtype)
            cents = {lab: X[y == lab].mean(0) for lab in labs}
            big_cent = np.vstack([cents[b] for b in big])
            for t in tiny:
                idx = np.where(y == t)[0]
                if idx.size == 0:
                    continue
                c_t = cents[t][None, :]
                from scipy.spatial.distance import cdist
                j = np.argmin(cdist(c_t, big_cent))
                to_lab = big[j]
                y[idx] = to_lab
            u = np.unique(y)
            remap = {u_i: i for i, u_i in enumerate(u)}
            y = np.vectorize(remap.get, otypes=[np.int32])(y)
        return y

    def _assign_noise_to_nearest(labels: np.ndarray, X: np.ndarray) -> np.ndarray:
        y = labels.astype(np.int32, copy=True)
        mask_noise = (y < 0)
        if not mask_noise.any():
            return y
        labs = np.unique(y[~mask_noise])
        if labs.size == 0:
            return y
        cents = np.vstack([X[y == lab].mean(0) for lab in labs])
        idx_noise = np.where(mask_noise)[0]
        X_noise = X[idx_noise]
        from scipy.spatial.distance import cdist
        D = cdist(X_noise, cents)
        nearest = labs[D.argmin(axis=1)]
        y[idx_noise] = nearest
        u = np.unique(y)
        remap = {u_i: i for i, u_i in enumerate(u)}
        y = np.vectorize(remap.get, otypes=[np.int32])(y)
        return y

    # Per-cell MetaSpot segmentation (parallel batching)

    def _iter_group_chunks(df: pd.DataFrame, chunk_size: int):
        buf = []
        for pair in df.groupby('cell_id', sort=False):
            buf.append(pair)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    n_groups = int(point_df['cell_id'].nunique(dropna=False)) if len(point_df) else 0

    # Deterministic RNG for MiniBatchKMeans sampling inside _fit_one
    rng = np.random.default_rng(42)

    def _fit_one(args):
        cid, g = args
        # Ensure minimal geometry present; composition may be provided per-cell
        g = g.dropna(subset=['x_location', 'y_location'])
        if len(g) == 0:
            return cid, pd.Series([], dtype=np.int32, index=g.index)
        n = len(g)
        if n < MIN_SUB_PTS:
            labels = np.zeros(n, dtype=np.int32)
            labels = _finalize_spatial_labels(g, labels)
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)
        # Build local expression features for this cell only, then release them
        # after clustering. This avoids materializing dense embeddings globally.
        local_density = None
        if composition_model is not None:
            X_comp, local_density = composition_model.transform_cell(g)
        else:
            comp_present = any((cc in g.columns) for cc in comp_cols)
        if composition_model is None and comp_present:
            X_comp = g[[c for c in comp_cols if c in g.columns]].to_numpy(np.float32, copy=False)
            # If some comp_cols are missing, pad zeros
            if X_comp.shape[1] < len(comp_cols):
                missing = len(comp_cols) - X_comp.shape[1]
                X_comp = np.hstack([X_comp, np.zeros((len(g), missing), dtype=np.float32)])
        elif composition_model is None:
            cell_emb = getattr(point_df, '_cell_comp_emb', None)
            if cell_emb is not None and cid in cell_emb.index:
                vec = cell_emb.loc[cid, comp_cols].to_numpy(dtype=np.float32)
                X_comp = np.repeat(vec.reshape(1, -1), len(g), axis=0)
            else:
                X_comp = np.zeros((len(g), len(comp_cols)), dtype=np.float32)
        # geometry
        geom_use = [c for c in geom_cols if c in g.columns and c != 'local_density']
        if len(geom_use) == 0:
            X_geom = np.zeros((len(g), 2), dtype=np.float32)
        else:
            X_geom = g[geom_use].to_numpy(np.float32, copy=False)
        if local_density is not None:
            X_geom = np.column_stack((X_geom, local_density)).astype(np.float32, copy=False)
        X_full = np.hstack([X_geom, X_comp]) if X_comp.size else X_geom
        X_full = _select_top_variance_features(X_full, n_feat_hdb)
        Xz, good = _safe_standardize(X_full)
        if Xz.shape[1] == 0 or not np.any(good):
            X_coord = g[['x_location', 'y_location']].to_numpy(np.float32, copy=False)
            Xz, _ = _safe_standardize(X_coord)
        # In the canonical MetaSpot mode, spatial resolution determines K.
        # HDBSCAN remains available as a legacy fallback when this target is off.
        if bool(enforce_spatial_connectivity) and TARGET_SITES_PER_METASPOT is not None:
            seed_labels = np.zeros(n, dtype=np.int32)
            labels = _finalize_spatial_labels(g, seed_labels, Xz)
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)
        # Fast path: large cells switch to sampled MiniBatchKMeans to avoid expensive HDBSCAN
        if n > int(hdbscan_max_points):
            K_init = _decide_K_for_cell(n)
            if bool(use_k_cap):
                K_init = max(int(kmeans_min_k), min(int(kmeans_max_k), int(K_init)))
            K_init = max(1, int(K_init))
            train_n = min(int(kmeans_train_size), n)
            K_init = min(K_init, max(1, train_n))
            idx_train = rng.choice(n, size=train_n, replace=False)
            X_train = Xz[idx_train]
            bs = max(64, min(int(kmeans_batch_size), train_n))
            km = MiniBatchKMeans(
                n_clusters=K_init,
                init="k-means++",
                n_init=3,
                random_state=42,
                batch_size=bs,
                max_iter=80,
                reassignment_ratio=0.01,
                verbose=0,
            )
            km.fit(X_train)
            labels = km.predict(Xz).astype(np.int32)
            _min_size = int(merge_min_size) if merge_min_size is not None else int(MIN_SUB_PTS)
            labels = _merge_small_clusters(labels, Xz, min_size=_min_size)
            labels = _finalize_spatial_labels(g, labels, Xz)
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)

        min_samples = max(3, int(round(MIN_SUB_PTS * MIN_SAMPLES_FRAC)))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_SUB_PTS,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            core_dist_n_jobs=1,
            prediction_data=False,
            allow_single_cluster=bool(hdbscan_allow_single_cluster),
            gen_min_span_tree=False,
            approx_min_span_tree=False,
        )
        hdb_labels = clusterer.fit_predict(Xz).astype(np.int32)
        mask_non_noise = (hdb_labels >= 0)
        uniq_clusters = np.unique(hdb_labels[mask_non_noise]) if mask_non_noise.any() else np.array([], dtype=int)
        n_clusters = len(uniq_clusters)
        K_min = max(int(kmeans_min_k), _decide_K_min(n))
        use_fallback = False
        if n_clusters == 0:
            use_fallback = True
        elif (n_clusters < K_min) and (n >= int(1.5 * MIN_SUB_PTS)):
            use_fallback = True
        if use_fallback:
            K_init = _decide_K_for_cell(n)
            if K_init <= 1:
                labels = np.zeros(n, dtype=np.int32)
            else:
                km = KMeans(n_clusters=K_init, n_init='auto', random_state=42, max_iter=200)
                labels = km.fit_predict(Xz).astype(np.int32)
            _min_size = int(merge_min_size) if merge_min_size is not None else int(MIN_SUB_PTS)
            labels = _merge_small_clusters(labels, Xz, min_size=_min_size)
            labels = _finalize_spatial_labels(g, labels, Xz)
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)
        labels = _assign_noise_to_nearest(hdb_labels, Xz)
        _min_size = int(merge_min_size) if merge_min_size is not None else int(MIN_SUB_PTS)
        labels = _merge_small_clusters(labels, Xz, min_size=_min_size)
        labels = _finalize_spatial_labels(g, labels, Xz)
        return cid, pd.Series(labels, index=g.index, dtype=np.int32)

    # Lightweight backend selection for small datasets to cut scheduler/IPC overhead
    _force_threads = bool(os.environ.get("CELLSCOPE_FORCE_THREADS", "").strip())
    use_sequential = n_groups <= int(sequential_cells)
    _backend = _joblib_backend()
    if _force_threads or n_groups <= int(small_backend_cells):
        _backend = "threading"
    if use_sequential:
        iterator = _progress_iter(range(n_groups), desc='MetaSpot segmentation (seq)', total=n_groups, show=show_internal_progress)
        results = []
        for chunk in _iter_group_chunks(point_df, 1):
            for pair in chunk:
                results.append(_fit_one(pair))
        labels_per_cell = pd.concat([s for _, s in results]).astype(np.int32)
        point_df = point_df.copy()
        point_df[METASPOT_LOCAL_COL] = labels_per_cell
        missing = point_df[METASPOT_LOCAL_COL].isna()
        if missing.any():
            point_df.loc[missing, METASPOT_LOCAL_COL] = -1
        point_df[METASPOT_LOCAL_COL] = point_df[METASPOT_LOCAL_COL].astype(np.int32)
        point_df[METASPOT_COL] = (
            point_df['cell_id'].astype(str)
            + ':'
            + point_df[METASPOT_LOCAL_COL].astype(str)
        )
        return point_df

    from joblib import Parallel, delayed
    # Keep module 3 conservative on memory: use at most 2 workers for large runs.
    N_JOBS = max(1, min(2, _preferred_n_jobs()))
    # _backend already chosen above, but keep env override precedence
    if os.environ.get("CELLSCOPE_FORCE_THREADS", "").strip():
        _backend = "threading"

    # prefer larger chunk to reduce scheduler overhead
    chunk_size = max(64, _cluster_chunk_size())
    results = []
    n_chunks = (n_groups + chunk_size - 1) // chunk_size if n_groups else 0
    iterator = _progress_iter(
        _iter_group_chunks(point_df, chunk_size),
        desc='MetaSpot segmentation (parallel)',
        total=n_chunks,
        show=show_internal_progress,
    )
    for chunk in iterator:
        try:
            out = Parallel(
                n_jobs=N_JOBS,
                backend=_backend,
                verbose=0,
                pre_dispatch=f"3*{N_JOBS}",
                batch_size="auto",
                max_nbytes=None,
            )(
                delayed(_fit_one)(pair) for pair in chunk
            )
        except Exception as e:
            # Fallback: sequential processing to avoid hard failure (e.g. OOM kill)
            print(f"[WARN] Parallel cluster chunk failed, fallback sequential: {e}")
            out = [ _fit_one(pair) for pair in chunk ]
        results.extend(out)
        gc.collect()

    labels_per_cell = pd.concat([s for _, s in results]).astype(np.int32)
    point_df = point_df.copy()
    point_df[METASPOT_LOCAL_COL] = labels_per_cell
    missing = point_df[METASPOT_LOCAL_COL].isna()
    if missing.any():
        point_df.loc[missing, METASPOT_LOCAL_COL] = -1
    point_df[METASPOT_LOCAL_COL] = point_df[METASPOT_LOCAL_COL].astype(np.int32)
    point_df[METASPOT_COL] = (
        point_df['cell_id'].astype(str)
        + ':'
        + point_df[METASPOT_LOCAL_COL].astype(str)
    )
    return point_df


def build_adata1(point_df: pd.DataFrame, show_internal_progress: bool = False) -> ad.AnnData:
    metaspot_cat = pd.Categorical(point_df[METASPOT_COL], ordered=False)
    row_codes = metaspot_cat.codes.astype(np.int64)
    row_cats = metaspot_cat.categories
    gene_codes, gene_uni = pd.factorize(point_df['feature_name'], sort=False)
    gene_codes = gene_codes.astype(np.int64)
    X = sp.coo_matrix(
        (np.ones(len(point_df), dtype=np.int32), (row_codes, gene_codes)),
        shape=(len(row_cats), len(gene_uni))
    ).tocsr()
    adata1 = ad.AnnData(X=X, obs=pd.DataFrame(index=row_cats), var=pd.DataFrame(index=gene_uni))

    need_cols = ['r_signed', 'd_cell_norm', 'd_nuc_norm', 'x_location', 'y_location']
    arr = point_df[need_cols].to_numpy(dtype=np.float32, copy=False)
    codes = row_codes
    order = np.argsort(codes, kind='mergesort')
    codes_sorted = codes[order]
    A = arr[order, :]
    unique_codes, counts = np.unique(codes_sorted, return_counts=True)
    starts = np.empty_like(counts)
    starts[0] = 0
    np.cumsum(counts[:-1], out=starts[1:])
    ends = starts + counts

    n_groups = len(unique_codes)
    EPS = 1e-8
    r_mean = np.full(n_groups, np.nan, np.float32)
    r_std = np.full(n_groups, np.nan, np.float32)
    r_p10 = np.full(n_groups, np.nan, np.float32)
    r_p50 = np.full(n_groups, np.nan, np.float32)
    r_p90 = np.full(n_groups, np.nan, np.float32)
    d1_mean = np.full(n_groups, np.nan, np.float32)
    d2_mean = np.full(n_groups, np.nan, np.float32)
    elong = np.full(n_groups, np.nan, np.float32)
    spread = np.full(n_groups, np.nan, np.float32)
    log_n = np.log1p(counts).astype(np.float32)

    rng_iter = _progress_iter(range(n_groups), desc='Aggregate domain stats', total=n_groups, show=show_internal_progress)
    for i in rng_iter:
        s, e = starts[i], ends[i]
        if e - s < 1:
            continue
        r = A[s:e, 0]
        d1 = A[s:e, 1]
        d2 = A[s:e, 2]
        xs = A[s:e, 3]
        ys = A[s:e, 4]
        r_mean[i] = np.nanmean(r)
        r_std[i] = np.nanstd(r)
        r_p10[i], r_p50[i], r_p90[i] = np.nanpercentile(r, [10, 50, 90])
        d1_mean[i] = np.nanmean(d1)
        d2_mean[i] = np.nanmean(d2)
        n = e - s
        if n >= 3:
            mx, my = np.nanmean(xs), np.nanmean(ys)
            cx, cy = xs - mx, ys - my
            mask_xy = np.isfinite(cx) & np.isfinite(cy)
            m = mask_xy.sum()
            if m >= 3:
                cxm, cym = cx[mask_xy], cy[mask_xy]
                denom = max(m - 1, 1)
                varx = float((cxm * cxm).sum() / denom)
                vary = float((cym * cym).sum() / denom)
                covxy = float((cxm * cym).sum() / denom)
                trace = max(varx + vary, 0.0)
                disc = max(trace * trace - 4.0 * (varx * vary - covxy * covxy), 0.0)
                sq = np.sqrt(disc)
                lam1 = max(0.5 * (trace + sq), EPS)
                lam2 = max(0.5 * (trace - sq), EPS)
                elong[i] = lam1 / lam2
                spread[i] = np.sqrt(trace)

    md_stats = pd.DataFrame({
        METASPOT_COL: row_cats,
        'r_mean': r_mean, 'r_std': r_std,
        'r_p10': r_p10, 'r_p50': r_p50, 'r_p90': r_p90,
        'd_cell_norm_mean': d1_mean, 'd_nuc_norm_mean': d2_mean,
        'elongation': elong, 'spread': spread, 'log_n': log_n
    }).set_index(METASPOT_COL)
    adata1.obs = adata1.obs.join(md_stats, how='left')
    emb_cols = [c for c in point_df.columns if c.startswith('comp_emb')]
    if len(emb_cols):
        agg_comp = (
            point_df.groupby(METASPOT_COL)[emb_cols + ['local_density']]
            .agg(['mean', 'std'])
        )
        agg_comp.columns = [f"{a}_{b}" for a, b in agg_comp.columns]
        adata1.obs = adata1.obs.join(agg_comp, how='left')
    return adata1


def _validate_adata1_point_df_alignment(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    *,
    geometry_atol: float = 1e-5,
) -> Tuple[bool, str]:
    """Verify that a cached Module-4 object matches the current MetaSpots.

    MetaSpot identifiers are local cluster numbers (for example ``cell:17``)
    and can be reassigned when Module 3 is recomputed.  Set equality alone is
    therefore insufficient: the transcript counts and geometry attached to
    every identifier must also agree before a cached ``adata1`` can be reused.
    """
    if METASPOT_COL not in point_df.columns:
        return False, f"point_df is missing {METASPOT_COL!r}"
    if not adata1.obs_names.is_unique:
        return False, "cached adata1 has duplicate MetaSpot identifiers"
    if len(point_df) == 0:
        return (
            (adata1.n_obs == 0),
            "empty point_df" if adata1.n_obs == 0 else "point_df is empty",
        )

    metaspot_ids = point_df[METASPOT_COL].astype(str)
    point_counts = (
        point_df.groupby(metaspot_ids, observed=True, sort=False)
        .size()
        .astype(np.int64)
    )
    cached_ids = pd.Index(adata1.obs_names.astype(str))
    point_ids = pd.Index(point_counts.index.astype(str))
    missing = point_ids.difference(cached_ids)
    extra = cached_ids.difference(point_ids)
    if len(missing) or len(extra):
        return (
            False,
            "MetaSpot identifier mismatch "
            f"(missing_in_cache={len(missing)}, extra_in_cache={len(extra)})",
        )

    point_counts.index = point_ids
    expected_counts = point_counts.reindex(cached_ids).to_numpy(np.int64)
    cached_counts = np.asarray(adata1.X.sum(axis=1)).ravel()
    if cached_counts.shape != expected_counts.shape or not np.allclose(
        cached_counts,
        expected_counts,
        rtol=0.0,
        atol=0.0,
    ):
        mismatched = int(
            np.count_nonzero(
                ~np.isclose(
                    cached_counts,
                    expected_counts,
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
        return False, f"per-MetaSpot transcript counts differ ({mismatched} rows)"

    if "r_signed" in point_df.columns and "r_mean" in adata1.obs.columns:
        point_r_mean = (
            pd.to_numeric(point_df["r_signed"], errors="coerce")
            .groupby(metaspot_ids, observed=True, sort=False)
            .mean()
        )
        point_r_mean.index = point_r_mean.index.astype(str)
        expected_r = point_r_mean.reindex(cached_ids).to_numpy(dtype=float)
        cached_r = pd.to_numeric(
            adata1.obs["r_mean"],
            errors="coerce",
        ).to_numpy(dtype=float)
        finite = np.isfinite(expected_r) & np.isfinite(cached_r)
        if not np.array_equal(np.isfinite(expected_r), np.isfinite(cached_r)):
            return False, "per-MetaSpot geometry missingness differs"
        if finite.any() and not np.allclose(
            cached_r[finite],
            expected_r[finite],
            rtol=0.0,
            atol=max(float(geometry_atol), 0.0),
        ):
            max_abs = float(
                np.max(np.abs(cached_r[finite] - expected_r[finite]))
            )
            return (
                False,
                "per-MetaSpot r_mean differs "
                f"(max_abs_difference={max_abs:.6g})",
            )

    return True, "aligned"


def _select_gc_feats_simple(adata1: ad.AnnData) -> List[str]:
    base_feats = [
        "r_mean", "r_std", "r_p10", "r_p50", "r_p90",
        "d_cell_norm_mean", "d_nuc_norm_mean",
        "elongation", "spread", "log_n",
    ]
    comp_feats = [c for c in adata1.obs.columns if c.startswith("comp_emb") and c.endswith("_mean")]
    comp_feats = comp_feats[: min(16, len(comp_feats))]
    if "local_density_mean" in adata1.obs.columns:
        base_feats.append("local_density_mean")
    return base_feats + comp_feats


def _spatial_feats_simple(adata1: ad.AnnData, point_df: pd.DataFrame) -> pd.DataFrame:
    metaspots = pd.Categorical(
        point_df[METASPOT_COL],
        categories=adata1.obs_names,
        ordered=False,
    )
    x = point_df["x_location"].to_numpy(np.float64, copy=False)
    y = point_df["y_location"].to_numpy(np.float64, copy=False)
    df = pd.DataFrame({METASPOT_COL: metaspots, "x": x, "y": y})
    agg = (
        df.groupby(METASPOT_COL, observed=True)
          .agg(
              n=("x", "size"),
              sx=("x", "sum"),
              sy=("y", "sum"),
              sxx=("x", lambda v: np.square(v).sum()),
              syy=("y", lambda v: np.square(v).sum()),
              sxy=("x", lambda v: (v * df.loc[v.index, "y"]).sum()),
          )
          .reindex(adata1.obs_names, fill_value=0)
    )
    n = agg["n"].to_numpy(np.float64)
    n_safe = np.maximum(n, 1)
    sx = agg["sx"].to_numpy(np.float64)
    sy = agg["sy"].to_numpy(np.float64)
    sxx = agg["sxx"].to_numpy(np.float64)
    syy = agg["syy"].to_numpy(np.float64)
    sxy = agg["sxy"].to_numpy(np.float64)
    mx, my = sx / n_safe, sy / n_safe
    Er2 = (sxx + syy) / n_safe
    Emu2 = mx * mx + my * my
    rms_radius = np.sqrt(np.maximum(Er2 - Emu2, 0.0))
    compactness = 1.0 / (rms_radius + 1.0)
    var_x = sxx / n_safe - mx * mx
    var_y = syy / n_safe - my * my
    cov_xy = sxy / n_safe - mx * my
    tr = var_x + var_y
    det = var_x * var_y - cov_xy * cov_xy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    sq = np.sqrt(disc)
    lam1 = 0.5 * (tr + sq)
    lam2 = 0.5 * (tr - sq)
    angle_consistency = lam1 / (lam2 + 1e-9)
    mask_small = n < 3
    compactness[mask_small] = 0.0
    angle_consistency[mask_small] = 0.0
    out = pd.DataFrame({
        "compactness": compactness.astype(np.float32),
        "angle_consistency": angle_consistency.astype(np.float32),
    }, index=adata1.obs_names)
    return out


def _construct_features_simple(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    speed_preset: str = "ultra",
    use_spatial_feats: bool = True,
    use_decell: bool = True,
    use_cosine: bool = True,
    w_gc: float = 0.4,
    w_expr: float = 0.6,
    w_sp: float = 0.0,
    use_gc: bool = True,
    use_expr: bool = True,
):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import normalize
    # 1) GC + composition
    feats_gc = _select_gc_feats_simple(adata1)
    X_gc = adata1.obs[feats_gc].to_numpy(np.float32, copy=False)
    X_gc = SimpleImputer(strategy="median").fit_transform(X_gc).astype(np.float32)
    X_gc = StandardScaler().fit_transform(X_gc).astype(np.float32)
    if not use_gc:
        X_gc = np.zeros_like(X_gc, dtype=np.float32)
    # 2) Expression TF-IDF + SVD
    X_counts = adata1.X.tocsr().astype(np.float32) if sp.issparse(adata1.X) else sp.csr_matrix(np.asarray(adata1.X, np.float32))
    row_sum = np.asarray(X_counts.sum(1)).ravel().astype(np.float32)
    row_sum[row_sum <= 0] = 1.0
    TF = X_counts.multiply(1.0 / row_sum[:, None])
    df_gene = np.asarray((X_counts > 0).sum(0)).ravel().astype(np.float32)
    N = X_counts.shape[0]
    idf = np.log1p(N / (1.0 + df_gene)).astype(np.float32)
    TFIDF = TF.multiply(idf)
    # Align expression SVD params with legacy defaults
    EXPR_NCOMP, SVD_N_ITER = 24, 3
    svd = TruncatedSVD(n_components=EXPR_NCOMP, random_state=42, n_iter=SVD_N_ITER)
    Z_expr = svd.fit_transform(TFIDF).astype(np.float32)
    Z_expr = StandardScaler().fit_transform(Z_expr).astype(np.float32)
    if not use_expr:
        Z_expr = np.zeros_like(Z_expr, dtype=np.float32)
    # 3) spatial shape
    if use_spatial_feats:
        spatial = _spatial_feats_simple(adata1, point_df)
        adata1.obs[["compactness", "angle_consistency"]] = spatial[["compactness", "angle_consistency"]]
        X_sp = adata1.obs[["compactness", "angle_consistency"]].fillna(0).to_numpy(np.float32)
        X_sp = StandardScaler().fit_transform(X_sp).astype(np.float32)
    else:
        X_sp = np.zeros((adata1.n_obs, 2), np.float32)
    # 4) combine with weights (configurable)
    X_base = np.hstack([float(w_gc) * X_gc, float(w_expr) * Z_expr, float(w_sp) * X_sp]).astype(np.float32)
    # 5) de-cell mean
    if "cell_id" not in adata1.obs.columns:
        tmp = (
            point_df[[METASPOT_COL, "cell_id"]]
            .drop_duplicates()
            .set_index(METASPOT_COL)
        )
        adata1.obs = adata1.obs.join(tmp, how="left")
    cell_ids = adata1.obs["cell_id"].astype(str).values
    u_cells, cell_codes = np.unique(cell_ids, return_inverse=True)
    if use_decell:
        d = X_base.shape[1]
        subs_per_cell = np.bincount(cell_codes)
        sum_by_cell = np.zeros((len(u_cells), d), np.float32)
        np.add.at(sum_by_cell, cell_codes, X_base)
        mean_by_cell = (sum_by_cell / np.maximum(1, subs_per_cell)[:, None]).astype(np.float32)
        X_all = X_base - mean_by_cell[cell_codes]
    else:
        X_all = X_base
    # 6) global standardize + L2
    X_all = StandardScaler().fit_transform(X_all).astype(np.float32)
    if use_cosine:
        X_all = normalize(X_all, norm="l2")
    X_all[~np.isfinite(X_all)] = 0.0
    return X_all, cell_ids, u_cells, cell_codes


def _choose_K_simple(
    n_sub: int,
    n_cells: int,
    target_sub_per_cluster: int = 800,
    target_cluster_per_cell: float = 1.0,
    K_min: int = 200,
    K_max: int = 8000,
):
    if n_sub <= 0:
        return max(2, K_min)
    K_by_size = max(50, int(n_sub / max(1, target_sub_per_cluster)))
    K_by_cells = int(np.clip(target_cluster_per_cell * n_cells, 50, K_max))
    K = min(K_by_size, K_by_cells)
    K = int(np.clip(K, K_min, min(K_max, n_sub // 10)))
    return max(2, K)


def _cap_relax_factor(n_c: int) -> float:
    return float(1.0 + 0.6 * np.exp(-n_c / 80.0))


def _cap_per_cell_simple(
    labels: np.ndarray,
    X: np.ndarray,
    cell_ids: np.ndarray,
    cap_ratio_base: float = 0.015,
    min_keep: int = 3,
    margin_tau: float = 0.015,
    max_frac_soft: float = 0.05,
    use_cosine: bool = True,
):
    from scipy.spatial.distance import cdist
    y = labels.astype(np.int32, copy=True)
    n, d = X.shape
    if n == 0:
        return y
    K = int(y.max()) + 1
    centers = np.zeros((K, d), np.float32)
    for k in range(K):
        idx = (y == k).nonzero()[0]
        if idx.size:
            centers[k] = X[idx].mean(0)
    if use_cosine:
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        D = 1.0 - (Xn @ centers.T)
    else:
        D = cdist(X, centers, metric="euclidean")
    uniq, inv = np.unique(cell_ids.astype(str), return_inverse=True)
    for ci in range(len(uniq)):
        idx_c = (inv == ci).nonzero()[0]
        if idx_c.size == 0:
            continue
        y_c = y[idx_c]
        n_c = len(idx_c)
        relax = _cap_relax_factor(n_c)
        cap_abs = int(np.floor(cap_ratio_base * relax * n_c))
        cap_abs = max(min_keep, cap_abs)
        cap_abs = min(cap_abs, max(int(np.ceil(max_frac_soft * n_c)), min_keep))
        vals, cnts = np.unique(y_c, return_counts=True)
        for k, cnum in zip(vals, cnts):
            if cnum <= cap_abs:
                continue
            overflow = int(cnum - cap_abs)
            idx_ck = idx_c[(y_c == k).nonzero()[0]]
            d_cur = D[idx_ck, k]
            D_alt = D[idx_ck].copy(); D_alt[:, k] = np.inf
            alt_idx = np.argmin(D_alt, axis=1)
            d_alt = D[idx_ck, alt_idx]
            margin = (d_alt - d_cur) / np.maximum(d_cur, 1e-8)
            cand = np.where(margin < margin_tau)[0]
            if cand.size == 0:
                continue
            order = np.argsort(margin[cand])
            take = cand[order[:overflow]]
            y[idx_ck[take]] = alt_idx[take]
    return y


def _mini_batch_kmeans_simple(X: np.ndarray, K: int, seed: int = 0, desc: str = "MBK") -> np.ndarray:
    n = X.shape[0]
    if n <= K:
        return np.arange(n, dtype=np.int32)
    bs = int(max(1024, min(8192, int(n * (1/5)))))
    mbk = MiniBatchKMeans(
        n_clusters=K,
        init="k-means++",
        n_init=1,
        random_state=seed,
        batch_size=bs,
        max_iter=60,
        reassignment_ratio=0.01,
        verbose=0,
    )
    y = mbk.fit_predict(X)
    return y.astype(np.int32)


def _two_stage_mb_kmeans_assign(
    X: np.ndarray,
    K: int,
    seed: int = 0,
    sample_n: int = 300000,
    batch_size: int = 8192,
    assign_chunk: int = 50000,
    use_cosine: bool = True,
    max_iter: int = 60,
):
    """Train MiniBatchKMeans on a subsample then assign all points in chunks.
    Keeps clustering quality close to full MBK while cutting runtime for very large blocks."""
    n = X.shape[0]
    if n <= K or n <= sample_n:
        return _mini_batch_kmeans_simple(X, K=K, seed=seed, desc="MBK-full")

    # 1) Train on a capped subsample (smaller default to reduce wall clock)
    sample_n = max(K, min(int(sample_n), n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_n, replace=False)
    X_train = X[idx]
    bs = max(1024, min(int(batch_size), sample_n))
    mbk = MiniBatchKMeans(
        n_clusters=K,
        init="k-means++",
        n_init=3,
        random_state=seed,
        batch_size=bs,
        max_iter=int(max_iter),
        reassignment_ratio=0.01,
        verbose=0,
    )
    mbk.fit(X_train)
    centers = mbk.cluster_centers_.astype(np.float32, copy=False)

    # 2) Chunked assignment (fast BLAS dot for cosine, stabilized quadratic form for L2)
    if use_cosine:
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    step = max(2048, int(assign_chunk))
    labels = np.empty(n, dtype=np.int32)
    if use_cosine:
        for s in range(0, n, step):
            e = min(n, s + step)
            X_chunk = X[s:e]
            sim = X_chunk @ centers.T  # cosine similarity via dot product
            labels[s:e] = np.argmax(sim, axis=1).astype(np.int32)
    else:
        c_norm = np.einsum('ij,ij->i', centers, centers)
        for s in range(0, n, step):
            e = min(n, s + step)
            X_chunk = X[s:e]
            x_norm = np.einsum('ij,ij->i', X_chunk, X_chunk)
            # d^2 = ||x||^2 + ||c||^2 - 2 x·c
            dist = x_norm[:, None] + c_norm[None, :] - 2.0 * (X_chunk @ centers.T)
            labels[s:e] = np.argmin(dist, axis=1).astype(np.int32)
    return labels


def run_module5_simple(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    speed_preset: str = "ultra",
    use_spatial_feats: bool = True,
    use_decell: bool = True,
    use_cosine: bool = True,
    enable_cap: bool = True,
    show_internal_progress: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    import time as _time
    _module_label = "Module 5 · Meta-domain Clustering"
    def _notify(desc: str):
        if progress_callback:
            try:
                progress_callback(f"{_module_label} >> Start: {desc}")
            except Exception:
                pass
    def _step_start(name: str):
        if show_internal_progress:
            print(f">> Start: Module 5 · {name}")
        _notify(name)
        return _time.perf_counter()
    def _step_done(name: str, t0: float):
        if show_internal_progress:
            dt = _time.perf_counter() - t0
            print(f"<< Done:  {name} in {dt:0.2f}s")
    # Load module5 config
    _m5 = {}
    try:
        _cfg = load_params_yaml()
        _m5 = _cfg.get('module5', {}) if _cfg else {}
    except Exception:
        _m5 = {}
    w_gc = float(_m5.get('w_gc', 0.4))
    w_expr = float(_m5.get('w_expr', 0.6))
    w_sp = float(_m5.get('w_sp', 0.0))
    speed_preset = str(_m5.get('speed_preset', speed_preset))
    use_spatial_feats = bool(_m5.get('use_spatial_feats', use_spatial_feats))
    use_decell = bool(_m5.get('use_decell', use_decell))
    use_cosine = bool(_m5.get('use_cosine', use_cosine))
    enable_cap = bool(_m5.get('enable_cap', enable_cap))
    use_gc = bool(_m5.get('use_gc', True))
    use_expr = bool(_m5.get('use_expr', True))
    # More aggressive speed defaults for large cohorts; can still be overridden in params.yaml
    train_sample_per_block = int(_m5.get('train_sample_per_block', 300000))
    assign_chunk_size = int(_m5.get('assign_chunk_size', _m5.get('block_size', 50000) or 50000))
    mbk_batch_size = int(_m5.get('mbk_batch_size', 8192))
    mbk_max_iter = int(_m5.get('mbk_max_iter', 60 if str(speed_preset).lower() == 'ultra' else 80))
    # feature construction
    X_all, cell_ids, u_cells, cell_codes = _construct_features_simple(
        adata1,
        point_df,
        speed_preset=speed_preset,
        use_spatial_feats=use_spatial_feats,
        use_decell=use_decell,
        use_cosine=use_cosine,
        w_gc=w_gc,
        w_expr=w_expr,
        w_sp=w_sp,
        use_gc=use_gc,
        use_expr=use_expr,
    )
    n_sub = X_all.shape[0]
    n_cells = len(u_cells)
    # K estimation params
    if str(_m5.get('estimate_k', 'auto')).lower() == 'fixed':
        K_total = int(_m5.get('fixed_k', 1024))
    else:
        K_total = _choose_K_simple(
            n_sub=n_sub,
            n_cells=n_cells,
            target_sub_per_cluster=int(_m5.get('target_sub_per_cluster', 800)),
            target_cluster_per_cell=float(_m5.get('target_cluster_per_cell', 1.0)),
            K_min=int(_m5.get('K_min', 200)),
            K_max=int(_m5.get('K_max', 8000)),
        )
    # split by nucleus/cytoplasm using r_p50
    split_by_nucleus = bool(_m5.get('split_by_nucleus', True))
    if split_by_nucleus and ("r_p50" in adata1.obs.columns):
        thr = float(_m5.get('nucleus_threshold', 0.0))
        is_nu = adata1.obs["r_p50"].to_numpy(np.float32) < thr
        blocks = [("N", np.where(is_nu)[0]), ("C", np.where(~is_nu)[0])]
    else:
        blocks = [("A", np.arange(n_sub, dtype=int))]
    global_labels = np.empty(n_sub, dtype=np.int32)
    offset = 0
    # Announce clustering start (single callback update)
    t0_main = _step_start('block clustering')
    rng_blocks = _progress_iter(blocks, desc='block clustering', total=len(blocks), show=show_internal_progress)
    for prefix, idx in rng_blocks:
        if idx.size == 0:
            continue
        Xi = X_all[idx]
        cell_ids_blk = cell_ids[idx]
        n_cells_blk = np.unique(cell_ids_blk).size
        K_blk = max(2, int(round(K_total * (n_cells_blk / max(1, n_cells)))))
        y = _two_stage_mb_kmeans_assign(
            Xi,
            K=K_blk,
            seed=42 + (0 if prefix == 'N' else 1),
            sample_n=train_sample_per_block,
            batch_size=mbk_batch_size,
            assign_chunk=assign_chunk_size,
            use_cosine=use_cosine,
            max_iter=mbk_max_iter,
        )
        if enable_cap:
            y = _cap_per_cell_simple(
                y,
                Xi,
                cell_ids_blk,
                cap_ratio_base=float(_m5.get('cap_ratio_base', 0.015)),
                min_keep=int(_m5.get('cap_min_keep', 3)),
                margin_tau=float(_m5.get('cap_margin_tau', 0.015)),
                max_frac_soft=float(_m5.get('cap_max_frac_soft', 0.05)),
                use_cosine=use_cosine,
            )
        u = np.unique(y); remap = {u_i: i for i, u_i in enumerate(u)}
        dense = np.vectorize(remap.get, otypes=[np.int32])(y)
        global_labels[idx] = dense + offset
        offset += dense.max() + 1
        del Xi; gc.collect()
    # Avoid duplicate print of final summary when internal iterator already printed timings.
    if not show_internal_progress:
        _step_done('block clustering', t0_main)
    adata1.obs["meta-domain"] = pd.Categorical(global_labels.astype(str))
    return adata1, pd.DataFrame(dict(n_sub=[n_sub], n_cells=[n_cells], K_total=[K_total], final_K=[int(adata1.obs["meta-domain"].nunique())]))


def build_adata2_from_adata1_and_pointdf(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    show_internal_progress: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ad.AnnData:
    import time as _time
    _module_label = "Module 6 · Aggregate Domains into Cluster Feature Table (adata2)"
    def _notify(desc: str):
        if progress_callback:
            try:
                progress_callback(f"{_module_label} >> Start: {desc}")
            except Exception:
                pass
    def _step_start(name: str):
        if show_internal_progress:
            print(f">> Start: Module 6 · {name}")
        _notify(name)
        return _time.perf_counter()
    def _step_done(name: str, t0: float):
        if show_internal_progress:
            dt = _time.perf_counter() - t0
            print(f"<< Done:  {name} in {dt:0.2f}s")
    # 1) normalize rows (TF/CPM-like)
    # Optional overall progress bar for M6 (fixed 5 steps)
    _m6_prog = None; _m6_task = None
    if show_internal_progress:
        try:
            from rich.console import Console as _Console
            from rich.progress import Progress as _Progress, SpinnerColumn as _Spin, BarColumn as _Bar, TextColumn as _Text, TimeElapsedColumn as _Time
            _m6_console = _Console(stderr=True)
            if _m6_console.is_terminal:
                _m6_prog = _Progress(_Spin(), _Text(" Building adata2"), _Bar(), _Time(), transient=True, console=_m6_console)
                _m6_task = _m6_prog.add_task("adata2", total=5)
                _m6_prog.start()
        except Exception:
            _m6_prog = None
    t0 = _step_start("Normalize counts (TF)")
    X1 = adata1.X.tocsr().astype(np.float32) if sp.issparse(adata1.X) else sp.csr_matrix(np.asarray(adata1.X, np.float32))
    rs = np.array(X1.sum(1)).ravel().astype(np.float32); rs[rs == 0] = 1.0
    X1_norm = X1.multiply(1.0 / rs[:, None])
    _step_done("Normalize counts (TF)", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 2) MetaSpot -> meta-domain mapping
    t0 = _step_start("Map metaspots to clusters")
    sub2clu = adata1.obs['meta-domain'].astype(str).values
    clu_codes, clu_uni = pd.factorize(sub2clu, sort=False)
    _step_done("Map metaspots to clusters", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 3) group mean to clusters.  Keep the true aggregate counts separate from
    # the compositional mean used by the downstream enrichment model.  SCVI
    # requires non-negative count-like input; feeding X2 (row-normalized
    # proportions) to a negative-binomial model is statistically invalid.
    t0 = _step_start("Aggregate metaspots -> cluster means")
    ones = np.ones_like(clu_codes, dtype=np.float32)
    G = sp.csr_matrix((ones, (clu_codes, np.arange(len(clu_codes)))), shape=(len(clu_uni), len(clu_codes)))
    row_sum = np.array(G.sum(1)).ravel(); row_sum[row_sum == 0] = 1.0
    X2 = (G.multiply(1.0 / row_sum[:, None]) @ X1_norm).tocsr()
    X2_counts = (G @ X1).tocsr().astype(np.float32, copy=False)
    adata2 = ad.AnnData(X=X2, obs=pd.DataFrame(index=clu_uni), var=adata1.var.copy())
    adata2.layers["counts"] = X2_counts
    adata2.uns["matrix_semantics"] = {
        "X": "mean MetaSpot transcript-frequency profile; rows sum to approximately one",
        "layers_counts": "sum of raw MetaSpot gene counts per MetaDomain",
    }
    _step_done("Aggregate metaspots -> cluster means", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 4) aggregate geometry stats
    t0 = _step_start("Aggregate geometry stats per cluster")
    want_cols = [c for c in ['r_mean','r_std','r_p50','d_cell_norm_mean','d_nuc_norm_mean','elongation','spread','log_n'] if c in adata1.obs.columns]
    agg = (adata1.obs[want_cols + ['meta-domain']]
            .groupby('meta-domain', sort=False, observed=True)
           .agg({'r_mean':'mean','r_std':'mean','r_p50':'mean','d_cell_norm_mean':'mean','d_nuc_norm_mean':'mean','elongation':'median','spread':'median','log_n':'mean'}))
    adata2.obs = adata2.obs.join(agg, how='left')
    _step_done("Aggregate geometry stats per cluster", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 5) representative cell id
    t0 = _step_start("Select representative cell per cluster")
    rep_cell = (adata1.obs[['meta-domain','cell_id']]
              .groupby('meta-domain', observed=True)['cell_id']
              .agg(lambda x: x.value_counts().index[0]))
    adata2.obs['rep_cell_id'] = adata2.obs.index.map(rep_cell.to_dict())
    _step_done("Select representative cell per cluster", t0)
    if _m6_prog and _m6_task is not None:
        try:
            _m6_prog.advance(_m6_task, 1)
            _m6_prog.stop()
        except Exception:
            pass
    return adata2


def _is_count_like_matrix(matrix, tolerance: float = 1e-4) -> bool:
    """Return True when a dense/sparse matrix is valid count-model input."""
    if matrix is None:
        return False
    X = matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix)
    values = X.data if sp.issparse(X) else X.ravel()
    if values.size == 0:
        return False
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < -float(tolerance)):
        return False
    return bool(
        np.max(np.abs(values - np.rint(values)), initial=0.0)
        <= float(tolerance)
    )


def build_module7_graph(
    adata2: ad.AnnData,
    show_internal_progress: bool = False,
    scvi_epochs: int = 100,
    out_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ad.AnnData:
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import kneighbors_graph
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse import diags
    import time as _time

    # Parameters (read from config when available)
    GEOM_FEATS = ['r_mean','r_std','r_p50','d_cell_norm_mean','d_nuc_norm_mean','elongation','spread','log_n']
    EDGE_NORM = 'rw'
    k_expr_base, k_geom_base = 32, 32
    # Fusion weights
    W_EXPR_GRAPH = 0.5
    W_GEOM_GRAPH = 0.5
    TOPK = 32
    BETA_SAMECELL = 0.5
    scvi_n_latent = 32
    scvi_lr = 1e-3
    scvi_batch_size = 512
    feature_scale_mode = 'standard'

    try:
        _cfg = load_params_yaml()
        _m7 = _cfg.get('module7', {}) if _cfg else {}
        scvi_epochs = int(_m7.get('scvi_epochs', scvi_epochs))
        # neighbors
        _nei = _m7.get('neighbors', None)
        if isinstance(_nei, dict):
            k_expr_base = int(_nei.get('k_expr', k_expr_base))
            k_geom_base = int(_nei.get('k_geom', k_geom_base))
        elif isinstance(_nei, (int, float)):
            k_expr_base = k_geom_base = int(_nei)
        # weights and prune
        W_EXPR_GRAPH = float(_m7.get('alpha_expr', W_EXPR_GRAPH))
        W_GEOM_GRAPH = float(_m7.get('alpha_geom', W_GEOM_GRAPH))
        TOPK = int(_m7.get('topk_prune', TOPK))
        BETA_SAMECELL = float(_m7.get('same_cell_beta', BETA_SAMECELL))
        EDGE_NORM = str(_m7.get('edge_norm', EDGE_NORM)).lower()
        feature_scale_mode = str(_m7.get('feature_scale', feature_scale_mode)).lower()
        scvi_n_latent = int(_m7.get('scvi_n_latent', scvi_n_latent))
        scvi_lr = float(_m7.get('scvi_lr', scvi_lr))
        scvi_batch_size = max(1, int(_m7.get('scvi_batch_size', scvi_batch_size)))
    except Exception:
        pass

    def compute_expr_latent_scvi_or_svd(
        adata2: ad.AnnData,
        n_latent: int = 32,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        # If epochs <= 0, force SVD path (skip SCVI entirely)
        if int(max_epochs) <= 0:
            X = adata2.X.tocsr().astype(np.float32) if sp.issparse(adata2.X) else sp.csr_matrix(np.asarray(adata2.X, np.float32))
            row_sum = np.array(X.sum(1)).ravel().astype(np.float32); row_sum[row_sum == 0] = 1.0
            TF = X.multiply(1.0 / row_sum[:, None])
            df = np.array((X > 0).sum(0)).ravel().astype(np.float32)
            N = X.shape[0]; idf = np.log1p(N / (1.0 + df)).astype(np.float32)
            TFIDF = TF.multiply(idf)
            svd = TruncatedSVD(n_components=n_latent, random_state=42)
            Z = svd.fit_transform(TFIDF).astype(np.float32)
            return Z, "SVD", {"n_components": int(n_latent), "tfidf": True}
        try:
            import scvi
            count_layer = adata2.layers.get("counts", None)
            if not _is_count_like_matrix(count_layer):
                raise ValueError(
                    "SCVI requires a non-negative integer-like counts layer; "
                    "falling back to TF-IDF/SVD."
                )
            # Quiet mode: suppress progress bars and logs
            try:
                scvi.settings.verbosity = 0
            except Exception:
                pass
            # Silence Lightning banners (again, in case import order differs)
            try:
                import logging as _logging
                for _name in ("lightning", "lightning.pytorch", "pytorch_lightning"):
                    _logging.getLogger(_name).setLevel(_logging.ERROR)
            except Exception:
                pass
            import warnings as _warnings
            with _warnings.catch_warnings():
                scvi.model.SCVI.setup_anndata(adata2, layer="counts")
            model = scvi.model.SCVI(adata2, n_latent=n_latent)
            import warnings as _warnings
            with _warnings.catch_warnings():
                model.train(
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    plan_kwargs={"lr": learning_rate},
                    check_val_every_n_epoch=None,
                    enable_progress_bar=False,
                    logger=False,
                    enable_model_summary=False,
                    log_every_n_steps=0,
                )
            Z = model.get_latent_representation().astype(np.float32)
            return Z, "SCVI", {
                "n_latent": int(n_latent),
                "epochs": int(max_epochs),
                "lr": float(learning_rate),
                "batch_size": int(batch_size),
            }
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "SCVI expression embedding unavailable or invalid; using "
                "TF-IDF/SVD instead: %s",
                exc,
            )
            X = adata2.X.tocsr().astype(np.float32) if sp.issparse(adata2.X) else sp.csr_matrix(np.asarray(adata2.X, np.float32))
            row_sum = np.array(X.sum(1)).ravel().astype(np.float32); row_sum[row_sum == 0] = 1.0
            TF = X.multiply(1.0 / row_sum[:, None])
            df = np.array((X > 0).sum(0)).ravel().astype(np.float32)
            N = X.shape[0]; idf = np.log1p(N / (1.0 + df)).astype(np.float32)
            TFIDF = TF.multiply(idf)
            svd = TruncatedSVD(n_components=n_latent, random_state=42)
            Z = svd.fit_transform(TFIDF).astype(np.float32)
            return Z, "SVD(fallback)", {"n_components": int(n_latent), "tfidf": True}

    def _build_knn_connectivities_by_scanpy(adata: ad.AnnData, rep_key: str, n_neighbors: int = 20) -> sp.csr_matrix:
            # Prefer sklearn kneighbors_graph for performance; fallback to scanpy
            try:
                G_ex = kneighbors_graph(adata.obsm[rep_key], n_neighbors=n_neighbors, mode='connectivity', include_self=False, n_jobs=_preferred_n_jobs())
                A = G_ex.maximum(G_ex.T).astype(np.float32).tocsr()
                A.eliminate_zeros(); A.sort_indices()
                return A
            except Exception as e:
                _logging.getLogger("cellscope").warning(f"kneighbors_graph failed: {e}; falling back to scanpy.pp.neighbors.")
                try:
                    import scanpy as sc
                    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=rep_key, method='umap', key_added=f'{rep_key}_nbrs')
                    C = adata.obsp.get(f'{rep_key}_nbrs_connectivities', None)
                    if C is None:
                        C = adata.obsp['connectivities']
                    C = C.tocsr().astype(np.float32)
                    C = ((C + C.T) * 0.5).tocsr()
                    C.eliminate_zeros(); C.sort_indices()
                    return C
                except Exception as e2:
                    _logging.getLogger("cellscope").error(f"scanpy neighbors failed: {e2}; returning empty CSR.")
                    n = adata.n_obs
                    return sp.csr_matrix((n, n), dtype=np.float32)

    def _row_topk_csr(A: sp.csr_matrix, k: int = 16) -> sp.csr_matrix:
        A = A.tocsr().astype(np.float32, copy=False)
        n = A.shape[0]; indptr, indices, data = A.indptr, A.indices, A.data
        rows, cols, vals = [], [], []
        for i in range(n):
            s, e = indptr[i], indptr[i + 1]
            if s == e:
                rows.append(np.array([i], dtype=np.int32)); cols.append(np.array([i], dtype=np.int32)); vals.append(np.array([1e-6], dtype=np.float32)); continue
            seg_idx, seg_val = indices[s:e], data[s:e]
            if e - s <= k:
                rows.append(np.full(e - s, i, dtype=np.int32)); cols.append(seg_idx); vals.append(seg_val)
            else:
                top = np.argpartition(-seg_val, k - 1)[:k]
                top = top[np.argsort(-seg_val[top])]
                rows.append(np.full(k, i, dtype=np.int32)); cols.append(seg_idx[top]); vals.append(seg_val[top])
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals).astype(np.float32)
        B = sp.csr_matrix((vals, (rows, cols)), shape=A.shape, dtype=np.float32)
        B = B.maximum(B.T).tocsr(); B.eliminate_zeros(); B.sort_indices()
        return B

    _module_label = "Module 7 · Graph Construction"
    def _notify(desc: str):
        if progress_callback:
            try:
                progress_callback(f"{_module_label} >> Start: {desc}")
            except Exception:
                pass
    def _step_start(name: str):
        if show_internal_progress:
            print(f">> Start: Module 7 · {name}")
        _notify(name)
        return _time.perf_counter()
    def _step_done(name: str, t0: float):
        if show_internal_progress:
            dt = _time.perf_counter() - t0
            print(f"<< Done:  {name} in {dt:0.2f}s")

    # Optional overall progress bar for M7 (fixed 7 steps)
    _m7_prog = None; _m7_task = None
    if show_internal_progress:
        try:
            from rich.console import Console as _Console
            from rich.progress import Progress as _Progress, SpinnerColumn as _Spin, BarColumn as _Bar, TextColumn as _Text, TimeElapsedColumn as _Time
            _m7_console = _Console(stderr=True)
            if _m7_console.is_terminal:
                _m7_prog = _Progress(_Spin(), _Text(" Build graph"), _Bar(), _Time(), transient=True, console=_m7_console)
                _m7_task = _m7_prog.add_task("graph", total=7)
                _m7_prog.start()
        except Exception:
            _m7_prog = None

    # 1) expression latent
    t0 = _step_start("Compute expression latent (SCVI/SVD)")
    Z_expr, _expr_method, _expr_params = compute_expr_latent_scvi_or_svd(
        adata2,
        n_latent=scvi_n_latent,
        max_epochs=scvi_epochs,
        learning_rate=scvi_lr,
        batch_size=scvi_batch_size,
    )
    adata2.obsm['X_expr'] = Z_expr
    # Record method + params for auditing and print a concise line when progress is on
    try:
        adata2.uns = adata2.uns if isinstance(adata2.uns, dict) else {}
        adata2.uns['module7_expr_latent'] = {"method": _expr_method, "params": _expr_params}
    except Exception:
        pass
    if show_internal_progress:
        try:
            print(f"Module 7 · Expr latent method: {_expr_method}; params={_expr_params}")
        except Exception:
            pass
    _step_done("Compute expression latent (SCVI/SVD)", t0)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass

    # 2) geometry stats
    t0 = _step_start("Impute + scale geometry features")
    coords = adata2.obs[GEOM_FEATS].to_numpy(dtype=np.float32, copy=True)
    coords = SimpleImputer(strategy='median').fit_transform(coords).astype(np.float32)
    Gm = StandardScaler().fit_transform(coords).astype(np.float32)
    adata2.obsm['X_geom'] = Gm
    _step_done("Impute + scale geometry features", t0)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass

    # 3) build graphs
    t0_graph = _step_start("Build expression kNN graph")
    n_obs2 = adata2.n_obs
    k_expr = int(np.clip(k_expr_base if n_obs2 < 80_000 else 16, 10, 30))
    k_geom = int(np.clip(k_geom_base if n_obs2 < 80_000 else 16, 10, 30))
    A_ex = _build_knn_connectivities_by_scanpy(adata2, 'X_expr', n_neighbors=k_expr)
    _step_done("Build expression kNN graph", t0_graph)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass
    t0_graph2 = _step_start("Build geometry kNN graph")
    A_gm = _build_knn_connectivities_by_scanpy(adata2, 'X_geom', n_neighbors=k_geom)
    _step_done("Build geometry kNN graph", t0_graph2)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass

    # 4) adaptive alpha fusion + same-cell downweight
    t0 = _step_start("Fuse graphs + same-cell downweight")
    inter = (A_ex.multiply(A_gm)).astype(bool).sum(1).A.ravel()
    deg_ex = A_ex.getnnz(axis=1); deg_gm = A_gm.getnnz(axis=1)
    union = np.maximum(deg_ex + deg_gm - inter, 1)
    alpha_vec = np.clip(inter / union, 0.15, 0.85).astype(np.float32)
    D_alpha = diags(alpha_vec, 0, dtype=np.float32)
    D_alpha1 = diags(1.0 - alpha_vec, 0, dtype=np.float32)
    A_mix = (W_EXPR_GRAPH * D_alpha @ A_ex + W_GEOM_GRAPH * D_alpha1 @ A_gm).tocsr()
    A_mix = ((A_mix + A_mix.T) * 0.5).tocsr(); A_mix.eliminate_zeros(); A_mix.sort_indices()
    if 'rep_cell_id' in adata2.obs.columns and BETA_SAMECELL != 1.0:
        cid = adata2.obs['rep_cell_id'].astype(str).values
        _, cidnum = np.unique(cid, return_inverse=True)
        rows, cols = A_mix.nonzero()
        same = (cidnum[rows] == cidnum[cols])
        A_mix.data[same] *= float(BETA_SAMECELL)
    _step_done("Fuse graphs + same-cell downweight", t0)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass

    t0 = _step_start("Row top-k pruning (k=TOPK)")
    A_pruned = _row_topk_csr(A_mix, TOPK)
    _step_done("Row top-k pruning (k=TOPK)", t0)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass

    # Save degree histogram visualization (intermediate)
    try:
        t_vis = _time.perf_counter()
        import matplotlib.pyplot as plt
        deg = np.asarray(A_pruned.getnnz(axis=1)).ravel()
        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()
        ax.hist(deg, bins=50, color="#4e79a7")
        ax.set_xlabel("Graph degree per node")
        ax.set_ylabel("count")
        plt.tight_layout()
        _save_fig(fig, out_dir or os.getcwd(), "module7_degree_hist")
        plt.close(fig)
        if show_internal_progress:
            print(f"<< Done:  Module 7 · Visualize degree distribution in {_time.perf_counter()-t_vis:0.2f}s")
    except Exception:
        pass

    # normalization
    t0 = _step_start("Normalize adjacency + scale features")
    if EDGE_NORM == 'rw':
        row_sum = np.array(A_pruned.sum(1)).ravel().astype(np.float32); row_sum[row_sum <= 0] = 1.0
        D_inv = diags((1.0 / row_sum).astype(np.float32))
        A_norm = (D_inv @ A_pruned).astype(np.float32)
    elif EDGE_NORM == 'gcn':
        from scipy.sparse import eye
        A_hat = (A_pruned + eye(A_pruned.shape[0], dtype=np.float32)).tocsr()
        d = np.array(A_hat.sum(1)).ravel().astype(np.float32) + 1e-8
        d_inv_sqrt = 1.0 / np.sqrt(d)
        D_inv_sqrt = diags(d_inv_sqrt.astype(np.float32), 0)
        A_norm = (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)
    else:
        A_norm = A_pruned.astype(np.float32)
    A_norm.eliminate_zeros(); A_norm.sort_indices()
    adata2.obsp['A_norm'] = A_norm
    if feature_scale_mode == 'standard':
        adata2.obsm['X_expr_scaled'] = StandardScaler().fit_transform(adata2.obsm['X_expr']).astype(np.float32)
        adata2.obsm['X_geom_scaled'] = StandardScaler().fit_transform(adata2.obsm['X_geom']).astype(np.float32)
    else:
        adata2.obsm['X_expr_scaled'] = adata2.obsm['X_expr'].astype(np.float32, copy=False)
        adata2.obsm['X_geom_scaled'] = adata2.obsm['X_geom'].astype(np.float32, copy=False)
    _step_done("Normalize adjacency + scale features", t0)
    if _m7_prog and _m7_task is not None:
        try:
            _m7_prog.advance(_m7_task, 1)
            _m7_prog.stop()
        except Exception:
            pass
    return adata2


def run_module8_dgi(adata1: ad.AnnData, adata2: ad.AnnData, show_internal_progress: bool = False, dgi_epochs: int = 100, enable_dgi_sage: bool = False, out_dir: Optional[str] = None, progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[ad.AnnData, ad.AnnData]:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv
    from torch_geometric.nn.models import DeepGraphInfomax
    from torch_geometric.utils import from_scipy_sparse_matrix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_norm = adata2.obsp["A_norm"].tocsr()
    edge_index, edge_weight = from_scipy_sparse_matrix(A_norm)
    edge_weight = torch.clamp(torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0), min=1e-8)
    X_all = np.concatenate([adata2.obsm["X_expr_scaled"], adata2.obsm["X_geom_scaled"]], axis=1).astype(np.float32)
    x = torch.from_numpy(np.nan_to_num(X_all, nan=0.0)).to(torch.float32).to(device)
    x = F.normalize(x, p=2, dim=1)
    ei = edge_index.to(device, non_blocking=True)
    ew = edge_weight.to(device, non_blocking=True).to(torch.float32)

    # Load DGI hyperparameters from config (with safe defaults)
    dgi_hidden = 64
    dgi_out = 32
    dgi_lr = 1e-3
    dgi_wd = 1e-4
    dgi_clip = 1.0
    dgi_amp = (device.type == "cuda")
    try:
        _cfg = load_params_yaml()
        _m8 = _cfg.get('module8', {}) if _cfg else {}
        dgi_epochs = int(_m8.get('dgi_epochs', dgi_epochs))
        enable_dgi_sage = bool(_m8.get('enable_dgi_sage', enable_dgi_sage))
        dgi_hidden = int(_m8.get('dgi_hidden', dgi_hidden))
        dgi_out = int(_m8.get('dgi_out', dgi_out))
        dgi_lr = float(_m8.get('dgi_lr', dgi_lr))
        dgi_wd = float(_m8.get('dgi_weight_decay', dgi_wd))
        dgi_clip = float(_m8.get('dgi_clip_grad', dgi_clip))
        dgi_amp = bool(_m8.get('amp', dgi_amp))
    except Exception:
        pass

    class Encoder(nn.Module):
        def __init__(self, in_channels, hidden=64, out_channels=32, conv_type="gcn", dropout=0.0):
            super().__init__()
            Conv = {"gcn": GCNConv, "sage": SAGEConv}[conv_type]
            if conv_type == "gcn":
                self.conv1 = Conv(in_channels, hidden, normalize=False, improved=True, add_self_loops=False)
                self.conv2 = Conv(hidden, out_channels, normalize=False, improved=True, add_self_loops=False)
            else:
                self.conv1 = Conv(in_channels, hidden)
                self.conv2 = Conv(hidden, out_channels)
            self.act = nn.PReLU(); self.drop = nn.Dropout(dropout); self.conv_type = conv_type
        def forward(self, x, edge_index, edge_weight=None):
            if self.conv_type == "gcn":
                x = self.conv1(x, edge_index, edge_weight=edge_weight); x = self.act(x); x = self.drop(x); x = self.conv2(x, edge_index, edge_weight=edge_weight)
            else:
                x = self.conv1(x, edge_index); x = self.act(x); x = self.drop(x); x = self.conv2(x, edge_index)
            return x
        @torch.no_grad()
        def get_embeddings(self, x, edge_index, edge_weight=None):
            self.eval(); return self.forward(x, edge_index, edge_weight)

    def build_dgi(conv_type="gcn", hidden=dgi_hidden, out_channels=dgi_out):
        enc = Encoder(in_channels=x.size(-1), hidden=hidden, out_channels=out_channels, conv_type=conv_type, dropout=0.0).to(device)
        def corruption(x_in, edge_index, edge_weight=None):
            perm = torch.randperm(x_in.size(0), device=x_in.device); x_cor = x_in[perm]; return x_cor, edge_index, edge_weight
        model = DeepGraphInfomax(hidden_channels=out_channels, encoder=enc, summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)), corruption=corruption).to(device)
        return model, enc

    def _make_scaler(use_amp: bool):
        # Compatibility wrapper for different torch versions
        try:
            from torch import amp as _amp
            return _amp.GradScaler(enabled=use_amp)
        except Exception:
            try:
                return torch.cuda.amp.GradScaler(enabled=use_amp)
            except Exception:
                class _Dummy:
                    def scale(self, loss): return loss
                    def unscale_(self, opt): pass
                    def step(self, opt): opt.step()
                    def update(self): pass
                return _Dummy()

    def train_dgi(model, epochs: int, lr=dgi_lr, wd=dgi_wd, clip_grad=dgi_clip, use_amp=dgi_amp, tag="GCN"):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        # Auto-detect and set medium precision on GPU for performance
        if device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("medium")
            except Exception:
                pass
        scaler = _make_scaler(use_amp)
        model.train()
        losses_hist, ema_hist = [], []
        ema = None
        # Progress bar per-epoch when internal progress enabled
        _m8_console = None; _m8_prog = None; _m8_task = None
        if show_internal_progress:
            try:
                from rich.console import Console as _Console
                from rich.progress import Progress as _Progress, BarColumn as _Bar, TextColumn as _Text, TimeElapsedColumn as _Time, TaskProgressColumn as _TaskProg
                _m8_console = _Console(stderr=True)
                if _m8_console.is_terminal:
                    _m8_prog = _Progress(_Text(f" [bold][M8] Train DGI-{tag}"), _Bar(), _TaskProg(), _Time(), transient=True, console=_m8_console)
                    _m8_task = _m8_prog.add_task(f"DGI-{tag}", total=epochs)
                    _m8_prog.start()
            except Exception:
                _m8_prog = None
        for ep in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            if use_amp:
                try:
                    from torch import amp as _amp
                    ctx = _amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
                except Exception:
                    ctx = torch.cuda.amp.autocast(dtype=torch.float16)
                with ctx:
                    pos_z, neg_z, summary = model(x, ei, ew); loss = model.loss(pos_z, neg_z, summary)
                try:
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                except Exception:
                    loss.backward()
                if clip_grad is not None: nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                try:
                    scaler.step(opt); scaler.update()
                except Exception:
                    opt.step()
            else:
                pos_z, neg_z, summary = model(x, ei, ew); loss = model.loss(pos_z, neg_z, summary)
                loss.backward();
                if clip_grad is not None: nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                opt.step()
            if show_internal_progress and ep % 25 == 0:
                print(f"[DGI-{tag}] epoch {ep}/{epochs} loss={float(loss):.4f}")
            if _m8_prog and _m8_task is not None:
                try: _m8_prog.advance(_m8_task, 1)
                except Exception: pass
            # record history
            l = float(loss)
            losses_hist.append(l)
            ema = l if ema is None else (0.9 * ema + 0.1 * l)
            ema_hist.append(ema)
        # save training curves
        try:
            import matplotlib.pyplot as plt
            import numpy as _np
            fig = plt.figure(figsize=(7,4))
            plt.plot(_np.array(losses_hist), label=f"{tag}-loss", linewidth=1.0)
            plt.plot(_np.array(ema_hist), label=f"{tag}-ema", linewidth=1.5, linestyle="--")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("DGI Training Curves"); plt.legend(frameon=False); plt.tight_layout()
            _save_fig(fig, out_dir or os.getcwd(), f"module8_dgi_{tag}_curves")
            plt.close(fig)
        except Exception:
            pass
        try:
            if _m8_prog:
                _m8_prog.stop()
        except Exception:
            pass
        return model

    @torch.no_grad()
    def get_embeddings(encoder):
        z = encoder.get_embeddings(x, ei, ew); z = z.detach().cpu().numpy().astype(np.float32, copy=False); return z

    import time as _time
    _module_label = "Module 8 · Graph Embeddings (DGI)"
    def _notify(desc: str):
        if progress_callback:
            try:
                progress_callback(f"{_module_label} >> Start: {desc}")
            except Exception:
                pass
    def _step_start(name: str):
        if show_internal_progress:
            print(f">> Start: Module 8 · {name}")
        _notify(name)
        return _time.perf_counter()
    def _step_done(name: str, t0: float):
        if show_internal_progress:
            dt = _time.perf_counter() - t0
            print(f"<< Done:  {name} in {dt:0.2f}s")
    # Train GCN
    t0 = _step_start("Build + train DGI (GCN)")
    dgi_gcn, enc_gcn = build_dgi(conv_type="gcn", hidden=dgi_hidden, out_channels=dgi_out)
    dgi_gcn = train_dgi(dgi_gcn, epochs=dgi_epochs, tag="GCN")
    Z_gcn = get_embeddings(enc_gcn)
    _step_done("Build + train DGI (GCN)", t0)
    del dgi_gcn, enc_gcn
    if device.type == "cuda": torch.cuda.empty_cache();
    # Optional SAGE
    Z_sage = None
    if enable_dgi_sage:
        try:
            t0 = _step_start("Build + train DGI (SAGE)")
            dgi_sage, enc_sage = build_dgi(conv_type="sage", hidden=dgi_hidden, out_channels=dgi_out)
            dgi_sage = train_dgi(dgi_sage, epochs=dgi_epochs, tag="SAGE")
            Z_sage = get_embeddings(enc_sage)
            _step_done("Build + train DGI (SAGE)", t0)
            del dgi_sage, enc_sage
            if device.type == "cuda": torch.cuda.empty_cache()
        except Exception:
            Z_sage = None

    # Fill to adata2
    t0 = _step_start("Attach embeddings to adata2 + broadcast to adata1")
    adata2.obsm["X_dgi_gcn"] = Z_gcn
    if Z_sage is not None:
        adata2.obsm["X_dgi_sage"] = Z_sage
    # Map back to adata1 by cluster
    clu_to_row = {clu: i for i, clu in enumerate(adata2.obs_names.astype(str))}
    row_idx_adata1 = adata1.obs["meta-domain"].astype(str).map(clu_to_row).to_numpy()
    def _broadcast(Z, row_index):
        out = np.zeros((len(row_index), Z.shape[1]), dtype=np.float32)
        mask = (row_index >= 0) & (row_index < Z.shape[0])
        out[mask] = Z[row_index[mask]]
        return out
    adata1.obsm["X_dgi_gcn"] = _broadcast(Z_gcn, row_idx_adata1)
    if Z_sage is not None:
        adata1.obsm["X_dgi_sage"] = _broadcast(Z_sage, row_idx_adata1)
    _step_done("Attach embeddings to adata2 + broadcast to adata1", t0)
    return adata1, adata2


def build_anndata(spatial_df: pd.DataFrame,
                  cell_df: pd.DataFrame,
                  nucleus_df: pd.DataFrame,
                  show_internal_progress: bool = False,
                  progress_callback: Optional[Callable[[str], None]] = None,
                  out_dir: Optional[str] = None,
                  scvi_epochs: int = 100,
                  dgi_epochs: int = 100,
                  enable_dgi_sage: bool = False) -> Tuple[ad.AnnData, Optional[ad.AnnData], pd.DataFrame]:
    # Fine-grained resume within build_anndata
    if out_dir is None:
        out_dir = os.getcwd()
    state_prev = _load_state(out_dir) or {}
    try:
        cfg = load_params_yaml()
    except Exception:
        cfg = {}
    cur_mod_fps = fingerprint_modules_set(cfg)
    prev_mod_fps = state_prev.get("module_fingerprints") or {}
    # First full module run: state has no 'modules_completed' flag yet
    _first_run = not bool(state_prev.get("modules_completed"))
    module_order = ['m1','m2','m3','m4','m5','m6','m7','m8']
    # Determine earliest changed module among those with params (m1,m2,m3,m5,m6,m7,m8)
    start_mod = 'm1'
    for m in ['m1','m2','m3','m5','m6','m7','m8']:
        if prev_mod_fps.get(m) != cur_mod_fps.get(m):
            start_mod = m
            break
    def _mod_le(m: str, target: str) -> bool:
        return module_order.index(m) <= module_order.index(target)
    def _mod_lt(m: str, target: str) -> bool:
        return module_order.index(m) < module_order.index(target)
    try:
        import logging as _logging
        _logging.getLogger("cellscope").info("Fine-grained resume: start module %s", start_mod)
    except Exception:
        pass
    # A downstream checkpoint is reusable only while every earlier module in
    # this invocation also came from its matching checkpoint.  With
    # lite_intermediate enabled, an upstream DataFrame cache may be absent even
    # though a later AnnData cache exists.  Recomputing the upstream module can
    # renumber MetaSpots, so loading any later cache would silently cross-wire
    # expression and geometry.
    _cache_chain_valid = True

    # Use unique cell ids to avoid row-level list blow-up (can be hundreds of millions).
    sampled_cell_ids = pd.unique(spatial_df['cell_id'])
    # Read Module 1 geometry options
    _m1_include_dir = True
    _m1_winsor = (1, 99)
    try:
        _cfg = load_params_yaml()
        _m1 = _cfg.get('module1', {}) if _cfg else {}
        _m1_include_dir = bool(_m1.get('include_directional', _m1_include_dir))
        wl = _m1.get('winsor_perc_low', _m1_winsor[0]); wh = _m1.get('winsor_perc_high', _m1_winsor[1])
        _m1_winsor = (int(wl), int(wh))
    except Exception:
        pass
    # Module 1: geometry features (resume if unchanged and cache exists)
    point_df = None
    _m1_recomputed = False
    if _mod_lt('m1', start_mod):
        # try load cache
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module1_point_features")
        if pd_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 1 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 1 cache (module1_point_features)")
            except Exception:
                pass
            point_df = pd_cache
        else:
            if progress_callback: progress_callback("Module 1 · Geometry Features")
            point_df = build_point_geometry_v2(
                transcripts_df=spatial_df,
                nucleus_boundaries_df=nucleus_df,
                cell_boundaries_df=cell_df,
                sampled_cells=sampled_cell_ids,
                include_directional=_m1_include_dir,
                winsor_perc=_m1_winsor,
                show_internal_progress=show_internal_progress,
            )
            _save_dataframe(point_df, out_dir, "module1_point_features")
            _m1_recomputed = True
    else:
        if progress_callback: progress_callback("Module 1 · Geometry Features")
        point_df = build_point_geometry_v2(
            transcripts_df=spatial_df,
            nucleus_boundaries_df=nucleus_df,
            cell_boundaries_df=cell_df,
            sampled_cells=sampled_cell_ids,
            include_directional=_m1_include_dir,
            winsor_perc=_m1_winsor,
            show_internal_progress=show_internal_progress,
        )
        _save_dataframe(point_df, out_dir, "module1_point_features")
        _m1_recomputed = True
    if _m1_recomputed:
        _cache_chain_valid = False
    # Report Module 1 completion
    try:
        import logging as _logging
        _suf = None if (_first_run and _m1_recomputed) else ("recomputed" if _m1_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 1 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m1_recomputed) else ("recomputed" if _m1_recomputed else "cache")
            progress_callback("Module 1 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass
    # Module 2: composition embeddings (+gmm features)
    _m2_use_point_source = False
    try:
        _cfg = load_params_yaml() or {}
        _m2_cfg = _cfg.get('module2', {}) if _cfg else {}
        _m2_use_point_source = bool(_m2_cfg.get('source_from_point_df', False))
    except Exception:
        pass
    # Module 2 only selects the columns it needs inside per-cell blocks. Passing
    # the original table avoids copying hundreds of millions of rows up front.
    if _m2_use_point_source and {'cell_id', 'x_location', 'y_location', 'feature_name'}.issubset(point_df.columns):
        _m2_source_df = point_df
    else:
        _m2_source_df = spatial_df

    gmm_cols: List[str] = []
    composition_model: Optional[_LocalCompositionModel] = None
    _m2_recomputed = False
    if _mod_lt('m2', start_mod) and _cache_chain_valid:
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module2_point_df")
        if pd_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 2 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 2 cache (module2_point_df)")
            except Exception:
                pass
            point_df = pd_cache
            _geom = ['r_signed', 'd_cell_norm', 'd_nuc_norm']
            _ang = ['sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta']
            _geom = [c for c in _geom + _ang if c in point_df.columns]
            gmm_cols = [c for c in point_df.columns if str(c).startswith("gmm_feat_")]
            if len(gmm_cols) == 0:
                # New low-memory path: Module 2 no longer materializes gmm_feat_*.
                _comp = [c for c in point_df.columns if str(c).startswith('comp_emb')]
                _ld = ['local_density'] if 'local_density' in point_df.columns else []
                gmm_cols = _geom + _comp + _ld
            model_path = os.path.join(_intermediate_dir(out_dir), "module2_local_composition.joblib")
            try:
                import joblib
                if os.path.exists(model_path):
                    composition_model = joblib.load(model_path)
                    gmm_cols = _geom + composition_model.feature_names + ['local_density']
            except Exception:
                composition_model = None
            if composition_model is None:
                _, gmm_cols, rebuilt_model = build_composition_embeddings(
                    point_df,
                    _m2_source_df,
                    sampled_cell_ids,
                    show_internal_progress=show_internal_progress,
                )
                if isinstance(rebuilt_model, _LocalCompositionModel):
                    composition_model = rebuilt_model
                    try:
                        import joblib
                        joblib.dump(composition_model, model_path)
                    except Exception:
                        pass
        else:
            if progress_callback: progress_callback("Module 2 · Composition Embeddings")
            point_df, gmm_cols, cell_emb = build_composition_embeddings(point_df, _m2_source_df, sampled_cell_ids, show_internal_progress=show_internal_progress)
            if isinstance(cell_emb, _LocalCompositionModel):
                composition_model = cell_emb
                try:
                    import joblib
                    joblib.dump(composition_model, os.path.join(_intermediate_dir(out_dir), "module2_local_composition.joblib"))
                except Exception:
                    pass
            # Attach cell-level composition embedding for downstream MetaSpot segmentation
            try:
                if isinstance(cell_emb, pd.DataFrame):
                    setattr(point_df, '_cell_comp_emb', cell_emb)
            except Exception:
                pass
            _save_dataframe(point_df, out_dir, "module2_point_df")
            _m2_recomputed = True
    else:
        if progress_callback: progress_callback("Module 2 · Composition Embeddings")
        point_df, gmm_cols, cell_emb = build_composition_embeddings(point_df, _m2_source_df, sampled_cell_ids, show_internal_progress=show_internal_progress)
        if isinstance(cell_emb, _LocalCompositionModel):
            composition_model = cell_emb
            try:
                import joblib
                joblib.dump(composition_model, os.path.join(_intermediate_dir(out_dir), "module2_local_composition.joblib"))
            except Exception:
                pass
        try:
            if isinstance(cell_emb, pd.DataFrame):
                setattr(point_df, '_cell_comp_emb', cell_emb)
        except Exception:
            pass
        _save_dataframe(point_df, out_dir, "module2_point_df")
        _m2_recomputed = True
    if _m2_recomputed:
        _cache_chain_valid = False
    try:
        import logging as _logging
        _suf = None if (_first_run and _m2_recomputed) else ("recomputed" if _m2_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 2 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m2_recomputed) else ("recomputed" if _m2_recomputed else "cache")
            progress_callback("Module 2 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass
    # Module 3: per-cell MetaSpot segmentation
    # Read Module 3 clustering options from YAML
    _m3_hdb_min = 6
    _m3_allow_single = True
    _m3_min_samples_frac = 0.3
    _m3_target_sub_size = 12
    _m3_n_feat_hdb = 24
    _m3_kmeans_min_k = 2
    _m3_kmeans_max_k = 999
    _m3_use_k_cap = False
    _m3_merge_min = None
    _m3_hdbscan_max_points = 512
    _m3_kmeans_train_size = 1024
    _m3_kmeans_batch_size = 2048
    _m3_small_backend_cells = 12000
    _m3_sequential_cells = 256
    _m3_enforce_spatial_connectivity = False
    _m3_spatial_connectivity_k = 8
    _m3_spatial_radius_quantile = 0.9
    _m3_spatial_radius_factor = 1.5
    _m3_spatial_smoothing_iterations = 1
    _m3_spatial_smoothing_min_fraction = 0.6
    _m3_spatial_min_component_size = None
    _m3_spatial_compactness = 1.0
    _m3_target_sites_per_metaspot = None
    _m3_min_metaspots_per_cell = 1
    _m3_max_metaspots_per_cell = None
    try:
        _cfg = load_params_yaml()
        _m3 = _cfg.get('module3', {}) if _cfg else {}
        if isinstance(_m3.get('hdbscan_min_cluster_size'), (int, float)):
            _m3_hdb_min = int(_m3.get('hdbscan_min_cluster_size'))
        if 'hdbscan_allow_single_cluster' in _m3:
            _m3_allow_single = bool(_m3.get('hdbscan_allow_single_cluster'))
        if isinstance(_m3.get('hdbscan_min_samples_frac'), (int, float)):
            _m3_min_samples_frac = float(_m3.get('hdbscan_min_samples_frac'))
        if isinstance(_m3.get('target_sub_size'), (int, float)):
            _m3_target_sub_size = int(_m3.get('target_sub_size')) if _m3.get('target_sub_size') is not None else _m3_target_sub_size
        if isinstance(_m3.get('n_feat_hdb'), (int, float)):
            _m3_n_feat_hdb = int(_m3.get('n_feat_hdb')) if _m3.get('n_feat_hdb') is not None else _m3_n_feat_hdb
        if isinstance(_m3.get('kmeans_min_k'), (int, float)):
            _m3_kmeans_min_k = int(_m3.get('kmeans_min_k'))
        if isinstance(_m3.get('kmeans_max_k'), (int, float)):
            _m3_kmeans_max_k = int(_m3.get('kmeans_max_k'))
        if 'use_k_cap' in _m3:
            _m3_use_k_cap = bool(_m3.get('use_k_cap'))
        # allow null (None) to mean: use MIN_SUB_PTS for merging
        if 'merge_min_size' in _m3:
            val = _m3.get('merge_min_size')
            _m3_merge_min = int(val) if isinstance(val, (int, float)) else None
        if isinstance(_m3.get('hdbscan_max_points'), (int, float)):
            _m3_hdbscan_max_points = int(_m3.get('hdbscan_max_points'))
        if isinstance(_m3.get('kmeans_train_size'), (int, float)):
            _m3_kmeans_train_size = int(_m3.get('kmeans_train_size'))
        if isinstance(_m3.get('kmeans_batch_size'), (int, float)):
            _m3_kmeans_batch_size = int(_m3.get('kmeans_batch_size'))
        if isinstance(_m3.get('small_backend_cells'), (int, float)):
            _m3_small_backend_cells = int(_m3.get('small_backend_cells'))
        if isinstance(_m3.get('sequential_cells'), (int, float)):
            _m3_sequential_cells = int(_m3.get('sequential_cells'))
        if 'enforce_spatial_connectivity' in _m3:
            _m3_enforce_spatial_connectivity = bool(_m3.get('enforce_spatial_connectivity'))
        if isinstance(_m3.get('spatial_connectivity_k'), (int, float)):
            _m3_spatial_connectivity_k = int(_m3.get('spatial_connectivity_k'))
        if isinstance(_m3.get('spatial_radius_quantile'), (int, float)):
            _m3_spatial_radius_quantile = float(_m3.get('spatial_radius_quantile'))
        if isinstance(_m3.get('spatial_radius_factor'), (int, float)):
            _m3_spatial_radius_factor = float(_m3.get('spatial_radius_factor'))
        if isinstance(_m3.get('spatial_smoothing_iterations'), (int, float)):
            _m3_spatial_smoothing_iterations = int(_m3.get('spatial_smoothing_iterations'))
        if isinstance(_m3.get('spatial_smoothing_min_fraction'), (int, float)):
            _m3_spatial_smoothing_min_fraction = float(_m3.get('spatial_smoothing_min_fraction'))
        if 'spatial_min_component_size' in _m3:
            val = _m3.get('spatial_min_component_size')
            _m3_spatial_min_component_size = int(val) if isinstance(val, (int, float)) else None
        if isinstance(_m3.get('spatial_compactness'), (int, float)):
            _m3_spatial_compactness = float(_m3.get('spatial_compactness'))
        if isinstance(_m3.get('target_sites_per_metaspot'), (int, float)):
            _m3_target_sites_per_metaspot = int(_m3.get('target_sites_per_metaspot'))
        if isinstance(_m3.get('min_metaspots_per_cell'), (int, float)):
            _m3_min_metaspots_per_cell = int(_m3.get('min_metaspots_per_cell'))
        if isinstance(_m3.get('max_metaspots_per_cell'), (int, float)):
            _m3_max_metaspots_per_cell = int(_m3.get('max_metaspots_per_cell'))
    except Exception:
        pass
    import time as _time
    _t_m3 = _time.perf_counter()
    _m3_recomputed = False
    if _mod_lt('m3', start_mod) and _cache_chain_valid:
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module3_point_df")
        if pd_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 3 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 3 cache (module3_point_df)")
            except Exception:
                pass
            point_df = pd_cache
        else:
            if progress_callback: progress_callback("Module 3 · MetaSpot Segmentation")
            point_df = per_cell_clustering(
                point_df,
                gmm_cols,
                composition_model=composition_model,
                show_internal_progress=show_internal_progress,
                hdbscan_min_cluster_size=_m3_hdb_min,
                hdbscan_allow_single_cluster=_m3_allow_single,
                hdbscan_min_samples_frac=_m3_min_samples_frac,
                target_sub_size=_m3_target_sub_size,
                n_feat_hdb=_m3_n_feat_hdb,
                kmeans_min_k=_m3_kmeans_min_k,
                kmeans_max_k=_m3_kmeans_max_k,
                use_k_cap=_m3_use_k_cap,
                merge_min_size=_m3_merge_min,
                hdbscan_max_points=_m3_hdbscan_max_points,
                kmeans_train_size=_m3_kmeans_train_size,
                kmeans_batch_size=_m3_kmeans_batch_size,
                small_backend_cells=_m3_small_backend_cells,
                sequential_cells=_m3_sequential_cells,
                enforce_spatial_connectivity=_m3_enforce_spatial_connectivity,
                spatial_connectivity_k=_m3_spatial_connectivity_k,
                spatial_radius_quantile=_m3_spatial_radius_quantile,
                spatial_radius_factor=_m3_spatial_radius_factor,
                spatial_smoothing_iterations=_m3_spatial_smoothing_iterations,
                spatial_smoothing_min_fraction=_m3_spatial_smoothing_min_fraction,
                spatial_min_component_size=_m3_spatial_min_component_size,
                spatial_compactness=_m3_spatial_compactness,
                target_sites_per_metaspot=_m3_target_sites_per_metaspot,
                min_metaspots_per_cell=_m3_min_metaspots_per_cell,
                max_metaspots_per_cell=_m3_max_metaspots_per_cell,
            )
            _save_dataframe(point_df, out_dir, "module3_point_df")
            _m3_recomputed = True
    else:
        if progress_callback: progress_callback("Module 3 · MetaSpot Segmentation")
        point_df = per_cell_clustering(
            point_df,
            gmm_cols,
            composition_model=composition_model,
            show_internal_progress=show_internal_progress,
            hdbscan_min_cluster_size=_m3_hdb_min,
            hdbscan_allow_single_cluster=_m3_allow_single,
            hdbscan_min_samples_frac=_m3_min_samples_frac,
            target_sub_size=_m3_target_sub_size,
            n_feat_hdb=_m3_n_feat_hdb,
            kmeans_min_k=_m3_kmeans_min_k,
            kmeans_max_k=_m3_kmeans_max_k,
            use_k_cap=_m3_use_k_cap,
            merge_min_size=_m3_merge_min,
            hdbscan_max_points=_m3_hdbscan_max_points,
            kmeans_train_size=_m3_kmeans_train_size,
            kmeans_batch_size=_m3_kmeans_batch_size,
            small_backend_cells=_m3_small_backend_cells,
            sequential_cells=_m3_sequential_cells,
            enforce_spatial_connectivity=_m3_enforce_spatial_connectivity,
            spatial_connectivity_k=_m3_spatial_connectivity_k,
            spatial_radius_quantile=_m3_spatial_radius_quantile,
            spatial_radius_factor=_m3_spatial_radius_factor,
            spatial_smoothing_iterations=_m3_spatial_smoothing_iterations,
            spatial_smoothing_min_fraction=_m3_spatial_smoothing_min_fraction,
            spatial_min_component_size=_m3_spatial_min_component_size,
            spatial_compactness=_m3_spatial_compactness,
            target_sites_per_metaspot=_m3_target_sites_per_metaspot,
            min_metaspots_per_cell=_m3_min_metaspots_per_cell,
            max_metaspots_per_cell=_m3_max_metaspots_per_cell,
        )
        _save_dataframe(point_df, out_dir, "module3_point_df")
        _m3_recomputed = True
    if _m3_recomputed:
        _cache_chain_valid = False
    try:
        _cells = int(point_df['cell_id'].nunique()) if 'cell_id' in point_df.columns else None
    except Exception:
        _cells = None
    _dt_m3 = _time.perf_counter() - _t_m3
    if show_internal_progress:
        msg = f"<< Done: Module 3 (MetaSpot segmentation) in {_dt_m3:0.2f}s" + (f" (cells={_cells})" if _cells is not None else "")
        print(msg)
    try:
        import logging as _logging
        _logging.getLogger("cellscope").info(
            "Module 3 done in %.2fs%s",
            _dt_m3,
            (f" (cells={_cells})" if _cells is not None else ""),
        )
    except Exception:
        pass
    if progress_callback:
        try:
            progress_callback("Module 3 · Done")
        except Exception:
            pass
    # Save histogram of metaspots per cell
    try:
        import matplotlib.pyplot as plt
        vc = point_df.groupby('cell_id')[METASPOT_LOCAL_COL].nunique()
        fig = plt.figure(figsize=(6,4))
        ax = vc.plot.hist(bins=50)
        ax.set_xlabel("n_metaspots per cell"); ax.set_ylabel("count"); plt.tight_layout()
        _save_fig(fig, out_dir or os.getcwd(), "module3_metaspots_per_cell_hist")
        plt.close(fig)
    except Exception:
        pass
    # Module 4: adata1 from point_df (derived; rerun if m1–m3 changed)
    _m4_recomputed = False
    if _mod_lt('m4', start_mod) and _cache_chain_valid:
        a1_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module4_adata1.h5ad"))
        if a1_cache is not None:
            aligned, alignment_reason = _validate_adata1_point_df_alignment(
                a1_cache,
                point_df,
            )
            if not aligned:
                _logging.getLogger("cellscope").warning(
                    "Rejecting Module 4 cache: %s",
                    alignment_reason,
                )
                a1_cache = None
        if a1_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 4 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 4 cache (module4_adata1.h5ad)")
            except Exception:
                pass
            adata1 = a1_cache
        else:
            if progress_callback: progress_callback("Module 4 · Build Per-cell Feature Table (adata1)")
            adata1 = build_adata1(point_df, show_internal_progress=show_internal_progress)
            _save_adata(adata1, out_dir, "module4_adata1")
            _m4_recomputed = True
    else:
        if progress_callback: progress_callback("Module 4 · Build Per-cell Feature Table (adata1)")
        adata1 = build_adata1(point_df, show_internal_progress=show_internal_progress)
        _save_adata(adata1, out_dir, "module4_adata1")
        _m4_recomputed = True
    aligned, alignment_reason = _validate_adata1_point_df_alignment(
        adata1,
        point_df,
    )
    if not aligned:
        raise RuntimeError(
            "Module 4 MetaSpot alignment check failed: "
            f"{alignment_reason}. Refusing to continue with mismatched "
            "expression and geometry."
        )
    _logging.getLogger("cellscope").info(
        "Module 4 MetaSpot alignment validated (%d MetaSpots)",
        adata1.n_obs,
    )
    if _m4_recomputed:
        _cache_chain_valid = False
    try:
        import logging as _logging
        _suf = None if (_first_run and _m4_recomputed) else ("recomputed" if _m4_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 4 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m4_recomputed) else ("recomputed" if _m4_recomputed else "cache")
            progress_callback("Module 4 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass
    # Module 5: meta-domain clustering (allow resume)
    _m5_recomputed = False
    if _mod_lt('m5', start_mod) and _cache_chain_valid:
        a1_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module5_adata1.h5ad"))
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module5_point_df")
        if a1_cache is not None and pd_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 5 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 5 cache (module5_adata1.h5ad, module5_point_df)")
            except Exception:
                pass
            adata1 = a1_cache
            point_df = pd_cache
            diag_simple = None
        else:
            if progress_callback: progress_callback("Module 5 · Meta-domain Clustering")
            adata1, diag_simple = run_module5_simple(
                adata1,
                point_df,
                speed_preset="ultra",
                use_spatial_feats=True,
                use_decell=True,
                use_cosine=True,
                enable_cap=True,
                show_internal_progress=show_internal_progress,
                progress_callback=progress_callback,
            )
            _m5_recomputed = True
    else:
        if progress_callback: progress_callback("Module 5 · Meta-domain Clustering")
        adata1, diag_simple = run_module5_simple(
            adata1,
            point_df,
            speed_preset="ultra",
            use_spatial_feats=True,
            use_decell=True,
            use_cosine=True,
            enable_cap=True,
            show_internal_progress=show_internal_progress,
            progress_callback=progress_callback,
        )
        _m5_recomputed = True
    if _m5_recomputed:
        _cache_chain_valid = False
    # Stats: transcripts count per meta-domain
    try:
        vc = point_df.groupby("meta-domain")["transcript_id"].count()
        stats = vc.describe().to_frame(name="value")
        _save_dataframe(stats.reset_index(), out_dir or os.getcwd(), "module5_domain_transcript_stats")
    except Exception:
        pass
    # Expose the canonical MetaSpot identifier for downstream joins.
    if METASPOT_COL not in adata1.obs.columns:
        adata1.obs = adata1.obs.copy()
        adata1.obs[METASPOT_COL] = adata1.obs.index.astype(str)
    # Attach meta-domain to point_df
    if "meta-domain" in point_df.columns:
        point_df = point_df.drop(columns=["meta-domain"])  # ensure fresh
    point_df = point_df.merge(
        adata1.obs[[METASPOT_COL, "meta-domain"]],
        on=METASPOT_COL,
        how="left",
    )
    try:
        _save_dataframe(point_df, out_dir, "module5_point_df")
    except Exception:
        pass
    try:
        import logging as _logging
        _suf = None if (_first_run and _m5_recomputed) else ("recomputed" if _m5_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 5 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m5_recomputed) else ("recomputed" if _m5_recomputed else "cache")
            progress_callback("Module 5 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass

    # Module 6: Build adata2 from adata1 + meta-domains
    # Module 6: adata2 (allow resume)
    _m6_recomputed = False
    if _mod_lt('m6', start_mod) and _cache_chain_valid:
        a2_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module6_adata2.h5ad"))
        if a2_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 6 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 6 cache (module6_adata2.h5ad)")
            except Exception:
                pass
            adata2 = a2_cache
        else:
            if progress_callback: progress_callback("Module 6 · Aggregate Domains into Cluster Feature Table (adata2)")
            adata2 = build_adata2_from_adata1_and_pointdf(
                adata1,
                point_df,
                show_internal_progress=show_internal_progress,
                progress_callback=progress_callback,
            )
            _m6_recomputed = True
    else:
        if progress_callback: progress_callback("Module 6 · Aggregate Domains into Cluster Feature Table (adata2)")
        adata2 = build_adata2_from_adata1_and_pointdf(
            adata1,
            point_df,
            show_internal_progress=show_internal_progress,
            progress_callback=progress_callback,
        )
        _m6_recomputed = True
    if _m6_recomputed:
        _cache_chain_valid = False
    try:
        _cfg = load_params_yaml()
        save_shape = bool((_cfg.get('module6', {}) if _cfg else {}).get('save_shape_summary', True))
        if save_shape:
            # log basic adata2 shape
            df = pd.DataFrame({"n_obs":[adata2.n_obs],"n_vars":[adata2.n_vars]})
            _save_dataframe(df, out_dir or os.getcwd(), "module6_adata2_shape")
    except Exception:
        pass
    try:
        import logging as _logging
        _suf = None if (_first_run and _m6_recomputed) else ("recomputed" if _m6_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 6 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m6_recomputed) else ("recomputed" if _m6_recomputed else "cache")
            progress_callback("Module 6 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass

    # Module 7: Graph construction on adata2 (weighted edges, optional same-cell downweight)
    # Module 7: Graph Construction (allow resume)
    _m7_recomputed = False
    if _mod_lt('m7', start_mod) and _cache_chain_valid:
        a2_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module7_adata2.h5ad"))
        if a2_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 7 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 7 cache (module7_adata2.h5ad)")
            except Exception:
                pass
            adata2 = a2_cache
        else:
            if progress_callback: progress_callback("Module 7 · Graph Construction")
            try:
                adata2 = build_module7_graph(
                    adata2,
                    show_internal_progress=show_internal_progress,
                    scvi_epochs=scvi_epochs,
                    out_dir=out_dir,
                    progress_callback=progress_callback,
                )
                try:
                    _save_adata(adata2, out_dir, "module7_adata2")
                except Exception:
                    pass
                _m7_recomputed = True
            except Exception as e:
                print(f"[WARN] Module 7 failed: {e}")
            else:
                pass
    else:
        if progress_callback: progress_callback("Module 7 · Graph Construction")
        try:
            adata2 = build_module7_graph(
                adata2,
                show_internal_progress=show_internal_progress,
                scvi_epochs=scvi_epochs,
                out_dir=out_dir,
                progress_callback=progress_callback,
            )
            try:
                _save_adata(adata2, out_dir, "module7_adata2")
            except Exception:
                pass
            _m7_recomputed = True
        except Exception as e:
            print(f"[WARN] Module 7 failed: {e}")
        else:
            pass
    if _m7_recomputed:
        _cache_chain_valid = False
    try:
        import logging as _logging
        _suf = None if (_first_run and _m7_recomputed) else ("recomputed" if _m7_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 7 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m7_recomputed) else ("recomputed" if _m7_recomputed else "cache")
            progress_callback("Module 7 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass
    # NOTE: single logging/progress_callback above is sufficient; avoid duplicate messages.

    # Module 8: Train DGI (GCN/SAGE) and backfill embeddings to adata2/adata1
    # Module 8: DGI embeddings (allow resume)
    _m8_recomputed = False
    if _mod_lt('m8', start_mod) and _cache_chain_valid:
        a1_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module8_adata1.h5ad"))
        a2_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module8_adata2.h5ad"))
        if a1_cache is not None and a2_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 8 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 8 cache (module8_adata1.h5ad, module8_adata2.h5ad)")
            except Exception:
                pass
            adata1, adata2 = a1_cache, a2_cache
        else:
            if progress_callback: progress_callback("Module 8 · Graph Embeddings (DGI)")
            try:
                adata1, adata2 = run_module8_dgi(
                    adata1,
                    adata2,
                    show_internal_progress=show_internal_progress,
                    dgi_epochs=dgi_epochs,
                    enable_dgi_sage=enable_dgi_sage,
                    out_dir=out_dir,
                    progress_callback=progress_callback,
                )
                try:
                    _save_adata(adata1, out_dir, "module8_adata1")
                    _save_adata(adata2, out_dir, "module8_adata2")
                except Exception:
                    pass
                _m8_recomputed = True
            except Exception as e:
                print(f"[WARN] Module 8 failed: {e}")
            else:
                pass
    else:
        if progress_callback: progress_callback("Module 8 · Graph Embeddings (DGI)")
        try:
            adata1, adata2 = run_module8_dgi(
                adata1,
                adata2,
                show_internal_progress=show_internal_progress,
                dgi_epochs=dgi_epochs,
                enable_dgi_sage=enable_dgi_sage,
                out_dir=out_dir,
                progress_callback=progress_callback,
            )
            try:
                _save_adata(adata1, out_dir, "module8_adata1")
                _save_adata(adata2, out_dir, "module8_adata2")
            except Exception:
                pass
            _m8_recomputed = True
        except Exception as e:
            print(f"[WARN] Module 8 failed: {e}")
        else:
            pass
    if _m8_recomputed:
        _cache_chain_valid = False
    try:
        import logging as _logging
        _suf = None if (_first_run and _m8_recomputed) else ("recomputed" if _m8_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 8 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m8_recomputed) else ("recomputed" if _m8_recomputed else "cache")
            progress_callback("Module 8 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass

    final_csv = point_df.copy()
    return adata1, adata2, final_csv


def _checkpoint_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, f".{name}.parquet")

# Global I/O switches for intermediates, set by run_pipeline and/or params.yaml
_WRITE_INTERMEDIATE: Optional[bool] = None
_LITE_INTERMEDIATE: Optional[bool] = None
_WRITE_FIGURES: Optional[bool] = None

def _io_should_write_intermediate() -> bool:
    """Decide whether to write intermediate artifacts.

    Priority:
      1) explicit global override set by run_pipeline
      2) io.write_intermediate (if present in params.yaml)
      3) io.keep_intermediate (legacy semantics; False disables writing)
      4) default True
    """
    global _WRITE_INTERMEDIATE
    if _WRITE_INTERMEDIATE is not None:
        return bool(_WRITE_INTERMEDIATE)
    try:
        _cfg = load_params_yaml()
        io_cfg = _cfg.get('io', {}) if _cfg else {}
        if 'write_intermediate' in io_cfg:
            return bool(io_cfg.get('write_intermediate'))
        if 'keep_intermediate' in io_cfg:
            return bool(io_cfg.get('keep_intermediate'))
    except Exception:
        pass
    return True

def _io_should_write_figures() -> bool:
    """Decide whether to write figures (PNG).

    Priority:
      1) explicit global override set by run_pipeline
      2) io.write_intermediate_figures (params.yaml)
      3) annotation.save_plots (best-effort if present)
      4) default True
    Note: figures are allowed even if other intermediates are disabled.
    """
    global _WRITE_FIGURES
    if _WRITE_FIGURES is not None:
        return bool(_WRITE_FIGURES)
    allow = True
    try:
        _cfg = load_params_yaml()
        io_cfg = _cfg.get('io', {}) if _cfg else {}
        if 'write_intermediate_figures' in io_cfg:
            allow = bool(io_cfg.get('write_intermediate_figures'))
        # fall back to annotation.save_plots if provided
        ann_cfg = _cfg.get('annotation', {}) if _cfg else {}
        if 'save_plots' in ann_cfg:
            allow = allow and bool(ann_cfg.get('save_plots'))
    except Exception:
        pass
    return allow

def _intermediate_dir(out_dir: str) -> str:
    """Return the path to the intermediate directory, configurable via params.yaml.

    io.intermediate_dirname: optional folder name (default 'intermediate').
    """
    try:
        _cfg = load_params_yaml()
        dname = (_cfg.get('io', {}) if _cfg else {}).get('intermediate_dirname', 'intermediate')
        dname = str(dname).strip() or 'intermediate'
    except Exception:
        dname = 'intermediate'
    p = os.path.join(out_dir, dname)
    os.makedirs(p, exist_ok=True)
    return p

def _save_dataframe(df: pd.DataFrame, out_dir: str, name: str):
    """Save a dataframe as CSV/Parquet according to io flags in params.yaml."""
    # Respect global/io switch: do nothing if disabled
    if not _io_should_write_intermediate():
        return
    base = _intermediate_dir(out_dir)
    write_csv = True
    write_pq = True
    try:
        _cfg = load_params_yaml()
        io_cfg = _cfg.get('io', {}) if _cfg else {}
        write_csv = bool(io_cfg.get('write_intermediate_csv', True))
        write_pq = bool(io_cfg.get('write_intermediate_parquet', True))
    except Exception:
        pass
    if not write_csv and not write_pq:
        _logging.getLogger("cellscope").debug(
            "Intermediate table disabled by I/O config: %s",
            name,
        )
        return
    _logging.getLogger("cellscope").info(
        "Saving intermediate table: %s",
        name,
    )
    written = []
    if write_csv:
        csv_path = os.path.join(base, f"{name}.csv")
        try:
            df.to_csv(csv_path, index=False)
            written.append(csv_path)
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "Failed to save intermediate CSV %s: %s",
                csv_path,
                exc,
            )
    if write_pq:
        try:
            pq_path = os.path.join(base, f"{name}.parquet")
            df.to_parquet(pq_path, index=False)
            written.append(pq_path)
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "Failed to save intermediate Parquet %s: %s",
                pq_path,
                exc,
            )
    if written:
        _logging.getLogger("cellscope").info(
            "Saved intermediate table: %s",
            ", ".join(written),
        )

def _preferred_n_jobs() -> int:
    """Return preferred parallel worker count.

    Env vars (checked in order):
      CELLSCOPE_N_JOBS         -> explicit override
      CELLSCOP3E_N_JOBS        -> (legacy typo) still honored

    params.yaml (runtime.n_jobs) is honored when env vars are absent.

    Fallback heuristic: use ~1/3 of available CPUs, cap <=32 (legacy behavior).
    """
    for k in ("CELLSCOPE_N_JOBS", "CELLSCOP3E_N_JOBS"):
        try:
            v_raw = os.environ.get(k, "").strip()
            if v_raw:
                v = int(v_raw)
                if v > 0:
                    return v
        except Exception:
            pass
    try:
        from .config import load_params_yaml
        cfg = load_params_yaml() or {}
        runtime_cfg = cfg.get("runtime", {}) or {}
        v = runtime_cfg.get("n_jobs")
        v_int = _coerce_pos_int(v)
        if v_int:
            return v_int
    except Exception:
        pass
    try:
        import multiprocessing
        cpu = max(1, multiprocessing.cpu_count())
    except Exception:
        cpu = 4
    # Default: use roughly half of available CPUs, capped at 32
    return max(1, min(32, cpu // 2 if cpu >= 2 else 1))

def _joblib_backend() -> str:
    # Default backend: 'loky' for maximum throughput on CPU-bound tasks.
    # Users can override via CELLSCOPE_JOBLIB_BACKEND env or runtime.joblib_backend.
    env_backend = os.environ.get("CELLSCOPE_JOBLIB_BACKEND")
    if env_backend:
        return env_backend
    try:
        from .config import load_params_yaml
        cfg = load_params_yaml() or {}
        runtime_cfg = cfg.get("runtime", {}) or {}
        rt_backend = runtime_cfg.get("joblib_backend")
        if isinstance(rt_backend, str) and rt_backend.strip():
            return rt_backend.strip()
    except Exception:
        pass
    return "loky"

def _cluster_chunk_size() -> int:
    """Chunk size for per-cell parallel loops.

    Env: CELLSCOPE_CLUSTER_CHUNK (int)
    Config: module2.parallel_chunk_cells
    Smaller chunks reduce peak memory (fewer simultaneous process payloads)
    at the cost of higher scheduler overhead.
    """
    try:
        env_v = os.environ.get("CELLSCOPE_CLUSTER_CHUNK", "").strip()
        if env_v:
            return max(16, min(1024, int(env_v)))
    except Exception:
        pass
    try:
        from .config import load_params_yaml
        cfg = load_params_yaml() or {}
        m2 = cfg.get("module2", {}) or {}
        cfg_v = m2.get("parallel_chunk_cells")
        if cfg_v is not None:
            return max(16, min(1024, int(cfg_v)))
    except Exception:
        pass
    try:
        v = int(os.environ.get("CELLSCOPE_CLUSTER_CHUNK", "512"))
        return max(128, min(1024, v))
    except Exception:
        return 128


def _state_path(out_dir: str) -> str:
    return os.path.join(out_dir, "pipeline_state.json")

def _load_state(out_dir: str) -> Optional[dict]:
    try:
        import json
        p = _state_path(out_dir)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _save_state(out_dir: str, last_completed: str, extra: Optional[dict] = None) -> None:
    try:
        import json, time
        state = {"last_completed": last_completed, "ts": time.time()}
        if extra:
            state.update(extra)
        with open(_state_path(out_dir), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _sanitize_obs_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    """Normalize obs values unsupported by older AnnData/HDF5 writers."""
    for col in adata.obs.columns:
        series = adata.obs[col]
        if (
            pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or series.dtype == object
        ):
            series = series.astype(object)
            series = series.where(pd.notnull(series), "")
            series = series.apply(
                lambda value: ",".join(map(str, value))
                if isinstance(value, (list, tuple))
                else str(value)
            )
            adata.obs[col] = series.astype(object)
    return adata


def _save_adata(adata: ad.AnnData, out_dir: str, name: str, compression: str = "lzf"):
    if not _io_should_write_intermediate():
        return
    tmp_path = None
    original_obs = adata.obs
    try:
        p = os.path.join(_intermediate_dir(out_dir), f"{name}.h5ad")
        tmp_path = f"{p}.tmp-{os.getpid()}"
        _logging.getLogger("cellscope").info(
            "Saving intermediate AnnData: %s",
            p,
        )
        # Avoid mutating the in-memory schema merely for writer compatibility.
        adata.obs = original_obs.copy()
        _sanitize_obs_for_h5ad(adata)
        try:
            import anndata as _ad
            _ad.settings.allow_write_nullable_strings = True
        except Exception:
            pass
        adata.write_h5ad(tmp_path, compression=compression)
        os.replace(tmp_path, p)
        tmp_path = None
        _logging.getLogger("cellscope").info(
            "Saved intermediate AnnData: %s",
            p,
        )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Failed to save intermediate AnnData %s: %s",
            name,
            exc,
            exc_info=True,
        )
    finally:
        adata.obs = original_obs
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def _save_fig(fig, out_dir: str, name: str):
    if not _io_should_write_figures():
        return
    try:
        p = os.path.join(_intermediate_dir(out_dir), f"{name}.png")
        fig.savefig(p, bbox_inches='tight', dpi=150)
    except Exception:
        pass


def run_pipeline(
    spatial_path: str,
    cell_boundaries_path: str,
    nucleus_boundaries_path: str,
    out_dir: str,
    gold_data: Optional[Union[str, pd.DataFrame]] = None,
    resume: bool = True,
    save_intermediate: bool = True,
    lite_intermediate: bool = False,
    final_format: Literal["parquet","csv"] = "csv",
    compression: str = "lzf",
    resume_from: Optional[Literal[
        "load_inputs",
        "prepare_adata",
        "final_table",
        "write_outputs",
    ]] = None,
    show_internal_progress: bool = False,
    enable_annotation: bool = True,
    annotation_method: str = "rapx",
    scvi_epochs: int = 100,
    dgi_epochs: int = 200,
    enable_dgi_sage: bool = False,
    resume_policy: Literal["auto","minimal","force"] = "minimal",
    dry_run_diff: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
    """
    Execute the CellScoper pipeline.

    Returns dict with output file paths.
    """
    ensure_dir(out_dir)
    logger = setup_logger(out_dir)
    logger.info("CellScoper pipeline start")
    # Load config early to allow overriding some runtime I/O defaults
    cfg = load_params_yaml()
    write_csv_copy = True
    try:
        io_cfg = cfg.get('io', {}) if cfg else {}
        # Allow YAML to override defaults (CLI flags still take precedence if supplied explicitly by caller)
        if isinstance(io_cfg.get('final_format'), str):
            final_format = str(io_cfg.get('final_format')).lower()
        if isinstance(io_cfg.get('h5ad_compression'), str):
            compression = str(io_cfg.get('h5ad_compression'))
        if 'write_csv_copy' in io_cfg:
            write_csv_copy = bool(io_cfg.get('write_csv_copy'))
        if 'lite_intermediate' in io_cfg:
            lite_intermediate = bool(io_cfg.get('lite_intermediate'))
        # New: allow disabling intermediate persistence via YAML
        # Priority: write_intermediate (if present) > keep_intermediate (legacy)
        if 'write_intermediate' in io_cfg:
            save_intermediate = bool(io_cfg.get('write_intermediate'))
        elif 'keep_intermediate' in io_cfg:
            ki = bool(io_cfg.get('keep_intermediate'))
            # when keep_intermediate is False, we disable saving intermediates
            if not ki:
                save_intermediate = False
        # Allow YAML resume_from
        if cfg and isinstance(io_cfg.get('resume_from'), (str, type(None))):
            rf = (io_cfg.get('resume_from') or '').strip()
            resume_from = rf if rf else resume_from
        # Optional resume policy from YAML (io.resume_policy)
        if isinstance(io_cfg.get('resume_policy'), str):
            resume_policy = str(io_cfg.get('resume_policy')).lower()
    except Exception:
        pass
    try:
        m7_cfg = cfg.get('module7', {}) if cfg else {}
        m8_cfg = cfg.get('module8', {}) if cfg else {}
        if isinstance(m7_cfg.get('scvi_epochs'), (int, float)):
            scvi_epochs = int(m7_cfg.get('scvi_epochs'))
        if isinstance(m8_cfg.get('dgi_epochs'), (int, float)):
            dgi_epochs = int(m8_cfg.get('dgi_epochs'))
        if 'enable_dgi_sage' in m8_cfg:
            enable_dgi_sage = bool(m8_cfg.get('enable_dgi_sage'))
    except Exception:
        pass
    try:
        runtime_cfg = cfg.get('runtime', {}) if cfg else {}
    except Exception:
        runtime_cfg = {}
    # Annotation toggles/method from YAML (default on, method=rapx)
    try:
        ann_cfg = cfg.get('annotation', {}) if cfg else {}
        if 'enable' in ann_cfg:
            enable_annotation = bool(ann_cfg.get('enable'))
        if 'method' in ann_cfg:
            _m = str(ann_cfg.get('method') or "").strip().lower()
            if _m:
                annotation_method = _m
    except Exception:
        pass
    # Set global I/O switches for helpers
    try:
        global _WRITE_INTERMEDIATE, _LITE_INTERMEDIATE, _WRITE_FIGURES
        _WRITE_INTERMEDIATE = bool(save_intermediate)
        _LITE_INTERMEDIATE = bool(lite_intermediate)
        # Figures default to True unless explicitly disabled
        if 'write_intermediate_figures' in (io_cfg or {}):
            _WRITE_FIGURES = bool(io_cfg.get('write_intermediate_figures'))
        else:
            _WRITE_FIGURES = True
        # Respect annotation.save_plots if present
        try:
            ann_cfg = cfg.get('annotation', {}) if cfg else {}
            if 'save_plots' in ann_cfg and _WRITE_FIGURES is not None:
                _WRITE_FIGURES = _WRITE_FIGURES and bool(ann_cfg.get('save_plots'))
        except Exception:
            pass
    except Exception:
        pass
    # Cap BLAS/OMP threads; allow runtime.blas_threads in params.yaml to take effect
    try:
        _rt_blas_threads = _coerce_pos_int(runtime_cfg.get('blas_threads')) if runtime_cfg else None
        if _rt_blas_threads:
            for _env_k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_MAX_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[_env_k] = str(_rt_blas_threads)
            if runtime_cfg.get('use_threadpoolctl'):
                try:
                    from threadpoolctl import threadpool_limits
                    threadpool_limits(int(_rt_blas_threads))
                except Exception:
                    pass
        else:
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
            os.environ.setdefault("MKL_NUM_THREADS",     "1")
            os.environ.setdefault("OMP_NUM_THREADS",     "1")
            os.environ.setdefault("NUMEXPR_MAX_THREADS", "64")
            os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")
        # Silence JAX GPU reminder by forcing CPU backend
        os.environ.setdefault("JAX_PLATFORM_NAME",    "cpu")
        os.environ.setdefault("JAX_PLATFORMS",        "cpu")
        os.environ.setdefault(
                                "PYTHONWARNINGS",
                                "ignore:.*'force_all_finite'.*'ensure_all_finite'.*:FutureWarning:sklearn.utils.deprecation"
                            )
    except Exception:
        pass
    # Suppress warnings globally to avoid noisy output; only errors should surface
    try:
        import warnings as _warnings
        _warnings.simplefilter("ignore")
        _warnings.filterwarnings("ignore", category=FutureWarning)
        _warnings.filterwarnings("ignore", category=UserWarning)
        _warnings.filterwarnings("ignore", category=RuntimeWarning)
        _warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Keep specific modules quiet as well
        _warnings.filterwarnings("ignore", category=FutureWarning, module=r"anndata.*")
        _warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn.utils.deprecation")
        # Filter common 3rd-party noisy warnings
        _warnings.filterwarnings("ignore", message=".*does not have many workers.*", category=UserWarning)
        _warnings.filterwarnings("ignore", message=".*Tensor Cores.*set `torch.set_float32_matmul_precision`.*", category=UserWarning)
    except Exception:
        pass
    # Downgrade Lightning/PL logs to ERROR to avoid hardware banners
    try:
        import logging as _logging
        for _name in ("lightning", "lightning.pytorch", "pytorch_lightning"):
            try:
                _logging.getLogger(_name).setLevel(_logging.ERROR)
            except Exception:
                pass
    except Exception:
        pass

    # Determine minimal rerun start using config fingerprints
    stage_order = ["load_inputs", "preprocess", "build_anndata", "annotate", "write_outputs"]
    start_from = None
    cur_fps = {s: fingerprint_stage(cfg, s) for s in stage_order}
    try:
        cur_mod_fps = fingerprint_modules_set(cfg)
    except Exception:
        cur_mod_fps = {}
    st = _load_state(out_dir) or {}
    prev_fps = st.get("fingerprints") or {}
    # default: if explicit resume_from provided, honor it
    if resume_from:
        legacy_map = {"prepare_adata": "build_anndata", "final_table": "annotate"}
        start_from = legacy_map.get(resume_from, resume_from)
        logger.info("Resume requested from %s", start_from)
    else:
        # Choose behavior based on resume_policy
        policy = str(resume_policy or "minimal").lower()
        if policy == "force":
            start_from = "load_inputs"
            logger.info("Resume policy=force; starting from %s", start_from)
        elif policy == "auto":
            # legacy: continue from last completed + 1 if available
            if st.get("last_completed") in stage_order:
                last = st["last_completed"]
                try:
                    idx = stage_order.index(last)
                    start_from = stage_order[min(idx + 1, len(stage_order) - 1)]
                except Exception:
                    start_from = None
            if start_from:
                logger.info("Resume policy=auto; continuing from %s (last completed: %s)", start_from, st.get("last_completed"))
            else:
                logger.info("Resume policy=auto; starting from %s", stage_order[0])
        else:  # minimal (diff-based)
            start_from = None
            for s in stage_order:
                if prev_fps.get(s) != cur_fps.get(s):
                    start_from = s
                    break
            if start_from:
                logger.info("Resume policy=minimal; config change → start from %s", start_from)
            else:
                # fall back to last_completed if exists
                if st.get("last_completed") in stage_order:
                    last = st["last_completed"]
                    try:
                        idx = stage_order.index(last)
                        start_from = stage_order[min(idx + 1, len(stage_order) - 1)]
                    except Exception:
                        start_from = None
                if start_from:
                    logger.info("Resume policy=minimal; auto-resume from %s (last completed: %s)", start_from, st.get("last_completed"))

    # Dry-run: show diff and planned start, then return immediately
    if dry_run_diff:
        try:
            changes = [s for s in stage_order if prev_fps.get(s) != cur_fps.get(s)]
        except Exception:
            changes = []
        logger.info("Dry-run diff: changed stages=%s; start_from=%s", ",".join(changes) if changes else "(none)", start_from or stage_order[0])
        return {
            "dry_run": True,
            "changed_stages": changes,
            "start_from": start_from or stage_order[0],
            "state_file": os.path.join(out_dir, "pipeline_state.json"),
        }

    stage_to_idx = {name: idx for idx, name in enumerate(stage_order)}
    resume_enabled = bool(resume)
    if resume_enabled:
        resume_stage_key = start_from if isinstance(start_from, str) else None
        if resume_stage_key and resume_stage_key not in stage_to_idx:
            logger.warning("Unknown resume stage '%s'; default to %s", resume_stage_key, stage_order[0])
            resume_stage_key = None
        start_idx = stage_to_idx.get(resume_stage_key, 0) if resume_stage_key else 0
        logger.info("Effective resume start stage: %s", stage_order[start_idx])
        # Nudge CLI progress early so users don't see a long 'Starting pipeline' while loading caches
        try:
            if progress_callback:
                progress_callback(f"Resume from {stage_order[start_idx]}")
        except Exception:
            pass
    else:
        start_idx = 0

    def _should_run(stage_name: str, has_cache: bool) -> bool:
        if not resume_enabled:
            return True
        idx = stage_to_idx[stage_name]
        if idx >= start_idx:
            return True
        return not has_cache

    # ==== Stage 1: Load inputs ====
    spatial_df = cell_df = nucleus_df = None
    adata1 = adata2 = None
    final_data = None

    # If resuming past load_inputs, we still need to load cached inputs; update progress text first.
    if resume_enabled and stage_to_idx["load_inputs"] < start_idx and progress_callback:
        try:
            progress_callback("Loading cached inputs")
        except Exception:
            pass
    raw_inputs_cache = _resume_raw_inputs(out_dir) if (resume_enabled and stage_to_idx["load_inputs"] < start_idx) else None
    if _should_run("load_inputs", raw_inputs_cache is not None):
        if resume_enabled and stage_to_idx["load_inputs"] < start_idx and raw_inputs_cache is None:
            logger.info("Resume requested skip for load_inputs but checkpoint missing → re-running")
        if progress_callback: progress_callback("Loading inputs")
        try:
            spatial_df, cell_df, nucleus_df = load_inputs(
                spatial_path, cell_boundaries_path, nucleus_boundaries_path, out_dir, resume, save_intermediate
            )
            raw_inputs_cache = (spatial_df, cell_df, nucleus_df)
            _save_state(out_dir, "load_inputs", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps})
            logger.info("Stage load_inputs done:spatial=%d rows, cells_boundaries=%d, nucleus_boundaries=%d", len(spatial_df), len(cell_df), len(nucleus_df))
        except Exception:
            logger.exception("Stage load_inputs failed")
            raise
    else:
        if raw_inputs_cache is None:
            raise RuntimeError("Missing 'load_inputs' checkpoint; cannot skip this stage.")
        spatial_df, cell_df, nucleus_df = raw_inputs_cache
        logger.info("Resume: skipping load_inputs (checkpoint available)")

    # ==== Stage 2: Preprocess ====
    # If resuming past preprocess, load cached preprocessed data; update progress text first.
    if resume_enabled and stage_to_idx["preprocess"] < start_idx and progress_callback:
        try:
            progress_callback("Loading cached preprocessed inputs")
        except Exception:
            pass
    preproc_cache = _resume_preprocessed_inputs(out_dir) if (resume_enabled and stage_to_idx["preprocess"] < start_idx) else None
    if _should_run("preprocess", preproc_cache is not None):
        if resume_enabled and stage_to_idx["preprocess"] < start_idx and preproc_cache is None:
            logger.info("Resume requested skip for preprocess but checkpoint missing → re-running")
        if progress_callback: progress_callback("Preprocess inputs")
        try:
            spatial_df, cell_df, nucleus_df = preprocess(spatial_df, cell_df, nucleus_df, show_internal_progress=show_internal_progress)
            if save_intermediate and not lite_intermediate:
                _save_dataframe(spatial_df, out_dir, "preprocess_spatial")
                _save_dataframe(cell_df, out_dir, "preprocess_cell_boundaries")
                _save_dataframe(nucleus_df, out_dir, "preprocess_nucleus_boundaries")
            try:
                save_checkpoint(spatial_df, _checkpoint_path(out_dir, "spatial_preproc"))
                save_checkpoint(cell_df, _checkpoint_path(out_dir, "cell_preproc"))
                save_checkpoint(nucleus_df, _checkpoint_path(out_dir, "nucleus_preproc"))
            except Exception:
                pass
            _save_state(out_dir, "preprocess", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps})
            logger.info("Stage preprocess done: spatial=%d rows, cells_boundaries=%d, nucleus_boundaries=%d", len(spatial_df), len(cell_df), len(nucleus_df))
        except Exception:
            logger.exception("Stage preprocess failed")
            raise
    else:
        if preproc_cache is None:
            raise RuntimeError("Missing 'preprocess' checkpoint; cannot skip this stage.")
        spatial_df, cell_df, nucleus_df = preproc_cache
        logger.info("Resume: skipping preprocess (checkpoint available)")

    # ==== Stage 3: Build AnnData + final table ====
    build_cache = _resume_build_outputs(out_dir) if (resume_enabled and stage_to_idx["build_anndata"] < start_idx) else None
    build_cache_ready = bool(build_cache and build_cache[0] is not None and build_cache[2] is not None)
    if _should_run("build_anndata", build_cache_ready):
        if resume_enabled and stage_to_idx["build_anndata"] < start_idx and not build_cache_ready:
            logger.info("Resume requested skip for build_anndata but checkpoint missing → re-running")
        if progress_callback: progress_callback("Core pipeline modules")
        try:
            adata1, adata2, final_data = build_anndata(
                spatial_df,
                cell_df,
                nucleus_df,
                show_internal_progress=show_internal_progress,
                progress_callback=progress_callback,
                out_dir=out_dir,
                scvi_epochs=scvi_epochs,
                dgi_epochs=dgi_epochs,
                enable_dgi_sage=enable_dgi_sage,
            )
            # Optional: prune final_data columns to reduce redundancy per config
            try:
                _cfg = load_params_yaml()
                _io_cfg = (_cfg.get('io', {}) if _cfg else {})
            except Exception:
                _io_cfg = {}
            drop_embeddings = bool(_io_cfg.get('drop_embeddings', False))
            keep_cols = _io_cfg.get('final_keep_columns', None)
            if isinstance(keep_cols, list) and len(keep_cols) > 0:
                existing = [c for c in keep_cols if c in final_data.columns]
                if len(existing) > 0:
                    final_data = final_data.loc[:, existing]
            elif drop_embeddings:
                # Drop large embedding columns when requested
                drop_prefixes = ("comp_emb", "gmm_feat_", "X_", "Z_", "umap_")
                final_data = final_data[[c for c in final_data.columns if not any(str(c).startswith(p) for p in drop_prefixes)]]
            persist_build = (save_intermediate and not lite_intermediate) or resume_enabled
            if persist_build:
                try:
                    _cfg = load_params_yaml()
                    comp = str((_cfg.get('io', {}) if _cfg else {}).get('h5ad_compression', compression))
                    m6_override = bool(
                        (_cfg.get('module6', {}) if _cfg else {}).get(
                            'h5ad_compression_override',
                            True,
                        )
                    )
                except Exception:
                    comp = compression
                    m6_override = True
                _save_adata(adata1, out_dir, "module5_adata1", compression=comp)
                if adata2 is not None:
                    _save_adata(
                        adata2,
                        out_dir,
                        "module6_adata2",
                        compression=comp if m6_override else "lzf",
                    )
            if resume_enabled or (save_intermediate and not lite_intermediate):
                _save_dataframe(final_data, out_dir, "module5_final_data")
            _save_state(out_dir, "build_anndata", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps, "modules_completed": True})
            logger.info("Stage build_anndata done: adata1_obs=%d adata2_obs=%s final_rows=%d", adata1.n_obs, (adata2.n_obs if adata2 is not None else None), len(final_data))
        except Exception:
            logger.exception("Stage build_anndata failed")
            raise
    else:
        logger.info("Resume: skipping build_anndata (checkpoint available)")
        if build_cache is None:
            raise RuntimeError("Missing 'build_anndata' checkpoint; cannot skip this stage.")
        adata1, adata2, cached_final = build_cache
        if cached_final is None:
            raise RuntimeError("Missing 'module5_final_data' checkpoint; cannot skip 'build_anndata' stage.")
        final_data = cached_final

    # ==== Stage 4: Annotation (optional/configurable) ====
    ann_method = (annotation_method or "rapx").lower()
    run_annotation = bool(enable_annotation) and (ann_method != "none") and adata2 is not None
    annotation_cache = None
    annotation_cache_ready = False
    if run_annotation and resume_enabled and stage_to_idx["annotate"] < start_idx:
        annotation_cache = _resume_annotation_outputs(out_dir)
        annotation_cache_ready = bool(annotation_cache and annotation_cache[1] is not None)
    if run_annotation and _should_run("annotate", annotation_cache_ready):
        if resume_enabled and stage_to_idx["annotate"] < start_idx and not annotation_cache_ready:
            logger.info("Resume requested skip for annotate but checkpoint missing → re-running")
        if progress_callback: progress_callback("Annotation (RAP-X)")
        try:
            try:
                _cfg = load_params_yaml()
                rapx_two_stage = bool(_cfg.get('annotation', {}).get('two_stage', False))
                rapx_spatial_channel = bool(_cfg.get('annotation', {}).get('spatial_channel', False))
            except Exception:
                rapx_two_stage = False
                rapx_spatial_channel = False
            adata1, adata2, final_data = _maybe_cluster_and_annotate(
                adata1, adata2, final_data,
                show_internal_progress=show_internal_progress,
                out_dir=out_dir,
                annotation_method=ann_method,
                rapx_two_stage=rapx_two_stage,
                rapx_spatial_channel=rapx_spatial_channel,
                gold_data=gold_data,
            )
            persist_annotation = (save_intermediate and not lite_intermediate) or resume_enabled
            if persist_annotation:
                if adata2 is not None:
                    _save_adata(adata2, out_dir, "annotated_adata2")
                _save_dataframe(final_data, out_dir, "annotated_final_csv")
            _save_state(out_dir, "annotate", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps})
            logger.info("Stage annotate done: final_rows=%d", len(final_data))
        except Exception as e:
            logger.warning("Annotation skipped: %s", e, exc_info=True)
    elif run_annotation and annotation_cache_ready and annotation_cache is not None:
        logger.info("Resume: skipping annotate (checkpoint available)")
        ann_adata2, ann_final = annotation_cache
        if ann_adata2 is not None:
            adata2 = ann_adata2
        final_data = ann_final

    if adata1 is None or final_data is None:
        raise RuntimeError("Cannot write outputs: missing 'adata1' or 'final_data'. Please check prior stages or checkpoints.")
    adata1 = cast(ad.AnnData, adata1)
    final_data = cast(pd.DataFrame, final_data)

    # Step 3: Write outputs
    adata1_path = os.path.join(out_dir, "adata1.h5ad")
    adata2_path = os.path.join(out_dir, "adata2.h5ad")
    final_parquet_path = os.path.join(out_dir, "final_data.parquet")
    final_csv_path = os.path.join(out_dir, "final_data.csv")

    # Optionally resume directly to writing outputs
    if resume_from == "write_outputs":
        pass

    # Write files

    if progress_callback: progress_callback("Writing outputs")
    try:
        # Compatibility for anndata<0.11 + pandas StringArray/categorical writes.
        # First sanitize obs columns, then opt-in to nullable-string writing when available.
        adata1 = _sanitize_obs_for_h5ad(adata1)
        try:
            import anndata as _ad
            _ad.settings.allow_write_nullable_strings = True
        except Exception:
            pass
        adata1.write(adata1_path, compression=compression)
        if adata2 is not None:
            adata2 = _sanitize_obs_for_h5ad(adata2)
            adata2.write(adata2_path, compression=compression)
        if final_format == "csv":
            final_data.to_csv(final_csv_path, index=False)
            # also write parquet copy silently
            try:
                final_data.to_parquet(final_parquet_path, index=False)
            except Exception:
                pass
        else:  # parquet chosen
            try:
                final_data.to_parquet(final_parquet_path, index=False)
                if write_csv_copy:
                    final_data.to_csv(final_csv_path, index=False)
            except Exception:
                final_data.to_csv(final_csv_path, index=False)
        _save_state(out_dir, "write_outputs", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps})
        logger.info("Stage write_outputs done")
    except Exception:
        logger.exception("Stage write_outputs failed")
        raise

    logger.info("Pipeline finished successfully")
    return {
        "adata1": adata1_path,
        "adata2": adata2_path if adata2 is not None else None,
        "final_parquet": final_parquet_path if final_format == "parquet" else None,
        "final_csv": final_csv_path if final_format == "csv" or write_csv_copy else None,
    }


# ====== Optional K1/K2: Clustering + RNALocate/APEX-based annotation ======

def _try_import_scanpy():
    try:
        import scanpy as sc  # noqa: F401
        return True
    except Exception:
        return False


def _choose_annotation_neighbors_key(
    adata2: ad.AnnData,
    cluster_rep: Optional[str],
    rep_priority: Optional[Sequence[str]],
) -> str:
    """Resolve the annotation graph without overriding an explicit choice."""
    if rep_priority is None:
        rep_priority = ["X_dgi_gcn", "X_dgi_sage", "X_expr", "X_pca"]

    def _neighbors_key(rep_name: str) -> str:
        rep = str(rep_name).strip()
        if rep == "X_pca":
            return "pca_nbrs"
        if rep == "X_dgi_gcn":
            return "gcn_nbrs"
        if rep == "X_dgi_sage":
            return "sage_nbrs"
        return f"{rep[2:]}_nbrs" if rep.startswith("X_") else rep

    if cluster_rep and str(cluster_rep).strip().lower() != "auto":
        explicit = _neighbors_key(str(cluster_rep))
        if f"{explicit}_connectivities" in adata2.obsp:
            return explicit
        _logging.getLogger("cellscope").warning(
            "Requested annotation.cluster_rep=%s has no neighbor graph; "
            "falling back to rep_priority.",
            cluster_rep,
        )

    for rep in rep_priority:
        candidate = _neighbors_key(str(rep))
        if f"{candidate}_connectivities" in adata2.obsp:
            return candidate
    return "pca_nbrs"


def _normalize_and_cluster_adata2(
    adata2: ad.AnnData,
    rep_priority: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    n_neighbors: int = 32,
    leiden_resolution: float = 1.0,
    hvg_n_top_genes: Optional[int] = None,
    pca_n_comps: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    umap_spread: Optional[float] = None,
    umap_neighbors_key: Optional[str] = None,
    cluster_rep: Optional[str] = None,
) -> ad.AnnData:
    """Normalize counts/log1p, run HVG+PCA+neighbors, and Leiden on preferred embeddings.
    Sets `adata2.obs['domain_type']` and aliases to `adata2.obs['leiden']` for downstream RAP-X.
    Requires scanpy; no-op if scanpy missing.
    """
    if not _try_import_scanpy():
        print("[WARN] scanpy not available; skip normalization+clustering.")
        return adata2
    import numpy as _np
    import scanpy as sc
    from scipy import sparse as _sparse

    # Preserve raw counts.  Normalization must happen in a separate layer:
    # mutating ``layers['counts']`` silently breaks SCVI and provenance.
    if 'counts' not in adata2.layers:
        if _is_count_like_matrix(adata2.X):
            adata2.layers['counts'] = adata2.X.copy()
        else:
            _logging.getLogger("cellscope").warning(
                "adata2 has no count-like counts layer; PCA will use the "
                "current compositional expression matrix."
            )
    counts_mat = adata2.layers.get('counts', adata2.X)
    if not _sparse.issparse(counts_mat):
        counts_mat = _sparse.csr_matrix(counts_mat)
    cell_sums = counts_mat.sum(axis=1).A1
    idx_keep = _np.where(cell_sums > 0)[0]
    if idx_keep.size < adata2.n_obs:
        adata2 = adata2[idx_keep].copy()
        counts_mat = counts_mat[idx_keep, :]
        if 'counts' in adata2.layers:
            adata2.layers['counts'] = counts_mat.copy()

    adata2.layers['log1p'] = counts_mat.copy()
    sc.pp.normalize_total(adata2, target_sum=1e4, layer='log1p', inplace=True)
    sc.pp.log1p(adata2, layer='log1p')
    adata2.X = adata2.layers['log1p']
    adata2.raw = adata2

    # PCA/Neighbors on log1p
    _hvg_top = int(hvg_n_top_genes) if (hvg_n_top_genes is not None) else min(2000, max(200, adata2.n_vars))
    sc.pp.highly_variable_genes(adata2, flavor='seurat', n_top_genes=_hvg_top)
    # Use mask_var to avoid deprecation warning
    _pca_comps = int(pca_n_comps) if (pca_n_comps is not None) else int(min(50, adata2.n_vars, adata2.n_obs))
    try:
        sc.pp.pca(adata2, mask_var="highly_variable", n_comps=_pca_comps)
    except Exception:
        sc.pp.pca(adata2, use_highly_variable=True, n_comps=_pca_comps)
    sc.pp.neighbors(adata2, use_rep='X_pca', key_added='pca_nbrs', n_neighbors=int(n_neighbors))

    # Also build neighbors on available learned reps if present
    reps = ['X_expr', 'X_dgi_sage', 'X_dgi_gcn']
    for rep in reps:
        if rep in adata2.obsm:
            try:
                # Explicit key names aligned with notebook expectations
                if rep == 'X_dgi_gcn':
                    key_std = 'gcn_nbrs'
                elif rep == 'X_dgi_sage':
                    key_std = 'sage_nbrs'
                elif rep == 'X_expr':
                    key_std = 'expr_nbrs'
                else:
                    key_std = f'{rep[2:]}_nbrs'
                sc.pp.neighbors(adata2, use_rep=rep, key_added=key_std, n_neighbors=int(n_neighbors))
            except Exception:
                pass

    # Pick priority for Leiden
    # Prefer explicit cluster_rep from config if available and corresponding neighbors exist
    try:
        if cluster_rep is None:
            _cfg = load_params_yaml()
            cluster_rep = _cfg.get('annotation', {}).get('cluster_rep', None)
    except Exception:
        pass
    chosen_key = _choose_annotation_neighbors_key(
        adata2,
        cluster_rep=cluster_rep,
        rep_priority=rep_priority,
    )
    # Future-proof: igraph flavor
    try:
        try:
            sc.tl.leiden(adata2, neighbors_key=chosen_key, key_added='domain_type', resolution=float(leiden_resolution), flavor='igraph', n_iterations=2, directed=False)
        except Exception as e:
            _logging.getLogger("cellscope").warning(f"Leiden(igraph) failed: {e}; falling back to 'leidenalg' flavor.")
            try:
                sc.tl.leiden(adata2, neighbors_key=chosen_key, key_added='domain_type', resolution=float(leiden_resolution), flavor='leidenalg', n_iterations=2, directed=False)
            except Exception as e2:
                _logging.getLogger("cellscope").error(f"Leiden fallback failed: {e2}; skipping clustering and setting 'domain_type' to 'unknown'.")
                adata2.obs['domain_type'] = 'unknown'
    except Exception:
        try:
            sc.tl.leiden(adata2, neighbors_key=chosen_key, key_added='domain_type', resolution=float(leiden_resolution))
        except Exception as e:
            _logging.getLogger("cellscope").warning(f"Leiden(default) failed: {e}; trying 'leidenalg' flavor.")
            try:
                sc.tl.leiden(adata2, neighbors_key=chosen_key, key_added='domain_type', resolution=float(leiden_resolution), flavor='leidenalg')
            except Exception as e2:
                _logging.getLogger("cellscope").error(f"Leiden fallback failed: {e2}; skipping clustering and setting 'domain_type' to 'unknown'.")
                adata2.obs['domain_type'] = 'unknown'
    # alias for RAP-X
    adata2.obs['leiden'] = adata2.obs['domain_type'].astype(str)
    # Optional UMAP visualization saved to intermediate (use chosen neighbors)
    try:
        # Read annotation plotting switches from config; default off for speed
        _cfg = load_params_yaml()
        ann = _cfg.get('annotation', {}) if _cfg else {}
        _compute_umap = bool(ann.get('compute_umap', False))
        _save_umap_plot = bool(ann.get('save_umap_plot', False))
        if _compute_umap or _save_umap_plot:
            import scanpy as sc
            import matplotlib.pyplot as plt
            _nbrs_for_umap = umap_neighbors_key if umap_neighbors_key else chosen_key
            if _compute_umap:
                # Build kwargs dynamically to avoid signature lint issues across scanpy versions
                _umap_kwargs = {'neighbors_key': _nbrs_for_umap}
                if umap_min_dist is not None:
                    _umap_kwargs['min_dist'] = float(umap_min_dist)
                if umap_spread is not None:
                    _umap_kwargs['spread'] = float(umap_spread)
                # Call with keyword-only to maximize compatibility
                sc.tl.umap(**dict(adata=adata2, **_umap_kwargs))
            if _save_umap_plot and ('X_umap' in adata2.obsm):
                fig = sc.pl.umap(adata2, color=['leiden'], return_fig=True, wspace=0.4)
                _save_fig(fig, out_dir or os.getcwd(), "annotation_umap_leiden")
                plt.close(fig)
    except Exception:
        pass
    return adata2


# --- RNALocate L1 mapping and projection to 7 compartments ---
import re as _re

SEVEN_KEYS   = ['NUCLEUS','NUCLEOLUS','LAMINA','NUCLEAR_PORE','CYTOSOL','ER','OMM']
SEVEN_PRETTY = {'NUCLEUS':'Nucleus','NUCLEOLUS':'Nucleolus','LAMINA':'Lamina',
                'NUCLEAR_PORE':'Nuclear_Pore','CYTOSOL':'Cytosol','ER':'ER','OMM':'OMM'}
SEVEN_COLS   = [SEVEN_PRETTY[k] for k in SEVEN_KEYS]

L1 = [
    'NUCLEUS','NUC_LEM','NUC_PORE','NUCLEOLUS','CYTOSOL','RNP_GRANULE','RIBOSOME_CYTO',
    'ER','GOLGI','ENDOLYSO','PEROXISOME','MITO_OMM','MITO_IMM','MITO_MATRIX','MITO_UNSPEC',
    'PM_JUNCTION','CYTO_CORTEX_POLARITY','NEURONAL','CELL_DIVISION','EV','RIBOSOME_ORGANELLAR','OTHER',
]

_MULTI_SEP = _re.compile(r'[;,/|]+')

def _norm_loc_txt(s: str) -> str:
    return _re.sub(r'\s+', ' ', str(s).strip().lower())

# Explicit mapping (subset from user-provided list; extended as needed)
L1_EXPLICIT = {
    'nucleolus': 'NUCLEOLUS',
    'nucleus(exclusion from nucleoli)': 'NUCLEUS', 'nucleus': 'NUCLEUS', 'nuclear ': 'NUCLEUS', 'nucleoplasm': 'NUCLEUS',
    'nuclear envelope': 'NUC_LEM', 'nuclear inner membrane': 'NUC_LEM', 'nuclear outer membrane': 'NUC_LEM', 'nuclear periphery': 'NUC_LEM',
    'nuclear pore complex': 'NUC_PORE', 'nuclear pore': 'NUC_PORE', 'npc': 'NUC_PORE',
    'cytosol': 'CYTOSOL', 'cytoplasm': 'CYTOSOL', 'cytosolic': 'CYTOSOL', 'perinuclear ': 'CYTOSOL',
    'stress granule': 'RNP_GRANULE', 'p-body': 'RNP_GRANULE',
    'cytosolic ribosome': 'RIBOSOME_CYTO', 'ribosome': 'RIBOSOME_CYTO',
    'endoplasmic reticulum': 'ER', 'er lumen': 'ER', 'er membrane': 'ER', 'rough endoplasmic reticulum': 'ER',
    'golgi apparatus': 'GOLGI', 'endosome': 'ENDOLYSO', 'lysosome': 'ENDOLYSO', 'autophagosome': 'ENDOLYSO',
    'peroxisome': 'PEROXISOME',
    'mitochondrial outer membrane': 'MITO_OMM', 'outer mitochondrial membrane': 'MITO_OMM',
    'mitochondrial inner membrane': 'MITO_IMM', 'mitochondrion': 'MITO_UNSPEC',
    'plasma membrane': 'PM_JUNCTION', 'tight junction': 'PM_JUNCTION', 'cell junction': 'PM_JUNCTION',
    'cell cortex': 'CYTO_CORTEX_POLARITY', 'lamellipodium': 'CYTO_CORTEX_POLARITY', 'pseudopodium': 'CYTO_CORTEX_POLARITY',
    'axon': 'NEURONAL', 'dendrite': 'NEURONAL', 'synapse': 'NEURONAL',
    'centrosome': 'CELL_DIVISION', 'spindle': 'CELL_DIVISION',
    'extracellular vesicle': 'EV', 'extracellular exosome': 'EV',
    'vesicle': 'OTHER', 'membrane': 'OTHER',
}

L1_REGEX = [
    ('NUCLEOLUS',     r'\bnucleol\w+'),
    ('NUC_LEM',       r'\bnuclear (envelope|lamina)\b|\binner nuclear membrane\b|\bouter nuclear membrane\b|\bnuclear periphery\b'),
    ('NUC_PORE',      r'\bnuclear pore( complex)?\b|\bNPC\b'),
    ('NUCLEUS',       r'\bnucleus\b|\bnuclear(?! (envelope|lamina|pore))\b|\bchromatin\b'),
    ('CYTOSOL',       r'\bcytosol(ic)?\b|\bcytoplasm(ic)?\b|\bperinuclear(?! (envelope|membrane))\b'),
    ('RNP_GRANULE',   r'\bstress granule\b|\bP-?body\b|\bRNP\b'),
    ('ER',            r'\b(endoplasmic reticulum|ER)\b|\ber lumen\b|\ber membrane\b'),
    ('MITO_OMM',      r'\bouter (mitochondrial|mitochondrion) membrane\b|\bmitochondrial outer membrane\b'),
    ('MITO_IMM',      r'\bmitochondrial inner membrane\b'),
    ('MITO_UNSPEC',   r'\bmitochondri(on|a)l? (?!inner|outer)\b|\bmitochondrion\b'),
    ('PM_JUNCTION',   r'\bplasma membrane\b|\b(cell )?junction\b|\btight junction\b'),
    ('CYTO_CORTEX_POLARITY', r'\bcortex\b|\bapical\b|\bbasal\b|\blamellipod\w+|\bpseudopod\w+|\bprotrusion\b'),
    ('NEURONAL',      r'\baxon\w*|\bdendrit\w*|\bsynaps\w*'),
    ('ENDOLYSO',      r'\bendosome\b|\blysosome\b|\bautophagosome\b'),
    ('EV',            r'\bextracellular (vesicle|exosome)\b|\b(apoptotic|micro|nano)vesicle\b'),
    ('CELL_DIVISION', r'\bcentrosome\b|\bspindle\b|\bmitotic apparatus\b'),
    ('PEROXISOME',    r'\bperoxisome\b'),
    ('GOLGI',         r'\bgolgi\b'),
]

def map_loc_to_l1(loc: str) -> str:
    if not isinstance(loc, str) or not loc.strip():
        return 'OTHER'
    s = _norm_loc_txt(loc)
    for key in sorted(L1_EXPLICIT.keys(), key=len, reverse=True):
        if key and key in s:
            return L1_EXPLICIT[key]
    for l1, patt in L1_REGEX:
        if _re.search(patt, s, flags=_re.IGNORECASE):
            return l1
    return 'OTHER'


def build_gold_from_rnalocate_l1(rna_locate_df: pd.DataFrame,
                                 species=("Homo sapiens","Mus musculus"),
                                 rna_types=("mRNA",),
                                 score_col='RNALocate_Score',
                                 agg_mode='prob_or',
                                 score_floor=0.5,
                                 use_pubmed_weight=True) -> pd.DataFrame:
    need = {'Species','RNA_Symbol','RNA_Type','Subcellular_Localization',score_col}
    miss = need - set(rna_locate_df.columns)
    if miss:
        raise ValueError(f"RNALocate is missing required columns: {miss}")
    df = rna_locate_df.copy()
    if species is not None:
        df = df[df['Species'].isin(species)]
    if rna_types is not None:
        df = df[df['RNA_Type'].isin(rna_types)]
    def _clean_symbol(x):
        if not isinstance(x, str): return None
        x = x.strip(); return x.upper() if x else None
    df['Common_Gene'] = df['RNA_Symbol'].map(_clean_symbol)
    df = df[df['Common_Gene'].notna()]
    df = df.assign(Subcellular_Localization=df['Subcellular_Localization'].astype(str))
    df = df.assign(_loc_split=df['Subcellular_Localization'].map(lambda s: [t.strip() for t in _MULTI_SEP.split(s) if t.strip()]))
    df = df.explode('_loc_split', ignore_index=True)
    df['L1'] = df['_loc_split'].map(map_loc_to_l1)
    df['score'] = pd.to_numeric(df[score_col], errors='coerce').fillna(0.0).clip(0,1)
    df = df[df['score'] >= float(score_floor)]
    if use_pubmed_weight and 'PubMed_ID' in df.columns:
        pm = df['PubMed_ID'].notna() & (df['PubMed_ID'].astype(str).str.strip()!='')
        df.loc[pm,'score'] = (df.loc[pm,'score'] + 0.05).clip(0,1)
    if agg_mode == 'prob_or':
        agg = df.groupby(['Common_Gene','L1'])['score'].apply(lambda s: 1.0 - np.prod(1.0 - s.values))
    elif agg_mode == 'max':
        agg = df.groupby(['Common_Gene','L1'])['score'].max()
    else:
        raise ValueError("'agg_mode' supports only 'prob_or' or 'max'.")
    tab = agg.unstack('L1').reindex(columns=L1).fillna(0.0)
    tab.index.name = 'Common_Gene'
    return tab


def project_l1_to_seven(l1_df: pd.DataFrame, mode='strict', mito_non_omm=None, fold_misc_to_cytosol=True) -> pd.DataFrame:
    def _get(col):
        return pd.to_numeric(l1_df.get(col, pd.Series(0.0, index=l1_df.index)), errors='coerce').fillna(0.0).astype(float)
    out = pd.DataFrame(index=l1_df.index, columns=SEVEN_COLS, dtype=float)
    out['Nucleus']       = _get('NUCLEUS')
    out['Nucleolus']     = _get('NUCLEOLUS')
    out['Lamina']        = _get('NUC_LEM')
    out['Nuclear_Pore']  = _get('NUC_PORE')
    cyt_base = _get('CYTOSOL') + _get('RNP_GRANULE') + _get('RIBOSOME_CYTO')
    if mode == 'relaxed' and fold_misc_to_cytosol:
        cyt_base = (cyt_base + _get('GOLGI') + _get('ENDOLYSO') + _get('PEROXISOME') + _get('PM_JUNCTION') + _get('CYTO_CORTEX_POLARITY') + _get('CELL_DIVISION'))
    out['Cytosol'] = np.clip(cyt_base, 0.0, 1.0)
    out['ER'] = _get('ER')
    omm = _get('MITO_OMM')
    if mode == 'relaxed' and mito_non_omm in ('OMM','Cytosol'):
        add_mito = _get('MITO_IMM') + _get('MITO_MATRIX') + _get('MITO_UNSPEC')
        if mito_non_omm == 'OMM':
            omm = 1.0 - (1.0 - omm) * (1.0 - add_mito)
        elif mito_non_omm == 'Cytosol':
            out['Cytosol'] = 1.0 - (1.0 - out['Cytosol']) * (1.0 - add_mito)
    out['OMM'] = np.clip(omm, 0.0, 1.0)
    out = out.fillna(0.0); out.index.name = 'Common_Gene'
    return out


def _dedup_columns_by(df: pd.DataFrame, how='prob_or'):
    if not df.columns.duplicated().any():
        return df
    X = df.T
    if how == 'prob_or':
        def por(x):
            x = pd.to_numeric(x, errors='coerce').fillna(0.0).astype(float)
            return 1.0 - np.prod(1.0 - x.values)
        agg = X.groupby(level=0, sort=False).agg(por)
    elif how == 'max':
        agg = X.groupby(level=0, sort=False).max()
    else:
        raise ValueError("'how' supports only 'prob_or' or 'max'.")
    return agg.T


def _dedup_index_by(df: pd.DataFrame, how='prob_or'):
    idx_up = df.index.astype(str).str.upper()
    X = df.copy(); X.index = idx_up
    if not idx_up.duplicated().any():
        return X
    if how == 'prob_or':
        def por(x):
            x = pd.to_numeric(x, errors='coerce').fillna(0.0).astype(float)
            return 1.0 - np.prod(1.0 - x.values)
        return X.groupby(level=0, sort=False).agg(por)
    elif how == 'max':
        return X.groupby(level=0, sort=False).max()
    else:
        raise ValueError("'how' supports only 'prob_or' or 'max'.")


def merge_gold(existing_gold_df: pd.DataFrame, new_gold_df: pd.DataFrame, fuse='max'):
    col_alias = {
        'NUCLEUS':'Nucleus','NUCLEOLUS':'Nucleolus','LAMINA':'Lamina',
        'NUCLEAR_PORE':'Nuclear_Pore','CYTOSOL':'Cytosol','ER':'ER','OMM':'OMM',
        'ERM':'ER','ER_LUMEN':'ER','ER_Lumen':'ER','ER Membrane':'ER','ER membrane':'ER'
    }
    agg_how = 'prob_or' if fuse=='prob_or' else 'max'
    def _prep(df):
        X = df.copy()
        if 'Common_Gene' in X.columns and X.index.name != 'Common_Gene':
            X = X.set_index('Common_Gene')
        X.columns = pd.Index([str(c).strip() for c in X.columns])
        X = X.rename(columns=col_alias)
        if X.columns.duplicated().any():
            X = _dedup_columns_by(X, how=agg_how)
        keep = [c for c in X.columns if c in SEVEN_COLS]
        X = X[keep]
        for c in SEVEN_COLS:
            if c not in X.columns:
                X[c] = 0.0
        X = X.reindex(columns=SEVEN_COLS, copy=False)
        X = _dedup_index_by(X, how=agg_how)
        X = X.apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0.0).astype(float)
        assert X.columns.is_unique, "Column names are still not unique. Please check upstream data or alias mapping."
        return X
    A = _prep(existing_gold_df)
    B = _prep(new_gold_df)
    all_idx = A.index.union(B.index)
    A = A.reindex(all_idx).fillna(0.0)
    B = B.reindex(all_idx).fillna(0.0)
    if fuse == 'prob_or':
        out = 1.0 - (1.0 - A) * (1.0 - B)
    elif fuse == 'max':
        out = np.maximum(A, B)
    else:
        raise ValueError("'fuse' supports only 'max' or 'prob_or'.")
    out.index.name = 'Common_Gene'
    return out


def _load_manual_marker_data(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load optional manual marker table and normalize row-wise to 0-1.

    Expected columns: `Common_Gene` + region columns (Nucleus/Nucleolus/.../OMM).
    If `Common_Gene` absent but index is the gene names, the index will be reset.
    Missing file or parse errors return None (non-fatal).
    """
    if path is None:
        path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'src', 'manual_marker.csv'))
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_table(path)
        except Exception:
            return None

    if 'Common_Gene' not in df.columns and df.index.name == 'Common_Gene':
        df = df.reset_index()
    if 'Common_Gene' not in df.columns:
        return None

    val_cols = [c for c in df.columns if c != 'Common_Gene']
    if len(val_cols) == 0:
        return None

    X = df[val_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    X = X.subtract(X.min(axis=1), axis=0)
    span = (X.max(axis=1) - X.min(axis=1)).replace(0, 1.0)
    X = X.div(span, axis=0)
    out = pd.concat([df[['Common_Gene']].reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    # Drop all-zero rows to avoid inflating sets with null markers.
    try:
        mask_nonzero = (X.sum(axis=1) > 0)
        out = out.loc[mask_nonzero].reset_index(drop=True)
    except Exception:
        pass
    return out


# ---------- 目标区域与别名 ----------
TARGET_REGIONS = ['NUCLEUS','NUCLEOLUS','LAMINA','NUCLEAR_PORE','CYTOSOL','ER','OMM']
GOLD_ALIAS = {
    'NUCLEUS':'NUCLEUS','NUCLEOLUS':'NUCLEOLUS','LAMINA':'LAMINA',
    'NUCLEAR_PORE':'NUCLEAR_PORE','NUCLEARPORE':'NUCLEAR_PORE',
    'CYTOSOL':'CYTOSOL','CYTOSOLIC':'CYTOSOL',
    'ER':'ER','ERM':'ER','ER_LUMEN':'ER','ER_LUMINAL':'ER',
    'OMM':'OMM','MITOCHONDRIA_OMM':'OMM',
}
PRETTY = {'NUCLEUS':'Nucleus','NUCLEOLUS':'Nucleolus','LAMINA':'Lamina',
          'NUCLEAR_PORE':'Nuclear Pore','CYTOSOL':'Cytosol','ER':'ER','OMM':'OMM'}
NUCLEAR_LABELS = frozenset({'Nucleus', 'Nucleolus', 'Lamina', 'Nuclear Pore'})
CYTOPLASMIC_LABELS = frozenset(
    {'Cytosol', 'ER', 'OMM', 'ER-OMM shared'}
)

# 可视化用的顺序与颜色
PRETTY_ORDER = ['Cytosol','ER','Lamina','Nuclear Pore','Nucleolus','Nucleus','OMM']
SIG_COLOR    = {
    '***': '#d62728',
    '**': '#ff7f0e',
    '*': '#2ca02c',
    'ns': '#7f7f7f',
    'NA': '#bdbdbd',
}

# ---------- 参数 ----------
TARGET_SET_SIZE = 600
MIN_SET_SIZE    = 200
MAX_SET_SIZE    = 1000
MIN_TOP_VALUE   = 0.1
USE_MARGIN      = True

TOPK_OVERLAP_FRAC = 0.3
TOPK_OVERLAP_MIN  = 250

FDR_CUTOFF      = 0.05
Z_CUTOFF        = 1
TIE_DELTA       = 0.35
ALWAYS_ASSIGN   = False

PROMOTE_TOP2    = True
DELTA_PROMOTE   = 1
REL_PROMOTE     = 0.9

# 新增：第4个通道权重（空间）
CHANNEL_WEIGHTS = (1, 1, 1, 0.25)  # MWU, Proj, Hyp, soft geometry

# ---------- Gold 基因大小写自适应 ----------
def harmonize_gold_gene_case(gold_df: pd.DataFrame, target_var_names: Optional[pd.Index] = None) -> pd.DataFrame:
    """Normalize `Common_Gene` casing and pick the style with better overlap to target genes.

    Mouse数据常用首字母大写+其余小写的写法（例如 Actb），而 Gold 表往往是全大写。
    这里同时计算两种写法与目标 adata.var_names 的重叠度，选择更优者并去重聚合。
    """
    gd = gold_df.copy()

    # Ensure Common_Gene is a column for processing.
    if 'Common_Gene' not in gd.columns and gd.index.name == 'Common_Gene':
        gd = gd.reset_index()
    if 'Common_Gene' not in gd.columns:
        return gd

    genes = gd['Common_Gene'].astype(str)
    upper = genes.str.upper()
    mouse_style = genes.str.capitalize()

    chosen = upper
    if target_var_names is not None:
        tgt = pd.Index(target_var_names.astype(str))
        tgt_up = tgt.str.upper()
        tgt_set = set(tgt)
        tgt_up_set = set(tgt_up)

        # Score exact case matches for mouse-style, and uppercase matches for both.
        hit_upper = len(tgt_up_set.intersection(set(upper)))
        hit_mouse_case = len(tgt_set.intersection(set(mouse_style)))
        hit_mouse_upper = len(tgt_up_set.intersection(set(mouse_style.str.upper())))

        # Prefer mouse-style when it improves exact-case overlap; otherwise fall back to upper.
        prefer_mouse = hit_mouse_case > hit_upper
        # Tie-breaker: if target genes大多不是全大写，则倾向 mouse-style
        frac_target_mixed = float((tgt != tgt_up).mean()) if len(tgt) else 0.0
        if not prefer_mouse and hit_mouse_case == hit_upper and frac_target_mixed >= 0.2:
            prefer_mouse = True

        chosen = mouse_style if prefer_mouse else upper
        try:
            _logging.getLogger('cellscope').info(
                "Gold gene case harmonization: upper=%d, mouse_case=%d, mouse_upper=%d, target_mixed=%.2f, chosen=%s",
                hit_upper, hit_mouse_case, hit_mouse_upper, frac_target_mixed,
                'mouse-style' if prefer_mouse else 'upper'
            )
        except Exception:
            pass

    gd['Common_Gene'] = chosen

    # Deduplicate after recasing; aggregate by max to keep conservative probabilities.
    if gd['Common_Gene'].duplicated().any():
        num_cols = [c for c in gd.columns if c != 'Common_Gene']
        agg = gd.groupby('Common_Gene', sort=False)[num_cols].max()
        gd = agg.reset_index()

    return gd

# ---------- Gold 处理 ----------
def prepare_gold_matrix(
    gold_df: pd.DataFrame,
    normalize_columns: bool = True,
) -> pd.DataFrame:
    gd = gold_df.copy()
    if 'Common_Gene' in gd.columns and gd.index.name != 'Common_Gene':
        gd = gd.set_index('Common_Gene')
    gd.index = gd.index.astype(str).str.upper()
    num_gd = gd.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna(axis=0, how='all')
    if num_gd.shape[1] == 0:
        raise ValueError("gold_data 数值化后无可用列。")
    if normalize_columns:
        vals = num_gd.to_numpy(dtype=np.float32, copy=True)
        cmin = np.nanmin(vals, axis=0, keepdims=True)
        cmax = np.nanmax(vals, axis=0, keepdims=True)
        span = np.maximum(cmax - cmin, 1e-8)
        vals = (vals - cmin) / span
        num_gd = pd.DataFrame(
            vals,
            index=num_gd.index,
            columns=num_gd.columns,
        )
    num_gd = num_gd.groupby(num_gd.index).max()
    return num_gd


def _canon_region_col(colname: str) -> str:
    s = str(colname).strip().upper().replace(' ', '_')
    s = re.sub(r'(_LOG2FC)$', '', s)
    s = re.sub(r'(_DEORPHAN(ED)?)$', '', s)
    s = re.sub(r'(_ORPHAN)$', '', s)
    s = re.sub(r'__+', '_', s)
    return GOLD_ALIAS.get(s, s)


def build_adaptive_sets(gold_mat: pd.DataFrame):
    gm = gold_mat.copy()
    gm.columns = [_canon_region_col(c) for c in gm.columns]
    keep = [c for c in gm.columns if c in TARGET_REGIONS]
    if not keep:
        raise ValueError(f"清洗后无目标区域列；现有列={list(set(gm.columns))}")
    gm = gm[keep].groupby(level=0, axis=1).max()

    vals = gm.to_numpy(dtype=np.float32, copy=True)
    cmin = np.nanmin(vals, axis=0, keepdims=True)
    cmax = np.nanmax(vals, axis=0, keepdims=True)
    span = np.maximum(cmax - cmin, 1e-8)
    vals = (vals - cmin) / span
    gm = pd.DataFrame(vals, index=gm.index, columns=gm.columns)

    row_sum = gm.sum(1).replace(0, np.nan)
    gm_norm = gm.div(row_sum, axis=0).fillna(0.0)

    arr = gm_norm.to_numpy()
    top_idx = arr.argmax(axis=1)
    top_val = arr[np.arange(arr.shape[0]), top_idx]
    second_val = np.partition(arr, -2, axis=1)[:, -2] if arr.shape[1] >= 2 else np.zeros_like(top_val)
    margin = top_val - second_val

    genes = gm_norm.index.to_numpy()
    regions_raw = list(gm_norm.columns)

    sets, weights = {}, {}
    for r_i, r in enumerate(regions_raw):
        pool = np.where((top_idx == r_i) & (top_val >= MIN_TOP_VALUE))[0]
        if pool.size == 0:
            sets[r] = set()
            weights[r] = pd.Series(dtype=np.float32)
            continue
        score = top_val[pool] * (margin[pool] if USE_MARGIN else 1.0)
        order = np.argsort(-score, kind='mergesort')
        take = min(max(TARGET_SET_SIZE, MIN_SET_SIZE), min(pool.size, MAX_SET_SIZE))
        sel = pool[order[:take]]
        w = (top_val[sel] * margin[sel]).astype(np.float32)
        if w.size > 0:
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        sets[r] = set(genes[sel])
        weights[r] = pd.Series(w, index=genes[sel], name=r).sort_values(ascending=False)

    sets = {PRETTY.get(k, k): v for k, v in sets.items()}
    weights = {PRETTY.get(k, k): s for k, s in weights.items()}
    regions = list(sets.keys())
    return sets, weights, regions


# ---------- 表达汇总与稳健 Z ----------
def build_group_means(adata, gold_gene_index_upper):
    if 'leiden' not in adata.obs.columns:
        raise KeyError("adata.obs is missing 'leiden'. Please complete the clustering first and write the cluster labels into adata.obs['leiden'].")
    common_genes_up = adata.var_names.astype(str).str.upper()
    mask_common = common_genes_up.isin(gold_gene_index_upper)
    if mask_common.sum() == 0:
        raise RuntimeError("No intersection with gold data. Please check naming/mapping.")
    X = adata[:, mask_common].X
    leiden_cat = adata.obs['leiden'].astype('category')
    codes = leiden_cat.cat.codes.to_numpy()
    n_groups = int(codes.max()) + 1
    n_obs = adata.n_obs
    M = sp.csr_matrix((np.ones(n_obs, dtype=np.float32), (codes, np.arange(n_obs, dtype=np.int64))),
                      shape=(n_groups, n_obs), dtype=np.float32)
    group_sum = M @ (X if sp.issparse(X) else sp.csr_matrix(X))
    counts = np.maximum(np.bincount(codes, minlength=n_groups).astype(np.float32), 1e-8)
    group_mean = group_sum.multiply(1.0 / counts[:, None]).tocsr()
    cluster_ids = list(leiden_cat.cat.categories)
    genes_up = common_genes_up[mask_common].to_numpy()
    return group_mean, cluster_ids, genes_up


def robust_gene_stats(group_mean_csr):
    if sp.issparse(group_mean_csr):
        G = group_mean_csr.toarray().astype(np.float32)
    else:
        G = np.asarray(group_mean_csr, dtype=np.float32)
    med = np.median(G, axis=0, keepdims=True)
    mad = np.median(np.abs(G - med), axis=0, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(G, axis=0, keepdims=True)
    unstable = scale < 1e-6
    scale[unstable] = std[unstable]
    constant = scale < 1e-6
    scale[constant] = 1.0
    S = (G - med) / scale
    S[:, constant.ravel()] = 0.0
    S[~np.isfinite(S)] = 0.0
    return S


def normalize_subanno(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    s = s.str.replace(r"[^a-z]+", "", regex=True)
    return s


def _nuclear_geometry_thresholds(
    values: np.ndarray,
    quantile: float,
) -> Tuple[float, float]:
    """Return boundary-low and deep-high ``r_mean`` thresholds.

    Inside the nucleus, ``r_signed`` is ``-d_cell / (d_cell + d_nuc)``.
    Values are therefore most negative at the nuclear envelope and increase
    toward the nuclear interior.  Keep the direction explicit here because
    reversing it makes Lamina/Nuclear Pore priors favor deep nuclear domains.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 4:
        return np.nan, np.nan
    q = float(np.clip(quantile, 0.0, 0.5))
    boundary = float(np.quantile(finite, q))
    deep = float(np.quantile(finite, 1.0 - q))
    return boundary, deep


def _subtype_geometry_scores(
    regions: List[str],
    r_value: float,
    center: float,
    scale: float,
) -> np.ndarray:
    """Build a continuous geometry prior without overwriting expression calls."""
    scores = np.zeros(len(regions), dtype=np.float32)
    if not np.isfinite(r_value):
        return scores
    z = float(np.clip((r_value - center) / max(scale, 1e-6), -3.0, 3.0))
    for idx, region in enumerate(regions):
        if region in {"Lamina", "Nuclear Pore"}:
            scores[idx] = -z
        elif region == "Nucleolus":
            scores[idx] = z
        elif region == "Nucleus":
            scores[idx] = -0.35 * abs(z)
        elif region == "ER":
            scores[idx] = -0.5 * z
        else:
            scores[idx] = 0.0
    return scores


def _compartment_geometry_scores(
    regions: List[str],
    r_value: float,
    tau: float = 0.2,
    min_abs_r: float = 0.05,
) -> np.ndarray:
    """Return a continuous nuclear/cytoplasmic consistency score.

    Signed geometry uses ``r < 0`` for the nuclear side and ``r > 0`` for
    the cytoplasmic side.  The score is positive for labels on the matching
    side and negative for labels on the conflicting side.  A dead zone around
    zero avoids turning uncertain boundary geometry into a hard compartment
    decision, while ``tanh`` keeps the prior bounded.
    """
    scores = np.zeros(len(regions), dtype=np.float32)
    if not np.isfinite(r_value):
        return scores

    r_value = float(r_value)
    dead_zone = max(float(min_abs_r), 0.0)
    distance = abs(r_value) - dead_zone
    if distance <= 0.0:
        return scores

    tau = max(abs(float(tau)), 1e-6)
    signed_strength = float(
        np.copysign(np.tanh(distance / tau), r_value)
    )
    for idx, region in enumerate(regions):
        compartment = _label_compartment(str(region))
        if compartment == "Nuclear":
            scores[idx] = -signed_strength
        elif compartment == "Cytoplasmic":
            scores[idx] = signed_strength
    return scores


def _label_compartment(label: str) -> Optional[str]:
    if label in NUCLEAR_LABELS:
        return "Nuclear"
    if label in CYTOPLASMIC_LABELS:
        return "Cytoplasmic"
    return None


def _empirical_marker_set_calibration(
    score_matrix: np.ndarray,
    cluster_ids: Sequence[str],
    regions: Sequence[str],
    region_indices: Dict[str, np.ndarray],
    region_weights: Dict[str, np.ndarray],
    *,
    expression_matrix: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    permutations: int = 500,
    min_effective_markers: int = 10,
    seed: int = 0,
    matching_bins: int = 5,
) -> pd.DataFrame:
    """Calibrate evidence against assay- and marker-matched random sets.

    The null is label-specific. Random sets match the number of measured
    markers, marker-weight profile, detection-rate distribution, and mean
    expression distribution of the observed set. Sampling is restricted to
    the measured marker universe and excludes the label's own genes.
    Empirical p-values use the add-one rule; BH correction is applied across
    annotation groups separately per label.

    This deliberately calibrates only the three expression channels.  Spatial
    compartment scores are omitted because they should constrain geometry, not
    manufacture subtype significance.
    """
    scores = np.asarray(score_matrix, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    n_groups, n_genes = scores.shape
    if len(cluster_ids) != n_groups:
        raise ValueError("cluster_ids length must match score_matrix rows")

    permutations = max(1, int(permutations))
    min_effective_markers = max(1, int(min_effective_markers))
    rng = np.random.default_rng(int(seed))
    index = pd.Index([str(value) for value in cluster_ids], name="leiden")
    result = pd.DataFrame(index=index)

    if n_genes == 0:
        for region in regions:
            result[f"effective_marker_count__{region}"] = 0
            result[f"subtype_testable__{region}"] = False
            result[f"subtype_empirical_p__{region}"] = np.nan
            result[f"subtype_empirical_q__{region}"] = np.nan
            result[f"subtype_calibrated_score__{region}"] = np.nan
        return result

    ranks = np.vstack(
        [rankdata(row, method="average") for row in scores]
    ).astype(np.float64, copy=False)
    score_std = scores.std(axis=1, dtype=np.float64) + 1e-8
    top_k = min(
        n_genes,
        max(TOPK_OVERLAP_MIN, int(TOPK_OVERLAP_FRAC * n_genes)),
    )
    top_mask = np.zeros((n_groups, n_genes), dtype=bool)
    top_indices = np.argpartition(scores, n_genes - top_k, axis=1)[:, -top_k:]
    top_mask[np.arange(n_groups)[:, None], top_indices] = True
    all_genes = np.arange(n_genes, dtype=np.int64)
    channel_denominator = math.sqrt(3.0)

    if expression_matrix is None:
        assay_values = np.abs(scores).astype(np.float64, copy=False)
    elif sp.issparse(expression_matrix):
        assay_values = expression_matrix.toarray().astype(
            np.float64,
            copy=False,
        )
    else:
        assay_values = np.asarray(
            expression_matrix,
            dtype=np.float64,
        )
    if assay_values.shape != scores.shape:
        raise ValueError(
            "expression_matrix must have the same shape as score_matrix"
        )
    assay_values[~np.isfinite(assay_values)] = 0.0
    detection_rate = np.mean(assay_values > 0.0, axis=0)
    mean_expression = np.log1p(
        np.maximum(assay_values, 0.0).mean(axis=0)
    )
    matching_bins = max(2, int(matching_bins))

    def _rank_bins(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(values, method="average")
        scaled = (ranks - 1.0) / max(float(len(values)), 1.0)
        return np.minimum(
            (scaled * matching_bins).astype(np.int16),
            matching_bins - 1,
        )

    detection_bin = _rank_bins(detection_rate)
    expression_bin = _rank_bins(mean_expression)
    strata = detection_bin.astype(np.int32) * matching_bins
    strata += expression_bin.astype(np.int32)

    def _mwu_from_rank_sums(rank_sums: np.ndarray, marker_count: int) -> np.ndarray:
        other_count = n_genes - marker_count
        if marker_count < 5 or other_count < 5:
            return np.zeros_like(rank_sums, dtype=np.float64)
        u_value = rank_sums - marker_count * (marker_count + 1) / 2.0
        mean_u = marker_count * other_count / 2.0
        std_u = math.sqrt(
            marker_count
            * other_count
            * (marker_count + other_count + 1)
            / 12.0
        )
        if std_u <= 0 or not np.isfinite(std_u):
            return np.zeros_like(rank_sums, dtype=np.float64)
        return (u_value - mean_u) / std_u

    def _hypergeom_z(overlap: np.ndarray, marker_count: int) -> np.ndarray:
        p_value = hypergeom.sf(
            np.maximum(np.asarray(overlap, dtype=np.int64), 0) - 1,
            n_genes,
            marker_count,
            top_k,
        )
        p_value = np.asarray(p_value, dtype=np.float64)
        z_value = norm.isf(np.clip(p_value, 1e-300, 1.0))
        z_value[p_value <= 0.0] = 8.0
        # A right-tail p-value of exactly one carries no positive enrichment
        # evidence. Represent it as zero rather than -inf so it does not erase
        # informative MWU/projection channels in the empirical statistic.
        z_value[~np.isfinite(z_value)] = 0.0
        return z_value

    for region in regions:
        marker_idx = np.asarray(
            region_indices.get(str(region), np.empty(0, dtype=np.int64)),
            dtype=np.int64,
        )
        marker_idx = marker_idx[
            (marker_idx >= 0) & (marker_idx < n_genes)
        ]
        marker_idx = np.unique(marker_idx)
        marker_count = int(marker_idx.size)
        count_col = f"effective_marker_count__{region}"
        testable_col = f"subtype_testable__{region}"
        p_col = f"subtype_empirical_p__{region}"
        q_col = f"subtype_empirical_q__{region}"
        calibrated_col = f"subtype_calibrated_score__{region}"
        result[count_col] = marker_count

        excluded = np.zeros(n_genes, dtype=bool)
        excluded[marker_idx] = True
        null_pool = all_genes[~excluded]
        testable = bool(
            marker_count >= min_effective_markers
            and marker_count >= 5
            and (n_genes - marker_count) >= 5
            and null_pool.size >= marker_count
        )
        result[testable_col] = testable
        if not testable:
            result[p_col] = np.nan
            result[q_col] = np.nan
            result[calibrated_col] = np.nan
            continue

        full_weights = np.asarray(
            region_weights.get(str(region), np.zeros(n_genes)),
            dtype=np.float64,
        )
        marker_weights = full_weights[marker_idx]
        if not np.any(marker_weights > 0):
            marker_weights = np.ones(marker_count, dtype=np.float64)
        marker_weights = marker_weights / max(
            float(np.linalg.norm(marker_weights)), 1e-8
        )
        marker_strata = strata[marker_idx]
        stratum_targets = []
        for stratum in np.unique(marker_strata):
            positions = np.flatnonzero(marker_strata == stratum)
            exact_available = int(
                np.count_nonzero(strata[null_pool] == stratum)
            )
            # Allocate the hardest-to-match strata first so abundant strata do
            # not consume their nearest fallback genes.
            scarcity = exact_available / max(int(positions.size), 1)
            stratum_targets.append(
                (float(scarcity), int(stratum), positions)
            )
        stratum_targets.sort(key=lambda item: (item[0], item[1]))

        observed_mwu = _mwu_from_rank_sums(
            ranks[:, marker_idx].sum(axis=1, dtype=np.float64),
            marker_count,
        )
        observed_proj = (
            scores[:, marker_idx].astype(np.float64) @ marker_weights
        ) / score_std
        observed_hyp = _hypergeom_z(
            top_mask[:, marker_idx].sum(axis=1),
            marker_count,
        )
        observed = (
            observed_mwu + observed_proj + observed_hyp
        ) / channel_denominator
        observed[~np.isfinite(observed)] = 0.0

        null_indices = np.empty(
            (permutations, marker_count),
            dtype=np.int64,
        )
        null_weights = np.empty(
            (permutations, marker_count),
            dtype=np.float64,
        )
        for permutation_idx in range(permutations):
            available = null_pool.copy()
            write_offset = 0
            for _, stratum, marker_positions in stratum_targets:
                needed = int(marker_positions.size)
                target_detection = stratum // matching_bins
                target_expression = stratum % matching_bins
                distance = (
                    np.abs(
                        detection_bin[available] - target_detection
                    )
                    + np.abs(
                        expression_bin[available] - target_expression
                    )
                )
                # Randomize ties before the stable distance sort.
                shuffled = rng.permutation(len(available))
                nearest_order = shuffled[
                    np.argsort(distance[shuffled], kind="stable")
                ]
                chosen_positions = nearest_order[:needed]
                chosen = available[chosen_positions]
                end = write_offset + needed
                null_indices[
                    permutation_idx,
                    write_offset:end,
                ] = chosen
                null_weights[
                    permutation_idx,
                    write_offset:end,
                ] = rng.permutation(
                    marker_weights[marker_positions]
                )
                keep_available = np.ones(len(available), dtype=bool)
                keep_available[chosen_positions] = False
                available = available[keep_available]
                write_offset = end
            if write_offset != marker_count:
                raise RuntimeError(
                    "Matched-null sampler produced the wrong marker count"
                )
        empirical_p = np.empty(n_groups, dtype=np.float64)
        calibrated_score = np.empty(n_groups, dtype=np.float64)
        for group_idx in range(n_groups):
            null_rank_sums = ranks[group_idx, null_indices].sum(
                axis=1,
                dtype=np.float64,
            )
            null_mwu = _mwu_from_rank_sums(
                null_rank_sums,
                marker_count,
            )
            null_proj = (
                scores[group_idx, null_indices].astype(np.float64)
                * null_weights
            ).sum(axis=1) / score_std[group_idx]
            null_overlap = top_mask[group_idx, null_indices].sum(axis=1)
            null_hyp = _hypergeom_z(null_overlap, marker_count)
            null_score = (
                null_mwu + null_proj + null_hyp
            ) / channel_denominator
            null_score[~np.isfinite(null_score)] = 0.0
            null_center = float(np.median(null_score))
            null_scale = float(
                1.4826 * np.median(np.abs(null_score - null_center))
            )
            if not np.isfinite(null_scale) or null_scale < 1e-8:
                null_scale = float(np.std(null_score))
            null_scale = max(null_scale, 1e-8)
            calibrated_score[group_idx] = np.clip(
                (observed[group_idx] - null_center) / null_scale,
                -20.0,
                20.0,
            )
            empirical_p[group_idx] = (
                1.0 + np.count_nonzero(null_score >= observed[group_idx])
            ) / (permutations + 1.0)

        empirical_q = multipletests(empirical_p, method="fdr_bh")[1]
        result[p_col] = empirical_p
        result[q_col] = empirical_q
        result[calibrated_col] = calibrated_score

    return result


def _choose_geometry_consistent_label(
    regions: List[str],
    z_scores: np.ndarray,
    q_scores: np.ndarray,
    channel_support: np.ndarray,
    chosen_idx: int,
    r_value: float,
    min_abs_r: float = 0.05,
    max_score_gap: float = 8.5,
    min_candidate_z: float = 1.0,
    max_candidate_q: float = 0.35,
    min_channel_support: int = 1,
) -> Tuple[int, bool, str, float]:
    """Softly prefer geometry-consistent labels in single-stage annotation."""
    if not np.isfinite(r_value) or abs(float(r_value)) < float(min_abs_r):
        return chosen_idx, False, "", np.nan

    preferred = "Nuclear" if float(r_value) < 0 else "Cytoplasmic"
    chosen_comp = _label_compartment(str(regions[int(chosen_idx)]))
    if chosen_comp == preferred:
        return chosen_idx, False, preferred, 0.0

    pref_idx = [
        i for i, label in enumerate(regions)
        if _label_compartment(str(label)) == preferred
    ]
    if not pref_idx:
        return chosen_idx, False, preferred, np.nan

    z = np.asarray(z_scores, dtype=float)
    q = np.asarray(q_scores, dtype=float)
    support = np.asarray(channel_support, dtype=float)
    valid = [
        i for i in pref_idx
        if np.isfinite(z[i])
        and np.isfinite(q[i])
        and z[i] >= float(min_candidate_z)
        and q[i] <= float(max_candidate_q)
        and support[i] >= int(min_channel_support)
    ]
    if not valid:
        return chosen_idx, False, preferred, np.nan

    candidate_idx = int(max(valid, key=lambda i: z[i]))
    score_gap = float(z[int(chosen_idx)] - z[candidate_idx])
    if score_gap <= float(max_score_gap):
        return candidate_idx, True, preferred, score_gap
    return chosen_idx, False, preferred, score_gap


# ---------- 核心标注逻辑（新增空间通道）----------
def _rapx_annotate_core(adata_sub, sets, regions,
                        set_weights=None,
                        use_spatial_channel=False,
                        geometry_channel_weight=0.25,
                        guard_mode="fixed",
                        subtype_fdr=0.05,
                        min_effective_markers=10,
                        null_permutations=500,
                        null_seed=0,
                        single_stage_compartment_score_weight=6.0,
                        single_stage_compartment_score_tau=0.2,
                        min_channel_support=2,
                        enforce_geometry_consistency=True,
                        geometry_quantile=0.4,
                        single_stage_compartment_guard=False,
                        single_stage_compartment_min_abs_r=0.05,
                        single_stage_compartment_max_score_gap=8.5,
                        single_stage_compartment_min_z=1.0,
                        single_stage_compartment_max_q=0.35,
                        single_stage_compartment_min_channel_support=1,
                        geom_before_top2=True,
                        top2_switch_z_floor_ratio=0.9,
                        top2_switch_q_ceiling_ratio=2.0,
                        top2_switch_z_not_much_worse=0.25):
    gold_genes_upper = pd.Index(sorted(set().union(*[set(map(str.upper, s)) for s in sets.values()])))
    group_mean, cluster_ids, genes_up = build_group_means(adata_sub, gold_genes_upper)
    S = robust_gene_stats(group_mean)

    Gn = len(genes_up)
    gene_pos = {g: i for i, g in enumerate(genes_up)}
    region_indices, region_weights = {}, {}
    for r in regions:
        idx = np.fromiter((gene_pos[g] for g in sets[r] if g in gene_pos), dtype=np.int64, count=-1)
        idx.sort()
        region_indices[r] = idx
        w = np.zeros(Gn, dtype=np.float32)
        if idx.size > 0:
            if set_weights is not None and r in set_weights:
                raw_weights = np.asarray(
                    [float(set_weights[r].get(genes_up[k], 0.0)) for k in idx],
                    dtype=np.float32,
                )
                if not np.any(raw_weights > 0):
                    raw_weights = np.ones(idx.size, dtype=np.float32)
            else:
                raw_weights = np.ones(idx.size, dtype=np.float32)
            norm_w = float(np.linalg.norm(raw_weights))
            w[idx] = raw_weights / max(norm_w, 1e-8)
        region_weights[r] = w

    def z_mwu_from_ranks(ranks: np.ndarray, idx_in: np.ndarray) -> float:
        n1 = idx_in.size
        n2 = ranks.size - n1
        if n1 < 5 or n2 < 5:
            return 0.0
        R_sum = ranks[idx_in].sum(dtype=np.float64)
        U = R_sum - n1 * (n1 + 1) / 2.0
        mean_U = n1 * n2 / 2.0
        std_U  = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        if std_U <= 0 or not np.isfinite(std_U):
            return 0.0
        return float((U - mean_U) / std_U)

    def p_to_z_right_tail(p: float, cap: float = 8.0) -> float:
        if p <= 0:
            return cap
        p = max(min(p, 1.0), 1e-300)
        return float(norm.isf(p))

    rmean_cache = (
        adata_sub.obs.groupby("leiden", observed=True)["r_mean"].mean().to_dict()
        if "r_mean" in adata_sub.obs.columns
        else {}
    )
    r_values_all = np.asarray(list(rmean_cache.values()), dtype=float)
    r_values_all = r_values_all[np.isfinite(r_values_all)]
    geometry_center = (
        float(np.median(r_values_all)) if r_values_all.size else 0.0
    )
    geometry_scale = (
        float(1.4826 * np.median(np.abs(r_values_all - geometry_center)))
        if r_values_all.size
        else 1.0
    )
    if not np.isfinite(geometry_scale) or geometry_scale < 1e-6:
        geometry_scale = (
            float(np.std(r_values_all)) if r_values_all.size else 1.0
        )
    geometry_scale = max(geometry_scale, 1e-6)
    NUC_GROUP = set(NUCLEAR_LABELS)
    CYT_GROUP = set(CYTOPLASMIC_LABELS)
    nuclear_subtype_mode = set(regions).issubset(NUC_GROUP)
    mixed_compartment_mode = bool(set(regions) & NUC_GROUP) and bool(set(regions) & CYT_GROUP)
    geometry_boundary_threshold = np.nan
    geometry_deep_threshold = np.nan
    if (
        use_spatial_channel
        and enforce_geometry_consistency
        and nuclear_subtype_mode
        and len(rmean_cache) >= 4
    ):
        r_values = np.asarray(list(rmean_cache.values()), dtype=float)
        (
            geometry_boundary_threshold,
            geometry_deep_threshold,
        ) = _nuclear_geometry_thresholds(
            r_values,
            geometry_quantile,
        )

    records = []
    for iC, clu in enumerate(tqdm(cluster_ids, desc="RAP-X annotation (with spatial)", ncols=80)):
        s_vec = S[iC, :].astype(np.float32, copy=False)
        ranks = rankdata(s_vec, method='average').astype(np.float64)

        G_eff = s_vec.size
        K = max(TOPK_OVERLAP_MIN, int(TOPK_OVERLAP_FRAC * G_eff))
        K = min(K, G_eff)
        topk_idx = np.argpartition(s_vec, G_eff - K)[-K:]
        topk_mask = np.zeros(G_eff, dtype=bool)
        topk_mask[topk_idx] = True

        s_std = float(np.std(s_vec)) + 1e-8

        z_mwu_arr  = np.zeros(len(regions), dtype=np.float32)
        z_proj_arr = np.zeros(len(regions), dtype=np.float32)
        z_hyp_arr  = np.zeros(len(regions), dtype=np.float32)
        z_spatial_arr = np.zeros(len(regions), dtype=np.float32)
        z_compartment_arr = np.zeros(len(regions), dtype=np.float32)

        for j, r in enumerate(regions):
            idx_in = region_indices[r]
            if idx_in.size == 0:
                z_mwu_arr[j] = 0.0; z_proj_arr[j] = 0.0; z_hyp_arr[j] = 0.0; z_spatial_arr[j] = 0.0
                continue

            z_mwu_arr[j] = z_mwu_from_ranks(ranks, idx_in)
            z_proj_arr[j] = float(np.dot(s_vec, region_weights[r]) / s_std) if s_std > 0 else 0.0
            x = int(topk_mask[idx_in].sum())
            p = float(hypergeom.sf(x - 1, G_eff, idx_in.size, K))
            z_hyp_arr[j] = p_to_z_right_tail(p)

        if use_spatial_channel:
            if mixed_compartment_mode:
                # In a joint seven-label competition, subtype priors are not
                # compartment priors.  In particular, ER=-0.5*z rewards ER on
                # the nuclear side.  Use a separate symmetric soft score here
                # and reserve subtype geometry for within-compartment ranking.
                z_compartment_arr = _compartment_geometry_scores(
                    regions,
                    rmean_cache.get(clu, np.nan),
                    tau=single_stage_compartment_score_tau,
                    min_abs_r=single_stage_compartment_min_abs_r,
                )
            else:
                z_spatial_arr = _subtype_geometry_scores(
                    regions,
                    rmean_cache.get(clu, np.nan),
                    geometry_center,
                    geometry_scale,
                )

        spatial_informative = False
        if use_spatial_channel:
            spatial_std = float(z_spatial_arr.std())
            spatial_informative = spatial_std > 1e-8
            if spatial_informative:
                z_spatial_arr = (
                    z_spatial_arr - z_spatial_arr.mean()
                ) / spatial_std
            else:
                z_spatial_arr.fill(0.0)

        if spatial_informative and len(CHANNEL_WEIGHTS) >= 4:
            w_mwu, w_proj, w_hyp = CHANNEL_WEIGHTS[:3]
            w_spatial = float(geometry_channel_weight)
            den = np.sqrt(w_mwu**2 + w_proj**2 + w_hyp**2 + w_spatial**2)
            z_comb = (w_mwu*z_mwu_arr + w_proj*z_proj_arr + w_hyp*z_hyp_arr + w_spatial*z_spatial_arr) / den
        else:
            w_mwu, w_proj, w_hyp = CHANNEL_WEIGHTS[:3]
            den = np.sqrt(w_mwu**2 + w_proj**2 + w_hyp**2)
            z_comb = (w_mwu*z_mwu_arr + w_proj*z_proj_arr + w_hyp*z_hyp_arr) / den

        # Add the bounded compartment prior before centering, FDR estimation,
        # and ranking.  All labels remain in the same argmax, so this is still
        # a single-stage model rather than a geometry gate or post-hoc flip.
        if (
            use_spatial_channel
            and mixed_compartment_mode
            and float(single_stage_compartment_score_weight) > 0.0
        ):
            z_comb = (
                z_comb
                + float(single_stage_compartment_score_weight)
                * z_compartment_arr
            )

        z_comb[~np.isfinite(z_comb)] = 0.0
        med = np.median(z_comb[np.isfinite(z_comb)])
        z_centered = z_comb - med

        p_for_fdr = 0.5 * erfc(z_centered / np.sqrt(2.0))
        p_for_fdr = np.clip(p_for_fdr, 1e-300, 1.0)
        order_p = np.argsort(p_for_fdr)
        ranks_p = np.arange(1, len(regions) + 1, dtype=np.float64)
        q_comb = np.empty_like(p_for_fdr)
        q_comb[order_p] = p_for_fdr[order_p] * len(regions) / ranks_p
        for t in range(len(regions) - 2, -1, -1):
            q_comb[order_p[t]] = min(q_comb[order_p[t]], q_comb[order_p[t + 1]])
        q_comb = np.clip(q_comb, 0.0, 1.0)

        ord_reg = np.argsort(-z_centered)
        i1 = ord_reg[0]
        i2 = ord_reg[1] if len(regions) > 1 else ord_reg[0]
        r1, r2 = regions[i1], regions[i2]
        z1, z2 = float(z_centered[i1]), float(z_centered[i2])
        q1, q2 = float(q_comb[i1]), float(q_comb[i2])

        promote = False
        if PROMOTE_TOP2 and (z2 >= Z_CUTOFF) and ((z1 - z2) < DELTA_PROMOTE) and (z2 >= REL_PROMOTE * max(z1, 1e-8)) and (q2 <= FDR_CUTOFF * 1.5):
            promote = True
            i_chosen, i_other = i2, i1
        else:
            i_chosen, i_other = i1, i2

        def _unpack_choice(i_chosen, i_other):
            return (regions[i_chosen], float(z_centered[i_chosen]), float(q_comb[i_chosen]),
                    regions[i_other],  float(z_centered[i_other]),  float(q_comb[i_other]))

        def _apply_geometry(i_chosen, i_other):
            if not (
                use_spatial_channel
                and enforce_geometry_consistency
            ):
                return i_chosen, i_other
            r_chosen, z_chosen, q_chosen, r_other, z_other, q_other = _unpack_choice(i_chosen, i_other)
            r_pref = rmean_cache.get(clu, np.nan)
            reg2idx = {r:i for i,r in enumerate(regions)}
            idx_nuc = [reg2idx[r] for r in regions if r in NUC_GROUP]
            idx_cyt = [reg2idx[r] for r in regions if r in CYT_GROUP]
            if (q_chosen > FDR_CUTOFF or (z_chosen - z_other) < TIE_DELTA) and np.isfinite(r_pref):
                prefer_nuc = (r_pref < 0)
                cand_idx = idx_nuc if prefer_nuc else idx_cyt
                if cand_idx:
                    j = int(cand_idx[np.argmax(z_centered[cand_idx])])
                    if z_centered[j] + 0.25 >= z_chosen:
                        i_chosen = j
                        i_other  = i1 if j != i1 else i2
            return i_chosen, i_other

        def _channel_support_for_idx(idx):
            return sum(
                int(x)
                for x in (
                    z_mwu_arr[idx] > 0,
                    z_proj_arr[idx] > 0,
                    z_hyp_arr[idx] > 0,
                )
            )

        def _is_low_conf(i_chosen, i_other):
            r_chosen, z_chosen, q_chosen, r_other, z_other, q_other = _unpack_choice(i_chosen, i_other)
            channel_support = _channel_support_for_idx(i_chosen)
            supported_channels = channel_support >= int(min_channel_support)
            if ALWAYS_ASSIGN:
                return not (
                    supported_channels
                    and (q_chosen <= FDR_CUTOFF)
                    and (z_chosen >= Z_CUTOFF)
                    and (
                        (z_chosen - z_other) >= TIE_DELTA
                        or q_other > FDR_CUTOFF * 2
                    )
                )
            return not (
                supported_channels
                and (q_chosen <= FDR_CUTOFF)
                and (z_chosen >= Z_CUTOFF)
            )

        def _apply_top2_if_lowconf(i_chosen, i_other):
            prelim_low_conf = _is_low_conf(i_chosen, i_other)
            used_top2 = False
            if prelim_low_conf:
                can_switch = (
                    (z2 >= (Z_CUTOFF * top2_switch_z_floor_ratio)) and
                    (q2 <= (FDR_CUTOFF * top2_switch_q_ceiling_ratio)) and
                    (z2 >= z1 - top2_switch_z_not_much_worse)
                )
                if can_switch:
                    i_chosen, i_other = i_other, i_chosen
                    used_top2 = True
            return i_chosen, i_other, prelim_low_conf, used_top2

        def _apply_single_stage_compartment_guard(i_chosen, i_other):
            if not (
                use_spatial_channel
                and enforce_geometry_consistency
                and
                single_stage_compartment_guard
                and mixed_compartment_mode
            ):
                return i_chosen, i_other, False, "", np.nan
            support_arr = np.asarray(
                [_channel_support_for_idx(i) for i in range(len(regions))],
                dtype=float,
            )
            new_idx, rescued, preferred, score_gap = _choose_geometry_consistent_label(
                regions,
                z_centered,
                q_comb,
                support_arr,
                int(i_chosen),
                rmean_cache.get(clu, np.nan),
                min_abs_r=single_stage_compartment_min_abs_r,
                max_score_gap=single_stage_compartment_max_score_gap,
                min_candidate_z=single_stage_compartment_min_z,
                max_candidate_q=single_stage_compartment_max_q,
                min_channel_support=single_stage_compartment_min_channel_support,
            )
            if rescued and int(new_idx) != int(i_chosen):
                return int(new_idx), int(i_chosen), True, preferred, score_gap
            return i_chosen, i_other, False, preferred, score_gap

        if geom_before_top2:
            i_chosen, i_other = _apply_geometry(i_chosen, i_other)
            i_chosen, i_other, prelim_low_conf, used_top2_due_to_low_conf = _apply_top2_if_lowconf(i_chosen, i_other)
        else:
            i_chosen, i_other, prelim_low_conf, used_top2_due_to_low_conf = _apply_top2_if_lowconf(i_chosen, i_other)
            i_chosen, i_other = _apply_geometry(i_chosen, i_other)

        (
            i_chosen,
            i_other,
            compartment_rescued,
            preferred_compartment,
            compartment_score_gap,
        ) = _apply_single_stage_compartment_guard(i_chosen, i_other)
        # Selection can change in the geometry, top-2, and compartment steps.
        # Confidence must describe the final core candidate, not an earlier
        # intermediate choice.
        prelim_low_conf = _is_low_conf(i_chosen, i_other)

        r_chosen, z_chosen, q_chosen, r_other, z_other, q_other = _unpack_choice(i_chosen, i_other)
        chosen_channel_support = _channel_support_for_idx(i_chosen)
        geometry_consistent = True
        r_pref = rmean_cache.get(clu, np.nan)
        if (
            use_spatial_channel
            and enforce_geometry_consistency
            and np.isfinite(r_pref)
            and nuclear_subtype_mode
        ):
            if (
                r_chosen in {"Lamina", "Nuclear Pore"}
                and np.isfinite(geometry_boundary_threshold)
            ):
                geometry_consistent = r_pref <= geometry_boundary_threshold
            elif (
                r_chosen == "Nucleolus"
                and np.isfinite(geometry_deep_threshold)
            ):
                geometry_consistent = r_pref >= geometry_deep_threshold
        if not geometry_consistent:
            prelim_low_conf = True

        rec = {
            'leiden': clu,
            'label': str(r_chosen),
            'low_conf': bool(prelim_low_conf),
            'top1': r1, 'z1': round(z1,3), 'q1': q1,
            'top2': r2, 'z2': round(z2,3), 'q2': q2,
            'selected_z': round(float(z_chosen), 3),
            'selected_q': float(q_chosen),
            'promoted_from_top2': bool(promote and int(i_chosen) == int(i2)),
            'used_top2_due_to_low_conf': bool(
                used_top2_due_to_low_conf and int(i_chosen) == int(i2)
            ),
            'channel_support': chosen_channel_support,
            'raw_combined_score': float(z_comb[i_chosen]),
            'geometry_consistent': bool(geometry_consistent),
            'compartment_rescued': bool(compartment_rescued),
            'preferred_compartment': str(preferred_compartment),
            'compartment_score_gap': float(compartment_score_gap) if np.isfinite(compartment_score_gap) else np.nan,
        }
        for j, r in enumerate(regions):
            rec[f'z_mwu__{r}']  = round(float(z_mwu_arr[j]), 3)
            rec[f'z_proj__{r}'] = round(float(z_proj_arr[j]), 3)
            rec[f'z_hyp__{r}']  = round(float(z_hyp_arr[j]), 3)
            rec[f'z_comb__{r}'] = round(float(z_comb[j]), 3)
            rec[f'q_comb__{r}'] = float(q_comb[j])
            if use_spatial_channel:
                rec[f'z_spatial__{r}'] = round(float(z_spatial_arr[j]), 3)
                rec[f'z_compartment__{r}'] = round(float(z_compartment_arr[j]), 3)
        records.append(rec)

    annot_df = pd.DataFrame.from_records(records).set_index('leiden')
    if str(guard_mode).strip().lower() == "adaptive":
        calibration = _empirical_marker_set_calibration(
            S,
            cluster_ids,
            regions,
            region_indices,
            region_weights,
            expression_matrix=group_mean,
            permutations=null_permutations,
            min_effective_markers=min_effective_markers,
            seed=null_seed,
        )
        # Group order is already deterministic and identical to annot_df.
        # Assign by position to avoid dtype differences between categorical
        # Leiden IDs and their serialized string representation.
        calibration.index = annot_df.index
        annot_df = annot_df.join(calibration)
        selected_p = []
        selected_q = []
        selected_testable = []
        selected_marker_count = []
        for group_id, label in annot_df["label"].astype(str).items():
            selected_p.append(
                annot_df.at[group_id, f"subtype_empirical_p__{label}"]
            )
            selected_q.append(
                annot_df.at[group_id, f"subtype_empirical_q__{label}"]
            )
            selected_testable.append(
                annot_df.at[group_id, f"subtype_testable__{label}"]
            )
            selected_marker_count.append(
                annot_df.at[group_id, f"effective_marker_count__{label}"]
            )
        annot_df["subtype_empirical_p"] = np.asarray(
            selected_p,
            dtype=np.float64,
        )
        annot_df["subtype_empirical_q"] = np.asarray(
            selected_q,
            dtype=np.float64,
        )
        annot_df["subtype_testable"] = np.asarray(
            selected_testable,
            dtype=bool,
        )
        annot_df["effective_marker_count"] = np.asarray(
            selected_marker_count,
            dtype=np.int64,
        )
        annot_df["subtype_empirical_supported"] = (
            annot_df["subtype_testable"].astype(bool)
            & annot_df["subtype_empirical_q"].le(float(subtype_fdr))
        )
    else:
        marker_counts = {
            str(region): int(
                np.asarray(
                    region_indices.get(region, np.empty(0, dtype=np.int64))
                ).size
            )
            for region in regions
        }
        annot_df["subtype_empirical_p"] = np.nan
        annot_df["subtype_empirical_q"] = np.nan
        annot_df["subtype_testable"] = False
        annot_df["effective_marker_count"] = (
            annot_df["label"].astype(str).map(marker_counts).fillna(0).astype(int)
        )
        annot_df["subtype_empirical_supported"] = False
    return annot_df


# ---------- 工具：初始化/清理目标列 dtype，避免 Categorical 写回错误 ----------
def _ensure_obs_columns_initialized(adata, produced_label_col, meta_cols):
    import numpy as _np
    import pandas as _pd

    if produced_label_col not in adata.obs.columns:
        adata.obs[produced_label_col] = _pd.Series([_pd.NA] * adata.n_obs, dtype="object", index=adata.obs_names)
    else:
        adata.obs[produced_label_col] = adata.obs[produced_label_col].astype("object")

    _meta_dtypes = {
        'top1': 'object', 'top2': 'object',
        'z1': 'float32', 'z2': 'float32',
        'q1': 'float64', 'q2': 'float64',
        'low_conf': 'boolean',
        'promoted_from_top2': 'boolean',
        'used_top2_due_to_low_conf': 'boolean',
        'weak_top1_demoted': 'boolean',
        'core_candidate_label': 'object',
        'subtype_empirical_p': 'float64',
        'subtype_empirical_q': 'float64',
        'subtype_testable': 'boolean',
        'subtype_empirical_supported': 'boolean',
        'effective_marker_count': 'float32',
        'strict_label': 'object',
        'exploratory_label': 'object',
        'annotation_tier': 'object',
        'primary_label': 'object',
        'secondary_label': 'object',
        'annotation_state': 'object',
        'supported_labels': 'object',
        'n_supported_labels': 'float32',
        'primary_evidence_score': 'float32',
        'secondary_evidence_score': 'float32',
        'evidence_score_gap': 'float32',
        'primary_empirical_q': 'float64',
        'secondary_empirical_q': 'float64',
    }

    for col in meta_cols:
        tgt = f"subcell_{col}"
        want = _meta_dtypes.get(col, 'object')

        if (tgt not in adata.obs.columns) or isinstance(adata.obs[tgt].dtype, _pd.CategoricalDtype):
            if want in ('float32', 'float64'):
                fill = _np.nan
            else:
                fill = _pd.NA
            adata.obs[tgt] = _pd.Series([fill] * adata.n_obs, dtype=want, index=adata.obs_names)


def _refresh_subcellular_group_columns(
    adata: ad.AnnData,
    label_col: str = "subcellular_annotation",
) -> None:
    """Keep object-level labels separate from Leiden-level majority labels."""
    if label_col not in adata.obs.columns or "leiden" not in adata.obs.columns:
        return
    labels = adata.obs[label_col].astype(str)
    leiden = adata.obs["leiden"].astype(str)
    dominant = (
        adata.obs.assign(_subcell_label=labels)
        .groupby("leiden", observed=True)["_subcell_label"]
        .agg(lambda s: s.value_counts().idxmax() if len(s) else "Unassigned")
    )
    adata.obs["leiden-dominant-subcellular"] = (
        adata.obs["leiden"].map(dominant).astype(str)
    )
    adata.obs["cluster-subcellular"] = leiden + ":" + labels


def _coerce_bool_series(values: pd.Series, default: bool = True) -> pd.Series:
    """Parse bool-like values, including categorical "True"/"False" strings."""
    raw = pd.Series(values, index=values.index, dtype="object")
    parsed = raw.map(
        {
            True: True,
            False: False,
            1: True,
            0: False,
            "1": True,
            "0": False,
            "true": True,
            "false": False,
            "True": True,
            "False": False,
        }
    )
    parsed.loc[parsed.isna()] = bool(default)
    return parsed.astype(bool)


def _apply_low_confidence_policy(
    annot_df: pd.DataFrame,
    compartment: str,
    policy: str = "supported",
    relaxed_z: float = 1.25,
    relaxed_q: float = 0.30,
) -> pd.DataFrame:
    """Attach detailed, conservative, and confidence-aware annotation labels."""
    out = annot_df.copy()
    if "candidate_label" not in out.columns:
        out["candidate_label"] = out["label"].astype(str)
    else:
        out["candidate_label"] = out["candidate_label"].astype(str)
    low_conf = _coerce_bool_series(out["low_conf"], default=True)
    selected_z = pd.to_numeric(
        out.get(
            "selected_z",
            out.get("z1", pd.Series(np.nan, index=out.index)),
        ),
        errors="coerce",
    )
    selected_q = pd.to_numeric(
        out.get(
            "selected_q",
            out.get("q1", pd.Series(np.nan, index=out.index)),
        ),
        errors="coerce",
    )
    channel_support = pd.to_numeric(
        out.get("channel_support", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0)
    geometry_consistent = _coerce_bool_series(
        out.get(
            "geometry_consistent",
            pd.Series(True, index=out.index),
        ),
        default=True,
    )
    statistically_supported = (
        (selected_z >= float(relaxed_z))
        & (selected_q <= float(relaxed_q))
        & (channel_support >= 2)
    )
    strong_single_channel = (
        (selected_z >= 3.0)
        & (selected_q <= 0.01)
        & (channel_support >= 1)
    )

    confidence = pd.Series("Low", index=out.index, dtype="object")
    confidence.loc[
        statistically_supported | strong_single_channel
    ] = "Medium"
    confidence.loc[~low_conf & geometry_consistent] = "High"
    confidence.loc[
        ~low_conf & ~geometry_consistent
    ] = "Medium"
    out["confidence_level"] = confidence

    fallback_label = "Nucleus" if compartment == "Nuclear" else "Cytosol"
    out["conservative_label"] = out["candidate_label"]
    out.loc[confidence != "High", "conservative_label"] = fallback_label

    policy = str(policy).strip().lower()
    if policy == "keep":
        return out
    if policy == "supported":
        keep_subtype = (confidence == "High") | (
            (confidence == "Medium") & geometry_consistent
        )
        out.loc[~keep_subtype, "label"] = fallback_label
        return out

    fallback = low_conf
    if policy == "adaptive":
        fallback = low_conf & ~statistically_supported
    out.loc[fallback, "label"] = fallback_label
    return out


def _apply_low_confidence_policy_by_candidate(
    annot_df: pd.DataFrame,
    policy: str = "supported",
    relaxed_z: float = 1.25,
    relaxed_q: float = 0.30,
) -> pd.DataFrame:
    """Apply confidence policy to mixed nuclear/cytoplasmic annotation rows."""
    out = annot_df.copy()
    if "candidate_label" not in out.columns:
        out["candidate_label"] = out["label"].astype(str)

    nuclear_mask = out["candidate_label"].isin(
        {"Nucleus", "Nucleolus", "Lamina", "Nuclear Pore"}
    )
    parts = []
    for mask, compartment in (
        (nuclear_mask, "Nuclear"),
        (~nuclear_mask, "Cytoplasmic"),
    ):
        if mask.any():
            parts.append(
                _apply_low_confidence_policy(
                    out.loc[mask].copy(),
                    compartment=compartment,
                    policy=policy,
                    relaxed_z=relaxed_z,
                    relaxed_q=relaxed_q,
                )
            )
    if not parts:
        return out
    return pd.concat(parts, axis=0).reindex(out.index)


def _apply_independent_evidence_policy(
    annot_df: pd.DataFrame,
    regions: Sequence[str],
    *,
    group_r_mean: Optional[pd.Series] = None,
    subtype_fdr: float = 0.05,
    min_effective_markers: int = 10,
    min_channel_support: int = 2,
    geometry_min_abs_r: float = 0.05,
    mixed_score_delta: float = 0.75,
    unknown_label: str = "Unknown",
    assignment_mode: str = "strict",
    exploratory_max_q: float = 0.30,
    exploratory_min_channel_support: int = 2,
    exploratory_min_score_gap: float = 0.50,
    subtype_parent_margin: float = 0.0,
    omm_competitor_margin: float = 0.0,
) -> pd.DataFrame:
    """Select labels from independent empirical evidence, not a seven-way gate.

    Expression evidence is evaluated for every label.  Signed geometry only
    marks broad nuclear/cytoplasmic feasibility; it never adds to the subtype
    score and never replaces one supported subtype with another.

    ``assignment_mode='strict'`` preserves the original behaviour: a label is
    emitted only when it passes the label-wise empirical FDR test.
    ``assignment_mode='tiered'`` emits the best geometry-feasible expression
    candidate for visualization and exploratory use.  The recommended
    ``assignment_mode='hierarchical'`` emits supported/exploratory subtypes,
    but collapses an unsupported organelle candidate to its broad nuclear or
    cytoplasmic parent.  ``strict_label`` and ``annotation_tier`` always retain
    the evidence boundary explicitly.
    """
    out = annot_df.copy()
    labels = [str(region) for region in regions]
    if not labels:
        return out
    assignment_mode = str(assignment_mode).strip().lower()
    if assignment_mode not in {"strict", "tiered", "hierarchical"}:
        raise ValueError(
            "assignment_mode must be strict, tiered, or hierarchical"
        )
    exploratory_max_q = float(np.clip(exploratory_max_q, 0.0, 1.0))
    exploratory_min_channel_support = max(
        0,
        int(exploratory_min_channel_support),
    )
    exploratory_min_score_gap = max(
        0.0,
        float(exploratory_min_score_gap),
    )
    subtype_parent_margin = max(0.0, float(subtype_parent_margin))
    omm_competitor_margin = max(0.0, float(omm_competitor_margin))

    weights = np.asarray(CHANNEL_WEIGHTS[:3], dtype=float)
    denominator = float(np.linalg.norm(weights))
    denominator = max(denominator, 1e-8)
    # Projection uses L2-normalized marker weights, so its expected magnitude
    # grows with roughly sqrt(marker-set size). It remains useful inside the
    # label-wise matched-null test, but including it in a direct ER-vs-OMM
    # choice systematically penalizes the smaller specific set. MWU and
    # hypergeometric channels already account for set size under their nulls.
    choice_weights = np.asarray([weights[0], 0.0, weights[2]], dtype=float)
    choice_denominator = max(
        float(np.linalg.norm(choice_weights)),
        1e-8,
    )
    raw_evidence = pd.DataFrame(
        index=out.index,
        columns=labels,
        dtype=float,
    )
    evidence = pd.DataFrame(index=out.index, columns=labels, dtype=float)
    empirical_q = pd.DataFrame(index=out.index, columns=labels, dtype=float)
    testable = pd.DataFrame(False, index=out.index, columns=labels)
    marker_count = pd.DataFrame(0, index=out.index, columns=labels, dtype=int)
    channel_support = pd.DataFrame(
        0,
        index=out.index,
        columns=labels,
        dtype=int,
    )

    for label in labels:
        channel_values = []
        for prefix in ("z_mwu", "z_proj", "z_hyp"):
            channel_values.append(
                pd.to_numeric(
                    out.get(
                        f"{prefix}__{label}",
                        pd.Series(0.0, index=out.index),
                    ),
                    errors="coerce",
                ).fillna(0.0)
            )
        matrix = np.column_stack(
            [value.to_numpy(dtype=float) for value in channel_values]
        )
        raw_score = pd.Series(
            (matrix @ weights) / denominator,
            index=out.index,
            dtype=float,
        )
        raw_evidence[label] = raw_score
        # The label-specific null calibration answers whether one marker
        # program is stronger than matched random gene sets. It is valid for
        # support/q-values, but its z-score must not be used as a seven-way
        # biological competition: each label has a different null. Cross-label
        # choice therefore uses the observed set-size-aware MWU and
        # hypergeometric channels; the L2 projection is retained only in the
        # raw diagnostic and matched-null support statistic.
        evidence[label] = pd.Series(
            (matrix @ choice_weights) / choice_denominator,
            index=out.index,
            dtype=float,
        )
        channel_support[label] = (matrix > 0).sum(axis=1)
        empirical_q[label] = pd.to_numeric(
            out.get(
                f"subtype_empirical_q__{label}",
                pd.Series(np.nan, index=out.index),
            ),
            errors="coerce",
        )
        testable[label] = _coerce_bool_series(
            out.get(
                f"subtype_testable__{label}",
                pd.Series(False, index=out.index),
            ),
            default=False,
        )
        marker_count[label] = pd.to_numeric(
            out.get(
                f"effective_marker_count__{label}",
                pd.Series(0, index=out.index),
            ),
            errors="coerce",
        ).fillna(0).astype(int)

    feasible = pd.DataFrame(True, index=out.index, columns=labels)
    if group_r_mean is not None:
        r_values = pd.to_numeric(
            pd.Series(group_r_mean).reindex(out.index),
            errors="coerce",
        )
        nuclear = r_values < -abs(float(geometry_min_abs_r))
        cytoplasmic = r_values > abs(float(geometry_min_abs_r))
        for label in labels:
            if label in NUCLEAR_LABELS:
                feasible.loc[cytoplasmic, label] = False
            elif label in CYTOPLASMIC_LABELS:
                feasible.loc[nuclear, label] = False

    statistically_supported = (
        testable
        & empirical_q.le(float(subtype_fdr))
        & marker_count.ge(int(min_effective_markers))
        & channel_support.ge(int(min_channel_support))
        & feasible
    )

    specificity_margin = pd.DataFrame(
        np.inf,
        index=out.index,
        columns=labels,
        dtype=float,
    )
    specificity_pass = pd.DataFrame(
        True,
        index=out.index,
        columns=labels,
        dtype=bool,
    )
    parent_competitors: Dict[str, Tuple[str, ...]] = {
        "Nucleolus": ("Nucleus",),
        "Lamina": ("Nucleus",),
        "Nuclear Pore": ("Nucleus",),
        "ER": ("Cytosol",),
        "OMM": ("ER", "Cytosol"),
    }
    omm_competitor_label = pd.Series(
        "",
        index=out.index,
        dtype="object",
    )
    omm_competitor_score = pd.Series(
        np.nan,
        index=out.index,
        dtype=float,
    )
    for label, requested_competitors in parent_competitors.items():
        if label not in labels:
            continue
        available = [
            competitor
            for competitor in requested_competitors
            if competitor in labels
        ]
        if not available:
            # A subtype cannot be called specific if its broad/membrane
            # alternative was absent from the competition.
            specificity_margin[label] = -np.inf
            specificity_pass[label] = False
            continue
        competitor_scores = evidence[available]
        competitor_max = competitor_scores.max(axis=1)
        margins = evidence[label] - competitor_max
        required_margin = (
            omm_competitor_margin
            if label == "OMM"
            else subtype_parent_margin
        )
        specificity_margin[label] = margins
        specificity_pass[label] = margins.ge(required_margin)
        if label == "OMM":
            omm_competitor_score = competitor_max.astype(float)
            omm_competitor_label = competitor_scores.idxmax(axis=1)

    supported = statistically_supported & specificity_pass

    if "OMM" in labels:
        omm_base = statistically_supported["OMM"]
        omm_specific = specificity_pass["OMM"]
        omm_vs_cytosol = pd.Series(
            np.nan,
            index=out.index,
            dtype=float,
        )
        if "Cytosol" in labels:
            omm_vs_cytosol = (
                evidence["OMM"] - evidence["Cytosol"]
            )
        shared = (
            omm_base
            & ~omm_specific
            & omm_competitor_label.eq("ER")
            & (
                omm_vs_cytosol.ge(omm_competitor_margin)
                | omm_vs_cytosol.isna()
            )
        )
        omm_domain_state = pd.Series(
            "Rejected",
            index=out.index,
            dtype="object",
        )
        omm_domain_state.loc[shared] = "ER-OMM shared"
        omm_domain_state.loc[omm_base & omm_specific] = "Confirmed"
    else:
        omm_domain_state = pd.Series(
            "Not evaluated",
            index=out.index,
            dtype="object",
        )

    # Relative evidence is useful for ranking and visualization, but is not a
    # calibrated biological posterior.  Keep that distinction in the column
    # name and metadata.
    masked_scores = evidence.where(feasible, -np.inf)
    row_max = masked_scores.replace([np.inf, -np.inf], np.nan).max(axis=1)
    shifted = masked_scores.sub(row_max, axis=0).clip(lower=-50.0)
    exp_score = np.exp(shifted).where(feasible, 0.0).fillna(0.0)
    denom = exp_score.sum(axis=1).replace(0.0, 1.0)
    relative = exp_score.div(denom, axis=0)

    for label in labels:
        out[f"raw_evidence_score__{label}"] = raw_evidence[label].astype(
            "float32"
        )
        out[f"evidence_score__{label}"] = evidence[label].astype("float32")
        out[f"relative_evidence__{label}"] = relative[label].astype("float32")
        out[f"channel_support__{label}"] = channel_support[label].astype("int8")
        out[f"geometry_feasible__{label}"] = feasible[label].astype(bool)
        out[f"statistically_supported__{label}"] = (
            statistically_supported[label].astype(bool)
        )
        out[f"specificity_margin__{label}"] = (
            specificity_margin[label].astype("float32")
        )
        out[f"specificity_pass__{label}"] = (
            specificity_pass[label].astype(bool)
        )
        out[f"empirically_supported__{label}"] = supported[label].astype(bool)
    out["omm_raw_competitor_label"] = omm_competitor_label
    out["omm_raw_competitor_score"] = omm_competitor_score.astype("float32")
    out["omm_specificity_margin"] = (
        specificity_margin["OMM"].astype("float32")
        if "OMM" in labels
        else np.nan
    )
    out["omm_specificity_pass"] = (
        specificity_pass["OMM"].astype(bool)
        if "OMM" in labels
        else False
    )
    out["omm_domain_state"] = omm_domain_state
    out["membrane_ambiguity"] = np.where(
        omm_domain_state.eq("ER-OMM shared"),
        "ER-OMM shared",
        "",
    )

    records = []

    def _broad_parent(label: str) -> str:
        if label in NUCLEAR_LABELS:
            return "Nucleus"
        if label in CYTOPLASMIC_LABELS:
            return "Cytosol"
        return str(unknown_label)

    for group_id in out.index:
        feasible_labels = [
            label for label in labels if bool(feasible.at[group_id, label])
        ]
        feasible_labels.sort(
            key=lambda label: float(evidence.at[group_id, label]),
            reverse=True,
        )
        supported_labels = [
            label
            for label in feasible_labels
            if bool(supported.at[group_id, label])
        ]
        raw_best = feasible_labels[0] if feasible_labels else str(unknown_label)
        raw_second = (
            feasible_labels[1] if len(feasible_labels) > 1 else ""
        )
        raw_best_score = (
            float(evidence.at[group_id, raw_best])
            if raw_best in evidence.columns
            else np.nan
        )
        raw_second_score = (
            float(evidence.at[group_id, raw_second])
            if raw_second
            else np.nan
        )
        raw_score_gap = (
            raw_best_score - raw_second_score
            if raw_second
            else np.inf
        )
        raw_best_q = (
            float(empirical_q.at[group_id, raw_best])
            if raw_best in empirical_q.columns
            else np.nan
        )
        raw_second_q = (
            float(empirical_q.at[group_id, raw_second])
            if raw_second
            else np.nan
        )
        raw_best_channels = (
            int(channel_support.at[group_id, raw_best])
            if raw_best in channel_support.columns
            else 0
        )
        if not supported_labels:
            exploratory_supported = (
                raw_best != str(unknown_label)
                and bool(
                    specificity_pass.at[group_id, raw_best]
                    if raw_best in specificity_pass.columns
                    else False
                )
                and np.isfinite(raw_best_q)
                and raw_best_q <= exploratory_max_q
                and raw_best_channels
                >= exploratory_min_channel_support
                and raw_score_gap >= exploratory_min_score_gap
            )
            evidence_tier = (
                "Exploratory" if exploratory_supported else "Candidate"
            )
            broad_parent = _broad_parent(raw_best)
            if assignment_mode == "tiered":
                emitted_label = raw_best
            elif assignment_mode == "hierarchical":
                emitted_label = (
                    raw_best if exploratory_supported else broad_parent
                )
            else:
                emitted_label = str(unknown_label)
            records.append(
                {
                    "label": emitted_label,
                    "candidate_label": raw_best,
                    "conservative_label": (
                        broad_parent
                        if assignment_mode == "hierarchical"
                        else str(unknown_label)
                    ),
                    "strict_label": str(unknown_label),
                    "exploratory_label": (
                        raw_best
                        if exploratory_supported
                        else str(unknown_label)
                    ),
                    "annotation_tier": evidence_tier,
                    "primary_label": raw_best,
                    "secondary_label": raw_second,
                    "annotation_state": (
                        (
                            "BroadFallback"
                            if assignment_mode == "hierarchical"
                            and not exploratory_supported
                            else evidence_tier
                        )
                        if assignment_mode in {"tiered", "hierarchical"}
                        else "Unknown"
                    ),
                    "supported_labels": "",
                    "n_supported_labels": 0,
                    "confidence_level": "Low",
                    "low_conf": True,
                    "geometry_consistent": True,
                    "primary_evidence_score": raw_best_score,
                    "secondary_evidence_score": raw_second_score,
                    "evidence_score_gap": raw_score_gap,
                    "primary_empirical_q": raw_best_q,
                    "secondary_empirical_q": raw_second_q,
                }
            )
            continue

        primary = supported_labels[0]
        secondary = supported_labels[1] if len(supported_labels) > 1 else ""
        primary_score = float(evidence.at[group_id, primary])
        secondary_score = (
            float(evidence.at[group_id, secondary]) if secondary else np.nan
        )
        score_gap = (
            primary_score - secondary_score if secondary else np.inf
        )
        state = (
            "Mixed"
            if secondary and score_gap <= float(mixed_score_delta)
            else "Resolved"
        )
        primary_q = float(empirical_q.at[group_id, primary])
        secondary_q = (
            float(empirical_q.at[group_id, secondary])
            if secondary
            else np.nan
        )
        confidence = (
            "High"
            if state == "Resolved"
            and primary_q <= float(subtype_fdr) * 0.5
            else "Medium"
        )
        records.append(
            {
                "label": primary,
                "candidate_label": primary,
                "conservative_label": primary,
                "strict_label": primary,
                "exploratory_label": primary,
                "annotation_tier": "Strict",
                "primary_label": primary,
                "secondary_label": secondary,
                "annotation_state": state,
                "supported_labels": "|".join(supported_labels),
                "n_supported_labels": len(supported_labels),
                "confidence_level": confidence,
                "low_conf": confidence != "High",
                "geometry_consistent": True,
                "primary_evidence_score": primary_score,
                "secondary_evidence_score": secondary_score,
                "evidence_score_gap": score_gap,
                "primary_empirical_q": primary_q,
                "secondary_empirical_q": secondary_q,
            }
        )

    decisions = pd.DataFrame(records, index=out.index)
    for column in decisions.columns:
        out[column] = decisions[column]

    # Keep selected evidence fields coherent for existing diagnostics.
    out["selected_z"] = pd.to_numeric(
        out["primary_evidence_score"],
        errors="coerce",
    )
    out["selected_q"] = pd.to_numeric(
        out["primary_empirical_q"],
        errors="coerce",
    )
    out["subtype_empirical_q"] = out["selected_q"]
    strict_supported = out["strict_label"].ne(str(unknown_label))
    out["subtype_empirical_supported"] = strict_supported
    out["subtype_testable"] = strict_supported
    return out


def _apply_local_compartment_guard(
    labels: pd.Series,
    r_values: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Prevent a domain label from crossing the local nuclear boundary."""
    out = pd.Series(labels, index=labels.index, dtype="object").astype(str)
    r = pd.to_numeric(
        pd.Series(r_values, index=labels.index),
        errors="coerce",
    )
    nuclear_labels = {"Nucleus", "Nucleolus", "Lamina", "Nuclear Pore"}
    cytoplasmic_labels = {"Cytosol", "ER", "OMM"}
    nuclear_conflict = out.isin(nuclear_labels) & (r > 0)
    cytoplasmic_conflict = out.isin(cytoplasmic_labels) & (r < 0)
    conflict = nuclear_conflict | cytoplasmic_conflict
    out.loc[nuclear_conflict] = "Cytosol"
    out.loc[cytoplasmic_conflict] = "Nucleus"
    return out, conflict


def _compose_omm_metaspot_labels(
    domain_labels: pd.Series,
    refined_states: pd.Series,
    domain_gates: pd.Series,
    competitor_labels: Optional[pd.Series] = None,
    *,
    identifiable: bool,
    ambiguous_policy: str = "explicit",
) -> pd.Series:
    """Compose the primary MetaSpot track from domain and local OMM evidence.

    Only a locally high OMM posterior inside a domain-confirmed gate can emit
    OMM.  The pipeline's ``competitor`` policy makes intermediate or locally
    unmeasured confirmed/shared spots fall back to the strongest ER/Cytosol
    alternative so the primary annotation remains in the seven-class label
    space.  The helper's ``explicit`` default preserves compatibility with
    callers that retain ``ER-OMM shared`` as a prediction class.

    Ambiguity remains available in ``omm_refined_state``,
    ``omm_domain_gate``, and ``membrane_ambiguity`` regardless of the primary
    label policy.
    """
    labels = pd.Series(
        domain_labels,
        index=domain_labels.index,
        dtype="object",
    ).astype(str)
    states = pd.Series(
        refined_states,
        index=labels.index,
        dtype="object",
    ).astype(str)
    gates = pd.Series(
        domain_gates,
        index=labels.index,
        dtype="object",
    ).astype(str).str.lower()
    if competitor_labels is None:
        competitors = pd.Series(
            "Cytosol",
            index=labels.index,
            dtype="object",
        )
    else:
        competitors = pd.Series(
            competitor_labels,
            index=labels.index,
            dtype="object",
        ).astype(str)
        competitors = competitors.where(
            competitors.isin(["ER", "Cytosol"]),
            "Cytosol",
        )

    policy = str(ambiguous_policy).strip().lower()
    if policy not in {"competitor", "explicit"}:
        raise ValueError(
            "ambiguous_policy must be one of {'competitor', 'explicit'}"
        )

    def _assign_ambiguous(mask: pd.Series) -> None:
        if policy == "explicit":
            labels.loc[mask] = "ER-OMM shared"
        else:
            labels.loc[mask] = competitors.loc[mask]

    def _finalize() -> pd.Series:
        if policy == "competitor":
            residual_shared = labels.eq("ER-OMM shared")
            labels.loc[residual_shared] = competitors.loc[residual_shared]
        return labels

    confirmed = gates.eq("confirmed")
    shared = gates.eq("shared")
    if not bool(identifiable):
        _assign_ambiguous(confirmed | shared)
        unsupported_omm = labels.eq("OMM")
        labels.loc[unsupported_omm] = competitors.loc[unsupported_omm]
        return _finalize()

    labels.loc[confirmed & states.eq("OMM-supported")] = "OMM"
    _assign_ambiguous(
        confirmed
        & states.isin(
            [
                "ER-OMM shared",
                "InsufficientEvidence",
                "DomainStateMissing",
            ]
        )
    )
    labels.loc[
        confirmed & states.eq("OMM-not-supported")
    ] = competitors.loc[
        confirmed & states.eq("OMM-not-supported")
    ]
    _assign_ambiguous(
        shared & ~states.eq("OMM-not-supported")
    )
    unsupported_omm = labels.eq("OMM") & ~(
        confirmed & states.eq("OMM-supported")
    )
    labels.loc[unsupported_omm] = competitors.loc[unsupported_omm]
    return _finalize()


def _demote_weak_top1(
    annot_df: pd.DataFrame,
    *,
    demote_labels: Sequence[str] = (),
    margin_lt: float = 0.35,
    q1_gt: float = 0.35,
) -> pd.DataFrame:
    """Replace a weak top-1 candidate with top-2 and keep its state coherent.

    ``core_candidate_label`` records the label selected by
    :func:`_rapx_annotate_core`.  This rule is only applied when that selected
    label is still top-1; a core candidate that already switched to top-2 must
    not be switched a second time.  All selected-score fields are updated to
    describe the replacement candidate.
    """
    need = {"label", "top1", "z1", "q1", "top2", "z2", "q2"}
    if not need.issubset(annot_df.columns):
        return annot_df

    out = annot_df.copy()
    if "core_candidate_label" not in out.columns:
        out["core_candidate_label"] = out["label"].astype(str)
    else:
        out["core_candidate_label"] = out["core_candidate_label"].astype(str)
    out["weak_top1_demoted"] = False

    z1 = pd.to_numeric(out["z1"], errors="coerce")
    z2 = pd.to_numeric(out["z2"], errors="coerce")
    q1 = pd.to_numeric(out["q1"], errors="coerce")
    q2 = pd.to_numeric(out["q2"], errors="coerce")
    margin = z1 - z2
    out["margin"] = margin

    top1 = out["top1"].astype(str)
    top2 = out["top2"].astype(str)
    selected = out["label"].astype(str)
    demote = (
        selected.eq(top1)
        & top1.isin(set(map(str, demote_labels)))
        & ((margin < float(margin_lt)) | (q1 > float(q1_gt)))
    )
    if not demote.any():
        return out

    out.loc[demote, "label"] = top2.loc[demote].to_numpy()
    out.loc[demote, "selected_z"] = z2.loc[demote].to_numpy()
    out.loc[demote, "selected_q"] = q2.loc[demote].to_numpy()
    out.loc[demote, "low_conf"] = True
    out.loc[demote, "weak_top1_demoted"] = True
    if "promoted_from_top2" in out.columns:
        out.loc[demote, "promoted_from_top2"] = False
    if "used_top2_due_to_low_conf" in out.columns:
        out.loc[demote, "used_top2_due_to_low_conf"] = False

    for idx in out.index[demote]:
        replacement = str(out.at[idx, "top2"])
        raw_score_col = f"z_comb__{replacement}"
        if raw_score_col in out.columns:
            out.at[idx, "raw_combined_score"] = pd.to_numeric(
                pd.Series([out.at[idx, raw_score_col]]),
                errors="coerce",
            ).iloc[0]
        expression_cols = [
            f"z_mwu__{replacement}",
            f"z_proj__{replacement}",
            f"z_hyp__{replacement}",
        ]
        if all(column in out.columns for column in expression_cols):
            out.at[idx, "channel_support"] = sum(
                int(
                    np.isfinite(pd.to_numeric(out.at[idx, column], errors="coerce"))
                    and pd.to_numeric(out.at[idx, column], errors="coerce") > 0
                )
                for column in expression_cols
            )
    return out


def _adaptive_subtype_support(
    annot_df: pd.DataFrame,
    label: str,
    *,
    subtype_fdr: float = 0.05,
    min_effective_markers: int = 10,
) -> pd.Series:
    """Return empirical-null support for one subtype candidate."""
    label_q_col = f"subtype_empirical_q__{label}"
    label_testable_col = f"subtype_testable__{label}"
    label_count_col = f"effective_marker_count__{label}"
    q_values = pd.to_numeric(
        annot_df.get(
            label_q_col,
            annot_df.get(
                "subtype_empirical_q",
                pd.Series(np.nan, index=annot_df.index),
            ),
        ),
        errors="coerce",
    )
    marker_count = pd.to_numeric(
        annot_df.get(
            label_count_col,
            annot_df.get(
                "effective_marker_count",
                pd.Series(0, index=annot_df.index),
            ),
        ),
        errors="coerce",
    ).fillna(0)
    testable = _coerce_bool_series(
        annot_df.get(
            label_testable_col,
            annot_df.get(
                "subtype_testable",
                pd.Series(False, index=annot_df.index),
            ),
        ),
        default=False,
    )
    return (
        testable
        & marker_count.ge(max(1, int(min_effective_markers)))
        & q_values.le(float(subtype_fdr))
    )


def _refresh_selected_empirical_calibration(
    annot_df: pd.DataFrame,
    label_col: str = "label",
    subtype_fdr: float = 0.05,
) -> pd.DataFrame:
    """Align selected empirical diagnostics after a candidate label changes."""
    out = annot_df.copy()
    if label_col not in out.columns:
        return out
    required_prefixes = (
        "subtype_empirical_p__",
        "subtype_empirical_q__",
        "subtype_testable__",
        "effective_marker_count__",
    )
    if not all(
        any(column.startswith(prefix) for column in out.columns)
        for prefix in required_prefixes
    ):
        return out

    selected_p = []
    selected_q = []
    selected_testable = []
    selected_count = []
    for group_id, label in out[label_col].astype(str).items():
        selected_p.append(out.at[group_id, f"subtype_empirical_p__{label}"])
        selected_q.append(out.at[group_id, f"subtype_empirical_q__{label}"])
        selected_testable.append(out.at[group_id, f"subtype_testable__{label}"])
        selected_count.append(out.at[group_id, f"effective_marker_count__{label}"])
    out["subtype_empirical_p"] = np.asarray(selected_p, dtype=np.float64)
    out["subtype_empirical_q"] = np.asarray(selected_q, dtype=np.float64)
    out["subtype_testable"] = np.asarray(selected_testable, dtype=bool)
    out["effective_marker_count"] = np.asarray(selected_count, dtype=np.int64)
    out["subtype_empirical_supported"] = (
        out["subtype_testable"].astype(bool)
        & out["subtype_empirical_q"].le(float(subtype_fdr))
    )
    return out


def _guard_nuclear_pore_overcall(
    annot_df: pd.DataFrame,
    *,
    enabled: bool = True,
    guard_mode: str = "fixed",
    subtype_fdr: float = 0.05,
    min_effective_markers: int = 10,
    min_margin: float = 0.75,
    non_nuclear_top2_min_margin: float = 5.0,
    max_q: float = 0.10,
    min_channel_support: int = 3,
    fallback_label: str = "Nucleus",
) -> pd.DataFrame:
    """Collapse only weak *selected* Nuclear Pore candidates to Nucleus."""
    out = annot_df.copy()
    out["nuclear_pore_guarded"] = False
    if not enabled:
        return out
    need = {"label", "top1", "top2", "z1", "z2", "q1"}
    if not need.issubset(out.columns):
        return out

    if "candidate_label" not in out.columns:
        out["candidate_label"] = out["label"].astype(str)
    else:
        out["candidate_label"] = out["candidate_label"].astype(str)

    top1 = out["top1"].astype(str)
    top2 = out["top2"].astype(str)
    label = out["label"].astype(str)
    z1 = pd.to_numeric(out["z1"], errors="coerce")
    z2 = pd.to_numeric(out["z2"], errors="coerce")
    raw_margin = z1 - z2
    np_margin = raw_margin.where(top1.eq("Nuclear Pore"), -raw_margin)
    selected_q = pd.to_numeric(
        out.get("selected_q", out["q1"]),
        errors="coerce",
    )
    expression_channels = [
        column
        for column in (
            "z_mwu__Nuclear Pore",
            "z_proj__Nuclear Pore",
            "z_hyp__Nuclear Pore",
        )
        if column in out.columns
    ]
    if len(expression_channels) == 3:
        support = sum(
            pd.to_numeric(out[column], errors="coerce").gt(0).astype(int)
            for column in expression_channels
        )
    else:
        support = pd.to_numeric(
            out.get("channel_support", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0)

    nuclear_alternatives = {"Nucleus", "Nucleolus", "Lamina"}
    np_mask = label.eq("Nuclear Pore")
    mode = str(guard_mode).strip().lower()
    if mode not in {"adaptive", "fixed"}:
        raise ValueError("guard_mode must be 'adaptive' or 'fixed'")
    if mode == "adaptive":
        supported = _adaptive_subtype_support(
            out,
            "Nuclear Pore",
            subtype_fdr=subtype_fdr,
            min_effective_markers=min_effective_markers,
        )
        demote = np_mask & ~supported
        if demote.any():
            out.loc[demote, "nuclear_pore_guarded"] = True
            out.loc[demote, "label"] = str(fallback_label)
            if "low_conf" in out.columns:
                out.loc[demote, "low_conf"] = True
        out["margin"] = raw_margin
        return out

    competitor = top2.where(top1.eq("Nuclear Pore"), top1)
    weak_basic = (
        (np_margin < float(min_margin))
        | (selected_q > float(max_q))
        | (support < int(min_channel_support))
    )
    weak_non_nuclear_runner_up = (
        ~competitor.isin(nuclear_alternatives)
        & (np_margin < float(non_nuclear_top2_min_margin))
    )
    demote = np_mask & (weak_basic | weak_non_nuclear_runner_up)
    if demote.any():
        out.loc[demote, "nuclear_pore_guarded"] = True
        out.loc[demote, "label"] = str(fallback_label)
        if "low_conf" in out.columns:
            out.loc[demote, "low_conf"] = True
    out["margin"] = raw_margin
    return out


def _guard_omm_overcall(
    annot_df: pd.DataFrame,
    *,
    enabled: bool = True,
    guard_mode: str = "fixed",
    subtype_fdr: float = 0.05,
    min_effective_markers: int = 10,
    min_margin: float = 2.0,
    max_q: float = 0.01,
    min_channel_support: int = 3,
    fallback_label: str = "Cytosol",
) -> pd.DataFrame:
    """Collapse weak OMM calls to the broad cytoplasmic fallback.

    Nuclear/cytoplasmic geometry can establish that a group is on the
    cytoplasmic side, but it cannot distinguish OMM from ER or Cytosol.  Use
    the expression-only channel signs plus the within-cytoplasmic score margin
    for that subtype decision.  ``q_comb__OMM`` remains a secondary filter;
    on its own it is not sufficient because the compartment prior contributes
    to the combined score in single-stage mode.
    """
    out = annot_df.copy()
    out["omm_guarded"] = False
    out["omm_margin"] = np.nan
    if not enabled or "label" not in out.columns:
        return out

    if "candidate_label" not in out.columns:
        out["candidate_label"] = out["label"].astype(str)
    else:
        out["candidate_label"] = out["candidate_label"].astype(str)

    label = out["label"].astype(str)
    omm_mask = label.eq("OMM")
    if not omm_mask.any():
        return out
    mode = str(guard_mode).strip().lower()
    if mode not in {"adaptive", "fixed"}:
        raise ValueError("guard_mode must be 'adaptive' or 'fixed'")
    if mode == "adaptive":
        supported = _adaptive_subtype_support(
            out,
            "OMM",
            subtype_fdr=subtype_fdr,
            min_effective_markers=min_effective_markers,
        )
        demote = omm_mask & ~supported
        if demote.any():
            out.loc[demote, "omm_guarded"] = True
            out.loc[demote, "label"] = str(fallback_label)
            if "low_conf" in out.columns:
                out.loc[demote, "low_conf"] = True
        return out

    # Prefer the OMM-vs-ER/Cytosol margin.  The compartment prior is identical
    # for these three labels, so it cancels from this comparison.
    omm_score = pd.to_numeric(
        out.get("z_comb__OMM", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    competitor_columns = [
        column
        for column in ("z_comb__Cytosol", "z_comb__ER")
        if column in out.columns
    ]
    if "z_comb__OMM" in out.columns and competitor_columns:
        competitor_score = pd.concat(
            [pd.to_numeric(out[column], errors="coerce") for column in competitor_columns],
            axis=1,
        ).max(axis=1)
        omm_margin = omm_score - competitor_score
    else:
        z1 = pd.to_numeric(
            out.get("z1", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        z2 = pd.to_numeric(
            out.get("z2", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        top1 = out.get("top1", pd.Series("", index=out.index)).astype(str)
        top2 = out.get("top2", pd.Series("", index=out.index)).astype(str)
        raw_margin = z1 - z2
        omm_margin = pd.Series(np.nan, index=out.index, dtype=float)
        omm_margin.loc[top1.eq("OMM")] = raw_margin.loc[top1.eq("OMM")]
        omm_margin.loc[top2.eq("OMM")] = -raw_margin.loc[top2.eq("OMM")]
    out["omm_margin"] = pd.to_numeric(omm_margin, errors="coerce")

    q_omm = pd.to_numeric(
        out.get(
            "q_comb__OMM",
            out.get(
                "selected_q",
                out.get("q1", pd.Series(np.nan, index=out.index)),
            ),
        ),
        errors="coerce",
    )

    expression_channels = [
        column
        for column in ("z_mwu__OMM", "z_proj__OMM", "z_hyp__OMM")
        if column in out.columns
    ]
    if len(expression_channels) == 3:
        support = sum(
            pd.to_numeric(out[column], errors="coerce").gt(0).astype(int)
            for column in expression_channels
        )
    else:
        support = pd.to_numeric(
            out.get("channel_support", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0)

    weak = (
        ~np.isfinite(out["omm_margin"])
        | (out["omm_margin"] < float(min_margin))
        | ~np.isfinite(q_omm)
        | (q_omm > float(max_q))
        | (support < int(min_channel_support))
    )
    demote = omm_mask & weak
    if demote.any():
        out.loc[demote, "omm_guarded"] = True
        out.loc[demote, "label"] = str(fallback_label)
        if "low_conf" in out.columns:
            out.loc[demote, "low_conf"] = True
    return out


def _guard_lamina_overcall(
    annot_df: pd.DataFrame,
    *,
    group_r_mean: Optional[pd.Series] = None,
    group_d_nuc_norm_mean: Optional[pd.Series] = None,
    enabled: bool = True,
    guard_mode: str = "fixed",
    subtype_fdr: float = 0.05,
    min_effective_markers: int = 10,
    min_margin: float = 2.0,
    non_nuclear_top2_min_margin: float = 3.0,
    max_q: float = 0.01,
    min_channel_support: int = 2,
    boundary_quantile: float = 0.5,
    fallback_label: str = "Nucleus",
) -> pd.DataFrame:
    """Collapse weak or non-peripheral Lamina calls to broad Nucleus.

    The expression guard mirrors the conservative Nuclear Pore guard, while
    the geometry guard uses ``d_nuc_norm_mean`` directly: smaller values are
    closer to the nuclear envelope.  The cutoff is estimated from annotation
    groups on the nuclear side so it adapts across assays and cell shapes.
    """
    out = annot_df.copy()
    out["lamina_guarded"] = False
    out["lamina_geometry_consistent"] = True
    out["lamina_boundary_threshold"] = np.nan
    if not enabled:
        return out

    need = {"label", "top1", "top2", "z1", "z2", "q1"}
    if not need.issubset(out.columns):
        return out

    if "candidate_label" not in out.columns:
        out["candidate_label"] = out["label"].astype(str)
    else:
        out["candidate_label"] = out["candidate_label"].astype(str)

    def _align_group_values(values: Optional[pd.Series]) -> pd.Series:
        if values is None:
            return pd.Series(np.nan, index=out.index, dtype=float)
        source = pd.Series(values).copy()
        source.index = source.index.astype(str)
        source = source.loc[~source.index.duplicated(keep="last")]
        lookup = pd.to_numeric(source, errors="coerce")
        keys = pd.Series(out.index.astype(str), index=out.index)
        return pd.to_numeric(keys.map(lookup), errors="coerce")

    top1 = out["top1"].astype(str)
    top2 = out["top2"].astype(str)
    label = out["label"].astype(str)
    z1 = pd.to_numeric(out["z1"], errors="coerce")
    z2 = pd.to_numeric(out["z2"], errors="coerce")
    raw_margin = z1 - z2
    # A Lamina label selected from top2 has negative expression superiority.
    lamina_margin = raw_margin.where(top1.eq("Lamina"), -raw_margin)
    selected_q = pd.to_numeric(
        out.get("selected_q", out["q1"]),
        errors="coerce",
    )
    support = pd.to_numeric(
        out.get("channel_support", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0)

    lamina_mask = label.eq("Lamina")
    competitor = top2.where(top1.eq("Lamina"), top1)
    cytoplasmic_labels = {"Cytosol", "ER", "OMM"}
    weak_basic = (
        (lamina_margin < float(min_margin))
        | (selected_q > float(max_q))
        | (support < int(min_channel_support))
    )
    weak_non_nuclear_runner_up = (
        competitor.isin(cytoplasmic_labels)
        & (lamina_margin < float(non_nuclear_top2_min_margin))
    )

    r_mean = _align_group_values(group_r_mean)
    d_nuc = _align_group_values(group_d_nuc_norm_mean)
    finite_d_nuc = d_nuc[np.isfinite(d_nuc)]
    if np.isfinite(r_mean).any():
        finite_d_nuc = d_nuc[np.isfinite(d_nuc) & (r_mean <= 0.0)]

    boundary_threshold = np.nan
    if finite_d_nuc.size >= 4:
        q = float(np.clip(boundary_quantile, 0.0, 1.0))
        boundary_threshold = float(np.quantile(finite_d_nuc, q))
        out["lamina_boundary_threshold"] = boundary_threshold

    geometry_inconsistent = pd.Series(False, index=out.index)
    geometry_inconsistent |= np.isfinite(r_mean) & (r_mean > 0.0)
    if np.isfinite(boundary_threshold):
        geometry_inconsistent |= (
            np.isfinite(d_nuc) & (d_nuc > boundary_threshold)
        )
    geometry_inconsistent &= lamina_mask
    out.loc[geometry_inconsistent, "lamina_geometry_consistent"] = False
    # Lamina has a dedicated envelope-distance check above.  Use that result
    # for the downstream confidence policy instead of applying a second,
    # potentially contradictory generic r-quantile gate.
    if "geometry_consistent" in out.columns:
        out.loc[lamina_mask, "geometry_consistent"] = (
            ~geometry_inconsistent.loc[lamina_mask]
        ).to_numpy()

    mode = str(guard_mode).strip().lower()
    if mode not in {"adaptive", "fixed"}:
        raise ValueError("guard_mode must be 'adaptive' or 'fixed'")
    if mode == "adaptive":
        supported = _adaptive_subtype_support(
            out,
            "Lamina",
            subtype_fdr=subtype_fdr,
            min_effective_markers=min_effective_markers,
        )
        demote = lamina_mask & (~supported | geometry_inconsistent)
    else:
        demote = lamina_mask & (
            weak_basic | weak_non_nuclear_runner_up | geometry_inconsistent
        )
    if demote.any():
        out.loc[demote, "lamina_guarded"] = True
        out.loc[demote, "label"] = str(fallback_label)
        if "low_conf" in out.columns:
            out.loc[demote, "low_conf"] = True
    out["margin"] = raw_margin
    return out


def build_specific_marker_sets(
    gold_mat: pd.DataFrame,
    target: int = 600,
    min_weight: float = 0.05,
    min_margin: float = 0.0,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]], List[str]]:
    """Build non-overlapping marker sets from each gene's strongest region."""
    W = gold_mat.copy()
    W.columns = [
        PRETTY.get(_canon_region_col(c), str(c).replace("_", " "))
        for c in W.columns
    ]
    W = W.T.groupby(level=0, sort=False).max().T
    W = W.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    regions = list(W.columns)
    values = W.to_numpy(dtype=np.float32, copy=False)

    if values.shape[1] == 0:
        return {}, {}, []

    winner = values.argmax(axis=1)
    winner_score = values[np.arange(values.shape[0]), winner]
    if values.shape[1] > 1:
        second_score = np.partition(values, -2, axis=1)[:, -2]
    else:
        second_score = np.zeros(values.shape[0], dtype=np.float32)
    margin = winner_score - second_score
    genes = W.index.astype(str).to_numpy()

    sets: Dict[str, List[str]] = {}
    weights: Dict[str, Dict[str, float]] = {}
    for region_idx, region in enumerate(regions):
        eligible = (
            (winner == region_idx)
            & (winner_score >= float(min_weight))
            & (margin >= float(min_margin))
        )
        rows = np.flatnonzero(eligible)
        if rows.size:
            score = winner_score[rows] * margin[rows]
            order = np.argsort(-score, kind="mergesort")[: max(1, int(target))]
            rows = rows[order]
            score = score[order]
            if score.max(initial=0.0) > 0:
                score = score / score.max()
            region_genes = genes[rows].tolist()
            sets[region] = region_genes
            weights[region] = dict(zip(region_genes, score.astype(float)))
        else:
            sets[region] = []
            weights[region] = {}

    return sets, weights, regions


# ---------- QUQ 诊断富集可视化 ----------
def improved_enrichment_visualization(
    adata,
    annot_df,
    sets,
    produced_label_col='subcellular_annotation',
    p_adj_max=0.1,
    lfc_min=0.0,
    selection_mode='quq',
    q_per_group=800,
    min_freq_abs=2,
    min_freq_frac=0.2,
    balance_labels=True,
    target_per_label=3000,
    min_per_label=500,
    max_per_label=20000,
    min_query_fallback=150,
    fallback_mode='weighted_union',
    show_tables=True
):
    import scanpy as sc

    sns.set(style='whitegrid', context='talk')
    diag_group_col = "leiden"
    cleanup_diag_group = False
    if "compartment_pred" in adata.obs.columns:
        diag_group_col = "_rapx_diag_group"
        diag_groups = (
            adata.obs["leiden"].astype(str)
            + "|"
            + adata.obs["compartment_pred"].astype(str)
        )
        group_counts = diag_groups.value_counts()
        rare_groups = group_counts[group_counts < 2].index
        for rare_group in rare_groups:
            rare_mask = diag_groups.eq(rare_group)
            compartment = adata.obs.loc[rare_mask, "compartment_pred"].astype(str).iloc[0]
            candidates = group_counts[
                (group_counts >= 2)
                & group_counts.index.to_series().str.endswith(f"|{compartment}")
            ]
            if len(candidates):
                diag_groups.loc[rare_mask] = str(candidates.idxmax())
        adata.obs[diag_group_col] = pd.Categorical(diag_groups)
        cleanup_diag_group = True

    def _map_var_upper(adata):
        vn = adata.var_names.astype(str)
        up = vn.str.upper()
        mp = {}
        for i,u in enumerate(up):
            if u not in mp:
                mp[u] = vn[i]
        return mp

    def _genes_to_var(genes, up2var):
        out = []
        for g in genes:
            if g is None:
                continue
            v = up2var.get(str(g).upper())
            if v is not None:
                out.append(v)
        return set(out)

    sc.tl.rank_genes_groups(
        adata,
        groupby=diag_group_col,
        method='wilcoxon',
        use_raw=False,
    )
    deg_df = sc.get.rank_genes_groups_df(adata, group=None).copy()
    deg_df['names'] = deg_df['names'].astype(str)
    for c in ['pvals_adj','logfoldchanges','scores']:
        deg_df[c] = pd.to_numeric(deg_df[c], errors='coerce')
    deg_df = deg_df[deg_df['logfoldchanges'] > 0]
    if p_adj_max is not None:
        deg_df = deg_df[deg_df['pvals_adj'] <= float(p_adj_max)]
    if lfc_min is not None:
        deg_df = deg_df[deg_df['logfoldchanges'] >= float(lfc_min)]

    groups_in_deg = sorted(deg_df['group'].astype(str).unique().tolist())
    top_by_group = {}
    for g in groups_in_deg:
        sub = deg_df[deg_df['group'].astype(str) == g].copy()
        sub = sub.sort_values(['pvals_adj','logfoldchanges','scores'],
                              ascending=[True, False, False], kind='mergesort')
        if q_per_group is not None:
            sub = sub.head(int(q_per_group))
        top_by_group[g] = sub

    print(f"Avg genes per cluster (Top-q): {np.mean([len(v) for v in top_by_group.values()]) if top_by_group else 0:.0f}")

    if cleanup_diag_group and produced_label_col in adata.obs.columns:
        grp2lab = (
            adata.obs.groupby(diag_group_col, observed=True)[produced_label_col]
            .agg(lambda s: s.astype(str).value_counts().idxmax())
            .astype(str)
            .to_dict()
        )
    else:
        grp2lab = annot_df['label'].astype(str).to_dict()

    label_cluster_counts = Counter(grp2lab.values())
    if produced_label_col in adata.obs.columns:
        final_object_counts = (
            adata.obs[produced_label_col]
            .astype(str)
            .value_counts()
            .to_dict()
        )
    else:
        final_object_counts = {}
    label_gene_hits = defaultdict(Counter)
    for g, sub in top_by_group.items():
        lab = grp2lab.get(str(g), "Unknown")
        if lab == "Unknown":
            continue
        label_gene_hits[lab].update(sub['names'].astype(str).tolist())

    genes_by_label = {}
    weights_by_label = {}

    def _build_union(lab):
        picked = list(label_gene_hits.get(lab, Counter()).keys())
        return set(picked), pd.Series(1.0, index=picked, dtype=float)

    def _build_weighted_union(lab):
        agg = []
        for g, sub in top_by_group.items():
            if grp2lab.get(str(g), 'Unknown') != lab:
                continue
            s = sub.copy()
            s['w'] = (-np.log10(s['pvals_adj'].clip(lower=1e-300)) + s['logfoldchanges'].clip(lower=0))
            agg.append(s[['names','w']])
        if len(agg):
            W = pd.concat(agg).groupby('names')['w'].sum().sort_values(ascending=False)
            order = W.index.tolist()
            return set(order), W
        else:
            return _build_union(lab)

    def _cap_balance(gset, wser):
        if not balance_labels:
            return gset, wser
        if len(gset) == 0:
            return gset, wser
        order = wser.sort_values(ascending=False).index
        cap = max(min_per_label, min(len(order), target_per_label))
        order = order[:min(cap, max_per_label)]
        return set(order), wser.reindex(order).fillna(0.0)

    for lab in label_gene_hits.keys():
        if selection_mode.lower() == 'union':
            gset, wser = _build_union(lab)
        elif selection_mode.lower() == 'weighted_union':
            gset, wser = _build_weighted_union(lab)
        else:
            n_lab = max(1, int(label_cluster_counts.get(lab, 1)))
            thr = max(int(min_freq_abs), int(math.ceil(min_freq_frac * n_lab)))
            picked = [g for g, c in label_gene_hits[lab].items() if c >= thr]
            wser = pd.Series({g: float(label_gene_hits[lab][g]) for g in picked}, dtype=float) if picked else pd.Series(dtype=float)
            gset = set(picked)
        gset, wser = _cap_balance(gset, wser)
        genes_by_label[lab] = gset
        weights_by_label[lab] = wser

    did_fallback = False
    for lab in list(genes_by_label.keys()):
        if 0 < len(genes_by_label[lab]) < int(min_query_fallback):
            did_fallback = True
            if fallback_mode == 'weighted_union':
                gset, wser = _build_weighted_union(lab)
            else:
                gset, wser = _build_union(lab)
            gset, wser = _cap_balance(gset, wser)
            genes_by_label[lab] = gset
            weights_by_label[lab] = wser
    print("Query sizes by label:", {k: len(v) for k, v in genes_by_label.items()})
    if did_fallback:
        print(f"[Note] Applied fallback for labels with query < {min_query_fallback} using mode='{fallback_mode}'")

    up2var = _map_var_upper(adata)
    background_genes = set(up2var.values())
    gold_sets_diag = {lab: (_genes_to_var(gset, up2var) & background_genes) for lab, gset in sets.items()}

    def _star(fdr):
        return '***' if fdr<=1e-3 else '**' if fdr<=1e-2 else '*' if fdr<=0.05 else 'ns'

    rows = []
    regions_all = [r for r in PRETTY_ORDER if r in sets.keys()]
    for lab in regions_all:
        gold = gold_sets_diag.get(lab, set()) & background_genes
        query = genes_by_label.get(lab, set())

        if len(query) and (next(iter(query)) not in background_genes):
            query = _genes_to_var(query, up2var)
        query = query & background_genes

        A = len(query & gold)
        B = len(query - gold)
        C = len(gold - query)
        D = len(background_genes - (query | gold))
        N = A + B + C + D

        base_row = {
            'Label': lab,
            'Final_objects': int(final_object_counts.get(lab, 0)),
            'Annotation_groups': int(label_cluster_counts.get(lab, 0)),
            'Overlap': int(A),
            'Query_genes': int(A+B),
            'Gold_genes': int(A+C),
            'Background': int(N),
        }
        if (A + B) == 0:
            status = (
                'No final-label objects'
                if base_row['Final_objects'] == 0
                else 'Insufficient positive DE genes'
            )
            rows.append({
                **base_row,
                'Expected': np.nan,
                'Fold_enrichment': np.nan,
                'log2FE': np.nan,
                'Odds_ratio': np.nan,
                'log2OR': np.nan,
                'log2OR_CI_lo': np.nan,
                'log2OR_CI_hi': np.nan,
                'P_value_hg': np.nan,
                'P_value_fisher': np.nan,
                'Testable': False,
                'Status': status,
            })
            continue
        if (A + C) == 0:
            rows.append({
                **base_row,
                'Expected': np.nan,
                'Fold_enrichment': np.nan,
                'log2FE': np.nan,
                'Odds_ratio': np.nan,
                'log2OR': np.nan,
                'log2OR_CI_lo': np.nan,
                'log2OR_CI_hi': np.nan,
                'P_value_hg': np.nan,
                'P_value_fisher': np.nan,
                'Testable': False,
                'Status': 'No gold genes in dataset',
            })
            continue

        a,b,c,d = A+0.5, B+0.5, C+0.5, D+0.5
        p1 = a / (a+b); p0 = c / (c+d)
        fe = (p1 / p0) if p0 > 0 else np.inf
        log2fe = np.log2(fe) if fe > 0 else -np.inf

        _, p_fisher = fisher_exact([[A, B],[C, D]], alternative='greater')
        p_hg = float(hypergeom.sf(max(A,0)-1, N, (A+C), (A+B))) if (A+B)>0 and (A+C)>0 else 1.0

        OR = (a*d)/(b*c)
        var_logOR = (1/a)+(1/b)+(1/c)+(1/d)
        se_logOR = math.sqrt(var_logOR)
        z = 1.96
        logOR = math.log(OR)
        lo, hi = logOR - z*se_logOR, logOR + z*se_logOR
        log2OR = logOR/math.log(2); lo2, hi2 = lo/math.log(2), hi/math.log(2)

        rows.append({
            **base_row,
            'Expected': float((A+B)*(A+C)/max(N,1)),
            'Fold_enrichment': float(fe),
            'log2FE': float(log2fe),
            'Odds_ratio': float(OR),
            'log2OR': float(log2OR),
            'log2OR_CI_lo': float(lo2),
            'log2OR_CI_hi': float(hi2),
            'P_value_hg': float(p_hg),
            'P_value_fisher': float(p_fisher),
            'Testable': True,
            'Status': 'OK',
        })

    enrich_df = pd.DataFrame(rows).set_index('Label')
    if len(enrich_df):
        enrich_df['FDR'] = np.nan
        enrich_df['Significance'] = 'NA'
        testable = (
            enrich_df['Testable'].astype(bool)
            & enrich_df['P_value_hg'].notna()
        )
        if testable.any():
            _, fdr, _, _ = multipletests(
                enrich_df.loc[testable, 'P_value_hg'].to_numpy(),
                method='fdr_bh',
            )
            enrich_df.loc[testable, 'FDR'] = fdr
            enrich_df.loc[testable, 'Significance'] = (
                enrich_df.loc[testable, 'FDR'].apply(_star)
            )
    else:
        enrich_df['FDR'] = []
        enrich_df['Significance'] = []

    regions = [r for r in PRETTY_ORDER if r in enrich_df.index]
    E = enrich_df.loc[regions]
    T = E.loc[E['Testable'].astype(bool)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios':[1.1,0.9]})

    def _show_no_testable(ax, title):
        ax.text(
            0.5,
            0.5,
            'No labels were testable',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.set_axis_off()

    ax = axes[0,0]
    if len(T):
        y = np.arange(len(T))[::-1]
        ax.hlines(y, xmin=T['log2OR_CI_lo'].values[::-1], xmax=T['log2OR_CI_hi'].values[::-1],
                  colors=[SIG_COLOR[s] for s in T['Significance'].values[::-1]], lw=3, alpha=0.9)
        ax.plot(T['log2OR'].values[::-1], y, 'o', color='black', ms=6)
        for i,(lab,r) in enumerate(T.iloc[::-1].iterrows()):
            ax.text(r['log2OR']+0.08, y[i], f"{r['log2FE']:.2f} FE {r['Significance']}", va='center', fontsize=10)
        ax.axvline(0, color='k', ls='--', alpha=0.6)
        ax.set_yticks(y); ax.set_yticklabels(T.index[::-1])
        ax.set_xlabel('log2(OR) with 95% CI'); ax.set_title('Enrichment')
    else:
        _show_no_testable(ax, 'Enrichment')

    ax = axes[0,1]
    if len(T):
        x = T['log2FE'].values; yv = -np.log10(T['FDR'].clip(lower=1e-300).values)
        sizes = 60 + 30*np.sqrt(np.clip(T['Overlap'].values,1,None))
        colors = [SIG_COLOR[s] for s in T['Significance']]
        ax.scatter(x, yv, s=sizes, c=colors, edgecolors='black', alpha=0.9)
        for lab, xv, yvi in zip(T.index, x, yv):
            ax.annotate(lab, (xv, yvi), xytext=(6,4), textcoords='offset points', fontsize=10)
        ax.axvline(0, color='k', ls='--', alpha=0.5)
        ax.axhline(-np.log10(0.05), color='red', ls='--', alpha=0.5, label='FDR=0.05')
        ax.set_xlabel('log2(FE)'); ax.set_ylabel('-log10(FDR)'); ax.set_title('Significance Volcano')
    else:
        _show_no_testable(ax, 'Significance Volcano')

    ax = axes[1,0]
    if len(T):
        ax.scatter(T['log2FE'], T.index, s=60+30*np.sqrt(np.clip(T['Overlap'],1,None)),
                   c=[SIG_COLOR[s] for s in T['Significance']], edgecolors='black', alpha=0.9)
        ax.axvline(0, color='k', ls='--', alpha=0.5)
        ax.set_xlabel('log2(FE)'); ax.set_ylabel(''); ax.set_title('Bubble: Overlap size & significance')
    else:
        _show_no_testable(ax, 'Bubble: Overlap size & significance')

    ax = axes[1,1]
    width = 0.35
    pos = np.arange(len(E))
    q = E['Query_genes'].values; g = E['Gold_genes'].values
    exp = E['Expected'].values
    ax.barh(pos-width/2, q, height=width, label='Query', color='#4C72B0', alpha=0.85)
    ax.barh(pos+width/2, g, height=width, label='Gold', color='#DD8452', alpha=0.85)
    for i,(ev) in enumerate(exp):
        ax.plot(ev, i, 'k|', ms=14)
    ax.set_yticks(pos); ax.set_yticklabels(E.index)
    ax.legend(loc='lower right'); ax.set_title('Power diagnostics (| marks Expected overlap)')
    plt.tight_layout(); plt.show()

    if show_tables:
        try:
            from IPython.display import display as _display
            print("\n[Enrichment table]")
            _display(E.round(4))
        except Exception:
            print("\n[Enrichment table]\n", E.round(4).to_string())

    if cleanup_diag_group:
        adata.obs.drop(columns=[diag_group_col], inplace=True, errors="ignore")
    return enrich_df


# ---------- 质量总览 ----------
def plot_annotation_quality_summary(adata, annot_df, produced_label_col='subcellular_annotation'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    label_counts = adata.obs[produced_label_col].value_counts()
    colors = plt.cm.Set3(np.arange(len(label_counts)))
    axes[0,0].pie(label_counts.values, labels=label_counts.index,
                  autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0,0].set_title('Meta-domain Label Distribution', fontsize=14, fontweight='bold')

    conf_counts = adata.obs['subcell_low_conf'].value_counts()
    conf_labels = ['High Confidence' if not x else 'Low Confidence' for x in conf_counts.index]
    axes[0,1].pie(conf_counts.values, labels=conf_labels, autopct='%1.1f%%',
                  colors=['lightgreen', 'lightcoral'], startangle=90)
    axes[0,1].set_title('Annotation Confidence Distribution', fontsize=14, fontweight='bold')

    z_column = (
        'subcell_selected_z'
        if 'subcell_selected_z' in adata.obs.columns
        else 'subcell_z1'
    )
    z_scores = pd.to_numeric(adata.obs[z_column], errors='coerce').dropna()
    axes[1,0].hist(z_scores, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    z_median = float(z_scores.median()) if len(z_scores) else np.nan
    if np.isfinite(z_median):
        axes[1,0].axvline(x=z_median, color='red', linestyle='--', label=f'Median: {z_median:.2f}')
        axes[1,0].legend()
    axes[1,0].set_xlabel('Selected candidate z-score'); axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Candidate Score Distribution', fontsize=14, fontweight='bold')
    axes[1,0].grid(alpha=0.3)

    if "subcell_primary_empirical_q" in adata.obs.columns:
        q_column = "subcell_primary_empirical_q"
        q_axis_label = "-log₁₀(label-wise empirical q)"
        q_summary_label = "Resolved primary labels with empirical q≤0.05"
    else:
        q_column = (
            'subcell_selected_q'
            if 'subcell_selected_q' in adata.obs.columns
            else 'subcell_q1'
        )
        q_axis_label = "-log₁₀(heuristic q-score)"
        q_summary_label = "Candidate ratio with heuristic q-score≤0.05"
    q_values = pd.to_numeric(adata.obs[q_column], errors='coerce').dropna()
    log_q = -np.log10(q_values + 1e-10) if len(q_values) else pd.Series([], dtype=float)
    axes[1,1].hist(log_q, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1,1].axvline(x=-np.log10(0.05), color='red', linestyle='--', label='q=0.05')
    axes[1,1].set_xlabel(q_axis_label); axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Candidate Q-score Distribution', fontsize=14, fontweight='bold')
    axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

    plt.tight_layout(); plt.show()

    print(f"\n=== Annotation Quality Summary ===")
    print(f"Total meta-domains: {adata.n_obs:,}")
    if {"leiden", "compartment_pred"}.issubset(adata.obs.columns):
        annotation_groups = (
            adata.obs[["leiden", "compartment_pred"]]
            .astype(str)
            .drop_duplicates()
            .shape[0]
        )
    else:
        annotation_groups = len(annot_df)
    print(f"Total annotation groups: {annotation_groups}")
    low_conf_values = _coerce_bool_series(
        adata.obs["subcell_low_conf"],
        default=True,
    )
    high_conf_ratio = (~low_conf_values).mean()
    print(f"High confidence annotation ratio: {high_conf_ratio:.1%}")
    if np.isfinite(z_median):
        print(f"Z-score median: {z_median:.2f}")
    print(
        f"{q_summary_label}: "
        f"{(q_values <= 0.05).mean():.1%}"
    )
    if "subcell_subtype_testable" in adata.obs.columns:
        testable = _coerce_bool_series(
            adata.obs["subcell_subtype_testable"],
            default=False,
        )
        supported = _coerce_bool_series(
            adata.obs.get(
                "subcell_subtype_empirical_supported",
                pd.Series(False, index=adata.obs.index),
            ),
            default=False,
        )
        print(
            "Empirically testable selected subtypes: "
            f"{int(testable.sum()):,}/{adata.n_obs:,}; "
            f"FDR-supported: {int(supported.sum()):,}/{adata.n_obs:,}"
        )
    print("\nFinal output label distribution:")
    for label, count in label_counts.items():
        pct = count / adata.n_obs * 100
        print(f"  {label}: {count:,} meta-domains ({pct:.1f}%)")

    strict_col = "subcellular_annotation_strict"
    if strict_col in adata.obs.columns:
        print("\nStrict FDR-supported label distribution:")
        strict_counts = adata.obs[strict_col].value_counts()
        for label, count in strict_counts.items():
            pct = count / adata.n_obs * 100
            print(f"  {label}: {count:,} meta-domains ({pct:.1f}%)")
    tier_col = "subcellular_annotation_tier"
    if tier_col in adata.obs.columns:
        print("\nEvidence tier distribution:")
        for tier, count in adata.obs[tier_col].value_counts(
            sort=False
        ).items():
            print(f"  {tier}: {count:,} meta-domains")

    if "subcellular_confidence" in adata.obs.columns:
        print("\nConfidence tier distribution:")
        for level, count in adata.obs["subcellular_confidence"].value_counts(
            sort=False
        ).items():
            print(f"  {level}: {count:,} meta-domains")
        confidence_by_label = pd.crosstab(
            adata.obs[produced_label_col].astype(str),
            adata.obs["subcellular_confidence"].astype(str),
        ).reindex(columns=["High", "Medium", "Low"], fill_value=0)
        print("\nDetailed labels by confidence:")
        print(confidence_by_label.to_string())
        no_high = confidence_by_label.index[
            confidence_by_label["High"].eq(0)
        ].tolist()
        no_high = [label for label in no_high if label != "Unknown"]
        if no_high:
            print(
                "\n[WARN] Output labels with no high-confidence MetaDomains: "
                + ", ".join(no_high)
            )
    if "subcell_core_candidate_label" in adata.obs.columns:
        print("\nCore candidate subtype distribution:")
        for label, count in (
            adata.obs["subcell_core_candidate_label"]
            .astype(str)
            .value_counts()
            .items()
        ):
            print(f"  {label}: {count:,} meta-domains")
    if "subcellular_candidate" in adata.obs.columns:
        print("\nRaw evidence candidate subtype distribution:")
        for label, count in (
            adata.obs["subcellular_candidate"]
            .astype(str)
            .value_counts()
            .items()
        ):
            print(f"  {label}: {count:,} meta-domains")
        candidate_values = adata.obs["subcellular_candidate"].astype(str)
        final_values = adata.obs[produced_label_col].astype(str)
        fallback_mask = candidate_values.ne(final_values)
        if fallback_mask.any():
            fallback_counts = (
                pd.DataFrame(
                    {
                        "candidate": candidate_values[fallback_mask],
                        "final": final_values[fallback_mask],
                    }
                )
                .value_counts()
                .sort_values(ascending=False)
            )
            print("\nEvidence-aware candidate fallbacks:")
            for (candidate, final), count in fallback_counts.items():
                print(f"  {candidate} -> {final}: {count:,} meta-domains")
    if "subcell_weak_top1_demoted" in adata.obs.columns:
        weak_demoted = _coerce_bool_series(
            adata.obs["subcell_weak_top1_demoted"],
            default=False,
        )
        if weak_demoted.any():
            print(
                "\nWeak-top1 replacement applied: "
                f"{int(weak_demoted.sum()):,} meta-domains"
            )
    if "subcell_lamina_guarded" in adata.obs.columns:
        lamina_guarded = _coerce_bool_series(
            adata.obs["subcell_lamina_guarded"],
            default=False,
        )
        guarded_count = int(lamina_guarded.sum())
        if guarded_count:
            print(
                "\nLamina guard applied: "
                f"{guarded_count:,} meta-domains"
            )
    if "subcell_omm_guarded" in adata.obs.columns:
        omm_guarded = _coerce_bool_series(
            adata.obs["subcell_omm_guarded"],
            default=False,
        )
        guarded_count = int(omm_guarded.sum())
        if guarded_count:
            print(
                "\nOMM guard applied: "
                f"{guarded_count:,} meta-domains"
            )
    if "subcell_compartment_rescued" in adata.obs.columns:
        rescued = _coerce_bool_series(
            adata.obs["subcell_compartment_rescued"],
            default=False,
        )
        rescued_count = int(rescued.sum())
        if rescued_count:
            print(
                "\nSingle-stage geometry rescues: "
                f"{rescued_count:,} meta-domains "
                f"({rescued_count / adata.n_obs * 100:.1f}%)"
            )

    conservative_col = "subcellular_annotation_conservative"
    if conservative_col in adata.obs.columns:
        print("\nConservative label distribution:")
        conservative_counts = adata.obs[conservative_col].value_counts()
        for label, count in conservative_counts.items():
            pct = count / adata.n_obs * 100
            print(f"  {label}: {count:,} meta-domains ({pct:.1f}%)")


# ---------- 主函数（支持两阶段标注）----------
def run_rapx(
    adata, gold_df=None, produced_label_col='subcellular_annotation',
    plot_umap=True, plot_diag_enrich=False,
    geom_before_top2=True,
    top2_switch_z_floor_ratio=0.9,
    top2_switch_q_ceiling_ratio=2.0,
    top2_switch_z_not_much_worse=0.25,
    use_two_stage=False,
    use_spatial_channel=False,

    # ✅ 新增：更通用的“特异性优先 set”
    set_target_size=600,          # 每个区域最多保留多少特异基因（越小越“尖锐”）
    set_min_weight=0.05,
    set_min_margin=0.0,
    gold_column_normalize=True,
    min_channel_support=2,
    enforce_geometry_consistency=True,
    geometry_quantile=0.4,
    geometry_channel_weight=0.25,
    guard_mode="adaptive",
    subtype_fdr=0.05,
    min_effective_markers=10,
    null_permutations=500,
    null_seed=0,
    single_stage_compartment_score_weight=6.0,
    single_stage_compartment_score_tau=0.2,
    single_stage_compartment_guard=False,
    single_stage_compartment_min_abs_r=0.05,
    single_stage_compartment_max_score_gap=8.5,
    single_stage_compartment_min_z=1.0,
    single_stage_compartment_max_q=0.35,
    single_stage_compartment_min_channel_support=1,

    # ✅ 新增：轻量去偏置（跨数据集通用）
    demote_if_margin_lt=0.35,     # z1 - z2 < 该阈值，认为 top1 证据不稳
    demote_if_q1_gt=0.35,         # q1 > 该阈值，认为富集不稳
    # Disabled by default: a weak label should fall back to its broad parent,
    # not be replaced by an arbitrary runner-up. Kept as an opt-in legacy rule.
    demote_labels=(),
    nuclear_pore_guard=True,
    nuclear_pore_min_margin=0.75,
    nuclear_pore_non_nuclear_top2_min_margin=5.0,
    nuclear_pore_max_q=0.10,
    nuclear_pore_min_channel_support=3,
    nuclear_pore_fallback_label="Nucleus",
    lamina_guard=True,
    lamina_min_margin=1.0,
    lamina_non_nuclear_top2_min_margin=3.0,
    lamina_max_q=0.05,
    lamina_min_channel_support=2,
    lamina_boundary_quantile=0.5,
    lamina_fallback_label="Nucleus",
    omm_guard=True,
    omm_min_margin=2.0,
    omm_max_q=0.01,
    omm_min_channel_support=3,
    omm_fallback_label="Cytosol",
    low_conf_policy="supported",
    low_conf_relaxed_z=1.25,
    low_conf_relaxed_q=0.30,
    decision_mode="legacy",
    independent_mixed_score_delta=0.75,
    independent_unknown_label="Unknown",
    independent_geometry_min_abs_r=0.05,
    independent_assignment_mode="strict",
    independent_exploratory_max_q=0.30,
    independent_exploratory_min_channel_support=2,
    independent_exploratory_min_score_gap=0.50,
    independent_subtype_parent_margin=0.0,
    independent_omm_competitor_margin=0.0,
):
    import numpy as np
    import pandas as pd
    import scanpy as sc
    try:
        annotation_cfg = (load_params_yaml() or {}).get("annotation", {}) or {}
        low_conf_policy = str(
            annotation_cfg.get("low_conf_policy", low_conf_policy)
        )
        low_conf_relaxed_z = float(
            annotation_cfg.get("low_conf_relaxed_z", low_conf_relaxed_z)
        )
        low_conf_relaxed_q = float(
            annotation_cfg.get("low_conf_relaxed_q", low_conf_relaxed_q)
        )
        decision_mode = str(
            annotation_cfg.get("decision_mode", decision_mode)
        ).strip().lower()
        if decision_mode not in {"legacy", "independent"}:
            raise ValueError(
                "annotation.decision_mode must be legacy or independent"
            )
        independent_mixed_score_delta = max(
            0.0,
            float(
                annotation_cfg.get(
                    "independent_mixed_score_delta",
                    independent_mixed_score_delta,
                )
            ),
        )
        independent_unknown_label = str(
            annotation_cfg.get(
                "independent_unknown_label",
                independent_unknown_label,
            )
        )
        independent_geometry_min_abs_r = max(
            0.0,
            float(
                annotation_cfg.get(
                    "independent_geometry_min_abs_r",
                    independent_geometry_min_abs_r,
                )
            ),
        )
        independent_assignment_mode = str(
            annotation_cfg.get(
                "independent_assignment_mode",
                independent_assignment_mode,
            )
        ).strip().lower()
        if independent_assignment_mode not in {
            "strict",
            "tiered",
            "hierarchical",
        }:
            raise ValueError(
                "annotation.independent_assignment_mode must be "
                "strict, tiered, or hierarchical"
            )
        independent_exploratory_max_q = float(
            np.clip(
                float(
                    annotation_cfg.get(
                        "independent_exploratory_max_q",
                        independent_exploratory_max_q,
                    )
                ),
                0.0,
                1.0,
            )
        )
        independent_exploratory_min_channel_support = max(
            0,
            int(
                annotation_cfg.get(
                    "independent_exploratory_min_channel_support",
                    independent_exploratory_min_channel_support,
                )
            ),
        )
        independent_exploratory_min_score_gap = max(
            0.0,
            float(
                annotation_cfg.get(
                    "independent_exploratory_min_score_gap",
                    independent_exploratory_min_score_gap,
                )
            ),
        )
        independent_subtype_parent_margin = max(
            0.0,
            float(
                annotation_cfg.get(
                    "independent_subtype_parent_margin",
                    independent_subtype_parent_margin,
                )
            ),
        )
        independent_omm_competitor_margin = max(
            0.0,
            float(
                annotation_cfg.get(
                    "independent_omm_competitor_margin",
                    independent_omm_competitor_margin,
                )
            ),
        )
        set_target_size = int(
            annotation_cfg.get("marker_set_target", set_target_size)
        )
        set_min_weight = float(
            annotation_cfg.get("marker_min_weight", set_min_weight)
        )
        set_min_margin = float(
            annotation_cfg.get("marker_min_margin", set_min_margin)
        )
        gold_column_normalize = bool(
            annotation_cfg.get(
                "gold_column_normalize",
                gold_column_normalize,
            )
        )
        min_channel_support = int(
            annotation_cfg.get(
                "min_channel_support",
                min_channel_support,
            )
        )
        enforce_geometry_consistency = bool(
            annotation_cfg.get(
                "enforce_geometry_consistency",
                enforce_geometry_consistency,
            )
        )
        geometry_quantile = float(
            annotation_cfg.get(
                "geometry_quantile",
                geometry_quantile,
            )
        )
        geometry_channel_weight = float(
            annotation_cfg.get(
                "geometry_channel_weight",
                geometry_channel_weight,
            )
        )
        guard_mode = str(
            annotation_cfg.get("guard_mode", guard_mode)
        ).strip().lower()
        if guard_mode not in {"adaptive", "fixed"}:
            raise ValueError("annotation.guard_mode must be adaptive or fixed")
        subtype_fdr = float(
            np.clip(
                float(annotation_cfg.get("subtype_fdr", subtype_fdr)),
                0.0,
                1.0,
            )
        )
        min_effective_markers = max(
            1,
            int(
                annotation_cfg.get(
                    "min_effective_markers",
                    min_effective_markers,
                )
            ),
        )
        null_permutations = max(
            1,
            int(
                annotation_cfg.get(
                    "null_permutations",
                    null_permutations,
                )
            ),
        )
        null_seed = int(annotation_cfg.get("null_seed", null_seed))
        single_stage_compartment_score_weight = max(
            0.0,
            float(
                annotation_cfg.get(
                    "single_stage_compartment_score_weight",
                    single_stage_compartment_score_weight,
                )
            ),
        )
        single_stage_compartment_score_tau = max(
            1e-6,
            abs(
                float(
                    annotation_cfg.get(
                        "single_stage_compartment_score_tau",
                        single_stage_compartment_score_tau,
                    )
                )
            ),
        )
        single_stage_compartment_guard = bool(
            annotation_cfg.get(
                "single_stage_compartment_guard",
                single_stage_compartment_guard,
            )
        )
        single_stage_compartment_min_abs_r = float(
            annotation_cfg.get(
                "single_stage_compartment_min_abs_r",
                single_stage_compartment_min_abs_r,
            )
        )
        single_stage_compartment_max_score_gap = float(
            annotation_cfg.get(
                "single_stage_compartment_max_score_gap",
                single_stage_compartment_max_score_gap,
            )
        )
        single_stage_compartment_min_z = float(
            annotation_cfg.get(
                "single_stage_compartment_min_z",
                single_stage_compartment_min_z,
            )
        )
        single_stage_compartment_max_q = float(
            annotation_cfg.get(
                "single_stage_compartment_max_q",
                single_stage_compartment_max_q,
            )
        )
        single_stage_compartment_min_channel_support = int(
            annotation_cfg.get(
                "single_stage_compartment_min_channel_support",
                single_stage_compartment_min_channel_support,
            )
        )
        nuclear_pore_guard = bool(
            annotation_cfg.get("nuclear_pore_guard", nuclear_pore_guard)
        )
        nuclear_pore_min_margin = float(
            annotation_cfg.get(
                "nuclear_pore_min_margin",
                nuclear_pore_min_margin,
            )
        )
        nuclear_pore_non_nuclear_top2_min_margin = float(
            annotation_cfg.get(
                "nuclear_pore_non_nuclear_top2_min_margin",
                nuclear_pore_non_nuclear_top2_min_margin,
            )
        )
        nuclear_pore_max_q = float(
            annotation_cfg.get("nuclear_pore_max_q", nuclear_pore_max_q)
        )
        nuclear_pore_min_channel_support = int(
            annotation_cfg.get(
                "nuclear_pore_min_channel_support",
                nuclear_pore_min_channel_support,
            )
        )
        nuclear_pore_fallback_label = str(
            annotation_cfg.get(
                "nuclear_pore_fallback_label",
                nuclear_pore_fallback_label,
            )
        )
        lamina_guard = bool(
            annotation_cfg.get("lamina_guard", lamina_guard)
        )
        lamina_min_margin = float(
            annotation_cfg.get("lamina_min_margin", lamina_min_margin)
        )
        lamina_non_nuclear_top2_min_margin = float(
            annotation_cfg.get(
                "lamina_non_nuclear_top2_min_margin",
                lamina_non_nuclear_top2_min_margin,
            )
        )
        lamina_max_q = float(
            annotation_cfg.get("lamina_max_q", lamina_max_q)
        )
        lamina_min_channel_support = int(
            annotation_cfg.get(
                "lamina_min_channel_support",
                lamina_min_channel_support,
            )
        )
        lamina_boundary_quantile = float(
            annotation_cfg.get(
                "lamina_boundary_quantile",
                lamina_boundary_quantile,
            )
        )
        lamina_fallback_label = str(
            annotation_cfg.get(
                "lamina_fallback_label",
                lamina_fallback_label,
            )
        )
        omm_guard = bool(
            annotation_cfg.get("omm_guard", omm_guard)
        )
        omm_min_margin = float(
            annotation_cfg.get("omm_min_margin", omm_min_margin)
        )
        omm_max_q = float(
            annotation_cfg.get("omm_max_q", omm_max_q)
        )
        omm_min_channel_support = int(
            annotation_cfg.get(
                "omm_min_channel_support",
                omm_min_channel_support,
            )
        )
        omm_fallback_label = str(
            annotation_cfg.get(
                "omm_fallback_label",
                omm_fallback_label,
            )
        )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Invalid annotation configuration value; using the successfully "
            "resolved values and function defaults for the remainder: %s",
            exc,
        )
    if decision_mode == "independent" and guard_mode != "adaptive":
        raise ValueError(
            "annotation.decision_mode=independent requires "
            "annotation.guard_mode=adaptive for empirical label-wise q-values"
        )

    # -----------------------------
    # helpers
    # -----------------------------
    def _case_insensitive_gold_map(gold_df: pd.DataFrame, var_names: pd.Index) -> pd.DataFrame:
        """
        把 gold_df 的基因名（通常 Common_Gene 列或 index）做大小写不敏感映射到 adata.var_names。
        不改权重，只改基因名到“adata 里存在的真实拼写”。
        """
        g = gold_df.copy()
        if 'Common_Gene' not in g.columns and g.index.name == 'Common_Gene':
            g = g.reset_index()
        if g.shape[1] == 0:
            raise ValueError("gold_data 无可用列。")
        gene_col = 'Common_Gene' if 'Common_Gene' in g.columns else g.columns[0]

        var_map = {str(x).strip().upper(): str(x) for x in var_names}
        upper = g[gene_col].astype(str).str.strip().str.upper()
        mapped = upper.map(var_map)

        keep = mapped.notna()
        dropped = int((~keep).sum())
        g = g.loc[keep].copy()

        # ✅ 修复：只用 keep 后的 mapped
        g[gene_col] = mapped.loc[keep].values

        print(f"[gold map] gene_col={gene_col!r}; kept={g.shape[0]} genes; dropped_unmatched_rows={dropped}")
        if g.shape[0] == 0:
            sample_gold = upper.head(5).tolist()
            sample_var = pd.Index(var_names.astype(str)).str.strip().tolist()[:5]
            raise ValueError(
                "gold_data 与 adata.var_names 没有匹配到任何基因（大小写不敏感）。"
                f" gene_col={gene_col!r}; gold示例={sample_gold}; adata.var_names示例={sample_var}"
            )
        return g

    # -----------------------------
    # input checks
    # -----------------------------
    if gold_df is None:
        if 'gold_data' not in globals():
            raise ValueError("请传 gold_df 或在全局提供 gold_data 变量。")
        gold_df = globals()['gold_data']

    if 'leiden' not in adata.obs.columns:
        raise KeyError("adata.obs 缺少 'leiden'，请先在 adata.obs['leiden'] 写入簇标签。")
    if use_two_stage and ('r_mean' not in adata.obs.columns):
        raise KeyError("两阶段需要 adata.obs['r_mean'] 作为核/质粗分依据，请先计算并写入。")

    # -----------------------------
    # gold -> sets (case-insensitive + specificity-first)
    # -----------------------------
    gold_df2 = _case_insensitive_gold_map(gold_df, adata.var_names)

    gold_mat0 = prepare_gold_matrix(
        gold_df2,
        normalize_columns=gold_column_normalize,
    )
    sets_ref, set_weights, regions_ref = build_specific_marker_sets(
        gold_mat0,
        target=set_target_size,
        min_weight=set_min_weight,
        min_margin=set_min_margin,
    )
    print("区域基因集大小（exclusive specificity-first）：", {k: len(v) for k, v in sets_ref.items()})
    marker_union = set().union(*(set(v) for v in sets_ref.values()))
    marker_total = sum(len(v) for v in sets_ref.values())
    print(
        "Marker set overlap diagnostics:",
        {
            "total_memberships": marker_total,
            "unique_genes": len(marker_union),
            "duplicate_memberships": marker_total - len(marker_union),
        },
    )
    if any(len(v) < 10 for v in sets_ref.values()):
        small = {k: len(v) for k, v in sets_ref.items() if len(v) < 10}
        _logging.getLogger("cellscope").warning(
            "Very small exclusive marker sets: %s", small
        )

    def _annotation_group_mean(adata_ref, column: str) -> Optional[pd.Series]:
        if column not in adata_ref.obs.columns:
            return None
        return (
            adata_ref.obs.groupby("leiden", observed=True)[column]
            .mean()
        )

    subtype_geometry_enabled = bool(
        use_spatial_channel and enforce_geometry_consistency
    )

    # -----------------------------
    # annotate
    # -----------------------------
    if use_two_stage:
        print("\n=== Stage 1: Nuclear vs Cytoplasmic classification ===")
        adata.obs['compartment_pred'] = np.where(
            adata.obs['r_mean'] < 0, 'Nuclear', 'Cytoplasmic'
        )

        nuclear_labels = {'Nucleus', 'Nucleolus', 'Lamina', 'Nuclear Pore'}
        cyto_labels    = {'Cytosol', 'ER', 'OMM'}

        nuclear_sets = {k: v for k, v in sets_ref.items() if k in nuclear_labels}
        cyto_sets    = {k: v for k, v in sets_ref.items() if k in cyto_labels}

        print(f"Predicted compartments: {adata.obs['compartment_pred'].value_counts().to_dict()}")
        print(f"Nuclear gold sets: {list(nuclear_sets.keys())}")
        print(f"Cytoplasmic gold sets: {list(cyto_sets.keys())}")

        meta_cols = ['top1','z1','q1','top2','z2','q2',
                     'selected_z','selected_q',
                     'channel_support','raw_combined_score',
                     'core_candidate_label','weak_top1_demoted',
                     'geometry_consistent',
                     'compartment_rescued','preferred_compartment','compartment_score_gap',
                     'low_conf','promoted_from_top2','used_top2_due_to_low_conf',
                     'nuclear_pore_guarded','lamina_guarded',
                     'lamina_geometry_consistent','lamina_boundary_threshold',
                     'omm_guarded','omm_margin',
                     'subtype_empirical_p','subtype_empirical_q',
                     'subtype_testable','subtype_empirical_supported',
                     'effective_marker_count',
                     'strict_label','exploratory_label','annotation_tier',
                     'primary_label','secondary_label',
                     'annotation_state','supported_labels',
                     'n_supported_labels',
                     'primary_evidence_score','secondary_evidence_score',
                     'evidence_score_gap',
                     'primary_empirical_q','secondary_empirical_q']
        meta_cols.extend([
            'omm_raw_competitor_label',
            'omm_raw_competitor_score',
            'omm_specificity_margin',
            'omm_specificity_pass',
            'omm_domain_state',
            'membrane_ambiguity',
        ])
        _ensure_obs_columns_initialized(adata, produced_label_col, meta_cols)
        assigned_labels = adata.obs[produced_label_col].astype(object).to_numpy(copy=True)
        conservative_labels = np.full(adata.n_obs, pd.NA, dtype=object)
        confidence_levels = np.full(adata.n_obs, pd.NA, dtype=object)
        if "subcellular_candidate" in adata.obs.columns:
            adata.obs["subcellular_candidate"] = (
                adata.obs["subcellular_candidate"].astype("object")
            )

        annot_df_list = []

        for comp, comp_sets in [('Nuclear', nuclear_sets), ('Cytoplasmic', cyto_sets)]:
            mask = (adata.obs['compartment_pred'] == comp)
            n_cells = int(mask.sum())
            if n_cells == 0:
                continue
            print(f"\n=== Stage 2: Annotating {comp} ({n_cells} cells) ===")

            adata_sub = adata[mask].copy()

            annot_sub = _rapx_annotate_core(
                adata_sub,
                comp_sets,
                list(comp_sets.keys()),
                set_weights={k: set_weights[k] for k in comp_sets},
                use_spatial_channel=use_spatial_channel,
                geometry_channel_weight=geometry_channel_weight,
                guard_mode=guard_mode,
                subtype_fdr=subtype_fdr,
                min_effective_markers=min_effective_markers,
                null_permutations=null_permutations,
                null_seed=null_seed,
                min_channel_support=min_channel_support,
                enforce_geometry_consistency=enforce_geometry_consistency,
                geometry_quantile=geometry_quantile,
                single_stage_compartment_guard=False,
                geom_before_top2=geom_before_top2,
                top2_switch_z_floor_ratio=top2_switch_z_floor_ratio,
                top2_switch_q_ceiling_ratio=top2_switch_q_ceiling_ratio,
                top2_switch_z_not_much_worse=top2_switch_z_not_much_worse
            )

            # ✅ 去偏置：对 weak top1（Nuclear Pore/ER 等）保守降级
            annot_sub = _demote_weak_top1(
                annot_sub,
                demote_labels=demote_labels,
                margin_lt=demote_if_margin_lt,
                q1_gt=demote_if_q1_gt,
            )
            annot_sub = _refresh_selected_empirical_calibration(
                annot_sub,
                subtype_fdr=subtype_fdr,
            )
            # Candidate means the state entering the structure-specific
            # guards; core_candidate_label retains the unmodified core call.
            annot_sub["candidate_label"] = annot_sub["label"].astype(str)
            annot_sub = _guard_nuclear_pore_overcall(
                annot_sub,
                enabled=nuclear_pore_guard,
                guard_mode=guard_mode,
                subtype_fdr=subtype_fdr,
                min_effective_markers=min_effective_markers,
                min_margin=nuclear_pore_min_margin,
                non_nuclear_top2_min_margin=(
                    nuclear_pore_non_nuclear_top2_min_margin
                ),
                max_q=nuclear_pore_max_q,
                min_channel_support=nuclear_pore_min_channel_support,
                fallback_label=nuclear_pore_fallback_label,
            )
            annot_sub = _guard_lamina_overcall(
                annot_sub,
                group_r_mean=(
                    _annotation_group_mean(adata_sub, "r_mean")
                    if subtype_geometry_enabled
                    else None
                ),
                group_d_nuc_norm_mean=(
                    _annotation_group_mean(adata_sub, "d_nuc_norm_mean")
                    if subtype_geometry_enabled
                    else None
                ),
                enabled=lamina_guard,
                guard_mode=guard_mode,
                subtype_fdr=subtype_fdr,
                min_effective_markers=min_effective_markers,
                min_margin=lamina_min_margin,
                non_nuclear_top2_min_margin=(
                    lamina_non_nuclear_top2_min_margin
                ),
                max_q=lamina_max_q,
                min_channel_support=lamina_min_channel_support,
                boundary_quantile=lamina_boundary_quantile,
                fallback_label=lamina_fallback_label,
            )
            annot_sub = _guard_omm_overcall(
                annot_sub,
                enabled=omm_guard,
                guard_mode=guard_mode,
                subtype_fdr=subtype_fdr,
                min_effective_markers=min_effective_markers,
                min_margin=omm_min_margin,
                max_q=omm_max_q,
                min_channel_support=omm_min_channel_support,
                fallback_label=omm_fallback_label,
            )
            annot_sub = _apply_low_confidence_policy(
                annot_sub,
                compartment=comp,
                policy=low_conf_policy,
                relaxed_z=low_conf_relaxed_z,
                relaxed_q=low_conf_relaxed_q,
            )
            if decision_mode == "independent":
                annot_sub = _apply_independent_evidence_policy(
                    annot_sub,
                    list(comp_sets.keys()),
                    group_r_mean=(
                        _annotation_group_mean(adata_sub, "r_mean")
                        if use_spatial_channel
                        else None
                    ),
                    subtype_fdr=subtype_fdr,
                    min_effective_markers=min_effective_markers,
                    min_channel_support=min_channel_support,
                    geometry_min_abs_r=independent_geometry_min_abs_r,
                    mixed_score_delta=independent_mixed_score_delta,
                    unknown_label=independent_unknown_label,
                    assignment_mode=independent_assignment_mode,
                    exploratory_max_q=(
                        independent_exploratory_max_q
                    ),
                    exploratory_min_channel_support=(
                        independent_exploratory_min_channel_support
                    ),
                    exploratory_min_score_gap=(
                        independent_exploratory_min_score_gap
                    ),
                    subtype_parent_margin=(
                        independent_subtype_parent_margin
                    ),
                    omm_competitor_margin=(
                        independent_omm_competitor_margin
                    ),
                )
            fallback_counts = (
                annot_sub.loc[
                    annot_sub["candidate_label"].astype(str)
                    != annot_sub["label"].astype(str),
                    ["candidate_label", "label"],
                ]
                .value_counts()
                .to_dict()
            )
            if fallback_counts:
                print(
                    f"{comp} low-confidence label fallbacks "
                    f"(candidate -> final): {fallback_counts}"
                )

            tmp_label = adata_sub.obs['leiden'].map(annot_sub['label']).astype('object')
            assigned_labels[np.flatnonzero(mask.to_numpy())] = tmp_label.values
            conservative_labels[np.flatnonzero(mask.to_numpy())] = (
                adata_sub.obs["leiden"]
                .map(annot_sub["conservative_label"])
                .astype("object")
                .values
            )
            confidence_levels[np.flatnonzero(mask.to_numpy())] = (
                adata_sub.obs["leiden"]
                .map(annot_sub["confidence_level"])
                .astype("object")
                .values
            )
            candidate = adata_sub.obs["leiden"].map(
                annot_sub["candidate_label"]
            ).astype("object")
            if "subcellular_candidate" not in adata.obs.columns:
                adata.obs["subcellular_candidate"] = pd.Series(
                    pd.NA,
                    index=adata.obs_names,
                    dtype="object",
                )
            adata.obs.loc[mask, "subcellular_candidate"] = candidate.values

            for col in meta_cols:
                if col not in annot_sub.columns:
                    continue
                src = adata_sub.obs['leiden'].map(annot_sub[col])
                tgt = f"subcell_{col}"
                if tgt not in adata.obs.columns:
                    adata.obs[tgt] = pd.NA
                if isinstance(adata.obs[tgt].dtype, pd.CategoricalDtype):
                    adata.obs[tgt] = adata.obs[tgt].astype('object')

                if col in (
                    'low_conf',
                    'promoted_from_top2',
                    'used_top2_due_to_low_conf',
                    'weak_top1_demoted',
                    'geometry_consistent',
                    'nuclear_pore_guarded',
                    'lamina_guarded',
                    'lamina_geometry_consistent',
                    'compartment_rescued',
                    'omm_guarded',
                    'subtype_testable',
                    'subtype_empirical_supported',
                    'omm_specificity_pass',
                ):
                    adata.obs.loc[mask, tgt] = pd.Series(src.values, index=adata.obs.index[mask]).astype('boolean').values
                elif col in (
                    'top1',
                    'top2',
                    'preferred_compartment',
                    'core_candidate_label',
                    'primary_label',
                    'secondary_label',
                    'strict_label',
                    'exploratory_label',
                    'annotation_tier',
                    'annotation_state',
                    'supported_labels',
                    'omm_raw_competitor_label',
                    'omm_domain_state',
                    'membrane_ambiguity',
                ):
                    adata.obs.loc[mask, tgt] = pd.Series(src.values, index=adata.obs.index[mask]).astype('object').values
                elif col in (
                    'z1','z2','selected_z','compartment_score_gap',
                    'lamina_boundary_threshold',
                    'omm_margin',
                    'effective_marker_count',
                    'n_supported_labels',
                    'primary_evidence_score',
                    'secondary_evidence_score',
                    'evidence_score_gap',
                    'omm_raw_competitor_score',
                    'omm_specificity_margin',
                ):
                    adata.obs.loc[mask, tgt] = pd.to_numeric(src, errors='coerce').astype('float32').values
                elif col in (
                    'q1','q2','selected_q',
                    'subtype_empirical_p','subtype_empirical_q',
                    'primary_empirical_q','secondary_empirical_q',
                ):
                    adata.obs.loc[mask, tgt] = pd.to_numeric(src, errors='coerce').astype('float64').values
                elif col == 'channel_support':
                    adata.obs.loc[mask, tgt] = pd.to_numeric(src, errors='coerce').astype('Int8').values
                elif col == 'raw_combined_score':
                    adata.obs.loc[mask, tgt] = pd.to_numeric(src, errors='coerce').astype('float32').values
                else:
                    adata.obs.loc[mask, tgt] = pd.Series(src.values, index=adata.obs.index[mask]).astype('object').values

            # margin 也写回去方便你看
            if "margin" in annot_sub.columns:
                adata.obs.loc[mask, "subcell_margin"] = pd.to_numeric(
                    adata_sub.obs['leiden'].map(annot_sub["margin"]), errors="coerce"
                ).astype("float32").values

            annot_sub2 = annot_sub.copy()
            annot_sub2['compartment'] = comp
            annot_sub2 = annot_sub2.reset_index().set_index(['leiden','compartment'])
            annot_df_list.append(annot_sub2)

        adata.obs[produced_label_col] = pd.Series(
            assigned_labels,
            index=adata.obs_names,
            dtype="object",
        )
        cats = pd.unique(adata.obs[produced_label_col].dropna().astype(str))
        adata.obs[produced_label_col] = pd.Categorical(
            adata.obs[produced_label_col].astype(str),
            categories=sorted(cats),
            ordered=False
        )
        adata.obs["subcellular_annotation_conservative"] = pd.Categorical(
            pd.Series(conservative_labels, index=adata.obs_names).astype(str)
        )
        if (
            "subcell_strict_label" in adata.obs.columns
            and adata.obs["subcell_strict_label"].notna().any()
        ):
            adata.obs["subcellular_annotation_strict"] = pd.Categorical(
                adata.obs["subcell_strict_label"].astype(str)
            )
        if (
            "subcell_exploratory_label" in adata.obs.columns
            and adata.obs["subcell_exploratory_label"].notna().any()
        ):
            adata.obs[
                "subcellular_annotation_exploratory"
            ] = pd.Categorical(
                adata.obs["subcell_exploratory_label"].astype(str)
            )
        if (
            "subcell_annotation_tier" in adata.obs.columns
            and adata.obs["subcell_annotation_tier"].notna().any()
        ):
            adata.obs["subcellular_annotation_tier"] = pd.Categorical(
                adata.obs["subcell_annotation_tier"].astype(str),
                categories=["Strict", "Exploratory", "Candidate"],
                ordered=True,
            )
        adata.obs["subcellular_confidence"] = pd.Categorical(
            pd.Series(confidence_levels, index=adata.obs_names).astype(str),
            categories=["Low", "Medium", "High"],
            ordered=True,
        )

        _refresh_subcellular_group_columns(adata, produced_label_col)

        annot_df = pd.concat(annot_df_list).sort_index() if len(annot_df_list) else pd.DataFrame(columns=['label'])

    else:
        annot_df = _rapx_annotate_core(
            adata, sets_ref, regions_ref,
            set_weights=set_weights,
            use_spatial_channel=use_spatial_channel,
            geometry_channel_weight=geometry_channel_weight,
            guard_mode=guard_mode,
            subtype_fdr=subtype_fdr,
            min_effective_markers=min_effective_markers,
            null_permutations=null_permutations,
            null_seed=null_seed,
            single_stage_compartment_score_weight=single_stage_compartment_score_weight,
            single_stage_compartment_score_tau=single_stage_compartment_score_tau,
            min_channel_support=min_channel_support,
            enforce_geometry_consistency=enforce_geometry_consistency,
            geometry_quantile=geometry_quantile,
            single_stage_compartment_guard=single_stage_compartment_guard,
            single_stage_compartment_min_abs_r=single_stage_compartment_min_abs_r,
            single_stage_compartment_max_score_gap=single_stage_compartment_max_score_gap,
            single_stage_compartment_min_z=single_stage_compartment_min_z,
            single_stage_compartment_max_q=single_stage_compartment_max_q,
            single_stage_compartment_min_channel_support=single_stage_compartment_min_channel_support,
            geom_before_top2=geom_before_top2,
            top2_switch_z_floor_ratio=top2_switch_z_floor_ratio,
            top2_switch_q_ceiling_ratio=top2_switch_q_ceiling_ratio,
            top2_switch_z_not_much_worse=top2_switch_z_not_much_worse
        )

        annot_df = _demote_weak_top1(
            annot_df,
            demote_labels=demote_labels,
            margin_lt=demote_if_margin_lt,
            q1_gt=demote_if_q1_gt,
        )
        annot_df = _refresh_selected_empirical_calibration(
            annot_df,
            subtype_fdr=subtype_fdr,
        )
        annot_df["candidate_label"] = annot_df["label"].astype(str)
        annot_df = _guard_nuclear_pore_overcall(
            annot_df,
            enabled=nuclear_pore_guard,
            guard_mode=guard_mode,
            subtype_fdr=subtype_fdr,
            min_effective_markers=min_effective_markers,
            min_margin=nuclear_pore_min_margin,
            non_nuclear_top2_min_margin=(
                nuclear_pore_non_nuclear_top2_min_margin
            ),
            max_q=nuclear_pore_max_q,
            min_channel_support=nuclear_pore_min_channel_support,
            fallback_label=nuclear_pore_fallback_label,
        )
        annot_df = _guard_lamina_overcall(
            annot_df,
            group_r_mean=(
                _annotation_group_mean(adata, "r_mean")
                if subtype_geometry_enabled
                else None
            ),
            group_d_nuc_norm_mean=(
                _annotation_group_mean(adata, "d_nuc_norm_mean")
                if subtype_geometry_enabled
                else None
            ),
            enabled=lamina_guard,
            guard_mode=guard_mode,
            subtype_fdr=subtype_fdr,
            min_effective_markers=min_effective_markers,
            min_margin=lamina_min_margin,
            non_nuclear_top2_min_margin=(
                lamina_non_nuclear_top2_min_margin
            ),
            max_q=lamina_max_q,
            min_channel_support=lamina_min_channel_support,
            boundary_quantile=lamina_boundary_quantile,
            fallback_label=lamina_fallback_label,
        )
        annot_df = _guard_omm_overcall(
            annot_df,
            enabled=omm_guard,
            guard_mode=guard_mode,
            subtype_fdr=subtype_fdr,
            min_effective_markers=min_effective_markers,
            min_margin=omm_min_margin,
            max_q=omm_max_q,
            min_channel_support=omm_min_channel_support,
            fallback_label=omm_fallback_label,
        )
        annot_df = _apply_low_confidence_policy_by_candidate(
            annot_df,
            policy=low_conf_policy,
            relaxed_z=low_conf_relaxed_z,
            relaxed_q=low_conf_relaxed_q,
        )
        if decision_mode == "independent":
            annot_df = _apply_independent_evidence_policy(
                annot_df,
                regions_ref,
                group_r_mean=(
                    _annotation_group_mean(adata, "r_mean")
                    if use_spatial_channel
                    else None
                ),
                subtype_fdr=subtype_fdr,
                min_effective_markers=min_effective_markers,
                min_channel_support=min_channel_support,
                geometry_min_abs_r=independent_geometry_min_abs_r,
                mixed_score_delta=independent_mixed_score_delta,
                unknown_label=independent_unknown_label,
                assignment_mode=independent_assignment_mode,
                exploratory_max_q=independent_exploratory_max_q,
                exploratory_min_channel_support=(
                    independent_exploratory_min_channel_support
                ),
                exploratory_min_score_gap=(
                    independent_exploratory_min_score_gap
                ),
                subtype_parent_margin=(
                    independent_subtype_parent_margin
                ),
                omm_competitor_margin=(
                    independent_omm_competitor_margin
                ),
            )

        adata.obs[produced_label_col] = adata.obs['leiden'].map(annot_df['label']).astype('category')
        adata.obs["subcellular_candidate"] = (
            adata.obs["leiden"].map(annot_df["candidate_label"]).astype("category")
        )
        adata.obs["subcellular_annotation_conservative"] = (
            adata.obs["leiden"]
            .map(annot_df["conservative_label"])
            .astype("category")
        )
        if "strict_label" in annot_df.columns:
            adata.obs["subcellular_annotation_strict"] = (
                adata.obs["leiden"]
                .map(annot_df["strict_label"])
                .astype("category")
            )
        if "exploratory_label" in annot_df.columns:
            adata.obs["subcellular_annotation_exploratory"] = (
                adata.obs["leiden"]
                .map(annot_df["exploratory_label"])
                .astype("category")
            )
        if "annotation_tier" in annot_df.columns:
            adata.obs["subcellular_annotation_tier"] = pd.Categorical(
                adata.obs["leiden"].map(annot_df["annotation_tier"]),
                categories=["Strict", "Exploratory", "Candidate"],
                ordered=True,
            )
        adata.obs["subcellular_confidence"] = pd.Categorical(
            adata.obs["leiden"].map(annot_df["confidence_level"]),
            categories=["Low", "Medium", "High"],
            ordered=True,
        )

        for col in [
            'top1','z1','q1','top2','z2','q2',
            'selected_z','selected_q',
            'channel_support','raw_combined_score',
            'core_candidate_label','weak_top1_demoted',
            'geometry_consistent',
            'compartment_rescued','preferred_compartment','compartment_score_gap',
            'low_conf','promoted_from_top2','used_top2_due_to_low_conf',
            'nuclear_pore_guarded',
            'lamina_guarded','lamina_geometry_consistent',
            'lamina_boundary_threshold',
            'omm_guarded','omm_margin',
            'subtype_empirical_p','subtype_empirical_q',
            'subtype_testable','subtype_empirical_supported',
            'effective_marker_count',
            'strict_label','exploratory_label','annotation_tier',
            'primary_label','secondary_label',
            'annotation_state','supported_labels',
            'n_supported_labels',
            'primary_evidence_score','secondary_evidence_score',
            'evidence_score_gap',
            'primary_empirical_q','secondary_empirical_q',
            'omm_raw_competitor_label','omm_raw_competitor_score',
            'omm_specificity_margin','omm_specificity_pass',
            'omm_domain_state','membrane_ambiguity',
        ]:
            if col in annot_df.columns:
                adata.obs[f'subcell_{col}'] = adata.obs['leiden'].map(annot_df[col])

        for region in regions_ref:
            for prefix in (
                "raw_evidence_score",
                "evidence_score",
                "relative_evidence",
                "channel_support",
                "geometry_feasible",
                "statistically_supported",
                "specificity_margin",
                "specificity_pass",
                "empirically_supported",
                "subtype_empirical_q",
                "subtype_calibrated_score",
            ):
                source = f"{prefix}__{region}"
                if source in annot_df.columns:
                    adata.obs[f"subcell_{source}"] = (
                        adata.obs["leiden"].map(annot_df[source])
                    )

        if "margin" in annot_df.columns:
            adata.obs["subcell_margin"] = pd.to_numeric(
                adata.obs['leiden'].map(annot_df["margin"]), errors="coerce"
            ).astype("float32").values

        _refresh_subcellular_group_columns(adata, produced_label_col)

    # -----------------------------
    # meta
    # -----------------------------
    adata.uns['subcellular_annotation_meta'] = {
        'regions': regions_ref,
        'set_sizes': {k: len(v) for k, v in sets_ref.items()},
        'score_interpretation': (
            'selected_q is a heuristic within-group q-score derived from the '
            'combined ranking score. subtype_empirical_q is the label-specific '
            'BH-adjusted empirical p-value from matched random marker sets and '
            'is used to establish support. subtype_calibrated_score__* is a '
            'label-wise null statistic and is not used for cross-label choice. '
            'evidence_score__* combines the set-size-aware MWU and '
            'hypergeometric channels for cross-label ranking; '
            'raw_evidence_score__* retains the original three-channel '
            'combination, including the L2 projection, for diagnostics. Fine '
            'subtypes must '
            'also beat their broad raw-evidence parent; OMM must beat both ER '
            'and Cytosol. In hierarchical assignment '
            'mode, unsupported subtype candidates fall back to Nucleus or '
            'Cytosol; '
            'subcellular_annotation_strict is the confirmatory FDR-controlled '
            'call and annotation_tier distinguishes Strict, Exploratory, and '
            'Candidate evidence.'
        ),
        'params': {
            'use_two_stage': use_two_stage,
            'use_spatial_channel': use_spatial_channel,
            'subtype_geometry_enabled': subtype_geometry_enabled,
            'CHANNEL_WEIGHTS': list(CHANNEL_WEIGHTS),
            'geom_before_top2': bool(geom_before_top2),
            'TOPK_OVERLAP_FRAC': TOPK_OVERLAP_FRAC,
            'TOPK_OVERLAP_MIN': TOPK_OVERLAP_MIN,
            'FDR_CUTOFF': FDR_CUTOFF,
            'Z_CUTOFF': Z_CUTOFF,
            'TIE_DELTA': TIE_DELTA,
            # new knobs
            'set_target_size': int(set_target_size),
            'marker_min_weight': float(set_min_weight),
            'marker_min_margin': float(set_min_margin),
            'gold_column_normalize': bool(gold_column_normalize),
            'min_channel_support': int(min_channel_support),
            'enforce_geometry_consistency': bool(enforce_geometry_consistency),
            'geometry_quantile': float(geometry_quantile),
            'geometry_channel_weight': float(geometry_channel_weight),
            'guard_mode': str(guard_mode),
            'subtype_fdr': float(subtype_fdr),
            'min_effective_markers': int(min_effective_markers),
            'null_permutations': int(null_permutations),
            'null_seed': int(null_seed),
            'single_stage_compartment_score_weight': float(single_stage_compartment_score_weight),
            'single_stage_compartment_score_tau': float(single_stage_compartment_score_tau),
            'single_stage_compartment_guard': bool(single_stage_compartment_guard),
            'single_stage_compartment_min_abs_r': float(single_stage_compartment_min_abs_r),
            'single_stage_compartment_max_score_gap': float(single_stage_compartment_max_score_gap),
            'single_stage_compartment_min_z': float(single_stage_compartment_min_z),
            'single_stage_compartment_max_q': float(single_stage_compartment_max_q),
            'single_stage_compartment_min_channel_support': int(single_stage_compartment_min_channel_support),
            'demote_if_margin_lt': float(demote_if_margin_lt),
            'demote_if_q1_gt': float(demote_if_q1_gt),
            'demote_labels': list(demote_labels),
            'nuclear_pore_guard': bool(nuclear_pore_guard),
            'nuclear_pore_min_margin': float(nuclear_pore_min_margin),
            'nuclear_pore_non_nuclear_top2_min_margin': float(nuclear_pore_non_nuclear_top2_min_margin),
            'nuclear_pore_max_q': float(nuclear_pore_max_q),
            'nuclear_pore_min_channel_support': int(nuclear_pore_min_channel_support),
            'nuclear_pore_fallback_label': str(nuclear_pore_fallback_label),
            'lamina_guard': bool(lamina_guard),
            'lamina_min_margin': float(lamina_min_margin),
            'lamina_non_nuclear_top2_min_margin': float(
                lamina_non_nuclear_top2_min_margin
            ),
            'lamina_max_q': float(lamina_max_q),
            'lamina_min_channel_support': int(
                lamina_min_channel_support
            ),
            'lamina_boundary_quantile': float(lamina_boundary_quantile),
            'lamina_fallback_label': str(lamina_fallback_label),
            'omm_guard': bool(omm_guard),
            'omm_min_margin': float(omm_min_margin),
            'omm_max_q': float(omm_max_q),
            'omm_min_channel_support': int(omm_min_channel_support),
            'omm_fallback_label': str(omm_fallback_label),
            'decision_mode': str(decision_mode),
            'independent_mixed_score_delta': float(
                independent_mixed_score_delta
            ),
            'independent_unknown_label': str(independent_unknown_label),
            'independent_geometry_min_abs_r': float(
                independent_geometry_min_abs_r
            ),
            'independent_assignment_mode': str(
                independent_assignment_mode
            ),
            'independent_exploratory_max_q': float(
                independent_exploratory_max_q
            ),
            'independent_exploratory_min_channel_support': int(
                independent_exploratory_min_channel_support
            ),
            'independent_exploratory_min_score_gap': float(
                independent_exploratory_min_score_gap
            ),
            'independent_subtype_parent_margin': float(
                independent_subtype_parent_margin
            ),
            'independent_omm_competitor_margin': float(
                independent_omm_competitor_margin
            ),
            'low_conf_policy': str(low_conf_policy),
            'low_conf_relaxed_z': float(low_conf_relaxed_z),
            'low_conf_relaxed_q': float(low_conf_relaxed_q),
        }
    }

    # -----------------------------
    # plotting
    # -----------------------------
    if plot_umap:
        def _sanitize_color_cols(cols):
            out = []
            for c in cols:
                if c not in adata.obs.columns and c not in adata.var_names:
                    continue
                if c not in adata.obs.columns:
                    out.append(c)
                    continue
                col = adata.obs[c]
                if pd.api.types.is_bool_dtype(col) or str(col.dtype) == 'boolean':
                    new_c = f"{c}__plotnum"
                    if new_c not in adata.obs.columns:
                        adata.obs[new_c] = col.astype('float32')
                    out.append(new_c)
                    continue
                if (
                    pd.api.types.is_object_dtype(col)
                    or pd.api.types.is_string_dtype(col)
                ):
                    # Scanpy can interpret an object/string column as a
                    # continuous vector and attempt unary numeric operations.
                    # Convert compact enums explicitly; skip identifiers with
                    # one value per MetaDomain because they are not readable.
                    if int(col.nunique(dropna=True)) > 64:
                        continue
                    adata.obs[c] = pd.Categorical(col.astype(str))
                out.append(c)
            return out

        primary_colors = [
            'subcell_core_candidate_label',
            'subcellular_annotation_strict',
            produced_label_col,
            'subcellular_annotation_tier',
            'cluster-subcellular',
        ]
        if use_two_stage:
            primary_colors.append('compartment_pred')
        colors1 = _sanitize_color_cols(primary_colors)
        colors2 = _sanitize_color_cols([
            'subcell_z1',
            'subcell_q1',
            'subcell_subtype_empirical_q',
            'subcell_subtype_testable',
            'subcell_low_conf',
            'subcell_margin',
        ])

        try:
            if colors1:
                sc.pl.umap(
                    adata,
                    color=colors1,
                    legend_fontsize="small",
                    wspace=0.6,
                )
            if colors2:
                sc.pl.umap(
                    adata,
                    color=colors2,
                    cmap='viridis',
                    wspace=0.4,
                )
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "RAP-X UMAP diagnostic plotting failed; annotation results "
                "are retained: %s",
                exc,
            )

    if plot_diag_enrich:
        print("\n=== Using improved enrichment analysis (QUQ default) ===")
        if isinstance(annot_df.index, pd.MultiIndex):
            if 'q1' in annot_df.columns:
                annot_df_vis = (
                    annot_df.reset_index()
                            .sort_values(['leiden','q1'], ascending=[True, True])
                            .drop_duplicates(subset=['leiden'], keep='first')
                            .set_index('leiden')
                )
            else:
                annot_df_vis = (
                    annot_df.reset_index()
                            .drop_duplicates(subset=['leiden'], keep='first')
                            .set_index('leiden')
                )
        else:
            annot_df_vis = annot_df

        group_sizes = adata.obs["leiden"].astype(str).value_counts()
        if int(group_sizes.max()) < 2:
            _logging.getLogger("cellscope").info(
                "RAP-X rank-genes enrichment diagnostics skipped for "
                "singleton MetaDomain annotation groups."
            )
        else:
            try:
                _ = improved_enrichment_visualization(
                    adata, annot_df_vis, sets_ref,
                    selection_mode='weighted_union',
                    q_per_group=500,
                    p_adj_max=0.5, lfc_min=0.0,
                    balance_labels=True,
                    target_per_label=TARGET_SET_SIZE,
                    min_per_label=TARGET_SET_SIZE,
                    max_per_label=TARGET_SET_SIZE,
                    min_query_fallback=80,
                    fallback_mode='weighted_union',
                    show_tables=True
                )
            except Exception as e:
                adata.obs.drop(columns=["_rapx_diag_group"], inplace=True, errors="ignore")
                try:
                    _logging.getLogger("cellscope").warning(
                        "RAP-X enrichment diagnostics skipped: %s",
                        e,
                    )
                except Exception:
                    pass

        if 'plot_annotation_quality_summary' in globals():
            try:
                plot_annotation_quality_summary(adata, annot_df_vis, produced_label_col)
            except Exception as e:
                print(f"[WARN] plot_annotation_quality_summary 运行失败：{e}")

    print("Exclusive marker set sizes:", {k: len(v) for k, v in sets_ref.items()})
    return annot_df



def _prepare_apex_gold_reference(
    apex_raw: pd.DataFrame,
    *,
    omm_specificity_filter: bool = True,
    omm_min_log2fc: float = 0.75,
    omm_min_delta: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the bundled APEX reference and audit OMM specificity.

    APEX-seq enrichment is a soft localization reference, not a census of
    molecules on an organelle.  In particular, the published OMM and ERM
    programs share many secretory-pathway RNAs.  An RNA is therefore allowed
    into the *OMM-specific* reference only when its raw OMM enrichment clears
    both a basal threshold and the stronger of ERM/Cytosol by ``omm_min_delta``.

    The gate is evaluated on the original log2 fold changes.  Row-wise
    normalization is performed afterwards, and failed rows have only their OMM
    value set to zero.  Their ER, Cytosol, and other localization evidence is
    deliberately retained.
    """
    required = {
        "Common_Gene",
        "OMM_log2FC",
        "ERM_log2FC",
        "Cytosol_log2FC",
    }
    missing = sorted(required - set(apex_raw.columns))
    if missing:
        raise ValueError(
            "Bundled APEX reference is missing raw columns required for the "
            f"OMM specificity audit: {missing}"
        )
    try:
        min_log2fc = float(omm_min_log2fc)
        min_delta = float(omm_min_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "APEX OMM thresholds must be finite numeric values"
        ) from exc
    if not np.isfinite(min_log2fc) or not np.isfinite(min_delta):
        raise ValueError("APEX OMM thresholds must be finite numeric values")
    if min_delta < 0.0:
        raise ValueError("apex_omm_min_delta must be non-negative")

    raw = apex_raw.copy()
    omm = pd.to_numeric(raw["OMM_log2FC"], errors="coerce")
    erm = pd.to_numeric(raw["ERM_log2FC"], errors="coerce")
    cytosol = pd.to_numeric(raw["Cytosol_log2FC"], errors="coerce")
    valid = omm.notna() & erm.notna() & cytosol.notna()
    competitor = pd.concat(
        [erm.rename("ERM"), cytosol.rename("Cytosol")],
        axis=1,
    ).max(axis=1)
    delta = omm - competitor
    basal_pass = valid & omm.ge(min_log2fc)
    specific_pass = basal_pass & delta.ge(min_delta)
    shared = basal_pass & ~specific_pass

    reason = pd.Series("invalid_numeric", index=raw.index, dtype="object")
    reason.loc[valid & ~basal_pass] = "below_basal"
    reason.loc[shared] = "shared_with_erm_or_cytosol"
    reason.loc[specific_pass] = "specific"
    audit = pd.DataFrame(
        {
            "Common_Gene": raw["Common_Gene"],
            "Source": raw.get(
                "Source",
                pd.Series("", index=raw.index, dtype="object"),
            ),
            "OMM_log2FC": omm,
            "ERM_log2FC": erm,
            "Cytosol_log2FC": cytosol,
            "omm_competitor_log2fc": competitor,
            "omm_specificity_delta": delta,
            "omm_basal_pass": basal_pass.astype(bool),
            "omm_specific_pass": specific_pass.astype(bool),
            "omm_shared": shared.astype(bool),
            "omm_filter_reason": reason,
        }
    ).reset_index(drop=True)

    cols = ["Common_Gene"] + [
        column
        for column in raw.columns
        if str(column).endswith("_log2FC")
    ]
    gold = raw[cols].rename(
        columns={
            column: column.replace("_log2FC", "")
            for column in cols
            if column != "Common_Gene"
        }
    )
    complete = gold.notna().all(axis=1)
    gold = gold.loc[complete].copy()
    values_columns = [
        column for column in gold.columns if column != "Common_Gene"
    ]
    values = (
        gold[values_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    values = values.subtract(values.min(axis=1), axis=0)
    span = (values.max(axis=1) - values.min(axis=1)).replace(0, 1.0)
    values = values.div(span, axis=0)
    if bool(omm_specificity_filter) and "OMM" in values.columns:
        values.loc[~specific_pass.reindex(values.index).fillna(False), "OMM"] = 0.0
    gold = pd.concat(
        [
            gold[["Common_Gene"]].reset_index(drop=True),
            values.reset_index(drop=True),
        ],
        axis=1,
    )
    return gold, audit


def _load_default_gold_tables(
    *,
    apex_omm_specificity_filter: bool = True,
    apex_omm_min_log2fc: float = 0.75,
    apex_omm_min_delta: float = 0.50,
    return_omm_audit: bool = False,
):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    apex = os.path.join(base, '2019_CELL_APEXSeq.tsv')
    rna = os.path.join(base, 'RNALocateV3.txt')
    if not os.path.exists(apex) or not os.path.exists(rna):
        raise FileNotFoundError(f"Gold files not found under {base}")
    apex_raw = pd.read_table(apex)
    rna_locate = pd.read_table(rna)
    gold_apex, omm_audit = _prepare_apex_gold_reference(
        apex_raw,
        omm_specificity_filter=apex_omm_specificity_filter,
        omm_min_log2fc=apex_omm_min_log2fc,
        omm_min_delta=apex_omm_min_delta,
    )
    if return_omm_audit:
        return gold_apex, rna_locate, omm_audit
    return gold_apex, rna_locate


def _configure_rnalocate_augmentation(
    gold_from_rna: pd.DataFrame,
    augment_er: bool = False,
) -> pd.DataFrame:
    """Keep broad RNALocate ER associations from overriding APEX specificity."""
    out = gold_from_rna.copy()
    if not augment_er:
        for column in ("ER", "ERM", "ER_Lumen"):
            if column in out.columns:
                out[column] = 0.0
    return out


def _configure_annotation_unit(
    adata2: ad.AnnData,
    unit: str = "leiden",
) -> ad.AnnData:
    """Choose the grain at which RAP-X makes independent decisions.

    ``metadomain`` preserves the embedding Leiden labels for visualization but
    gives each adata2 row its own annotation group.  This prevents a single
    cross-cell Leiden label from being broadcast over biologically different
    MetaDomains.
    """
    mode = str(unit).strip().lower().replace("_", "-")
    if mode in {"leiden", "cluster", "annotation-cluster"}:
        adata2.obs["annotation_group"] = adata2.obs["leiden"].astype(str)
        adata2.uns["annotation_unit"] = "leiden"
        return adata2
    if mode not in {"metadomain", "meta-domain", "domain"}:
        raise ValueError(
            "annotation.unit must be 'metadomain' or 'leiden', "
            f"got {unit!r}"
        )
    if not adata2.obs_names.is_unique:
        raise ValueError(
            "MetaDomain annotation requires unique adata2.obs_names"
        )
    if "leiden" in adata2.obs.columns:
        adata2.obs["leiden_embedding"] = adata2.obs["leiden"].astype(str)
    groups = pd.Index(adata2.obs_names.astype(str), name="annotation_group")
    adata2.obs["annotation_group"] = pd.Categorical(groups)
    adata2.obs["leiden"] = pd.Categorical(groups)
    adata2.uns["annotation_unit"] = "metadomain"
    return adata2


def  _maybe_cluster_and_annotate(adata1: ad.AnnData, adata2: ad.AnnData, final_df: pd.DataFrame, show_internal_progress: bool = False, out_dir: Optional[str] = None,
                                annotation_method: str = "rapx",
                                rapx_two_stage: bool = False, rapx_spatial_channel: bool = False,
                                gold_data: Optional[Union[str, pd.DataFrame]] = None):
    # Step 0: choose annotation method (future-proof)
    _method = (annotation_method or "rapx").lower()
    if _method not in {"rapx", "rap-x"}:
        try:
            _logging.getLogger('cellscope').warning("Annotation method '%s' not supported; skipping annotation stage.", _method)
        except Exception:
            pass
        return adata1, adata2, final_df

    # Step 1: normalize + cluster on adata2 (scanpy)
    # Pull clustering params from config if available
    rep_priority = None
    n_neighbors = 32
    leiden_resolution = 1.0
    hvg_n_top_genes = None
    pca_n_comps = None
    umap_min_dist = None
    umap_spread = None
    umap_neighbors_key = None
    cluster_rep = None
    annotation_unit = "leiden"
    rnalocate_augment_er = False
    rnalocate_species: Optional[Tuple[str, ...]] = (
        "Homo sapiens",
        "Mus musculus",
    )
    apex_omm_specificity_filter = True
    apex_omm_min_log2fc = 0.75
    apex_omm_min_delta = 0.50
    omm_metaspot_refinement = True
    omm_metaspot_apply_to_primary = True
    omm_metaspot_ambiguous_policy = "competitor"
    omm_metaspot_min_informative_transcripts = 5
    omm_metaspot_shrinkage_count = 20.0
    omm_metaspot_posterior_high = 0.80
    omm_metaspot_posterior_low = 0.20
    try:
        _cfg = load_params_yaml()
        rep_priority = _cfg.get('module7', {}).get('rep_priority', None)
        if isinstance(rep_priority, str):
            rep_priority = [s.strip() for s in rep_priority.split(',') if s.strip()]
        # Clustering neighbors for annotation stage (from annotation config)
        n_neighbors = int(_cfg.get('annotation', {}).get('neighbors_k', 15))
        # Use annotation's Leiden resolution (no legacy fallbacks)
        leiden_resolution = float(_cfg.get('annotation', {}).get('leiden_resolution', 1.0))
        # annotation-level params
        ann = _cfg.get('annotation', {})
        hvg_n_top_genes = ann.get('hvg_n_top_genes', None)
        pca_n_comps = ann.get('pca_n_comps', None)
        umap_min_dist = ann.get('umap_min_dist', None)
        umap_spread = ann.get('umap_spread', None)
        umap_neighbors_key = ann.get('umap_neighbors_key', None)
        cluster_rep = ann.get('cluster_rep', None)
        annotation_unit = str(ann.get("unit", annotation_unit))
        rnalocate_augment_er = bool(
            ann.get("rnalocate_augment_er", False)
        )
        apex_omm_specificity_filter = bool(
            ann.get(
                "apex_omm_specificity_filter",
                apex_omm_specificity_filter,
            )
        )
        apex_omm_min_log2fc = float(
            ann.get("apex_omm_min_log2fc", apex_omm_min_log2fc)
        )
        apex_omm_min_delta = float(
            ann.get("apex_omm_min_delta", apex_omm_min_delta)
        )
        species_cfg = ann.get("rnalocate_species", rnalocate_species)
        if species_cfg is None:
            rnalocate_species = None
        elif isinstance(species_cfg, str):
            rnalocate_species = tuple(
                value.strip()
                for value in species_cfg.split(",")
                if value.strip()
            )
        else:
            rnalocate_species = tuple(str(value) for value in species_cfg)
        ann_fast = bool(ann.get('fast', False))
        ann_save_plots = bool(ann.get('save_plots', False))
    except Exception:
        ann_fast = False
        ann_save_plots = False
    # Reference thresholds affect scientific identity and resume validity.
    # Parse them outside the broad optional-clustering fallback above so a
    # malformed value cannot be silently replaced by a default.
    strict_annotation_cfg = (
        (load_params_yaml() or {}).get("annotation", {}) or {}
    )
    apex_omm_specificity_filter = bool(
        strict_annotation_cfg.get(
            "apex_omm_specificity_filter",
            apex_omm_specificity_filter,
        )
    )
    omm_metaspot_refinement = bool(
        strict_annotation_cfg.get(
            "omm_metaspot_refinement",
            omm_metaspot_refinement,
        )
    )
    omm_metaspot_apply_to_primary = bool(
        strict_annotation_cfg.get(
            "omm_metaspot_apply_to_primary",
            omm_metaspot_apply_to_primary,
        )
    )
    omm_metaspot_ambiguous_policy = str(
        strict_annotation_cfg.get(
            "omm_metaspot_ambiguous_policy",
            omm_metaspot_ambiguous_policy,
        )
    ).strip().lower()
    if omm_metaspot_ambiguous_policy not in {
        "competitor",
        "explicit",
    }:
        raise ValueError(
            "annotation.omm_metaspot_ambiguous_policy must be one of "
            "{'competitor', 'explicit'}"
        )
    try:
        apex_omm_min_log2fc = float(
            strict_annotation_cfg.get(
                "apex_omm_min_log2fc",
                apex_omm_min_log2fc,
            )
        )
        apex_omm_min_delta = float(
            strict_annotation_cfg.get(
                "apex_omm_min_delta",
                apex_omm_min_delta,
            )
        )
        omm_metaspot_min_informative_transcripts = int(
            strict_annotation_cfg.get(
                "omm_metaspot_min_informative_transcripts",
                omm_metaspot_min_informative_transcripts,
            )
        )
        omm_metaspot_shrinkage_count = float(
            strict_annotation_cfg.get(
                "omm_metaspot_shrinkage_count",
                omm_metaspot_shrinkage_count,
            )
        )
        omm_metaspot_posterior_high = float(
            strict_annotation_cfg.get(
                "omm_metaspot_posterior_high",
                omm_metaspot_posterior_high,
            )
        )
        omm_metaspot_posterior_low = float(
            strict_annotation_cfg.get(
                "omm_metaspot_posterior_low",
                omm_metaspot_posterior_low,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "APEX OMM and OMM MetaSpot refinement thresholds in annotation "
            "must be numeric"
        ) from exc
    if (
        not np.isfinite(apex_omm_min_log2fc)
        or not np.isfinite(apex_omm_min_delta)
        or apex_omm_min_delta < 0.0
    ):
        raise ValueError(
            "annotation.apex_omm_min_log2fc must be finite and "
            "annotation.apex_omm_min_delta must be finite and non-negative"
        )
    if omm_metaspot_min_informative_transcripts < 1:
        raise ValueError(
            "annotation.omm_metaspot_min_informative_transcripts must be >= 1"
        )
    if (
        not np.isfinite(omm_metaspot_shrinkage_count)
        or omm_metaspot_shrinkage_count < 0.0
    ):
        raise ValueError(
            "annotation.omm_metaspot_shrinkage_count must be finite and >= 0"
        )
    if not (
        0.0
        < omm_metaspot_posterior_low
        < omm_metaspot_posterior_high
        < 1.0
    ):
        raise ValueError(
            "annotation OMM MetaSpot posterior thresholds must satisfy "
            "0 < posterior_low < posterior_high < 1"
        )
    # Upstream Leiden groups are intentionally coarse and should not silently
    # become annotation groups. Reuse them only when fast mode is explicit.
    if not (ann_fast and ('leiden' in adata2.obs.columns)):
        adata2 = _normalize_and_cluster_adata2(
            adata2,
            rep_priority=cast(Optional[List[str]], rep_priority),
            out_dir=out_dir,
            n_neighbors=n_neighbors,
            leiden_resolution=leiden_resolution,
            hvg_n_top_genes=hvg_n_top_genes,
            pca_n_comps=pca_n_comps,
            umap_min_dist=umap_min_dist,
            umap_spread=umap_spread,
            umap_neighbors_key=umap_neighbors_key,
            cluster_rep=cluster_rep,
        )
        # Type cast retained for analyzers
        rep_priority = cast(Optional[List[str]], rep_priority)
    adata2 = _configure_annotation_unit(
        adata2,
        unit=annotation_unit,
    )
    # Step 2: load gold tables (or use user-supplied `gold_data`).
    apex_omm_audit: Optional[pd.DataFrame] = None
    reference_source = "bundled_apex_plus_rnalocate"
    if gold_data is not None:
        try:
            if isinstance(gold_data, str):
                # Interpret string as path to parquet/csv
                if os.path.exists(gold_data):
                    try:
                        _gdf = pd.read_parquet(gold_data)
                    except Exception:
                        _gdf = pd.read_csv(gold_data)
                else:
                    raise FileNotFoundError(f"gold_data path not found: {gold_data}")
            elif isinstance(gold_data, pd.DataFrame):
                _gdf = gold_data.copy()
            else:
                _gdf = pd.DataFrame(gold_data)
            # Ensure gene column present
            if 'Common_Gene' not in _gdf.columns:
                # try reset index as gene names
                try:
                    _gdf = _gdf.reset_index()
                except Exception:
                    pass
            if 'Common_Gene' not in _gdf.columns:
                raise ValueError("Provided gold_data must contain a 'Common_Gene' column")
            # Normalize numeric columns per-row to 0-1 (same logic as default loader)
            val_cols = [c for c in _gdf.columns if c != 'Common_Gene']
            if len(val_cols) == 0:
                raise ValueError("Provided gold_data must contain numeric region columns in addition to 'Common_Gene'")
            X = _gdf[val_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
            X = X.subtract(X.min(axis=1), axis=0)
            span = (X.max(axis=1) - X.min(axis=1)).replace(0, 1.0)
            X = X.div(span, axis=0)
            gold_aug = pd.concat([_gdf[['Common_Gene']].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        except Exception as e:
            raise ValueError(
                "Failed to load or validate the user-supplied gold_data. "
                "CellScope will not silently replace it with the bundled "
                "human APEX reference."
            ) from e
        reference_source = "user_supplied"
    else:
        gold_apex, rna_loc, apex_omm_audit = _load_default_gold_tables(
            apex_omm_specificity_filter=apex_omm_specificity_filter,
            apex_omm_min_log2fc=apex_omm_min_log2fc,
            apex_omm_min_delta=apex_omm_min_delta,
            return_omm_audit=True,
        )
        l1 = build_gold_from_rnalocate_l1(
            rna_loc,
            species=rnalocate_species,
            agg_mode='prob_or',
            score_floor=0.6,
            use_pubmed_weight=True,
        )
        gold_from_rna = _configure_rnalocate_augmentation(
            project_l1_to_seven(
                l1,
                mode='strict',
                mito_non_omm=None,
            ),
            augment_er=rnalocate_augment_er,
        )
        gold_aug = merge_gold(gold_apex, gold_from_rna, fuse='max')
        if out_dir is not None and apex_omm_audit is not None:
            _save_dataframe(
                apex_omm_audit,
                out_dir,
                "apex_omm_reference_audit",
            )

    # Optional manual markers. Disabled by default because marker provenance and
    # compartment specificity must be curated explicitly.
    try:
        use_manual_markers = bool(
            (load_params_yaml() or {}).get("annotation", {}).get(
                "use_manual_markers", False
            )
        )
    except Exception:
        use_manual_markers = False
    manual_df = _load_manual_marker_data() if use_manual_markers else None
    if manual_df is not None:
        try:
            gold_aug = merge_gold(gold_aug, manual_df, fuse='max')
            _logging.getLogger('cellscope').info("Manual marker table merged: %d genes", len(manual_df))
        except Exception:
            try:
                _logging.getLogger('cellscope').warning("Manual marker merge failed; continuing without manual markers")
            except Exception:
                pass

    # Case-harmonize gold genes to improve mouse coverage when gold is all upper-case.
    try:
        gold_aug = harmonize_gold_gene_case(gold_aug, target_var_names=adata2.var_names)
    except Exception:
        pass
    if reference_source != "user_supplied":
        _logging.getLogger("cellscope").warning(
            "The bundled APEX-seq reference is human HEK293T. Gene matching "
            "to this dataset uses case-insensitive same-symbol intersection; "
            "it is not a validated human-to-mouse ortholog conversion."
        )
    # Step 3: RAP-X annotation; write to adata2.obs
    rapx_use_two_stage = False
    rapx_use_spatial_channel = False
    rapx_plot_umap = False
    rapx_plot_diag = False
    try:
        _cfg2 = load_params_yaml()
        ann_cfg = _cfg2.get('annotation', {}) if _cfg2 else {}
        rapx_use_two_stage = bool(ann_cfg.get('two_stage', rapx_use_two_stage))
        rapx_use_spatial_channel = bool(ann_cfg.get('spatial_channel', rapx_use_spatial_channel))
        rapx_plot_umap = bool(ann_cfg.get('save_umap_plot', False) or ann_cfg.get('save_plots', False))
        rapx_plot_diag = bool(ann_cfg.get('save_plots', False))
    except Exception:
        pass
    t_rapx0 = __import__('time').perf_counter()
    annot_df = run_rapx(
        adata2,
        gold_df=gold_aug,
        produced_label_col='subcellular_annotation',
        plot_umap=rapx_plot_umap,
        plot_diag_enrich=rapx_plot_diag,
        use_two_stage=rapx_use_two_stage,
        use_spatial_channel=rapx_use_spatial_channel,
    )
    try:
        annotation_meta = adata2.uns.setdefault(
            "subcellular_annotation_meta",
            {},
        )
        annotation_meta["annotation_unit"] = str(
            adata2.uns.get("annotation_unit", annotation_unit)
        )
        apex_audit_summary: Dict[str, Any] = {
            "filter_applied": bool(
                reference_source != "user_supplied"
                and apex_omm_specificity_filter
            ),
            "min_log2fc": float(apex_omm_min_log2fc),
            "min_delta_vs_erm_cytosol": float(apex_omm_min_delta),
        }
        if apex_omm_audit is not None:
            apex_audit_summary.update(
                {
                    "rows": int(len(apex_omm_audit)),
                    "basal_pass_rows": int(
                        apex_omm_audit["omm_basal_pass"].sum()
                    ),
                    "specific_pass_rows": int(
                        apex_omm_audit["omm_specific_pass"].sum()
                    ),
                    "shared_rows": int(
                        apex_omm_audit["omm_shared"].sum()
                    ),
                    "specific_unique_named_genes": int(
                        apex_omm_audit.loc[
                            apex_omm_audit["omm_specific_pass"],
                            "Common_Gene",
                        ]
                        .dropna()
                        .astype(str)
                        .nunique()
                    ),
                }
            )
        if reference_source == "user_supplied":
            annotation_meta["reference_provenance"] = {
                "reference_source": reference_source,
                "apex_used": False,
                "apex_omm_specificity": {
                    "filter_applied": False,
                    "reason": (
                        "The bundled APEX-only gate is never imposed on a "
                        "user-supplied reference."
                    ),
                },
                "rnalocate_used": False,
                "gene_mapping_mode": (
                    "case-insensitive same-symbol intersection"
                ),
            }
        else:
            annotation_meta["reference_provenance"] = {
                "reference_source": reference_source,
                "apex": (
                    "Cell 2019 APEX-seq; human; soft localization reference"
                ),
                "apex_source_species": "Homo sapiens",
                "apex_source_cell_context": "HEK293T",
                "apex_omm_specificity": apex_audit_summary,
                "gene_mapping_mode": (
                    "case-insensitive same-symbol intersection"
                ),
                "ortholog_mapping_performed": False,
                "cross_species_limitation": (
                    "Mouse matches are conserved same-symbol matches, not "
                    "validated one-to-one ortholog conversions."
                ),
                "rnalocate": "RNALocate v3",
                "rnalocate_species": (
                    list(rnalocate_species)
                    if rnalocate_species is not None
                    else "all"
                ),
                "rnalocate_er_enabled": bool(rnalocate_augment_er),
                "fusion": "maximum across sources",
            }
        annotation_meta["relative_evidence_note"] = (
            "relative_evidence__* is a within-observation softmax used for "
            "ranking and visualization; it is not a calibrated biological "
            "probability. Empirical subtype q-values control support."
        )
    except Exception:
        pass
    t_rapx1 = __import__('time').perf_counter()
    print(
        f"[RAP-X timing] annotation took {t_rapx1 - t_rapx0:0.2f}s "
        f"(annotation_groups={len(annot_df)}, "
        f"score_columns={len(annot_df.columns)})"
    )
    # Save RAP-X diagnostics plots and annotation scores
    try:
        # _plot_rapx_enrichment_and_quality(annot_df, adata2, out_dir=out_dir)
        _save_dataframe(annot_df.reset_index(), out_dir or os.getcwd(), "rapx_annotation_scores")
        # Keep object labels separate from the Leiden-level majority label.
        try:
            if 'subcellular_annotation' in adata2.obs and 'leiden' in adata2.obs:
                _refresh_subcellular_group_columns(
                    adata2,
                    label_col='subcellular_annotation',
                )
        except Exception:
            pass
        # Additional UMAP annotation visuals (optional; default off for speed)
        if ann_save_plots:
            try:
                import scanpy as sc
                import matplotlib.pyplot as plt
                # Ensure UMAP exists (compute using an available neighbors graph)
                if 'X_umap' not in adata2.obsm:
                    cand = ['expr_nbrs','dgi_sage_nbrs','dgi_gcn_nbrs','pca_nbrs']
                    used = None
                    for k in cand:
                        if f"{k}_connectivities" in adata2.obsp:
                            try:
                                sc.tl.umap(**dict(adata=adata2, neighbors_key=k))
                                used = k
                                break
                            except Exception:
                                continue
                    if 'X_umap' not in adata2.obsm:
                        try:
                            sc.pp.neighbors(adata2, use_rep='X_pca', n_neighbors=32)
                            sc.tl.umap(**dict(adata=adata2))
                        except Exception:
                            pass
                # Plot and save
                group_color = (
                    "leiden_embedding"
                    if "leiden_embedding" in adata2.obs.columns
                    else "leiden"
                )
                umap_colors = [
                    column
                    for column in [
                        'subcell_core_candidate_label',
                        'subcellular_annotation_strict',
                        'subcellular_annotation',
                        'subcellular_annotation_tier',
                        group_color,
                    ]
                    if column in adata2.obs.columns
                ]
                sc.pl.umap(
                    adata2,
                    color=umap_colors,
                    legend_fontsize='small',
                    wspace=0.5,
                    show=False,
                )
                _save_fig(plt.gcf(), out_dir or os.getcwd(), "rapx_umap_cluster_annotation")
                plt.close()
            except Exception:
                pass
    except Exception:
        pass

    omm_refinement_meta: Optional[Dict[str, Any]] = None
    if omm_metaspot_refinement:
        from .omm_refinement import (
            OMM_REFINEMENT_COLUMNS,
            build_omm_contrast_marker_weights,
            refine_omm_metaspots,
        )

        if apex_omm_audit is not None:
            omm_contrast_reference = apex_omm_audit[
                [
                    "Common_Gene",
                    "OMM_log2FC",
                    "ERM_log2FC",
                    "Cytosol_log2FC",
                ]
            ].rename(
                columns={
                    "OMM_log2FC": "OMM",
                    "ERM_log2FC": "ER",
                    "Cytosol_log2FC": "Cytosol",
                }
            )
            contrast_min_weight = float(apex_omm_min_log2fc)
            contrast_min_margin = float(apex_omm_min_delta)
        else:
            omm_contrast_reference = gold_aug.copy()
            if "Common_Gene" not in omm_contrast_reference.columns:
                omm_contrast_reference = (
                    omm_contrast_reference.reset_index()
                )
            contrast_min_weight = 0.75
            contrast_min_margin = 0.50

        (
            omm_positive_weights,
            omm_alternative_weights,
            omm_marker_meta,
        ) = build_omm_contrast_marker_weights(
            omm_contrast_reference,
            min_region_weight=contrast_min_weight,
            min_margin=contrast_min_margin,
        )
        if "subcell_omm_domain_state" in adata2.obs.columns:
            omm_domain_states = pd.Series(
                adata2.obs["subcell_omm_domain_state"].astype(str).to_numpy(),
                index=adata2.obs_names.astype(str),
                dtype="object",
            ).to_dict()
        else:
            omm_domain_states = {}
        adata1, final_df, omm_refinement_meta = refine_omm_metaspots(
            adata1,
            final_df,
            omm_positive_weights,
            omm_alternative_weights,
            domain_states=omm_domain_states,
            min_informative_transcripts=(
                omm_metaspot_min_informative_transcripts
            ),
            shrinkage_count=omm_metaspot_shrinkage_count,
            posterior_high=omm_metaspot_posterior_high,
            posterior_low=omm_metaspot_posterior_low,
            random_state=0,
            copy_adata=False,
        )
        omm_refinement_meta["marker_builder"] = omm_marker_meta
        omm_refinement_meta["applied_to_primary_annotation"] = bool(
            omm_metaspot_apply_to_primary
            and omm_refinement_meta.get("identifiable", False)
        )
        omm_refinement_meta["ambiguous_primary_label_policy"] = (
            omm_metaspot_ambiguous_policy
        )
        if (
            omm_refinement_meta.get("identifiable", False)
            and "subcell_statistically_supported__OMM"
            in adata2.obs.columns
            and "subcell_omm_specificity_margin" in adata2.obs.columns
            and "meta-domain" in adata1.obs.columns
        ):
            domain_index = pd.Index(adata2.obs_names.astype(str))
            statistically_supported_omm = _coerce_bool_series(
                adata2.obs["subcell_statistically_supported__OMM"],
                default=False,
            )
            statistically_supported_omm.index = domain_index
            domain_margin = pd.to_numeric(
                adata2.obs["subcell_omm_specificity_margin"],
                errors="coerce",
            )
            domain_margin.index = domain_index
            spot_domain = adata1.obs["meta-domain"].astype(str)
            posterior = pd.to_numeric(
                adata1.obs["omm_reference_posterior"],
                errors="coerce",
            ).to_numpy(dtype=float)
            spot_counts = np.asarray(adata1.X.sum(axis=1)).ravel()
            total_counts = float(spot_counts.sum())
            sensitivity_rows = []
            for sensitivity_margin in (0.0, 0.25, 0.5, 0.75, 1.0):
                domain_confirmed = (
                    statistically_supported_omm
                    & domain_margin.ge(sensitivity_margin)
                )
                spot_confirmed = (
                    spot_domain.map(domain_confirmed)
                    .fillna(False)
                    .to_numpy(dtype=bool)
                )
                finite = np.isfinite(posterior)
                soft_weights = np.where(
                    spot_confirmed & finite,
                    posterior,
                    0.0,
                )
                hard_mask = (
                    spot_confirmed
                    & finite
                    & (
                        posterior
                        >= float(omm_metaspot_posterior_high)
                    )
                )
                sensitivity_rows.append(
                    {
                        "required_domain_margin": float(
                            sensitivity_margin
                        ),
                        "confirmed_domains": int(domain_confirmed.sum()),
                        "confirmed_metaspots_with_posterior": int(
                            np.sum(spot_confirmed & finite)
                        ),
                        "transcript_weighted_soft_confirmed": (
                            float(
                                np.dot(spot_counts, soft_weights)
                                / total_counts
                            )
                            if total_counts > 0
                            else None
                        ),
                        "transcript_weighted_hard_posterior_high": (
                            float(
                                spot_counts[hard_mask].sum()
                                / total_counts
                            )
                            if total_counts > 0
                            else None
                        ),
                    }
                )
            omm_refinement_meta[
                "domain_specificity_margin_sensitivity"
            ] = {
                str(column): [
                    row.get(column)
                    for row in sensitivity_rows
                ]
                for column in sensitivity_rows[0]
            }

        def _uns_safe(value):
            if value is None:
                return "not_available"
            if isinstance(value, dict):
                return {
                    str(key): _uns_safe(item)
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                if value and all(
                    isinstance(item, dict) for item in value
                ):
                    return {
                        f"item_{index}": _uns_safe(item)
                        for index, item in enumerate(value)
                    }
                return [_uns_safe(item) for item in value]
            return value

        safe_omm_meta = _uns_safe(omm_refinement_meta)
        adata2.uns["omm_metaspot_refinement"] = safe_omm_meta
        adata1.uns["omm_metaspot_refinement"] = safe_omm_meta
        if out_dir is not None:
            refinement_table = adata1.obs[
                [
                    column
                    for column in OMM_REFINEMENT_COLUMNS
                    if column in adata1.obs.columns
                ]
            ].copy()
            refinement_table.insert(
                0,
                "metaspot",
                adata1.obs.get(
                    "metaspot",
                    pd.Series(
                        adata1.obs_names,
                        index=adata1.obs_names,
                    ),
                ).astype(str).to_numpy(),
            )
            _save_dataframe(
                refinement_table.reset_index(drop=True),
                out_dir,
                "omm_metaspot_refinement",
            )
            prevalence_values = omm_refinement_meta.get(
                "soft_prevalence",
                {},
            )
            prevalence_intervals = omm_refinement_meta.get(
                "soft_prevalence_cell_bootstrap_95ci",
                {},
            )
            prevalence_rows = []
            for metric, estimate in prevalence_values.items():
                interval = prevalence_intervals.get(metric) or {}
                prevalence_rows.append(
                    {
                        "metric": str(metric),
                        "estimate": estimate,
                        "lower_95_cell_bootstrap": interval.get(
                            "lower_95"
                        ),
                        "upper_95_cell_bootstrap": interval.get(
                            "upper_95"
                        ),
                        "interpretation": (
                            "posterior_coverage"
                            if "coverage" in str(metric)
                            else (
                                "confirmed_lower"
                                if "confirmed_or_shared" not in str(metric)
                                else (
                                    "confirmed_plus_shared_upper_envelope"
                                )
                            )
                        ),
                    }
                )
            _save_dataframe(
                pd.DataFrame(prevalence_rows),
                out_dir,
                "omm_soft_prevalence_summary",
            )
        prevalence = omm_refinement_meta.get("soft_prevalence", {})
        _logging.getLogger("cellscope").info(
            "OMM MetaSpot refinement identifiable=%s; confirmed soft "
            "transcript prevalence=%s; confirmed+shared envelope=%s",
            omm_refinement_meta.get("identifiable", False),
            prevalence.get("transcript_weighted_soft_confirmed"),
            prevalence.get(
                "transcript_weighted_soft_confirmed_or_shared"
            ),
        )

    # Step 4: map annotation back to final point table via meta-domain
    if 'meta-domain' in final_df.columns:
        # Faster mapping vs merge for large final_df
        m = adata2.obs['subcellular_annotation'].astype(str)
        # Ensure key types align
        final_df = final_df.copy()
        final_df['meta-domain'] = final_df['meta-domain'].astype(str)
        # Vectorized map
        domain_labels = final_df['meta-domain'].map(m)
        final_df['subcellular_annotation_domain'] = pd.Categorical(
            domain_labels
        )
        if "r_signed" in final_df.columns:
            (
                guarded_labels,
                final_df["subcellular_geometry_conflict"],
            ) = _apply_local_compartment_guard(
                domain_labels,
                final_df["r_signed"],
            )
            final_df["subcellular_annotation_geometry_guarded"] = (
                pd.Categorical(guarded_labels)
            )
        else:
            final_df["subcellular_annotation_geometry_guarded"] = (
                pd.Categorical(domain_labels)
            )
            final_df["subcellular_geometry_conflict"] = False
        final_df["subcellular_annotation"] = pd.Categorical(domain_labels)
        for column in (
            "subcellular_annotation_strict",
            "subcellular_annotation_exploratory",
            "subcellular_annotation_tier",
        ):
            if column in adata2.obs.columns:
                final_df[column] = pd.Categorical(
                    final_df["meta-domain"].map(
                        adata2.obs[column].astype(str)
                    )
                )
        if "subcellular_candidate" in adata2.obs.columns:
            candidate_map = adata2.obs["subcellular_candidate"].astype(str)
            final_df["subcellular_candidate"] = final_df["meta-domain"].map(
                candidate_map
            )
        evidence_output_map = {
            "subcell_primary_label": "subcellular_primary_label",
            "subcell_secondary_label": "subcellular_secondary_label",
            "subcell_annotation_state": "subcellular_annotation_state",
            "subcell_supported_labels": "subcellular_supported_labels",
            "subcell_n_supported_labels": "subcellular_n_supported_labels",
            "subcell_primary_evidence_score": (
                "subcellular_primary_evidence_score"
            ),
            "subcell_secondary_evidence_score": (
                "subcellular_secondary_evidence_score"
            ),
            "subcell_evidence_score_gap": (
                "subcellular_evidence_score_gap"
            ),
            "subcell_primary_empirical_q": (
                "subcellular_primary_empirical_q"
            ),
            "subcell_secondary_empirical_q": (
                "subcellular_secondary_empirical_q"
            ),
            "subcell_omm_domain_state": "omm_domain_state",
            "subcell_omm_specificity_margin": (
                "omm_specificity_margin"
            ),
            "subcell_omm_specificity_pass": "omm_specificity_pass",
            "subcell_omm_raw_competitor_label": (
                "omm_raw_competitor_label"
            ),
            "subcell_membrane_ambiguity": "membrane_ambiguity",
        }
        for source, target in evidence_output_map.items():
            if source in adata2.obs.columns:
                final_df[target] = final_df["meta-domain"].map(
                    adata2.obs[source]
                )
        for column in (
            "subcellular_annotation_conservative",
            "subcellular_confidence",
        ):
            if column in adata2.obs.columns:
                final_df[column] = final_df["meta-domain"].map(
                    adata2.obs[column].astype(str)
                )
        if "subcell_low_conf" in adata2.obs.columns:
            confidence_map = _coerce_bool_series(
                adata2.obs["subcell_low_conf"],
                default=True,
            )
            final_df["subcell_low_conf"] = final_df["meta-domain"].map(
                confidence_map
            )

    if "meta-domain" in adata1.obs.columns:
        domain_key = adata1.obs["meta-domain"].astype(str)
        domain_labels = domain_key.map(
            adata2.obs["subcellular_annotation"].astype(str)
        )
        adata1.obs["subcellular_annotation_domain"] = pd.Categorical(
            domain_labels
        )
        if "r_mean" in adata1.obs.columns:
            (
                guarded_labels,
                geometry_conflict,
            ) = _apply_local_compartment_guard(
                domain_labels,
                adata1.obs["r_mean"],
            )
            adata1.obs[
                "subcellular_annotation_geometry_guarded"
            ] = pd.Categorical(
                guarded_labels
            )
            adata1.obs["subcellular_geometry_conflict"] = (
                geometry_conflict.values
            )
        else:
            adata1.obs[
                "subcellular_annotation_geometry_guarded"
            ] = pd.Categorical(
                domain_labels
            )
            adata1.obs["subcellular_geometry_conflict"] = False
        adata1.obs["subcellular_annotation"] = pd.Categorical(
            domain_labels
        )
        for column in (
            "subcellular_annotation_strict",
            "subcellular_annotation_exploratory",
            "subcellular_annotation_tier",
        ):
            if column in adata2.obs.columns:
                adata1.obs[column] = pd.Categorical(
                    domain_key.map(adata2.obs[column].astype(str))
                )
        if "subcellular_candidate" in adata2.obs.columns:
            adata1.obs["subcellular_candidate"] = domain_key.map(
                adata2.obs["subcellular_candidate"].astype(str)
            ).values
        evidence_output_map = {
            "subcell_primary_label": "subcellular_primary_label",
            "subcell_secondary_label": "subcellular_secondary_label",
            "subcell_annotation_state": "subcellular_annotation_state",
            "subcell_supported_labels": "subcellular_supported_labels",
            "subcell_n_supported_labels": "subcellular_n_supported_labels",
            "subcell_primary_evidence_score": (
                "subcellular_primary_evidence_score"
            ),
            "subcell_secondary_evidence_score": (
                "subcellular_secondary_evidence_score"
            ),
            "subcell_evidence_score_gap": (
                "subcellular_evidence_score_gap"
            ),
            "subcell_primary_empirical_q": (
                "subcellular_primary_empirical_q"
            ),
            "subcell_secondary_empirical_q": (
                "subcellular_secondary_empirical_q"
            ),
            "subcell_omm_domain_state": "omm_domain_state",
            "subcell_omm_specificity_margin": (
                "omm_specificity_margin"
            ),
            "subcell_omm_specificity_pass": "omm_specificity_pass",
            "subcell_omm_raw_competitor_label": (
                "omm_raw_competitor_label"
            ),
            "subcell_membrane_ambiguity": "membrane_ambiguity",
        }
        for source, target in evidence_output_map.items():
            if source in adata2.obs.columns:
                adata1.obs[target] = domain_key.map(
                    adata2.obs[source]
                ).values
        for source in adata2.obs.columns:
            if source.startswith(
                (
                    "subcell_evidence_score__",
                    "subcell_raw_evidence_score__",
                    "subcell_relative_evidence__",
                    "subcell_geometry_feasible__",
                    "subcell_statistically_supported__",
                    "subcell_specificity_margin__",
                    "subcell_specificity_pass__",
                    "subcell_empirically_supported__",
                    "subcell_subtype_empirical_q__",
                    "subcell_subtype_calibrated_score__",
                )
            ):
                adata1.obs[source] = domain_key.map(
                    adata2.obs[source]
                ).values
        for column in (
            "subcellular_annotation_conservative",
            "subcellular_confidence",
        ):
            if column in adata2.obs.columns:
                adata1.obs[column] = domain_key.map(
                    adata2.obs[column].astype(str)
                ).values
        if "subcell_low_conf" in adata2.obs.columns:
            confidence_map = _coerce_bool_series(
                adata2.obs["subcell_low_conf"],
                default=True,
            )
            adata1.obs["subcell_low_conf"] = domain_key.map(
                confidence_map
            ).values
        if {
            "cell_id",
            "subcellular_annotation",
            "subcellular_confidence",
        }.issubset(adata1.obs.columns):
            high = (
                adata1.obs["subcellular_confidence"].astype(str) == "High"
            )
            cell_labels = (
                adata1.obs.loc[
                    high,
                    ["cell_id", "subcellular_annotation"],
                ]
                .astype(str)
                .drop_duplicates()
            )
            total_cells = adata1.obs["cell_id"].nunique()
            coverage = (
                cell_labels.groupby(
                    "subcellular_annotation",
                    observed=True,
                )["cell_id"]
                .nunique()
                .sort_values(ascending=False)
            )
            print(
                "\nOriginal-cell coverage by high-confidence domains "
                "(labels are not mutually exclusive):"
            )
            for label, count in coverage.items():
                pct = count / total_cells * 100 if total_cells else 0.0
                print(f"  {label}: {count:,} cells ({pct:.1f}%)")

    if (
        omm_refinement_meta is not None
        and "subcellular_annotation_domain" in adata1.obs.columns
        and "omm_refined_state" in adata1.obs.columns
        and "omm_domain_gate" in adata1.obs.columns
    ):
        metaspot_labels = _compose_omm_metaspot_labels(
            adata1.obs["subcellular_annotation_domain"],
            adata1.obs["omm_refined_state"],
            adata1.obs["omm_domain_gate"],
            adata1.obs.get("omm_raw_competitor_label"),
            identifiable=bool(
                omm_refinement_meta.get("identifiable", False)
            ),
            ambiguous_policy=omm_metaspot_ambiguous_policy,
        )
        adata1.obs["subcellular_annotation_metaspot"] = pd.Categorical(
            metaspot_labels
        )
        if (
            omm_metaspot_apply_to_primary
            and bool(omm_refinement_meta.get("identifiable", False))
        ):
            adata1.obs["subcellular_annotation"] = pd.Categorical(
                metaspot_labels
            )

        if "metaspot" in adata1.obs.columns:
            adata1_metaspot_ids = adata1.obs["metaspot"].astype(str)
        else:
            adata1_metaspot_ids = pd.Series(
                adata1.obs_names.astype(str),
                index=adata1.obs_names,
            )
        label_by_metaspot = pd.Series(
            metaspot_labels.to_numpy(),
            index=pd.Index(adata1_metaspot_ids.to_numpy(), dtype=str),
            dtype="object",
        )
        if not label_by_metaspot.index.is_unique:
            raise ValueError(
                "MetaSpot annotation refinement requires unique MetaSpot IDs"
            )
        final_refined_labels = final_df["metaspot"].astype(str).map(
            label_by_metaspot
        )
        if final_refined_labels.isna().any():
            raise ValueError(
                "Failed to map OMM-refined MetaSpot labels back to every "
                "transcript"
            )
        final_df["subcellular_annotation_metaspot"] = pd.Categorical(
            final_refined_labels
        )
        if (
            omm_metaspot_apply_to_primary
            and bool(omm_refinement_meta.get("identifiable", False))
        ):
            final_df["subcellular_annotation"] = pd.Categorical(
                final_refined_labels
            )

    return adata1, adata2, final_df
