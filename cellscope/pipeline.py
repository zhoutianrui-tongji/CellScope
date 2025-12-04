from typing import Optional, Literal, Tuple, List, Callable, cast, Dict, Any, Union
import os

# Configure BLAS/OMP threads before importing numpy/scipy.
# Use a balanced default based on CPU cores, but allow user overrides via env.
_cpu = os.cpu_count() or 8
_threads_default = str(min(32, max(64, _cpu // 2)))
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
del _THREAD_ENV_DEFAULTS, _k, _v, _cpu, _threads_default

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
import scipy.sparse as sp
import gc

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
            return pd.read_parquet(pq_path)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
    except Exception:
        return None
    return None


def _load_adata_checkpoint(path: str) -> Optional[ad.AnnData]:
    if not os.path.exists(path):
        return None
    try:
        return ad.read_h5ad(path)
    except Exception:
        return None


def _resume_build_outputs(out_dir: str) -> Optional[Tuple[ad.AnnData, Optional[ad.AnnData], Optional[pd.DataFrame]]]:
    base = os.path.join(out_dir, "intermediate")
    a1_path = os.path.join(base, "module5_adata1.h5ad")
    adata1 = _load_adata_checkpoint(a1_path)
    if adata1 is None:
        return None
    adata2 = _load_adata_checkpoint(os.path.join(base, "module6_adata2.h5ad"))
    final_df = _load_dataframe_checkpoint(base, "module5_final_data")
    return adata1, adata2, final_df


def _resume_annotation_outputs(out_dir: str) -> Optional[Tuple[Optional[ad.AnnData], Optional[pd.DataFrame]]]:
    base = os.path.join(out_dir, "intermediate")
    final_df = _load_dataframe_checkpoint(base, "annotated_final_csv")
    if final_df is None:
        return None
    ann_adata2 = _load_adata_checkpoint(os.path.join(base, "annotated_adata2.h5ad"))
    return ann_adata2, final_df


def preprocess(spatial_df: pd.DataFrame, cell_df: pd.DataFrame, nucleus_df: pd.DataFrame, show_internal_progress: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Keep only cells where `overlaps_nucleus` has both 0 and 1; enforce consistent filtering across the three input tables.
    """
    unique_counts = (
        spatial_df.groupby("cell_id")["overlaps_nucleus"].nunique().reset_index(name="n_unique")
    )
    valid_cells = unique_counts.query("n_unique == 2")["cell_id"]

    spatial_df = spatial_df.query("cell_id in @valid_cells").copy()
    cell_df = cell_df.query("cell_id in @valid_cells").copy()
    nucleus_df = nucleus_df.query("cell_id in @valid_cells").copy()

    return spatial_df, cell_df, nucleus_df


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

    df = transcripts_df[transcripts_df['cell_id'].isin(sampled_cells)].copy()
    cell_kd = _build_kdtrees(cell_boundaries_df, sampled_cells, desc="Cell KDTree")
    nuc_kd = _build_kdtrees(nucleus_boundaries_df, sampled_cells, desc="Nucleus KDTree")
    nuc_paths = _build_nucleus_paths(nucleus_boundaries_df, sampled_cells)

    df['d_cell'] = _compute_min_dist(df, cell_kd, 'd_cell')
    df['d_nuc'] = _compute_min_dist(df, nuc_kd, 'd_nuc')

    flags = []
    it = df.groupby('cell_id', sort=False)
    iterator = _progress_iter(it, desc='Point-in-nucleus', total=it.ngroups, show=show_internal_progress)
    for cid, g in iterator:
        p = nuc_paths.get(cid, None)
        if p is None:
            flags.append(pd.Series(False, index=g.index))
        else:
            pts = g[['x_location', 'y_location']].to_numpy()
            flags.append(pd.Series(p.contains_points(pts), index=g.index))
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

    Q_LOW, Q_HIGH = 5.0, 95.0
    ZERO_BAND = 0.05

    def _stretch_per_cell_v2(v: np.ndarray, cell_size: int) -> np.ndarray:
        if cell_size < 16:
            x = v
        else:
            lo, hi = np.nanpercentile(v, [Q_LOW, Q_HIGH])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                x = v
            else:
                x = (v - lo) / max(hi - lo, 1e-6) * 2 - 1
        x = np.clip(x, -2, 2)
        adaptive_band = ZERO_BAND * (1.0 + 0.5 * np.exp(-cell_size / 1000))
        x[np.abs(x) < adaptive_band] = 0.0
        return x.astype(DTYPE)

    df['_rs_tmp'] = rs_norm.astype(DTYPE, copy=False)
    cell_sizes = df.groupby('cell_id').size()
    df['_cell_size'] = df['cell_id'].map(cell_sizes)
    df['r_signed'] = df.groupby('cell_id', sort=False).apply(
        lambda g: pd.Series(
            _stretch_per_cell_v2(g['_rs_tmp'].values, len(g)),
            index=g.index
        ),
        include_groups=False
    ).droplevel(0).astype(DTYPE)
    df.drop(columns=['_rs_tmp', '_nuc_r', '_cell_size'], errors='ignore', inplace=True)

    _winsorize_inplace(df, ['d_cell', 'd_nuc', 'r_signed', 'd_cell_norm', 'd_nuc_norm'], winsor_perc)
    return df


def build_composition_embeddings(point_df: pd.DataFrame, transcripts_df: pd.DataFrame, sampled_cell_ids: List, show_internal_progress: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    # Build vocabulary
    sub_tr_all = transcripts_df[transcripts_df['cell_id'].isin(sampled_cell_ids)]
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
    except Exception:
        pass
    vocab_genes = gene_counts.head(TOP_G).index.astype(str).tolist()
    gene2idx = {g: i for i, g in enumerate(vocab_genes)}

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
    RANDOM_STATE = 42
    rng = np.random.default_rng(RANDOM_STATE)
    sample_n = min(200_000, len(sub_tr_all))
    sample_idx = sub_tr_all.sample(n=sample_n, random_state=RANDOM_STATE).index.to_numpy()

    from joblib import Parallel, delayed
    # speed-oriented parallel settings (bounded without env vars)
    N_JOBS = max(2, min(8, _preferred_n_jobs()))
    _backend = _joblib_backend()

    groups = list(sub_tr_all.loc[sample_idx].groupby('cell_id', sort=False))

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
    iterator = _progress_iter(range(0, len(groups), 128), desc='SVD sample TF (parallel)', total=len(list(range(0, len(groups), 128))), show=show_internal_progress)
    for chunk_start in iterator:
        chunk = groups[chunk_start:chunk_start + 128]
        out = Parallel(n_jobs=N_JOBS, backend=_backend, verbose=0)(
            delayed(_svd_tf_block)(pair) for pair in chunk
        )
        for tf in out:
            if tf is None:
                continue
            bow_samples.append(tf)
            total_rows += tf.shape[0]
            if total_rows >= MAX_SVD_ROWS:
                break
        if total_rows >= MAX_SVD_ROWS:
            break

    if len(bow_samples) == 0:
        raise RuntimeError('SVD samples empty.')
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
    svd_local = TruncatedSVD(n_components=N_COMP, random_state=RANDOM_STATE)
    svd_local.fit(B_sample)

    # Embed all rows (full pass)
    sub_tr = sub_tr_all.copy()
    sub_tr['_rowid_'] = np.arange(len(sub_tr), dtype=np.int64)
    comp_emb = np.zeros((len(sub_tr), N_COMP), dtype=np.float32)
    local_deg = np.zeros(len(sub_tr), dtype=np.float32)
    groups2 = list(sub_tr.groupby('cell_id', sort=False))

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
        k_used = min(K_NEI, len(dfc))
        if dist is not None and dist.shape[1] >= k_used:
            rk = dist[:, k_used - 1].astype(np.float32) + 1e-3
            dens = (k_used) / (np.pi * rk * rk)
        else:
            dens = np.full(len(dfc), k_used, np.float32)
        rows = dfc['_rowid_'].to_numpy()
        return rows, Z, dens

    chunk_size2 = _cluster_chunk_size()
    iterator2 = _progress_iter(range(0, len(groups2), chunk_size2), desc='Per-cell KNN & BoW (parallel)', total=len(list(range(0, len(groups2), chunk_size2))), show=show_internal_progress)
    for chunk_start in iterator2:
        chunk = groups2[chunk_start:chunk_start + chunk_size2]
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
            comp_emb[rows] = Z
            local_deg[rows] = dens

    comp_cols = [f'comp_emb{i}' for i in range(N_COMP)]
    emb_df = pd.DataFrame(comp_emb, index=sub_tr.index, columns=comp_cols)
    emb_df['local_density'] = local_deg
    point_df = point_df.join(emb_df, how='left')

    # Combine geometry and composition features to produce GMM inputs
    geom_cols = ['r_signed', 'd_cell_norm', 'd_nuc_norm'] + (
        ['sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta'] if 'sin_theta' in point_df.columns else []
    )
    all_needed = geom_cols + comp_cols + ['local_density']
    F_raw = point_df[all_needed].to_numpy(dtype=np.float32, copy=False)
    _sc_dom = StandardScaler()
    F_all = _sc_dom.fit_transform(np.nan_to_num(F_raw, nan=0.0, posinf=0.0, neginf=0.0))
    gmm_cols = [f"gmm_feat_{i}" for i in range(F_all.shape[1])]
    point_df[gmm_cols] = F_all
    del B_sample, bow_samples, comp_emb, local_deg, F_raw
    gc.collect()
    return point_df, gmm_cols


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


def per_cell_clustering(
    point_df: pd.DataFrame,
    gmm_cols: List[str],
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
) -> pd.DataFrame:
    import hdbscan  # lazy import

    MIN_SUB_PTS = int(hdbscan_min_cluster_size)
    # 使用更小的目标簇大小以促进细分，且不与最小簇大小绑定
    TARGET_SUB_SIZE = max(8, int(target_sub_size))
    K_CAP_BASE = 999
    MIN_SAMPLES_FRAC = float(hdbscan_min_samples_frac)

    # Select features with the highest variance
    X_gmm_sample = point_df[gmm_cols].to_numpy(np.float32, copy=False)
    X_gmm_sample[~np.isfinite(X_gmm_sample)] = 0.0
    feat_var = X_gmm_sample.var(axis=0)
    order = np.argsort(-feat_var)
    N_FEAT_HDB = min(int(n_feat_hdb), len(gmm_cols))
    idx_use = order[:N_FEAT_HDB]
    feat_for_hdb = [gmm_cols[i] for i in idx_use]
    del X_gmm_sample, feat_var, order
    gc.collect()

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

    # Per-cell clustering (parallel batching)

    groups = list(point_df.groupby('cell_id', sort=False))

    def _fit_one(args):
        cid, g = args
        g = g.dropna(subset=feat_for_hdb)
        if len(g) == 0:
            return cid, pd.Series([], dtype=np.int32, index=g.index)
        n = len(g)
        if n < MIN_SUB_PTS:
            labels = np.zeros(n, dtype=np.int32)
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)
        X_full = g[feat_for_hdb].to_numpy(np.float32, copy=False)
        Xz, good = _safe_standardize(X_full)
        if Xz.shape[1] == 0 or not np.any(good):
            X_coord = g[['x_location', 'y_location']].to_numpy(np.float32, copy=False)
            Xz, _ = _safe_standardize(X_coord)
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
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)
        labels = _assign_noise_to_nearest(hdb_labels, Xz)
        _min_size = int(merge_min_size) if merge_min_size is not None else int(MIN_SUB_PTS)
        labels = _merge_small_clusters(labels, Xz, min_size=_min_size)
        return cid, pd.Series(labels, index=g.index, dtype=np.int32)

    from joblib import Parallel, delayed
    N_JOBS = _preferred_n_jobs()
    _backend = _joblib_backend()
    if os.environ.get("CELLSCOPE_FORCE_THREADS", "").strip():
        _backend = "threading"

    # prefer larger chunk to reduce scheduler overhead
    chunk_size = max(64, _cluster_chunk_size())
    results = []
    rng_chunks = range(0, len(groups), chunk_size)
    iterator = _progress_iter(rng_chunks, desc='Per-cell clustering (parallel)', total=len(list(rng_chunks)), show=show_internal_progress)
    for chunk_start in iterator:
        chunk = groups[chunk_start:chunk_start + chunk_size]
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
    point_df['cluster_in_cell'] = labels_per_cell
    missing = point_df['cluster_in_cell'].isna()
    if missing.any():
        point_df.loc[missing, 'cluster_in_cell'] = -1
    point_df['cluster_in_cell'] = point_df['cluster_in_cell'].astype(np.int32)
    point_df['cell_sub'] = (
        point_df['cell_id'].astype(str) + ':' + point_df['cluster_in_cell'].astype(np.int32).astype(str)
    )
    return point_df


def build_adata1(point_df: pd.DataFrame, show_internal_progress: bool = False) -> ad.AnnData:
    cell_sub_cat = pd.Categorical(point_df['cell_sub'], ordered=False)
    row_codes = cell_sub_cat.codes.astype(np.int64)
    row_cats = cell_sub_cat.categories
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
        'cell_sub': row_cats,
        'r_mean': r_mean, 'r_std': r_std,
        'r_p10': r_p10, 'r_p50': r_p50, 'r_p90': r_p90,
        'd_cell_norm_mean': d1_mean, 'd_nuc_norm_mean': d2_mean,
        'elongation': elong, 'spread': spread, 'log_n': log_n
    }).set_index('cell_sub')
    adata1.obs = adata1.obs.join(md_stats, how='left')
    emb_cols = [c for c in point_df.columns if c.startswith('comp_emb')]
    if len(emb_cols):
        agg_comp = (point_df.groupby('cell_sub')[emb_cols + ['local_density']].agg(['mean', 'std']))
        agg_comp.columns = [f"{a}_{b}" for a, b in agg_comp.columns]
        adata1.obs = adata1.obs.join(agg_comp, how='left')
    return adata1


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
    cs = pd.Categorical(point_df["cell_sub"], categories=adata1.obs_names, ordered=False)
    x = point_df["x_location"].to_numpy(np.float64, copy=False)
    y = point_df["y_location"].to_numpy(np.float64, copy=False)
    df = pd.DataFrame({"cell_sub": cs, "x": x, "y": y})
    agg = (
        df.groupby("cell_sub", observed=True)
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
        tmp = point_df[["cell_sub", "cell_id"]].drop_duplicates().set_index("cell_sub")
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
        y = _mini_batch_kmeans_simple(
            Xi, K=K_blk, seed=42 + (0 if prefix == 'N' else 1), desc=f"[{prefix}] MBK K={K_blk}"
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
    # 2) cell_sub -> cluster mapping
    t0 = _step_start("Map metaspots to clusters")
    sub2clu = adata1.obs['meta-domain'].astype(str).values
    clu_codes, clu_uni = pd.factorize(sub2clu, sort=False)
    _step_done("Map metaspots to clusters", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 3) group mean to clusters
    t0 = _step_start("Aggregate metaspots -> cluster means")
    ones = np.ones_like(clu_codes, dtype=np.float32)
    G = sp.csr_matrix((ones, (clu_codes, np.arange(len(clu_codes)))), shape=(len(clu_uni), len(clu_codes)))
    row_sum = np.array(G.sum(1)).ravel(); row_sum[row_sum == 0] = 1.0
    X2 = (G.multiply(1.0 / row_sum[:, None]) @ X1_norm).tocsr()
    adata2 = ad.AnnData(X=X2, obs=pd.DataFrame(index=clu_uni), var=adata1.var.copy())
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
    feature_scale_mode = 'standard'

    try:
        _cfg = load_params_yaml()
        _m7 = _cfg.get('module7', {}) if _cfg else {}
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
    except Exception:
        pass

    def compute_expr_latent_scvi_or_svd(adata2: ad.AnnData, n_latent: int = 32, max_epochs: int = 50) -> Tuple[np.ndarray, str, Dict[str, Any]]:
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
                _warnings.filterwarnings("ignore", message="adata.X does not contain unnormalized count data", category=UserWarning)
                scvi.model.SCVI.setup_anndata(adata2)
            model = scvi.model.SCVI(adata2, n_latent=n_latent)
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message="adata.X does not contain unnormalized count data", category=UserWarning)
                _warnings.filterwarnings("ignore", message="The value argument must be within the support of the distribution", category=UserWarning)
                model.train(
                    max_epochs=max_epochs,
                    batch_size=512,
                    plan_kwargs={"lr": 1e-3},
                    check_val_every_n_epoch=None,
                    enable_progress_bar=False,
                    logger=False,
                    enable_model_summary=False,
                    log_every_n_steps=0,
                )
            Z = model.get_latent_representation().astype(np.float32)
            return Z, "SCVI", {"n_latent": int(n_latent), "epochs": int(max_epochs), "lr": 1e-3, "batch_size": 512}
        except Exception:
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
    Z_expr, _expr_method, _expr_params = compute_expr_latent_scvi_or_svd(adata2, n_latent=scvi_n_latent, max_epochs=scvi_epochs)
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

    sampled_cell_ids = spatial_df['cell_id'].tolist()
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
    gmm_cols: List[str] = []
    _m2_recomputed = False
    if _mod_lt('m2', start_mod):
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module2_point_df")
        if pd_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 2 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 2 cache (module2_point_df)")
            except Exception:
                pass
            point_df = pd_cache
            gmm_cols = [c for c in point_df.columns if str(c).startswith("gmm_feat_")]
        else:
            if progress_callback: progress_callback("Module 2 · Composition Embeddings")
            point_df, gmm_cols = build_composition_embeddings(point_df, spatial_df, sampled_cell_ids, show_internal_progress=show_internal_progress)
            _save_dataframe(point_df, out_dir, "module2_point_df")
            _m2_recomputed = True
    else:
        if progress_callback: progress_callback("Module 2 · Composition Embeddings")
        point_df, gmm_cols = build_composition_embeddings(point_df, spatial_df, sampled_cell_ids, show_internal_progress=show_internal_progress)
        _save_dataframe(point_df, out_dir, "module2_point_df")
        _m2_recomputed = True
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
    # Module 3: per-cell clustering
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
    try:
        _cfg = load_params_yaml()
        _m3 = _cfg.get('module3', {}) if _cfg else {}
        if isinstance(_m3.get('hdbscan_min_cluster_size'), (int, float)):
            _m3_hdb_min = int(_m3.get('hdbscan_min_cluster_size'))
        if 'hdbscan_allow_single_cluster' in _m3:
            _m3_allow_single = bool(_m3.get('hdbscan_allow_single_cluster'))
        if isinstance(_m3.get('hdbscan_min_samples_frac'), (int, float)):
            _m3_min_samples_frac = float(_m3.get('hdbscan_min_samples_frac'))
            _m3_target_sub_size = int(_m3.get('target_sub_size')) if _m3.get('target_sub_size') is not None else _m3_target_sub_size
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
    except Exception:
        pass
    import time as _time
    _t_m3 = _time.perf_counter()
    if _mod_lt('m3', start_mod):
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
            if progress_callback: progress_callback("Module 3 · Per-cell Clustering")
            point_df = per_cell_clustering(
                point_df,
                gmm_cols,
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
            )
            _save_dataframe(point_df, out_dir, "module3_point_df")
    else:
        if progress_callback: progress_callback("Module 3 · Per-cell Clustering")
        point_df = per_cell_clustering(
            point_df,
            gmm_cols,
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
        )
        _save_dataframe(point_df, out_dir, "module3_point_df")
    try:
        _cells = int(point_df['cell_id'].nunique()) if 'cell_id' in point_df.columns else None
    except Exception:
        _cells = None
    _dt_m3 = _time.perf_counter() - _t_m3
    if show_internal_progress:
        msg = f"<< Done: Module 3 (Per-cell clustering) in {_dt_m3:0.2f}s" + (f" (cells={_cells})" if _cells is not None else "")
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
        vc = point_df.groupby('cell_id')['cluster_in_cell'].nunique()
        fig = plt.figure(figsize=(6,4))
        ax = vc.plot.hist(bins=50)
        ax.set_xlabel("n_metaspots per cell"); ax.set_ylabel("count"); plt.tight_layout()
        _save_fig(fig, out_dir or os.getcwd(), "module3_metaspots_per_cell_hist")
        plt.close(fig)
    except Exception:
        pass
    # Module 4: adata1 from point_df (derived; rerun if m1–m3 changed)
    _m4_recomputed = False
    if _mod_lt('m4', start_mod):
        a1_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module4_adata1.h5ad"))
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
    if _mod_lt('m5', start_mod):
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
    # Stats: transcripts count per meta-domain
    try:
        vc = point_df.groupby("meta-domain")["transcript_id"].count()
        stats = vc.describe().to_frame(name="value")
        _save_dataframe(stats.reset_index(), out_dir or os.getcwd(), "module5_domain_transcript_stats")
    except Exception:
        pass
    # Ensure 'cell_sub' exists in adata1.obs for downstream merge
    if 'cell_sub' not in adata1.obs.columns:
        adata1.obs = adata1.obs.copy()
        adata1.obs['cell_sub'] = adata1.obs.index.astype(str)
    # Attach meta-domain to point_df
    if "meta-domain" in point_df.columns:
        point_df = point_df.drop(columns=["meta-domain"])  # ensure fresh
    point_df = point_df.merge(
        adata1.obs[["cell_sub", "meta-domain"]], on="cell_sub", how="left"
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
    if _mod_lt('m6', start_mod):
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
    if _mod_lt('m7', start_mod):
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
    if _mod_lt('m8', start_mod):
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
    try:
        import logging as _logging
        _logging.getLogger("cellscope").info("Saving intermediate: %s", f"{name}.csv" if write_csv else (f"{name}.parquet" if write_pq else name))
    except Exception:
        pass
    if write_csv:
        csv_path = os.path.join(base, f"{name}.csv")
        try:
            df.to_csv(csv_path, index=False)
        except Exception:
            pass
    if write_pq:
        try:
            pq_path = os.path.join(base, f"{name}.parquet")
            df.to_parquet(pq_path, index=False)
        except Exception:
            pass
    try:
        import logging as _logging
        _logging.getLogger("cellscope").info("Saved intermediate: %s%s", name, " (csv)" if write_csv else "")
    except Exception:
        pass

def _preferred_n_jobs() -> int:
    """Return preferred parallel worker count.

    Env vars (checked in order):
      CELLSCOPE_N_JOBS         -> explicit override
      CELLSCOP3E_N_JOBS        -> (legacy typo) still honored

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
        import multiprocessing
        cpu = max(1, multiprocessing.cpu_count())
    except Exception:
        cpu = 4
    return max(1, min(32, cpu // 3 if cpu >= 3 else 1))

def _joblib_backend() -> str:
    # Default backend: 'loky' for maximum throughput on CPU-bound tasks.
    # Users can override via CELLSCOPE_JOBLIB_BACKEND (e.g., to 'threading').
    return os.environ.get("CELLSCOPE_JOBLIB_BACKEND", "loky")

def _cluster_chunk_size() -> int:
    """Chunk size for per-cell parallel loops.

    Env: CELLSCOPE_CLUSTER_CHUNK (int, default 128)
    Smaller chunks reduce peak memory (fewer simultaneous process payloads)
    at the cost of higher scheduler overhead.
    """
    try:
        v = int(os.environ.get("CELLSCOPE_CLUSTER_CHUNK", "32"))
        return max(8, min(512, v))
    except Exception:
        return 32


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

def _save_adata(adata: ad.AnnData, out_dir: str, name: str, compression: str = "lzf"):
    if not _io_should_write_intermediate():
        return
    try:
        p = os.path.join(_intermediate_dir(out_dir), f"{name}.h5ad")
        try:
            import logging as _logging
            _logging.getLogger("cellscope").info("Saving intermediate: %s", f"{name}.h5ad")
        except Exception:
            pass
        adata.write(p, compression=compression)
        try:
            import logging as _logging
            _logging.getLogger("cellscope").info("Saved intermediate: %s", f"{name}.h5ad")
        except Exception:
            pass
    except Exception:
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
    scvi_epochs: int = 100,
    dgi_epochs: int = 200,
    enable_dgi_sage: bool = False,
    resume_policy: Literal["auto","minimal","force"] = "minimal",
    dry_run_diff: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
    """
    Execute CellScope pipeline.

    Returns dict with output file paths.
    """
    ensure_dir(out_dir)
    logger = setup_logger(out_dir)
    logger.info("CellScope pipeline start")
    # Load config early to allow overriding some runtime I/O defaults
    cfg = load_params_yaml()
    try:
        io_cfg = cfg.get('io', {}) if cfg else {}
        # Allow YAML to override defaults (CLI flags still take precedence if supplied explicitly by caller)
        if isinstance(io_cfg.get('final_format'), str):
            final_format = str(io_cfg.get('final_format')).lower()
        if isinstance(io_cfg.get('h5ad_compression'), str):
            compression = str(io_cfg.get('h5ad_compression'))
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
    # Cap BLAS/OMP threads to avoid OpenBLAS NUM_THREADS crash on many-core hosts
    try:
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
                except Exception:
                    comp = compression
                _save_adata(adata1, out_dir, "module5_adata1", compression=comp)
                if adata2 is not None:
                    _save_adata(adata2, out_dir, "module6_adata2", compression=comp)
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

    # ==== Stage 4: Annotation (optional) ====
    run_annotation = enable_annotation and adata2 is not None
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
        adata1.write(adata1_path, compression=compression)
        if adata2 is not None:
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
        "final_csv": final_csv_path if final_format == "csv" else None,
    }


# ====== Optional K1/K2: Clustering + RNALocate/APEX-based annotation ======

def _try_import_scanpy():
    try:
        import scanpy as sc  # noqa: F401
        return True
    except Exception:
        return False


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

    # Ensure counts in layer
    if 'counts' not in adata2.layers:
        adata2.layers['counts'] = adata2.X.copy()
    counts_mat = adata2.layers.get('counts', adata2.X)
    if not _sparse.issparse(counts_mat):
        counts_mat = _sparse.csr_matrix(counts_mat)
    cell_sums = counts_mat.sum(axis=1).A1
    idx_keep = _np.where(cell_sums > 0)[0]
    if idx_keep.size < adata2.n_obs:
        adata2 = adata2[idx_keep].copy()
        counts_mat = counts_mat[idx_keep, :]
        adata2.layers['counts'] = counts_mat

    sc.pp.normalize_total(adata2, target_sum=1e4, layer='counts', inplace=True)
    adata2.layers['log1p'] = adata2.layers['counts'].copy()
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
    if rep_priority is None:
        rep_priority = ['X_dgi_gcn', 'X_dgi_sage', 'X_expr', 'X_pca']
    chosen_key = None
    if cluster_rep:
        rep = str(cluster_rep).strip()
        if rep == 'X_pca' and 'pca_nbrs_connectivities' in adata2.obsp:
            chosen_key = 'pca_nbrs'
        else:
            if rep == 'X_dgi_gcn':
                suf = 'gcn_nbrs'
            elif rep == 'X_dgi_sage':
                suf = 'sage_nbrs'
            else:
                suf = f'{rep[2:]}_nbrs' if rep.startswith('X_') else rep
            if f'{suf}_connectivities' in adata2.obsp:
                chosen_key = suf
    for rep in rep_priority:
        if rep == 'X_pca' and 'pca_nbrs_connectivities' in adata2.obsp:
            chosen_key = 'pca_nbrs'
            break
        if rep == 'X_dgi_gcn':
            suf = 'gcn_nbrs'
        elif rep == 'X_dgi_sage':
            suf = 'sage_nbrs'
        else:
            suf = f'{rep[2:]}_nbrs'
        if f'{suf}_connectivities' in adata2.obsp:
            chosen_key = suf
            break
    if chosen_key is None:
        chosen_key = 'pca_nbrs'
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


def _prepare_gold_matrix(gold_df: pd.DataFrame) -> pd.DataFrame:
    gd = gold_df.copy()
    if 'Common_Gene' in gd.columns and gd.index.name != 'Common_Gene':
        gd = gd.set_index('Common_Gene')
    gd.index = gd.index.astype(str).str.upper()
    num_gd = gd.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna(axis=0, how='all')
    vals = num_gd.to_numpy(dtype=np.float32, copy=True)
    cmin = np.nanmin(vals, axis=0, keepdims=True)
    cmax = np.nanmax(vals, axis=0, keepdims=True)
    span = np.maximum(cmax - cmin, 1e-8)
    vals = (vals - cmin) / span
    num_gd = pd.DataFrame(vals, index=num_gd.index, columns=num_gd.columns)
    num_gd = num_gd.groupby(num_gd.index).max()
    return num_gd


def _build_group_means(adata: ad.AnnData, gold_gene_index_upper: pd.Index):
    from scipy.sparse import csr_matrix as _csr
    from scipy.sparse import issparse as _iss
    if 'leiden' not in adata.obs.columns:
        raise KeyError("adata.obs is missing 'leiden'. Please cluster first and write labels into adata.obs['leiden'].")
    common_genes_up = adata.var_names.astype(str).str.upper()
    mask_common = common_genes_up.isin(gold_gene_index_upper)
    if mask_common.sum() == 0:
        raise RuntimeError("No gene intersection with gold. Please check naming/mapping.")
    X = adata[:, mask_common].X
    leiden_cat = adata.obs['leiden'].astype('category')
    codes = leiden_cat.cat.codes.to_numpy()
    n_groups = int(codes.max()) + 1
    n_obs = adata.n_obs
    M = _csr((np.ones(n_obs, dtype=np.float32), (codes, np.arange(n_obs, dtype=np.int64))), shape=(n_groups, n_obs), dtype=np.float32)
    group_sum = M @ (X if _iss(X) else _csr(X))
    counts = np.maximum(np.bincount(codes, minlength=n_groups).astype(np.float32), 1e-8)
    group_mean = group_sum.multiply(1.0 / counts[:, None]).tocsr()
    cluster_ids = list(leiden_cat.cat.categories)
    genes_up = common_genes_up[mask_common].to_numpy()
    return group_mean, cluster_ids, genes_up


def _robust_gene_stats(group_mean_csr):
    G = group_mean_csr.toarray().astype(np.float32)
    med = np.median(G, axis=0, keepdims=True)
    mad = np.median(np.abs(G - med), axis=0, keepdims=True)
    scale = 1.4826 * (mad + 1e-8)
    S = (G - med) / scale
    S[~np.isfinite(S)] = 0.0
    return S


def _rapx_annotate_core(adata_sub: ad.AnnData, sets: dict, regions: List[str]) -> pd.DataFrame:
    from scipy.stats import rankdata, norm, hypergeom
    from scipy.special import erfc
    # build labels over gold genes
    gold_genes_upper = pd.Index(sorted(set().union(*[set(map(str.upper, s)) for s in sets.values()])))
    group_mean, cluster_ids, genes_up = _build_group_means(adata_sub, gold_genes_upper)
    S = _robust_gene_stats(group_mean)
    Gn = len(genes_up)
    gene_pos = {g: i for i, g in enumerate(genes_up)}
    region_indices, region_weights = {}, {}
    for r in regions:
        idx = np.fromiter((gene_pos[g] for g in sets[r] if g in gene_pos), dtype=np.int64, count=-1)
        idx.sort(); region_indices[r] = idx
        w = np.zeros(Gn, dtype=np.float32)
        if idx.size > 0:
            w[idx] = 1.0 / np.sqrt(max(idx.size, 1))
        region_weights[r] = w

    def z_mwu_from_ranks(ranks: np.ndarray, idx_in: np.ndarray) -> float:
        n1 = idx_in.size; n2 = ranks.size - n1
        if n1 < 5 or n2 < 5: return 0.0
        R_sum = ranks[idx_in].sum(dtype=np.float64)
        U = R_sum - n1 * (n1 + 1) / 2.0
        mean_U = n1 * n2 / 2.0
        std_U  = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        if std_U <= 0 or not np.isfinite(std_U): return 0.0
        return float((U - mean_U) / std_U)

    def p_to_z_right_tail(p: float, cap: float = 8.0) -> float:
        if p <= 0: return cap
        p = max(min(p, 1.0), 1e-300)
        return float(norm.isf(p))

    records = []
    for iC, clu in enumerate(cluster_ids):
        s_vec = S[iC, :].astype(np.float32, copy=False)
        ranks = rankdata(s_vec, method='average').astype(np.float64)
        G_eff = s_vec.size
        K = min(max(250, int(0.3 * G_eff)), G_eff)
        topk_idx = np.argpartition(s_vec, G_eff - K)[-K:]
        topk_mask = np.zeros(G_eff, dtype=bool); topk_mask[topk_idx] = True
        s_std = float(np.std(s_vec)) + 1e-8
        z_comb = np.zeros(len(regions), dtype=np.float32)
        for j, r in enumerate(regions):
            idx_in = region_indices[r]
            if idx_in.size == 0: z_comb[j] = 0.0; continue
            z_mwu = z_mwu_from_ranks(ranks, idx_in)
            z_proj = float(np.dot(s_vec, region_weights[r]) / s_std) if s_std > 0 else 0.0
            x = int(topk_mask[idx_in].sum())
            from scipy.stats import hypergeom as _hyper
            p = float(_hyper.sf(x - 1, G_eff, idx_in.size, K))
            z_hyp = p_to_z_right_tail(p)
            # simple average of three channels
            z_comb[j] = (z_mwu + z_proj + z_hyp) / np.sqrt(3.0)
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
        i1 = int(np.argmax(z_centered)); r1 = regions[i1]
        rec = { 'leiden': clu, 'label': str(r1), 'z1': float(z_centered[i1]), 'q1': float(q_comb[i1]) }
        for j, r in enumerate(regions):
            rec[f'z_comb__{r}'] = float(z_comb[j]); rec[f'q_comb__{r}'] = float(q_comb[j])
        records.append(rec)
    annot_df = pd.DataFrame.from_records(records).set_index('leiden')
    return annot_df


def run_rapx(adata: ad.AnnData, gold_df: pd.DataFrame, produced_label_col: str = 'subcellular_annotation',
            max_genes_per_region: int = 1200, min_genes_per_region: int = 300, frac_genes: float = 0.25) -> pd.DataFrame:
    """Execute RAP-X annotation.

    Performance knobs:
      max_genes_per_region: hard cap of genes selected per region (default 1200).
      min_genes_per_region: lower bound (default 300).
      frac_genes: fraction of available genes to consider (default 0.25).
    """
    gold_mat0 = _prepare_gold_matrix(gold_df)
    vals = gold_mat0.to_numpy(dtype=np.float32, copy=True)
    row_sum = np.nansum(vals, axis=1, keepdims=True); row_sum[row_sum <= 0] = 1.0
    gm_norm = (vals / row_sum).astype(np.float32)
    genes = gold_mat0.index.to_numpy()
    regions_raw = [c for c in gold_mat0.columns if c in SEVEN_COLS]
    sets = {}
    n_total = len(genes)
    # derive dynamic take size once
    dyn_take = int(max(min_genes_per_region, int(frac_genes * n_total)))
    dyn_take = int(min(max_genes_per_region, dyn_take))
    for i, r in enumerate(regions_raw):
        scores = gm_norm[:, i]
        order = np.argsort(-scores)
        take = dyn_take
        sets[SEVEN_PRETTY.get(r, r)] = set(genes[order[:take]].astype(str))
    regions = list(sets.keys())
    annot_df = _rapx_annotate_core(adata, sets, regions)
    adata.obs[produced_label_col] = adata.obs['leiden'].map(annot_df['label']).astype('category')
    return annot_df


def _plot_rapx_enrichment_and_quality(annot_df: pd.DataFrame, adata2: ad.AnnData, out_dir: Optional[str] = None):
    """Save RAP-X enrichment heatmaps and quality summary panels.
    - Enrichment: heatmaps of z_comb and -log10(q_comb) per cluster vs region
    - Quality: label distribution, q1 distribution, z1 vs -log10(q1) scatter sized by cluster size
    """
    try:
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns  # optional but nicer
            _has_sns = True
        except Exception:
            _has_sns = False

        # Align clusters order to adata2 leiden categories if available
        idx = annot_df.index.astype(str)
        try:
            cats = adata2.obs['leiden'].astype('category').cat.categories.astype(str)
            order_clusters = [c for c in cats if c in set(idx)]
            annot_ord = annot_df.loc[order_clusters]
        except Exception:
            annot_ord = annot_df.sort_index()

        # Determine regions from columns (prefer SEVEN_COLS ordering)
        regions_all = [c for c in SEVEN_COLS if (f'z_comb__{c}' in annot_ord.columns)]
        if not regions_all:
            regions_all = sorted([c.replace('z_comb__', '') for c in annot_ord.columns if c.startswith('z_comb__')])

        # Build matrices
        Z = np.vstack([annot_ord[f'z_comb__{r}'].to_numpy(dtype=float, copy=False) for r in regions_all]).T if regions_all else None
        Q = np.vstack([annot_ord[f'q_comb__{r}'].to_numpy(dtype=float, copy=False) for r in regions_all]).T if regions_all else None

        # 1) Enrichment heatmaps
        if Z is not None and Q is not None and Z.size > 0 and Q.size > 0:
            fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.2*len(regions_all)), max(4, 0.35*len(annot_ord))))
            ax1, ax2 = axes
            if _has_sns:
                sns.heatmap(Z, ax=ax1, cmap='coolwarm', center=0.0, cbar_kws={'label': 'z-score'})
            else:
                im1 = ax1.imshow(Z, aspect='auto', cmap='coolwarm', vmin=np.nanmin(Z), vmax=np.nanmax(Z))
                fig.colorbar(im1, ax=ax1, label='z-score')
            ax1.set_title('RAP-X Enrichment (z)')
            ax1.set_xticks(np.arange(len(regions_all)))
            ax1.set_xticklabels(regions_all, rotation=45, ha='right')
            ax1.set_yticks(np.arange(len(annot_ord)))
            ax1.set_yticklabels(annot_ord.index.astype(str))

            q_neglog = -np.log10(np.clip(Q, 1e-12, 1.0))
            if _has_sns:
                sns.heatmap(q_neglog, ax=ax2, cmap='viridis', cbar_kws={'label': '-log10(q)'})
            else:
                im2 = ax2.imshow(q_neglog, aspect='auto', cmap='viridis', vmin=np.nanmin(q_neglog), vmax=np.nanmax(q_neglog))
                fig.colorbar(im2, ax=ax2, label='-log10(q)')
            ax2.set_title('RAP-X Enrichment (-log10 q)')
            ax2.set_xticks(np.arange(len(regions_all)))
            ax2.set_xticklabels(regions_all, rotation=45, ha='right')
            ax2.set_yticks(np.arange(len(annot_ord)))
            ax2.set_yticklabels(annot_ord.index.astype(str))
            fig.tight_layout()
            _save_fig(fig, out_dir or os.getcwd(), "rapx_enrichment_panels")
            plt.close(fig)

        # 2) Quality summary
        try:
            # Label distribution
            label_counts = annot_df['label'].astype(str).value_counts().sort_values(ascending=False)
        except Exception:
            label_counts = pd.Series(dtype=int)
        q1 = annot_df.get('q1', pd.Series(index=annot_df.index, dtype=float)).astype(float)
        z1 = annot_df.get('z1', pd.Series(index=annot_df.index, dtype=float)).astype(float)
        q1_neglog = -np.log10(np.clip(q1.to_numpy(dtype=float, copy=False), 1e-12, 1.0))
        # Cluster sizes from adata2
        try:
            clu_sizes = adata2.obs['leiden'].value_counts()
            clu_sizes = clu_sizes.reindex(annot_df.index.astype(str)).fillna(0).astype(int)
        except Exception:
            clu_sizes = pd.Series(1, index=annot_df.index)

        fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
        # Left: label distribution
        ax = axes2[0]
        if len(label_counts) > 0:
            ax.bar(label_counts.index.astype(str), label_counts.values, color='#4C72B0')
            ax.set_xticks(np.arange(len(label_counts)))
            ax.set_xticklabels(label_counts.index.astype(str), rotation=45, ha='right')
        ax.set_title('Annotation Label Counts')
        ax.set_ylabel('Clusters')

        # Middle: q1 distribution
        ax = axes2[1]
        ax.hist(q1_neglog, bins=20, color='#55A868', alpha=0.9)
        ax.set_title('-log10(q1) across clusters')
        ax.set_xlabel('-log10(q1)')

        # Right: z1 vs -log10(q1), size by cluster size
        ax = axes2[2]
        sizes = np.clip(clu_sizes.to_numpy(dtype=float, copy=False), 1.0, np.inf)
        s_norm = 20.0 * (sizes / np.nanmax(sizes)) if np.nanmax(sizes) > 0 else 20.0
        ax.scatter(z1.to_numpy(dtype=float, copy=False), q1_neglog, s=s_norm, alpha=0.8, c='#C44E52', edgecolors='none')
        ax.set_xlabel('z1 (top region)')
        ax.set_ylabel('-log10(q1)')
        ax.set_title('Top Enrichment Strength per Cluster')
        fig2.tight_layout()
        _save_fig(fig2, out_dir or os.getcwd(), "rapx_quality_summary")
        plt.close(fig2)
    except Exception:
        # Never break the pipeline for plotting
        pass


def _load_default_gold_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    apex = os.path.join(base, '2019_CELL_APEXSeq.tsv')
    rna = os.path.join(base, 'RNALocateV3.txt')
    if not os.path.exists(apex) or not os.path.exists(rna):
        raise FileNotFoundError(f"Gold files not found under {base}")
    gold_apex = pd.read_table(apex)
    rna_locate = pd.read_table(rna)
    # Massage APEX: keep Common_Gene + *_log2FC and normalize 0-1 by row
    cols = ['Common_Gene'] + [c for c in gold_apex.columns if str(c).endswith('_log2FC')]
    gold_apex = gold_apex[cols].rename(columns={c: c.replace('_log2FC','') for c in cols if c!='Common_Gene'})
    gold_apex = gold_apex.dropna()
    val_cols = [c for c in gold_apex.columns if c != 'Common_Gene']
    X = gold_apex[val_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    X = X.subtract(X.min(axis=1), axis=0)
    span = (X.max(axis=1) - X.min(axis=1)).replace(0, 1.0)
    X = X.div(span, axis=0)
    gold_apex = pd.concat([gold_apex[['Common_Gene']], X], axis=1)
    return gold_apex, rna_locate


def _maybe_cluster_and_annotate(adata1: ad.AnnData, adata2: ad.AnnData, final_df: pd.DataFrame, show_internal_progress: bool = False, out_dir: Optional[str] = None,
                                rapx_two_stage: bool = False, rapx_spatial_channel: bool = False,
                                gold_data: Optional[Union[str, pd.DataFrame]] = None):
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
        ann_fast = bool(ann.get('fast', True))
        ann_save_plots = bool(ann.get('save_plots', False))
    except Exception:
        ann_fast = True
        ann_save_plots = False
    # Fast path: if already clustered (has 'leiden') and fast mode enabled, skip scanpy normalization/clustering
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
    # Step 2: load gold tables (or use user-supplied `gold_data`).
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
            # If loading user gold fails, fall back to default tables (but warn)
            try:
                import logging as _logging
                _logging.getLogger('cellscope').warning('Failed to load provided gold_data (%s); falling back to defaults', e)
            except Exception:
                pass
            gold_apex, rna_loc = _load_default_gold_tables()
            l1 = build_gold_from_rnalocate_l1(rna_loc, species=('Homo sapiens','Mus musculus'), rna_types=('mRNA',), agg_mode='prob_or', score_floor=0.6, use_pubmed_weight=True)
            gold_from_rna = project_l1_to_seven(l1, mode='strict', mito_non_omm=None)
            gold_aug = merge_gold(gold_apex, gold_from_rna, fuse='max')
    else:
        gold_apex, rna_loc = _load_default_gold_tables()
        l1 = build_gold_from_rnalocate_l1(rna_loc, species=('Homo sapiens','Mus musculus'), rna_types=('mRNA',), agg_mode='prob_or', score_floor=0.6, use_pubmed_weight=True)
        gold_from_rna = project_l1_to_seven(l1, mode='strict', mito_non_omm=None)
        gold_aug = merge_gold(gold_apex, gold_from_rna, fuse='max')
    # Step 3: RAP-X annotation; write to adata2.obs
    # Read RAP-X tuning knobs from config (optional)
    rapx_max_genes = 1200
    rapx_min_genes = 300
    rapx_frac_genes = 0.25
    try:
        _cfg2 = load_params_yaml()
        ann_cfg = _cfg2.get('annotation', {}) if _cfg2 else {}
        rapx_max_genes = int(ann_cfg.get('rapx_set_max_genes', rapx_max_genes))
        rapx_min_genes = int(ann_cfg.get('rapx_set_min_genes', rapx_min_genes))
        rapx_frac_genes = float(ann_cfg.get('rapx_set_frac_genes', rapx_frac_genes))
    except Exception:
        pass
    t_rapx0 = __import__('time').perf_counter()
    annot_df = run_rapx(
        adata2,
        gold_df=gold_aug,
        produced_label_col='subcellular_annotation',
        max_genes_per_region=rapx_max_genes,
        min_genes_per_region=rapx_min_genes,
        frac_genes=rapx_frac_genes,
    )
    t_rapx1 = __import__('time').perf_counter()
    print(f"[RAP-X timing] annotate_core took {t_rapx1 - t_rapx0:0.2f}s (regions={len(annot_df.columns)})")
    # Save RAP-X diagnostics plots and annotation scores
    try:
        # _plot_rapx_enrichment_and_quality(annot_df, adata2, out_dir=out_dir)
        _save_dataframe(annot_df.reset_index(), out_dir or os.getcwd(), "rapx_annotation_scores")
        # Derive per-cluster dominant subcellular label for UMAP overlay
        try:
            if 'subcellular_annotation' in adata2.obs and 'leiden' in adata2.obs:
                # Explicitly pass observed=True to groupby to match future pandas default
                grp = (
                    adata2.obs
                    .groupby('leiden', observed=True)['subcellular_annotation']
                    .agg(lambda s: (s.value_counts().idxmax() if len(s) else 'Unassigned'))
                )
                adata2.obs['cluster-subcellular'] = adata2.obs['leiden'].map(grp).astype(str)
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
                sc.pl.umap(adata2, color=['subcellular_annotation','leiden'], legend_fontsize='small', wspace=0.2, show=False)
                _save_fig(plt.gcf(), out_dir or os.getcwd(), "rapx_umap_cluster_annotation")
                plt.close()
            except Exception:
                pass
    except Exception:
        pass
    # Step 4: map annotation back to final point table via meta-domain
    if 'meta-domain' in final_df.columns:
        # Faster mapping vs merge for large final_df
        m = adata2.obs['subcellular_annotation'].astype(str)
        # Ensure key types align
        final_df = final_df.copy()
        final_df['meta-domain'] = final_df['meta-domain'].astype(str)
        # Vectorized map
        final_df['subcellular_annotation'] = final_df['meta-domain'].map(m)
    return adata1, adata2, final_df
