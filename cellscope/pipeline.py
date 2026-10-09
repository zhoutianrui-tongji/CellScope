from typing import Optional, Literal, Tuple, List, Callable, cast, Dict, Any, Union, Sequence, Mapping
from contextlib import contextmanager
import functools
import os
import re
import math
import hashlib
import logging as _logging
import tempfile
import time as _time
import warnings as _warnings
from collections import defaultdict, Counter

# Numba and Matplotlib otherwise try to cache beside read-only site-packages or
# under a non-writable home in containers/HPC jobs. Keep caches process-local
# and writable without changing numerical behavior.
_runtime_cache_root = os.path.join(tempfile.gettempdir(), "cellscope-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_runtime_cache_root, "numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_runtime_cache_root, "matplotlib"))

# Per-cell spatial graphs may contain several legitimate islands.  sklearn
# connects those islands internally before constrained Ward clustering and
# otherwise emits this warning once per affected cell.  The connectivity
# contract is enforced again below, so keep only this exact repeated notice
# out of user-facing progress output.
def _quiet_expected_disconnected_connectivity_warning() -> None:
    _warnings.filterwarnings(
        "ignore",
        message=(
            r"the number of connected components of the connectivity matrix is "
            r"\d+ > 1\. Completing it to avoid stopping the tree early\."
        ),
        category=UserWarning,
        module=r"sklearn\.cluster\._agglomerative",
    )


_quiet_expected_disconnected_connectivity_warning()

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

# Configure BLAS/OMP threads before importing numpy/scipy.
# Priority: params.yaml runtime.blas_threads / runtime.n_jobs > fallback heuristic;
# inherited BLAS environment values are deliberately overridden for reproducibility.
_runtime_cfg_threads = {}
try:
    if _load_params_yaml_threads:
        _runtime_cfg_threads = (_load_params_yaml_threads() or {}).get("runtime", {}) or {}
except Exception:
    _runtime_cfg_threads = {}

_cpu = os.cpu_count() or 8
_fallback_threads = max(1, min(32, _cpu // 2 or 1))
_yaml_threads = _coerce_pos_int(_runtime_cfg_threads.get("blas_threads")) or _coerce_pos_int(_runtime_cfg_threads.get("n_jobs"))
_outer_jobs = (
    _coerce_pos_int(os.environ.get("CELLSCOPE_N_JOBS"))
    or _coerce_pos_int(_runtime_cfg_threads.get("n_jobs"))
    or 1
)
_threaded_outer_parallelism = (
    str(
        os.environ.get("CELLSCOPE_JOBLIB_BACKEND")
        or _runtime_cfg_threads.get("joblib_backend", "")
    ).strip().lower()
    == "threading"
    and _outer_jobs > 1
)
# The bundled OpenBLAS was built with a maximum thread table of 64.  Above
# that value it may warn at import time and subsequently segfault under BLAS
# load, so keep both the initial import environment and runtime limits within
# the compiled capability.
_MAX_SAFE_BLAS_THREADS = 64
_threads_default = str(
    1
    if _threaded_outer_parallelism
    else min(_yaml_threads or _fallback_threads, _MAX_SAFE_BLAS_THREADS)
)
_numba_safe_threads = (
    max(1, _MAX_SAFE_BLAS_THREADS // _outer_jobs)
    if _threaded_outer_parallelism
    else _MAX_SAFE_BLAS_THREADS
)
_numba_threads_default = str(
    min(
        _coerce_pos_int(_runtime_cfg_threads.get("numba_threads"))
        or (1 if _threaded_outer_parallelism else (_yaml_threads or _fallback_threads)),
        _numba_safe_threads,
    )
)
_scanpy_threads_default = str(
    min(
        _coerce_pos_int(os.environ.get("SCANPY_N_JOBS"))
        or _coerce_pos_int(_runtime_cfg_threads.get("scanpy_n_jobs"))
        or 4,
        int(_numba_threads_default),
    )
)
_THREAD_ENV_DEFAULTS = {
    # These must be assigned before importing numpy/scipy.  ``setdefault``
    # previously preserved a user's inherited OPENBLAS_NUM_THREADS value even
    # when runtime.blas_threads requested a smaller safe limit; OpenBLAS then
    # initialized with an oversized thread table and could segfault later.
    "OPENBLAS_NUM_THREADS": _threads_default,
    "OMP_NUM_THREADS": _threads_default,
    "MKL_NUM_THREADS": _threads_default,
    "NUMEXPR_MAX_THREADS": _threads_default,
    "NUMEXPR_NUM_THREADS": _threads_default,
    "BLIS_NUM_THREADS": _threads_default,
    "VECLIB_MAXIMUM_THREADS": _threads_default,
    "ACCELERATE_NUM_THREADS": _threads_default,
    # Numba-backed UMAP kernels can call BLAS from their worker threads.  On
    # hosts with more cores than OpenBLAS was compiled for, leaving Numba at
    # os.cpu_count() can exhaust OpenBLAS' thread-metadata table even though
    # OPENBLAS_NUM_THREADS itself is 1.
    "NUMBA_NUM_THREADS": _numba_threads_default,
    "SCANPY_N_JOBS": _scanpy_threads_default,
}
for _k, _v in _THREAD_ENV_DEFAULTS.items():
    os.environ[_k] = str(_v)
del _THREAD_ENV_DEFAULTS, _k, _v, _cpu, _threads_default, _fallback_threads, _yaml_threads, _runtime_cfg_threads, _outer_jobs, _threaded_outer_parallelism, _numba_safe_threads, _numba_threads_default, _scanpy_threads_default

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import anndata as ad
from scipy.stats import binom, hypergeom, norm, rankdata, fisher_exact
from scipy.special import erfc
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
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
from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin
import scipy.sparse as sp
import gc

from .plotting_style import (
    PUBLICATION_PLOT_STYLE as _PUBLICATION_PLOT_STYLE,
    SUBCELLULAR_LABEL_GROUPS as _SUBCELLULAR_LABEL_GROUPS,
    SUBCELLULAR_LABEL_ORDER as _SUBCELLULAR_LABEL_ORDER,
    SUBCELLULAR_LABEL_PALETTE as _SUBCELLULAR_LABEL_PALETTE,
)


METASPOT_COL = "metaspot"
METASPOT_LOCAL_COL = "metaspot_in_cell"
_LEGACY_METASPOT_COL = "cell_sub"
_LEGACY_METASPOT_LOCAL_COL = "cluster_in_cell"

METASPOT_GRAPH_SCHEMA_VERSION = 4
METASPOT_DGI_SCHEMA_VERSION = 4
METASPOT_DGI_KEY = "X_dgi_gcn_metaspot"
METASPOT_DGI_SAGE_KEY = "X_dgi_sage_metaspot"
METADOMAIN_DGI_KEY = "X_dgi_gcn_metadomain"
METASPOT_GRAPH_KEY = "A_metaspot_norm"
METASPOT_DMGI_EXPRESSION_KEY = "X_dmgi_expression_metaspot"
METASPOT_DMGI_SPATIAL_KEY = "X_dmgi_spatial_metaspot"


def _replace_frame_columns(
    frame: pd.DataFrame,
    columns: Mapping[str, Any],
) -> pd.DataFrame:
    """Return ``frame`` with many columns replaced in one column concat.

    Repeated ``frame[name] = values`` calls create one pandas block at a time.
    Annotation writes more than one hundred audit columns, so that pattern
    fragments both ``AnnData.obs`` and transcript tables and makes every later
    insertion emit ``PerformanceWarning``. Building the update frame first
    keeps dtype inference (including categoricals and nullable booleans) while
    avoiding the repeated insertion path.
    """
    if not columns:
        return frame

    updates: Dict[str, Any] = {}
    for name, values in columns.items():
        if isinstance(values, pd.Series):
            updates[str(name)] = values.reindex(frame.index)
        else:
            updates[str(name)] = values
    update_frame = pd.DataFrame(updates, index=frame.index)
    base = frame.drop(columns=list(update_frame.columns), errors="ignore")
    return pd.concat([base, update_frame], axis=1, copy=False)


def _replace_obs_columns(
    adata: ad.AnnData,
    columns: Mapping[str, Any],
) -> None:
    """Bulk-replace AnnData observation columns without fragmenting obs."""
    if columns:
        adata.obs = _replace_frame_columns(adata.obs, columns)


def _obs_names_checksum(obs_names: Sequence[object]) -> str:
    """Return an order-sensitive digest for an AnnData observation index."""
    digest = hashlib.sha256()
    for value in obs_names:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _membership_checksum(
    obs_names: Sequence[object],
    memberships: Sequence[object],
) -> str:
    """Return an order-sensitive digest of observation-to-group membership."""
    digest = hashlib.sha256()
    for obs_name, membership in zip(obs_names, memberships):
        digest.update(str(obs_name).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(membership).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


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


# Keep Scanpy/PyNNDescent within Numba's import-time thread ceiling.
try:
    import scanpy as sc
    sc.settings.n_jobs = max(
        1,
        min(
            int(os.environ.get("SCANPY_N_JOBS", "4")),
            int(os.environ["NUMBA_NUM_THREADS"]),
        ),
    )
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
from .config import (
    dgi_unit_from_config,
    fingerprint_modules_set,
    fingerprint_stage,
    load_params_yaml,
    resolve_annotation_configuration,
)
from .provenance import (
    atomic_write_json,
    build_input_provenance,
    bundled_reference_provenance,
    sha256_mapping,
    source_tree_provenance,
)
from .resource_paths import bundled_resource_path


def load_inputs(
    spatial_path: str,
    cell_boundaries_path: str,
    nucleus_boundaries_path: str,
    out_dir: str,
    resume: bool,
    save_intermediate: bool,
    *,
    reuse_cached_inputs: Optional[bool] = None,
):
    """Load raw inputs and optionally maintain raw-input checkpoints.

    ``resume`` controls whether raw checkpoints are persisted. Reuse is a
    separate decision: when provenance selects ``load_inputs`` as the first
    stage to rerun, existing checkpoints are stale and must be overwritten
    from the source paths.
    """
    cp_spatial = _checkpoint_path(out_dir, "spatial")
    cp_cell = _checkpoint_path(out_dir, "cell_boundaries")
    cp_nuc = _checkpoint_path(out_dir, "nucleus_boundaries")
    cache_inputs = bool(resume or save_intermediate)
    reuse_cached_inputs = (
        bool(resume)
        if reuse_cached_inputs is None
        else bool(reuse_cached_inputs)
    )

    if reuse_cached_inputs and os.path.exists(cp_spatial):
        spatial_df = load_checkpoint(cp_spatial)
    else:
        spatial_df = read_spatial_transcripts(spatial_path)
        if cache_inputs:
            save_checkpoint(spatial_df, cp_spatial)

    if reuse_cached_inputs and os.path.exists(cp_cell):
        cell_df = load_checkpoint(cp_cell)
    else:
        cell_df = read_boundaries(cell_boundaries_path)
        if cache_inputs:
            save_checkpoint(cell_df, cp_cell)

    if reuse_cached_inputs and os.path.exists(cp_nuc):
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


def _module3_membership_manifest_path(out_dir: str) -> str:
    return os.path.join(
        _intermediate_dir(out_dir),
        "module3_metaspot_membership.manifest.json",
    )


def _save_module3_membership_checkpoint(
    point_df: pd.DataFrame,
    out_dir: str,
    *,
    module_fingerprints: Mapping[str, Any],
) -> None:
    """Persist compact M3 membership plus the upstream fingerprint chain."""
    membership_columns = [
        column
        for column in (
            "transcript_id",
            "cell_id",
            METASPOT_LOCAL_COL,
            METASPOT_COL,
        )
        if column in point_df.columns
    ]
    required = {"cell_id", METASPOT_LOCAL_COL, METASPOT_COL}
    if not required.issubset(membership_columns):
        raise KeyError(
            "Module 3 membership checkpoint requires cell_id, "
            f"{METASPOT_LOCAL_COL}, and {METASPOT_COL}"
        )
    path = os.path.join(
        _intermediate_dir(out_dir),
        "module3_metaspot_membership.parquet",
    )
    _atomic_dataframe_write(
        point_df.loc[:, membership_columns],
        path,
        file_format="parquet",
    )
    atomic_write_json(
        _module3_membership_manifest_path(out_dir),
        {
            "schema_version": 1,
            "rows": int(len(point_df)),
            "n_cells": int(point_df["cell_id"].nunique(dropna=False)),
            "n_metaspots": int(point_df[METASPOT_COL].nunique(dropna=False)),
            "columns": membership_columns,
            "module_fingerprints": {
                key: str(module_fingerprints.get(key, ""))
                for key in ("m1", "m2", "m3")
            },
        },
    )


def _load_module3_membership_checkpoint(
    point_df: pd.DataFrame,
    out_dir: str,
    *,
    module_fingerprints: Mapping[str, Any],
) -> Optional[pd.DataFrame]:
    """Restore M3 labels after a row-group-wise owner-cell alignment audit."""
    import json

    parquet_path = os.path.join(
        _intermediate_dir(out_dir),
        "module3_metaspot_membership.parquet",
    )
    manifest_path = _module3_membership_manifest_path(out_dir)
    if not os.path.isfile(parquet_path) or not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected_fingerprints = {
            key: str(module_fingerprints.get(key, ""))
            for key in ("m1", "m2", "m3")
        }
        if manifest.get("module_fingerprints") != expected_fingerprints:
            _logging.getLogger("cellscope").warning(
                "Rejecting compact Module 3 membership: upstream module "
                "fingerprints differ"
            )
            return None
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(parquet_path)
        row_count = int(parquet.metadata.num_rows)
        if row_count != len(point_df) or int(manifest.get("rows", -1)) != row_count:
            _logging.getLogger("cellscope").warning(
                "Rejecting compact Module 3 membership: rows=%d input_rows=%d",
                row_count,
                len(point_df),
            )
            return None
        available = set(parquet.schema.names)
        required = {"cell_id", METASPOT_LOCAL_COL, METASPOT_COL}
        if not required.issubset(available):
            return None

        local_labels = np.empty(row_count, dtype=np.int32)
        metaspot_chunks = []
        offset = 0
        owner_cells = point_df["cell_id"]
        logger = _logging.getLogger("cellscope")
        logger.info(
            "Validating compact Module 3 membership | rows=%d | row_groups=%d",
            row_count,
            int(parquet.metadata.num_row_groups),
        )
        for row_group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                row_group,
                columns=["cell_id", METASPOT_LOCAL_COL, METASPOT_COL],
            )
            chunk_rows = int(table.num_rows)
            stop = offset + chunk_rows
            cached_cells = table["cell_id"].combine_chunks().to_numpy(
                zero_copy_only=False
            )
            current_cells = owner_cells.iloc[offset:stop].astype(str).to_numpy()
            if not np.array_equal(
                np.asarray(cached_cells, dtype=str),
                np.asarray(current_cells, dtype=str),
            ):
                logger.warning(
                    "Rejecting compact Module 3 membership: owner-cell order "
                    "differs in row group %d",
                    row_group,
                )
                return None
            local_labels[offset:stop] = (
                table[METASPOT_LOCAL_COL]
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
                .astype(np.int32, copy=False)
            )
            metaspot_chunks.append(table[METASPOT_COL].combine_chunks())
            offset = stop
        if offset != row_count:
            return None

        restored = point_df.copy(deep=False)
        restored[METASPOT_LOCAL_COL] = local_labels
        chunked_metaspots = pa.chunked_array(metaspot_chunks)
        try:
            restored[METASPOT_COL] = pd.Series(
                pd.arrays.ArrowStringArray(chunked_metaspots),
                index=restored.index,
            )
        except (AttributeError, TypeError):
            restored[METASPOT_COL] = chunked_metaspots.to_pandas()
        if int(restored[METASPOT_COL].nunique(dropna=False)) != int(
            manifest.get("n_metaspots", -1)
        ):
            _logging.getLogger("cellscope").warning(
                "Rejecting compact Module 3 membership: MetaSpot count differs"
            )
            return None
        logger.info(
            "Compact Module 3 membership validated and restored | "
            "MetaSpots=%d",
            int(manifest["n_metaspots"]),
        )
        return restored
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Could not load compact Module 3 membership: %s",
            exc,
        )
        return None


def _load_late_point_resume_checkpoint(
    out_dir: str,
    spatial_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Recover the aligned transcript-to-MetaSpot table in lite I/O mode.

    ``module5_final_data`` is deliberately persisted for resume even when the
    much larger Module 1--3 tables are omitted. It still contains every
    upstream geometry and MetaSpot membership column. Reusing it is safe only
    when its row count matches the current preprocessed input and its MetaSpot
    population is strictly aligned with the Module 4 AnnData checkpoint.
    """
    base = _intermediate_dir(out_dir)
    point_df = _load_dataframe_checkpoint(base, "module5_final_data")
    if point_df is None:
        return None
    required = {
        "cell_id",
        "feature_name",
        "x_location",
        "y_location",
        METASPOT_LOCAL_COL,
        METASPOT_COL,
    }
    missing = sorted(required.difference(point_df.columns))
    if missing or len(point_df) != len(spatial_df):
        _logging.getLogger("cellscope").warning(
            "Rejecting late point-table resume checkpoint: missing=%s | "
            "cached_rows=%d | input_rows=%d",
            missing,
            len(point_df),
            len(spatial_df),
        )
        return None
    module4 = _load_adata_checkpoint(
        os.path.join(base, "module4_adata1.h5ad")
    )
    if module4 is None:
        return None
    aligned, reason = _validate_adata1_point_df_alignment(module4, point_df)
    if not aligned:
        _logging.getLogger("cellscope").warning(
            "Rejecting late point-table resume checkpoint: %s",
            reason,
        )
        return None
    return point_df


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
    try:
        resume_cfg = load_params_yaml() or {}
        resume_graph_unit = _graph_unit_from_config(resume_cfg)
    except Exception:
        resume_graph_unit = "metadomain"
    if resume_graph_unit == "metaspot":
        if adata2 is None:
            return None
        dgi_meta = (
            adata1.uns.get("dgi_embeddings", {}).get(METASPOT_DGI_KEY, {})
        )
        if (
            METASPOT_DGI_KEY not in adata1.obsm
            or adata1.obsm[METASPOT_DGI_KEY].shape[0] != adata1.n_obs
            or dgi_meta.get("node_unit") != "metaspot"
            or int(dgi_meta.get("schema_version", -1))
            != METASPOT_DGI_SCHEMA_VERSION
            or dgi_meta.get("node_id_checksum")
            != _obs_names_checksum(adata1.obs_names)
            or METADOMAIN_DGI_KEY not in adata2.obsm
        ):
            return None
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

    if (
        {METASPOT_COL, "meta-domain"}.issubset(final_df.columns)
        and "meta-domain" in adata1.obs.columns
    ):
        mapping = (
            final_df[[METASPOT_COL, "meta-domain"]]
            .assign(
                **{
                    METASPOT_COL: lambda frame: frame[METASPOT_COL].astype(str),
                    "meta-domain": lambda frame: frame["meta-domain"].astype(str),
                }
            )
            .drop_duplicates()
        )
        if mapping[METASPOT_COL].duplicated().any():
            return None
        observed_mapping = mapping.set_index(METASPOT_COL)["meta-domain"]
        expected_index = pd.Index(adata1.obs_names.astype(str))
        expected_mapping = pd.Series(
            adata1.obs["meta-domain"].astype(str).to_numpy(),
            index=expected_index,
        )
        if not observed_mapping.reindex(expected_index).equals(expected_mapping):
            return None

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


def _resume_annotation_outputs(
    out_dir: str,
) -> Optional[
    Tuple[
        Optional[ad.AnnData],
        Optional[ad.AnnData],
        Optional[pd.DataFrame],
    ]
]:
    base = _intermediate_dir(out_dir)
    final_df = _load_dataframe_checkpoint(base, "annotated_final_csv")
    if final_df is None:
        return None
    ann_adata1 = _load_adata_checkpoint(
        os.path.join(base, "annotated_adata1.h5ad")
    )
    ann_adata2 = _load_adata_checkpoint(os.path.join(base, "annotated_adata2.h5ad"))
    return ann_adata1, ann_adata2, final_df


def preprocess(
    spatial_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    nucleus_df: pd.DataFrame,
    show_internal_progress: bool = False,
    use_input_nucleus_flags: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align input cells without leaking unused nuclear truth into selection.

    When input nuclear flags are part of Module 1, retain the historical
    requirement that a cell contain both nuclear and non-nuclear transcripts.
    When geometry is recomputed from polygons, select only on membership in the
    three input tables; ``overlaps_nucleus`` then has no influence on the sample.
    """
    del show_internal_progress  # reserved for future streaming validation
    common_cells = (
        set(spatial_df["cell_id"].dropna())
        & set(cell_df["cell_id"].dropna())
        & set(nucleus_df["cell_id"].dropna())
    )

    if use_input_nucleus_flags:
        if "overlaps_nucleus" not in spatial_df.columns:
            raise ValueError(
                "spatial input is missing overlaps_nucleus while "
                "module1.use_input_nucleus_flags=true"
            )
        unique_counts = spatial_df.groupby("cell_id", observed=True)[
            "overlaps_nucleus"
        ].nunique(dropna=True)
        common_cells &= set(unique_counts[unique_counts == 2].index)

    spatial_df = spatial_df[spatial_df["cell_id"].isin(common_cells)].copy()
    cell_df = cell_df[cell_df["cell_id"].isin(common_cells)].copy()
    nucleus_df = nucleus_df[nucleus_df["cell_id"].isin(common_cells)].copy()

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
    use_input_nucleus_flags: bool = True,
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
        """Build valid nucleus polygons from boundary-point tables.

        Some spatial formats store boundary pixels/vertices without contour
        ordering. ``matplotlib.path.Path`` assumes ordered polygon vertices;
        feeding an unordered point cloud can collapse the inferred nuclear
        area and severely reduce nuclear recall. We preserve apparently
        ordered contours and reconstruct clearly unordered contours with a
        convex hull. Nuclei are typically close to convex, making this a
        conservative and stable fallback for cross-platform input.
        """
        from scipy.spatial import ConvexHull
        from scipy.spatial import QhullError

        paths = {}
        sub = nucleus_df[nucleus_df['cell_id'].isin(cells)]
        it = sub.groupby('cell_id', sort=False)
        iterator = _progress_iter(
            it,
            desc='Build nucleus Path',
            total=it.ngroups,
            show=show_internal_progress,
        )

        for cid, g in iterator:
            pts = (
                g[['vertex_x', 'vertex_y']]
                .apply(pd.to_numeric, errors='coerce')
                .dropna()
                .to_numpy(dtype=np.float64)
            )
            # ``np.unique(..., axis=0)`` sorts rows, which destroys contour
            # order. Keep the first occurrence of each vertex instead.
            if len(pts):
                _, first_idx = np.unique(pts, axis=0, return_index=True)
                pts = pts[np.sort(first_idx)]

            if len(pts) < 3:
                paths[cid] = None
                continue

            try:
                tree = KDTree(pts)
                nn_dist, _ = tree.query(pts, k=2, return_distance=True)
                local_scale = float(np.nanmedian(nn_dist[:, 1]))
            except Exception:
                local_scale = np.nan

            closed_pts = np.vstack([pts, pts[:1]])
            path_steps = np.sqrt(
                np.sum(np.diff(closed_pts, axis=0) ** 2, axis=1)
            )
            typical_step = float(np.nanmedian(path_steps))
            upper_step = float(np.nanpercentile(path_steps, 95))

            ordered = (
                np.isfinite(local_scale)
                and local_scale > 0
                and typical_step <= 3.0 * local_scale
                and upper_step <= 10.0 * local_scale
            )

            polygon_pts = pts

            # A short vertex list can pass the step-size heuristic while still
            # being self-intersecting (for example, bow-tie ordering). Treat
            # only a valid, non-zero-area Shapely polygon as an ordered contour.
            if ordered:
                try:
                    from shapely.geometry import Polygon as _Polygon

                    candidate = _Polygon(pts)
                    ordered = bool(
                        candidate.is_valid
                        and not candidate.is_empty
                        and float(candidate.area) > 0.0
                    )
                except Exception:
                    ordered = False

            if not ordered:
                try:
                    hull = ConvexHull(pts)
                    polygon_pts = pts[hull.vertices]
                except QhullError:
                    center = np.nanmedian(pts, axis=0)
                    angle = np.arctan2(
                        pts[:, 1] - center[1],
                        pts[:, 0] - center[0],
                    )
                    polygon_pts = pts[np.argsort(angle)]

            paths[cid] = (
                # With ``closed=True`` Matplotlib treats the last vertex as a
                # CLOSEPOLY placeholder. Repeat the first point explicitly so
                # the final real vertex is not discarded.
                Path(np.vstack([polygon_pts, polygon_pts[:1]]), closed=True)
                if len(polygon_pts) >= 3
                else None
            )

        return paths

    # Avoid copying the full transcript table. We keep the original rows and
    # build geometry columns in chunks so Module 1 can run without doubling RAM.
    df = transcripts_df[transcripts_df['cell_id'].isin(sampled_cells)]
    cell_kd = _build_kdtrees(cell_boundaries_df, sampled_cells, desc="Cell KDTree")
    nuc_kd = _build_kdtrees(nucleus_boundaries_df, sampled_cells, desc="Nucleus KDTree")
    use_input_nucleus_flags = bool(
        use_input_nucleus_flags
        and "overlaps_nucleus" in df.columns
    )
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


def _target_metaspot_counts_from_site_counts(
    site_counts: np.ndarray,
    target_sites_per_metaspot: int,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
) -> np.ndarray:
    """Vectorize the configured MetaSpot budget over per-cell site counts."""
    n_sites = np.asarray(site_counts, dtype=np.int64)
    if np.any(n_sites < 0):
        raise ValueError("site_counts must be non-negative")

    target_sites = max(1, int(target_sites_per_metaspot))
    target_count = (n_sites + target_sites - 1) // target_sites
    lower_bound = max(1, int(min_metaspots_per_cell))
    target_count = np.maximum(lower_bound, target_count)
    if isinstance(max_metaspots_per_cell, (int, float)):
        upper_bound = max(lower_bound, int(max_metaspots_per_cell))
        target_count = np.minimum(target_count, upper_bound)
    target_count = np.minimum(target_count, n_sites)
    return np.maximum(1, target_count).astype(np.int64, copy=False)


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
    n_sites = int(np.unique(finite_xy, axis=0).shape[0])
    return int(
        _target_metaspot_counts_from_site_counts(
            np.asarray([n_sites], dtype=np.int64),
            target_sites_per_metaspot=target_sites_per_metaspot,
            min_metaspots_per_cell=min_metaspots_per_cell,
            max_metaspots_per_cell=max_metaspots_per_cell,
        )[0]
    )


def _allocate_partition_target_counts(
    partition_coordinates: Sequence[np.ndarray],
    target_sites_per_metaspot: int,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
) -> List[int]:
    """Allocate one cell-level MetaSpot budget across spatial compartments.

    Every non-empty compartment receives at least one MetaSpot.  The remaining
    budget is assigned in proportion to unique XY sites, with deterministic
    largest-remainder tie breaking.  This preserves the configured per-cell
    cap while preventing any MetaSpot from crossing a compartment boundary.
    """
    partitions = [
        np.asarray(coordinates, dtype=np.float32)
        for coordinates in partition_coordinates
        if len(coordinates)
    ]
    if not partitions:
        return []

    site_counts = np.asarray(
        [
            np.unique(
                coordinates[np.isfinite(coordinates).all(axis=1)],
                axis=0,
            ).shape[0]
            for coordinates in partitions
        ],
        dtype=np.int64,
    )
    site_counts = np.maximum(site_counts, 1)
    combined = np.vstack(partitions)
    total_target = _target_metaspot_count(
        combined,
        target_sites_per_metaspot=target_sites_per_metaspot,
        min_metaspots_per_cell=max(
            int(min_metaspots_per_cell),
            len(partitions),
        ),
        max_metaspots_per_cell=max_metaspots_per_cell,
    )
    total_target = max(len(partitions), int(total_target))
    total_target = min(int(site_counts.sum()), total_target)

    raw = total_target * site_counts.astype(np.float64) / site_counts.sum()
    allocation = np.floor(raw).astype(np.int64)
    allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, site_counts)

    while int(allocation.sum()) < total_target:
        candidates = np.flatnonzero(allocation < site_counts)
        if candidates.size == 0:
            break
        residual = raw[candidates] - allocation[candidates]
        chosen = int(candidates[np.argmax(residual)])
        allocation[chosen] += 1

    while int(allocation.sum()) > total_target:
        candidates = np.flatnonzero(allocation > 1)
        if candidates.size == 0:
            break
        residual = raw[candidates] - allocation[candidates]
        chosen = int(candidates[np.argmin(residual)])
        allocation[chosen] -= 1

    return allocation.astype(int).tolist()


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

    # Canonical spatial Ward segmentation starts from one constant label. Skip
    # thousands of tiny majority-vote reductions in that common path.
    point_label_values = np.unique(point_labels)
    if point_label_values.size == 1:
        site_labels = np.full(
            n_sites,
            int(point_label_values[0]),
            dtype=np.int32,
        )
    else:
        # Pick a deterministic majority label at coordinates containing
        # multiple transcripts.
        site_labels = np.empty(n_sites, dtype=np.int32)
        order = np.argsort(inverse, kind="stable")
        sorted_sites = inverse[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_sites)) + 1]
        stops = np.r_[starts[1:], len(order)]
        for start, stop in zip(starts, stops):
            site = int(sorted_sites[start])
            values, counts = np.unique(
                point_labels[order[start:stop]],
                return_counts=True,
            )
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

    edge_rows = np.repeat(
        np.arange(n_sites, dtype=np.int64),
        neighbor_idx.shape[1],
    )
    edge_cols = np.asarray(neighbor_idx, dtype=np.int64).ravel()
    edge_distances = np.asarray(neighbor_dist, dtype=np.float64).ravel()
    valid_edges = (
        (edge_rows != edge_cols)
        & np.isfinite(edge_distances)
        & (edge_distances > 0.0)
    )
    edge_rows = edge_rows[valid_edges]
    edge_cols = edge_cols[valid_edges]
    edge_distances = edge_distances[valid_edges]
    edge_low = np.minimum(edge_rows, edge_cols)
    edge_high = np.maximum(edge_rows, edge_cols)
    edge_codes = edge_low * np.int64(n_sites) + edge_high
    knn_codes = np.unique(edge_codes)
    local_codes = np.unique(edge_codes[edge_distances <= radius])

    # Feature-aware Ward clustering needs one connected neighborhood graph to
    # preserve its requested cluster count. Direct label splitting keeps the
    # stricter native-spacing radius so genuinely separated islands stay split.
    selected_codes = knn_codes if features is not None else local_codes
    if selected_codes.size == 0:
        return point_labels.copy()

    edge_a = (selected_codes // np.int64(n_sites)).astype(
        np.int32,
        copy=False,
    )
    edge_b = (selected_codes % np.int64(n_sites)).astype(
        np.int32,
        copy=False,
    )

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
            rows = np.concatenate((edge_a, edge_b))
            cols = np.concatenate((edge_b, edge_a))
            connectivity = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(n_sites, n_sites),
            )
            try:
                # Some runners prepend their own warning filters after this
                # module is imported, so reassert the precise rule at the call
                # site without wrapping or serializing parallel clustering.
                _quiet_expected_disconnected_connectivity_warning()
                site_labels = AgglomerativeClustering(
                    n_clusters=requested_clusters,
                    linkage="ward",
                    connectivity=connectivity,
                    compute_full_tree=False,
                ).fit_predict(combined_features).astype(np.int32)
                used_constrained_clustering = True
            except Exception as exc:
                _logging.getLogger("cellscope").warning(
                    "Spatially constrained clustering failed; using label "
                    "smoothing fallback: %s",
                    exc,
                )

    if not used_constrained_clustering:
        adjacency = [[] for _ in range(n_sites)]
        for a, b in zip(edge_a, edge_b):
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
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
    for a, b in zip(edge_a, edge_b):
        if site_labels[a] == site_labels[b]:
            union(int(a), int(b))

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
        for a, b in zip(edge_a, edge_b):
            root_a, root_b = find(int(a)), find(int(b))
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
    partition_by_nucleus: bool = False,
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
    parallel_chunk_cells: Optional[int] = None,
    parallel_n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:
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
        target_count_override: Optional[int] = None,
    ) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int32)
        if not bool(enforce_spatial_connectivity) or len(labels) <= 1:
            return labels
        coordinates = g[["x_location", "y_location"]].to_numpy(
            np.float32,
            copy=False,
        )
        target_count = int(np.unique(labels).size)
        if isinstance(target_count_override, (int, float)):
            target_count = max(1, int(target_count_override))
        elif TARGET_SITES_PER_METASPOT is not None:
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
    # Keep process workers from serializing the full transcript table.
    cell_comp_emb = getattr(point_df, '_cell_comp_emb', None)

    def _fit_one(
        args,
        allow_nucleus_partition: bool = True,
        target_count_override: Optional[int] = None,
    ):
        cid, g = args
        # Ensure minimal geometry present; composition may be provided per-cell
        g = g.dropna(subset=['x_location', 'y_location'])
        if len(g) == 0:
            return cid, pd.Series([], dtype=np.int32, index=g.index)

        if bool(partition_by_nucleus) and allow_nucleus_partition:
            if "in_nucleus" in g.columns:
                nucleus_flags = _coerce_bool_series(
                    g["in_nucleus"],
                    default=False,
                )
            elif "r_signed" in g.columns:
                nucleus_flags = (
                    pd.to_numeric(g["r_signed"], errors="coerce")
                    .fillna(0.0)
                    .lt(0.0)
                )
            else:
                raise ValueError(
                    "module3.partition_by_nucleus requires either "
                    "'in_nucleus' or 'r_signed' in the Module 3 point table"
                )

            present_partitions = [
                flag
                for flag in (True, False)
                if bool(nucleus_flags.eq(flag).any())
            ]
            if len(present_partitions) > 1:
                partition_frames = [
                    g.loc[nucleus_flags.eq(flag)]
                    for flag in present_partitions
                ]
                if TARGET_SITES_PER_METASPOT is not None:
                    partition_targets = _allocate_partition_target_counts(
                        [
                            frame[
                                ["x_location", "y_location"]
                            ].to_numpy(np.float32, copy=False)
                            for frame in partition_frames
                        ],
                        target_sites_per_metaspot=TARGET_SITES_PER_METASPOT,
                        min_metaspots_per_cell=min_metaspots_per_cell,
                        max_metaspots_per_cell=max_metaspots_per_cell,
                    )
                else:
                    partition_targets = [None] * len(partition_frames)

                combined = pd.Series(
                    index=g.index,
                    dtype=np.int32,
                )
                label_offset = 0
                for frame, partition_target in zip(
                    partition_frames,
                    partition_targets,
                ):
                    _, partition_labels = _fit_one(
                        (cid, frame),
                        allow_nucleus_partition=False,
                        target_count_override=partition_target,
                    )
                    dense_labels = pd.factorize(
                        partition_labels,
                        sort=True,
                    )[0].astype(np.int32)
                    combined.loc[partition_labels.index] = (
                        dense_labels + label_offset
                    )
                    label_offset += max(
                        1,
                        int(np.unique(dense_labels).size),
                    )
                return cid, combined.astype(np.int32)

        n = len(g)
        if n < MIN_SUB_PTS:
            labels = np.zeros(n, dtype=np.int32)
            labels = _finalize_spatial_labels(
                g,
                labels,
                target_count_override=target_count_override,
            )
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
            if cell_comp_emb is not None and cid in cell_comp_emb.index:
                vec = cell_comp_emb.loc[cid, comp_cols].to_numpy(dtype=np.float32)
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
            labels = _finalize_spatial_labels(
                g,
                seed_labels,
                Xz,
                target_count_override=target_count_override,
            )
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
            labels = _finalize_spatial_labels(
                g,
                labels,
                Xz,
                target_count_override=target_count_override,
            )
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)

        min_samples = max(3, int(round(MIN_SUB_PTS * MIN_SAMPLES_FRAC)))
        import hdbscan

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
            labels = _finalize_spatial_labels(
                g,
                labels,
                Xz,
                target_count_override=target_count_override,
            )
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)
        labels = _assign_noise_to_nearest(hdb_labels, Xz)
        _min_size = int(merge_min_size) if merge_min_size is not None else int(MIN_SUB_PTS)
        labels = _merge_small_clusters(labels, Xz, min_size=_min_size)
        labels = _finalize_spatial_labels(
            g,
            labels,
            Xz,
            target_count_override=target_count_override,
        )
        return cid, pd.Series(labels, index=g.index, dtype=np.int32)

    # Lightweight backend selection for small datasets to cut scheduler/IPC overhead
    _force_threads = bool(os.environ.get("CELLSCOPE_FORCE_THREADS", "").strip())
    use_sequential = n_groups <= int(sequential_cells)
    _backend = _joblib_backend()
    if _force_threads or n_groups <= int(small_backend_cells):
        _backend = "threading"
    if use_sequential:
        iterator = _progress_iter(range(n_groups), desc='MetaSpot segmentation (seq)', total=n_groups, show=show_internal_progress)
        labels_per_cell = pd.Series(
            -1,
            index=point_df.index,
            dtype=np.int32,
        )
        for chunk in _iter_group_chunks(point_df, 1):
            for pair in chunk:
                _, labels = _fit_one(pair)
                labels_per_cell.loc[labels.index] = labels.to_numpy(
                    dtype=np.int32,
                    copy=False,
                )
        point_df = point_df.copy(deep=False)
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
    # Ward/k-d tree work releases the GIL and each cell is bounded in size.
    # Use several cells concurrently by default while keeping an explicit
    # upper bound for large production runs.
    available_jobs = _preferred_n_jobs()
    if isinstance(parallel_n_jobs, (int, float)) and int(parallel_n_jobs) > 0:
        N_JOBS = max(1, min(int(parallel_n_jobs), available_jobs))
    else:
        N_JOBS = max(1, min(8, available_jobs))
    # _backend already chosen above, but keep env override precedence
    if os.environ.get("CELLSCOPE_FORCE_THREADS", "").strip():
        _backend = "threading"

    # prefer larger chunk to reduce scheduler overhead
    chunk_size = max(64, _cluster_chunk_size(parallel_chunk_cells))
    labels_per_cell = pd.Series(
        -1,
        index=point_df.index,
        dtype=np.int32,
    )
    n_chunks = (n_groups + chunk_size - 1) // chunk_size if n_groups else 0
    iterator = _progress_iter(
        _iter_group_chunks(point_df, chunk_size),
        desc='MetaSpot segmentation (parallel)',
        total=n_chunks,
        show=show_internal_progress,
    )
    _logging.getLogger("cellscope").info(
        "Module 3 parallel segmentation | cells=%d | workers=%d | backend=%s | chunk_cells=%d",
        n_groups,
        N_JOBS,
        _backend,
        chunk_size,
    )
    # Keep one executor for all chunks.  Recreating it for every small chunk
    # dominated scheduling time on large COAD runs.
    from contextlib import nullcontext

    thread_limit = nullcontext()
    if N_JOBS > 1:
        try:
            from threadpoolctl import threadpool_limits

            # Prevent every concurrently segmented cell from starting another
            # BLAS/OpenMP pool. This context is process-wide and therefore
            # belongs outside the persistent joblib executor.
            thread_limit = threadpool_limits(limits=1)
        except Exception:
            thread_limit = nullcontext()
    with thread_limit:
        with Parallel(
            n_jobs=N_JOBS,
            backend=_backend,
            verbose=0,
            pre_dispatch=f"3*{N_JOBS}",
            batch_size="auto",
            max_nbytes=None,
        ) as parallel:
            for chunk_index, chunk in enumerate(iterator, start=1):
                try:
                    out = parallel(delayed(_fit_one)(pair) for pair in chunk)
                except Exception as e:
                    # Fallback: sequential processing to avoid hard failure
                    # (for example, a worker memory allocation failure).
                    print(
                        "[WARN] Parallel cluster chunk failed, fallback "
                        f"sequential: {e}"
                    )
                    out = [_fit_one(pair) for pair in chunk]
                if out:
                    chunk_labels = pd.concat(
                        [series for _, series in out]
                    ).astype(np.int32, copy=False)
                    labels_per_cell.loc[
                        chunk_labels.index
                    ] = chunk_labels.to_numpy(
                        dtype=np.int32,
                        copy=False,
                    )
                del out
                # Periodic collection retains bounded temporary lifetimes
                # without making collection the primary segmentation cost.
                if chunk_index % 16 == 0:
                    gc.collect()
    gc.collect()

    point_df = point_df.copy(deep=False)
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


def _metaspot_segmentation_audit(
    point_df: pd.DataFrame,
    *,
    target_sites_per_metaspot: Optional[int] = None,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
    partition_by_nucleus: bool = False,
) -> pd.DataFrame:
    """Return compact per-cell checks for MetaSpot size and compartment mixing."""
    required = {
        "cell_id",
        METASPOT_COL,
        "x_location",
        "y_location",
    }
    missing = sorted(required.difference(point_df.columns))
    if missing:
        raise KeyError(
            "MetaSpot segmentation audit requires columns: "
            + ", ".join(missing)
        )

    audit_source = point_df[
        [
            "cell_id",
            METASPOT_COL,
            "x_location",
            "y_location",
        ]
    ].copy()
    if "in_nucleus" in point_df.columns:
        audit_source["_in_nucleus"] = _coerce_bool_series(
            point_df["in_nucleus"],
            default=False,
        ).to_numpy()
    elif "r_signed" in point_df.columns:
        audit_source["_in_nucleus"] = (
            pd.to_numeric(point_df["r_signed"], errors="coerce")
            .fillna(0.0)
            .lt(0.0)
            .to_numpy()
        )
    else:
        audit_source["_in_nucleus"] = False

    grouped_spot = audit_source.groupby(
        ["cell_id", METASPOT_COL],
        sort=False,
        observed=True,
    )
    spot_stats = grouped_spot.agg(
        n_transcripts=(METASPOT_COL, "size"),
        nucleus_min=("_in_nucleus", "min"),
        nucleus_max=("_in_nucleus", "max"),
    )
    unique_sites = (
        audit_source[
            ["cell_id", METASPOT_COL, "x_location", "y_location"]
        ].drop_duplicates(
            ["cell_id", METASPOT_COL, "x_location", "y_location"]
        )
        .groupby(
            ["cell_id", METASPOT_COL],
            sort=False,
            observed=True,
        )
        .size()
        .rename("n_unique_sites")
    )
    spot_stats = spot_stats.join(unique_sites, how="left")
    spot_stats["nucleus_mixed"] = (
        spot_stats["nucleus_min"] != spot_stats["nucleus_max"]
    )
    spot_stats["mixed_transcripts"] = (
        spot_stats["n_transcripts"]
        * spot_stats["nucleus_mixed"].astype(np.int64)
    )

    per_cell = spot_stats.groupby(
        level="cell_id",
        sort=False,
        observed=True,
    ).agg(
        actual_metaspots=("n_transcripts", "size"),
        median_transcripts_per_metaspot=("n_transcripts", "median"),
        max_transcripts_per_metaspot=("n_transcripts", "max"),
        median_sites_per_metaspot=("n_unique_sites", "median"),
        max_sites_per_metaspot=("n_unique_sites", "max"),
        nucleus_mixed_metaspots=("nucleus_mixed", "sum"),
        mixed_transcripts=("mixed_transcripts", "sum"),
        n_transcripts=("n_transcripts", "sum"),
    )
    per_cell["nucleus_mixed_metaspot_fraction"] = (
        per_cell["nucleus_mixed_metaspots"]
        / per_cell["actual_metaspots"].clip(lower=1)
    )
    per_cell["nucleus_mixed_transcript_fraction"] = (
        per_cell["mixed_transcripts"]
        / per_cell["n_transcripts"].clip(lower=1)
    )

    unique_cell_sites = audit_source[
        ["cell_id", "x_location", "y_location"]
    ].drop_duplicates()
    site_counts = unique_cell_sites.groupby(
        "cell_id",
        sort=False,
        observed=True,
    ).size()
    per_cell["n_unique_sites"] = site_counts.reindex(per_cell.index).astype(int)
    if isinstance(target_sites_per_metaspot, (int, float)):
        unique_xy = unique_cell_sites[
            ["x_location", "y_location"]
        ].to_numpy(np.float32, copy=False)
        finite_site_mask = np.isfinite(unique_xy).all(axis=1)
        finite_site_counts = (
            unique_cell_sites.loc[finite_site_mask, ["cell_id"]]
            .groupby("cell_id", sort=False, observed=True)
            .size()
            .reindex(per_cell.index, fill_value=0)
            .to_numpy(dtype=np.int64, copy=False)
        )
        per_cell["requested_metaspots"] = (
            _target_metaspot_counts_from_site_counts(
                finite_site_counts,
                target_sites_per_metaspot=int(target_sites_per_metaspot),
                min_metaspots_per_cell=min_metaspots_per_cell,
                max_metaspots_per_cell=max_metaspots_per_cell,
            )
        )
    else:
        per_cell["requested_metaspots"] = np.nan
    per_cell["actual_to_requested_ratio"] = (
        per_cell["actual_metaspots"]
        / pd.to_numeric(
            per_cell["requested_metaspots"],
            errors="coerce",
        ).replace(0, np.nan)
    )
    per_cell["partition_by_nucleus"] = bool(partition_by_nucleus)
    return per_cell.reset_index()


def _grouped_nanmean_std(
    codes: np.ndarray,
    values: np.ndarray,
    n_groups: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute NaN-aware population moments for contiguous integer groups."""
    valid = np.isfinite(values)
    counts = np.bincount(codes[valid], minlength=n_groups).astype(
        np.int64,
        copy=False,
    )
    sums = np.bincount(
        codes[valid],
        weights=values[valid],
        minlength=n_groups,
    )
    sumsq = np.bincount(
        codes[valid],
        weights=np.square(values[valid], dtype=np.float64),
        minlength=n_groups,
    )
    mean = np.full(n_groups, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean, where=counts > 0)
    variance = np.full(n_groups, np.nan, dtype=np.float64)
    np.divide(sumsq, counts, out=variance, where=counts > 0)
    variance -= np.square(mean)
    variance = np.maximum(variance, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _grouped_geometry_shape(
    codes: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_groups: int,
    *,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sample-covariance elongation and spread for each MetaSpot."""
    x_mean, _ = _grouped_nanmean_std(codes, x_values, n_groups)
    y_mean, _ = _grouped_nanmean_std(codes, y_values, n_groups)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    counts = np.bincount(codes[valid], minlength=n_groups).astype(
        np.int64,
        copy=False,
    )
    x = x_values[valid].astype(np.float64, copy=False)
    y = y_values[valid].astype(np.float64, copy=False)
    grouped = lambda weights: np.bincount(
        codes[valid], weights=weights, minlength=n_groups
    )
    sum_x = grouped(x)
    sum_y = grouped(y)
    sum_x2 = grouped(np.square(x))
    sum_y2 = grouped(np.square(y))
    sum_xy = grouped(x * y)
    mx = x_mean.astype(np.float64, copy=False)
    my = y_mean.astype(np.float64, copy=False)
    count_float = counts.astype(np.float64, copy=False)
    var_x = sum_x2 - 2.0 * mx * sum_x + count_float * np.square(mx)
    var_y = sum_y2 - 2.0 * my * sum_y + count_float * np.square(my)
    cov_xy = sum_xy - mx * sum_y - my * sum_x + count_float * mx * my
    denominator = np.maximum(count_float - 1.0, 1.0)
    var_x /= denominator
    var_y /= denominator
    cov_xy /= denominator
    trace = np.maximum(var_x + var_y, 0.0)
    discriminant = np.maximum(
        np.square(trace) - 4.0 * (var_x * var_y - np.square(cov_xy)),
        0.0,
    )
    root = np.sqrt(discriminant)
    lambda_1 = np.maximum(0.5 * (trace + root), eps)
    lambda_2 = np.maximum(0.5 * (trace - root), eps)
    elongation = np.full(n_groups, np.nan, dtype=np.float32)
    spread = np.full(n_groups, np.nan, dtype=np.float32)
    valid_shape = counts >= 3
    elongation[valid_shape] = (
        lambda_1[valid_shape] / lambda_2[valid_shape]
    ).astype(np.float32)
    spread[valid_shape] = np.sqrt(trace[valid_shape]).astype(np.float32)
    return elongation, spread


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
    del gene_codes
    adata1 = ad.AnnData(X=X, obs=pd.DataFrame(index=row_cats), var=pd.DataFrame(index=gene_uni))

    n_groups = len(row_cats)
    # Do not materialize a five-column, row-sorted transcript array.  The
    # COAD run contains hundreds of millions of rows, so that copy alone was
    # several GiB before any per-MetaSpot statistics were calculated.
    codes = row_codes.astype(
        np.int32 if n_groups <= np.iinfo(np.int32).max else np.int64,
        copy=False,
    )
    if codes is not row_codes:
        del row_codes
    counts = np.bincount(codes, minlength=n_groups).astype(np.int64, copy=False)

    r_values = point_df['r_signed'].to_numpy(dtype=np.float32, copy=False)
    r_mean, r_std = _grouped_nanmean_std(codes, r_values, n_groups)
    # Pandas executes grouped quantiles in native code.  This preserves the
    # exact NumPy interpolation rule used previously while avoiding one Python
    # percentile call per MetaSpot.
    quantiles = (
        pd.Series(r_values, copy=False)
        .groupby(codes, sort=True, observed=True)
        .quantile([0.1, 0.5, 0.9])
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_groups, 3)
    )
    r_p10, r_p50, r_p90 = quantiles.T
    del quantiles, r_values

    d1_values = point_df['d_cell_norm'].to_numpy(dtype=np.float32, copy=False)
    d1_mean, _ = _grouped_nanmean_std(codes, d1_values, n_groups)
    del d1_values
    d2_values = point_df['d_nuc_norm'].to_numpy(dtype=np.float32, copy=False)
    d2_mean, _ = _grouped_nanmean_std(codes, d2_values, n_groups)
    del d2_values
    x_values = point_df['x_location'].to_numpy(dtype=np.float32, copy=False)
    y_values = point_df['y_location'].to_numpy(dtype=np.float32, copy=False)
    elong, spread = _grouped_geometry_shape(
        codes,
        x_values,
        y_values,
        n_groups,
        eps=1e-8,
    )
    del x_values, y_values
    log_n = np.log1p(counts).astype(np.float32)

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


def _attach_metaspot_spatial_metadata(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
) -> ad.AnnData:
    """Attach one owning cell and one physical centroid to every MetaSpot."""
    required = {METASPOT_COL, "x_location", "y_location"}
    missing = sorted(required.difference(point_df.columns))
    if missing:
        raise ValueError(
            "MetaSpot graph construction requires point_df columns "
            f"{missing}"
        )

    columns = [METASPOT_COL, "x_location", "y_location"]
    if "cell_id" in point_df.columns:
        columns.append("cell_id")
    frame = point_df.loc[:, columns].copy()
    frame[METASPOT_COL] = frame[METASPOT_COL].astype(str)

    grouped = frame.groupby(METASPOT_COL, observed=True, sort=False)
    centroids = grouped[["x_location", "y_location"]].mean()
    centroids.columns = ["x_centroid", "y_centroid"]

    if "cell_id" in frame.columns:
        owner_counts = grouped["cell_id"].nunique(dropna=False)
        if owner_counts.ne(1).any():
            bad = owner_counts.index[owner_counts.ne(1)].astype(str).tolist()[:5]
            raise ValueError(
                "Each MetaSpot must belong to exactly one cell; conflicting "
                f"owners found for {bad}"
            )
        owners = grouped["cell_id"].first().astype(str).rename("cell_id")
    elif "cell_id" in adata1.obs.columns:
        owners = adata1.obs["cell_id"].astype(str)
    else:
        raise ValueError(
            "MetaSpot graph construction requires cell_id in point_df or "
            "adata1.obs"
        )

    target = pd.Index(adata1.obs_names.astype(str))
    centroids.index = centroids.index.astype(str)
    owners.index = owners.index.astype(str)
    centroids = centroids.reindex(target)
    owners = owners.reindex(target)
    if centroids.isna().any(axis=None) or owners.isna().any():
        missing_ids = target[
            centroids.isna().any(axis=1).to_numpy() | owners.isna().to_numpy()
        ].tolist()[:5]
        raise ValueError(
            "Missing MetaSpot centroid or owner for observations "
            f"{missing_ids}"
        )

    adata1.obs = adata1.obs.copy()
    adata1.obs["cell_id"] = owners.to_numpy(dtype=str)
    adata1.obs["x_centroid"] = centroids["x_centroid"].to_numpy(np.float32)
    adata1.obs["y_centroid"] = centroids["y_centroid"].to_numpy(np.float32)
    return adata1


def _metaspot_expression_embedding(
    adata1: ad.AnnData,
    n_components: int = 32,
) -> np.ndarray:
    """Build a compact TF-IDF/SVD expression representation per MetaSpot."""
    X = (
        adata1.X.tocsr().astype(np.float32)
        if sp.issparse(adata1.X)
        else sp.csr_matrix(np.asarray(adata1.X, dtype=np.float32))
    )
    if X.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)

    row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    row_sum[row_sum <= 0] = 1.0
    tf = X.multiply(1.0 / row_sum[:, None])
    gene_df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log1p(X.shape[0] / (1.0 + gene_df)).astype(np.float32)
    tfidf = tf.multiply(idf)

    max_components = min(
        max(1, int(n_components)),
        max(1, X.shape[0] - 1),
        max(1, X.shape[1] - 1),
    )
    if X.shape[0] == 1 or X.shape[1] == 1:
        latent = tfidf.toarray().astype(np.float32, copy=False)
    else:
        latent = TruncatedSVD(
            n_components=max_components,
            random_state=42,
            n_iter=5,
        ).fit_transform(tfidf).astype(np.float32)
    latent = StandardScaler().fit_transform(latent).astype(np.float32)
    latent[~np.isfinite(latent)] = 0.0
    return latent


def _symmetric_knn_connectivity(
    values: np.ndarray,
    n_neighbors: int,
) -> sp.csr_matrix:
    """Return a symmetric binary kNN graph with k clamped to data size."""
    from sklearn.neighbors import kneighbors_graph

    n_obs = int(values.shape[0])
    if n_obs <= 1 or int(n_neighbors) <= 0:
        return sp.csr_matrix((n_obs, n_obs), dtype=np.float32)
    k = min(max(1, int(n_neighbors)), n_obs - 1)
    # sklearn's brute-force distance kernel is already vectorized. Spawning a
    # joblib task per query chunk is slower for both the global graph and the
    # many small within-cell graphs, and can multiply BLAS pools. Keep the
    # exact graph deterministic and single-scheduled; BLAS/OpenMP remains
    # controlled by the process-level runtime settings.
    graph = kneighbors_graph(
        values,
        n_neighbors=k,
        mode="connectivity",
        include_self=False,
        n_jobs=1,
    ).astype(np.float32)
    graph = graph.maximum(graph.T).tocsr()
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _within_cell_spatial_connectivity(
    coordinates: np.ndarray,
    cell_ids: Sequence[object],
    n_neighbors: int,
) -> sp.csr_matrix:
    """Build physical kNN edges, strictly within each owning cell."""
    n_obs = int(coordinates.shape[0])
    if n_obs == 0 or int(n_neighbors) <= 0:
        return sp.csr_matrix((n_obs, n_obs), dtype=np.float32)

    cells = np.asarray(cell_ids).astype(str)
    cell_codes, _ = pd.factorize(cells, sort=False)
    cell_codes = np.asarray(cell_codes, dtype=np.int64)
    order = np.argsort(cell_codes, kind="stable").astype(
        np.int64,
        copy=False,
    )
    counts = np.bincount(cell_codes).astype(np.int64, copy=False)
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for group_index in range(counts.size):
        indices = order[offsets[group_index]:offsets[group_index + 1]]
        if indices.size <= 1:
            continue
        local = _symmetric_knn_connectivity(
            coordinates[indices],
            min(int(n_neighbors), indices.size - 1),
        ).tocoo()
        if local.nnz:
            rows.append(indices[local.row].astype(np.int64, copy=False))
            cols.append(indices[local.col].astype(np.int64, copy=False))

    if not rows:
        return sp.csr_matrix((n_obs, n_obs), dtype=np.float32)
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    data = np.ones(row.size, dtype=np.float32)
    graph = sp.csr_matrix((data, (row, col)), shape=(n_obs, n_obs))
    graph = graph.maximum(graph.T).tocsr()
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _normalize_sparse_adjacency(
    adjacency: sp.csr_matrix,
    mode: str = "gcn",
) -> sp.csr_matrix:
    """Normalize an adjacency matrix after adding explicit self loops."""
    from scipy.sparse import diags, eye

    graph = adjacency.tocsr().astype(np.float32, copy=False)
    graph = graph + eye(graph.shape[0], dtype=np.float32, format="csr")
    graph.sum_duplicates()
    graph.eliminate_zeros()
    norm_mode = str(mode).strip().lower()
    degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    degree[degree <= 0] = 1.0
    if norm_mode == "gcn":
        scale = 1.0 / np.sqrt(degree)
        graph = (diags(scale) @ graph @ diags(scale)).tocsr()
    elif norm_mode == "rw":
        graph = (diags(1.0 / degree) @ graph).tocsr()
    elif norm_mode != "none":
        raise ValueError(
            "MetaSpot graph edge_norm must be one of {'gcn', 'rw', 'none'}"
        )
    graph = graph.astype(np.float32, copy=False)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def build_metaspot_graph(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    *,
    k_expr: int = 15,
    k_spatial: int = 6,
    expr_n_components: int = 32,
    alpha_expr: float = 0.8,
    alpha_spatial: float = 0.2,
    edge_norm: str = "gcn",
    use_geometry_features: bool = False,
    expression_cross_cell_only: bool = False,
    max_full_batch_nodes: int = 200_000,
    execution_profile: str = "exact",
    knn_backend: str = "exact",
    ann_candidate_multiplier: int = 4,
    ann_query_chunk_nodes: int = 100_000,
    ann_train_sample: int = 1_000_000,
    faiss_hnsw_m: int = 32,
    faiss_nprobe: int = 32,
    faiss_pq_subvectors: int = 8,
    faiss_pq_bits: int = 8,
    faiss_use_gpu: Any = "auto",
    faiss_gpu_device: int = 0,
) -> ad.AnnData:
    """Construct a genuine MetaSpot expression + within-cell spatial graph."""
    graph_started = _time.perf_counter()
    timing_seconds: Dict[str, float] = {}
    if adata1.n_obs == 0:
        raise ValueError("Cannot build a MetaSpot graph with zero observations")
    profile = str(execution_profile).strip().lower()
    if profile not in {"exact", "sampled", "streaming"}:
        raise ValueError(
            "execution_profile must be one of {'exact','sampled','streaming'}"
        )
    effective_knn_backend = str(knn_backend).strip().lower().replace(
        "-", "_"
    )
    if effective_knn_backend == "auto":
        effective_knn_backend = "exact"
    if (
        profile == "exact"
        and int(max_full_batch_nodes) > 0
        and adata1.n_obs > int(max_full_batch_nodes)
    ):
        raise ValueError(
            f"MetaSpot full-batch graph mode received {adata1.n_obs:,} nodes, "
            f"above module7.max_full_batch_nodes={int(max_full_batch_nodes):,}. "
            "Use execution.profile: sampled/streaming with a supported ANN "
            "backend; increasing this limit only removes the exact-path guard."
        )
    if float(alpha_expr) < 0 or float(alpha_spatial) < 0:
        raise ValueError("MetaSpot graph fusion weights must be non-negative")
    weight_sum = float(alpha_expr) + float(alpha_spatial)
    if weight_sum <= 0:
        raise ValueError("At least one MetaSpot graph fusion weight must be > 0")

    step_started = _time.perf_counter()
    adata1 = _attach_metaspot_spatial_metadata(adata1, point_df)
    timing_seconds["attach_spatial_metadata"] = (
        _time.perf_counter() - step_started
    )
    step_started = _time.perf_counter()
    expression = _metaspot_expression_embedding(
        adata1,
        n_components=expr_n_components,
    )
    timing_seconds["expression_embedding"] = _time.perf_counter() - step_started
    coordinates = adata1.obs[
        ["x_centroid", "y_centroid"]
    ].to_numpy(np.float32)
    from .scalable_graph import build_knn_connectivity

    step_started = _time.perf_counter()
    expression_graph, ann_metadata = build_knn_connectivity(
        expression,
        n_neighbors=k_expr,
        backend=effective_knn_backend,
        excluded_groups=(
            adata1.obs["cell_id"].astype(str).to_numpy()
            if expression_cross_cell_only
            else None
        ),
        candidate_multiplier=ann_candidate_multiplier,
        query_chunk_size=ann_query_chunk_nodes,
        train_sample=ann_train_sample,
        hnsw_m=faiss_hnsw_m,
        nprobe=faiss_nprobe,
        pq_subvectors=faiss_pq_subvectors,
        pq_bits=faiss_pq_bits,
        faiss_use_gpu=faiss_use_gpu,
        faiss_gpu_device=faiss_gpu_device,
        random_seed=42,
        n_jobs=_preferred_n_jobs(),
    )
    timing_seconds["expression_knn"] = _time.perf_counter() - step_started
    step_started = _time.perf_counter()
    spatial_graph = _within_cell_spatial_connectivity(
        coordinates,
        adata1.obs["cell_id"].astype(str).to_numpy(),
        k_spatial,
    )
    timing_seconds["within_cell_spatial"] = _time.perf_counter() - step_started
    if expression_cross_cell_only:
        if int(ann_metadata.get("nodes_below_available_degree", 0)) > 0:
            raise RuntimeError(
                "Cross-cell MetaSpot expression graph could not provide all "
                "globally available neighbors; increase ANN search quality "
                "or use module7.knn_backend='exact'"
            )
        if expression_graph.nnz == 0:
            raise ValueError(
                "Cross-cell MetaSpot expression graph contains no edges; "
                "disable module7.metaspot_expr_cross_cell_only or provide "
                "more than one cell"
            )

    step_started = _time.perf_counter()
    expr_weight = float(alpha_expr) / weight_sum
    spatial_weight = float(alpha_spatial) / weight_sum
    fused = (
        expr_weight * expression_graph
        + spatial_weight * spatial_graph
    ).tocsr()
    fused = ((fused + fused.T) * 0.5).tocsr()
    fused.eliminate_zeros()
    fused.sort_indices()

    adata1.obsm["X_expr_metaspot"] = expression
    adata1.obsm["X_expr_metaspot_scaled"] = expression.copy()
    feature_keys = ["X_expr_metaspot_scaled"]

    if use_geometry_features:
        geometry_columns = [
            column
            for column in [
                "r_mean",
                "r_std",
                "r_p50",
                "d_cell_norm_mean",
                "d_nuc_norm_mean",
                "elongation",
                "spread",
                "log_n",
            ]
            if column in adata1.obs.columns
        ]
        geometry = adata1.obs[geometry_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        for column_index in range(geometry.shape[1]):
            values = geometry[:, column_index]
            finite = np.isfinite(values)
            fill_value = float(np.median(values[finite])) if finite.any() else 0.0
            values[~finite] = fill_value
            geometry[:, column_index] = values
        geometry = StandardScaler().fit_transform(geometry).astype(np.float32)
        geometry[~np.isfinite(geometry)] = 0.0
        adata1.obsm["X_geom_metaspot_scaled"] = geometry
        feature_keys.append("X_geom_metaspot_scaled")

    adata1.obsp["metaspot_expr_connectivities"] = expression_graph
    adata1.obsp["metaspot_spatial_connectivities"] = spatial_graph
    adata1.obsp[METASPOT_GRAPH_KEY] = _normalize_sparse_adjacency(
        fused,
        mode=edge_norm,
    )
    checksum = _obs_names_checksum(adata1.obs_names)
    timing_seconds["fusion_and_attach"] = _time.perf_counter() - step_started
    timing_seconds["total"] = _time.perf_counter() - graph_started
    adata1.uns["metaspot_graph"] = {
        "schema_version": int(METASPOT_GRAPH_SCHEMA_VERSION),
        "node_unit": "metaspot",
        "node_id_checksum": checksum,
        "n_nodes": int(adata1.n_obs),
        "k_expr": int(min(max(0, k_expr), max(0, adata1.n_obs - 1))),
        "k_spatial": int(max(0, k_spatial)),
        "alpha_expr": expr_weight,
        "alpha_spatial": spatial_weight,
        "edge_norm": str(edge_norm).lower(),
        "expression_edges_nnz": int(expression_graph.nnz),
        "spatial_edges_nnz": int(spatial_graph.nnz),
        "fused_edges_nnz": int(fused.nnz),
        "spatial_edges_within_cell_only": True,
        "expression_edges_cross_cell_only": bool(
            expression_cross_cell_only
        ),
        "uses_boundary_geometry_features": bool(use_geometry_features),
        "max_full_batch_nodes": int(max_full_batch_nodes),
        "execution_profile": profile,
        "knn_backend": str(ann_metadata.get("backend", knn_backend)),
        "ann_metadata": ann_metadata,
        "timing_seconds": {
            key: float(value) for key, value in timing_seconds.items()
        },
        "feature_keys": feature_keys,
    }
    return adata1


def _metaspot_cache_matches(
    cached: ad.AnnData,
    reference: ad.AnnData,
    *,
    required_obsm: Sequence[str] = (),
    required_obsp: Sequence[str] = (),
) -> Tuple[bool, str]:
    """Validate MetaSpot graph/embedding caches before downstream reuse."""
    cached_names = pd.Index(cached.obs_names.astype(str))
    reference_names = pd.Index(reference.obs_names.astype(str))
    if not cached_names.equals(reference_names):
        return False, "MetaSpot observation order differs"
    for key in required_obsm:
        if key not in cached.obsm or cached.obsm[key].shape[0] != cached.n_obs:
            return False, f"missing or misaligned obsm[{key!r}]"
    for key in required_obsp:
        if key not in cached.obsp or cached.obsp[key].shape != (
            cached.n_obs,
            cached.n_obs,
        ):
            return False, f"missing or misaligned obsp[{key!r}]"
    expected_checksum = _obs_names_checksum(reference_names)
    graph_meta = cached.uns.get("metaspot_graph", {})
    if graph_meta and graph_meta.get("node_id_checksum") != expected_checksum:
        return False, "MetaSpot graph node checksum differs"
    return True, "aligned"


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


def _stratified_metaspot_sample(
    cell_ids: np.ndarray,
    sample_n: int,
    *,
    seed: int,
) -> np.ndarray:
    """Sample MetaSpots while preventing large cells from dominating."""
    cell_ids = np.asarray(cell_ids).astype(str)
    n = int(cell_ids.size)
    sample_n = max(1, min(int(sample_n), n))
    rng = np.random.default_rng(int(seed))
    if sample_n >= n:
        return rng.permutation(n).astype(np.int64, copy=False)

    _, cell_codes = np.unique(cell_ids, return_inverse=True)
    n_cells = int(cell_codes.max()) + 1 if cell_codes.size else 0
    if n_cells == 0:
        return rng.choice(n, size=sample_n, replace=False)

    # Sort once so looking up a cell does not repeatedly scan all MetaSpots.
    # This matters when both n and the number of cells are very large.
    order = np.argsort(cell_codes, kind="stable")
    cell_counts = np.bincount(cell_codes, minlength=n_cells)
    cell_offsets = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.cumsum(cell_counts, dtype=np.int64),
        ]
    )

    def _members(code: int) -> np.ndarray:
        return order[cell_offsets[code]:cell_offsets[code + 1]]

    selected: List[np.ndarray] = []
    if sample_n < n_cells:
        chosen_cells = rng.choice(n_cells, size=sample_n, replace=False)
        for code in chosen_cells:
            members = _members(int(code))
            selected.append(
                np.asarray([rng.choice(members)], dtype=np.int64)
            )
    else:
        quota = max(1, sample_n // n_cells)
        for code in range(n_cells):
            members = _members(code)
            take = min(quota, members.size)
            if take:
                selected.append(
                    rng.choice(members, size=take, replace=False)
                )

    chosen = (
        np.concatenate(selected).astype(np.int64, copy=False)
        if selected
        else np.empty(0, dtype=np.int64)
    )
    chosen = np.unique(chosen)
    remaining = sample_n - chosen.size
    if remaining > 0:
        available = np.ones(n, dtype=bool)
        available[chosen] = False
        pool = np.flatnonzero(available)
        chosen = np.concatenate(
            [chosen, rng.choice(pool, size=remaining, replace=False)]
        )
    rng.shuffle(chosen)
    return chosen.astype(np.int64, copy=False)


def _assign_to_metadomain_centers(
    values: np.ndarray,
    centers: np.ndarray,
    *,
    use_cosine: bool,
    chunk_size: int,
) -> Tuple[np.ndarray, float]:
    """Assign rows to centers and return mean held-out distortion."""
    from sklearn.preprocessing import normalize

    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    n = int(values.shape[0])
    labels = np.empty(n, dtype=np.int32)
    total_loss = 0.0
    step = max(256, int(chunk_size))

    if use_cosine:
        centers_eval = normalize(centers, norm="l2").astype(
            np.float32,
            copy=False,
        )
        for start in range(0, n, step):
            end = min(n, start + step)
            similarity = values[start:end] @ centers_eval.T
            local_labels = np.argmax(similarity, axis=1)
            labels[start:end] = local_labels.astype(np.int32, copy=False)
            total_loss += float(
                np.sum(
                    1.0
                    - similarity[
                        np.arange(end - start),
                        local_labels,
                    ]
                )
            )
    else:
        center_norm = np.einsum("ij,ij->i", centers, centers)
        for start in range(0, n, step):
            end = min(n, start + step)
            chunk = values[start:end]
            chunk_norm = np.einsum("ij,ij->i", chunk, chunk)
            distance = (
                chunk_norm[:, None]
                + center_norm[None, :]
                - 2.0 * (chunk @ centers.T)
            )
            local_labels = np.argmin(distance, axis=1)
            labels[start:end] = local_labels.astype(np.int32, copy=False)
            total_loss += float(
                np.sum(
                    distance[
                        np.arange(end - start),
                        local_labels,
                    ]
                )
            )
    return labels, total_loss / max(1, n)


def _sampled_knee_candidates(
    *,
    n_sub: int,
    train_n: int,
    auto_k_min: int,
    auto_k_max: int,
    min_average_size: int,
    max_average_size: int,
    min_train_per_cluster: int,
    n_candidates: int,
) -> np.ndarray:
    """Build a logarithmic K grid with data-size and sample-size guards."""
    low = max(
        2,
        int(auto_k_min),
        int(math.ceil(n_sub / max(1, int(max_average_size)))),
    )
    high = min(
        int(auto_k_max),
        max(2, int(n_sub // max(1, int(min_average_size)))),
        max(2, int(train_n // max(1, int(min_train_per_cluster)))),
    )
    if high < low:
        low = high
    if high <= 2:
        return np.asarray([2], dtype=np.int32)
    if low == high:
        return np.asarray([low], dtype=np.int32)

    candidates = np.unique(
        np.rint(
            np.geomspace(
                low,
                high,
                num=max(3, int(n_candidates)),
            )
        ).astype(np.int32)
    )
    return candidates[(candidates >= 2) & (candidates <= high)]


def _select_sampled_knee(
    candidate_k: np.ndarray,
    distortion: np.ndarray,
    stability: np.ndarray,
    small_fraction: np.ndarray,
    *,
    stability_min: float,
    max_small_fraction: float,
    fallback_k: int,
) -> Tuple[int, int, str, np.ndarray]:
    """Select the distortion knee, backing off when quality guards fail."""
    candidate_k = np.asarray(candidate_k, dtype=np.int32)
    distortion = np.asarray(distortion, dtype=float)
    stability = np.asarray(stability, dtype=float)
    small_fraction = np.asarray(small_fraction, dtype=float)
    finite = np.isfinite(distortion)
    if not finite.any():
        chosen = int(
            candidate_k[
                np.argmin(np.abs(candidate_k.astype(float) - fallback_k))
            ]
        )
        return chosen, chosen, "sqrt_fallback_no_valid_fit", finite

    valid_indices = np.flatnonzero(finite)
    k_valid = candidate_k[valid_indices].astype(float)
    d_valid = distortion[valid_indices]
    if len(valid_indices) >= 3 and np.ptp(d_valid) > 1e-12:
        x = np.log(k_valid)
        x = (x - x.min()) / max(np.ptp(x), 1e-12)
        y = (d_valid - d_valid.min()) / max(np.ptp(d_valid), 1e-12)
        knee_local = int(np.argmax((1.0 - x) - y))
        knee_index = int(valid_indices[knee_local])
    else:
        knee_index = int(
            valid_indices[
                np.argmin(
                    np.abs(k_valid - float(fallback_k))
                )
            ]
        )
    knee_k = int(candidate_k[knee_index])

    stability_ok = np.isnan(stability) | (
        stability >= float(stability_min)
    )
    small_ok = np.isnan(small_fraction) | (
        small_fraction <= float(max_small_fraction)
    )
    feasible = finite & stability_ok & small_ok

    at_or_below_knee = np.flatnonzero(
        feasible & (candidate_k <= knee_k)
    )
    if at_or_below_knee.size:
        selected_index = int(at_or_below_knee[-1])
        reason = (
            "distortion_knee"
            if selected_index == knee_index
            else "guarded_lower_k"
        )
        return (
            int(candidate_k[selected_index]),
            knee_k,
            reason,
            feasible,
        )

    feasible_indices = np.flatnonzero(feasible)
    if feasible_indices.size:
        selected_index = int(
            feasible_indices[
                np.argmin(
                    np.abs(
                        candidate_k[feasible_indices].astype(float)
                        - float(knee_k)
                    )
                )
            ]
        )
        return (
            int(candidate_k[selected_index]),
            knee_k,
            "nearest_feasible_k",
            feasible,
        )

    selected_index = int(
        valid_indices[
            np.argmin(
                np.abs(k_valid - float(fallback_k))
            )
        ]
    )
    return (
        int(candidate_k[selected_index]),
        knee_k,
        "sqrt_fallback_no_feasible_k",
        feasible,
    )


def _choose_K_sampled_knee(
    values: np.ndarray,
    cell_ids: np.ndarray,
    *,
    use_cosine: bool,
    config: Mapping[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    """Choose MetaDomain K on a bounded sample using a held-out knee.

    Candidate models are fit only on a cell-balanced sample. The selected
    model size is the held-out cosine/Euclidean distortion knee, with
    cross-seed stability and tiny-cluster guards. This keeps automatic K
    selection bounded for very large datasets.
    """
    values = np.asarray(values, dtype=np.float32)
    n_sub = int(values.shape[0])
    if n_sub < 4:
        selected = max(2, min(n_sub, int(round(math.sqrt(n_sub)))))
        return selected, {
            "method": "sampled_knee",
            "selection_reason": "tiny_dataset_fallback",
            "selected_k": int(selected),
            "knee_k": int(selected),
            "candidate_k": np.asarray([selected], dtype=np.int32),
            "distortion": np.asarray([np.nan], dtype=np.float32),
            "stability_ari": np.asarray([np.nan], dtype=np.float32),
            "small_cluster_fraction": np.asarray(
                [np.nan],
                dtype=np.float32,
            ),
        }

    train_cap = max(20, int(config.get("auto_k_train_sample", 50000)))
    validation_cap = max(
        20,
        int(config.get("auto_k_validation_sample", 3000)),
    )
    validation_n = min(
        validation_cap,
        max(20, int(round(0.25 * n_sub))),
        n_sub - 2,
    )
    train_n = min(train_cap, n_sub - validation_n)
    if train_n < 20:
        train_n = max(2, n_sub - max(1, n_sub // 5))
        validation_n = n_sub - train_n

    seeds_raw = config.get("auto_k_seeds", [42, 43])
    if isinstance(seeds_raw, (int, np.integer)):
        seeds = [int(seeds_raw)]
    elif isinstance(seeds_raw, Sequence) and not isinstance(
        seeds_raw,
        (str, bytes),
    ):
        seeds = [int(seed) for seed in seeds_raw]
    else:
        seeds = [42, 43]
    seeds = list(dict.fromkeys(seeds)) or [42, 43]

    sample_seed = int(config.get("auto_k_sample_seed", 2026))
    sampled = _stratified_metaspot_sample(
        np.asarray(cell_ids),
        train_n + validation_n,
        seed=sample_seed,
    )
    train_indices = sampled[:train_n]
    validation_indices = sampled[train_n:]
    X_train = values[train_indices]
    X_validation = values[validation_indices]

    auto_k_max = int(
        config.get("auto_k_max", config.get("K_max", 8000))
    )
    candidates = _sampled_knee_candidates(
        n_sub=n_sub,
        train_n=train_n,
        auto_k_min=int(config.get("auto_k_min", 20)),
        auto_k_max=auto_k_max,
        min_average_size=int(
            config.get("auto_k_min_average_size", 20)
        ),
        max_average_size=int(
            config.get("auto_k_max_average_size", 500)
        ),
        min_train_per_cluster=int(
            config.get("auto_k_min_train_per_cluster", 10)
        ),
        n_candidates=int(config.get("auto_k_n_candidates", 8)),
    )

    search_batch_size = max(
        256,
        min(
            int(config.get("auto_k_batch_size", 2048)),
            train_n,
        ),
    )
    search_max_iter = max(
        5,
        int(config.get("auto_k_max_iter", 30)),
    )
    assign_chunk = max(
        256,
        int(config.get("assign_chunk_size", 50000)),
    )
    full_eval_max = max(
        validation_n,
        int(config.get("auto_k_full_eval_max", 100000)),
    )
    if n_sub <= full_eval_max:
        X_occupancy = values
        occupancy_scale = 1.0
    else:
        X_occupancy = values[sampled]
        occupancy_scale = n_sub / max(1, sampled.size)

    rows: List[Dict[str, float]] = []
    for candidate in candidates:
        validation_labels: List[np.ndarray] = []
        candidate_distortion: List[float] = []
        first_centers: Optional[np.ndarray] = None
        for seed in seeds:
            try:
                model = MiniBatchKMeans(
                    n_clusters=int(candidate),
                    init="k-means++",
                    n_init=1,
                    random_state=int(seed),
                    batch_size=search_batch_size,
                    max_iter=search_max_iter,
                    reassignment_ratio=0.01,
                    verbose=0,
                )
                model.fit(X_train)
                labels, loss = _assign_to_metadomain_centers(
                    X_validation,
                    model.cluster_centers_,
                    use_cosine=use_cosine,
                    chunk_size=assign_chunk,
                )
                validation_labels.append(labels)
                candidate_distortion.append(float(loss))
                if first_centers is None:
                    first_centers = model.cluster_centers_.astype(
                        np.float32,
                        copy=True,
                    )
            except Exception:
                validation_labels = []
                candidate_distortion = []
                first_centers = None
                break

        if not candidate_distortion or first_centers is None:
            rows.append(
                {
                    "k": float(candidate),
                    "distortion": np.nan,
                    "stability_ari": np.nan,
                    "small_cluster_fraction": np.nan,
                    "empty_cluster_fraction": np.nan,
                }
            )
            continue

        pairwise_stability: List[float] = []
        for left in range(len(validation_labels)):
            for right in range(left + 1, len(validation_labels)):
                pairwise_stability.append(
                    float(
                        adjusted_rand_score(
                            validation_labels[left],
                            validation_labels[right],
                        )
                    )
                )
        stability = (
            float(np.mean(pairwise_stability))
            if pairwise_stability
            else np.nan
        )

        occupancy_labels, _ = _assign_to_metadomain_centers(
            X_occupancy,
            first_centers,
            use_cosine=use_cosine,
            chunk_size=assign_chunk,
        )
        occupancy = np.bincount(
            occupancy_labels,
            minlength=int(candidate),
        ).astype(float)
        expected_full_occupancy = occupancy * occupancy_scale
        tiny = expected_full_occupancy < int(
            config.get("min_domain_size", 5)
        )
        small_fraction = float(
            occupancy[tiny].sum() / max(1, occupancy.sum())
        )
        rows.append(
            {
                "k": float(candidate),
                "distortion": float(np.mean(candidate_distortion)),
                "stability_ari": stability,
                "small_cluster_fraction": small_fraction,
                "empty_cluster_fraction": float(np.mean(occupancy == 0)),
            }
        )

    diagnostics = pd.DataFrame(rows).sort_values("k")
    candidate_array = diagnostics["k"].to_numpy(np.int32)
    fallback_k = int(
        np.clip(
            round(math.sqrt(n_sub)),
            candidate_array.min(),
            candidate_array.max(),
        )
    )
    stability_min = float(config.get("auto_k_stability_min", 0.4))
    max_small_fraction = float(
        config.get("auto_k_max_small_cluster_fraction", 0.01)
    )
    selected_k, knee_k, reason, feasible = _select_sampled_knee(
        candidate_array,
        diagnostics["distortion"].to_numpy(float),
        diagnostics["stability_ari"].to_numpy(float),
        diagnostics["small_cluster_fraction"].to_numpy(float),
        stability_min=stability_min,
        max_small_fraction=max_small_fraction,
        fallback_k=fallback_k,
    )

    result: Dict[str, Any] = {
        "method": "sampled_knee",
        "selection_reason": reason,
        "selected_k": int(selected_k),
        "knee_k": int(knee_k),
        "fallback_k": int(fallback_k),
        "train_size": int(train_n),
        "validation_size": int(validation_n),
        "use_cosine": bool(use_cosine),
        "stability_min": stability_min,
        "max_small_cluster_fraction": max_small_fraction,
        "candidate_k": candidate_array,
        "distortion": diagnostics["distortion"].to_numpy(np.float32),
        "stability_ari": diagnostics["stability_ari"].to_numpy(
            np.float32
        ),
        "small_cluster_fraction": diagnostics[
            "small_cluster_fraction"
        ].to_numpy(np.float32),
        "empty_cluster_fraction": diagnostics[
            "empty_cluster_fraction"
        ].to_numpy(np.float32),
        "feasible": np.asarray(feasible, dtype=bool),
    }
    return int(selected_k), result


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


def _metadomain_feature_matrix(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    *,
    cluster_rep: str,
    cluster_rep_decell: bool,
    speed_preset: str,
    use_spatial_feats: bool,
    use_decell: bool,
    use_cosine: bool,
    w_gc: float,
    w_expr: float,
    w_sp: float,
    use_gc: bool,
    use_expr: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select either the learned MetaSpot representation or legacy features."""
    from sklearn.preprocessing import normalize

    rep = str(cluster_rep).strip()
    if rep.lower() in {"", "legacy", "legacy_features", "auto"}:
        return _construct_features_simple(
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

    if rep not in adata1.obsm:
        raise KeyError(
            f"module5.cluster_rep={rep!r} is missing from adata1.obsm; "
            "refusing to fall back to legacy features"
        )
    values = np.asarray(adata1.obsm[rep], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != adata1.n_obs:
        raise ValueError(
            f"adata1.obsm[{rep!r}] must have shape (n_metaspots, d), "
            f"got {values.shape}"
        )
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(
            f"adata1.obsm[{rep!r}] must be finite and have at least one feature"
        )

    if "cell_id" not in adata1.obs.columns:
        if not {METASPOT_COL, "cell_id"}.issubset(point_df.columns):
            raise ValueError(
                "Embedding-based MetaDomain clustering requires cell_id"
            )
        owners = (
            point_df[[METASPOT_COL, "cell_id"]]
            .assign(**{METASPOT_COL: lambda frame: frame[METASPOT_COL].astype(str)})
            .drop_duplicates()
            .set_index(METASPOT_COL)["cell_id"]
            .astype(str)
        )
        if owners.index.duplicated().any():
            raise ValueError("A MetaSpot has more than one owning cell")
        adata1.obs = adata1.obs.copy()
        adata1.obs["cell_id"] = owners.reindex(
            adata1.obs_names.astype(str)
        ).to_numpy()
    cell_ids = adata1.obs["cell_id"].astype(str).to_numpy()
    unique_cells, cell_codes = np.unique(cell_ids, return_inverse=True)

    matrix = StandardScaler().fit_transform(values).astype(np.float32)
    if cluster_rep_decell:
        sums = np.zeros((len(unique_cells), matrix.shape[1]), dtype=np.float32)
        np.add.at(sums, cell_codes, matrix)
        counts = np.bincount(cell_codes).astype(np.float32)
        means = sums / np.maximum(counts[:, None], 1.0)
        matrix = matrix - means[cell_codes]
        matrix = StandardScaler().fit_transform(matrix).astype(np.float32)
    if use_cosine:
        matrix = normalize(matrix, norm="l2").astype(np.float32)
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix, cell_ids, unique_cells, cell_codes


def _merge_small_metadomain_clusters(
    labels: np.ndarray,
    values: np.ndarray,
    min_size: int,
    *,
    use_cosine: bool,
) -> np.ndarray:
    """Merge tiny clusters into the closest sufficiently sized prototype."""
    labels = np.asarray(labels, dtype=np.int32).copy()
    min_size = max(1, int(min_size))
    if min_size <= 1 or labels.size == 0:
        return labels

    cluster_ids, counts = np.unique(labels, return_counts=True)
    large_ids = cluster_ids[counts >= min_size]
    small_ids = cluster_ids[counts < min_size]
    if large_ids.size == 0 or small_ids.size == 0:
        return labels

    centers = np.vstack(
        [values[labels == cluster_id].mean(axis=0) for cluster_id in large_ids]
    ).astype(np.float32)
    points = values
    if use_cosine:
        centers = centers / (
            np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
        )
        points = values / (
            np.linalg.norm(values, axis=1, keepdims=True) + 1e-8
        )
        for cluster_id in small_ids:
            indices = np.flatnonzero(labels == cluster_id)
            nearest = np.argmax(points[indices] @ centers.T, axis=1)
            labels[indices] = large_ids[nearest]
    else:
        for cluster_id in small_ids:
            indices = np.flatnonzero(labels == cluster_id)
            nearest = pairwise_distances_argmin(values[indices], centers)
            labels[indices] = large_ids[nearest]
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
    module5_config_override: Optional[Mapping[str, Any]] = None,
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
    if module5_config_override:
        _m5 = {
            **_m5,
            **dict(module5_config_override),
        }
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
    cluster_rep = str(_m5.get("cluster_rep", "legacy_features"))
    cluster_rep_decell = bool(_m5.get("cluster_rep_decell", False))
    clustering_method = str(
        _m5.get("clustering_method", "minibatch_kmeans")
    ).strip().lower().replace("-", "_")
    clustering_aliases = {
        "minibatch_kmeans": "minibatch_kmeans",
        "kmeans": "minibatch_kmeans",
        "legacy": "minibatch_kmeans",
        "consensus_leiden": "consensus_leiden",
        "stability_leiden": "consensus_leiden",
        "leiden": "consensus_leiden",
    }
    if clustering_method not in clustering_aliases:
        raise ValueError(
            "module5.clustering_method must be one of "
            "{'minibatch_kmeans', 'consensus_leiden'}"
        )
    clustering_method = clustering_aliases[clustering_method]
    large_scale_method = str(
        _m5.get("large_scale_method", "auto")
    ).strip().lower().replace("-", "_")
    if large_scale_method == "auto":
        effective_clustering_method = clustering_method
    elif large_scale_method in {
        "hierarchical_prototype",
        "hierarchical",
        "prototype",
    }:
        effective_clustering_method = "hierarchical_prototype"
    elif large_scale_method in {"consensus_leiden", "exact"}:
        effective_clustering_method = "consensus_leiden"
    elif large_scale_method == "minibatch_kmeans":
        effective_clustering_method = "minibatch_kmeans"
    else:
        raise ValueError(
            "module5.large_scale_method must be one of "
            "{'auto','minibatch_kmeans','consensus_leiden',"
            "'hierarchical_prototype'}"
        )
    min_domain_size = max(1, int(_m5.get("min_domain_size", 1)))
    random_seed = int(_m5.get("random_seed", 42))
    # More aggressive speed defaults for large cohorts; can still be overridden in params.yaml
    train_sample_per_block = int(_m5.get('train_sample_per_block', 300000))
    assign_chunk_size = int(_m5.get('assign_chunk_size', _m5.get('block_size', 50000) or 50000))
    mbk_batch_size = int(_m5.get('mbk_batch_size', 8192))
    mbk_max_iter = int(_m5.get('mbk_max_iter', 60 if str(speed_preset).lower() == 'ultra' else 80))
    # feature construction
    X_all, cell_ids, u_cells, cell_codes = _metadomain_feature_matrix(
        adata1,
        point_df,
        cluster_rep=cluster_rep,
        cluster_rep_decell=cluster_rep_decell,
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
    if effective_clustering_method == "hierarchical_prototype":
        from .hierarchical_metadomain import (
            run_hierarchical_prototype_clustering,
        )

        split_by_nucleus = bool(_m5.get("split_by_nucleus", True))
        if split_by_nucleus and "r_p50" in adata1.obs.columns:
            threshold = float(_m5.get("nucleus_threshold", 0.0))
            side_labels = np.where(
                adata1.obs["r_p50"].to_numpy(np.float32) < threshold,
                "N",
                "C",
            )
        else:
            side_labels = np.full(n_sub, "A", dtype=object)
        t0_hierarchical = _step_start(
            "cell-aligned local prototypes and global prototype clustering"
        )
        global_labels, metadata, diagnostic_table = (
            run_hierarchical_prototype_clustering(
                X_all,
                cell_ids,
                side_labels=side_labels,
                config=_m5,
            )
        )
        global_labels = _merge_small_metadomain_clusters(
            global_labels,
            X_all,
            min_domain_size,
            use_cosine=use_cosine,
        )
        unique = np.unique(global_labels)
        remap = {int(value): index for index, value in enumerate(unique)}
        dense = np.asarray(
            [remap[int(value)] for value in global_labels],
            dtype=np.int32,
        )
        _step_done("hierarchical prototype clustering", t0_hierarchical)
        adata1.obs["meta-domain"] = pd.Categorical(dense.astype(str))
        metadata = {
            **metadata,
            "cluster_rep": cluster_rep,
            "cluster_rep_decell": bool(cluster_rep_decell),
            "split_by_nucleus": bool(split_by_nucleus),
            "min_domain_size": int(min_domain_size),
            "random_seed": int(random_seed),
            "n_metadomains_after_small_cluster_merge": int(unique.size),
        }
        adata1.uns["module5_metadomain_clustering"] = metadata
        _logging.getLogger("cellscope").info(
            "Module 5 hierarchical prototypes: MetaSpots=%d "
            "local_prototypes=%d MetaDomains=%d blocks=%d",
            int(n_sub),
            int(metadata["n_local_prototypes"]),
            int(unique.size),
            int(metadata["n_blocks"]),
        )
        return adata1, diagnostic_table

    if effective_clustering_method == "consensus_leiden":
        from .consensus_clustering import run_stability_consensus_leiden

        split_by_nucleus = bool(_m5.get("split_by_nucleus", True))
        if split_by_nucleus and "r_p50" in adata1.obs.columns:
            threshold = float(_m5.get("nucleus_threshold", 0.0))
            is_nuclear = (
                adata1.obs["r_p50"].to_numpy(np.float32) < threshold
            )
            blocks = [
                ("N", np.flatnonzero(is_nuclear)),
                ("C", np.flatnonzero(~is_nuclear)),
            ]
        else:
            blocks = [("A", np.arange(n_sub, dtype=np.int64))]
        global_labels = np.empty(n_sub, dtype=np.int32)
        offset = 0
        consensus_by_block: Dict[str, Any] = {}
        diagnostic_rows: List[Dict[str, Any]] = []
        t0_consensus = _step_start("stability consensus Leiden")
        for prefix, indices in blocks:
            if indices.size == 0:
                continue
            local_labels, metadata = run_stability_consensus_leiden(
                X_all[indices],
                cell_ids[indices],
                config=_m5,
                n_jobs=_preferred_n_jobs(),
            )
            local_labels = _merge_small_metadomain_clusters(
                local_labels,
                X_all[indices],
                min_domain_size,
                use_cosine=use_cosine,
            )
            unique = np.unique(local_labels)
            remap = {value: index for index, value in enumerate(unique)}
            dense = np.asarray(
                [remap[value] for value in local_labels],
                dtype=np.int32,
            )
            global_labels[indices] = dense + offset
            offset += int(dense.max()) + 1
            consensus_by_block[prefix] = metadata
            candidate_resolution = np.asarray(
                metadata["candidate_resolution"], dtype=float
            )
            selected_resolution = float(metadata["selected_resolution"])
            selected_index = int(
                np.argmin(np.abs(candidate_resolution - selected_resolution))
            )
            diagnostic_rows.append(
                {
                    "block": prefix,
                    "n_metaspots": int(indices.size),
                    "selected_resolution": selected_resolution,
                    "selected_stability_ari": float(
                        np.asarray(
                            metadata["candidate_stability_ari"], dtype=float
                        )[selected_index]
                    ),
                    "selected_candidate_clusters": int(
                        np.asarray(
                            metadata["candidate_n_clusters"], dtype=int
                        )[selected_index]
                    ),
                    "final_clusters": int(unique.size),
                    "selection_reason": str(metadata["selection_reason"]),
                }
            )
        _step_done("stability consensus Leiden", t0_consensus)
        adata1.obs["meta-domain"] = pd.Categorical(
            global_labels.astype(str)
        )
        adata1.uns["module5_metadomain_clustering"] = {
            "clustering_method": "consensus_leiden",
            "cluster_rep": cluster_rep,
            "cluster_rep_decell": bool(cluster_rep_decell),
            "split_by_nucleus": bool(split_by_nucleus),
            "min_domain_size": int(min_domain_size),
            "random_seed": int(random_seed),
            "n_metaspots": int(n_sub),
            "n_metadomains": int(
                adata1.obs["meta-domain"].nunique()
            ),
            "consensus_by_block": consensus_by_block,
        }
        _logging.getLogger("cellscope").info(
            "Module 5 consensus Leiden: MetaSpots=%d MetaDomains=%d blocks=%s",
            int(n_sub),
            int(adata1.obs["meta-domain"].nunique()),
            [row["block"] for row in diagnostic_rows],
        )
        return adata1, pd.DataFrame(diagnostic_rows)

    # K estimation. "auto" is a sampled held-out distortion knee; the
    # historical count-only heuristic remains available as "legacy_auto".
    estimate_k_mode = str(_m5.get("estimate_k", "auto")).strip().lower()
    if estimate_k_mode == "fixed":
        K_total = int(_m5.get('fixed_k', 1024))
        k_selection: Dict[str, Any] = {
            "method": "fixed",
            "selection_reason": "configured_fixed_k",
            "selected_k": int(K_total),
            "knee_k": int(K_total),
            "candidate_k": np.asarray([K_total], dtype=np.int32),
        }
    elif estimate_k_mode in {"legacy_auto", "legacy"}:
        K_total = _choose_K_simple(
            n_sub=n_sub,
            n_cells=n_cells,
            target_sub_per_cluster=int(_m5.get('target_sub_per_cluster', 800)),
            target_cluster_per_cell=float(_m5.get('target_cluster_per_cell', 1.0)),
            K_min=int(_m5.get('K_min', 200)),
            K_max=int(_m5.get('K_max', 8000)),
        )
        k_selection = {
            "method": "legacy_auto",
            "selection_reason": "legacy_count_heuristic",
            "selected_k": int(K_total),
            "knee_k": int(K_total),
            "candidate_k": np.asarray([K_total], dtype=np.int32),
        }
    elif estimate_k_mode in {"auto", "sampled_knee"}:
        t0_auto_k = _step_start("sampled automatic K selection")
        K_total, k_selection = _choose_K_sampled_knee(
            X_all,
            cell_ids,
            use_cosine=use_cosine,
            config=_m5,
        )
        _step_done("sampled automatic K selection", t0_auto_k)
        _logging.getLogger("cellscope").info(
            "Module 5 automatic K: selected=%d knee=%d reason=%s "
            "candidates=%s",
            int(K_total),
            int(k_selection.get("knee_k", K_total)),
            str(k_selection.get("selection_reason", "")),
            np.asarray(
                k_selection.get("candidate_k", [K_total])
            ).tolist(),
        )
    else:
        raise ValueError(
            "module5.estimate_k must be one of "
            "{'auto', 'sampled_knee', 'legacy_auto', 'fixed'}, "
            f"got {estimate_k_mode!r}"
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
            seed=random_seed + (1 if prefix == 'C' else 0),
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
        y = _merge_small_metadomain_clusters(
            y,
            Xi,
            min_domain_size,
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
    adata1.uns["module5_metadomain_clustering"] = {
        "clustering_method": "minibatch_kmeans",
        "cluster_rep": cluster_rep,
        "cluster_rep_decell": bool(cluster_rep_decell),
        "split_by_nucleus": bool(split_by_nucleus),
        "min_domain_size": int(min_domain_size),
        "random_seed": int(random_seed),
        "n_metaspots": int(n_sub),
        "n_metadomains": int(adata1.obs["meta-domain"].nunique()),
        "k_selection": k_selection,
    }
    return adata1, pd.DataFrame(
        {
            "n_sub": [n_sub],
            "n_cells": [n_cells],
            "K_total": [K_total],
            "knee_K": [int(k_selection.get("knee_k", K_total))],
            "selection_method": [
                str(k_selection.get("method", estimate_k_mode))
            ],
            "selection_reason": [
                str(k_selection.get("selection_reason", ""))
            ],
            "final_K": [
                int(adata1.obs["meta-domain"].nunique())
            ],
        }
    )


def build_adata2_from_adata1_and_pointdf(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    show_internal_progress: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    module6_config_override: Optional[Mapping[str, Any]] = None,
    graph_unit_override: Optional[str] = None,
    pool_output_key_map: Optional[Mapping[str, str]] = None,
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
    adata2.obs["n_metaspots"] = row_sum.astype(np.int64)
    adata2.uns["matrix_semantics"] = {
        "X": "mean MetaSpot transcript-frequency profile; rows sum to approximately one",
        "layers_counts": "sum of raw MetaSpot gene counts per MetaDomain",
    }

    try:
        module6_config = (load_params_yaml() or {}).get("module6", {}) or {}
    except Exception:
        module6_config = {}
    if module6_config_override:
        module6_config = {
            **module6_config,
            **dict(module6_config_override),
        }
    pool_reps = module6_config.get("pool_reps", [])
    if isinstance(pool_reps, str):
        pool_reps = [
            value.strip()
            for value in pool_reps.split(",")
            if value.strip()
        ]
    elif not isinstance(pool_reps, (list, tuple)):
        pool_reps = []
    pool_normalize = bool(module6_config.get("pool_normalize", True))
    if graph_unit_override is None:
        graph_unit = str(
            ((load_params_yaml() or {}).get("module7", {}) or {}).get(
                "graph_unit",
                "metadomain",
            )
        ).strip().lower()
    else:
        graph_unit = str(graph_unit_override).strip().lower()

    pooled_metadata: Dict[str, Any] = {}
    group_mean = G.multiply(1.0 / row_sum[:, None]).tocsr()
    output_key_map = {
        METASPOT_DGI_KEY: METADOMAIN_DGI_KEY,
        METASPOT_DGI_SAGE_KEY: "X_dgi_sage_metadomain",
        "X_expr_metaspot": "X_expr_metadomain",
    }
    if pool_output_key_map:
        output_key_map.update(
            {
                str(source): str(target)
                for source, target in pool_output_key_map.items()
            }
        )
    compatibility_aliases = {
        METASPOT_DGI_KEY: "X_dgi_gcn",
        METASPOT_DGI_SAGE_KEY: "X_dgi_sage",
        "X_expr_metaspot": "X_expr",
    }
    for source_key in [str(value) for value in pool_reps]:
        if source_key not in adata1.obsm:
            if graph_unit == "metaspot":
                raise KeyError(
                    f"module6.pool_reps requested {source_key!r}, but it is "
                    "missing from adata1.obsm"
                )
            continue
        source = np.asarray(adata1.obsm[source_key], dtype=np.float32)
        if source.ndim != 2 or source.shape[0] != adata1.n_obs:
            raise ValueError(
                f"adata1.obsm[{source_key!r}] is not aligned to MetaSpots"
            )
        if not np.isfinite(source).all():
            raise ValueError(
                f"adata1.obsm[{source_key!r}] contains non-finite values"
            )
        if pool_normalize:
            source_norm = np.linalg.norm(source, axis=1, keepdims=True)
            source = source / np.maximum(source_norm, 1e-8)
        pooled = np.asarray(group_mean @ source, dtype=np.float32)
        if pool_normalize:
            pooled_norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.maximum(pooled_norm, 1e-8)
        output_key = output_key_map.get(
            source_key,
            f"{source_key}_metadomain",
        )
        adata2.obsm[output_key] = pooled
        alias = compatibility_aliases.get(source_key)
        if alias:
            adata2.obsm[alias] = pooled.copy()
        pooled_metadata[output_key] = {
            "source_key": source_key,
            "source_unit": "metaspot",
            "target_unit": "metadomain",
            "pooling": "equal_metaspot_mean",
            "l2_normalized": bool(pool_normalize),
            "compatibility_alias": alias or "",
        }
    if graph_unit == "metaspot" and METADOMAIN_DGI_KEY not in adata2.obsm:
        raise RuntimeError(
            "MetaSpot graph mode requires pooled MetaDomain DGI embedding "
            f"{METADOMAIN_DGI_KEY!r}"
        )
    if pooled_metadata:
        adata2.uns["embedding_pooling"] = {
            "schema_version": 1,
            "membership_checksum": _membership_checksum(
                adata1.obs_names,
                adata1.obs["meta-domain"].astype(str).to_numpy(),
            ),
            "representations": pooled_metadata,
        }
    _step_done("Aggregate metaspots -> cluster means", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 4) aggregate geometry stats
    t0 = _step_start("Aggregate geometry stats per cluster")
    geometry_columns = [
        column
        for column in (
            "r_mean",
            "r_std",
            "r_p50",
            "d_cell_norm_mean",
            "d_nuc_norm_mean",
            "elongation",
            "spread",
            "log_n",
        )
        if column in adata1.obs.columns
    ]
    if geometry_columns:
        aggregation = {
            column: (
                "median"
                if column in {"elongation", "spread"}
                else "mean"
            )
            for column in geometry_columns
        }
        agg = (
            adata1.obs[geometry_columns + ["meta-domain"]]
            .groupby("meta-domain", sort=False, observed=True)
            .agg(aggregation)
        )
        adata2.obs = adata2.obs.join(agg, how="left")
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


def _build_groupwise_corruption_plan(
    group_values: Sequence[Any],
) -> Dict[str, np.ndarray]:
    """Precompute indices for a random non-self permutation within groups.

    Missing group labels are treated as separate singleton groups so that they
    can never leak features between cells.
    """
    values = np.asarray(group_values, dtype=object).reshape(-1).copy()
    missing = pd.isna(values)
    for index in np.flatnonzero(missing):
        values[index] = f"__cellscope_missing_group_{int(index)}"
    codes, _ = pd.factorize(values, sort=False)
    codes = np.asarray(codes, dtype=np.int64)
    order = np.argsort(codes, kind="stable").astype(np.int64, copy=False)
    position = np.empty(order.size, dtype=np.int64)
    position[order] = np.arange(order.size, dtype=np.int64)
    n_groups = int(codes.max()) + 1 if codes.size else 0
    sizes = np.bincount(codes, minlength=n_groups).astype(np.int64)
    starts = np.zeros(n_groups, dtype=np.int64)
    if n_groups > 1:
        starts[1:] = np.cumsum(sizes[:-1], dtype=np.int64)
    return {
        "codes": codes,
        "order": order,
        "position": position,
        "starts": starts,
        "sizes": sizes,
        "missing_count": np.asarray([int(missing.sum())], dtype=np.int64),
    }


def _draw_groupwise_corruption_index(
    plan: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one cyclic, non-self feature permutation per non-singleton group."""
    codes = np.asarray(plan["codes"], dtype=np.int64)
    if codes.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.asarray(plan["order"], dtype=np.int64)
    position = np.asarray(plan["position"], dtype=np.int64)
    starts = np.asarray(plan["starts"], dtype=np.int64)
    sizes = np.asarray(plan["sizes"], dtype=np.int64)
    shifts = np.zeros(sizes.size, dtype=np.int64)
    for group_index, size in enumerate(sizes):
        if size > 1:
            shifts[group_index] = int(rng.integers(1, int(size)))
    within_position = position - starts[codes]
    corrupted_position = (
        starts[codes]
        + (within_position + shifts[codes]) % sizes[codes]
    )
    return order[corrupted_position]


def _sample_local_contrastive_pairs(
    adjacency: sp.spmatrix,
    group_values: Optional[Sequence[Any]],
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample positive graph edges and non-edge negatives for local DGI loss.

    When group labels are supplied, negatives are restricted to the source
    node's group. Pairs without an eligible within-group non-neighbor are
    skipped instead of silently crossing cell boundaries.
    """
    requested = max(0, int(n_samples))
    graph = adjacency.tocsr().astype(np.float32)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    upper = sp.triu(graph, k=1, format="coo")
    if group_values is None:
        codes = np.zeros(graph.shape[0], dtype=np.int64)
        members = [np.arange(graph.shape[0], dtype=np.int64)]
    else:
        plan = _build_groupwise_corruption_plan(group_values)
        codes = np.asarray(plan["codes"], dtype=np.int64)
        if codes.size != graph.shape[0]:
            raise ValueError(
                "Local contrastive group labels are not aligned to graph nodes"
            )
        same_group_edge = codes[upper.row] == codes[upper.col]
        upper = sp.coo_matrix(
            (
                upper.data[same_group_edge],
                (upper.row[same_group_edge], upper.col[same_group_edge]),
            ),
            shape=upper.shape,
        )
        members = [
            plan["order"][start:start + size]
            for start, size in zip(plan["starts"], plan["sizes"])
        ]
    if requested == 0 or upper.nnz == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()

    rng = np.random.default_rng(int(seed))
    take = min(requested, int(upper.nnz))
    selected = (
        np.arange(upper.nnz, dtype=np.int64)
        if take == upper.nnz
        else rng.choice(upper.nnz, size=take, replace=False)
    )
    source = np.asarray(upper.row[selected], dtype=np.int64)
    positive = np.asarray(upper.col[selected], dtype=np.int64)

    kept_source: List[int] = []
    kept_positive: List[int] = []
    kept_negative: List[int] = []
    for src, pos in zip(source, positive):
        candidates = members[int(codes[src])]
        if candidates.size <= 1:
            continue
        row_start, row_end = graph.indptr[src], graph.indptr[src + 1]
        neighbors = graph.indices[row_start:row_end]
        negative = -1
        # Sparse graphs normally succeed immediately; bounded random trials
        # avoid materializing a large candidate mask for every sampled edge.
        for _ in range(min(32, max(4, int(candidates.size)))):
            candidate = int(candidates[int(rng.integers(candidates.size))])
            if candidate == src:
                continue
            location = np.searchsorted(neighbors, candidate)
            if location >= neighbors.size or neighbors[location] != candidate:
                negative = candidate
                break
        if negative < 0:
            # Deterministic fallback makes small/dense test graphs exact.
            for candidate_value in candidates:
                candidate = int(candidate_value)
                if candidate == src:
                    continue
                location = np.searchsorted(neighbors, candidate)
                if location >= neighbors.size or neighbors[location] != candidate:
                    negative = candidate
                    break
        if negative < 0:
            continue
        kept_source.append(int(src))
        kept_positive.append(int(pos))
        kept_negative.append(negative)

    return (
        np.asarray(kept_source, dtype=np.int64),
        np.asarray(kept_positive, dtype=np.int64),
        np.asarray(kept_negative, dtype=np.int64),
    )


def _run_module8_cell_aware_dmgi(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    *,
    module8_config: Mapping[str, Any],
    feature_keys: Sequence[str],
    output_key: str,
    compatibility_alias: bool,
    broadcast_to_adata1: bool,
    out_dir: Optional[str],
    progress_callback: Optional[Callable[[str], None]],
    show_internal_progress: bool,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Train the MetaSpot-only CA-DMGI backbone."""
    if broadcast_to_adata1:
        raise ValueError(
            "module8.dgi_framework='cell_aware_dmgi' requires native "
            "MetaSpot execution (module8.dgi_unit='metaspot')"
        )
    expression_graph_key = str(
        module8_config.get(
            "dmgi_expression_graph_key",
            "metaspot_expr_connectivities",
        )
    ).strip()
    spatial_graph_key = str(
        module8_config.get(
            "dmgi_spatial_graph_key",
            "metaspot_spatial_connectivities",
        )
    ).strip()
    missing_graphs = [
        key
        for key in (expression_graph_key, spatial_graph_key)
        if key not in adata2.obsp
    ]
    if missing_graphs:
        raise KeyError(f"Missing CA-DMGI relation graphs: {missing_graphs}")
    if "cell_id" not in adata2.obs.columns:
        raise KeyError("CA-DMGI requires adata.obs['cell_id']")
    missing_features = [
        key for key in feature_keys if key not in adata2.obsm
    ]
    if missing_features:
        raise KeyError(f"Missing CA-DMGI node features: {missing_features}")
    feature_arrays = [
        np.asarray(adata2.obsm[key], dtype=np.float32)
        for key in feature_keys
    ]
    if any(array.shape[0] != adata2.n_obs for array in feature_arrays):
        raise ValueError("CA-DMGI node features are not aligned to observations")
    features = np.concatenate(feature_arrays, axis=1).astype(
        np.float32,
        copy=False,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if progress_callback:
        progress_callback(
            "Module 8 · CA-DMGI relation-specific expression/spatial training"
        )

    requested_training_mode = str(
        module8_config.get("training_mode", "auto")
    ).strip().lower().replace("-", "_")
    execution_plan = adata2.uns.get("execution_plan", {})
    if requested_training_mode == "auto":
        effective_training_mode = str(
            execution_plan.get("dgi_training_mode", "full_batch")
        ).strip().lower().replace("-", "_")
    else:
        effective_training_mode = requested_training_mode
    if effective_training_mode not in {"full_batch", "neighbor_sampled"}:
        raise ValueError(
            "module8.training_mode must resolve to 'full_batch' or "
            "'neighbor_sampled'"
        )
    effective_module8_config = dict(module8_config)
    effective_module8_config["training_mode"] = effective_training_mode
    if effective_training_mode == "neighbor_sampled":
        effective_module8_config["dgi_epochs"] = int(
            module8_config.get(
                "sampled_dgi_epochs",
                module8_config.get("dgi_epochs", 100),
            )
        )
        if out_dir:
            effective_module8_config["sampled_checkpoint_dir"] = os.path.join(
                _intermediate_dir(out_dir), "module8_sampled_checkpoints"
            )
        from .sampled_multiplex_dgi import (
            train_sampled_cell_aware_dmgi as train_cell_aware_dmgi,
        )
    else:
        from .multiplex_dgi import train_cell_aware_dmgi

    def _training_progress(epoch: int, epochs: int, loss: float) -> None:
        _logging.getLogger("cellscope").info(
            "[CA-DMGI] epoch %d/%d loss=%.4f",
            epoch,
            epochs,
            loss,
        )

    result = train_cell_aware_dmgi(
        features,
        adata2.obsp[expression_graph_key],
        adata2.obsp[spatial_graph_key],
        adata2.obs["cell_id"].astype(str).to_numpy(),
        config=effective_module8_config,
        progress_callback=_training_progress,
    )
    consensus = np.asarray(result["embedding"], dtype=np.float32)
    expression = np.asarray(
        result["expression_embedding"], dtype=np.float32
    )
    spatial = np.asarray(result["spatial_embedding"], dtype=np.float32)
    expression_gate = np.asarray(
        result["expression_gate"], dtype=np.float32
    )
    adata2.obsm[output_key] = consensus
    adata2.obsm[METASPOT_DMGI_EXPRESSION_KEY] = expression
    adata2.obsm[METASPOT_DMGI_SPATIAL_KEY] = spatial
    adata2.obs = adata2.obs.copy()
    adata2.obs["dmgi_expression_gate"] = expression_gate
    adata2.obs["dmgi_spatial_gate"] = 1.0 - expression_gate
    if compatibility_alias and output_key != "X_dgi_gcn":
        adata2.obsm["X_dgi_gcn"] = consensus.copy()

    history = result["history"]
    metrics = result["metrics"]
    runtime = result["runtime"]
    provenance = dict(adata2.uns.get("dgi_embeddings", {}))
    provenance[output_key] = {
        "schema_version": int(METASPOT_DGI_SCHEMA_VERSION),
        "framework": "cell_aware_dmgi",
        "training_mode_requested": requested_training_mode,
        "training_mode": effective_training_mode,
        "objective": "relation_specific_local_infomax",
        "node_unit": "metaspot",
        "node_id_checksum": _obs_names_checksum(adata2.obs_names),
        "graph_key": "multiplex",
        "expression_graph_key": expression_graph_key,
        "spatial_graph_key": spatial_graph_key,
        "feature_keys": [str(key) for key in feature_keys],
        "n_nodes": int(adata2.n_obs),
        "n_features": int(features.shape[1]),
        "embedding_dim": int(consensus.shape[1]),
        "expression_embedding_key": METASPOT_DMGI_EXPRESSION_KEY,
        "spatial_embedding_key": METASPOT_DMGI_SPATIAL_KEY,
        "gate_obs_key": "dmgi_expression_gate",
        "encoder": str(module8_config.get("dgi_encoder", "residual")),
        "message_passing_layers": int(
            module8_config.get("dgi_layers", 2)
        ),
        "hidden_dim": int(module8_config.get("dgi_hidden", 64)),
        "dropout": float(module8_config.get("dgi_dropout", 0.1)),
        "residual_alpha": float(
            module8_config.get("dgi_residual_alpha", 0.1)
        ),
        "layer_norm": bool(module8_config.get("dgi_layer_norm", True)),
        "corruption_mode_effective": "same_cell",
        "negative_semantics": "corrupted_features_not_real_nodes",
        "expression_loss_weight": float(
            module8_config.get("dmgi_expression_weight", 1.0)
        ),
        "spatial_loss_weight": float(
            module8_config.get("dmgi_spatial_weight", 1.0)
        ),
        "consensus_loss_weight": float(
            module8_config.get("dmgi_consensus_weight", 0.1)
        ),
        "gate_floor": float(module8_config.get("dmgi_gate_floor", 0.1)),
        "epochs_requested": int(
            effective_module8_config.get("dgi_epochs", 100)
        ),
        "epochs": int(metrics["epochs_run"]),
        "best_epoch": int(metrics["best_epoch"]),
        "early_stopped": bool(metrics["early_stopped"]),
        "learning_rate": float(module8_config.get("dgi_lr", 1e-3)),
        "weight_decay": float(
            module8_config.get("dgi_weight_decay", 1e-4)
        ),
        "gradient_clip": float(
            module8_config.get("dgi_clip_grad", 1.0)
        ),
        "random_seed": int(module8_config.get("random_seed", 42)),
        "training_loss": history["loss"],
        "validation_loss": history.get("validation_loss", []),
        "training_expression_infomax_loss": history[
            "expression_infomax_loss"
        ],
        "training_spatial_infomax_loss": history[
            "spatial_infomax_loss"
        ],
        "training_consensus_loss": history["consensus_loss"],
        "training_loss_ema": history["loss_ema"],
        "training_expression_gate_mean": history[
            "expression_gate_mean"
        ],
        "final_training_loss": float(history["loss"][-1]),
        "final_expression_infomax_loss": float(
            history["expression_infomax_loss"][-1]
        ),
        "final_spatial_infomax_loss": float(
            history["spatial_infomax_loss"][-1]
        ),
        "final_consensus_loss": float(history["consensus_loss"][-1]),
        **{str(key): value for key, value in metrics.items()},
        **{str(key): value for key, value in runtime.items()},
        "compatibility_alias": (
            "X_dgi_gcn"
            if compatibility_alias and output_key != "X_dgi_gcn"
            else ""
        ),
    }
    adata2.uns["dgi_embeddings"] = provenance

    training_history = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["loss"]) + 1),
            "total_loss": history["loss"],
            "validation_loss": history.get(
                "validation_loss",
                np.full(len(history["loss"]), np.nan, dtype=np.float32),
            ),
            "expression_infomax_loss": history[
                "expression_infomax_loss"
            ],
            "spatial_infomax_loss": history["spatial_infomax_loss"],
            "consensus_loss": history["consensus_loss"],
            "total_loss_ema": history["loss_ema"],
            "mean_expression_gate": history["expression_gate_mean"],
        }
    )
    summary_fields = {
        "framework": "cell_aware_dmgi",
        "training_mode_requested": requested_training_mode,
        "training_mode": effective_training_mode,
        "objective": "relation_specific_local_infomax",
        "negative_semantics": "corrupted_features_not_real_nodes",
        "n_nodes": int(adata2.n_obs),
        "n_features": int(features.shape[1]),
        "embedding_dim": int(consensus.shape[1]),
        "epochs_requested": int(
            effective_module8_config.get("dgi_epochs", 100)
        ),
        "epochs_run": int(metrics["epochs_run"]),
        "best_epoch": int(metrics["best_epoch"]),
        "early_stopped": bool(metrics["early_stopped"]),
        "device": str(runtime["device"]),
        "device_requested": str(runtime["device_requested"]),
        "cuda_device_name": str(runtime["cuda_device_name"]),
        "cpu_threads": int(runtime["cpu_threads"]),
        "sparse_layout": str(runtime["sparse_layout"]),
        "sparse_index_dtype": str(runtime["sparse_index_dtype"]),
        "fused_corruption_pass": bool(
            runtime["fused_corruption_pass"]
        ),
        "tf32_enabled": bool(runtime["tf32_enabled"]),
        "expression_edges_nnz": int(runtime["expression_edges_nnz"]),
        "spatial_edges_nnz": int(runtime["spatial_edges_nnz"]),
        "corruption_group_count": int(runtime["corruption_group_count"]),
        "final_expression_gate_mean": float(
            metrics["final_expression_gate_mean"]
        ),
        "final_expression_gate_std": float(
            metrics["final_expression_gate_std"]
        ),
        "final_relation_cosine": float(metrics["final_relation_cosine"]),
        "final_expression_context_cosine": float(
            metrics["final_expression_context_cosine"]
        ),
        "final_spatial_context_cosine": float(
            metrics["final_spatial_context_cosine"]
        ),
    }
    try:
        _save_dataframe(
            training_history,
            out_dir or os.getcwd(),
            "module8_ca_dmgi_training_history",
            diagnostic=True,
        )
        _save_dataframe(
            pd.DataFrame([summary_fields]),
            out_dir or os.getcwd(),
            "module8_ca_dmgi_summary",
            diagnostic=True,
        )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "CA-DMGI training audit tables skipped: %s",
            exc,
        )

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        epochs_axis = np.arange(1, len(history["loss"]) + 1)
        axes[0].plot(
            epochs_axis,
            history["expression_infomax_loss"],
            label="Expression-relation Infomax",
        )
        axes[0].plot(
            epochs_axis,
            history["spatial_infomax_loss"],
            label="Within-cell spatial Infomax",
        )
        axes[0].plot(
            epochs_axis,
            history["loss_ema"],
            linestyle="--",
            label="Total loss EMA",
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Relation-specific training objectives")
        axes[0].legend(frameon=False)
        axes[1].plot(
            epochs_axis,
            history["expression_gate_mean"],
            color="#4C78A8",
        )
        axes[1].axhline(0.5, color="#999999", linestyle="--")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean expression-relation gate")
        axes[1].set_title("Learned multiplex consensus gate")
        fig.suptitle(
            "Cell-Aware Deep Multiplex Graph Infomax (CA-DMGI) training",
            fontweight="bold",
        )
        fig.tight_layout()
        _save_fig(
            fig,
            out_dir or os.getcwd(),
            "module8_ca_dmgi_training",
        )
        plt.close(fig)
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "CA-DMGI training plot skipped: %s", exc
        )
    return adata1, adata2


def run_module8_dgi(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    show_internal_progress: bool = False,
    dgi_epochs: int = 100,
    enable_dgi_sage: bool = False,
    out_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    *,
    graph_key: str = "A_norm",
    feature_keys: Optional[Sequence[str]] = None,
    output_key: str = "X_dgi_gcn",
    sage_output_key: str = "X_dgi_sage",
    broadcast_to_adata1: bool = True,
    compatibility_alias: bool = False,
) -> Tuple[ad.AnnData, ad.AnnData]:
    import numpy as np
    import torch
    if feature_keys is None:
        feature_keys = ("X_expr_scaled", "X_geom_scaled")

    # Load DGI hyperparameters from config. The residual encoder separates
    # representation depth from graph-propagation depth: two message-passing
    # layers remain the default, while input/output projections provide the
    # additional nonlinear capacity without forcing a wider graph receptive
    # field.
    dgi_encoder = "residual"
    dgi_layers = 2
    dgi_hidden = 64
    dgi_out = 32
    dgi_dropout = 0.1
    dgi_residual_alpha = 0.1
    dgi_layer_norm = True
    dgi_framework = "standard_dgi"
    dgi_corruption_mode = "auto"
    dgi_local_loss_weight = 0.2
    dgi_local_temperature = 0.2
    dgi_local_samples = 65536
    dgi_local_graph_key = "auto"
    dgi_cpu_backend = "auto"
    dgi_lr = 1e-3
    dgi_wd = 1e-4
    dgi_clip = 1.0
    dgi_device_requested = "auto"
    dgi_cpu_threads = 0
    dgi_tf32 = True
    dgi_amp_requested = True
    dgi_seed = 42
    _m8: Mapping[str, Any] = {}
    try:
        _cfg = load_params_yaml() or {}
        _m8 = _cfg.get('module8', {}) if _cfg else {}
        if not isinstance(_m8, Mapping):
            raise TypeError("module8 must be a mapping")
        dgi_epochs = int(_m8.get('dgi_epochs', dgi_epochs))
        enable_dgi_sage = bool(_m8.get('enable_dgi_sage', enable_dgi_sage))
        dgi_encoder = str(_m8.get("dgi_encoder", dgi_encoder)).strip().lower()
        dgi_layers = int(_m8.get("dgi_layers", dgi_layers))
        dgi_hidden = int(_m8.get('dgi_hidden', dgi_hidden))
        dgi_out = int(_m8.get('dgi_out', dgi_out))
        dgi_dropout = float(_m8.get("dgi_dropout", dgi_dropout))
        dgi_residual_alpha = float(
            _m8.get("dgi_residual_alpha", dgi_residual_alpha)
        )
        dgi_layer_norm = bool(_m8.get("dgi_layer_norm", dgi_layer_norm))
        dgi_framework = str(
            _m8.get("dgi_framework", dgi_framework)
        ).strip().lower()
        dgi_corruption_mode = str(
            _m8.get("dgi_corruption_mode", dgi_corruption_mode)
        ).strip().lower()
        dgi_local_loss_weight = float(
            _m8.get("dgi_local_loss_weight", dgi_local_loss_weight)
        )
        dgi_local_temperature = float(
            _m8.get("dgi_local_temperature", dgi_local_temperature)
        )
        dgi_local_samples = int(
            _m8.get("dgi_local_samples", dgi_local_samples)
        )
        dgi_local_graph_key = str(
            _m8.get("dgi_local_graph_key", dgi_local_graph_key)
        ).strip()
        dgi_cpu_backend = str(
            _m8.get("dgi_cpu_backend", dgi_cpu_backend)
        ).strip().lower()
        dgi_lr = float(_m8.get('dgi_lr', dgi_lr))
        dgi_wd = float(_m8.get('dgi_weight_decay', dgi_wd))
        dgi_clip = float(_m8.get('dgi_clip_grad', dgi_clip))
        dgi_device_requested = str(
            _m8.get("dgi_device", dgi_device_requested)
        ).strip().lower()
        dgi_cpu_threads = int(
            _m8.get("dgi_cpu_threads", dgi_cpu_threads)
        )
        dgi_tf32 = bool(_m8.get("dgi_tf32", dgi_tf32))
        dgi_amp_requested = bool(_m8.get('amp', dgi_amp_requested))
        dgi_seed = int(_m8.get("random_seed", dgi_seed))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid module8 DGI configuration: {exc}") from exc

    encoder_aliases = {
        "plain": "plain",
        "gcn": "plain",
        "residual": "residual",
        "residual_gcn": "residual",
        "residual-gcn": "residual",
    }
    if dgi_encoder not in encoder_aliases:
        raise ValueError(
            "module8.dgi_encoder must be one of "
            "{'plain', 'residual'}, "
            f"got {dgi_encoder!r}"
        )
    dgi_encoder = encoder_aliases[dgi_encoder]
    framework_aliases = {
        "standard_dgi": "standard_dgi",
        "standard-dgi": "standard_dgi",
        "dgi": "standard_dgi",
        "cell_aware_dmgi": "cell_aware_dmgi",
        "cell-aware-dmgi": "cell_aware_dmgi",
        "ca_dmgi": "cell_aware_dmgi",
        "ca-dmgi": "cell_aware_dmgi",
        "ch_dmgi": "cell_aware_dmgi",  # legacy compatibility only
        "ch-dmgi": "cell_aware_dmgi",  # legacy compatibility only
        "dmgi": "cell_aware_dmgi",
    }
    if dgi_framework not in framework_aliases:
        raise ValueError(
            "module8.dgi_framework must be one of "
            "{'standard_dgi', 'cell_aware_dmgi'}"
        )
    dgi_framework = framework_aliases[dgi_framework]
    corruption_aliases = {
        "auto": "auto",
        "global": "global",
        "global_shuffle": "global",
        "global-shuffle": "global",
        "same_cell": "same_cell",
        "same-cell": "same_cell",
        "within_cell": "same_cell",
        "within-cell": "same_cell",
    }
    if dgi_corruption_mode not in corruption_aliases:
        raise ValueError(
            "module8.dgi_corruption_mode must be one of "
            "{'auto', 'global', 'same_cell'}, "
            f"got {dgi_corruption_mode!r}"
        )
    dgi_corruption_mode = corruption_aliases[dgi_corruption_mode]
    if dgi_epochs < 1:
        raise ValueError("module8.dgi_epochs must be >= 1")
    if dgi_layers < 1:
        raise ValueError("module8.dgi_layers must be >= 1")
    if dgi_hidden < 1 or dgi_out < 1:
        raise ValueError("module8.dgi_hidden and module8.dgi_out must be >= 1")
    if not 0.0 <= dgi_dropout < 1.0:
        raise ValueError("module8.dgi_dropout must be in [0, 1)")
    if not 0.0 <= dgi_residual_alpha <= 1.0:
        raise ValueError("module8.dgi_residual_alpha must be in [0, 1]")
    if dgi_lr <= 0.0 or dgi_wd < 0.0:
        raise ValueError(
            "module8.dgi_lr must be > 0 and dgi_weight_decay must be >= 0"
        )
    if dgi_clip is not None and dgi_clip <= 0.0:
        raise ValueError("module8.dgi_clip_grad must be > 0")
    if dgi_cpu_threads < 0:
        raise ValueError("module8.dgi_cpu_threads must be >= 0")
    if dgi_local_loss_weight < 0.0:
        raise ValueError("module8.dgi_local_loss_weight must be >= 0")
    if dgi_local_temperature <= 0.0:
        raise ValueError("module8.dgi_local_temperature must be > 0")
    if dgi_local_samples < 0:
        raise ValueError("module8.dgi_local_samples must be >= 0")
    if not dgi_local_graph_key:
        raise ValueError("module8.dgi_local_graph_key must not be empty")
    cpu_backend_aliases = {
        "auto": "auto",
        "pyg": "pyg",
        "coo": "pyg",
        "torch_csr": "torch_csr",
        "torch-csr": "torch_csr",
        "csr": "torch_csr",
    }
    if dgi_cpu_backend not in cpu_backend_aliases:
        raise ValueError(
            "module8.dgi_cpu_backend must be one of "
            "{'auto', 'pyg', 'torch_csr'}"
        )
    dgi_cpu_backend = cpu_backend_aliases[dgi_cpu_backend]
    if dgi_framework == "cell_aware_dmgi":
        effective_dmgi_config = {
            **dict(_m8),
            "dgi_framework": dgi_framework,
            "dgi_epochs": int(dgi_epochs),
            "dgi_encoder": dgi_encoder,
            "dgi_layers": int(dgi_layers),
            "dgi_hidden": int(dgi_hidden),
            "dgi_out": int(dgi_out),
            "dgi_dropout": float(dgi_dropout),
            "dgi_residual_alpha": float(dgi_residual_alpha),
            "dgi_layer_norm": bool(dgi_layer_norm),
            "dgi_lr": float(dgi_lr),
            "dgi_weight_decay": float(dgi_wd),
            "dgi_clip_grad": float(dgi_clip),
            "random_seed": int(dgi_seed),
        }
        return _run_module8_cell_aware_dmgi(
            adata1,
            adata2,
            module8_config=effective_dmgi_config,
            feature_keys=tuple(str(key) for key in feature_keys),
            output_key=output_key,
            compatibility_alias=compatibility_alias,
            broadcast_to_adata1=broadcast_to_adata1,
            out_dir=out_dir,
            progress_callback=progress_callback,
            show_internal_progress=show_internal_progress,
        )

    # Standard DGI uses the fused graph. Delay this comparatively expensive
    # PyG conversion and feature copy until after the CA-DMGI early return;
    # CA-DMGI consumes its two native sparse relation graphs directly.
    from .multiplex_dgi import resolve_dgi_device
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv
    from torch_geometric.nn.models import DeepGraphInfomax
    from torch_geometric.utils import from_scipy_sparse_matrix

    device = resolve_dgi_device(torch, dgi_device_requested)
    if device.type == "cpu":
        available_cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else int(os.cpu_count() or 1)
        )
        torch.set_num_threads(
            dgi_cpu_threads
            if dgi_cpu_threads > 0
            else min(32, max(1, available_cpus))
        )
    elif hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = dgi_tf32
    dgi_amp = dgi_amp_requested and device.type == "cuda"
    if graph_key not in adata2.obsp:
        raise KeyError(f"Missing DGI graph adata.obsp[{graph_key!r}]")
    A_norm = adata2.obsp[graph_key].tocsr()
    edge_index, edge_weight = from_scipy_sparse_matrix(A_norm)
    edge_weight = torch.clamp(
        torch.nan_to_num(
            edge_weight, nan=0.0, posinf=0.0, neginf=0.0
        ),
        min=1e-8,
    )
    missing_features = [
        key for key in feature_keys if key not in adata2.obsm
    ]
    if missing_features:
        raise KeyError(f"Missing DGI node features: {missing_features}")
    feature_arrays = [
        np.asarray(adata2.obsm[key], dtype=np.float32)
        for key in feature_keys
    ]
    if any(array.shape[0] != adata2.n_obs for array in feature_arrays):
        raise ValueError("DGI node features are not aligned to observations")
    X_all = np.concatenate(feature_arrays, axis=1).astype(np.float32)
    x = torch.from_numpy(np.nan_to_num(X_all, nan=0.0)).to(
        dtype=torch.float32,
        device=device,
    )
    x = F.normalize(x, p=2, dim=1)
    ei = edge_index.to(device, non_blocking=True)
    ew = edge_weight.to(device, non_blocking=True).to(torch.float32)
    effective_gcn_backend = (
        "torch_csr"
        if device.type == "cpu" and dgi_cpu_backend in {"auto", "torch_csr"}
        else "pyg"
    )
    np.random.seed(dgi_seed)
    torch.manual_seed(dgi_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(dgi_seed)

    cell_group_key = "cell_id"
    has_cell_groups = cell_group_key in adata2.obs.columns
    if dgi_corruption_mode == "auto":
        effective_corruption_mode = (
            "same_cell"
            if not broadcast_to_adata1 and has_cell_groups
            else "global"
        )
    else:
        effective_corruption_mode = dgi_corruption_mode
    if effective_corruption_mode == "same_cell" and not has_cell_groups:
        raise ValueError(
            "module8.dgi_corruption_mode='same_cell' requires "
            "adata.obs['cell_id'] aligned to DGI graph nodes"
        )

    corruption_plan = None
    corruption_tensors = None
    if effective_corruption_mode == "same_cell":
        corruption_plan = _build_groupwise_corruption_plan(
            adata2.obs[cell_group_key].to_numpy()
        )
        corruption_tensors = {
            key: torch.as_tensor(
                corruption_plan[key], dtype=torch.long, device=device
            )
            for key in ("codes", "order", "position", "starts", "sizes")
        }

    effective_local_graph_key = ""
    local_pair_arrays = (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    if dgi_local_loss_weight > 0.0 and dgi_local_samples > 0:
        if dgi_local_graph_key.lower() == "auto":
            effective_local_graph_key = (
                "metaspot_spatial_connectivities"
                if "metaspot_spatial_connectivities" in adata2.obsp
                else graph_key
            )
        else:
            effective_local_graph_key = dgi_local_graph_key
        if effective_local_graph_key not in adata2.obsp:
            raise KeyError(
                "Missing local DGI graph "
                f"adata.obsp[{effective_local_graph_key!r}]"
            )
        local_groups = (
            adata2.obs[cell_group_key].to_numpy()
            if has_cell_groups
            else None
        )
        local_pair_arrays = _sample_local_contrastive_pairs(
            adata2.obsp[effective_local_graph_key],
            local_groups,
            n_samples=dgi_local_samples,
            seed=dgi_seed + 17,
        )
    local_pair_tensors = tuple(
        torch.as_tensor(values, dtype=torch.long, device=device)
        for values in local_pair_arrays
    )

    cpu_gcn_adjacency = None
    if effective_gcn_backend == "torch_csr":
        # GCNConv with ``normalize=False`` computes A^T @ (XW) under PyG's
        # source-to-target edge convention.  The graph is already normalized,
        # so a single CSR SpMM is mathematically equivalent and substantially
        # faster than COO scatter on CPU for million-edge MetaSpot graphs.
        cpu_graph = A_norm.transpose().tocsr().astype(
            np.float32,
            copy=True,
        )
        cpu_graph.data = np.clip(
            np.nan_to_num(
                cpu_graph.data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            1e-8,
            None,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            cpu_gcn_adjacency = torch.sparse_csr_tensor(
                torch.from_numpy(cpu_graph.indptr.astype(np.int64, copy=False)),
                torch.from_numpy(cpu_graph.indices.astype(np.int64, copy=False)),
                torch.from_numpy(cpu_graph.data),
                size=cpu_graph.shape,
                dtype=torch.float32,
                device=device,
            )
        _logging.getLogger("cellscope").info(
            "Module 8 CPU GCN backend | requested=%s | effective=%s | "
            "nodes=%d | edges=%d",
            dgi_cpu_backend,
            effective_gcn_backend,
            int(A_norm.shape[0]),
            int(A_norm.nnz),
        )

    class _TorchCSRGCNConv(nn.Module):
        """GCN linear transform plus pre-normalized CSR aggregation."""

        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
            self.bias = nn.Parameter(torch.empty(output_dim))
            # ``GCNConv`` constructs and initializes its internal Linear once,
            # then calls ``GCNConv.reset_parameters`` and initializes it a
            # second time. Mirror that RNG consumption so the optimized CSR
            # backend preserves seeded initialization as well as graph math.
            self.reset_parameters()
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

        def forward(self, values, edge_index=None, edge_weight=None):
            if cpu_gcn_adjacency is None:
                raise RuntimeError("CPU CSR GCN adjacency is unavailable")
            transformed = F.linear(values, self.weight, bias=None)
            return torch.sparse.mm(
                cpu_gcn_adjacency,
                transformed,
            ) + self.bias

    class Encoder(nn.Module):
        """Configurable plain or initial-residual graph encoder for DGI."""

        def __init__(
            self,
            in_channels,
            hidden=64,
            out_channels=32,
            conv_type="gcn",
            architecture="residual",
            num_layers=2,
            dropout=0.1,
            residual_alpha=0.1,
            layer_norm=True,
        ):
            super().__init__()
            Conv = {"gcn": GCNConv, "sage": SAGEConv}[conv_type]
            self.architecture = str(architecture)
            self.num_layers = int(num_layers)
            self.conv_type = conv_type
            self.residual_alpha = float(residual_alpha)
            self.drop = nn.Dropout(float(dropout))

            def make_conv(input_dim, output_dim):
                if conv_type == "gcn":
                    if effective_gcn_backend == "torch_csr":
                        return _TorchCSRGCNConv(input_dim, output_dim)
                    return Conv(
                        input_dim,
                        output_dim,
                        normalize=False,
                        improved=True,
                        add_self_loops=False,
                    )
                return Conv(input_dim, output_dim)

            if self.architecture == "plain":
                dimensions = (
                    [int(in_channels)]
                    + [int(hidden)] * max(0, self.num_layers - 1)
                    + [int(out_channels)]
                )
                self.convs = nn.ModuleList(
                    make_conv(dimensions[index], dimensions[index + 1])
                    for index in range(self.num_layers)
                )
                self.activations = nn.ModuleList(
                    nn.PReLU() for _ in range(max(0, self.num_layers - 1))
                )
                self.input_projection = None
                self.output_projection = None
                self.norms = nn.ModuleList()
                self.input_activation = None
            else:
                self.input_projection = nn.Linear(in_channels, hidden)
                self.input_activation = nn.PReLU()
                self.convs = nn.ModuleList(
                    make_conv(hidden, hidden) for _ in range(self.num_layers)
                )
                self.activations = nn.ModuleList(
                    nn.PReLU() for _ in range(self.num_layers)
                )
                self.norms = nn.ModuleList(
                    nn.LayerNorm(hidden) if layer_norm else nn.Identity()
                    for _ in range(self.num_layers)
                )
                self.output_projection = nn.Linear(hidden, out_channels)

        def _convolve(self, conv, values, edge_index, edge_weight):
            if self.conv_type == "gcn":
                return conv(values, edge_index, edge_weight=edge_weight)
            return conv(values, edge_index)

        def forward(self, x, edge_index, edge_weight=None):
            if self.architecture == "plain":
                values = x
                for layer_index, conv in enumerate(self.convs):
                    values = self._convolve(
                        conv,
                        values,
                        edge_index,
                        edge_weight,
                    )
                    if layer_index < self.num_layers - 1:
                        values = self.activations[layer_index](values)
                        values = self.drop(values)
                return values

            initial = self.input_activation(self.input_projection(x))
            values = self.drop(initial)
            for layer_index, conv in enumerate(self.convs):
                previous = values
                message = self._convolve(
                    conv,
                    values,
                    edge_index,
                    edge_weight,
                )
                message = self.activations[layer_index](message)
                message = self.drop(message)
                message = (
                    (1.0 - self.residual_alpha) * message
                    + self.residual_alpha * initial
                )
                values = self.norms[layer_index](previous + message)
            return self.output_projection(values)

        @torch.no_grad()
        def get_embeddings(self, x, edge_index, edge_weight=None):
            self.eval()
            return self.forward(x, edge_index, edge_weight)

    def build_dgi(conv_type="gcn", hidden=dgi_hidden, out_channels=dgi_out):
        enc = Encoder(
            in_channels=x.size(-1),
            hidden=hidden,
            out_channels=out_channels,
            conv_type=conv_type,
            architecture=dgi_encoder,
            num_layers=dgi_layers,
            dropout=dgi_dropout,
            residual_alpha=dgi_residual_alpha,
            layer_norm=dgi_layer_norm,
        ).to(device)
        def corruption(x_in, edge_index, edge_weight=None):
            if effective_corruption_mode == "global":
                permutation = torch.randperm(
                    x_in.size(0), device=x_in.device
                )
            else:
                assert corruption_tensors is not None
                codes = corruption_tensors["codes"]
                order = corruption_tensors["order"]
                position = corruption_tensors["position"]
                starts = corruption_tensors["starts"]
                sizes = corruption_tensors["sizes"]
                shifts = torch.zeros_like(sizes)
                nonsingleton = sizes > 1
                shifts[nonsingleton] = (
                    torch.floor(
                        torch.rand(
                            int(nonsingleton.sum().item()),
                            device=x_in.device,
                        )
                        * (sizes[nonsingleton] - 1).to(torch.float32)
                    ).to(torch.long)
                    + 1
                )
                within_position = position - starts[codes]
                corrupted_position = (
                    starts[codes]
                    + (within_position + shifts[codes]) % sizes[codes]
                )
                permutation = order[corrupted_position]
            return x_in[permutation], edge_index, edge_weight
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
        losses_hist, dgi_losses_hist, local_losses_hist, ema_hist = [], [], [], []
        ema = None
        local_source, local_positive, local_negative = local_pair_tensors

        def _objective():
            pos_z, neg_z, summary = model(x, ei, ew)
            dgi_loss = model.loss(pos_z, neg_z, summary)
            if (
                dgi_local_loss_weight > 0.0
                and local_source.numel() > 0
            ):
                normalized = F.normalize(pos_z, p=2, dim=1)
                positive_similarity = (
                    normalized[local_source]
                    * normalized[local_positive]
                ).sum(dim=1) / dgi_local_temperature
                negative_similarity = (
                    normalized[local_source]
                    * normalized[local_negative]
                ).sum(dim=1) / dgi_local_temperature
                local_loss = F.softplus(
                    negative_similarity - positive_similarity
                ).mean()
            else:
                local_loss = pos_z.sum() * 0.0
            total_loss = dgi_loss + dgi_local_loss_weight * local_loss
            return total_loss, dgi_loss, local_loss
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
                    loss, dgi_loss_value, local_loss_value = _objective()
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
                loss, dgi_loss_value, local_loss_value = _objective()
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
            dgi_losses_hist.append(float(dgi_loss_value))
            local_losses_hist.append(float(local_loss_value))
            ema = l if ema is None else (0.9 * ema + 0.1 * l)
            ema_hist.append(ema)
        # save training curves
        try:
            import matplotlib.pyplot as plt
            import numpy as _np
            fig = plt.figure(figsize=(7,4))
            plt.plot(_np.array(losses_hist), label=f"{tag}-total", linewidth=1.2)
            plt.plot(_np.array(dgi_losses_hist), label=f"{tag}-DGI", linewidth=1.0)
            if dgi_local_loss_weight > 0.0 and local_source.numel() > 0:
                plt.plot(
                    dgi_local_loss_weight * _np.array(local_losses_hist),
                    label=f"{tag}-weighted-local",
                    linewidth=1.0,
                )
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
        history = {
            "loss": np.asarray(losses_hist, dtype=np.float32),
            "dgi_loss": np.asarray(dgi_losses_hist, dtype=np.float32),
            "local_loss": np.asarray(local_losses_hist, dtype=np.float32),
            "loss_ema": np.asarray(ema_hist, dtype=np.float32),
        }
        return model, history

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
    dgi_gcn, gcn_history = train_dgi(
        dgi_gcn,
        epochs=dgi_epochs,
        tag="GCN",
    )
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
            dgi_sage, _sage_history = train_dgi(
                dgi_sage,
                epochs=dgi_epochs,
                tag="SAGE",
            )
            Z_sage = get_embeddings(enc_sage)
            _step_done("Build + train DGI (SAGE)", t0)
            del dgi_sage, enc_sage
            if device.type == "cuda": torch.cuda.empty_cache()
        except Exception:
            Z_sage = None

    # Attach to the graph's native observation unit.
    attach_label = (
        "Attach native-unit embeddings"
        if not broadcast_to_adata1
        else "Attach embeddings to adata2 + broadcast to adata1"
    )
    t0 = _step_start(attach_label)
    adata2.obsm[output_key] = Z_gcn
    if Z_sage is not None:
        adata2.obsm[sage_output_key] = Z_sage
    if compatibility_alias and output_key != "X_dgi_gcn":
        adata2.obsm["X_dgi_gcn"] = Z_gcn.copy()
    if (
        compatibility_alias
        and Z_sage is not None
        and sage_output_key != "X_dgi_sage"
    ):
        adata2.obsm["X_dgi_sage"] = Z_sage.copy()

    local_positive_cosine = float("nan")
    local_negative_cosine = float("nan")
    local_cosine_margin = float("nan")
    if local_pair_arrays[0].size > 0:
        embedding_norm = np.linalg.norm(Z_gcn, axis=1, keepdims=True)
        normalized_embedding = Z_gcn / np.maximum(embedding_norm, 1e-12)
        local_source_np, local_positive_np, local_negative_np = (
            local_pair_arrays
        )
        positive_cosines = np.sum(
            normalized_embedding[local_source_np]
            * normalized_embedding[local_positive_np],
            axis=1,
        )
        negative_cosines = np.sum(
            normalized_embedding[local_source_np]
            * normalized_embedding[local_negative_np],
            axis=1,
        )
        local_positive_cosine = float(np.mean(positive_cosines))
        local_negative_cosine = float(np.mean(negative_cosines))
        local_cosine_margin = float(
            local_positive_cosine - local_negative_cosine
        )

    node_unit = "metaspot" if not broadcast_to_adata1 else "metadomain"
    provenance = dict(adata2.uns.get("dgi_embeddings", {}))
    provenance[output_key] = {
        "schema_version": int(
            METASPOT_DGI_SCHEMA_VERSION
            if node_unit == "metaspot"
            else 1
        ),
        "node_unit": node_unit,
        "framework": "standard_dgi",
        "objective": "global_dgi_plus_pairwise_local_ranking",
        "node_id_checksum": _obs_names_checksum(adata2.obs_names),
        "graph_key": graph_key,
        "feature_keys": [str(key) for key in feature_keys],
        "n_nodes": int(adata2.n_obs),
        "n_features": int(X_all.shape[1]),
        "embedding_dim": int(Z_gcn.shape[1]),
        "encoder": str(dgi_encoder),
        "message_passing_layers": int(dgi_layers),
        "hidden_dim": int(dgi_hidden),
        "dropout": float(dgi_dropout),
        "residual_alpha": float(dgi_residual_alpha),
        "layer_norm": bool(dgi_layer_norm),
        "corruption_mode_requested": str(dgi_corruption_mode),
        "corruption_mode_effective": str(effective_corruption_mode),
        "corruption_group_key": (
            cell_group_key
            if effective_corruption_mode == "same_cell"
            else ""
        ),
        "corruption_group_count": int(
            len(corruption_plan["sizes"])
            if corruption_plan is not None
            else 0
        ),
        "corruption_singleton_group_count": int(
            np.sum(corruption_plan["sizes"] == 1)
            if corruption_plan is not None
            else 0
        ),
        "local_graph_key": str(effective_local_graph_key),
        "gcn_backend_requested": str(dgi_cpu_backend),
        "gcn_backend_effective": str(effective_gcn_backend),
        "local_loss_weight": float(dgi_local_loss_weight),
        "local_temperature": float(dgi_local_temperature),
        "local_samples_requested": int(dgi_local_samples),
        "local_pairs_sampled": int(local_pair_arrays[0].size),
        "final_local_positive_cosine": local_positive_cosine,
        "final_local_negative_cosine": local_negative_cosine,
        "final_local_cosine_margin": local_cosine_margin,
        "epochs": int(dgi_epochs),
        "learning_rate": float(dgi_lr),
        "weight_decay": float(dgi_wd),
        "gradient_clip": float(dgi_clip),
        "amp": bool(dgi_amp),
        "random_seed": int(dgi_seed),
        "training_loss": gcn_history["loss"],
        "training_dgi_loss": gcn_history["dgi_loss"],
        "training_local_loss": gcn_history["local_loss"],
        "training_loss_ema": gcn_history["loss_ema"],
        "final_training_loss": float(gcn_history["loss"][-1]),
        "final_dgi_loss": float(gcn_history["dgi_loss"][-1]),
        "final_local_loss": float(gcn_history["local_loss"][-1]),
        "compatibility_alias": (
            "X_dgi_gcn"
            if compatibility_alias and output_key != "X_dgi_gcn"
            else ""
        ),
    }
    adata2.uns["dgi_embeddings"] = provenance

    if broadcast_to_adata1:
        if "meta-domain" not in adata1.obs.columns:
            raise KeyError(
                "Legacy MetaDomain DGI broadcast requires "
                "adata1.obs['meta-domain']"
            )
        clu_to_row = {
            clu: i for i, clu in enumerate(adata2.obs_names.astype(str))
        }
        row_idx_adata1 = (
            adata1.obs["meta-domain"].astype(str).map(clu_to_row).to_numpy()
        )

        def _broadcast(Z, row_index):
            out = np.zeros((len(row_index), Z.shape[1]), dtype=np.float32)
            mask = (row_index >= 0) & (row_index < Z.shape[0])
            out[mask] = Z[row_index[mask]]
            return out

        adata1.obsm["X_dgi_gcn"] = _broadcast(Z_gcn, row_idx_adata1)
        if Z_sage is not None:
            adata1.obsm["X_dgi_sage"] = _broadcast(Z_sage, row_idx_adata1)
    _step_done(attach_label, t0)
    return adata1, adata2


def _graph_unit_from_config(config: Dict[str, Any]) -> str:
    return dgi_unit_from_config(config)


def _build_module_execution_order(config: Dict[str, Any]) -> List[str]:
    """Return the dependency order for the selected graph observation unit."""
    if _graph_unit_from_config(config) == "metaspot":
        return ["m1", "m2", "m3", "m4", "m7", "m8", "m5", "m6"]
    return ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]


def build_anndata(spatial_df: pd.DataFrame,
                  cell_df: pd.DataFrame,
                  nucleus_df: pd.DataFrame,
                  show_internal_progress: bool = False,
                  progress_callback: Optional[Callable[[str], None]] = None,
                  out_dir: Optional[str] = None,
                  scvi_epochs: int = 100,
                  dgi_epochs: int = 100,
                  enable_dgi_sage: bool = False,
                  fingerprint_context: Optional[Mapping[str, Any]] = None,
                  ) -> Tuple[ad.AnnData, Optional[ad.AnnData], pd.DataFrame]:
    # Fine-grained resume within build_anndata
    if out_dir is None:
        out_dir = os.getcwd()
    state_prev = _load_state(out_dir) or {}
    try:
        cfg = load_params_yaml()
    except Exception:
        cfg = {}
    cur_mod_fps = fingerprint_modules_set(
        cfg,
        context=fingerprint_context,
    )
    # Compact M3 membership is a semantic data artifact. Its reuse depends on
    # the inputs and M1-M3 parameters/schema, not on unrelated source edits in
    # later modules (the normal module fingerprints intentionally include the
    # whole source-tree provenance).
    membership_fingerprint_context = {
        "inputs": (
            fingerprint_context.get("inputs")
            if isinstance(fingerprint_context, Mapping)
            else None
        )
    }
    membership_mod_fps = fingerprint_modules_set(
        cfg,
        context=membership_fingerprint_context,
    )
    prev_mod_fps = state_prev.get("module_fingerprints") or {}
    # First full module run: state has no 'modules_completed' flag yet
    _first_run = not bool(state_prev.get("modules_completed"))
    graph_unit = _graph_unit_from_config(cfg)
    module_order = _build_module_execution_order(cfg)
    fingerprint_order = [m for m in module_order if m != "m4"]
    # Determine the earliest changed module in the actual dependency order.
    start_mod = fingerprint_order[0]
    for m in fingerprint_order:
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
    _late_point_df_cache: Optional[pd.DataFrame] = None
    if graph_unit == "metaspot" and _mod_lt("m4", start_mod):
        _late_point_df_cache = _load_late_point_resume_checkpoint(
            out_dir,
            spatial_df,
        )
        if _late_point_df_cache is not None:
            _logging.getLogger("cellscope").info(
                "Resume: using aligned late point-table checkpoint "
                "(module5_final_data) for Modules 1-3"
            )

    # Use unique cell ids to avoid row-level list blow-up (can be hundreds of millions).
    sampled_cell_ids = pd.unique(spatial_df['cell_id'])
    # Read Module 1 geometry options
    _m1_include_dir = True
    _m1_use_input_nucleus_flags = True
    _m1_winsor = (1, 99)
    try:
        _cfg = load_params_yaml()
        _m1 = _cfg.get('module1', {}) if _cfg else {}
        _m1_include_dir = bool(_m1.get('include_directional', _m1_include_dir))
        _m1_use_input_nucleus_flags = bool(
            _m1.get(
                'use_input_nucleus_flags',
                _m1_use_input_nucleus_flags,
            )
        )
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
        elif _late_point_df_cache is not None:
            point_df = _late_point_df_cache
        else:
            if progress_callback: progress_callback("Module 1 · Geometry Features")
            point_df = build_point_geometry_v2(
                transcripts_df=spatial_df,
                nucleus_boundaries_df=nucleus_df,
                cell_boundaries_df=cell_df,
                sampled_cells=sampled_cell_ids,
                include_directional=_m1_include_dir,
                use_input_nucleus_flags=_m1_use_input_nucleus_flags,
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
            use_input_nucleus_flags=_m1_use_input_nucleus_flags,
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
        if pd_cache is None and _late_point_df_cache is not None:
            pd_cache = _late_point_df_cache
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
            if composition_model is None and _late_point_df_cache is None:
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
    _m3_partition_by_nucleus = False
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
    _m3_parallel_chunk_cells = None
    _m3_parallel_n_jobs = None
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
        if 'partition_by_nucleus' in _m3:
            _m3_partition_by_nucleus = bool(
                _m3.get('partition_by_nucleus')
            )
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
        if isinstance(_m3.get('parallel_chunk_cells'), (int, float)):
            _m3_parallel_chunk_cells = int(_m3.get('parallel_chunk_cells'))
        if isinstance(_m3.get('parallel_n_jobs'), (int, float)):
            _m3_parallel_n_jobs = int(_m3.get('parallel_n_jobs'))
    except Exception:
        pass
    import time as _time
    _t_m3 = _time.perf_counter()
    _m3_recomputed = False
    _m3_compact_reused = False
    pd_cache = None
    if _mod_lt('m3', start_mod) and _cache_chain_valid:
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module3_point_df")
        if pd_cache is None and _late_point_df_cache is not None:
            pd_cache = _late_point_df_cache
    if pd_cache is None:
        pd_cache = _load_module3_membership_checkpoint(
            point_df,
            out_dir,
            module_fingerprints=membership_mod_fps,
        )
        _m3_compact_reused = pd_cache is not None
    if pd_cache is not None:
        if progress_callback:
            progress_callback("Resume: load Module 3 membership")
        point_df = pd_cache
        _logging.getLogger("cellscope").info(
            "Resume: using %s Module 3 checkpoint",
            "compact membership" if _m3_compact_reused else "full point-table",
        )
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
            partition_by_nucleus=_m3_partition_by_nucleus,
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
            parallel_chunk_cells=_m3_parallel_chunk_cells,
            parallel_n_jobs=_m3_parallel_n_jobs,
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
            "Module 3 segmentation done in %.2fs%s",
            _dt_m3,
            (f" (cells={_cells})" if _cells is not None else ""),
        )
    except Exception:
        pass
    _audit_t0 = _time.perf_counter()
    if progress_callback:
        try:
            progress_callback("Module 3 · Audit segmentation quality")
        except Exception:
            pass
    _logging.getLogger("cellscope").info(
        "Module 3 segmentation audit started | rows=%d | cells=%s",
        len(point_df),
        _cells,
    )
    try:
        segmentation_audit = (
            _load_dataframe_checkpoint(
                _intermediate_dir(out_dir),
                "module3_metaspot_segmentation_audit",
            )
            if _m3_compact_reused
            else None
        )
        if segmentation_audit is None:
            segmentation_audit = _metaspot_segmentation_audit(
                point_df,
                target_sites_per_metaspot=_m3_target_sites_per_metaspot,
                min_metaspots_per_cell=_m3_min_metaspots_per_cell,
                max_metaspots_per_cell=_m3_max_metaspots_per_cell,
                partition_by_nucleus=_m3_partition_by_nucleus,
            )
            _save_dataframe(
                segmentation_audit,
                out_dir,
                "module3_metaspot_segmentation_audit",
                diagnostic=True,
            )
        _logging.getLogger("cellscope").info(
            "Module 3 audit: MetaSpots/cell median=%.1f, "
            "actual/requested median=%.3f, "
            "nucleus-mixed transcript fraction=%.6f",
            float(segmentation_audit["actual_metaspots"].median()),
            float(
                segmentation_audit[
                    "actual_to_requested_ratio"
                ].median()
            ),
            float(
                segmentation_audit["mixed_transcripts"].sum()
                / max(
                    1,
                    segmentation_audit["n_transcripts"].sum(),
                )
            ),
        )
        _logging.getLogger("cellscope").info(
            "Module 3 segmentation audit done in %.2fs",
            _time.perf_counter() - _audit_t0,
        )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Module 3 segmentation audit failed after %.2fs: %s",
            _time.perf_counter() - _audit_t0,
            exc,
        )
    # This compact membership contract is mandatory even when bulky
    # intermediate tables are disabled.  It is sufficient to resume or audit
    # segmentation without serializing every Module 1/2 feature column.
    if _m3_recomputed:
        try:
            _save_module3_membership_checkpoint(
                point_df,
                out_dir,
                module_fingerprints=membership_mod_fps,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not persist the mandatory Module 3 MetaSpot "
                "membership checkpoint"
            ) from exc
    # Save histogram of metaspots per cell
    if not _m3_compact_reused:
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
    if progress_callback:
        try:
            progress_callback("Module 3 · Done")
        except Exception:
            pass
    execution_plan: Optional[Dict[str, Any]] = None
    if graph_unit == "metaspot":
        from .scalable_execution import resolve_execution_plan

        resolved_plan = resolve_execution_plan(
            cfg,
            n_metaspots=int(point_df[METASPOT_COL].nunique()),
            n_transcripts=int(len(point_df)),
        )
        execution_plan = resolved_plan.to_dict()
        if not bool(execution_plan.get("full_diagnostics", True)):
            _logging.getLogger("cellscope").info(
                "Streaming profile: disabling full diagnostic tables; "
                "downstream figures follow io.write_intermediate_figures=%s",
                _io_should_write_figures(),
            )
    # Module 4: adata1 from point_df (derived; rerun if m1–m3 changed)
    _m4_recomputed = False
    if (
        (_mod_lt('m4', start_mod) and _cache_chain_valid)
        or _m3_compact_reused
    ):
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

    if graph_unit == "metaspot":
        if execution_plan is None:
            raise RuntimeError("MetaSpot execution plan was not resolved")
        if int(execution_plan["n_metaspots"]) != int(adata1.n_obs):
            raise RuntimeError(
                "Execution-plan MetaSpot count differs from Module 4 AnnData"
            )
        adata1.uns["execution_plan"] = dict(execution_plan)
        try:
            atomic_write_json(
                os.path.join(
                    _intermediate_dir(out_dir or os.getcwd()),
                    "execution_plan.json",
                ),
                execution_plan,
            )
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "Execution-plan manifest could not be written: %s",
                exc,
            )
        _logging.getLogger("cellscope").info(
            "Scale-aware execution plan | profile=%s | MetaSpots=%d | "
            "kNN=%s | DGI=%s | MetaDomain=%s",
            execution_plan["effective_profile"],
            int(adata1.n_obs),
            execution_plan["knn_backend"],
            execution_plan["dgi_training_mode"],
            execution_plan["metadomain_method"],
        )

    # In MetaSpot graph mode, Modules 7 and 8 are true upstream dependencies
    # of MetaDomain construction.  Keep separate checkpoints so an old
    # MetaDomain-broadcast embedding can never masquerade as a MetaSpot model.
    if graph_unit == "metaspot":
        module7_config = (cfg.get("module7", {}) or {})
        neighbor_config = module7_config.get("neighbors", {}) or {}
        if not isinstance(neighbor_config, dict):
            neighbor_config = {}

        _m7_metaspot_recomputed = False
        if _mod_lt("m7", start_mod) and _cache_chain_valid:
            graph_cache = _load_adata_checkpoint(
                os.path.join(
                    _intermediate_dir(out_dir),
                    "module7_metaspot_adata1.h5ad",
                )
            )
            if graph_cache is not None:
                valid, reason = _metaspot_cache_matches(
                    graph_cache,
                    adata1,
                    required_obsm=("X_expr_metaspot_scaled",),
                    required_obsp=(METASPOT_GRAPH_KEY,),
                )
                graph_meta = graph_cache.uns.get("metaspot_graph", {})
                valid = bool(
                    valid
                    and graph_meta.get("node_unit") == "metaspot"
                    and int(graph_meta.get("schema_version", -1))
                    == METASPOT_GRAPH_SCHEMA_VERSION
                    and graph_meta.get("execution_profile")
                    == execution_plan["effective_profile"]
                    and graph_meta.get("knn_backend")
                    == execution_plan["knn_backend"]
                )
                if not valid:
                    _logging.getLogger("cellscope").warning(
                        "Rejecting MetaSpot graph cache: %s",
                        reason,
                    )
                    graph_cache = None
            if graph_cache is not None:
                if progress_callback:
                    progress_callback("Resume: load Module 7 MetaSpot graph")
                adata1 = graph_cache
            else:
                if progress_callback:
                    progress_callback("Module 7 · MetaSpot Graph Construction")
                adata1 = build_metaspot_graph(
                    adata1,
                    point_df,
                    k_expr=int(neighbor_config.get("k_expr", 15)),
                    k_spatial=int(neighbor_config.get("k_spatial", 6)),
                    expr_n_components=int(
                        module7_config.get(
                            "metaspot_expr_n_components",
                            module7_config.get("scvi_n_latent", 32),
                        )
                    ),
                    alpha_expr=float(
                        module7_config.get("metaspot_alpha_expr", 0.8)
                    ),
                    alpha_spatial=float(
                        module7_config.get("metaspot_alpha_spatial", 0.2)
                    ),
                    edge_norm=str(
                        module7_config.get("metaspot_edge_norm", "gcn")
                    ),
                    use_geometry_features=bool(
                        module7_config.get(
                            "metaspot_use_geometry_features",
                            False,
                        )
                    ),
                    expression_cross_cell_only=bool(
                        module7_config.get(
                            "metaspot_expr_cross_cell_only",
                            False,
                        )
                    ),
                    max_full_batch_nodes=int(
                        module7_config.get(
                            "max_full_batch_nodes",
                            200_000,
                        )
                    ),
                    execution_profile=str(
                        execution_plan["effective_profile"]
                    ),
                    knn_backend=str(execution_plan["knn_backend"]),
                    ann_candidate_multiplier=int(
                        module7_config.get("ann_candidate_multiplier", 4)
                    ),
                    ann_query_chunk_nodes=int(
                        module7_config.get("ann_query_chunk_nodes", 100_000)
                    ),
                    ann_train_sample=int(
                        module7_config.get("ann_train_sample", 1_000_000)
                    ),
                    faiss_hnsw_m=int(
                        module7_config.get("faiss_hnsw_m", 32)
                    ),
                    faiss_nprobe=int(
                        module7_config.get("faiss_nprobe", 32)
                    ),
                    faiss_pq_subvectors=int(
                        module7_config.get("faiss_pq_subvectors", 8)
                    ),
                    faiss_pq_bits=int(
                        module7_config.get("faiss_pq_bits", 8)
                    ),
                    faiss_use_gpu=module7_config.get(
                        "faiss_use_gpu", "auto"
                    ),
                    faiss_gpu_device=int(
                        module7_config.get("faiss_gpu_device", 0)
                    ),
                )
                _save_adata(adata1, out_dir, "module7_metaspot_adata1")
                _m7_metaspot_recomputed = True
        else:
            if progress_callback:
                progress_callback("Module 7 · MetaSpot Graph Construction")
            adata1 = build_metaspot_graph(
                adata1,
                point_df,
                k_expr=int(neighbor_config.get("k_expr", 15)),
                k_spatial=int(neighbor_config.get("k_spatial", 6)),
                expr_n_components=int(
                    module7_config.get(
                        "metaspot_expr_n_components",
                        module7_config.get("scvi_n_latent", 32),
                    )
                ),
                alpha_expr=float(
                    module7_config.get("metaspot_alpha_expr", 0.8)
                ),
                alpha_spatial=float(
                    module7_config.get("metaspot_alpha_spatial", 0.2)
                ),
                edge_norm=str(
                    module7_config.get("metaspot_edge_norm", "gcn")
                ),
                use_geometry_features=bool(
                    module7_config.get(
                        "metaspot_use_geometry_features",
                        False,
                    )
                ),
                expression_cross_cell_only=bool(
                    module7_config.get(
                        "metaspot_expr_cross_cell_only",
                        False,
                    )
                ),
                max_full_batch_nodes=int(
                    module7_config.get("max_full_batch_nodes", 200_000)
                ),
                execution_profile=str(execution_plan["effective_profile"]),
                knn_backend=str(execution_plan["knn_backend"]),
                ann_candidate_multiplier=int(
                    module7_config.get("ann_candidate_multiplier", 4)
                ),
                ann_query_chunk_nodes=int(
                    module7_config.get("ann_query_chunk_nodes", 100_000)
                ),
                ann_train_sample=int(
                    module7_config.get("ann_train_sample", 1_000_000)
                ),
                faiss_hnsw_m=int(module7_config.get("faiss_hnsw_m", 32)),
                faiss_nprobe=int(module7_config.get("faiss_nprobe", 32)),
                faiss_pq_subvectors=int(
                    module7_config.get("faiss_pq_subvectors", 8)
                ),
                faiss_pq_bits=int(module7_config.get("faiss_pq_bits", 8)),
                faiss_use_gpu=module7_config.get("faiss_use_gpu", "auto"),
                faiss_gpu_device=int(
                    module7_config.get("faiss_gpu_device", 0)
                ),
            )
            _save_adata(adata1, out_dir, "module7_metaspot_adata1")
            _m7_metaspot_recomputed = True
        if _m7_metaspot_recomputed:
            _cache_chain_valid = False
        _logging.getLogger("cellscope").info(
            "Module 7 MetaSpot graph ready: nodes=%d edges=%d",
            adata1.n_obs,
            int(adata1.obsp[METASPOT_GRAPH_KEY].nnz),
        )

        _m8_metaspot_recomputed = False
        if _mod_lt("m8", start_mod) and _cache_chain_valid:
            dgi_cache = _load_adata_checkpoint(
                os.path.join(
                    _intermediate_dir(out_dir),
                    "module8_metaspot_adata1.h5ad",
                )
            )
            if dgi_cache is not None:
                valid, reason = _metaspot_cache_matches(
                    dgi_cache,
                    adata1,
                    required_obsm=(METASPOT_DGI_KEY,),
                    required_obsp=(METASPOT_GRAPH_KEY,),
                )
                dgi_meta = (
                    dgi_cache.uns.get("dgi_embeddings", {}).get(
                        METASPOT_DGI_KEY,
                        {},
                    )
                )
                valid = bool(
                    valid
                    and dgi_meta.get("node_unit") == "metaspot"
                    and int(dgi_meta.get("schema_version", -1))
                    == METASPOT_DGI_SCHEMA_VERSION
                    and dgi_meta.get("node_id_checksum")
                    == _obs_names_checksum(adata1.obs_names)
                    and dgi_meta.get("training_mode")
                    == execution_plan["dgi_training_mode"]
                )
                if not valid:
                    _logging.getLogger("cellscope").warning(
                        "Rejecting MetaSpot DGI cache: %s",
                        reason,
                    )
                    dgi_cache = None
            if dgi_cache is not None:
                if progress_callback:
                    progress_callback("Resume: load Module 8 MetaSpot DGI")
                adata1 = dgi_cache
            else:
                if progress_callback:
                    progress_callback("Module 8 · MetaSpot DGI Embedding")
                feature_keys = tuple(
                    adata1.uns["metaspot_graph"]["feature_keys"]
                )
                _, embedded = run_module8_dgi(
                    adata1,
                    adata1,
                    show_internal_progress=show_internal_progress,
                    dgi_epochs=dgi_epochs,
                    enable_dgi_sage=enable_dgi_sage,
                    out_dir=out_dir,
                    progress_callback=progress_callback,
                    graph_key=METASPOT_GRAPH_KEY,
                    feature_keys=feature_keys,
                    output_key=METASPOT_DGI_KEY,
                    sage_output_key=METASPOT_DGI_SAGE_KEY,
                    broadcast_to_adata1=False,
                    compatibility_alias=True,
                )
                adata1 = embedded
                _save_adata(adata1, out_dir, "module8_metaspot_adata1")
                _m8_metaspot_recomputed = True
        else:
            if progress_callback:
                progress_callback("Module 8 · MetaSpot DGI Embedding")
            feature_keys = tuple(adata1.uns["metaspot_graph"]["feature_keys"])
            _, embedded = run_module8_dgi(
                adata1,
                adata1,
                show_internal_progress=show_internal_progress,
                dgi_epochs=dgi_epochs,
                enable_dgi_sage=enable_dgi_sage,
                out_dir=out_dir,
                progress_callback=progress_callback,
                graph_key=METASPOT_GRAPH_KEY,
                feature_keys=feature_keys,
                output_key=METASPOT_DGI_KEY,
                sage_output_key=METASPOT_DGI_SAGE_KEY,
                broadcast_to_adata1=False,
                compatibility_alias=True,
            )
            adata1 = embedded
            _save_adata(adata1, out_dir, "module8_metaspot_adata1")
            _m8_metaspot_recomputed = True
        if _m8_metaspot_recomputed:
            _cache_chain_valid = False
        _logging.getLogger("cellscope").info(
            "Module 8 MetaSpot DGI ready: shape=%s",
            tuple(adata1.obsm[METASPOT_DGI_KEY].shape),
        )

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
                module5_config_override={
                    "large_scale_method": (
                        execution_plan["metadomain_method"]
                        if execution_plan is not None
                        else "auto"
                    )
                },
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
            module5_config_override={
                "large_scale_method": (
                    execution_plan["metadomain_method"]
                    if execution_plan is not None
                    else "auto"
                )
            },
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
    if _io_should_write_figures():
        try:
            _plot_metadomain_k_selection_diagnostics(
                adata1,
                out_dir or os.getcwd(),
            )
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "MetaDomain clustering-diagnostic plotting failed: %s",
                exc,
            )

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

    if graph_unit == "metaspot":
        # Modules 7/8 already ran on adata1 before MetaDomain construction.
        # adata2 now carries pooled MetaSpot representations for annotation
        # Leiden; do not train a second DGI in the first implementation.
        final_csv = point_df.copy()
        return adata1, adata2, final_csv

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
    else:
        if progress_callback: progress_callback("Module 7 · Graph Construction")
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
    else:
        if progress_callback: progress_callback("Module 8 · Graph Embeddings (DGI)")
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


def _io_should_write_diagnostic_tables() -> bool:
    """Return whether diagnostic tables are requested by the active config."""
    try:
        cfg = load_params_yaml()
        io_cfg = cfg.get("io", {}) if cfg else {}
        return bool(io_cfg.get("write_diagnostic_tables", True)) and bool(
            io_cfg.get("write_diagnostic_csv", True)
            or io_cfg.get("write_diagnostic_parquet", False)
        )
    except Exception:
        return True


def _full_metaspot_diagnostics_enabled(adata1: ad.AnnData) -> bool:
    """Honor both scale-aware planning and explicit diagnostic I/O policy."""
    execution_plan = adata1.uns.get("execution_plan", {})
    scale_allows = bool(execution_plan.get("full_diagnostics", True))
    return scale_allows and _io_should_write_diagnostic_tables()


def _broadcast_metaspot_diagnostics_enabled(adata1: ad.AnnData) -> bool:
    """Avoid duplicating wide MetaSpot audits onto every streaming transcript."""
    execution_plan = adata1.uns.get("execution_plan", {})
    default = bool(execution_plan.get("full_diagnostics", True))
    try:
        cfg = load_params_yaml()
        io_cfg = cfg.get("io", {}) if cfg else {}
        if "broadcast_metaspot_diagnostics_to_transcripts" in io_cfg:
            return bool(
                io_cfg.get("broadcast_metaspot_diagnostics_to_transcripts")
            )
    except Exception:
        pass
    return default

def _intermediate_dirname() -> str:
    """Return the configured intermediate folder name without creating it."""
    try:
        _cfg = load_params_yaml()
        dname = (_cfg.get('io', {}) if _cfg else {}).get('intermediate_dirname', 'intermediate')
        dname = str(dname).strip() or 'intermediate'
    except Exception:
        dname = 'intermediate'
    return dname


def _intermediate_dir(out_dir: str) -> str:
    """Return and create the configured intermediate output directory."""
    p = os.path.join(out_dir, _intermediate_dirname())
    os.makedirs(p, exist_ok=True)
    return p

def _atomic_dataframe_write(
    df: pd.DataFrame,
    path: str,
    *,
    file_format: str,
) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=f".tmp.{file_format}",
        dir=directory,
    )
    os.close(fd)
    try:
        if file_format == "csv":
            df.to_csv(temporary, index=False)
        elif file_format == "parquet":
            df.to_parquet(temporary, index=False)
        else:
            raise ValueError(f"Unsupported dataframe format: {file_format}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _quarantine_stale_file(path: str) -> Optional[str]:
    """Move an older output aside so it cannot masquerade as this run."""
    if not os.path.isfile(path):
        return None
    candidate = f"{path}.stale"
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{path}.stale.{suffix}"
        suffix += 1
    os.replace(path, candidate)
    return candidate


def _atomic_partitioned_parquet_write(
    df: pd.DataFrame,
    directory: str,
    *,
    target_rows_per_part: int = 1_000_000,
) -> Dict[str, Any]:
    """Write a cell-stable Parquet dataset with an atomic directory swap."""
    import json
    import shutil

    parent = os.path.dirname(os.path.abspath(directory))
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(
        prefix=f".{os.path.basename(directory)}.",
        suffix=".tmp",
        dir=parent,
    )
    target_rows = max(1, int(target_rows_per_part))
    n_parts = max(1, int(math.ceil(len(df) / target_rows)))
    if "cell_id" in df.columns:
        cell_codes, _ = pd.factorize(df["cell_id"], sort=False)
        partition_ids = np.mod(cell_codes, n_parts).astype(np.int32)
    else:
        partition_ids = np.minimum(
            np.arange(len(df), dtype=np.int64) // target_rows,
            n_parts - 1,
        ).astype(np.int32)
    order = np.argsort(partition_ids, kind="stable")
    sorted_parts = partition_ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_parts)) + 1]
    ends = np.r_[starts[1:], len(order)]
    parts = []
    try:
        for start, end in zip(starts, ends):
            part_id = int(sorted_parts[start]) if end > start else 0
            name = f"part-{part_id:05d}.parquet"
            path = os.path.join(temporary, name)
            frame = df.iloc[order[start:end]]
            frame.to_parquet(path, index=False)
            parts.append(
                {
                    "file": name,
                    "rows": int(len(frame)),
                    "cells": (
                        int(frame["cell_id"].nunique())
                        if "cell_id" in frame.columns
                        else None
                    ),
                }
            )
        manifest = {
            "schema_version": 1,
            "format": "partitioned_parquet",
            "partition_strategy": (
                "stable_cell_hash" if "cell_id" in df.columns else "row_range"
            ),
            "rows": int(len(df)),
            "columns": [str(column) for column in df.columns],
            "n_parts": int(len(parts)),
            "target_rows_per_part": target_rows,
            "parts": parts,
        }
        with open(os.path.join(temporary, "manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        open(os.path.join(temporary, "_SUCCESS"), "w").close()
        previous = f"{directory}.previous"
        if os.path.exists(previous):
            if os.path.isdir(previous):
                shutil.rmtree(previous)
            else:
                os.unlink(previous)
        if os.path.exists(directory):
            os.replace(directory, previous)
        os.replace(temporary, directory)
        if os.path.exists(previous):
            if os.path.isdir(previous):
                shutil.rmtree(previous)
            else:
                os.unlink(previous)
        return manifest
    finally:
        if os.path.isdir(temporary):
            shutil.rmtree(temporary)


_DEPRECATED_OUTPUT_ARTIFACTS = (
    "rapx_umap_cluster_annotation.png",
    "annotation_umap_leiden.png",
    "fine_annotation_proportions.png",
    "fine_annotation_proportions.csv",
    "fine_annotation_proportions.parquet",
)


def _remove_deprecated_output_artifacts(out_dir: str) -> List[str]:
    """Remove superseded files that would misrepresent a resumed run.

    These paths are deliberately exact and confined to the configured
    intermediate directory.  This prevents stale figures from surviving when
    a newer code version no longer emits them.
    """
    intermediate = os.path.join(out_dir, _intermediate_dirname())
    if not os.path.isdir(intermediate):
        return []
    removed: List[str] = []
    for name in _DEPRECATED_OUTPUT_ARTIFACTS:
        path = os.path.join(intermediate, name)
        if not os.path.isfile(path):
            continue
        try:
            os.unlink(path)
            removed.append(path)
            _logging.getLogger("cellscope").info(
                "Removed deprecated output artifact: %s",
                path,
            )
        except OSError as exc:
            _logging.getLogger("cellscope").warning(
                "Could not remove deprecated output artifact %s: %s",
                path,
                exc,
            )
    return removed


def _remove_inactive_method_artifacts(
    out_dir: str,
    config: Mapping[str, Any],
) -> List[str]:
    """Remove exact QC files emitted only by the inactive model path.

    Resumed runs intentionally share one output directory. Without this
    cleanup, a CA-DMGI/consensus-Leiden run can appear beside stale
    standard-DGI/KMeans figures and incorrectly imply that both generated the
    final result.
    """
    module5 = config.get("module5", {}) if isinstance(config, Mapping) else {}
    module8 = config.get("module8", {}) if isinstance(config, Mapping) else {}
    clustering_method = str(
        module5.get("clustering_method", "minibatch_kmeans")
        if isinstance(module5, Mapping)
        else "minibatch_kmeans"
    ).strip().lower().replace("-", "_")
    framework = str(
        module8.get("dgi_framework", "standard_dgi")
        if isinstance(module8, Mapping)
        else "standard_dgi"
    ).strip().lower().replace("-", "_")
    inactive_names: List[str] = []
    if clustering_method in {"consensus_leiden", "stability_leiden", "leiden"}:
        inactive_names.extend(
            [
                "metadomain_k_selection_diagnostics.png",
                "module5_k_selection_diagnostics.csv",
                "module5_k_selection_diagnostics.parquet",
            ]
        )
    else:
        inactive_names.extend(
            [
                "metadomain_consensus_leiden_diagnostics.png",
                "module5_consensus_leiden_diagnostics.csv",
                "module5_consensus_leiden_diagnostics.parquet",
            ]
        )
    legacy_framework = "ch_dmgi"  # legacy compatibility only
    if framework in {"cell_aware_dmgi", "ca_dmgi", legacy_framework, "dmgi"}:
        inactive_names.append("module8_dgi_GCN_curves.png")
    else:
        inactive_names.extend(
            [
                "module8_ca_dmgi_training.png",
                "module8_ca_dmgi_training_history.csv",
                "module8_ca_dmgi_training_history.parquet",
                "module8_ca_dmgi_summary.csv",
                "module8_ca_dmgi_summary.parquet",
            ]
        )
    inactive_names.extend(
        [
            "module8_ch_dmgi_training.png",  # legacy compatibility only
            "module8_ch_dmgi_training_history.csv",  # legacy compatibility only
            "module8_ch_dmgi_training_history.parquet",  # legacy compatibility only
            "module8_ch_dmgi_summary.csv",  # legacy compatibility only
            "module8_ch_dmgi_summary.parquet",  # legacy compatibility only
        ]
    )

    intermediate = os.path.join(out_dir, _intermediate_dirname())
    removed: List[str] = []
    for name in inactive_names:
        path = os.path.join(intermediate, name)
        if not os.path.isfile(path):
            continue
        try:
            os.unlink(path)
            removed.append(path)
            _logging.getLogger("cellscope").info(
                "Removed inactive method artifact: %s",
                path,
            )
        except OSError as exc:
            _logging.getLogger("cellscope").warning(
                "Could not remove inactive method artifact %s: %s",
                path,
                exc,
            )
    return removed


def _write_output_guide(
    out_dir: str,
    *,
    counts: Mapping[str, Any],
    annotation_enabled: bool,
    primary_data_product: str = "final_data.parquet",
) -> str:
    """Write a concise human-readable index of production and audit outputs."""
    intermediate_name = _intermediate_dirname()
    guide_path = os.path.join(out_dir, "OUTPUT_GUIDE.md")
    count_text = (
        f"{int(counts.get('n_transcripts') or 0):,} transcripts, "
        f"{int(counts.get('n_cells') or 0):,} cells, "
        f"{int(counts.get('n_metaspots') or 0):,} MetaSpots, and "
        f"{int(counts.get('n_metadomains') or 0):,} MetaDomains"
    )
    annotation_section = ""
    if annotation_enabled:
        annotation_section = f"""
## Recommended result figures (when figure output is enabled)

- `{intermediate_name}/metaspot_annotation_result.png`: canonical annotation
  result at MetaSpot resolution, including final labels and evidence tiers.
- `{intermediate_name}/final_annotation_proportions.png`: per-cell distribution
  of the canonical seven-class transcript annotation.

## Cell × Structure atlas (after Phase 1 audit)

- `{intermediate_name}/cell_structure_atlas.csv`: one row per cell and fine
  structure with four-state detection status, multi-source evidence, loci, and
  total-cell/compartment burden.
- `{intermediate_name}/cell_structure_summary.csv`: per-cell fine-structure
  counts and detection coverage normalized by assay detectability.
- `{intermediate_name}/cell_structure_depth_summary.csv`: marker evidence,
  detectability, detected structures, and coverage by transcript depth.

## Clustering and method diagnostics (when available)

- `{intermediate_name}/metadomain_leiden_clusters.png`: MetaDomain UMAP colored
  by the final `Subcellular Annotation` label. The legacy filename is retained
  for output compatibility; Leiden cluster IDs remain available in the AnnData
  audit fields rather than defining the figure colors.
- `{intermediate_name}/metadomain_annotation_quality.png`: MetaDomain aggregation
  and transfer into final MetaSpot labels.
- `{intermediate_name}/metadomain_consensus_leiden_diagnostics.png`: default
  multi-resolution, multi-seed clustering stability/occupancy audit. The
  companion `module5_consensus_leiden_diagnostics.*` contains exact values.
- `{intermediate_name}/module8_ca_dmgi_training.png`: default Cell-Aware Deep
  Multiplex Graph Infomax (CA-DMGI) relation-specific expression/spatial losses
  and learned consensus-gate trajectory. The companion
  `module8_ca_dmgi_training_history.*` and `module8_ca_dmgi_summary.*` contain
  exact values and graph semantics.
- `metadomain_k_selection_diagnostics.png` and `module8_dgi_GCN_curves.png`
  are produced only by the legacy MiniBatchKMeans/standard-DGI fallback paths.

## Audit-only annotation tables

- `{intermediate_name}/rapx_annotation_scores.csv`: full RAP-X evidence matrix.
- `{intermediate_name}/rapx_metadomain_evidence.csv`,
  `{intermediate_name}/rapx_candidate_to_strict.csv`, and
  `{intermediate_name}/rapx_candidate_evidence_tier.csv`: confirmatory evidence
  audit. `Unknown` in the strict track means "not confirmatory-FDR supported";
  it is not the final public annotation.
- `{intermediate_name}/metaspot_local_annotation_scores.csv` and
  `{intermediate_name}/metaspot_fine_decision_provenance.csv`: local marker,
  DGI-neighborhood, rescue, and fallback evidence.
- `adata1.obs["subcellular_metaspot_diminishing_return_*"]`: original label,
  locus rank, evidence threshold, burden, and pruning decision for repeated
  fine-structure loci. Pruned loci fall back to their broad parent label.
- `adata1.obs["subcellular_metaspot_multistructure_first_locus_*"]`: unified
  five-structure first-locus selection, evidence scores, and projected-burden
  safety decision. These fields alter labels only when the corresponding
  option in `effective_params.yaml` is enabled.
- `{intermediate_name}/omm_cell_detection_tiers.csv`: supported OMM calls and
  the separate audit-only exploratory sensitivity layer. Exploratory rows do
  not overwrite the canonical public label.
"""
    primary_name = os.path.basename(primary_data_product)
    optional_name = (
        "final_data.csv"
        if primary_name == "final_data.parquet"
        else "final_data.parquet"
    )
    text = f"""# CellScoper output guide

This successful run contains {count_text}.

## Primary data products

- `{primary_name}`: transcript-level production result written successfully by
  this run. `{optional_name}` is optional and must not be treated as current
  unless it is also recorded in `run_manifest.json`.
- `adata1.h5ad`: MetaSpot-level production object. The canonical public label is
  `obs[\"subcellular_annotation\"]`. Multi-structure Phase 2 evidence is stored in
  `obsm[\"X_subcellular_structure_presence\"]`,
  `obsm[\"X_subcellular_structure_localization\"]`, and
  `obsm[\"X_subcellular_structure_evidence\"]`; structure order and score
  semantics are recorded in `uns[\"metaspot_local_annotation\"]`.
- `adata2.h5ad`: internal MetaDomain evidence and clustering object; it is not
  the final annotation grain.
- `effective_params.yaml`, `params_used.yaml`, and `run_manifest.json`:
  configuration and provenance required to reproduce the run.
{annotation_section}
## Checkpoints and compact QC

Other files under `{intermediate_name}/` are restart checkpoints or compact
diagnostic tables. They should not be presented as final biological results
unless their observation unit and decision stage are stated explicitly.
"""
    fd, temporary = tempfile.mkstemp(
        prefix=".OUTPUT_GUIDE.",
        suffix=".tmp.md",
        dir=out_dir,
    )
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text.strip() + "\n")
        os.replace(temporary, guide_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return guide_path


def _save_dataframe(
    df: pd.DataFrame,
    out_dir: str,
    name: str,
    *,
    diagnostic: bool = False,
    force_parquet: bool = False,
):
    """Atomically save an intermediate or compact diagnostic table."""
    if not diagnostic and not _io_should_write_intermediate():
        return
    base = _intermediate_dir(out_dir)
    write_csv = True
    write_pq = True
    try:
        _cfg = load_params_yaml()
        io_cfg = _cfg.get('io', {}) if _cfg else {}
        if diagnostic:
            if not bool(io_cfg.get("write_diagnostic_tables", True)):
                return
            write_csv = bool(io_cfg.get("write_diagnostic_csv", True))
            write_pq = bool(io_cfg.get("write_diagnostic_parquet", False))
        else:
            write_csv = bool(io_cfg.get('write_intermediate_csv', True))
            write_pq = bool(io_cfg.get('write_intermediate_parquet', True))
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Could not read I/O settings for table %s: %s; using defaults",
            name,
            exc,
        )
    if force_parquet:
        write_pq = True
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
            _atomic_dataframe_write(
                df,
                csv_path,
                file_format="csv",
            )
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
            _atomic_dataframe_write(
                df,
                pq_path,
                file_format="parquet",
            )
            written.append(pq_path)
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "Failed to save intermediate Parquet %s: %s",
                pq_path,
                exc,
            )
    if written:
        _logging.getLogger("cellscope").info(
            "Saved %s table | name=%s | rows=%d | columns=%d | files=%s",
            "diagnostic" if diagnostic else "intermediate",
            name,
            len(df),
            int(df.shape[1]),
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

def _cluster_chunk_size(configured: Optional[int] = None) -> int:
    """Chunk size for per-cell parallel loops.

    Env: CELLSCOPE_CLUSTER_CHUNK (int)
    Config: module3.parallel_chunk_cells (with module2's legacy setting as a
    fallback when called outside the pipeline).
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
        if configured is not None:
            return max(16, min(4096, int(configured)))
    except Exception:
        pass
    try:
        from .config import load_params_yaml
        cfg = load_params_yaml() or {}
        m3 = cfg.get("module3", {}) or {}
        cfg_v = m3.get("parallel_chunk_cells")
        if cfg_v is not None:
            return max(16, min(4096, int(cfg_v)))
        m2 = cfg.get("module2", {}) or {}
        cfg_v = m2.get("parallel_chunk_cells")
        if cfg_v is not None:
            return max(16, min(4096, int(cfg_v)))
    except Exception:
        pass
    try:
        v = int(os.environ.get("CELLSCOPE_CLUSTER_CHUNK", "512"))
        return max(128, min(4096, v))
    except Exception:
        return 256


def _state_path(out_dir: str) -> str:
    return os.path.join(out_dir, "pipeline_state.json")

def _load_state(out_dir: str) -> Optional[dict]:
    try:
        import json
        p = _state_path(out_dir)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Could not read pipeline state from %s: %s",
            _state_path(out_dir),
            exc,
        )
    return None

def _save_state(out_dir: str, last_completed: str, extra: Optional[dict] = None) -> None:
    try:
        import time
        state = {"last_completed": last_completed, "ts": time.time()}
        if extra:
            state.update(extra)
        atomic_write_json(_state_path(out_dir), state)
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Could not persist pipeline state after stage %s: %s",
            last_completed,
            exc,
            exc_info=True,
        )

def _sanitize_obs_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    """Normalize obs values unsupported by older AnnData/HDF5 writers."""
    for col in adata.obs.columns:
        series = adata.obs[col]
        # AnnData/HDF5 natively supports categorical and nullable string
        # columns. Expanding them to Python objects used to perform one Python
        # call per value and destroyed compact category codes on large runs.
        if isinstance(series.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_string_dtype(series.dtype) and series.dtype != object:
            continue
        if series.dtype != object:
            continue

        non_missing = series[series.notna()]
        sample = non_missing.iloc[:1024].to_numpy(dtype=object, copy=False)
        contains_sequence = any(
            isinstance(value, (list, tuple)) for value in sample
        )
        all_strings = all(isinstance(value, str) for value in sample)
        if contains_sequence:
            normalized = series.map(
                lambda value: ",".join(map(str, value))
                if isinstance(value, (list, tuple))
                else ("" if pd.isna(value) else str(value))
            )
            adata.obs[col] = normalized.astype(object)
        elif all_strings:
            if series.isna().any():
                adata.obs[col] = series.where(series.notna(), "")
        else:
            adata.obs[col] = series.where(series.notna(), "").astype(str)
    return adata


def _atomic_write_adata(
    adata: ad.AnnData,
    path: str,
    *,
    compression: str,
) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp.h5ad",
        dir=directory,
    )
    os.close(fd)
    try:
        adata.write_h5ad(temporary, compression=compression)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _compact_final_adata1_for_write(
    adata1: ad.AnnData,
    *,
    enabled: bool,
):
    """Temporarily retain only the canonical MetaSpot DGI representation.

    Final output compaction must not mutate the in-memory object or any resume
    checkpoint: the pipeline may still need all relations while writing later
    artifacts.  Holding references while the mappings are cleared avoids a
    second multi-GiB AnnData copy.
    """
    if not enabled:
        yield adata1
        return

    original_obsm = list(adata1.obsm.items())
    original_obsp = list(adata1.obsp.items())
    kept_obsm = [
        (key, value)
        for key, value in original_obsm
        if key == METASPOT_DGI_KEY
    ]
    marker_key = "cellscope_final_output_compaction"
    sentinel = object()
    original_marker = adata1.uns.get(marker_key, sentinel)
    adata1.obsm.clear()
    adata1.obsm.update(kept_obsm)
    adata1.obsp.clear()
    adata1.uns[marker_key] = {
        "enabled": True,
        "kept_obsm": [key for key, _ in kept_obsm],
        "omitted_obsm": [
            key for key, _ in original_obsm if key != METASPOT_DGI_KEY
        ],
        "omitted_obsp": [key for key, _ in original_obsp],
        "reason": "io.drop_embeddings",
    }
    try:
        yield adata1
    finally:
        adata1.obsm.clear()
        adata1.obsm.update(original_obsm)
        adata1.obsp.clear()
        adata1.obsp.update(original_obsp)
        if original_marker is sentinel:
            adata1.uns.pop(marker_key, None)
        else:
            adata1.uns[marker_key] = original_marker


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
        adata.obs = original_obs.copy(deep=False)
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

def _save_fig(fig, out_dir: str, name: str, *, dpi: int = 150):
    if not _io_should_write_figures():
        return
    temporary = None
    try:
        p = os.path.join(_intermediate_dir(out_dir), f"{name}.png")
        fd, temporary = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp.png",
            dir=os.path.dirname(p),
        )
        os.close(fd)
        fig.savefig(temporary, bbox_inches='tight', dpi=int(dpi))
        os.replace(temporary, p)
        temporary = None
        _logging.getLogger("cellscope").info(
            "Saved diagnostic figure | name=%s | path=%s",
            name,
            p,
        )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Failed to save diagnostic figure %s: %s",
            name,
            exc,
            exc_info=True,
        )
    finally:
        if temporary and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _plot_metadomain_k_selection_diagnostics(
    adata1: ad.AnnData,
    out_dir: str,
) -> Optional[pd.DataFrame]:
    """Save the sampled automatic-K evidence alongside Module 5 outputs."""
    metadata = adata1.uns.get("module5_metadomain_clustering", {})
    if metadata.get("clustering_method") == "consensus_leiden":
        rows: List[Dict[str, Any]] = []
        selected_by_block: Dict[str, float] = {}
        for block, block_meta in dict(
            metadata.get("consensus_by_block", {})
        ).items():
            resolutions = np.asarray(
                block_meta.get("candidate_resolution", []), dtype=float
            )
            stability = np.asarray(
                block_meta.get("candidate_stability_ari", []), dtype=float
            )
            n_clusters = np.asarray(
                block_meta.get("candidate_n_clusters", []), dtype=int
            )
            average_size = np.asarray(
                block_meta.get("candidate_average_size", []), dtype=float
            )
            small_fraction = np.asarray(
                block_meta.get("candidate_small_cluster_fraction", []),
                dtype=float,
            )
            nonrecurrent_fraction = np.asarray(
                block_meta.get("candidate_nonrecurrent_fraction", []),
                dtype=float,
            )
            feasible = np.asarray(
                block_meta.get("candidate_feasible", []), dtype=bool
            )
            if not (
                resolutions.size
                and all(
                    values.size == resolutions.size
                    for values in (
                        stability,
                        n_clusters,
                        average_size,
                        small_fraction,
                        nonrecurrent_fraction,
                        feasible,
                    )
                )
            ):
                continue
            selected_resolution = float(
                block_meta.get("selected_resolution", resolutions[0])
            )
            selected_by_block[str(block)] = selected_resolution
            for index, resolution in enumerate(resolutions):
                rows.append(
                    {
                        "block": str(block),
                        "resolution": float(resolution),
                        "stability_ari": float(stability[index]),
                        "n_clusters": int(n_clusters[index]),
                        "average_metaspots_per_cluster": float(
                            average_size[index]
                        ),
                        "small_cluster_fraction": float(
                            small_fraction[index]
                        ),
                        "nonrecurrent_fraction": float(
                            nonrecurrent_fraction[index]
                        ),
                        "feasible": bool(feasible[index]),
                        "selected": bool(
                            np.isclose(resolution, selected_resolution)
                        ),
                    }
                )
        diagnostics = pd.DataFrame(rows)
        if diagnostics.empty:
            return None
        fig, axes = plt.subplots(
            1, 3, figsize=(16, 4.6), constrained_layout=True
        )
        palette = {
            "N": _SUBCELLULAR_LABEL_PALETTE["Nucleus"],
            "C": _SUBCELLULAR_LABEL_PALETTE["Cytosol"],
            "A": _SUBCELLULAR_LABEL_PALETTE["Unknown"],
        }
        for block, group in diagnostics.groupby("block", sort=False):
            color = palette.get(str(block), "#777777")
            axes[0].plot(
                group["resolution"],
                group["stability_ari"],
                marker="o",
                color=color,
                label=str(block),
            )
            axes[1].plot(
                group["resolution"],
                group["n_clusters"],
                marker="o",
                color=color,
                label=str(block),
            )
            axes[2].plot(
                group["resolution"],
                100.0 * group["nonrecurrent_fraction"],
                marker="o",
                color=color,
                label=f"{block}: nonrecurrent",
            )
            axes[2].plot(
                group["resolution"],
                100.0 * group["small_cluster_fraction"],
                marker="s",
                linestyle="--",
                color=color,
                label=f"{block}: small",
            )
            selected = selected_by_block[str(block)]
            for axis in axes:
                axis.axvline(selected, color=color, linestyle=":", alpha=0.7)
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("Cross-seed adjusted Rand index")
        axes[0].set_title("Assignment stability")
        axes[1].set_ylabel("Candidate MetaDomains")
        axes[1].set_title("Resolution and granularity")
        axes[2].set_ylabel("MetaSpots (%)")
        axes[2].set_title("Occupancy and cross-cell recurrence")
        for axis in axes:
            axis.set_xlabel("Leiden resolution")
            axis.legend(frameon=False, fontsize="small")
        fig.suptitle(
            "Stability-selected consensus Leiden",
            fontsize=16,
            fontweight="bold",
        )
        _save_fig(fig, out_dir, "metadomain_consensus_leiden_diagnostics")
        plt.close(fig)
        _save_dataframe(
            diagnostics,
            out_dir,
            "module5_consensus_leiden_diagnostics",
            diagnostic=True,
        )
        return diagnostics

    selection = metadata.get("k_selection", {})
    candidate_k = np.asarray(selection.get("candidate_k", []))
    distortion = np.asarray(selection.get("distortion", []), dtype=float)
    if candidate_k.size < 2 or distortion.size != candidate_k.size:
        return None

    def _aligned(name: str, default: float = np.nan) -> np.ndarray:
        values = np.asarray(selection.get(name, []), dtype=float)
        if values.size == candidate_k.size:
            return values
        return np.full(candidate_k.size, default, dtype=float)

    diagnostics = pd.DataFrame(
        {
            "K": candidate_k.astype(int),
            "heldout_distortion": distortion,
            "stability_ari": _aligned("stability_ari"),
            "small_cluster_fraction": _aligned(
                "small_cluster_fraction"
            ),
            "empty_cluster_fraction": _aligned(
                "empty_cluster_fraction"
            ),
        }
    ).sort_values("K")
    selected_k = int(selection.get("selected_k", candidate_k[0]))
    knee_k = int(selection.get("knee_k", selected_k))
    feasible = np.asarray(selection.get("feasible", []), dtype=bool)
    if feasible.size == candidate_k.size:
        diagnostics["feasible"] = feasible
    else:
        diagnostics["feasible"] = True
    diagnostics["selected"] = diagnostics["K"].eq(selected_k)
    diagnostics["distortion_knee"] = diagnostics["K"].eq(knee_k)

    blue = "#4C78A8"
    orange = "#F28E2B"
    grey = "#9D9D9D"
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.6),
        constrained_layout=True,
    )

    axes[0].plot(
        diagnostics["K"],
        diagnostics["heldout_distortion"],
        marker="o",
        color=blue,
        linewidth=2,
    )
    axes[0].scatter(
        [selected_k],
        diagnostics.loc[
            diagnostics["selected"],
            "heldout_distortion",
        ],
        s=90,
        color=orange,
        edgecolor="black",
        zorder=4,
        label=f"Selected K={selected_k}",
    )
    axes[0].axvline(knee_k, color=grey, linestyle="--")
    axes[0].set_xscale("log")
    axes[0].set_title("Held-out clustering distortion")
    axes[0].set_xlabel("Candidate K (log scale)")
    axes[0].set_ylabel(
        "Mean 1 − cosine similarity"
        if bool(selection.get("use_cosine", True))
        else "Mean squared distance"
    )
    axes[0].legend(frameon=False)

    axes[1].plot(
        diagnostics["K"],
        diagnostics["stability_ari"],
        marker="o",
        color=blue,
        linewidth=2,
    )
    stability_min = float(selection.get("stability_min", 0.4))
    axes[1].axhline(
        stability_min,
        color=grey,
        linestyle="--",
        label=f"Minimum={stability_min:.2f}",
    )
    axes[1].axvline(selected_k, color=orange, linestyle=":")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Cross-seed assignment stability")
    axes[1].set_xlabel("Candidate K (log scale)")
    axes[1].set_ylabel("Adjusted Rand index")
    axes[1].legend(frameon=False)

    axes[2].plot(
        diagnostics["K"],
        100.0 * diagnostics["small_cluster_fraction"],
        marker="o",
        color=blue,
        linewidth=2,
        label="Small-cluster MetaSpots",
    )
    axes[2].plot(
        diagnostics["K"],
        100.0 * diagnostics["empty_cluster_fraction"],
        marker="s",
        color=grey,
        linewidth=1.5,
        label="Empty clusters",
    )
    max_small = float(
        selection.get("max_small_cluster_fraction", 0.01)
    )
    axes[2].axhline(
        100.0 * max_small,
        color=grey,
        linestyle="--",
        label=f"Maximum={100.0 * max_small:.1f}%",
    )
    axes[2].axvline(selected_k, color=orange, linestyle=":")
    axes[2].set_xscale("log")
    axes[2].set_title("Candidate occupancy diagnostics")
    axes[2].set_xlabel("Candidate K (log scale)")
    axes[2].set_ylabel("Fraction (%)")
    axes[2].legend(frameon=False, fontsize="small")

    reason = str(selection.get("selection_reason", ""))
    fig.suptitle(
        "MetaDomain automatic K selection"
        f" · {reason.replace('_', ' ')}",
        fontsize=16,
        fontweight="bold",
    )
    _save_fig(fig, out_dir, "metadomain_k_selection_diagnostics")
    plt.close(fig)
    _save_dataframe(
        diagnostics,
        out_dir,
        "module5_k_selection_diagnostics",
        diagnostic=True,
    )
    return diagnostics


def _metadomain_broad_compartment(values: Sequence[object]) -> np.ndarray:
    values = np.asarray(values, dtype=object)
    broad = np.full(values.size, "Other", dtype=object)
    broad[
        np.isin(
            values,
            [
                "Nucleus",
                "Nucleolus",
                "Lamina",
                "Nuclear Pore",
            ],
        )
    ] = "Nuclear"
    broad[
        np.isin(
            values,
            ["Cytosol", "ER", "OMM"],
        )
    ] = "Cytoplasmic"
    return broad


def _attach_annotation_provenance_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Describe how production labels relate to MetaDomain evidence."""
    required = {
        "subcellular_annotation_domain",
        "subcellular_annotation",
    }
    if not required.issubset(frame.columns):
        return frame
    comparison_column = (
        "subcellular_annotation_compatibility"
        if "subcellular_annotation_compatibility" in frame.columns
        else "subcellular_annotation"
    )
    categories = list(_SUBCELLULAR_LABEL_ORDER)
    unknown_code = categories.index("Unknown")

    def _annotation_codes(values: pd.Series) -> np.ndarray:
        codes = pd.Categorical(values, categories=categories).codes.astype(
            np.int8,
            copy=False,
        )
        if np.any(codes < 0):
            codes = codes.copy()
            codes[codes < 0] = unknown_code
        return codes

    domain_codes = _annotation_codes(
        frame["subcellular_annotation_domain"]
    )
    final_codes = _annotation_codes(frame[comparison_column])
    changed = domain_codes != final_codes
    broad_lookup = np.zeros(len(categories), dtype=np.int8)
    broad_lookup[
        [
            categories.index(label)
            for label in _SUBCELLULAR_LABEL_GROUPS["Nuclear side"]
        ]
    ] = 1
    broad_lookup[
        [
            categories.index(label)
            for label in _SUBCELLULAR_LABEL_GROUPS["Cytoplasmic side"]
        ]
    ] = 2
    domain_broad = broad_lookup[domain_codes]
    final_broad = broad_lookup[final_codes]
    broad_changed = domain_broad != final_broad
    provenance_codes = np.zeros(
        len(frame),
        dtype=np.int8,
    )
    provenance_codes[changed & ~broad_changed] = 1
    provenance_codes[changed & broad_changed] = 2
    supported = ~changed
    for column in (
        "subcellular_annotation_domain_strict",
        "subcellular_annotation_domain_exploratory",
    ):
        if column in frame.columns:
            evidence_codes = _annotation_codes(frame[column])
            supported |= (
                (evidence_codes == final_codes)
                & (evidence_codes != unknown_code)
            )
    return _replace_frame_columns(
        frame,
        {
            "subcellular_annotation_provenance": (
                pd.Categorical.from_codes(
                    provenance_codes,
                    categories=[
                        "MetaDomain RAP-X",
                        "MetaSpot subtype/local refinement",
                        "MetaSpot broad-compartment correction",
                    ],
                )
            ),
            "final_label_matches_domain_annotation": ~changed,
            "final_label_supported_by_domain_evidence": supported,
        },
    )


def _build_metadomain_annotation_quality_tables(
    adata1: ad.AnnData,
) -> Dict[str, Any]:
    """Build scalable MetaDomain aggregation and label-transfer diagnostics."""
    required = {
        "meta-domain",
        "subcellular_annotation_domain",
        "subcellular_annotation",
    }
    missing = sorted(required.difference(adata1.obs.columns))
    if missing:
        raise KeyError(
            "MetaDomain quality diagnostics require adata1.obs columns: "
            + ", ".join(missing)
        )

    domains = adata1.obs["meta-domain"].astype(str)
    domain_codes, domain_names_raw = pd.factorize(domains, sort=True)
    domain_names = pd.Index(
        domain_names_raw.astype(str),
        name="meta-domain",
    )
    n_domains = len(domain_names)
    n_spots = adata1.n_obs
    sizes = np.bincount(domain_codes, minlength=n_domains)

    domain_labels = (
        adata1.obs["subcellular_annotation_domain"]
        .astype("string")
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )
    compatibility_column = (
        "subcellular_annotation_compatibility"
        if "subcellular_annotation_compatibility" in adata1.obs.columns
        else "subcellular_annotation"
    )
    final_labels = (
        adata1.obs[compatibility_column]
        .astype("string")
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )
    exact_agreement = domain_labels == final_labels
    broad_domain = _metadomain_broad_compartment(domain_labels)
    broad_final = _metadomain_broad_compartment(final_labels)
    broad_agreement = broad_domain == broad_final

    geometry = np.full(n_spots, "Unavailable", dtype=object)
    if "r_mean" in adata1.obs.columns:
        r_mean = pd.to_numeric(
            adata1.obs["r_mean"],
            errors="coerce",
        ).to_numpy(float)
        finite = np.isfinite(r_mean)
        geometry[finite & (r_mean < 0)] = "Nuclear"
        geometry[finite & (r_mean >= 0)] = "Cytoplasmic"

    loo_cosine = np.full(n_spots, np.nan, dtype=np.float32)
    clustering_meta = adata1.uns.get(
        "module5_metadomain_clustering",
        {},
    )
    representation_key = str(
        clustering_meta.get("cluster_rep", "")
    )
    if (
        representation_key == METASPOT_DGI_KEY
        and representation_key not in adata1.obsm
        and "X_dgi_gcn" in adata1.obsm
    ):
        representation_key = "X_dgi_gcn"
    representation_available = representation_key in adata1.obsm
    if representation_available:
        from sklearn.preprocessing import normalize

        representation = np.asarray(
            adata1.obsm[representation_key],
            dtype=np.float32,
        )
        representation = StandardScaler().fit_transform(
            representation
        ).astype(np.float32)
        representation = normalize(
            representation,
            norm="l2",
        ).astype(np.float32)
        sums = np.zeros(
            (n_domains, representation.shape[1]),
            dtype=np.float32,
        )
        np.add.at(sums, domain_codes, representation)
        valid_loo = sizes[domain_codes] > 1
        own_loo = normalize(
            sums[domain_codes[valid_loo]]
            - representation[valid_loo],
            norm="l2",
        )
        loo_cosine[valid_loo] = np.einsum(
            "ij,ij->i",
            representation[valid_loo],
            own_loo,
        ).astype(np.float32)

    spot_table = pd.DataFrame(
        {
            "meta-domain": domains.to_numpy(),
            "cell_id": (
                adata1.obs["cell_id"].astype(str).to_numpy()
                if "cell_id" in adata1.obs.columns
                else "Unavailable"
            ),
            "geometry_compartment": geometry,
            "domain_annotation": domain_labels,
            "final_annotation": final_labels,
            "domain_final_exact_agreement": exact_agreement,
            "domain_final_broad_agreement": broad_agreement,
            "embedding_loo_cosine": loo_cosine,
        },
        index=adata1.obs_names,
    )
    grouped = spot_table.groupby("meta-domain", sort=False)

    def _purity(series: pd.Series) -> float:
        counts = series.value_counts(dropna=False)
        return float(counts.iloc[0] / counts.sum())

    def _mode(series: pd.Series) -> str:
        return str(series.value_counts(dropna=False).index[0])

    domain_table = pd.DataFrame(index=domain_names)
    domain_table["n_metaspots"] = grouped.size().reindex(domain_names)
    domain_table["domain_label"] = (
        grouped["domain_annotation"].apply(_mode).reindex(domain_names)
    )
    domain_table["dominant_final_label"] = (
        grouped["final_annotation"].apply(_mode).reindex(domain_names)
    )
    domain_table["geometry_purity"] = (
        grouped["geometry_compartment"]
        .apply(_purity)
        .reindex(domain_names)
    )
    domain_table["final_label_purity"] = (
        grouped["final_annotation"]
        .apply(_purity)
        .reindex(domain_names)
    )
    domain_table["domain_final_exact_agreement"] = (
        grouped["domain_final_exact_agreement"]
        .mean()
        .reindex(domain_names)
    )
    domain_table["domain_final_broad_agreement"] = (
        grouped["domain_final_broad_agreement"]
        .mean()
        .reindex(domain_names)
    )
    domain_table["embedding_loo_cosine_median"] = (
        grouped["embedding_loo_cosine"]
        .median()
        .reindex(domain_names)
    )
    domain_table["representation_key"] = representation_key
    domain_table["representation_available"] = representation_available

    if "cell_id" in adata1.obs.columns:
        cell_counts = (
            spot_table.groupby(["meta-domain", "cell_id"]).size()
        )
        domain_table["n_cells"] = (
            cell_counts.groupby(level=0)
            .size()
            .reindex(domain_names)
        )
        domain_table["dominant_cell_fraction"] = (
            cell_counts.groupby(level=0).max()
            / cell_counts.groupby(level=0).sum()
        ).reindex(domain_names)

    transfer_counts = pd.crosstab(
        spot_table["domain_annotation"],
        spot_table["final_annotation"],
    )
    transfer_fraction = transfer_counts.div(
        transfer_counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)
    transfer_long = (
        transfer_counts.rename_axis(
            index="domain_annotation",
            columns="final_annotation",
        )
        .stack()
        .rename("n_metaspots")
        .reset_index()
    )
    fraction_lookup = (
        transfer_fraction.rename_axis(
            index="domain_annotation",
            columns="final_annotation",
        )
        .stack()
        .rename("row_fraction")
        .reset_index()
    )
    transfer_long = transfer_long.merge(
        fraction_lookup,
        on=["domain_annotation", "final_annotation"],
        how="left",
    )

    domain_composition = pd.Series(
        domain_labels,
        dtype="object",
    ).value_counts(normalize=True)
    final_composition = pd.Series(
        final_labels,
        dtype="object",
    ).value_counts(normalize=True)
    composition_labels = list(
        dict.fromkeys(
            [
                "Nucleus",
                "Nucleolus",
                "Lamina",
                "Nuclear Pore",
                "Cytosol",
                "ER",
                "OMM",
                "Unknown",
            ]
            + domain_composition.index.tolist()
            + final_composition.index.tolist()
        )
    )
    composition = pd.DataFrame(
        {
            "label": composition_labels,
            "domain_evidence_fraction": domain_composition.reindex(
                composition_labels,
                fill_value=0.0,
            ).to_numpy(),
            "final_metaspot_fraction": final_composition.reindex(
                composition_labels,
                fill_value=0.0,
            ).to_numpy(),
        }
    )
    composition = composition[
        (
            composition["domain_evidence_fraction"]
            + composition["final_metaspot_fraction"]
        )
        > 0
    ].reset_index(drop=True)

    summary = {
        "n_metaspots": int(n_spots),
        "n_metadomains": int(n_domains),
        "weighted_geometry_purity": float(
            np.average(
                domain_table["geometry_purity"],
                weights=domain_table["n_metaspots"],
            )
        ),
        "domain_final_exact_agreement": float(
            np.mean(exact_agreement)
        ),
        "domain_final_broad_agreement": float(
            np.mean(broad_agreement)
        ),
        "representation_key": representation_key,
        "representation_available": representation_available,
    }
    return {
        "domains": domain_table,
        "spots": spot_table,
        "transfer_counts": transfer_counts,
        "transfer_fraction": transfer_fraction,
        "transfer_long": transfer_long,
        "composition": composition,
        "summary": summary,
    }


def _plot_metadomain_annotation_quality(
    adata1: ad.AnnData,
    out_dir: str,
) -> Optional[Dict[str, Any]]:
    """Save MetaDomain aggregation and annotation-transfer diagnostics."""
    try:
        tables = _build_metadomain_annotation_quality_tables(adata1)
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "MetaDomain quality plotting skipped: %s",
            exc,
        )
        return None

    domains = tables["domains"]
    transfer_fraction = tables["transfer_fraction"]
    composition = tables["composition"]
    summary = tables["summary"]
    blue = "#4C78A8"
    orange = "#F28E2B"
    grey = "#9D9D9D"

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
        constrained_layout=True,
    )

    axes[0, 0].hist(
        domains["n_metaspots"],
        bins=min(25, max(8, int(np.sqrt(len(domains))))),
        color=blue,
        edgecolor="white",
    )
    median_size = float(domains["n_metaspots"].median())
    axes[0, 0].axvline(
        median_size,
        color=orange,
        linestyle="--",
        label=f"Median={median_size:.0f}",
    )
    axes[0, 0].set_title("MetaSpots per MetaDomain")
    axes[0, 0].set_xlabel("MetaSpots")
    axes[0, 0].set_ylabel("MetaDomains")
    axes[0, 0].legend(frameon=False)

    agreement_labels = ["Exact label", "Broad compartment"]
    agreement_values = 100.0 * np.asarray(
        [
            summary["domain_final_exact_agreement"],
            summary["domain_final_broad_agreement"],
        ],
        dtype=float,
    )
    agreement_bars = axes[0, 1].bar(
        agreement_labels,
        agreement_values,
        color=[blue, orange],
        edgecolor="white",
        width=0.62,
    )
    axes[0, 1].bar_label(
        agreement_bars,
        labels=[f"{value:.1f}%" for value in agreement_values],
        padding=4,
    )
    axes[0, 1].set_ylim(0.0, 100.0)
    axes[0, 1].set_title(
        "Domain evidence retained after MetaSpot update"
    )
    axes[0, 1].set_ylabel("Agreement with final MetaSpot label (%)")
    axes[0, 1].tick_params(axis="x", rotation=12)

    axes[0, 2].hist(
        domains["geometry_purity"],
        bins=np.linspace(0.5, 1.0, 16),
        color=blue,
        edgecolor="white",
    )
    axes[0, 2].axvline(
        summary["weighted_geometry_purity"],
        color=orange,
        linestyle="--",
        label=(
            "MetaSpot-weighted="
            f"{summary['weighted_geometry_purity']:.3f}"
        ),
    )
    axes[0, 2].set_xlim(0.48, 1.02)
    axes[0, 2].set_title(
        "Nuclear/cytoplasmic purity by MetaDomain"
    )
    axes[0, 2].set_xlabel("Within-domain geometry purity")
    axes[0, 2].set_ylabel("MetaDomains")
    axes[0, 2].legend(frameon=False)

    sizes = domains["n_metaspots"].to_numpy(float)
    point_sizes = 25.0 + 155.0 * sizes / max(1.0, sizes.max())
    color_values = domains[
        "embedding_loo_cosine_median"
    ].to_numpy(float)
    finite_color = np.isfinite(color_values)
    color_range = (
        float(np.nanmax(color_values) - np.nanmin(color_values))
        if finite_color.any()
        else np.nan
    )
    if finite_color.any() and color_range > 1e-4:
        fill_value = float(np.nanmedian(color_values))
        scatter = axes[1, 0].scatter(
            domains["geometry_purity"],
            domains["domain_final_broad_agreement"],
            c=np.where(finite_color, color_values, fill_value),
            s=point_sizes,
            cmap="viridis",
            alpha=0.78,
            edgecolor="white",
            linewidth=0.4,
        )
        fig.colorbar(
            scatter,
            ax=axes[1, 0],
            label="Median embedding cosine",
        )
    else:
        axes[1, 0].scatter(
            domains["geometry_purity"],
            domains["domain_final_broad_agreement"],
            s=point_sizes,
            color=blue,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.4,
        )
    axes[1, 0].axvline(0.8, color=grey, linestyle=":")
    axes[1, 0].axhline(0.5, color=grey, linestyle=":")
    axes[1, 0].set_xlim(0.48, 1.02)
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title("Geometry purity and label transfer")
    axes[1, 0].set_xlabel("Within-domain geometry purity")
    axes[1, 0].set_ylabel("Domain-to-final broad-label agreement")

    sns.heatmap(
        transfer_fraction,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annot=(
            transfer_fraction.shape[0] <= 9
            and transfer_fraction.shape[1] <= 9
        ),
        fmt=".0%",
        cbar_kws={"label": "Row fraction"},
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(
        "Domain evidence to final MetaSpot labels"
    )
    axes[1, 1].set_xlabel("Final MetaSpot label")
    axes[1, 1].set_ylabel("MetaDomain evidence label")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].tick_params(axis="y", rotation=0)

    y = np.arange(len(composition))
    height = 0.36
    axes[1, 2].barh(
        y - height / 2,
        100.0 * composition["domain_evidence_fraction"],
        height,
        color=grey,
        label="Domain evidence",
    )
    axes[1, 2].barh(
        y + height / 2,
        100.0 * composition["final_metaspot_fraction"],
        height,
        color=blue,
        label="Final MetaSpot",
    )
    axes[1, 2].set_yticks(y)
    axes[1, 2].set_yticklabels(composition["label"])
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_title("MetaSpot-weighted label composition")
    axes[1, 2].set_xlabel("MetaSpots (%)")
    axes[1, 2].legend(frameon=False)

    fig.suptitle(
        "MetaDomain aggregation and annotation quality"
        f" · {summary['n_metaspots']:,} MetaSpots"
        f" · {summary['n_metadomains']:,} MetaDomains"
        f" · broad-label agreement="
        f"{summary['domain_final_broad_agreement']:.1%}"
        f" · corrected="
        f"{1.0 - summary['domain_final_broad_agreement']:.1%}",
        fontsize=16,
        fontweight="bold",
    )
    _save_fig(fig, out_dir, "metadomain_annotation_quality")
    plt.close(fig)

    _save_dataframe(
        domains.reset_index(),
        out_dir,
        "metadomain_quality_by_domain",
        diagnostic=True,
    )
    _save_dataframe(
        tables["transfer_long"],
        out_dir,
        "metadomain_annotation_transfer",
        diagnostic=True,
    )
    _save_dataframe(
        composition,
        out_dir,
        "metadomain_label_composition",
        diagnostic=True,
    )
    return tables


def _plot_metadomain_subcellular_annotation(
    adata2,
    out_dir: str,
    *,
    label_col: str = "subcellular_annotation",
) -> bool:
    """Save the MetaDomain UMAP using final production annotation colors."""
    if "X_umap" not in adata2.obsm:
        _logging.getLogger("cellscope").warning(
            "MetaDomain annotation UMAP skipped: X_umap is unavailable."
        )
        return False
    if label_col not in adata2.obs.columns:
        _logging.getLogger("cellscope").warning(
            "MetaDomain annotation UMAP skipped: label column %s is unavailable.",
            label_col,
        )
        return False

    labels = (
        adata2.obs[label_col]
        .astype("string")
        .fillna("Unknown")
        .astype(str)
    )
    observed = set(labels.unique())
    categories = [
        label for label in _SUBCELLULAR_LABEL_ORDER if label in observed
    ]
    categories.extend(sorted(observed.difference(categories)))

    coordinates = np.asarray(adata2.obsm["X_umap"])
    if coordinates.ndim != 2 or coordinates.shape != (adata2.n_obs, 2):
        raise ValueError(
            "X_umap must have shape (n_metadomains, 2); "
            f"received {coordinates.shape}."
        )

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    with mpl.rc_context(_PUBLICATION_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        draw_order = [
            "Unknown",
            "Cytosol",
            "Nucleus",
            "ER",
            "OMM",
            "Lamina",
            "Nuclear Pore",
            "Nucleolus",
        ]
        draw_order.extend(
            sorted(set(categories).difference(draw_order))
        )
        handles = {}
        for label in draw_order:
            mask = labels.eq(label).to_numpy()
            if not mask.any():
                continue
            collection = ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=10,
                linewidth=0,
                alpha=0.8,
                rasterized=True,
                color=_SUBCELLULAR_LABEL_PALETTE.get(label, "#7F7F7F"),
                label=label,
            )
            handles[label] = collection

        ax.set_title("MetaDomain Subcellular Annotation")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        nuclear_handles = [
            handles[label]
            for label in _SUBCELLULAR_LABEL_GROUPS["Nuclear side"]
            if label in handles
        ]
        cytoplasmic_handles = [
            handles[label]
            for label in _SUBCELLULAR_LABEL_GROUPS["Cytoplasmic side"]
            if label in handles
        ]
        if nuclear_handles:
            nuclear_legend = ax.legend(
                handles=nuclear_handles,
                title="Nuclear side",
                loc="upper left",
                bbox_to_anchor=(1.01, 0.98),
                borderaxespad=0,
                markerscale=1.7,
            )
            nuclear_legend.get_title().set_fontweight("bold")
            ax.add_artist(nuclear_legend)
        if cytoplasmic_handles:
            cytoplasmic_legend = ax.legend(
                handles=cytoplasmic_handles,
                title="Cytoplasmic side",
                loc="upper left",
                bbox_to_anchor=(1.01, 0.48),
                borderaxespad=0,
                markerscale=1.7,
            )
            cytoplasmic_legend.get_title().set_fontweight("bold")
        if "Unknown" in handles:
            if cytoplasmic_handles:
                ax.add_artist(cytoplasmic_legend)
            unknown_legend = ax.legend(
                handles=[handles["Unknown"]],
                title="Other",
                loc="upper left",
                bbox_to_anchor=(1.01, 0.18),
                borderaxespad=0,
                markerscale=1.7,
            )
            unknown_legend.get_title().set_fontweight("bold")

        fig.subplots_adjust(right=0.74)
        _save_fig(
            fig,
            out_dir,
            "metadomain_leiden_clusters",
            dpi=800,
        )
    plt.close(fig)
    return True


def _normalize_overlaps_nucleus_labels(values: pd.Series) -> pd.Series:
    """Map geometric nucleus membership to explicit nuclear/cytosolic labels."""
    series = pd.Series(values, copy=False)
    codes = np.full(len(series), -1, dtype=np.int8)

    if pd.api.types.is_bool_dtype(series.dtype):
        valid = series.notna().to_numpy()
        truth = series.fillna(False).to_numpy(dtype=bool)
        codes[valid & truth] = 0
        codes[valid & ~truth] = 1
    elif pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="coerce")
        valid = series.notna()
        invalid = valid & ~numeric.isin([0, 1])
        if invalid.any():
            preview = series.loc[invalid].value_counts().head(10).to_dict()
            raise ValueError(
                "overlaps_nucleus must contain only boolean or 0/1 values; "
                f"unrecognized values={preview}"
            )
        codes[numeric.eq(1).to_numpy()] = 0
        codes[numeric.eq(0).to_numpy()] = 1
    else:
        normalized = series.astype("string").str.strip().str.lower()
        nuclear_values = {"true", "1", "1.0", "t", "yes"}
        cytosolic_values = {"false", "0", "0.0", "f", "no"}
        nuclear = normalized.isin(nuclear_values)
        cytosolic = normalized.isin(cytosolic_values)
        invalid = series.notna() & ~(nuclear | cytosolic)
        if invalid.any():
            preview = series.loc[invalid].value_counts().head(10).to_dict()
            raise ValueError(
                "overlaps_nucleus must contain only boolean or 0/1 values; "
                f"unrecognized values={preview}"
            )
        codes[nuclear.to_numpy()] = 0
        codes[cytosolic.to_numpy()] = 1

    categorical = pd.Categorical.from_codes(
        codes,
        categories=["Nucleus", "Cytosol"],
    )
    return pd.Series(categorical, index=series.index, name="label")


def _build_per_cell_label_proportion_tables(
    cell_ids: pd.Series,
    labels: pd.Series,
    *,
    preferred_order: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """Build zero-complete per-cell percentages and their exact summary."""
    cells = pd.Series(cell_ids, copy=False)
    label_values = pd.Series(labels, copy=False)
    valid = cells.notna() & label_values.notna()
    if not valid.any():
        raise ValueError("No transcripts have both a cell ID and a label")

    valid_cells = cells.loc[valid]
    valid_labels = label_values.loc[valid]
    cell_codes, cell_index = pd.factorize(valid_cells, sort=False)
    observed_codes, observed_index = pd.factorize(
        valid_labels,
        sort=False,
    )
    observed_labels = [str(value) for value in observed_index]
    label_order = list(
        dict.fromkeys(
            [str(value) for value in preferred_order] + observed_labels
        )
    )
    if not label_order:
        raise ValueError("No label categories are available for plotting")

    label_position = {label: idx for idx, label in enumerate(label_order)}
    observed_to_order = np.asarray(
        [label_position[label] for label in observed_labels],
        dtype=np.int64,
    )
    ordered_codes = observed_to_order[observed_codes]
    n_cells = len(cell_index)
    n_labels = len(label_order)
    flat_codes = cell_codes.astype(np.int64, copy=False) * n_labels
    flat_codes += ordered_codes
    counts_array = np.bincount(
        flat_codes,
        minlength=n_cells * n_labels,
    ).reshape(n_cells, n_labels)
    counts = pd.DataFrame(
        counts_array,
        index=pd.Index(cell_index, name="cell_id"),
        columns=pd.Index(label_order, name="label"),
    )
    cell_totals = counts.sum(axis=1)
    proportions = counts.div(cell_totals, axis=0).mul(100.0)

    transcript_counts = counts.sum(axis=0)
    total_transcripts = int(transcript_counts.sum())
    cells_with_label = counts.gt(0).sum(axis=0)
    quartiles = proportions.quantile([0.25, 0.75]).T
    summary = pd.DataFrame(
        {
            "label": label_order,
            "n_transcripts": transcript_counts.to_numpy(dtype=np.int64),
            "global_transcript_percentage": (
                transcript_counts.to_numpy(dtype=float)
                / max(1, total_transcripts)
                * 100.0
            ),
            "n_cells_total": n_cells,
            "n_cells_with_label": cells_with_label.to_numpy(dtype=np.int64),
            "cell_detection_percentage": (
                cells_with_label.to_numpy(dtype=float)
                / max(1, n_cells)
                * 100.0
            ),
            "median_cell_percentage": proportions.median(axis=0).to_numpy(),
            "q1_cell_percentage": quartiles[0.25].to_numpy(),
            "q3_cell_percentage": quartiles[0.75].to_numpy(),
            "mean_cell_percentage": proportions.mean(axis=0).to_numpy(),
            "std_cell_percentage": proportions.std(axis=0).to_numpy(),
            "min_cell_percentage": proportions.min(axis=0).to_numpy(),
            "max_cell_percentage": proportions.max(axis=0).to_numpy(),
        }
    )
    return {
        "counts": counts,
        "proportions": proportions,
        "summary": summary,
    }


def _build_ground_truth_nuclear_cytoplasmic_proportions(
    final_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Summarize per-cell geometric nuclear/cytoplasmic transcript shares."""
    required = {"cell_id", "overlaps_nucleus"}
    missing = sorted(required.difference(final_df.columns))
    if missing:
        raise KeyError(
            "Ground-truth proportion plot requires final-data columns: "
            + ", ".join(missing)
        )
    labels = _normalize_overlaps_nucleus_labels(
        final_df["overlaps_nucleus"]
    )
    return _build_per_cell_label_proportion_tables(
        final_df["cell_id"],
        labels,
        preferred_order=["Nucleus", "Cytosol"],
    )


def _build_final_annotation_proportions(
    final_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Summarize per-cell transcript shares for the final production labels."""
    required = {"cell_id", "subcellular_annotation"}
    missing = sorted(required.difference(final_df.columns))
    if missing:
        raise KeyError(
            "Final-annotation proportion plot requires final-data columns: "
            + ", ".join(missing)
        )
    labels = final_df["subcellular_annotation"]
    if isinstance(labels.dtype, pd.CategoricalDtype):
        if "Unknown" not in labels.cat.categories:
            labels = labels.cat.add_categories(["Unknown"])
        labels = labels.fillna("Unknown")
    else:
        labels = labels.where(labels.notna(), "Unknown")
    preferred = [
        label for label in _SUBCELLULAR_LABEL_ORDER if label != "Unknown"
    ]
    return _build_per_cell_label_proportion_tables(
        final_df["cell_id"],
        labels,
        preferred_order=preferred,
    )


def _build_fine_annotation_proportions(
    final_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Summarize seven-class labels; uncertainty lives in state columns."""
    required = {"cell_id", "subcellular_fine_annotation"}
    missing = sorted(required.difference(final_df.columns))
    if missing:
        raise KeyError(
            "Fine-annotation proportion plot requires final-data columns: "
            + ", ".join(missing)
        )
    labels = final_df["subcellular_fine_annotation"]
    if isinstance(labels.dtype, pd.CategoricalDtype):
        if "Unknown" not in labels.cat.categories:
            labels = labels.cat.add_categories(["Unknown"])
        labels = labels.fillna("Unknown")
    else:
        labels = labels.where(labels.notna(), "Unknown")
    preferred = [
        label for label in _SUBCELLULAR_LABEL_ORDER if label != "Unknown"
    ]
    return _build_per_cell_label_proportion_tables(
        final_df["cell_id"],
        labels,
        preferred_order=preferred,
    )


def _plot_per_cell_label_proportions(
    tables: Mapping[str, pd.DataFrame],
    out_dir: str,
    *,
    name: str,
    title: str,
    color: Optional[str] = None,
    palette: Optional[Mapping[str, str]] = None,
) -> None:
    """Save one full-scale per-cell boxplot plus its exact summary table."""
    proportions = tables["proportions"]
    summary = tables["summary"]
    long = (
        proportions.rename_axis(index="cell_id", columns="label")
        .reset_index()
        .melt(
            id_vars="cell_id",
            var_name="label",
            value_name="percentage",
        )
    )
    label_order = proportions.columns.astype(str).tolist()
    summary_by_label = summary.set_index("label").reindex(label_order)
    tick_labels = [
        f"{label}\n{summary_by_label.loc[label, 'cell_detection_percentage']:.1f}% cells"
        for label in label_order
    ]

    fig_width = max(7.0, 1.35 * len(label_order))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))
    try:
        boxplot_kwargs: Dict[str, Any] = {}
        if palette is not None:
            boxplot_kwargs.update(
                hue="label",
                palette={
                    label: palette.get(label, "#9AA0A6")
                    for label in label_order
                },
                legend=False,
            )
        else:
            boxplot_kwargs["color"] = color
        sns.boxplot(
            data=long,
            x="label",
            y="percentage",
            order=label_order,
            width=0.65,
            showfliers=False,
            ax=ax,
            **boxplot_kwargs,
        )
        ax.scatter(
            np.arange(len(label_order)),
            summary_by_label["global_transcript_percentage"].to_numpy(),
            marker="D",
            s=38,
            color="#30343B",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label="Global transcript percentage",
        )
        ax.set_xticks(np.arange(len(label_order)))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        ax.set_xlabel("Label (fraction of cells with >0 transcripts)")
        ax.set_ylabel("Transcripts per cell (%)")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0.0, 100.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.legend(frameon=False, loc="upper right")
        sns.despine(ax=ax)
        fig.tight_layout()
        _save_fig(fig, out_dir, name)
    finally:
        plt.close(fig)

    _save_dataframe(
        summary,
        out_dir,
        name,
        diagnostic=True,
    )


def _save_transcript_annotation_proportion_diagnostics(
    final_df: pd.DataFrame,
    out_dir: str,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Save geometric-truth and final-annotation per-cell proportion plots."""
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    specifications = [
        (
            "ground_truth_nuclear_cytoplasmic_proportions",
            _build_ground_truth_nuclear_cytoplasmic_proportions,
            "Geometric reference: per-cell nuclear/cytoplasmic proportions",
            _SUBCELLULAR_LABEL_PALETTE,
        ),
        (
            "final_annotation_proportions",
            _build_final_annotation_proportions,
            "Final annotation: per-cell subcellular-label proportions",
            _SUBCELLULAR_LABEL_PALETTE,
        ),
    ]
    # ``subcellular_fine_annotation`` is a compatibility alias of the public
    # seven-class ``subcellular_annotation``. Emitting a second identical plot
    # and table suggests two distinct results, so only the canonical final
    # annotation summary is written.
    for name, builder, title, palette in specifications:
        try:
            tables = builder(final_df)
            _plot_per_cell_label_proportions(
                tables,
                out_dir,
                name=name,
                title=title,
                palette=palette,
            )
            results[name] = tables
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "Transcript proportion diagnostic skipped | name=%s | "
                "reason=%s",
                name,
                exc,
                exc_info=True,
            )
    return results


def _build_metaspot_annotation_result_tables(
    adata1: ad.AnnData,
) -> Dict[str, Any]:
    """Summarize the final user-facing annotation at MetaSpot resolution."""
    required = {
        "subcellular_annotation_domain",
        "subcellular_annotation",
    }
    missing = sorted(required.difference(adata1.obs.columns))
    if missing:
        raise KeyError(
            "MetaSpot annotation results require adata1.obs columns: "
            + ", ".join(missing)
        )

    domain_labels = (
        adata1.obs["subcellular_annotation_domain"]
        .astype("string")
        .fillna("Unknown")
        .astype(str)
    )
    compatibility_column = (
        "subcellular_annotation_compatibility"
        if "subcellular_annotation_compatibility" in adata1.obs.columns
        else "subcellular_annotation"
    )
    final_labels = (
        adata1.obs[compatibility_column]
        .astype("string")
        .fillna("Unknown")
        .astype(str)
    )
    changed = domain_labels.to_numpy() != final_labels.to_numpy()
    domain_broad = _metadomain_broad_compartment(domain_labels.to_numpy())
    final_broad = _metadomain_broad_compartment(final_labels.to_numpy())
    broad_changed = changed & (domain_broad != final_broad)

    update = np.full(
        adata1.n_obs,
        "Unchanged",
        dtype=object,
    )
    update[changed & ~broad_changed] = "Subtype/local refinement"
    update[broad_changed] = "Broad-compartment correction"

    spots = pd.DataFrame(
        {
            "domain_annotation": domain_labels.to_numpy(),
            "final_annotation": final_labels.to_numpy(),
            "annotation_update": update,
        },
        index=adata1.obs_names.copy(),
    )
    if "subcellular_fine_annotation" in adata1.obs.columns:
        spots["fine_annotation"] = (
            adata1.obs["subcellular_fine_annotation"]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
            .to_numpy()
        )
    else:
        spots["fine_annotation"] = (
            adata1.obs["subcellular_annotation"]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
            .to_numpy()
        )
    if "subcellular_fine_state" in adata1.obs.columns:
        spots["fine_state"] = (
            adata1.obs["subcellular_fine_state"]
            .astype("string")
            .fillna("Unavailable")
            .astype(str)
            .to_numpy()
        )
    else:
        spots["fine_state"] = "Unavailable"
    if "subcellular_metaspot_decision_provenance" in adata1.obs.columns:
        spots["decision_provenance"] = (
            adata1.obs["subcellular_metaspot_decision_provenance"]
            .astype("string")
            .fillna("unavailable")
            .astype(str)
            .to_numpy()
        )
    else:
        spots["decision_provenance"] = update
    tier_column = (
        "subcellular_annotation_domain_tier"
        if "subcellular_annotation_domain_tier" in adata1.obs.columns
        else "subcellular_annotation_tier"
    )
    if tier_column in adata1.obs.columns:
        spots["domain_evidence_tier"] = (
            adata1.obs[tier_column]
            .astype("string")
            .fillna("Unavailable")
            .astype(str)
            .to_numpy()
        )
    else:
        spots["domain_evidence_tier"] = "Unavailable"
    if "subcellular_annotation_tier" in adata1.obs.columns:
        spots["final_evidence_tier"] = (
            adata1.obs["subcellular_annotation_tier"]
            .astype("string")
            .fillna("Unavailable")
            .astype(str)
            .to_numpy()
        )
    else:
        spots["final_evidence_tier"] = "Unavailable"
    if "cell_id" in adata1.obs.columns:
        spots["cell_id"] = adata1.obs["cell_id"].astype(str).to_numpy()

    label_order = list(
        dict.fromkeys(
            list(_SUBCELLULAR_LABEL_ORDER)
            + domain_labels.value_counts().index.astype(str).tolist()
            + final_labels.value_counts().index.astype(str).tolist()
        )
    )
    domain_counts = domain_labels.value_counts()
    final_counts = final_labels.value_counts()
    rows = []
    for label in label_order:
        domain_n = int(domain_counts.get(label, 0))
        final_n = int(final_counts.get(label, 0))
        if domain_n + final_n == 0:
            continue
        final_mask = final_labels.to_numpy() == label
        row: Dict[str, Any] = {
            "label": label,
            "domain_evidence_n_metaspots": domain_n,
            "domain_evidence_fraction": domain_n / max(1, adata1.n_obs),
            "final_n_metaspots": final_n,
            "final_fraction": final_n / max(1, adata1.n_obs),
            "changed_to_label_n_metaspots": int(
                np.sum(final_mask & changed)
            ),
        }
        if "cell_id" in spots.columns:
            row["final_n_cells"] = int(
                spots.loc[final_mask, "cell_id"].nunique()
            )
        rows.append(row)
    label_summary = pd.DataFrame(rows)

    fine_groups = spots.groupby(
        ["fine_annotation", "fine_state"],
        observed=True,
        dropna=False,
    )
    fine_summary = fine_groups.size().rename("n_metaspots").reset_index()
    if "cell_id" in spots.columns:
        fine_summary["n_cells"] = (
            fine_groups["cell_id"].nunique().to_numpy()
        )
    fine_summary["fraction"] = (
        fine_summary["n_metaspots"] / max(1, adata1.n_obs)
    )

    update_summary = (
        spots["annotation_update"]
        .value_counts()
        .rename_axis("annotation_update")
        .rename("n_metaspots")
        .reset_index()
    )
    update_summary["fraction"] = (
        update_summary["n_metaspots"] / max(1, adata1.n_obs)
    )

    transfer_counts = pd.crosstab(
        spots["domain_annotation"],
        spots["final_annotation"],
    ).reindex(
        index=[
            label
            for label in label_order
            if label in set(spots["domain_annotation"])
        ],
        columns=[
            label
            for label in label_order
            if label in set(spots["final_annotation"])
        ],
        fill_value=0,
    )
    transfer_fraction = transfer_counts.div(
        transfer_counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)
    transfer_long = (
        transfer_counts.rename_axis(
            index="domain_annotation",
            columns="final_annotation",
        )
        .stack()
        .rename("n_metaspots")
        .reset_index()
    )
    transfer_long["row_fraction"] = (
        transfer_fraction.rename_axis(
            index="domain_annotation",
            columns="final_annotation",
        )
        .stack()
        .to_numpy()
    )

    provenance_order = [
        "Unchanged",
        "Subtype/local refinement",
        "Broad-compartment correction",
    ]
    provenance_counts = pd.crosstab(
        spots["final_annotation"],
        spots["annotation_update"],
    ).reindex(
        index=[
            label
            for label in label_order
            if label in set(spots["final_annotation"])
        ],
        columns=provenance_order,
        fill_value=0,
    )
    provenance_fraction = provenance_counts.div(
        provenance_counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)
    provenance_long = (
        provenance_counts.rename_axis(
            index="final_annotation",
            columns="final_annotation_provenance",
        )
        .stack()
        .rename("n_metaspots")
        .reset_index()
    )
    provenance_long["within_final_label_fraction"] = (
        provenance_fraction.rename_axis(
            index="final_annotation",
            columns="final_annotation_provenance",
        )
        .stack()
        .to_numpy()
    )

    preferred_decision_provenance = [
        "broad_prior_fallback",
        "local_exploratory",
        "local_marker_plus_dgi_exploratory",
        "local_strict",
        "local_plus_omm_confirmation_exploratory",
        "local_marker_plus_dgi_plus_omm_confirmation_exploratory",
        "local_plus_omm_confirmation_strict",
        "local_only_strict_omm_confirmation_unidentifiable",
        "local_only_strict_omm_confirmation_unavailable",
    ]
    observed_decision_provenance = (
        spots["decision_provenance"].value_counts().index.astype(str).tolist()
    )
    decision_provenance_order = list(
        dict.fromkeys(
            preferred_decision_provenance + observed_decision_provenance
        )
    )
    decision_counts = pd.crosstab(
        spots["fine_annotation"],
        spots["decision_provenance"],
    ).reindex(
        index=spots["fine_annotation"].value_counts().index.astype(str),
        columns=decision_provenance_order,
        fill_value=0,
    )
    decision_fraction = decision_counts.div(
        decision_counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)
    decision_long = (
        decision_counts.rename_axis(
            index="fine_annotation",
            columns="decision_provenance",
        )
        .stack()
        .rename("n_metaspots")
        .reset_index()
    )
    decision_long["within_fine_label_fraction"] = (
        decision_fraction.rename_axis(
            index="fine_annotation",
            columns="decision_provenance",
        )
        .stack()
        .to_numpy()
    )

    preferred_final_tiers = [
        "Broad-prior fallback",
        "DGI-supported exploratory evidence",
        "Exploratory local evidence",
        "Strict local evidence",
        "Cell-coherent OMM evidence",
        "Cell-coherent fine-structure evidence",
        "Tier-B callable-cell multiscale evidence",
        "Unavailable",
    ]
    observed_final_tiers = (
        spots["final_evidence_tier"].value_counts().index.astype(str).tolist()
    )
    final_tier_order = list(
        dict.fromkeys(preferred_final_tiers + observed_final_tiers)
    )
    final_tier_counts = pd.crosstab(
        spots["fine_annotation"],
        spots["final_evidence_tier"],
    ).reindex(
        index=spots["fine_annotation"].value_counts().index.astype(str),
        columns=final_tier_order,
        fill_value=0,
    )
    final_tier_fraction = final_tier_counts.div(
        final_tier_counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)

    return {
        "spots": spots,
        "label_summary": label_summary,
        "fine_summary": fine_summary,
        "update_summary": update_summary,
        "transfer_counts": transfer_counts,
        "transfer_fraction": transfer_fraction,
        "transfer_long": transfer_long,
        "provenance_counts": provenance_counts,
        "provenance_fraction": provenance_fraction,
        "provenance_long": provenance_long,
        "decision_counts": decision_counts,
        "decision_fraction": decision_fraction,
        "decision_long": decision_long,
        "decision_provenance_order": decision_provenance_order,
        "final_tier_counts": final_tier_counts,
        "final_tier_fraction": final_tier_fraction,
        "final_tier_order": final_tier_order,
        "summary": {
            "n_metaspots": int(adata1.n_obs),
            "n_cells": (
                int(adata1.obs["cell_id"].nunique())
                if "cell_id" in adata1.obs.columns
                else None
            ),
            "changed_fraction": float(np.mean(changed)),
            "broad_compartment_corrected_fraction": float(
                np.mean(broad_changed)
            ),
        },
    }


def _metaspot_plot_representation(
    adata1: ad.AnnData,
    indices: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """Return a bounded MetaSpot representation for result visualization."""
    from sklearn.preprocessing import normalize

    clustering_meta = adata1.uns.get(
        "module5_metadomain_clustering",
        {},
    )
    configured = str(clustering_meta.get("cluster_rep", "")).strip()
    candidates = [
        configured,
        METASPOT_DGI_KEY,
        "X_dgi_gcn",
        METASPOT_DGI_SAGE_KEY,
        "X_dgi_sage",
        "X_expr_metaspot_scaled",
        "X_expr_metaspot",
    ]
    seen = set()
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        if key not in adata1.obsm:
            continue
        values = np.asarray(
            adata1.obsm[key][indices],
            dtype=np.float32,
        )
        if values.ndim != 2 or values.shape[1] < 2:
            continue
        # Historical adata1 files may contain a MetaDomain embedding broadcast
        # back to every member MetaSpot under a generic key such as
        # ``X_dgi_gcn``.  Reusing it would only redraw the MetaDomain UMAP with
        # overlapping points, not show MetaSpot-level structure.
        rounded = np.round(values, decimals=6)
        n_unique = int(np.unique(rounded, axis=0).shape[0])
        n_domains = (
            int(adata1.obs["meta-domain"].astype(str).nunique())
            if "meta-domain" in adata1.obs.columns
            else 0
        )
        broadcast_like = (
            n_unique < max(20, int(math.ceil(0.25 * len(indices))))
            and (
                n_domains == 0
                or n_unique <= max(20, int(math.ceil(1.25 * n_domains)))
            )
        )
        if broadcast_like:
            _logging.getLogger("cellscope").info(
                "Skipping %s for MetaSpot UMAP: only %d unique rows among "
                "%d MetaSpots (MetaDomains=%d), consistent with a "
                "domain-level embedding broadcast.",
                key,
                n_unique,
                len(indices),
                n_domains,
            )
            continue
        values = StandardScaler().fit_transform(values).astype(
            np.float32,
            copy=False,
        )
        values = normalize(values, norm="l2").astype(
            np.float32,
            copy=False,
        )
        return values, key

    expression = adata1.X[indices]
    if sp.issparse(expression):
        expression = expression.astype(np.float32, copy=True)
        expression = normalize(expression, norm="l1")
        expression.data = np.log1p(expression.data * 1.0e4)
    else:
        expression = np.asarray(expression, dtype=np.float32)
        expression = normalize(expression, norm="l1")
        expression = np.log1p(expression * 1.0e4)
    max_components = min(
        50,
        expression.shape[0] - 1,
        expression.shape[1] - 1,
    )
    if max_components < 2:
        dense = (
            expression.toarray()
            if sp.issparse(expression)
            else np.asarray(expression)
        )
        values = np.zeros((expression.shape[0], 2), dtype=np.float32)
        values[:, : min(2, dense.shape[1])] = dense[:, :2]
        return values, "normalized_expression"
    values = TruncatedSVD(
        n_components=max_components,
        random_state=0,
    ).fit_transform(expression)
    values = normalize(values, norm="l2").astype(
        np.float32,
        copy=False,
    )
    return values, "normalized_expression_svd"


def _compute_metaspot_result_umap(
    adata1: ad.AnnData,
    spots: pd.DataFrame,
    *,
    max_points: int = 25_000,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Compute a deterministic, cell/label-stratified MetaSpot UMAP."""
    n_obs = int(adata1.n_obs)
    if n_obs < 3:
        raise ValueError("MetaSpot UMAP requires at least three MetaSpots")

    if n_obs > int(max_points):
        if "cell_id" in spots.columns:
            strata = (
                spots["cell_id"].astype(str)
                + "\x1f"
                + spots["final_annotation"].astype(str)
            ).to_numpy()
        else:
            strata = spots["final_annotation"].astype(str).to_numpy()
        indices = _stratified_metaspot_sample(
            strata,
            int(max_points),
            seed=2026,
        )
    else:
        indices = np.arange(n_obs, dtype=np.int64)

    existing = adata1.obsm.get("X_umap_metaspot")
    if (
        existing is not None
        and np.asarray(existing).shape == (n_obs, 2)
        and np.isfinite(np.asarray(existing)).all()
    ):
        return (
            indices,
            np.asarray(existing, dtype=np.float32)[indices],
            str(
                adata1.uns.get(
                    "metaspot_annotation_visualization",
                    {},
                ).get("representation_key", "stored_X_umap_metaspot")
            ),
        )

    values, representation_key = _metaspot_plot_representation(
        adata1,
        indices,
    )
    import scanpy as sc

    plot_adata = ad.AnnData(
        X=np.zeros((len(indices), 1), dtype=np.float32),
    )
    plot_adata.obsm["X_metaspot_result"] = values
    n_neighbors = min(30, max(2, len(indices) - 1))
    sc.pp.neighbors(
        plot_adata,
        n_neighbors=n_neighbors,
        use_rep="X_metaspot_result",
        metric="cosine",
        random_state=0,
    )
    sc.tl.umap(
        plot_adata,
        min_dist=0.30,
        spread=1.0,
        random_state=0,
    )
    coordinates = np.asarray(
        plot_adata.obsm["X_umap"],
        dtype=np.float32,
    )
    if len(indices) == n_obs:
        adata1.obsm["X_umap_metaspot"] = coordinates
    adata1.uns["metaspot_annotation_visualization"] = {
        "representation_key": representation_key,
        "n_metaspots_total": n_obs,
        "n_metaspots_displayed": int(len(indices)),
        "sampling": (
            "all"
            if len(indices) == n_obs
            else "cell_and_final_label_stratified"
        ),
        "max_points": int(max_points),
        "neighbors": int(n_neighbors),
        "metric": "cosine",
        "random_state": 0,
    }
    return indices, coordinates, representation_key


def _save_rapx_metadomain_evidence_audit_tables(
    adata2: ad.AnnData,
    out_dir: str,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Save RAP-X confirmatory evidence without a misleading label UMAP.

    The strict track is intentionally audit-only: ``Unknown`` means that a
    MetaDomain did not pass confirmatory FDR, not that its final public label
    is unknown.  Exact tables preserve this information more honestly than a
    sparse strict-label scatter plot.
    """
    required = {
        "subcell_core_candidate_label",
        "subcellular_annotation_strict",
        "subcellular_annotation_tier",
    }
    missing = sorted(required.difference(adata2.obs.columns))
    if missing:
        return None

    if "X_umap" in adata2.obsm:
        coordinates = np.asarray(adata2.obsm["X_umap"], dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            coordinates = np.full((adata2.n_obs, 2), np.nan, dtype=np.float32)
    else:
        coordinates = np.full((adata2.n_obs, 2), np.nan, dtype=np.float32)

    evidence = pd.DataFrame(
        {
            "core_candidate": (
                adata2.obs["subcell_core_candidate_label"]
                .astype("string")
                .fillna("Unknown")
                .astype(str)
                .to_numpy()
            ),
            "strict_call": (
                adata2.obs["subcellular_annotation_strict"]
                .astype("string")
                .fillna("Unknown")
                .astype(str)
                .to_numpy()
            ),
            "domain_evidence_tier": (
                adata2.obs["subcellular_annotation_tier"]
                .astype("string")
                .fillna("Unavailable")
                .astype(str)
                .to_numpy()
            ),
            "UMAP1": coordinates[:, 0],
            "UMAP2": coordinates[:, 1],
        },
        index=adata2.obs_names.copy(),
    )
    label_order = [
        label
        for label in _SUBCELLULAR_LABEL_ORDER
        if (
            label in set(evidence["core_candidate"])
            or label in set(evidence["strict_call"])
        )
    ]
    tier_order = ["Strict", "Exploratory", "Candidate", "Unavailable"]

    transition_counts = pd.crosstab(
        evidence["core_candidate"],
        evidence["strict_call"],
    ).reindex(
        index=[
            label
            for label in label_order
            if label in set(evidence["core_candidate"])
        ],
        columns=[
            label
            for label in label_order
            if label in set(evidence["strict_call"])
        ],
        fill_value=0,
    )
    transition_fraction = transition_counts.div(
        transition_counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)
    tier_counts = pd.crosstab(
        evidence["core_candidate"],
        evidence["domain_evidence_tier"],
    ).reindex(
        index=transition_counts.index,
        columns=tier_order,
        fill_value=0,
    )
    tier_fraction = tier_counts.div(
        tier_counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)

    transition_long = (
        transition_counts.rename_axis(
            index="core_candidate",
            columns="strict_call",
        )
        .stack()
        .rename("n_metadomains")
        .reset_index()
    )
    transition_long["within_candidate_fraction"] = (
        transition_fraction.rename_axis(
            index="core_candidate",
            columns="strict_call",
        )
        .stack()
        .to_numpy()
    )
    tier_long = (
        tier_counts.rename_axis(
            index="core_candidate",
            columns="domain_evidence_tier",
        )
        .stack()
        .rename("n_metadomains")
        .reset_index()
    )
    tier_long["within_candidate_fraction"] = (
        tier_fraction.rename_axis(
            index="core_candidate",
            columns="domain_evidence_tier",
        )
        .stack()
        .to_numpy()
    )
    _save_dataframe(
        evidence.reset_index(names="metadomain_id"),
        out_dir,
        "rapx_metadomain_evidence",
        diagnostic=True,
    )
    _save_dataframe(
        transition_long,
        out_dir,
        "rapx_candidate_to_strict",
        diagnostic=True,
    )
    _save_dataframe(
        tier_long,
        out_dir,
        "rapx_candidate_evidence_tier",
        diagnostic=True,
    )
    return {
        "evidence": evidence,
        "transition": transition_long,
        "tier": tier_long,
    }


def _plot_metaspot_annotation_result(
    adata1: ad.AnnData,
    out_dir: str,
) -> Optional[Dict[str, Any]]:
    """Save the final MetaSpot-level annotation result and exact summaries."""
    try:
        tables = _build_metaspot_annotation_result_tables(adata1)
        spots = tables["spots"]
        plot_cfg = load_params_yaml() or {}
        annotation_plot_cfg = plot_cfg.get("annotation", {}) or {}
        max_points = int(
            annotation_plot_cfg.get(
                "metaspot_result_umap_max_points",
                25_000,
            )
        )
        if max_points < 1_000:
            raise ValueError(
                "annotation.metaspot_result_umap_max_points must be >= 1000"
            )
        indices, coordinates, representation_key = (
            _compute_metaspot_result_umap(
                adata1,
                spots,
                max_points=max_points,
            )
        )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "MetaSpot annotation result plotting skipped: %s",
            exc,
            exc_info=True,
        )
        return None

    displayed = spots.iloc[indices].copy()
    displayed["UMAP1"] = coordinates[:, 0]
    displayed["UMAP2"] = coordinates[:, 1]
    label_palette = _SUBCELLULAR_LABEL_PALETTE
    def _ordered_categories(
        values: pd.Series,
        preferred: Sequence[str],
    ) -> List[str]:
        observed = values.astype(str).value_counts().index.tolist()
        return list(
            dict.fromkeys(
                value
                for value in list(preferred) + observed
                if value in observed
            )
        )

    def _scatter_panel(
        ax,
        column: str,
        title: str,
        palette: Mapping[str, str],
        preferred: Sequence[str],
    ) -> None:
        values = displayed[column].astype(str)
        categories = list(dict.fromkeys(_ordered_categories(
            values,
            preferred,
        )))
        extra_colors = sns.color_palette(
            "husl",
            n_colors=max(1, len(categories)),
        )
        point_size = float(
            np.clip(24_000.0 / max(1, len(displayed)), 2.0, 24.0)
        )
        rng = np.random.default_rng(2026)
        for position, category in enumerate(categories):
            member = np.flatnonzero(values.to_numpy() == category)
            rng.shuffle(member)
            ax.scatter(
                displayed["UMAP1"].to_numpy()[member],
                displayed["UMAP2"].to_numpy()[member],
                s=point_size,
                c=[
                    palette.get(
                        category,
                        extra_colors[position % len(extra_colors)],
                    )
                ],
                alpha=0.78,
                linewidths=0,
                rasterized=len(displayed) > 10_000,
                label=category,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(
            frameon=True,
            framealpha=0.90,
            facecolor="white",
            edgecolor="none",
            fontsize="small",
            markerscale=1.8,
            loc="upper right",
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17, 12),
        constrained_layout=False,
    )
    _scatter_panel(
        axes[0, 0],
        "domain_annotation",
        "MetaDomain evidence mapped to MetaSpots",
        label_palette,
        _SUBCELLULAR_LABEL_ORDER,
    )
    _scatter_panel(
        axes[0, 1],
        "fine_annotation",
        "Evidence-qualified fine MetaSpot annotation",
        label_palette,
        _SUBCELLULAR_LABEL_ORDER,
    )

    transfer_counts = tables["transfer_counts"]
    transfer_fraction = tables["transfer_fraction"]
    transfer_annotations = transfer_fraction.copy().astype(object)
    for row in transfer_fraction.index:
        for column in transfer_fraction.columns:
            fraction = float(transfer_fraction.loc[row, column])
            count = int(transfer_counts.loc[row, column])
            transfer_annotations.loc[row, column] = (
                f"{fraction:.0%}\n(n={count})" if count else ""
            )
    sns.heatmap(
        transfer_fraction,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annot=transfer_annotations,
        fmt="",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Fraction within domain-evidence label"},
        ax=axes[1, 0],
    )
    axes[1, 0].set_title(
        "MetaDomain evidence → broad-compatible MetaSpot annotation",
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Final MetaSpot annotation")
    axes[1, 0].set_ylabel("MetaDomain evidence label")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].tick_params(axis="y", rotation=0)

    provenance_fraction = tables["final_tier_fraction"]
    provenance_counts = tables["final_tier_counts"]
    final_labels = provenance_fraction.index.tolist()
    y = np.arange(len(final_labels))
    left = np.zeros(len(final_labels), dtype=float)
    provenance_order = tables["final_tier_order"]
    tier_palette = {
        "Broad-prior fallback": "#BAB0AC",
        "DGI-supported exploratory evidence": "#76B7B2",
        "Exploratory local evidence": "#F28E2B",
        "Strict local evidence": "#4C78A8",
        "Cell-coherent OMM evidence": "#B07AA1",
        "Cell-coherent fine-structure evidence": "#59A14F",
        "Tier-B callable-cell multiscale evidence": "#EDC948",
        "Unavailable": "#D9D9D9",
    }
    for provenance in provenance_order:
        values = (
            100.0
            * provenance_fraction[provenance].to_numpy(dtype=float)
        )
        axes[1, 1].barh(
            y,
            values,
            left=left,
            color=tier_palette.get(provenance, "#BAB0AC"),
            edgecolor="white",
            linewidth=0.5,
            label=provenance,
        )
        counts = provenance_counts[provenance].to_numpy(dtype=int)
        for position, (start, value, count) in enumerate(
            zip(left, values, counts)
        ):
            if value >= 8.0 and count > 0:
                axes[1, 1].text(
                    start + value / 2.0,
                    position,
                    f"{value:.0f}%\n(n={count})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white"
                        if provenance in {
                            "local_strict",
                            "local_plus_omm_confirmation_strict",
                        }
                        else "#222222"
                    ),
                )
        left += values
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels(final_labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(0.0, 100.0)
    axes[1, 1].set_title(
        "Evidence tier within each fine MetaSpot label",
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("MetaSpots within fine label (%)")
    axes[1, 1].set_ylabel("Fine MetaSpot annotation")
    axes[1, 1].legend(
        frameon=False,
        fontsize="small",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
    )

    total = int(tables["summary"]["n_metaspots"])
    shown = int(len(displayed))
    display_note = (
        f"{shown:,}/{total:,} MetaSpots displayed"
        if shown < total
        else f"{total:,} MetaSpots"
    )
    correction_fraction = float(
        tables["summary"]["broad_compartment_corrected_fraction"]
    )
    fig.suptitle(
        "MetaDomain evidence and fine MetaSpot annotation",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.953,
        (
            f"{display_note} · one point per MetaSpot · "
            f"UMAP of {representation_key} · "
            f"broad-compartment corrections={correction_fraction:.1%}"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(
        rect=[0.0, 0.105, 1.0, 0.91],
        h_pad=2.4,
        w_pad=2.0,
    )
    _save_fig(fig, out_dir, "metaspot_annotation_result")
    plt.close(fig)

    coordinate_table = displayed[
        [
            "UMAP1",
            "UMAP2",
            "domain_annotation",
            "final_annotation",
            "fine_annotation",
            "fine_state",
            "decision_provenance",
            "domain_evidence_tier",
            "annotation_update",
        ]
    ].copy()
    metaspot_ids = (
        adata1.obs["metaspot"].astype(str).iloc[indices].to_numpy()
        if "metaspot" in adata1.obs.columns
        else coordinate_table.index.astype(str)
    )
    coordinate_table.insert(
        0,
        "metaspot_id",
        metaspot_ids,
    )
    if "cell_id" in displayed.columns:
        coordinate_table.insert(
            1,
            "cell_id",
            displayed["cell_id"].astype(str).to_numpy(),
        )
    _save_dataframe(
        tables["label_summary"],
        out_dir,
        "metaspot_annotation_summary",
        diagnostic=True,
    )
    _save_dataframe(
        tables["fine_summary"],
        out_dir,
        "metaspot_fine_annotation_summary",
        diagnostic=True,
    )
    _save_dataframe(
        tables["update_summary"],
        out_dir,
        "metaspot_annotation_updates",
        diagnostic=True,
    )
    _save_dataframe(
        tables["transfer_long"],
        out_dir,
        "metaspot_annotation_transfer",
        diagnostic=True,
    )
    _save_dataframe(
        tables["provenance_long"],
        out_dir,
        "metaspot_annotation_provenance_by_final",
        diagnostic=True,
    )
    _save_dataframe(
        tables["decision_long"],
        out_dir,
        "metaspot_fine_decision_provenance",
        diagnostic=True,
    )
    _save_dataframe(
        coordinate_table.reset_index(drop=True),
        out_dir,
        "metaspot_annotation_umap_coordinates",
        diagnostic=True,
    )
    return tables


def _run_pipeline_impl(
    spatial_path: str,
    cell_boundaries_path: str,
    nucleus_boundaries_path: str,
    out_dir: str,
    gold_data: Optional[Union[str, pd.DataFrame]] = None,
    resume: bool = True,
    save_intermediate: Optional[bool] = None,
    lite_intermediate: Optional[bool] = None,
    final_format: Optional[Literal["parquet","csv"]] = None,
    compression: Optional[str] = None,
    resume_from: Optional[Literal[
        "load_inputs",
        "prepare_adata",
        "final_table",
        "write_outputs",
    ]] = None,
    show_internal_progress: bool = False,
    enable_annotation: Optional[bool] = None,
    annotation_method: Optional[str] = None,
    scvi_epochs: Optional[int] = None,
    dgi_epochs: Optional[int] = None,
    enable_dgi_sage: Optional[bool] = None,
    resume_policy: Optional[Literal["auto","minimal","force"]] = None,
    dry_run_diff: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    input_provenance: Optional[Mapping[str, Any]] = None,
    ) -> dict:
    """
    Execute the CellScoper pipeline.

    Returns dict with output file paths.
    """
    ensure_dir(out_dir)
    logger = setup_logger(out_dir)
    logger.info("CellScoper pipeline start | out_dir=%s", out_dir)
    # Load config early to allow overriding some runtime I/O defaults
    cfg = load_params_yaml()
    write_csv_copy = True
    io_cfg: Dict[str, Any] = {}
    try:
        io_cfg = cfg.get('io', {}) if cfg else {}
        # Explicit Python/CLI arguments win. YAML supplies values only when the
        # caller left an argument unset.
        if final_format is None and isinstance(io_cfg.get('final_format'), str):
            final_format = str(io_cfg.get('final_format')).lower()
        if compression is None and isinstance(io_cfg.get('h5ad_compression'), str):
            compression = str(io_cfg.get('h5ad_compression'))
        if 'write_csv_copy' in io_cfg:
            write_csv_copy = bool(io_cfg.get('write_csv_copy'))
        if lite_intermediate is None and 'lite_intermediate' in io_cfg:
            lite_intermediate = bool(io_cfg.get('lite_intermediate'))
        if save_intermediate is None and 'write_intermediate' in io_cfg:
            save_intermediate = bool(io_cfg.get('write_intermediate'))
        elif save_intermediate is None and 'keep_intermediate' in io_cfg:
            ki = bool(io_cfg.get('keep_intermediate'))
            save_intermediate = bool(ki)
        if resume_from is None and isinstance(
            io_cfg.get('resume_from'),
            (str, type(None)),
        ):
            rf = (io_cfg.get('resume_from') or '').strip()
            resume_from = rf if rf else resume_from
        if resume_policy is None and isinstance(io_cfg.get('resume_policy'), str):
            resume_policy = str(io_cfg.get('resume_policy')).lower()
    except Exception as exc:
        logger.warning(
            "Could not resolve runtime I/O configuration: %s",
            exc,
            exc_info=True,
        )
    final_format = str(final_format or "csv").lower()
    if final_format not in {"csv", "parquet"}:
        raise ValueError("final_format must be 'csv' or 'parquet'")
    compression = str(compression or "lzf")
    lite_intermediate = bool(
        False if lite_intermediate is None else lite_intermediate
    )
    save_intermediate = bool(
        True if save_intermediate is None else save_intermediate
    )
    resume_policy = str(resume_policy or "minimal").lower()
    if resume_policy not in {"auto", "minimal", "force"}:
        raise ValueError(
            "resume_policy must be one of: auto, minimal, force"
        )
    try:
        m7_cfg = cfg.get('module7', {}) if cfg else {}
        m8_cfg = cfg.get('module8', {}) if cfg else {}
        if scvi_epochs is None and isinstance(
            m7_cfg.get('scvi_epochs'),
            (int, float),
        ):
            scvi_epochs = int(m7_cfg.get('scvi_epochs'))
        if dgi_epochs is None and isinstance(
            m8_cfg.get('dgi_epochs'),
            (int, float),
        ):
            dgi_epochs = int(m8_cfg.get('dgi_epochs'))
        if enable_dgi_sage is None and 'enable_dgi_sage' in m8_cfg:
            enable_dgi_sage = bool(m8_cfg.get('enable_dgi_sage'))
    except Exception as exc:
        logger.warning("Could not resolve module runtime settings: %s", exc)
    scvi_epochs = int(100 if scvi_epochs is None else scvi_epochs)
    dgi_epochs = int(200 if dgi_epochs is None else dgi_epochs)
    enable_dgi_sage = bool(
        False if enable_dgi_sage is None else enable_dgi_sage
    )
    try:
        runtime_cfg = cfg.get('runtime', {}) if cfg else {}
    except Exception:
        runtime_cfg = {}
    # Annotation toggles/method from YAML (default on, method=rapx)
    try:
        ann_cfg = cfg.get('annotation', {}) if cfg else {}
        if enable_annotation is None and 'enable' in ann_cfg:
            enable_annotation = bool(ann_cfg.get('enable'))
        if annotation_method is None and 'method' in ann_cfg:
            _m = str(ann_cfg.get('method') or "").strip().lower()
            if _m:
                annotation_method = _m
    except Exception as exc:
        logger.warning(
            "Could not resolve annotation runtime settings: %s",
            exc,
        )
    enable_annotation = bool(
        True if enable_annotation is None else enable_annotation
    )
    annotation_method = str(annotation_method or "rapx").lower()
    logger.info(
        "Effective runtime | config=%s | config_sha256=%s | resume=%s | "
        "resume_policy=%s | save_intermediate=%s | lite_intermediate=%s | "
        "final_format=%s | compression=%s | annotation=%s:%s",
        os.environ.get("CELLSCOPE_CONFIG", "(package default)"),
        sha256_mapping(cfg),
        bool(resume),
        resume_policy,
        save_intermediate,
        lite_intermediate,
        final_format,
        compression,
        enable_annotation,
        annotation_method,
    )
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
    # Cap BLAS/OMP threads; allow runtime.blas_threads in params.yaml to take effect.
    # OpenBLAS cannot safely exceed the package's compiled thread-table limit.
    try:
        _requested_blas_threads = (
            _coerce_pos_int(runtime_cfg.get('blas_threads'))
            if runtime_cfg
            else None
        )
        _threaded_outer_parallelism = (
            _joblib_backend().strip().lower() == "threading"
            and _preferred_n_jobs() > 1
        )
        _rt_blas_threads = (
            1
            if _threaded_outer_parallelism
            else (
                min(_requested_blas_threads, _MAX_SAFE_BLAS_THREADS)
                if _requested_blas_threads
                else None
            )
        )
        if _threaded_outer_parallelism and (
            _requested_blas_threads is None or _requested_blas_threads > 1
        ):
            logger.warning(
                "Threaded outer parallelism requires single-threaded BLAS; "
                "using 1 instead of runtime.blas_threads=%s.",
                _requested_blas_threads,
            )
        elif _requested_blas_threads and _requested_blas_threads > _MAX_SAFE_BLAS_THREADS:
            logger.warning(
                "runtime.blas_threads=%d exceeds the OpenBLAS safe limit %d; "
                "using %d.",
                _requested_blas_threads,
                _MAX_SAFE_BLAS_THREADS,
                _rt_blas_threads,
            )
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
    except Exception:
        pass
    logger.info(
        "Numerical threading | backend=%s | outer_jobs=%d | blas=%s | "
        "numba=%s(requested=%s) | scanpy=%s(requested=%s)",
        _joblib_backend(),
        _preferred_n_jobs(),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMBA_NUM_THREADS"),
        runtime_cfg.get("numba_threads"),
        os.environ.get("SCANPY_N_JOBS"),
        runtime_cfg.get("scanpy_n_jobs"),
    )
    # Suppress only stable third-party compatibility notices. Runtime warnings
    # and numerical problems must remain observable.
    try:
        import warnings as _warnings
        _warnings.filterwarnings("ignore", category=FutureWarning, module=r"anndata.*")
        _warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn.utils.deprecation")
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

    # Freeze current inputs, references, and source identity into resume
    # fingerprints so an in-place file or code change cannot reuse stale
    # checkpoints merely because params.yaml is unchanged.
    hash_inputs = bool(io_cfg.get("provenance_hash_inputs", True))
    if input_provenance is None:
        input_provenance = build_input_provenance(
            spatial_path=spatial_path,
            cell_boundaries_path=cell_boundaries_path,
            nucleus_boundaries_path=nucleus_boundaries_path,
            gold_data=gold_data,
            hash_inputs=hash_inputs,
        )
    else:
        input_provenance = dict(input_provenance)
    package_dir = os.path.dirname(__file__)
    reference_provenance = bundled_reference_provenance(
        package_dir,
        hash_content=hash_inputs,
    )
    code_provenance = source_tree_provenance(package_dir)
    raw_input_provenance = {
        key: input_provenance.get(key)
        for key in (
            "spatial",
            "cell_boundaries",
            "nucleus_boundaries",
        )
    }
    build_fingerprint_context = {
        "inputs": raw_input_provenance,
        "code": code_provenance,
    }
    annotation_fingerprint_context = {
        "gold_data": input_provenance.get("gold_data"),
        "references": reference_provenance,
        "code": code_provenance,
    }
    stage_fingerprint_contexts = {
        "load_inputs": build_fingerprint_context,
        "preprocess": build_fingerprint_context,
        "build_anndata": build_fingerprint_context,
        "annotate": annotation_fingerprint_context,
        "write_outputs": {"code": code_provenance},
    }
    for input_name, record in input_provenance.items():
        if not isinstance(record, Mapping):
            logger.info("Input provenance | name=%s | value=%s", input_name, record)
            continue
        digest = str(record.get("sha256", "not-computed"))
        logger.info(
            "Input provenance | name=%s | path=%s | size_bytes=%s | "
            "sha256=%s",
            input_name,
            record.get("path", record.get("kind", "in-memory")),
            record.get("size_bytes", record.get("n_rows")),
            digest,
        )
    logger.info(
        "Source provenance | python_source_sha256=%s | python_files=%s",
        code_provenance.get("python_source_sha256"),
        code_provenance.get("n_python_files"),
    )
    for reference_name, record in reference_provenance.items():
        if isinstance(record, Mapping):
            logger.info(
                "Reference provenance | name=%s | size_bytes=%s | "
                "sha256=%s",
                reference_name,
                record.get("size_bytes"),
                record.get("sha256", "not-computed"),
            )

    # Determine minimal rerun start using config and provenance fingerprints.
    stage_order = ["load_inputs", "preprocess", "build_anndata", "annotate", "write_outputs"]
    start_from = None
    cur_fps = {
        stage: fingerprint_stage(
            cfg,
            stage,
            context=stage_fingerprint_contexts.get(stage),
        )
        for stage in stage_order
    }
    try:
        cur_mod_fps = fingerprint_modules_set(
            cfg,
            context=build_fingerprint_context,
        )
    except Exception as exc:
        logger.warning(
            "Could not build module fingerprints: %s",
            exc,
            exc_info=True,
        )
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
                logger.info(
                    "Resume policy=minimal; configuration/provenance "
                    "fingerprint changed | start_from=%s | previous=%s | "
                    "current=%s",
                    start_from,
                    str(prev_fps.get(start_from, "missing"))[:12],
                    str(cur_fps.get(start_from, "missing"))[:12],
                )
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
        for stage in stage_order:
            logger.info(
                "Dry-run fingerprint | stage=%s | changed=%s | "
                "previous=%s | current=%s",
                stage,
                prev_fps.get(stage) != cur_fps.get(stage),
                str(prev_fps.get(stage, "missing"))[:12],
                str(cur_fps.get(stage, "missing"))[:12],
            )
        return {
            "dry_run": True,
            "changed_stages": changes,
            "start_from": start_from or stage_order[0],
            "state_file": os.path.join(out_dir, "pipeline_state.json"),
        }

    # Resumed runs reuse the same output directory. Remove only exact files
    # retired by the current output contract so stale visuals cannot be
    # mistaken for results from this invocation.
    _remove_deprecated_output_artifacts(out_dir)
    _remove_inactive_method_artifacts(out_dir, cfg)

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
                spatial_path,
                cell_boundaries_path,
                nucleus_boundaries_path,
                out_dir,
                resume,
                save_intermediate,
                # Reaching this branch means the stage must run. Existing raw
                # checkpoints may have a different input fingerprint.
                reuse_cached_inputs=False,
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
            module1_cfg = cfg.get("module1", {}) if cfg else {}
            spatial_df, cell_df, nucleus_df = preprocess(
                spatial_df,
                cell_df,
                nucleus_df,
                show_internal_progress=show_internal_progress,
                use_input_nucleus_flags=bool(
                    module1_cfg.get("use_input_nucleus_flags", True)
                ),
            )
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
                fingerprint_context=build_fingerprint_context,
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
                # Resume needs the transcript-to-MetaSpot/MetaDomain ownership
                # table even in lite mode. Force one compressed Parquet rather
                # than silently making an annotate-only resume impossible.
                _save_dataframe(
                    final_data,
                    out_dir,
                    "module5_final_data",
                    force_parquet=True,
                )
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
    if bool(enable_annotation) and ann_method not in {"rapx", "rap-x", "none"}:
        raise ValueError(
            f"Unsupported annotation method {ann_method!r}; expected 'rapx' "
            "or explicitly disable annotation with method='none'."
        )
    if bool(enable_annotation) and ann_method != "none" and adata2 is None:
        raise RuntimeError(
            "Annotation is enabled but MetaDomain AnnData (adata2) is missing."
        )
    run_annotation = bool(enable_annotation) and (ann_method != "none") and adata2 is not None
    annotation_cache = None
    annotation_cache_ready = False
    if run_annotation and resume_enabled and stage_to_idx["annotate"] < start_idx:
        annotation_cache = _resume_annotation_outputs(out_dir)
        annotation_cache_ready = bool(
            annotation_cache
            and annotation_cache[0] is not None
            and annotation_cache[2] is not None
        )
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
            required_annotation_columns = {
                "subcellular_annotation",
                "subcellular_annotation_domain",
            }
            missing_final_columns = sorted(
                required_annotation_columns.difference(final_data.columns)
            )
            missing_metaspot_columns = sorted(
                required_annotation_columns.difference(adata1.obs.columns)
            )
            if missing_final_columns or missing_metaspot_columns:
                raise RuntimeError(
                    "Annotation completed without required production columns; "
                    f"final_data_missing={missing_final_columns}, "
                    f"adata1_missing={missing_metaspot_columns}"
                )
            # A lite run suppresses the annotated transcript checkpoint. Saving
            # only annotated H5AD files cannot satisfy
            # ``_resume_annotation_outputs`` and merely duplicates the final
            # H5AD write, so keep this checkpoint atomic as a complete set.
            persist_annotation = save_intermediate and not lite_intermediate
            if persist_annotation:
                _save_adata(adata1, out_dir, "annotated_adata1")
                if adata2 is not None:
                    _save_adata(adata2, out_dir, "annotated_adata2")
                _save_dataframe(final_data, out_dir, "annotated_final_csv")
            elif save_intermediate:
                logger.info(
                    "Lite intermediate mode: skipping incomplete annotation "
                    "checkpoint; canonical outputs will be written once"
                )
            _save_state(out_dir, "annotate", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps})
            logger.info("Stage annotate done: final_rows=%d", len(final_data))
        except Exception as exc:
            logger.exception(
                "Stage annotate failed; refusing to write a successful "
                "production output"
            )
            raise RuntimeError(
                f"Annotation stage failed: {exc}"
            ) from exc
    elif run_annotation and annotation_cache_ready and annotation_cache is not None:
        logger.info("Resume: skipping annotate (checkpoint available)")
        ann_adata1, ann_adata2, ann_final = annotation_cache
        if ann_adata1 is not None:
            adata1 = ann_adata1
        if ann_adata2 is not None:
            adata2 = ann_adata2
        final_data = ann_final
        required_annotation_columns = {
            "subcellular_annotation",
            "subcellular_annotation_domain",
        }
        missing_cached_columns = sorted(
            required_annotation_columns.difference(final_data.columns)
        )
        if missing_cached_columns:
            raise RuntimeError(
                "Cached annotation output is incomplete; missing "
                f"{missing_cached_columns}. Rerun annotation."
            )
        missing_cached_metaspot_columns = sorted(
            required_annotation_columns.difference(adata1.obs.columns)
        )
        if missing_cached_metaspot_columns:
            raise RuntimeError(
                "Cached MetaSpot annotation output is incomplete; missing "
                f"{missing_cached_metaspot_columns}. Rerun annotation."
            )
    else:
        logger.info(
            "Annotation disabled explicitly | enable=%s | method=%s",
            bool(enable_annotation),
            ann_method,
        )

    if adata1 is None or final_data is None:
        raise RuntimeError("Cannot write outputs: missing 'adata1' or 'final_data'. Please check prior stages or checkpoints.")
    adata1 = cast(ad.AnnData, adata1)
    final_data = cast(pd.DataFrame, final_data)

    # Step 3: Write outputs
    adata1_path = os.path.join(out_dir, "adata1.h5ad")
    adata2_path = os.path.join(out_dir, "adata2.h5ad")
    final_parquet_path = os.path.join(out_dir, "final_data.parquet")
    final_csv_path = os.path.join(out_dir, "final_data.csv")
    final_dataset_path = os.path.join(out_dir, "final_data_dataset")

    # Optionally resume directly to writing outputs
    if resume_from == "write_outputs":
        pass

    # Write files

    if progress_callback: progress_callback("Writing outputs")
    written_outputs: Dict[str, Optional[str]] = {
        "adata1": None,
        "adata2": None,
        "final_parquet": None,
        "final_csv": None,
        "final_dataset": None,
        "stale_final_parquet": None,
    }
    try:
        # Compatibility for anndata<0.11 + pandas StringArray/categorical writes.
        # First sanitize obs columns, then opt-in to nullable-string writing when available.
        adata1 = _sanitize_obs_for_h5ad(adata1)
        try:
            _write_cfg = load_params_yaml() or {}
            _drop_final_h5ad_embeddings = bool(
                (_write_cfg.get("io", {}) or {}).get(
                    "drop_embeddings",
                    False,
                )
            )
        except Exception:
            _drop_final_h5ad_embeddings = False
        try:
            import anndata as _ad
            _ad.settings.allow_write_nullable_strings = True
        except Exception:
            pass
        with _compact_final_adata1_for_write(
            adata1,
            enabled=_drop_final_h5ad_embeddings,
        ):
            _atomic_write_adata(
                adata1,
                adata1_path,
                compression=compression,
            )
        written_outputs["adata1"] = adata1_path
        if adata2 is not None:
            adata2 = _sanitize_obs_for_h5ad(adata2)
            _atomic_write_adata(
                adata2,
                adata2_path,
                compression=compression,
            )
            written_outputs["adata2"] = adata2_path
        if final_format == "csv":
            _atomic_dataframe_write(
                final_data,
                final_csv_path,
                file_format="csv",
            )
            written_outputs["final_csv"] = final_csv_path
            try:
                _atomic_dataframe_write(
                    final_data,
                    final_parquet_path,
                    file_format="parquet",
                )
                written_outputs["final_parquet"] = final_parquet_path
            except Exception as exc:
                stale_path = _quarantine_stale_file(final_parquet_path)
                written_outputs["stale_final_parquet"] = stale_path
                logger.warning(
                    "Optional Parquet copy was not written; CSV production "
                    "output remains valid | stale_parquet=%s | error=%s",
                    stale_path,
                    exc,
                )
        else:  # parquet chosen
            try:
                _atomic_dataframe_write(
                    final_data,
                    final_parquet_path,
                    file_format="parquet",
                )
                written_outputs["final_parquet"] = final_parquet_path
                if write_csv_copy:
                    try:
                        _atomic_dataframe_write(
                            final_data,
                            final_csv_path,
                            file_format="csv",
                        )
                        written_outputs["final_csv"] = final_csv_path
                    except Exception as exc:
                        logger.warning(
                            "Optional CSV copy was not written; Parquet "
                            "production output remains valid | error=%s",
                            exc,
                        )
                elif os.path.isfile(final_csv_path):
                    # Do not leave a stale CSV from an older run beside the
                    # newly written canonical Parquet output.
                    os.unlink(final_csv_path)
                    logger.info(
                        "Removed stale optional CSV output: %s",
                        final_csv_path,
                    )
            except Exception as exc:
                logger.warning(
                    "Parquet production write failed; falling back to CSV | "
                    "error=%s",
                    exc,
                    exc_info=True,
                )
                _atomic_dataframe_write(
                    final_data,
                    final_csv_path,
                    file_format="csv",
                )
                written_outputs["final_csv"] = final_csv_path
                stale_path = _quarantine_stale_file(final_parquet_path)
                written_outputs["stale_final_parquet"] = stale_path
                if stale_path:
                    logger.warning(
                        "Quarantined stale Parquet output: %s",
                        stale_path,
                    )
        execution_plan = adata1.uns.get("execution_plan", {})
        write_partitioned_dataset = bool(
            io_cfg.get("write_partitioned_dataset", False)
        )
        if (
            bool(execution_plan.get("partitioned_annotation", False))
            and write_partitioned_dataset
        ):
            partition_manifest = _atomic_partitioned_parquet_write(
                final_data,
                final_dataset_path,
                target_rows_per_part=int(
                    execution_plan.get("partition_rows", 1_000_000)
                ),
            )
            written_outputs["final_dataset"] = final_dataset_path
            logger.info(
                "Streaming final dataset written | path=%s | parts=%d",
                final_dataset_path,
                int(partition_manifest["n_parts"]),
            )
        _save_state(out_dir, "write_outputs", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps})
        logger.info(
            "Stage write_outputs done | adata1=%s | adata2=%s | "
            "final_parquet=%s | final_csv=%s | final_dataset=%s",
            written_outputs["adata1"],
            written_outputs["adata2"],
            written_outputs["final_parquet"],
            written_outputs["final_csv"],
            written_outputs["final_dataset"],
        )
    except Exception:
        logger.exception("Stage write_outputs failed")
        raise

    counts = {
        "n_transcripts": int(len(final_data)),
        "n_cells": (
            int(final_data["cell_id"].nunique())
            if "cell_id" in final_data.columns
            else None
        ),
        "n_metaspots": int(adata1.n_obs),
        "n_metadomains": (
            int(adata2.n_obs) if adata2 is not None else None
        ),
    }
    output_guide_path = _write_output_guide(
        out_dir,
        counts=counts,
        annotation_enabled=bool(run_annotation),
        primary_data_product=str(
            (
                written_outputs["final_csv"]
                if final_format == "csv"
                else written_outputs["final_parquet"]
            )
            or written_outputs["final_csv"]
        ),
    )
    written_outputs["output_guide"] = output_guide_path
    logger.info(
        "Pipeline finished successfully | transcripts=%s | cells=%s | "
        "metaspots=%s | metadomains=%s",
        counts["n_transcripts"],
        counts["n_cells"],
        counts["n_metaspots"],
        counts["n_metadomains"],
    )
    return {**written_outputs, "counts": counts}


@functools.wraps(_run_pipeline_impl)
def run_pipeline(*args, **kwargs) -> dict:
    """Run one isolated pipeline invocation.

    Module helpers use temporary process-global I/O switches while a run is
    active. Always restore them so failed runs and notebook dry-runs cannot
    change the behavior of a later invocation in the same Python process.
    """
    global _WRITE_INTERMEDIATE, _LITE_INTERMEDIATE, _WRITE_FIGURES
    previous = (
        _WRITE_INTERMEDIATE,
        _LITE_INTERMEDIATE,
        _WRITE_FIGURES,
    )
    try:
        return _run_pipeline_impl(*args, **kwargs)
    finally:
        (
            _WRITE_INTERMEDIATE,
            _LITE_INTERMEDIATE,
            _WRITE_FIGURES,
        ) = previous


# ====== Optional K1/K2: Clustering + RNALocate/APEX-based annotation ======

def _try_import_scanpy():
    try:
        __import__("scanpy")
        return True
    except Exception:
        return False


def _annotation_neighbors_key_for_rep(rep_name: str) -> str:
    rep = str(rep_name).strip()
    if rep == "X_pca":
        return "pca_nbrs"
    if rep == "X_dgi_gcn":
        return "gcn_nbrs"
    if rep == "X_dgi_sage":
        return "sage_nbrs"
    return f"{rep[2:]}_nbrs" if rep.startswith("X_") else rep


def _choose_annotation_neighbors_key(
    adata2: ad.AnnData,
    cluster_rep: Optional[str],
    rep_priority: Optional[Sequence[str]],
    require_cluster_rep: bool = False,
) -> str:
    """Resolve the annotation graph without overriding an explicit choice."""
    if rep_priority is None:
        rep_priority = ["X_dgi_gcn", "X_dgi_sage", "X_expr", "X_pca"]

    if cluster_rep and str(cluster_rep).strip().lower() != "auto":
        explicit = _annotation_neighbors_key_for_rep(str(cluster_rep))
        if f"{explicit}_connectivities" in adata2.obsp:
            return explicit
        if require_cluster_rep:
            raise KeyError(
                "Requested annotation.cluster_rep="
                f"{cluster_rep!r} has no neighbor graph"
            )
        _logging.getLogger("cellscope").warning(
            "Requested annotation.cluster_rep=%s has no neighbor graph; "
            "falling back to rep_priority.",
            cluster_rep,
        )

    for rep in rep_priority:
        candidate = _annotation_neighbors_key_for_rep(str(rep))
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
    require_cluster_rep: bool = False,
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
    # PCA runs on the highly-variable subset only, and the arpack solver
    # requires n_components < min(n_samples, n_features). Clamp to the
    # actual HVG count and n_obs - 1 so small MetaDomain sets (or few HVGs)
    # do not raise ValueError.
    _n_hvg = (
        int(adata2.var["highly_variable"].sum())
        if "highly_variable" in adata2.var.columns
        else adata2.n_vars
    )
    _pca_comps = int(max(1, min(_pca_comps, _n_hvg, adata2.n_obs - 1)))
    try:
        sc.pp.pca(adata2, mask_var="highly_variable", n_comps=_pca_comps)
    except Exception:
        sc.pp.pca(adata2, use_highly_variable=True, n_comps=_pca_comps)
    sc.pp.neighbors(adata2, use_rep='X_pca', key_added='pca_nbrs', n_neighbors=int(n_neighbors))

    # Also build neighbors on available learned reps if present
    reps = list(
        dict.fromkeys(
            [
                *(
                    [str(cluster_rep)]
                    if cluster_rep
                    and str(cluster_rep).strip().lower() != "auto"
                    else []
                ),
                *([str(rep) for rep in rep_priority] if rep_priority else []),
                "X_expr",
                "X_dgi_sage",
                "X_dgi_gcn",
                METADOMAIN_DGI_KEY,
                "X_expr_metadomain",
            ]
        )
    )
    for rep in reps:
        if rep in adata2.obsm and rep != "X_pca":
            try:
                key_std = _annotation_neighbors_key_for_rep(rep)
                sc.pp.neighbors(adata2, use_rep=rep, key_added=key_std, n_neighbors=int(n_neighbors))
            except Exception:
                pass

    # Pick priority for Leiden
    # Prefer explicit cluster_rep from config if available and corresponding neighbors exist
    try:
        if cluster_rep is None:
            _cfg = load_params_yaml()
            cluster_rep = resolve_annotation_configuration(_cfg).get(
                "cluster_rep"
            )
    except Exception:
        pass
    chosen_key = _choose_annotation_neighbors_key(
        adata2,
        cluster_rep=cluster_rep,
        rep_priority=rep_priority,
        require_cluster_rep=require_cluster_rep,
    )
    adata2.uns["annotation_neighbors_key"] = chosen_key
    adata2.uns["annotation_neighbors_rep"] = (
        str(cluster_rep)
        if cluster_rep and str(cluster_rep).strip().lower() != "auto"
        else "priority_fallback"
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
    # Compute the optional UMAP from the chosen neighbors.  The production
    # annotation colors are not available until RAP-X completes, so the named
    # figure is saved later instead of coloring it with Leiden IDs here.
    try:
        # Read resolved annotation plotting switches.
        _cfg = load_params_yaml()
        ann = resolve_annotation_configuration(_cfg)
        _compute_umap = bool(ann.get('compute_umap', False))
        _save_umap_plot = bool(ann.get('save_umap_plot', False))
        if _compute_umap or _save_umap_plot:
            import scanpy as sc
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
                adata2.uns["annotation_umap_neighbors_key"] = str(
                    _nbrs_for_umap
                )
                adata2.uns["annotation_umap_representation"] = (
                    str(adata2.uns.get("annotation_neighbors_rep"))
                    if _nbrs_for_umap == chosen_key
                    else f"neighbor graph {_nbrs_for_umap}"
                )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Annotation UMAP generation failed: %s",
            exc,
            exc_info=True,
        )
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
        path = bundled_resource_path("manual_marker.csv")
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

    updates: Dict[str, Any] = {}
    if produced_label_col not in adata.obs.columns:
        updates[produced_label_col] = _pd.Series(
            [_pd.NA] * adata.n_obs,
            dtype="object",
            index=adata.obs_names,
        )
    else:
        updates[produced_label_col] = adata.obs[produced_label_col].astype(
            "object"
        )

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
            updates[tgt] = _pd.Series(
                [fill] * adata.n_obs,
                dtype=want,
                index=adata.obs_names,
            )
    _replace_obs_columns(adata, updates)


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
    _replace_obs_columns(
        adata,
        {
            "leiden-dominant-subcellular": (
                adata.obs["leiden"].map(dominant).astype(str)
            ),
            "cluster-subcellular": leiden + ":" + labels,
        },
    )


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


def _apply_independent_evidence_tracks(
    annot_df: pd.DataFrame,
    regions: Sequence[str],
    *,
    group_r_mean: Optional[pd.Series] = None,
    **policy_kwargs,
) -> pd.DataFrame:
    """Return compartment-aware decisions plus an expression-only audit track.

    The independent label policy is evaluated first without geometry so the
    exact expression-only counterfactual is never lost.  When ``group_r_mean``
    is supplied, a second evaluation restricts candidates to the broad
    nuclear/cytoplasmic side while retaining the same seven-label,
    single-stage evidence policy.
    """
    expression_only = _apply_independent_evidence_policy(
        annot_df,
        regions,
        group_r_mean=None,
        **policy_kwargs,
    )
    if group_r_mean is None:
        selected = expression_only.copy()
        feasibility_applied = False
    else:
        selected = _apply_independent_evidence_policy(
            annot_df,
            regions,
            group_r_mean=group_r_mean,
            **policy_kwargs,
        )
        feasibility_applied = True

    audit_columns = pd.DataFrame(
        {
            "expression_only_label": (
                expression_only["label"].astype(str)
            ),
            "expression_only_candidate_label": (
                expression_only["candidate_label"].astype(str)
            ),
            "domain_compartment_feasibility_applied": bool(
                feasibility_applied
            ),
        },
        index=selected.index,
    )
    selected = selected.drop(
        columns=audit_columns.columns,
        errors="ignore",
    )
    return pd.concat([selected, audit_columns], axis=1)


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


def _metaspot_compartment_guard_values(
    obs: pd.DataFrame,
) -> Optional[pd.Series]:
    """Return the signed majority-side statistic for each MetaSpot.

    ``r_p50`` is the median signed transcript location, so its sign follows
    the majority nuclear/cytoplasmic side of a MetaSpot.  Do not fall back to
    ``r_mean``: a minority of transcripts far from the boundary can dominate
    the mean distance and reverse the local majority.
    """
    if "r_p50" not in obs.columns:
        return None
    return pd.to_numeric(obs["r_p50"], errors="coerce")


def _map_metaspot_labels_to_transcripts(
    adata: ad.AnnData,
    final_df: pd.DataFrame,
    label_col: str = "subcellular_annotation",
) -> pd.Series:
    """Broadcast one primary MetaSpot label to all member transcripts."""
    if label_col not in adata.obs.columns:
        raise KeyError(f"Missing MetaSpot label column: {label_col}")
    if "metaspot" not in final_df.columns:
        raise KeyError("Transcript table is missing the metaspot column")

    if "metaspot" in adata.obs.columns:
        metaspot_ids = adata.obs["metaspot"].astype(str)
    else:
        metaspot_ids = pd.Series(
            adata.obs_names.astype(str),
            index=adata.obs_names,
        )
    label_by_metaspot = pd.Series(
        adata.obs[label_col].astype(str).to_numpy(),
        index=pd.Index(metaspot_ids.to_numpy(), dtype=str),
        dtype="object",
    )
    if not label_by_metaspot.index.is_unique:
        raise ValueError(
            "Primary annotation mapping requires unique MetaSpot IDs"
        )

    mapped = final_df["metaspot"].astype(str).map(label_by_metaspot)
    if mapped.isna().any():
        raise ValueError(
            "Failed to map a primary annotation to every transcript"
        )
    return mapped


def _metaspot_transcript_row_indexer(
    adata: ad.AnnData,
    final_df: pd.DataFrame,
) -> np.ndarray:
    """Resolve transcript MetaSpots to AnnData rows exactly once."""
    if "metaspot" not in final_df.columns:
        raise KeyError("Transcript table is missing the metaspot column")
    if "metaspot" in adata.obs.columns:
        metaspot_ids = pd.Index(adata.obs["metaspot"])
    else:
        metaspot_ids = pd.Index(adata.obs_names)
    return _key_row_indexer(
        metaspot_ids,
        final_df["metaspot"],
        error_context="primary annotation",
    )


def _key_row_indexer(
    source_ids: pd.Index,
    target_ids: pd.Series,
    *,
    error_context: str,
) -> np.ndarray:
    """Resolve string keys once and retain compact integer row codes."""
    source_ids = pd.Index(source_ids)
    if not source_ids.is_unique:
        raise ValueError(f"{error_context} mapping requires unique source IDs")
    # MetaSpot/MetaDomain columns are normally already string or categorical.
    # Try their native compact representation first; converting hundreds of
    # millions of exact string keys to Python objects just to hash them again
    # is both slower and substantially more memory-intensive.
    row_indexer = source_ids.get_indexer(target_ids)
    if np.any(row_indexer < 0):
        string_source_ids = pd.Index(source_ids.astype(str))
        if not string_source_ids.is_unique:
            raise ValueError(
                f"{error_context} mapping requires unique string source IDs"
            )
        row_indexer = string_source_ids.get_indexer(target_ids.astype(str))
    if np.any(row_indexer < 0):
        raise ValueError(f"Failed to map {error_context} to every target row")
    if len(source_ids) <= np.iinfo(np.int32).max:
        row_indexer = row_indexer.astype(np.int32, copy=False)
    return row_indexer


def _take_obs_column_by_row_indexer(
    values: pd.Series,
    row_indexer: np.ndarray,
    target_index: pd.Index,
) -> pd.Series:
    """Take one obs column while retaining compact categorical codes."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        categorical = values.array
        taken_codes = np.take(categorical.codes, row_indexer)
        taken = pd.Categorical.from_codes(
            taken_codes,
            categories=categorical.categories,
            ordered=categorical.ordered,
        )
        return pd.Series(taken, index=target_index, name=values.name)
    taken = np.take(values.to_numpy(), row_indexer)
    return pd.Series(taken, index=target_index, name=values.name)


def _compose_omm_metaspot_labels(
    domain_labels: pd.Series,
    refined_states: pd.Series,
    domain_gates: pd.Series,
    competitor_labels: Optional[pd.Series] = None,
    reference_posteriors: Optional[pd.Series] = None,
    *,
    identifiable: bool,
    ambiguous_policy: str = "explicit",
    decision_policy: str = "strict",
    posterior_high: float = 0.80,
) -> pd.Series:
    """Compose the primary MetaSpot track from domain and local OMM evidence.

    ``strict`` requires a locally high OMM posterior inside a domain-confirmed
    gate. ``balanced`` also lets a locally high posterior resolve a domain-level
    ER/OMM shared state to OMM. The pipeline's ``competitor`` ambiguity policy
    keeps unresolved spots in the seven-class primary label space.

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
    decision = str(decision_policy).strip().lower()
    if decision not in {"strict", "balanced"}:
        raise ValueError(
            "decision_policy must be one of {'strict', 'balanced'}"
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

    if decision == "balanced":
        if reference_posteriors is None:
            raise ValueError(
                "balanced OMM decision_policy requires reference_posteriors"
            )
        posterior = pd.to_numeric(
            pd.Series(reference_posteriors, index=labels.index),
            errors="coerce",
        )
        locally_high = posterior.ge(float(posterior_high))
        supported = (confirmed | shared) & locally_high
        labels.loc[supported] = "OMM"

        unresolved = (confirmed | shared) & ~supported
        locally_rejected = unresolved & states.eq("OMM-not-supported")
        labels.loc[locally_rejected] = competitors.loc[locally_rejected]
        _assign_ambiguous(unresolved & ~locally_rejected)

        unsupported_omm = labels.eq("OMM") & ~supported
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

    print("\n=== Annotation Quality Summary ===")
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
    independent_domain_compartment_feasibility=False,
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
        annotation_cfg = resolve_annotation_configuration(
            load_params_yaml() or {}
        )
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
        independent_domain_compartment_feasibility = bool(
            annotation_cfg.get(
                "independent_domain_compartment_feasibility",
                independent_domain_compartment_feasibility,
            )
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
        _replace_obs_columns(
            adata,
            {
                "compartment_pred": np.where(
                    adata.obs["r_mean"] < 0,
                    "Nuclear",
                    "Cytoplasmic",
                )
            },
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
        _replace_obs_columns(
            adata,
            {
                "subcellular_candidate": (
                    adata.obs["subcellular_candidate"].astype("object")
                    if "subcellular_candidate" in adata.obs.columns
                    else pd.Series(
                        pd.NA,
                        index=adata.obs_names,
                        dtype="object",
                    )
                ),
                "subcell_margin": (
                    pd.to_numeric(
                        adata.obs["subcell_margin"],
                        errors="coerce",
                    ).astype("float32")
                    if "subcell_margin" in adata.obs.columns
                    else pd.Series(
                        np.nan,
                        index=adata.obs_names,
                        dtype="float32",
                    )
                ),
            },
        )
        assigned_labels = adata.obs[produced_label_col].astype(object).to_numpy(copy=True)
        conservative_labels = np.full(adata.n_obs, pd.NA, dtype=object)
        confidence_levels = np.full(adata.n_obs, pd.NA, dtype=object)
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

        produced_values = pd.Series(
            assigned_labels,
            index=adata.obs_names,
            dtype="object",
        )
        cats = pd.unique(produced_values.dropna().astype(str))
        two_stage_updates: Dict[str, Any] = {
            produced_label_col: pd.Categorical(
                produced_values.astype(str),
                categories=sorted(cats),
                ordered=False,
            ),
            "subcellular_annotation_conservative": pd.Categorical(
                pd.Series(conservative_labels, index=adata.obs_names).astype(str)
            ),
            "subcellular_confidence": pd.Categorical(
                pd.Series(confidence_levels, index=adata.obs_names).astype(str),
                categories=["Low", "Medium", "High"],
                ordered=True,
            ),
        }
        if (
            "subcell_strict_label" in adata.obs.columns
            and adata.obs["subcell_strict_label"].notna().any()
        ):
            two_stage_updates["subcellular_annotation_strict"] = pd.Categorical(
                adata.obs["subcell_strict_label"].astype(str)
            )
        if (
            "subcell_exploratory_label" in adata.obs.columns
            and adata.obs["subcell_exploratory_label"].notna().any()
        ):
            two_stage_updates[
                "subcellular_annotation_exploratory"
            ] = pd.Categorical(
                adata.obs["subcell_exploratory_label"].astype(str)
            )
        if (
            "subcell_annotation_tier" in adata.obs.columns
            and adata.obs["subcell_annotation_tier"].notna().any()
        ):
            two_stage_updates["subcellular_annotation_tier"] = pd.Categorical(
                adata.obs["subcell_annotation_tier"].astype(str),
                categories=["Strict", "Exploratory", "Candidate"],
                ordered=True,
            )
        _replace_obs_columns(adata, two_stage_updates)

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
            group_r_mean = None
            if (
                independent_domain_compartment_feasibility
                and use_spatial_channel
                and "r_mean" in adata.obs.columns
            ):
                group_r_mean = _annotation_group_mean(adata, "r_mean")
            elif independent_domain_compartment_feasibility:
                _logging.getLogger("cellscope").warning(
                    "Independent MetaDomain compartment feasibility was "
                    "requested but spatial r_mean is unavailable; using the "
                    "expression-only decision"
                )
            _logging.getLogger("cellscope").info(
                "Independent single-stage annotation | "
                "domain_compartment_feasibility=%s | "
                "expression_only_track_retained=true",
                group_r_mean is not None,
            )
            annot_df = _apply_independent_evidence_tracks(
                annot_df,
                regions_ref,
                group_r_mean=group_r_mean,
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

        obs_updates: Dict[str, Any] = {
            produced_label_col: (
                adata.obs["leiden"].map(annot_df["label"]).astype("category")
            ),
            "subcellular_candidate": (
                adata.obs["leiden"]
                .map(annot_df["candidate_label"])
                .astype("category")
            ),
            "subcellular_annotation_conservative": (
                adata.obs["leiden"]
                .map(annot_df["conservative_label"])
                .astype("category")
            ),
            "subcellular_confidence": pd.Categorical(
                adata.obs["leiden"].map(annot_df["confidence_level"]),
                categories=["Low", "Medium", "High"],
                ordered=True,
            ),
        }
        if "expression_only_label" in annot_df.columns:
            obs_updates["subcellular_annotation_expression_only"] = (
                adata.obs["leiden"]
                .map(annot_df["expression_only_label"])
                .astype("category")
            )
        if "expression_only_candidate_label" in annot_df.columns:
            obs_updates["subcellular_candidate_expression_only"] = (
                adata.obs["leiden"]
                .map(annot_df["expression_only_candidate_label"])
                .astype("category")
            )
        if "strict_label" in annot_df.columns:
            obs_updates["subcellular_annotation_strict"] = (
                adata.obs["leiden"]
                .map(annot_df["strict_label"])
                .astype("category")
            )
        if "exploratory_label" in annot_df.columns:
            obs_updates["subcellular_annotation_exploratory"] = (
                adata.obs["leiden"]
                .map(annot_df["exploratory_label"])
                .astype("category")
            )
        if "annotation_tier" in annot_df.columns:
            obs_updates["subcellular_annotation_tier"] = pd.Categorical(
                adata.obs["leiden"].map(annot_df["annotation_tier"]),
                categories=["Strict", "Exploratory", "Candidate"],
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
            'expression_only_label','expression_only_candidate_label',
            'domain_compartment_feasibility_applied',
            'omm_raw_competitor_label','omm_raw_competitor_score',
            'omm_specificity_margin','omm_specificity_pass',
            'omm_domain_state','membrane_ambiguity',
        ]:
            if col in annot_df.columns:
                obs_updates[f"subcell_{col}"] = adata.obs["leiden"].map(
                    annot_df[col]
                )

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
                    obs_updates[f"subcell_{source}"] = (
                        adata.obs["leiden"].map(annot_df[source])
                    )

        if "margin" in annot_df.columns:
            obs_updates["subcell_margin"] = pd.to_numeric(
                adata.obs['leiden'].map(annot_df["margin"]), errors="coerce"
            ).astype("float32").values

        _replace_obs_columns(adata, obs_updates)
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
            'independent_domain_compartment_feasibility': bool(
                independent_domain_compartment_feasibility
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
            plot_updates: Dict[str, Any] = {}
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
                        plot_updates[new_c] = col.astype('float32')
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
                    categorical = pd.Categorical(col.astype(str))
                    plot_updates[c] = categorical
                    category_labels = [
                        str(value) for value in categorical.categories
                    ]
                    if category_labels and all(
                        label in _SUBCELLULAR_LABEL_PALETTE
                        for label in category_labels
                    ):
                        adata.uns[f"{c}_colors"] = [
                            _SUBCELLULAR_LABEL_PALETTE[label]
                            for label in category_labels
                        ]
                out.append(c)
            _replace_obs_columns(adata, plot_updates)
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
    apex = bundled_resource_path("2019_CELL_APEXSeq.tsv")
    rna = bundled_resource_path("RNALocateV3.txt")
    if not os.path.exists(apex) or not os.path.exists(rna):
        raise FileNotFoundError(
            "Bundled APEX/RNALocate reference files are missing from the "
            "CellScoper installation"
        )
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


def _stable_group_layout(
    group_codes: np.ndarray,
    n_groups: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable member order, offsets, and sizes for dense group codes."""
    codes = np.asarray(group_codes)
    if codes.ndim != 1:
        raise ValueError("group_codes must be one-dimensional")
    if n_groups < 0:
        raise ValueError("n_groups must be non-negative")
    if codes.size:
        if np.any(codes < 0) or int(codes.max()) >= int(n_groups):
            raise ValueError(
                "group_codes must be in the dense range [0, n_groups)"
            )

    member_order = np.argsort(codes, kind="stable")
    if codes.size <= np.iinfo(np.int32).max:
        member_order = member_order.astype(np.int32, copy=False)
    group_sizes = np.bincount(
        codes.astype(np.int64, copy=False),
        minlength=int(n_groups),
    ).astype(np.int64, copy=False)
    group_offsets = np.empty(int(n_groups) + 1, dtype=np.int64)
    group_offsets[0] = 0
    np.cumsum(group_sizes, out=group_offsets[1:])
    return member_order, group_offsets, group_sizes


def _unique_cell_counts_for_masks(
    cell_ids: Sequence[object],
    masks: Sequence[Sequence[bool]],
) -> Tuple[int, List[int]]:
    """Count unique cells under several masks using compact integer codes."""
    cell_codes, unique_cells = pd.factorize(cell_ids, sort=False)
    if len(unique_cells) <= np.iinfo(np.int32).max:
        cell_codes = cell_codes.astype(np.int32, copy=False)
    valid_cells = cell_codes >= 0
    counts: List[int] = []
    for mask in masks:
        mask_values = np.asarray(mask, dtype=bool)
        if mask_values.shape != cell_codes.shape:
            raise ValueError("Cell coverage mask length does not match cell IDs")
        selected_codes = cell_codes[valid_cells & mask_values]
        if selected_codes.size == 0:
            counts.append(0)
            continue
        occupied = np.bincount(
            selected_codes,
            minlength=len(unique_cells),
        )
        counts.append(int(np.count_nonzero(occupied)))
    return int(len(unique_cells)), counts


def _dense_group_codes(
    group_ids: Sequence[object],
    *,
    error_context: str,
) -> Tuple[np.ndarray, int]:
    """Return dense non-negative codes, reusing an existing code array."""
    values = np.asarray(group_ids)
    if values.ndim != 1:
        raise ValueError(f"{error_context} IDs must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.int32), 0
    if np.issubdtype(values.dtype, np.integer):
        minimum = int(values.min())
        maximum = int(values.max())
        if minimum >= 0 and maximum < values.size:
            dtype = np.int32 if maximum <= np.iinfo(np.int32).max else np.int64
            return values.astype(dtype, copy=False), maximum + 1

    codes, unique_groups = pd.factorize(group_ids, sort=False)
    if np.any(codes < 0):
        raise ValueError(f"{error_context} IDs contain missing values")
    if len(unique_groups) <= np.iinfo(np.int32).max:
        codes = codes.astype(np.int32, copy=False)
    return codes, int(len(unique_groups))


def _high_confidence_cell_coverage(
    cell_ids: Sequence[object],
    labels: Sequence[object],
    confidence: Sequence[object],
) -> Tuple[int, pd.Series]:
    """Return exact label-wise cell coverage without object-table deduping."""
    label_codes, unique_labels = pd.factorize(labels, sort=False)
    high = np.asarray(pd.Series(confidence).eq("High"), dtype=bool)
    masks = [
        high & (label_codes == label_index)
        for label_index in range(len(unique_labels))
    ]
    total_cells, counts = _unique_cell_counts_for_masks(cell_ids, masks)
    coverage = pd.Series(
        counts,
        index=pd.Index(unique_labels).astype(str),
        dtype=np.int64,
    ).sort_values(ascending=False)
    return total_cells, coverage


def _build_same_cell_metaspot_dgi_pool(
    adata1: ad.AnnData,
    broad_prior: Sequence[object],
    *,
    rep_key: str = METASPOT_DGI_KEY,
    n_neighbors: int = 8,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Return an unweighted DGI-neighborhood pooling matrix.

    Neighbors are selected independently inside each original cell and broad
    compartment.  L2-normalizing the DGI vectors makes Euclidean KDTree
    neighbors equivalent to cosine-nearest neighbors.  The diagonal is always
    present, so a small compartment in a cell safely pools only with itself.
    """
    if "cell_id" not in adata1.obs.columns:
        raise KeyError(
            "MetaSpot annotation.unit='metaspot' requires adata1.obs['cell_id']"
        )
    if rep_key not in adata1.obsm:
        raise KeyError(
            "MetaSpot annotation.unit='metaspot' requires the native MetaSpot "
            f"DGI representation adata1.obsm[{rep_key!r}]. Set "
            "module8.dgi_unit: metaspot and rerun build_anndata."
        )
    try:
        k = int(n_neighbors)
    except (TypeError, ValueError) as exc:
        raise ValueError("annotation.metaspot_pool_k must be an integer") from exc
    if k < 1:
        raise ValueError("annotation.metaspot_pool_k must be >= 1")

    logger = _logging.getLogger("cellscope")
    started_at = _time.perf_counter()
    logger.info(
        "MetaSpot same-cell DGI pooling start | metaspots=%d | neighbors=%d",
        int(adata1.n_obs),
        int(k),
    )

    embedding = adata1.obsm[rep_key]
    if sp.issparse(embedding):
        embedding = embedding.toarray()
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[0] != adata1.n_obs:
        raise ValueError(
            f"adata1.obsm[{rep_key!r}] must have shape "
            f"({adata1.n_obs}, n_features)"
        )
    embedding = np.nan_to_num(
        embedding,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=True,
    )
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding /= np.maximum(norms, 1e-8)

    broad = pd.Series(broad_prior, index=adata1.obs_names, dtype="string")
    if len(broad) != adata1.n_obs:
        raise ValueError("broad_prior length must match adata1.n_obs")
    cells = adata1.obs["cell_id"].astype("string").fillna("<missing-cell>")
    broad = broad.fillna("Unknown")
    group_index = pd.MultiIndex.from_arrays(
        [cells.to_numpy(), broad.to_numpy()],
        names=["cell_id", "broad_prior"],
    )
    group_codes, group_levels = pd.factorize(group_index, sort=False)
    n_groups = len(group_levels)
    logger.info(
        "MetaSpot same-cell DGI pooling grouped | groups=%d | "
        "elapsed_seconds=%.1f",
        int(n_groups),
        _time.perf_counter() - started_at,
    )

    # Build group membership once. The previous implementation evaluated
    # ``group_codes == group_code`` for every group, repeatedly scanning all
    # MetaSpots and making the work proportional to n_obs * n_groups.
    member_order, group_offsets, group_sizes = _stable_group_layout(
        group_codes,
        n_groups,
    )
    poolable_group = np.fromiter(
        (
            str(level[1]) in {"Nuclear", "Cytoplasmic"}
            for level in group_levels
        ),
        dtype=bool,
        count=n_groups,
    )
    group_neighbor_counts = np.where(
        poolable_group,
        np.minimum(group_sizes, int(k)),
        1,
    ).astype(np.int32, copy=False)
    row_neighbor_counts = group_neighbor_counts[group_codes]

    indptr = np.empty(adata1.n_obs + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_neighbor_counts, dtype=np.int64, out=indptr[1:])
    n_edges = int(indptr[-1])
    index_dtype = (
        np.int32
        if adata1.n_obs <= np.iinfo(np.int32).max
        else np.int64
    )
    indices = np.empty(n_edges, dtype=index_dtype)

    progress_every = max(1, n_groups // 10)
    for group_code, level in enumerate(group_levels):
        start = int(group_offsets[group_code])
        stop = int(group_offsets[group_code + 1])
        members = member_order[start:stop]
        group_size = int(members.size)
        local_k = int(group_neighbor_counts[group_code])
        positions = (
            indptr[members, None]
            + np.arange(local_k, dtype=np.int64)[None, :]
        ).reshape(-1)

        group_broad = str(level[1])
        if group_broad not in {"Nuclear", "Cytoplasmic"}:
            neighbor_ids = members
        elif group_size <= k:
            # Querying every point in a group returns the full group. Filling
            # it directly avoids constructing a KDTree for small groups.
            neighbor_ids = np.tile(members, group_size)
        else:
            tree = KDTree(embedding[members])
            local_indices = np.asarray(
                tree.query(
                    embedding[members],
                    k=local_k,
                    return_distance=False,
                ),
                dtype=np.int64,
            )
            local_self = np.arange(group_size, dtype=np.int64)
            missing_self = ~np.any(
                local_indices == local_self[:, None],
                axis=1,
            )
            if missing_self.any():
                local_indices[missing_self, -1] = local_self[missing_self]
            neighbor_ids = members[local_indices.reshape(-1)]
        indices[positions] = neighbor_ids

        processed = group_code + 1
        if processed % progress_every == 0 or processed == n_groups:
            logger.info(
                "MetaSpot same-cell DGI pooling progress | groups=%d/%d "
                "| elapsed_seconds=%.1f",
                int(processed),
                int(n_groups),
                _time.perf_counter() - started_at,
            )

    data = np.ones(n_edges, dtype=np.float32)
    pooling = sp.csr_matrix(
        (data, indices, indptr),
        shape=(adata1.n_obs, adata1.n_obs),
        dtype=np.float32,
    )
    pooling.sum_duplicates()
    pooling.data[:] = 1.0
    pool_size = row_neighbor_counts.astype(np.int32, copy=False)
    if np.any(pool_size < 1):
        raise RuntimeError("Every MetaSpot must include itself in local pooling")
    logger.info(
        "MetaSpot same-cell DGI pooling done | metaspots=%d | groups=%d "
        "| edges=%d | elapsed_seconds=%.1f",
        int(adata1.n_obs),
        int(n_groups),
        int(pooling.nnz),
        _time.perf_counter() - started_at,
    )
    return pooling, pool_size


def _prepare_metaspot_marker_sets(
    gold_df: pd.DataFrame,
    var_names: Sequence[object],
    annotation_cfg: Mapping[str, Any],
) -> Dict[str, List[str]]:
    """Build exclusive, measured marker sets for local MetaSpot evidence."""
    gold_mat = prepare_gold_matrix(
        gold_df,
        normalize_columns=bool(
            annotation_cfg.get("gold_column_normalize", False)
        ),
    )
    marker_sets, _, _ = build_specific_marker_sets(
        gold_mat,
        target=int(annotation_cfg.get("marker_set_target", 600)),
        min_weight=float(annotation_cfg.get("marker_min_weight", 0.05)),
        min_margin=float(annotation_cfg.get("marker_min_margin", 0.05)),
    )
    measured = {str(gene).upper() for gene in var_names}
    labels = [
        "Nucleus",
        "Nucleolus",
        "Lamina",
        "Nuclear Pore",
        "Cytosol",
        "ER",
        "OMM",
    ]
    return {
        label: sorted(
            {
                str(gene).upper()
                for gene in marker_sets.get(label, [])
                if str(gene).upper() in measured
            }
        )
        for label in labels
    }


def _resolve_metaspot_omm_confirmation(
    adata1: ad.AnnData,
    *,
    require_confirmation: bool,
    posterior_high: float,
) -> Tuple[str, np.ndarray]:
    """Resolve global OMM identifiability and per-MetaSpot confirmation.

    Returns a global mode (``disabled``, ``identifiable``,
    ``unidentifiable``, or ``unavailable``) and a boolean confirmation mask.
    ``unavailable`` is distinct from a negative confirmation: it means the
    independent confirmation model or its required output columns are absent.
    """
    n_obs = adata1.n_obs
    if not require_confirmation:
        return "disabled", np.ones(n_obs, dtype=bool)

    refinement_meta = adata1.uns.get("omm_metaspot_refinement")
    identifiable: Optional[bool] = None
    if (
        isinstance(refinement_meta, Mapping)
        and "identifiable" in refinement_meta
    ):
        raw_identifiable = refinement_meta.get("identifiable")
        if isinstance(raw_identifiable, str):
            normalized = raw_identifiable.strip().lower()
            if normalized in {"true", "1", "yes"}:
                identifiable = True
            elif normalized in {"false", "0", "no"}:
                identifiable = False
        else:
            try:
                raw_array = np.asarray(raw_identifiable)
                if raw_array.size == 1:
                    identifiable = bool(raw_array.reshape(-1)[0])
            except (TypeError, ValueError):
                identifiable = None

    required_columns = {"omm_domain_gate", "omm_refined_state"}
    if identifiable is False:
        return "unidentifiable", np.zeros(n_obs, dtype=bool)
    if identifiable is not True or not required_columns.issubset(
        adata1.obs.columns
    ):
        return "unavailable", np.zeros(n_obs, dtype=bool)

    gates = (
        adata1.obs["omm_domain_gate"]
        .astype("string")
        .fillna("")
        .str.lower()
    )
    states = adata1.obs["omm_refined_state"].astype("string").fillna("")
    confirmed_state = gates.eq("confirmed") & states.eq("OMM-supported")
    posterior_supported = pd.Series(False, index=adata1.obs_names)
    if "omm_reference_posterior" in adata1.obs.columns:
        posterior = pd.to_numeric(
            adata1.obs["omm_reference_posterior"],
            errors="coerce",
        )
        posterior_supported = (
            gates.isin(["confirmed", "shared"])
            & posterior.ge(float(posterior_high))
        )
    confirmed = (confirmed_state | posterior_supported).to_numpy(dtype=bool)
    return "identifiable", confirmed


def _resolve_metaspot_label_thresholds(
    annotation_cfg: Mapping[str, Any],
    labels: Sequence[str],
    defaults: Mapping[str, Any],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """Resolve validated per-label overrides for the MetaSpot fine head."""
    raw_overrides = annotation_cfg.get("metaspot_label_thresholds", {}) or {}
    if not isinstance(raw_overrides, Mapping):
        raise ValueError(
            "annotation.metaspot_label_thresholds must be a mapping"
        )

    integer_fields = {
        "min_marker_transcripts",
        "min_marker_genes",
    }
    probability_fields = {
        "subtype_fdr",
        "exploratory_max_q",
        "min_dgi_support",
        "min_dgi_margin",
    }
    allowed_fields = set(defaults)
    label_lookup = {str(label).casefold(): str(label) for label in labels}
    normalized_overrides: Dict[str, Mapping[str, Any]] = {}
    for raw_label, raw_values in raw_overrides.items():
        normalized_label = label_lookup.get(str(raw_label).strip().casefold())
        if normalized_label is None:
            raise ValueError(
                "annotation.metaspot_label_thresholds contains an unknown "
                f"label: {raw_label!r}"
            )
        if not isinstance(raw_values, Mapping):
            raise ValueError(
                "Each annotation.metaspot_label_thresholds entry must be "
                f"a mapping; got {raw_label!r}"
            )
        unknown_fields = sorted(set(raw_values).difference(allowed_fields))
        if unknown_fields:
            raise ValueError(
                "Unknown MetaSpot label threshold fields for "
                f"{normalized_label!r}: {unknown_fields}"
            )
        normalized_overrides[normalized_label] = raw_values

    arrays: Dict[str, np.ndarray] = {}
    resolved_by_label: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        resolved_by_label[str(label)] = {}
    for field, default in defaults.items():
        values: List[Any] = []
        for label in labels:
            raw_value = normalized_overrides.get(str(label), {}).get(
                field,
                default,
            )
            if field in integer_fields:
                try:
                    value: Any = int(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"MetaSpot threshold {field!r} for {label!r} must "
                        "be an integer"
                    ) from exc
                if value < 1:
                    raise ValueError(
                        f"MetaSpot threshold {field!r} for {label!r} must "
                        "be >= 1"
                    )
            else:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"MetaSpot threshold {field!r} for {label!r} must "
                        "be numeric"
                    ) from exc
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"MetaSpot threshold {field!r} for {label!r} must "
                        "be finite and >= 0"
                    )
                if field in probability_fields and value > 1.0:
                    raise ValueError(
                        f"MetaSpot threshold {field!r} for {label!r} must "
                        "be <= 1"
                    )
            values.append(value)
            resolved_by_label[str(label)][field] = value
        dtype = np.int32 if field in integer_fields else np.float64
        arrays[field] = np.asarray(values, dtype=dtype)

    for label_idx, label in enumerate(labels):
        if (
            arrays["exploratory_max_q"][label_idx]
            < arrays["subtype_fdr"][label_idx]
        ):
            raise ValueError(
                "MetaSpot exploratory_max_q must be >= subtype_fdr for "
                f"{label!r}"
            )
    return arrays, resolved_by_label


def _compute_metaspot_dgi_consensus(
    pooling: sp.csr_matrix,
    raw_scores: np.ndarray,
    raw_counts: np.ndarray,
    raw_distinct_genes: np.ndarray,
    broad_values: np.ndarray,
    labels: Sequence[str],
    allowed: Mapping[str, Sequence[str]],
    *,
    seed_min_transcripts: int,
    seed_min_genes: int,
    seed_min_enrichment: float,
    min_informative_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate marker-only subtype seeds over leave-one-out DGI edges.

    The seed labels are computed from unpooled counts.  Removing the diagonal
    prevents a MetaSpot from manufacturing DGI support for its own marker
    candidate.  This makes the returned consensus an independently auditable
    representation branch rather than a second name for pooled counts.
    """
    n_obs, n_labels = raw_scores.shape
    label_to_idx = {str(label): idx for idx, label in enumerate(labels)}
    subtype_labels = {
        str(label)
        for label in labels
        if str(label) not in {"Nucleus", "Cytosol", "Unknown"}
    }
    marker_candidate_indices = np.full(n_obs, -1, dtype=np.int32)
    marker_candidate_gap = np.full(n_obs, np.nan, dtype=np.float64)
    for compartment, compartment_labels in allowed.items():
        rows = np.flatnonzero(broad_values == compartment)
        label_indices = np.asarray(
            [label_to_idx[str(label)] for label in compartment_labels],
            dtype=np.int64,
        )
        if rows.size == 0 or label_indices.size == 0:
            continue
        local_scores = raw_scores[np.ix_(rows, label_indices)]
        best_local = np.argmax(local_scores, axis=1)
        best_scores = local_scores[np.arange(rows.size), best_local]
        finite_best = np.isfinite(best_scores)
        chosen = label_indices[best_local]
        marker_candidate_indices[rows[finite_best]] = chosen[finite_best]
        if label_indices.size > 1:
            sorted_scores = np.sort(local_scores, axis=1)
            gaps = sorted_scores[:, -1] - sorted_scores[:, -2]
            gaps[~np.isfinite(gaps)] = np.nan
            marker_candidate_gap[rows] = gaps

    row_ids = np.arange(n_obs)
    safe_indices = np.maximum(marker_candidate_indices, 0)
    candidate_counts = raw_counts[row_ids, safe_indices]
    candidate_genes = raw_distinct_genes[row_ids, safe_indices]
    candidate_scores = raw_scores[row_ids, safe_indices]
    label_values = np.asarray(labels, dtype=object)
    marker_candidate_labels = np.full(n_obs, "Unknown", dtype=object)
    valid_candidate = marker_candidate_indices >= 0
    marker_candidate_labels[valid_candidate] = label_values[
        marker_candidate_indices[valid_candidate]
    ]
    seed_mask = (
        np.isin(marker_candidate_labels, list(subtype_labels))
        & (candidate_counts >= int(seed_min_transcripts))
        & (candidate_genes >= int(seed_min_genes))
        & (candidate_scores >= float(seed_min_enrichment))
        & np.isfinite(candidate_scores)
    )

    seed_onehot = np.zeros((n_obs, n_labels), dtype=np.float32)
    seed_rows = np.flatnonzero(seed_mask)
    if seed_rows.size:
        seed_onehot[
            seed_rows,
            marker_candidate_indices[seed_rows],
        ] = 1.0

    # The pooling contract guarantees a unit diagonal. Subtracting the focal
    # seed after multiplication is the exact leave-one-out result and avoids a
    # second multi-gigabyte CSR copy solely to clear the diagonal.
    votes = np.asarray(pooling @ seed_onehot, dtype=np.float32)
    votes -= seed_onehot
    seed_values = seed_mask.astype(np.float32)
    informative = np.asarray(pooling @ seed_values).ravel() - seed_values
    support = np.zeros((n_obs, n_labels), dtype=np.float32)
    available = informative >= int(min_informative_neighbors)
    support[available] = votes[available] / informative[available, None]
    return (
        marker_candidate_indices,
        marker_candidate_gap,
        support,
        informative.astype(np.int32),
    )


def _select_metaspot_cell_coverage_rescues(
    cell_ids: np.ndarray,
    candidate: np.ndarray,
    already_supported: np.ndarray,
    marker_base_evidence: np.ndarray,
    dgi_available: np.ndarray,
    candidate_dgi_support: np.ndarray,
    candidate_dgi_margin: np.ndarray,
    candidate_scores: np.ndarray,
    score_gap: np.ndarray,
    candidate_q: np.ndarray,
    candidate_counts: np.ndarray,
    cell_marker_counts: np.ndarray,
    cell_marker_genes: np.ndarray,
    cell_marker_enrichment: np.ndarray,
    cell_marker_binomial_p: np.ndarray,
    *,
    labels: Sequence[str],
    min_cell_transcripts: int,
    min_cell_genes: int,
    min_cell_enrichment: float,
    max_cell_p: float,
    min_dgi_support: float,
    min_dgi_margin: float,
    max_per_cell_label: int,
) -> np.ndarray:
    """Select bounded, evidence-backed fine labels missing from each cell.

    This is a discovery rescue, not a forced seven-label assignment. A label
    is eligible only when the cell has coherent multi-transcript, multi-gene
    evidence for it, the focal MetaSpot passes its own marker gates, and an
    independent leave-one-out DGI neighborhood supports that local placement.
    At most ``max_per_cell_label`` MetaSpots are selected for a label that is
    currently absent from a cell.
    """
    n_obs = len(candidate)
    rescued = np.zeros(n_obs, dtype=bool)
    if n_obs == 0 or max_per_cell_label <= 0 or not labels:
        return rescued

    cell_values, n_cells = _dense_group_codes(
        cell_ids,
        error_context="MetaSpot cell-coverage rescue",
    )
    candidate_values = np.asarray(candidate, dtype=object)
    supported_values = np.asarray(already_supported, dtype=bool)
    eligible_base = (
        np.asarray(marker_base_evidence, dtype=bool)
        & np.asarray(dgi_available, dtype=bool)
        & np.isfinite(candidate_dgi_support)
        & np.isfinite(candidate_dgi_margin)
        & (candidate_dgi_support >= float(min_dgi_support))
        & (candidate_dgi_margin >= float(min_dgi_margin))
        & (np.asarray(cell_marker_counts, dtype=float) >= int(min_cell_transcripts))
        & (np.asarray(cell_marker_genes, dtype=float) >= int(min_cell_genes))
        & (
            np.asarray(cell_marker_enrichment, dtype=float)
            >= float(min_cell_enrichment)
        )
        & np.isfinite(np.asarray(cell_marker_binomial_p, dtype=float))
        & (
            np.asarray(cell_marker_binomial_p, dtype=float)
            <= float(max_cell_p)
        )
        & ~supported_values
    )

    for label in labels:
        label_value = str(label)
        present_by_cell = np.bincount(
            cell_values[
                supported_values & (candidate_values == label_value)
            ],
            minlength=n_cells,
        ).astype(bool, copy=False)
        eligible_rows = np.flatnonzero(
            eligible_base
            & (candidate_values == label_value)
            & ~present_by_cell[cell_values]
        )
        if eligible_rows.size == 0:
            continue

        ranked = pd.DataFrame(
            {
                "row": eligible_rows,
                "cell_id": cell_values[eligible_rows],
                "dgi_support": candidate_dgi_support[eligible_rows],
                "dgi_margin": candidate_dgi_margin[eligible_rows],
                "enrichment": candidate_scores[eligible_rows],
                "score_gap": score_gap[eligible_rows],
                "q": candidate_q[eligible_rows],
                "marker_transcripts": candidate_counts[eligible_rows],
                "cell_marker_transcripts": np.asarray(cell_marker_counts)[
                    eligible_rows
                ],
                "cell_marker_genes": np.asarray(cell_marker_genes)[
                    eligible_rows
                ],
                "cell_marker_enrichment": np.asarray(
                    cell_marker_enrichment
                )[eligible_rows],
                "cell_marker_binomial_p": np.asarray(
                    cell_marker_binomial_p
                )[eligible_rows],
            }
        )
        ranked.sort_values(
            [
                "cell_id",
                "dgi_support",
                "dgi_margin",
                "enrichment",
                "score_gap",
                "q",
                "marker_transcripts",
                "cell_marker_enrichment",
                "cell_marker_transcripts",
                "cell_marker_genes",
                "cell_marker_binomial_p",
                "row",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            kind="mergesort",
            inplace=True,
        )
        selected = ranked.groupby("cell_id", sort=False).head(
            int(max_per_cell_label)
        )
        rescued[selected["row"].to_numpy(dtype=np.int64)] = True
    return rescued


def _select_metaspot_labelwise_cell_coverage_rescues(
    cell_ids: np.ndarray,
    broad_values: np.ndarray,
    current_labels: np.ndarray,
    already_supported: np.ndarray,
    local_counts: np.ndarray,
    local_genes: np.ndarray,
    local_enrichment: np.ndarray,
    local_q: np.ndarray,
    dgi_available: np.ndarray,
    dgi_support: np.ndarray,
    cell_marker_counts: np.ndarray,
    cell_marker_genes: np.ndarray,
    cell_marker_enrichment: np.ndarray,
    cell_marker_binomial_p: np.ndarray,
    *,
    labels: Sequence[str],
    all_labels: Sequence[str],
    allowed: Mapping[str, Sequence[str]],
    min_local_transcripts: np.ndarray,
    min_local_genes: np.ndarray,
    min_local_enrichment: np.ndarray,
    min_cell_transcripts: int,
    min_cell_genes: int,
    min_cell_enrichment: float,
    max_cell_p: float,
    min_dgi_support: float,
    min_dgi_margin: float,
    max_per_cell_label: int,
) -> np.ndarray:
    """Return independently supported fine labels for bounded rescue rows.

    The ordinary MetaSpot head first chooses one label with the largest local
    marker enrichment. That winner-takes-all step is useful for a primary
    candidate, but it can hide a second fine structure even when that label
    has its own focal marker, coherent cell-level multi-gene evidence, and
    leave-one-out DGI neighborhood support. This helper evaluates each allowed
    rescue label against its own evidence arrays before candidate competition.

    No label is invented from DGI alone. Every selected row must contain a
    measured marker for the emitted label, the cell must pass exact pooled
    marker gates in the matching broad compartment, and the label must tie or
    beat every same-compartment label in the independent DGI vote. Selection
    is bounded per cell/label and a MetaSpot can receive at most one label.
    """
    n_obs = len(cell_ids)
    rescue_labels = np.full(n_obs, "", dtype=object)
    if n_obs == 0 or int(max_per_cell_label) <= 0 or not labels:
        return rescue_labels

    matrices = {
        "local_counts": np.asarray(local_counts),
        "local_genes": np.asarray(local_genes),
        "local_enrichment": np.asarray(local_enrichment),
        "local_q": np.asarray(local_q),
        "dgi_support": np.asarray(dgi_support),
        "cell_marker_counts": np.asarray(cell_marker_counts),
        "cell_marker_genes": np.asarray(cell_marker_genes),
        "cell_marker_enrichment": np.asarray(cell_marker_enrichment),
        "cell_marker_binomial_p": np.asarray(cell_marker_binomial_p),
    }
    n_labels = len(all_labels)
    for name, matrix in matrices.items():
        if matrix.shape != (n_obs, n_labels):
            raise ValueError(
                f"{name} must have shape {(n_obs, n_labels)}, got "
                f"{matrix.shape}"
            )

    cell_values, n_cells = _dense_group_codes(
        cell_ids,
        error_context="MetaSpot labelwise cell-coverage rescue",
    )
    broad = np.asarray(broad_values, dtype=object)
    current = np.asarray(current_labels, dtype=object)
    supported = np.asarray(already_supported, dtype=bool)
    available = np.asarray(dgi_available, dtype=bool)
    if any(
        array.shape != (n_obs,)
        for array in (broad, current, supported, available)
    ):
        raise ValueError(
            "MetaSpot labelwise rescue row arrays must be one-dimensional"
        )

    label_to_idx = {
        str(label): idx for idx, label in enumerate(all_labels)
    }
    label_to_compartment = {
        str(label): str(compartment)
        for compartment, compartment_labels in allowed.items()
        for label in compartment_labels
    }
    rows: List[pd.DataFrame] = []
    for raw_label in labels:
        label = str(raw_label)
        if label not in label_to_idx or label not in label_to_compartment:
            raise ValueError(
                f"Unknown labelwise cell-coverage rescue label: {label!r}"
            )
        label_idx = label_to_idx[label]
        compartment = label_to_compartment[label]
        alternative_indices = np.asarray(
            [
                label_to_idx[str(other)]
                for other in allowed[compartment]
                if str(other) != label
            ],
            dtype=np.int64,
        )
        label_dgi_support = matrices["dgi_support"][:, label_idx].astype(
            float, copy=False
        )
        if alternative_indices.size:
            alternative_support = np.max(
                matrices["dgi_support"][:, alternative_indices], axis=1
            )
        else:
            alternative_support = np.zeros(n_obs, dtype=float)
        label_dgi_margin = label_dgi_support - alternative_support

        present_by_cell = np.bincount(
            cell_values[supported & (current == label)],
            minlength=n_cells,
        ).astype(bool, copy=False)
        eligible = (
            ~supported
            & (broad == compartment)
            & ~present_by_cell[cell_values]
            & (
                matrices["local_counts"][:, label_idx]
                >= int(min_local_transcripts[label_idx])
            )
            & (
                matrices["local_genes"][:, label_idx]
                >= int(min_local_genes[label_idx])
            )
            & np.isfinite(matrices["local_enrichment"][:, label_idx])
            & (
                matrices["local_enrichment"][:, label_idx]
                >= float(min_local_enrichment[label_idx])
            )
            & np.isfinite(matrices["local_q"][:, label_idx])
            & available
            & np.isfinite(label_dgi_support)
            & np.isfinite(label_dgi_margin)
            & (label_dgi_support >= float(min_dgi_support))
            & (label_dgi_margin >= float(min_dgi_margin))
            & (
                matrices["cell_marker_counts"][:, label_idx]
                >= int(min_cell_transcripts)
            )
            & (
                matrices["cell_marker_genes"][:, label_idx]
                >= int(min_cell_genes)
            )
            & np.isfinite(
                matrices["cell_marker_enrichment"][:, label_idx]
            )
            & (
                matrices["cell_marker_enrichment"][:, label_idx]
                >= float(min_cell_enrichment)
            )
            & np.isfinite(
                matrices["cell_marker_binomial_p"][:, label_idx]
            )
            & (
                matrices["cell_marker_binomial_p"][:, label_idx]
                <= float(max_cell_p)
            )
        )
        eligible_rows = np.flatnonzero(eligible)
        if eligible_rows.size == 0:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "row": eligible_rows,
                    "cell_id": cell_values[eligible_rows],
                    "label": label,
                    "dgi_support": label_dgi_support[eligible_rows],
                    "dgi_margin": label_dgi_margin[eligible_rows],
                    "local_enrichment": matrices["local_enrichment"][
                        eligible_rows, label_idx
                    ],
                    "local_transcripts": matrices["local_counts"][
                        eligible_rows, label_idx
                    ],
                    "local_genes": matrices["local_genes"][
                        eligible_rows, label_idx
                    ],
                    "cell_enrichment": matrices[
                        "cell_marker_enrichment"
                    ][eligible_rows, label_idx],
                    "cell_transcripts": matrices["cell_marker_counts"][
                        eligible_rows, label_idx
                    ],
                    "cell_genes": matrices["cell_marker_genes"][
                        eligible_rows, label_idx
                    ],
                    "cell_p": matrices["cell_marker_binomial_p"][
                        eligible_rows, label_idx
                    ],
                    "label_order": label_idx,
                }
            )
        )

    if not rows:
        return rescue_labels

    ranked = pd.concat(rows, ignore_index=True)
    ranked.sort_values(
        [
            "dgi_support",
            "dgi_margin",
            "local_enrichment",
            "local_transcripts",
            "local_genes",
            "cell_enrichment",
            "cell_transcripts",
            "cell_genes",
            "cell_p",
            "label_order",
            "cell_id",
            "row",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
        inplace=True,
    )

    used_rows: set[int] = set()
    selected_per_cell_label: Counter[Tuple[int, str]] = Counter()
    for record in ranked.itertuples(index=False):
        row = int(record.row)
        key = (int(record.cell_id), str(record.label))
        if row in used_rows:
            continue
        if selected_per_cell_label[key] >= int(max_per_cell_label):
            continue
        rescue_labels[row] = str(record.label)
        used_rows.add(row)
        selected_per_cell_label[key] += 1
    return rescue_labels


def _select_metaspot_multistructure_first_loci(
    cell_ids: Sequence[object],
    current_labels: Sequence[object],
    already_supported: Sequence[bool],
    presence_score: np.ndarray,
    localization_score: np.ndarray,
    structure_evidence: np.ndarray,
    spatial_coherence: np.ndarray,
    geometry_compatible: np.ndarray,
    local_marker_transcripts: np.ndarray,
    local_marker_genes: np.ndarray,
    cell_marker_transcripts: np.ndarray,
    cell_marker_genes: np.ndarray,
    cell_marker_enrichment: np.ndarray,
    cell_marker_binomial_p: np.ndarray,
    dgi_available: Sequence[bool],
    metaspot_total_transcripts: Sequence[float],
    *,
    structures: Sequence[str],
    min_presence_score: float,
    min_localization_score: float,
    min_spatial_coherence: Sequence[float],
    min_evidence_score: Sequence[float],
    min_local_marker_transcripts: Sequence[int],
    min_local_marker_genes: Sequence[int],
    min_cell_marker_transcripts: Sequence[int],
    min_cell_marker_genes: Sequence[int],
    min_cell_marker_enrichment: Sequence[float],
    max_cell_marker_p: Sequence[float],
    max_projected_burden: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Select one evidence-supported missing locus per cell and structure."""
    n_obs = len(cell_ids)
    n_structures = len(structures)
    selected_labels = np.full(n_obs, "", dtype=object)
    diagnostics = {
        "eligible": np.zeros(n_obs, dtype=bool),
        "selected": np.zeros(n_obs, dtype=bool),
        "burden_rejected": np.zeros(n_obs, dtype=bool),
        "presence_score": np.full(n_obs, np.nan, dtype=float),
        "localization_score": np.full(n_obs, np.nan, dtype=float),
        "evidence_score": np.full(n_obs, np.nan, dtype=float),
        "spatial_coherence": np.full(n_obs, np.nan, dtype=float),
        "projected_burden": np.full(n_obs, np.nan, dtype=float),
    }
    if n_obs == 0 or n_structures == 0:
        return selected_labels, diagnostics

    matrices = {
        "presence_score": np.asarray(presence_score, dtype=float),
        "localization_score": np.asarray(localization_score, dtype=float),
        "structure_evidence": np.asarray(structure_evidence, dtype=float),
        "spatial_coherence": np.asarray(spatial_coherence, dtype=float),
        "geometry_compatible": np.asarray(geometry_compatible, dtype=bool),
        "local_marker_transcripts": np.asarray(local_marker_transcripts),
        "local_marker_genes": np.asarray(local_marker_genes),
        "cell_marker_transcripts": np.asarray(cell_marker_transcripts),
        "cell_marker_genes": np.asarray(cell_marker_genes),
        "cell_marker_enrichment": np.asarray(cell_marker_enrichment),
        "cell_marker_binomial_p": np.asarray(cell_marker_binomial_p),
    }
    expected_shape = (n_obs, n_structures)
    for name, values in matrices.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {values.shape}"
            )

    vector_parameters = {
        "min_evidence_score": np.asarray(min_evidence_score, dtype=float),
        "min_local_marker_transcripts": np.asarray(
            min_local_marker_transcripts, dtype=int
        ),
        "min_local_marker_genes": np.asarray(min_local_marker_genes, dtype=int),
        "min_cell_marker_transcripts": np.asarray(
            min_cell_marker_transcripts, dtype=int
        ),
        "min_cell_marker_genes": np.asarray(min_cell_marker_genes, dtype=int),
        "min_cell_marker_enrichment": np.asarray(
            min_cell_marker_enrichment, dtype=float
        ),
        "max_cell_marker_p": np.asarray(max_cell_marker_p, dtype=float),
        "min_spatial_coherence": np.asarray(
            min_spatial_coherence, dtype=float
        ),
    }
    for name, values in vector_parameters.items():
        if values.shape != (n_structures,):
            raise ValueError(
                f"{name} must have shape {(n_structures,)}, got {values.shape}"
            )
    score_parameters = np.concatenate(
        [
            np.asarray(
                [
                    min_presence_score,
                    min_localization_score,
                    max_projected_burden,
                ],
                dtype=float,
            ),
            vector_parameters["min_evidence_score"],
            vector_parameters["min_spatial_coherence"],
        ]
    )
    if not np.isfinite(score_parameters).all() or (
        (score_parameters < 0.0) | (score_parameters > 1.0)
    ).any():
        raise ValueError("First-locus score and burden thresholds must be in [0, 1]")
    if any(
        (values < 1).any()
        for name, values in vector_parameters.items()
        if name not in {
            "min_evidence_score",
            "min_cell_marker_enrichment",
            "max_cell_marker_p",
            "min_spatial_coherence",
        }
    ):
        raise ValueError("First-locus marker minima must be >= 1")

    supported = np.asarray(already_supported, dtype=bool)
    current = np.asarray(current_labels, dtype=object)
    available = np.asarray(dgi_available, dtype=bool)
    transcripts = np.asarray(metaspot_total_transcripts, dtype=float)
    if any(
        values.shape != (n_obs,)
        for values in (supported, current, available, transcripts)
    ):
        raise ValueError("First-locus row arrays must match the MetaSpot count")
    if not np.isfinite(transcripts).all() or (transcripts < 0).any():
        raise ValueError(
            "metaspot_total_transcripts must contain finite non-negative values"
        )

    cell_codes, n_cells = _dense_group_codes(
        np.asarray(cell_ids, dtype=object),
        error_context="MetaSpot multi-structure first-locus selection",
    )
    cell_totals = np.bincount(
        cell_codes, weights=transcripts, minlength=n_cells
    ).astype(float)
    projected = np.divide(
        transcripts,
        cell_totals[cell_codes],
        out=np.full(n_obs, np.inf),
        where=cell_totals[cell_codes] > 0,
    )
    ranked_parts: List[pd.DataFrame] = []
    for structure_idx, raw_structure in enumerate(structures):
        structure = str(raw_structure)
        present_by_cell = np.bincount(
            cell_codes[supported & (current == structure)],
            minlength=n_cells,
        ).astype(bool, copy=False)
        evidence_eligible = (
            ~supported
            & ~present_by_cell[cell_codes]
            & available
            & matrices["geometry_compatible"][:, structure_idx]
            & (
                matrices["local_marker_transcripts"][:, structure_idx]
                >= vector_parameters["min_local_marker_transcripts"][structure_idx]
            )
            & (
                matrices["local_marker_genes"][:, structure_idx]
                >= vector_parameters["min_local_marker_genes"][structure_idx]
            )
            & (
                matrices["cell_marker_transcripts"][:, structure_idx]
                >= vector_parameters["min_cell_marker_transcripts"][structure_idx]
            )
            & (
                matrices["cell_marker_genes"][:, structure_idx]
                >= vector_parameters["min_cell_marker_genes"][structure_idx]
            )
            & np.isfinite(matrices["cell_marker_enrichment"][:, structure_idx])
            & (
                matrices["cell_marker_enrichment"][:, structure_idx]
                >= vector_parameters["min_cell_marker_enrichment"][structure_idx]
            )
            & np.isfinite(matrices["cell_marker_binomial_p"][:, structure_idx])
            & (
                matrices["cell_marker_binomial_p"][:, structure_idx]
                <= vector_parameters["max_cell_marker_p"][structure_idx]
            )
            & np.isfinite(matrices["presence_score"][:, structure_idx])
            & (
                matrices["presence_score"][:, structure_idx]
                >= float(min_presence_score)
            )
            & np.isfinite(matrices["localization_score"][:, structure_idx])
            & (
                matrices["localization_score"][:, structure_idx]
                >= float(min_localization_score)
            )
            & np.isfinite(matrices["structure_evidence"][:, structure_idx])
            & (
                matrices["structure_evidence"][:, structure_idx]
                >= vector_parameters["min_evidence_score"][structure_idx]
            )
            & np.isfinite(matrices["spatial_coherence"][:, structure_idx])
            & (
                matrices["spatial_coherence"][:, structure_idx]
                >= vector_parameters["min_spatial_coherence"][structure_idx]
            )
        )
        diagnostics["burden_rejected"] |= evidence_eligible & (
            projected > float(max_projected_burden)
        )
        rows = np.flatnonzero(
            evidence_eligible & (projected <= float(max_projected_burden))
        )
        diagnostics["eligible"][rows] = True
        if rows.size == 0:
            continue
        ranked_parts.append(
            pd.DataFrame(
                {
                    "row": rows,
                    "cell": cell_codes[rows],
                    "structure": structure,
                    "structure_idx": structure_idx,
                    "evidence": matrices["structure_evidence"][
                        rows, structure_idx
                    ],
                    "spatial_coherence": matrices["spatial_coherence"][
                        rows, structure_idx
                    ],
                    "localization": matrices["localization_score"][
                        rows, structure_idx
                    ],
                    "presence": matrices["presence_score"][
                        rows, structure_idx
                    ],
                    "projected_burden": projected[rows],
                }
            )
        )

    if not ranked_parts:
        return selected_labels, diagnostics
    ranked = pd.concat(ranked_parts, ignore_index=True).sort_values(
        [
            "evidence",
            "localization",
            "presence",
            "projected_burden",
            "structure_idx",
            "row",
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    )
    used_rows: set[int] = set()
    selected_cell_structures: set[Tuple[int, str]] = set()
    added_burden = np.zeros(n_cells, dtype=float)
    for record in ranked.itertuples(index=False):
        row = int(record.row)
        cell = int(record.cell)
        structure = str(record.structure)
        key = (cell, structure)
        projected_total = added_burden[cell] + float(record.projected_burden)
        if row in used_rows or key in selected_cell_structures:
            continue
        if projected_total > float(max_projected_burden):
            diagnostics["burden_rejected"][row] = True
            continue
        selected_labels[row] = structure
        diagnostics["selected"][row] = True
        diagnostics["presence_score"][row] = float(record.presence)
        diagnostics["localization_score"][row] = float(record.localization)
        diagnostics["evidence_score"][row] = float(record.evidence)
        diagnostics["spatial_coherence"][row] = float(record.spatial_coherence)
        diagnostics["projected_burden"][row] = float(record.projected_burden)
        used_rows.add(row)
        selected_cell_structures.add(key)
        added_burden[cell] = projected_total
    return selected_labels, diagnostics


def _apply_metaspot_diminishing_return_selection(
    cell_ids: Sequence[object],
    final_labels: Sequence[object],
    strict_supported: Sequence[bool],
    broad_values: Sequence[object],
    local_enrichment: np.ndarray,
    dgi_support: np.ndarray,
    metaspot_total_transcripts: Sequence[float],
    *,
    enabled_labels: Sequence[str],
    all_labels: Sequence[str],
    allowed: Mapping[str, Sequence[str]],
    min_local_enrichment: Sequence[float],
    dgi_weight: float,
    min_dgi_support: float,
    log_penalty: float,
    burden_penalty: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Keep the first supported locus and raise evidence for later loci."""
    labels = np.asarray(final_labels, dtype=object)
    n_obs = labels.size
    diagnostics = {
        "eligible": np.zeros(n_obs, dtype=bool),
        "kept": np.ones(n_obs, dtype=bool),
        "pruned": np.zeros(n_obs, dtype=bool),
        "first_locus": np.zeros(n_obs, dtype=bool),
        "rank": np.zeros(n_obs, dtype=np.int32),
        "evidence_score": np.full(n_obs, np.nan, dtype=float),
        "required_score": np.full(n_obs, np.nan, dtype=float),
        "burden_before": np.full(n_obs, np.nan, dtype=float),
    }
    if n_obs == 0 or not enabled_labels:
        return diagnostics["kept"], diagnostics

    enrichment = np.asarray(local_enrichment, dtype=float)
    support = np.asarray(dgi_support, dtype=float)
    strict = np.asarray(strict_supported, dtype=bool)
    broad = np.asarray(broad_values, dtype=object)
    transcripts = np.asarray(metaspot_total_transcripts, dtype=float)
    minimum_enrichment = np.asarray(min_local_enrichment, dtype=float)
    n_labels = len(all_labels)
    if enrichment.shape != (n_obs, n_labels) or support.shape != (
        n_obs,
        n_labels,
    ):
        raise ValueError(
            "Diminishing-return enrichment and DGI matrices must match "
            "(n_metaspots, n_labels)"
        )
    if any(array.shape != (n_obs,) for array in (strict, broad, transcripts)):
        raise ValueError(
            "Diminishing-return row arrays must match the MetaSpot count"
        )
    if minimum_enrichment.shape != (n_labels,):
        raise ValueError(
            "Diminishing-return label thresholds must match all_labels"
        )
    numeric_parameters = np.asarray(
        [dgi_weight, min_dgi_support, log_penalty, burden_penalty],
        dtype=float,
    )
    if not np.isfinite(numeric_parameters).all() or (
        numeric_parameters < 0
    ).any():
        raise ValueError(
            "Diminishing-return parameters must be finite and non-negative"
        )
    if min_dgi_support > 1.0:
        raise ValueError(
            "Diminishing-return min_dgi_support must be <= 1"
        )
    if not np.isfinite(transcripts).all() or (transcripts < 0).any():
        raise ValueError(
            "Diminishing-return transcript counts must be finite and "
            "non-negative"
        )

    cell_codes, n_cells = _dense_group_codes(
        np.asarray(cell_ids, dtype=object),
        error_context="MetaSpot diminishing-return selection",
    )
    label_to_idx = {str(label): idx for idx, label in enumerate(all_labels)}
    label_to_compartment = {
        str(label): str(compartment)
        for compartment, compartment_labels in allowed.items()
        for label in compartment_labels
    }
    compartment_totals = {
        compartment: np.bincount(
            cell_codes[broad == compartment],
            weights=transcripts[broad == compartment],
            minlength=n_cells,
        )
        for compartment in allowed
    }

    for raw_label in enabled_labels:
        label = str(raw_label)
        if label not in label_to_idx or label not in label_to_compartment:
            raise ValueError(
                f"Unknown diminishing-return label: {label!r}"
            )
        label_idx = label_to_idx[label]
        compartment = label_to_compartment[label]
        rows = np.flatnonzero(labels == label)
        if rows.size == 0:
            continue
        label_support = support[rows, label_idx]
        label_enrichment = enrichment[rows, label_idx]
        evidence = label_enrichment + float(dgi_weight) * label_support
        if not np.isfinite(evidence).all():
            raise ValueError(
                f"Selected {label!r} loci require finite local and DGI evidence"
            )
        diagnostics["eligible"][rows] = True
        diagnostics["evidence_score"][rows] = evidence
        ranked = pd.DataFrame(
            {
                "row": rows,
                "cell": cell_codes[rows],
                "strict": strict[rows],
                "evidence": evidence,
                "transcripts": transcripts[rows],
            }
        ).sort_values(
            ["cell", "strict", "evidence", "transcripts", "row"],
            ascending=[True, False, False, True, True],
            kind="mergesort",
        )
        for cell_code, group in ranked.groupby("cell", sort=False):
            denominator = float(compartment_totals[compartment][cell_code])
            if denominator <= 0.0:
                raise ValueError(
                    f"Selected {label!r} locus has no {compartment} denominator"
                )
            kept_count = 0
            kept_transcripts = 0.0
            for rank, record in enumerate(group.itertuples(index=False), 1):
                row = int(record.row)
                burden_before = kept_transcripts / denominator
                diagnostics["rank"][row] = rank
                diagnostics["burden_before"][row] = burden_before
                first_locus = kept_count == 0
                diagnostics["first_locus"][row] = first_locus
                required = (
                    minimum_enrichment[label_idx]
                    + float(dgi_weight) * float(min_dgi_support)
                    + float(log_penalty) * np.log1p(kept_count)
                    + float(burden_penalty) * burden_before
                )
                diagnostics["required_score"][row] = required
                keep = bool(record.strict) or first_locus or (
                    float(record.evidence) >= required
                )
                diagnostics["kept"][row] = keep
                if keep:
                    kept_count += 1
                    kept_transcripts += float(record.transcripts)

    diagnostics["pruned"] = diagnostics["eligible"] & ~diagnostics["kept"]
    if np.any(diagnostics["pruned"] & strict):
        raise RuntimeError("Diminishing-return selection pruned strict evidence")
    return diagnostics["kept"], diagnostics


def _select_metaspot_hierarchical_cell_rescues(
    cell_ids: np.ndarray,
    broad_values: np.ndarray,
    already_supported: np.ndarray,
    local_counts: np.ndarray,
    local_genes: np.ndarray,
    local_enrichment: np.ndarray,
    dgi_available: np.ndarray,
    dgi_support: np.ndarray,
    cell_marker_counts: np.ndarray,
    cell_marker_genes: np.ndarray,
    cell_marker_enrichment: np.ndarray,
    cell_marker_binomial_p: np.ndarray,
    metaspot_total_transcripts: np.ndarray,
    *,
    labels: Sequence[str],
    all_labels: Sequence[str],
    allowed: Mapping[str, Sequence[str]],
    min_cell_transcripts: int,
    min_metaspots: int,
    min_presence_score: float,
    min_localization_score: float,
    min_joint_score: float,
    min_local_marker_transcripts: int,
    min_local_marker_genes: int,
    min_cell_marker_transcripts: int,
    min_cell_marker_genes: int,
    min_dgi_support: float,
    min_dgi_margin: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Rescue one fine structure in callable cells using graded evidence.

    This is the final Tier-B coverage branch.  It is evaluated only for cells
    that do not already contain an evidence-supported fine structure.  Cell
    callability is defined from total transcript and MetaSpot depth, while a
    label must retain three independent anchors: cell-level marker presence,
    a measured marker in the focal MetaSpot, and leave-one-out DGI support for
    the local placement.  The best label/MetaSpot pair is selected once per
    cell; cells are never assigned a quota of structures.

    The bounded scores are evidence scores, not calibrated probabilities.
    Their fixed weights are returned through the annotation metadata so the
    decision remains reproducible and auditable.
    """
    n_obs = len(cell_ids)
    n_labels = len(all_labels)
    rescue_labels = np.full(n_obs, "", dtype=object)
    diagnostics: Dict[str, np.ndarray] = {
        "callable": np.zeros(n_obs, dtype=bool),
        "eligible_cell": np.zeros(n_obs, dtype=bool),
        "status": np.full(n_obs, "not_callable", dtype=object),
        "cell_total_transcripts": np.full(n_obs, np.nan, dtype=float),
        "cell_metaspots": np.full(n_obs, np.nan, dtype=float),
        "presence_score": np.full(n_obs, np.nan, dtype=float),
        "localization_score": np.full(n_obs, np.nan, dtype=float),
        "joint_score": np.full(n_obs, np.nan, dtype=float),
        "best_label": np.full(n_obs, "", dtype=object),
        "best_presence_score": np.full(n_obs, np.nan, dtype=float),
        "best_localization_score": np.full(n_obs, np.nan, dtype=float),
        "best_joint_score": np.full(n_obs, np.nan, dtype=float),
        "best_dgi_support": np.full(n_obs, np.nan, dtype=float),
        "best_dgi_margin": np.full(n_obs, np.nan, dtype=float),
    }
    if n_obs == 0:
        return rescue_labels, diagnostics

    matrices = {
        "local_counts": np.asarray(local_counts),
        "local_genes": np.asarray(local_genes),
        "local_enrichment": np.asarray(local_enrichment),
        "dgi_support": np.asarray(dgi_support),
        "cell_marker_counts": np.asarray(cell_marker_counts),
        "cell_marker_genes": np.asarray(cell_marker_genes),
        "cell_marker_enrichment": np.asarray(cell_marker_enrichment),
        "cell_marker_binomial_p": np.asarray(cell_marker_binomial_p),
    }
    for name, values in matrices.items():
        if values.shape != (n_obs, n_labels):
            raise ValueError(
                f"{name} must have shape {(n_obs, n_labels)}, got "
                f"{values.shape}"
            )

    cell_values, n_cells = _dense_group_codes(
        cell_ids,
        error_context="MetaSpot hierarchical cell rescue",
    )
    broad = np.asarray(broad_values, dtype=object)
    supported = np.asarray(already_supported, dtype=bool)
    available = np.asarray(dgi_available, dtype=bool)
    spot_transcripts = np.asarray(
        metaspot_total_transcripts, dtype=float
    ).reshape(-1)
    if any(
        values.shape != (n_obs,)
        for values in (broad, supported, available, spot_transcripts)
    ):
        raise ValueError(
            "MetaSpot hierarchical rescue row arrays must be one-dimensional"
        )
    if not np.isfinite(spot_transcripts).all() or (
        spot_transcripts < 0
    ).any():
        raise ValueError(
            "metaspot_total_transcripts must contain finite non-negative values"
        )

    cell_total_values = np.bincount(
        cell_values,
        weights=spot_transcripts,
        minlength=n_cells,
    ).astype(float)
    cell_metaspot_values = np.bincount(
        cell_values,
        minlength=n_cells,
    ).astype(float)
    cell_total_by_row = cell_total_values[cell_values]
    cell_metaspots_by_row = cell_metaspot_values[cell_values]
    callable_rows = (
        (cell_total_by_row >= int(min_cell_transcripts))
        & (cell_metaspots_by_row >= int(min_metaspots))
    )
    supported_by_cell = np.bincount(
        cell_values[supported],
        minlength=n_cells,
    ).astype(bool, copy=False)
    eligible_cell_rows = callable_rows & ~supported_by_cell[cell_values]
    diagnostics["callable"] = callable_rows
    diagnostics["eligible_cell"] = eligible_cell_rows
    diagnostics["status"][callable_rows] = "already_has_fine_structure"
    diagnostics["status"][eligible_cell_rows] = "no_anchored_candidate"
    diagnostics["cell_total_transcripts"] = cell_total_by_row
    diagnostics["cell_metaspots"] = cell_metaspots_by_row
    if not labels or not eligible_cell_rows.any():
        return rescue_labels, diagnostics

    label_to_idx = {
        str(label): idx for idx, label in enumerate(all_labels)
    }
    label_to_compartment = {
        str(label): str(compartment)
        for compartment, compartment_labels in allowed.items()
        for label in compartment_labels
    }

    def _enrichment_score(values: np.ndarray) -> np.ndarray:
        scaled = np.clip(np.asarray(values, dtype=float) / 0.5, -12.0, 12.0)
        return 1.0 / (1.0 + np.exp(-scaled))

    ranked_parts: List[pd.DataFrame] = []
    for raw_label in labels:
        label = str(raw_label)
        if label not in label_to_idx or label not in label_to_compartment:
            raise ValueError(
                f"Unknown hierarchical cell rescue label: {label!r}"
            )
        label_idx = label_to_idx[label]
        compartment = label_to_compartment[label]
        alternative_indices = np.asarray(
            [
                label_to_idx[str(other)]
                for other in allowed[compartment]
                if str(other) != label
            ],
            dtype=np.int64,
        )
        label_dgi_support = matrices["dgi_support"][:, label_idx].astype(
            float, copy=False
        )
        if alternative_indices.size:
            alternative_support = np.max(
                matrices["dgi_support"][:, alternative_indices], axis=1
            )
        else:
            alternative_support = np.zeros(n_obs, dtype=float)
        label_dgi_margin = label_dgi_support - alternative_support

        local_count = matrices["local_counts"][:, label_idx].astype(float)
        local_gene = matrices["local_genes"][:, label_idx].astype(float)
        local_enrich = matrices["local_enrichment"][:, label_idx].astype(
            float
        )
        cell_count = matrices["cell_marker_counts"][:, label_idx].astype(
            float
        )
        cell_gene = matrices["cell_marker_genes"][:, label_idx].astype(float)
        cell_enrich = matrices["cell_marker_enrichment"][:, label_idx].astype(
            float
        )
        cell_p = matrices["cell_marker_binomial_p"][:, label_idx].astype(
            float
        )

        # These scores deliberately saturate at modest marker counts. Fine
        # structures are sparse, so extra copies strengthen confidence without
        # allowing abundant transcripts to dominate the DGI placement term.
        presence_score = (
            0.30 * np.clip(cell_count / 2.0, 0.0, 1.0)
            + 0.25 * np.clip(cell_gene / 2.0, 0.0, 1.0)
            + 0.25 * _enrichment_score(cell_enrich)
            + 0.20 * np.clip(1.0 - cell_p, 0.0, 1.0)
        )
        localization_score = (
            0.15 * np.clip(local_count / 2.0, 0.0, 1.0)
            + 0.10 * np.clip(local_gene / 2.0, 0.0, 1.0)
            + 0.20 * _enrichment_score(local_enrich)
            + 0.45 * np.clip(label_dgi_support, 0.0, 1.0)
            + 0.10 * np.clip((label_dgi_margin + 1.0) / 2.0, 0.0, 1.0)
        )
        joint_score = np.sqrt(
            np.clip(presence_score, 0.0, 1.0)
            * np.clip(localization_score, 0.0, 1.0)
        )

        anchored = (
            eligible_cell_rows
            & (broad == compartment)
            & (local_count >= int(min_local_marker_transcripts))
            & (local_gene >= int(min_local_marker_genes))
            & np.isfinite(local_enrich)
            & (cell_count >= int(min_cell_marker_transcripts))
            & (cell_gene >= int(min_cell_marker_genes))
            & np.isfinite(cell_enrich)
            & np.isfinite(cell_p)
            & available
            & np.isfinite(label_dgi_support)
            & np.isfinite(label_dgi_margin)
            & (label_dgi_support >= float(min_dgi_support))
            & (label_dgi_margin >= float(min_dgi_margin))
            & np.isfinite(presence_score)
            & np.isfinite(localization_score)
            & np.isfinite(joint_score)
        )
        rows = np.flatnonzero(anchored)
        if rows.size == 0:
            continue
        ranked_parts.append(
            pd.DataFrame(
                {
                    "row": rows,
                    "cell_id": cell_values[rows],
                    "label": label,
                    "presence_score": presence_score[rows],
                    "localization_score": localization_score[rows],
                    "joint_score": joint_score[rows],
                    "dgi_support": label_dgi_support[rows],
                    "dgi_margin": label_dgi_margin[rows],
                    "local_enrichment": local_enrich[rows],
                    "local_transcripts": local_count[rows],
                    "cell_enrichment": cell_enrich[rows],
                    "cell_transcripts": cell_count[rows],
                    "passes_score_thresholds": (
                        (presence_score[rows] >= float(min_presence_score))
                        & (
                            localization_score[rows]
                            >= float(min_localization_score)
                        )
                        & (joint_score[rows] >= float(min_joint_score))
                    ),
                    "label_order": label_idx,
                }
            )
        )

    if not ranked_parts:
        return rescue_labels, diagnostics

    ranked = pd.concat(ranked_parts, ignore_index=True)
    ranked.sort_values(
        [
            "cell_id",
            "joint_score",
            "presence_score",
            "localization_score",
            "dgi_support",
            "dgi_margin",
            "local_enrichment",
            "local_transcripts",
            "cell_enrichment",
            "cell_transcripts",
            "label_order",
            "row",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
        kind="mergesort",
        inplace=True,
    )
    best = ranked.groupby("cell_id", sort=False).head(1)
    best_rows = best["row"].to_numpy(dtype=np.int64)
    diagnostics["best_label"][best_rows] = best["label"].astype(str).to_numpy()
    for source, target in (
        ("presence_score", "best_presence_score"),
        ("localization_score", "best_localization_score"),
        ("joint_score", "best_joint_score"),
        ("dgi_support", "best_dgi_support"),
        ("dgi_margin", "best_dgi_margin"),
    ):
        diagnostics[target][best_rows] = best[source].to_numpy(dtype=float)
    diagnostics["status"][
        np.isin(cell_values, ranked["cell_id"].unique())
    ] = "score_below_threshold"

    selected = ranked.loc[
        ranked["passes_score_thresholds"]
    ].groupby("cell_id", sort=False).head(1)
    if selected.empty:
        return rescue_labels, diagnostics
    selected_rows = selected["row"].to_numpy(dtype=np.int64)
    rescue_labels[selected_rows] = selected["label"].astype(str).to_numpy()
    diagnostics["status"][
        np.isin(cell_values, selected["cell_id"].unique())
    ] = "selected"
    for name in ("presence_score", "localization_score", "joint_score"):
        diagnostics[name][selected_rows] = selected[name].to_numpy(dtype=float)
    return rescue_labels, diagnostics


def _select_metaspot_omm_cell_coherent_rescues(
    cell_ids: np.ndarray,
    candidate: np.ndarray,
    already_supported: np.ndarray,
    broad_values: np.ndarray,
    local_counts: np.ndarray,
    local_genes: np.ndarray,
    local_enrichment: np.ndarray,
    local_score_gap: np.ndarray,
    dgi_available: np.ndarray,
    dgi_support: np.ndarray,
    dgi_margin: np.ndarray,
    cell_counts: np.ndarray,
    cell_genes: np.ndarray,
    cell_enrichment: np.ndarray,
    cell_binomial_p: np.ndarray,
    *,
    min_cell_transcripts: int,
    min_cell_genes: int,
    min_cell_enrichment: float,
    max_cell_p: float,
    min_local_transcripts: int,
    min_local_genes: int,
    min_local_enrichment: float,
    min_local_score_gap: float,
    min_dgi_support: float,
    min_dgi_margin: float,
    max_per_cell: int,
) -> np.ndarray:
    """Select OMM loci only inside cells with coherent multi-gene evidence.

    A MetaSpot-level binomial test is underpowered when mitochondrial RNAs are
    split across several small MetaSpots.  This rescue therefore establishes
    OMM presence at the cell level, using exact pooled marker counts and
    distinct genes, before placing the label on a bounded number of local
    MetaSpots.  Local OMM-over-ER/Cytosol separation and leave-one-out DGI
    agreement remain mandatory, so this is not a per-cell OMM quota.
    """
    n_obs = len(candidate)
    rescued = np.zeros(n_obs, dtype=bool)
    if n_obs == 0 or int(max_per_cell) <= 0:
        return rescued

    cell_values, n_cells = _dense_group_codes(
        cell_ids,
        error_context="MetaSpot OMM cell-coherent rescue",
    )
    candidate_values = np.asarray(candidate, dtype=object)
    supported_values = np.asarray(already_supported, dtype=bool)
    broad = np.asarray(broad_values, dtype=object)

    present_by_cell = np.bincount(
        cell_values[supported_values & (candidate_values == "OMM")],
        minlength=n_cells,
    ).astype(bool, copy=False)
    eligible = (
        (candidate_values == "OMM")
        & ~supported_values
        & (broad == "Cytoplasmic")
        & ~present_by_cell[cell_values]
        & (np.asarray(local_counts, dtype=float) >= int(min_local_transcripts))
        & (np.asarray(local_genes, dtype=float) >= int(min_local_genes))
        & (np.asarray(local_enrichment, dtype=float) >= float(min_local_enrichment))
        & (np.asarray(local_score_gap, dtype=float) >= float(min_local_score_gap))
        & np.asarray(dgi_available, dtype=bool)
        & (np.asarray(dgi_support, dtype=float) >= float(min_dgi_support))
        & (np.asarray(dgi_margin, dtype=float) >= float(min_dgi_margin))
        & (np.asarray(cell_counts, dtype=float) >= int(min_cell_transcripts))
        & (np.asarray(cell_genes, dtype=float) >= int(min_cell_genes))
        & (np.asarray(cell_enrichment, dtype=float) >= float(min_cell_enrichment))
        & np.isfinite(np.asarray(cell_binomial_p, dtype=float))
        & (np.asarray(cell_binomial_p, dtype=float) <= float(max_cell_p))
    )
    eligible_rows = np.flatnonzero(eligible)
    if eligible_rows.size == 0:
        return rescued

    ranked = pd.DataFrame(
        {
            "row": eligible_rows,
            "cell_id": cell_values[eligible_rows],
            "dgi_support": np.asarray(dgi_support)[eligible_rows],
            "dgi_margin": np.asarray(dgi_margin)[eligible_rows],
            "local_enrichment": np.asarray(local_enrichment)[eligible_rows],
            "local_score_gap": np.asarray(local_score_gap)[eligible_rows],
            "local_transcripts": np.asarray(local_counts)[eligible_rows],
        }
    )
    ranked.sort_values(
        [
            "cell_id",
            "dgi_support",
            "dgi_margin",
            "local_enrichment",
            "local_score_gap",
            "local_transcripts",
            "row",
        ],
        ascending=[True, False, False, False, False, False, True],
        kind="mergesort",
        inplace=True,
    )
    selected = ranked.groupby("cell_id", sort=False).head(int(max_per_cell))
    rescued[selected["row"].to_numpy(dtype=np.int64)] = True
    return rescued


def _compute_multistructure_evidence(
    local_counts: np.ndarray,
    local_genes: np.ndarray,
    local_enrichment: np.ndarray,
    metadomain_affinity: np.ndarray,
    dgi_support: np.ndarray,
    geometry_compatible: np.ndarray,
    competitor_margin: np.ndarray,
    rapx_calibrated_score: np.ndarray,
    cell_counts: np.ndarray,
    cell_genes: np.ndarray,
    cell_enrichment: np.ndarray,
    cell_binomial_p: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose bounded cell-presence and MetaSpot-localization evidence."""
    arrays = [
        np.asarray(values, dtype=float)
        for values in (
            local_counts,
            local_genes,
            local_enrichment,
            metadomain_affinity,
            dgi_support,
            geometry_compatible,
            competitor_margin,
            rapx_calibrated_score,
            cell_counts,
            cell_genes,
            cell_enrichment,
            cell_binomial_p,
        )
    ]
    if any(values.shape != arrays[0].shape for values in arrays):
        raise ValueError("Multi-structure evidence arrays must share one shape")

    def _sigmoid(values: np.ndarray, scale: float) -> np.ndarray:
        output = np.zeros(values.shape, dtype=float)
        finite = np.isfinite(values)
        output[finite] = 1.0 / (
            1.0
            + np.exp(-np.clip(values[finite] / float(scale), -12.0, 12.0))
        )
        return output

    (
        local_count,
        local_gene,
        local_enrich,
        domain_affinity,
        local_dgi,
        geometry,
        contrast,
        rapx_score,
        cell_count,
        cell_gene,
        cell_enrich,
        cell_p,
    ) = arrays
    one_minus_p = np.zeros(cell_p.shape, dtype=float)
    finite_p = np.isfinite(cell_p)
    one_minus_p[finite_p] = np.clip(1.0 - cell_p[finite_p], 0.0, 1.0)
    presence = (
        0.30 * np.clip(cell_count / 2.0, 0.0, 1.0)
        + 0.25 * np.clip(cell_gene / 2.0, 0.0, 1.0)
        + 0.25 * _sigmoid(cell_enrich, 0.5)
        + 0.20 * one_minus_p
    )
    marker = (
        0.30 * np.clip(local_count / 2.0, 0.0, 1.0)
        + 0.25 * np.clip(local_gene / 2.0, 0.0, 1.0)
        + 0.45 * _sigmoid(local_enrich, 0.5)
    )
    geometry_mask = np.asarray(geometry, dtype=bool)
    localization = (
        0.25 * marker
        + 0.20 * np.clip(domain_affinity, 0.0, 1.0)
        + 0.20 * np.clip(local_dgi, 0.0, 1.0)
        + 0.10 * geometry_mask
        + 0.15 * _sigmoid(contrast, 0.5)
        + 0.10 * _sigmoid(rapx_score, 1.0)
    )
    localization *= geometry_mask
    evidence = np.sqrt(
        np.clip(presence, 0.0, 1.0)
        * np.clip(localization, 0.0, 1.0)
    )
    return (
        presence.astype(np.float32),
        localization.astype(np.float32),
        evidence.astype(np.float32),
    )


def _compute_cell_structure_spatial_coherence(
    cell_codes: np.ndarray,
    broad_values: Sequence[object],
    local_marker_counts: np.ndarray,
    metaspot_total_transcripts: Sequence[float],
    structures: Sequence[str],
    structure_compartments: Mapping[str, str],
) -> np.ndarray:
    """Measure marker concentration within each cell compartment."""
    codes = np.asarray(cell_codes)
    broad = np.asarray(broad_values, dtype=object)
    marker_counts = np.asarray(local_marker_counts, dtype=float)
    transcripts = np.asarray(metaspot_total_transcripts, dtype=float)
    if marker_counts.ndim != 2 or marker_counts.shape[0] != len(codes):
        raise ValueError("Spatial coherence marker counts must match cell codes")
    if broad.shape != codes.shape or transcripts.shape != codes.shape:
        raise ValueError("Spatial coherence row arrays must match cell codes")
    n_cells = int(codes.max()) + 1 if codes.size else 0
    coherence = np.zeros((len(codes), len(structures)), dtype=float)
    for structure_idx, structure in enumerate(structures):
        compartment = str(structure_compartments[str(structure)])
        rows = broad == compartment
        if not rows.any():
            continue
        marker_totals = np.bincount(
            codes[rows],
            weights=marker_counts[rows, structure_idx],
            minlength=n_cells,
        )
        compartment_totals = np.bincount(
            codes[rows],
            weights=transcripts[rows],
            minlength=n_cells,
        )
        compartment_marker_counts = np.where(
            rows, marker_counts[:, structure_idx], 0.0
        )
        marker_share = np.divide(
            compartment_marker_counts,
            marker_totals[codes],
            out=np.zeros(len(codes), dtype=float),
            where=marker_totals[codes] > 0,
        )
        transcript_share = np.divide(
            transcripts,
            compartment_totals[codes],
            out=np.zeros(len(codes), dtype=float),
            where=rows & (compartment_totals[codes] > 0),
        )
        distance = np.abs(marker_share - transcript_share)
        total_variation = 0.5 * np.bincount(
            codes,
            weights=distance,
            minlength=n_cells,
        )
        coherence[:, structure_idx] = total_variation[codes]
    return np.clip(coherence, 0.0, 1.0).astype(np.float32)


def _refine_metaspot_annotations_with_local_dgi(
    adata1: ad.AnnData,
    gold_df: pd.DataFrame,
    annotation_cfg: Mapping[str, Any],
    *,
    build_score_table: bool = True,
) -> Tuple[ad.AnnData, Optional[pd.DataFrame], Dict[str, Any]]:
    """Refine a MetaDomain broad prior with local MetaSpot marker evidence.

    The broad nuclear/cytoplasmic parent is fixed before DGI pooling.  Fine
    labels can therefore compete only with labels from the same compartment.
    Strict subtypes require pooled marker count, distinct genes, enrichment,
    score separation, and a label-wise BH q-value. A separately tagged
    exploratory track may rescue marker-qualified candidates only when
    leave-one-out DGI neighbors agree strongly. Every unsupported decision is
    exposed as an unresolved fine state while the compatibility label falls
    back to the broad parent.
    """
    if "subcellular_annotation" not in adata1.obs.columns:
        raise KeyError(
            "MetaSpot refinement requires a mapped MetaDomain annotation"
        )

    def _finite_float(name: str, default: float, lower: float = 0.0) -> float:
        try:
            value = float(annotation_cfg.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"annotation.{name} must be numeric") from exc
        if not np.isfinite(value) or value < lower:
            raise ValueError(
                f"annotation.{name} must be finite and >= {lower}"
            )
        return value

    try:
        min_transcripts = int(
            annotation_cfg.get("metaspot_min_marker_transcripts", 2)
        )
        min_genes = int(annotation_cfg.get("metaspot_min_marker_genes", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "annotation.metaspot_min_marker_transcripts and "
            "metaspot_min_marker_genes must be integers"
        ) from exc
    if min_transcripts < 1 or min_genes < 1:
        raise ValueError(
            "annotation MetaSpot marker minima must both be >= 1"
        )
    min_enrichment = _finite_float(
        "metaspot_min_log2_enrichment", 0.3
    )
    min_gap = _finite_float("metaspot_min_score_gap", 0.1)
    strict_fdr = _finite_float("metaspot_subtype_fdr", 0.10)
    exploratory_q = _finite_float("metaspot_exploratory_max_q", 0.40)
    prior_strength = _finite_float(
        "metaspot_background_prior_strength", 2.0
    )
    dgi_consensus_weight = _finite_float(
        "metaspot_dgi_consensus_weight", 0.75
    )
    dgi_seed_min_enrichment = _finite_float(
        "metaspot_dgi_seed_min_log2_enrichment", 0.5
    )
    dgi_exploratory_enabled = bool(
        annotation_cfg.get("metaspot_dgi_exploratory", True)
    )
    dgi_exploratory_min_support = _finite_float(
        "metaspot_dgi_exploratory_min_support",
        0.75,
    )
    dgi_exploratory_min_margin = _finite_float(
        "metaspot_dgi_exploratory_min_margin",
        0.25,
    )
    if (
        dgi_exploratory_min_support > 1.0
        or dgi_exploratory_min_margin > 1.0
    ):
        raise ValueError(
            "annotation MetaSpot DGI exploratory support and margin must "
            "both be <= 1"
        )
    try:
        dgi_seed_min_transcripts = int(
            annotation_cfg.get(
                "metaspot_dgi_seed_min_marker_transcripts",
                2,
            )
        )
        dgi_seed_min_genes = int(
            annotation_cfg.get("metaspot_dgi_seed_min_marker_genes", 1)
        )
        dgi_min_informative_neighbors = int(
            annotation_cfg.get(
                "metaspot_dgi_min_informative_neighbors",
                2,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "MetaSpot DGI seed and neighbor minima must be integers"
        ) from exc
    if (
        dgi_seed_min_transcripts < 1
        or dgi_seed_min_genes < 1
        or dgi_min_informative_neighbors < 1
    ):
        raise ValueError(
            "MetaSpot DGI seed and neighbor minima must be >= 1"
        )
    require_dgi_support = bool(
        annotation_cfg.get("metaspot_require_dgi_support", False)
    )
    if strict_fdr > 1.0 or exploratory_q > 1.0:
        raise ValueError("MetaSpot q-value thresholds must be <= 1")
    if exploratory_q < strict_fdr:
        raise ValueError(
            "annotation.metaspot_exploratory_max_q must be >= "
            "annotation.metaspot_subtype_fdr"
        )

    omm_unidentifiable_policy = str(
        annotation_cfg.get(
            "metaspot_omm_unidentifiable_policy",
            "unresolved",
        )
    ).strip().lower()
    if omm_unidentifiable_policy not in {"unresolved", "strict_local"}:
        raise ValueError(
            "annotation.metaspot_omm_unidentifiable_policy must be "
            "'unresolved' or 'strict_local'"
        )

    prior_column = (
        "subcellular_annotation_domain"
        if "subcellular_annotation_domain" in adata1.obs.columns
        else "subcellular_annotation"
    )
    prior_labels = (
        adata1.obs[prior_column]
        .astype("string")
        .fillna("Unknown")
    )
    broad = prior_labels.map(
        lambda value: _label_compartment(str(value)) or "Unknown"
    )
    geometry = _metaspot_compartment_guard_values(adata1.obs)
    if geometry is not None and bool(
        annotation_cfg.get("metaspot_compartment_correction", True)
    ):
        numeric_geometry = pd.to_numeric(geometry, errors="coerce")
        broad.loc[numeric_geometry.lt(0)] = "Nuclear"
        broad.loc[numeric_geometry.gt(0)] = "Cytoplasmic"
    fallback = broad.map(
        {"Nuclear": "Nucleus", "Cytoplasmic": "Cytosol"}
    ).fillna("Unknown")

    rep_key = str(
        annotation_cfg.get("metaspot_pool_rep", METASPOT_DGI_KEY)
    )
    pooling, pool_size = _build_same_cell_metaspot_dgi_pool(
        adata1,
        broad,
        rep_key=rep_key,
        n_neighbors=int(annotation_cfg.get("metaspot_pool_k", 8)),
    )
    marker_sets = _prepare_metaspot_marker_sets(
        gold_df,
        adata1.var_names,
        annotation_cfg,
    )
    labels = [
        "Nucleus",
        "Nucleolus",
        "Lamina",
        "Nuclear Pore",
        "Cytosol",
        "ER",
        "OMM",
    ]
    fine_structure_labels = [
        "Nucleolus",
        "Lamina",
        "Nuclear Pore",
        "ER",
        "OMM",
    ]
    cell_coverage_rescue_enabled = bool(
        annotation_cfg.get("metaspot_cell_coverage_rescue", False)
    )
    cell_coverage_min_dgi_support = _finite_float(
        "metaspot_cell_coverage_min_dgi_support",
        0.50,
    )
    cell_coverage_min_dgi_margin = _finite_float(
        "metaspot_cell_coverage_min_dgi_margin",
        0.0,
        lower=-1.0,
    )
    if (
        cell_coverage_min_dgi_support > 1.0
        or cell_coverage_min_dgi_margin > 1.0
    ):
        raise ValueError(
            "annotation MetaSpot cell-coverage DGI support and margin "
            "must both be <= 1"
        )
    try:
        cell_coverage_min_transcripts = int(
            annotation_cfg.get(
                "metaspot_cell_coverage_min_marker_transcripts",
                4,
            )
        )
        cell_coverage_min_genes = int(
            annotation_cfg.get(
                "metaspot_cell_coverage_min_marker_genes",
                3,
            )
        )
        cell_coverage_min_local_transcripts = int(
            annotation_cfg.get(
                "metaspot_cell_coverage_min_local_marker_transcripts", 1
            )
        )
        cell_coverage_min_local_genes = int(
            annotation_cfg.get(
                "metaspot_cell_coverage_min_local_marker_genes", 1
            )
        )
        cell_coverage_max_per_label = int(
            annotation_cfg.get(
                "metaspot_cell_coverage_max_per_label",
                1,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "MetaSpot cell-coverage marker minima and max_per_label must "
            "be integers"
        ) from exc
    if min(
        cell_coverage_min_transcripts,
        cell_coverage_min_genes,
        cell_coverage_min_local_transcripts,
        cell_coverage_min_local_genes,
        cell_coverage_max_per_label,
    ) < 1:
        raise ValueError(
            "MetaSpot cell-coverage marker minima and max_per_label must "
            "be >= 1"
        )
    cell_coverage_min_enrichment = _finite_float(
        "metaspot_cell_coverage_min_log2_enrichment",
        0.0,
        lower=-1.0,
    )
    cell_coverage_min_local_enrichment = _finite_float(
        "metaspot_cell_coverage_min_local_log2_enrichment", 0.0
    )
    cell_coverage_max_p = _finite_float(
        "metaspot_cell_coverage_max_binomial_p",
        0.50,
    )
    if cell_coverage_max_p > 1.0:
        raise ValueError(
            "annotation.metaspot_cell_coverage_max_binomial_p must be <= 1"
        )

    coverage_labels_raw = annotation_cfg.get(
        "metaspot_cell_coverage_labels",
        ["Nucleolus", "Lamina", "Nuclear Pore", "ER"],
    )
    if isinstance(coverage_labels_raw, str):
        coverage_labels_raw = [
            value.strip()
            for value in coverage_labels_raw.split(",")
            if value.strip()
        ]
    if not isinstance(coverage_labels_raw, (list, tuple)):
        raise ValueError(
            "annotation.metaspot_cell_coverage_labels must be a list or "
            "comma-separated string"
        )
    coverage_label_lookup = {
        label.casefold().replace("_", " "): label for label in labels
    }
    cell_coverage_labels: list[str] = []
    for raw_label in coverage_labels_raw:
        normalized = str(raw_label).strip().casefold().replace("_", " ")
        resolved_label = coverage_label_lookup.get(normalized)
        if resolved_label is None:
            raise ValueError(
                "annotation.metaspot_cell_coverage_labels contains "
                f"unknown label {raw_label!r}"
            )
        if resolved_label in {"Nucleus", "Cytosol", "OMM"}:
            raise ValueError(
                "MetaSpot cell-coverage rescue accepts only Nucleolus, "
                "Lamina, Nuclear Pore, and ER; OMM keeps its independent "
                "confirmation/strict-local gate"
            )
        if resolved_label not in cell_coverage_labels:
            cell_coverage_labels.append(resolved_label)

    first_locus_enabled = bool(
        annotation_cfg.get("metaspot_multistructure_first_locus", False)
    )
    first_locus_min_presence = _finite_float(
        "metaspot_multistructure_first_locus_min_presence", 0.55
    )
    first_locus_min_localization = _finite_float(
        "metaspot_multistructure_first_locus_min_localization", 0.55
    )
    spatial_coherence_raw = annotation_cfg.get(
        "metaspot_fine_min_spatial_coherence", 0.50
    )
    if isinstance(spatial_coherence_raw, Mapping):
        unknown_coherence_labels = set(spatial_coherence_raw).difference(
            fine_structure_labels
        )
        if unknown_coherence_labels:
            raise ValueError(
                "Unknown spatial-coherence structures: "
                f"{sorted(unknown_coherence_labels)}"
            )
        min_spatial_coherence = np.asarray(
            [
                float(spatial_coherence_raw.get(label, 0.50))
                for label in fine_structure_labels
            ],
            dtype=float,
        )
    else:
        min_spatial_coherence = np.full(
            len(fine_structure_labels),
            float(spatial_coherence_raw),
            dtype=float,
        )
    if (
        not np.isfinite(min_spatial_coherence).all()
        or (min_spatial_coherence < 0.0).any()
        or (min_spatial_coherence > 1.0).any()
    ):
        raise ValueError(
            "annotation.metaspot_fine_min_spatial_coherence values must be in [0, 1]"
        )
    first_locus_max_projected_burden = _finite_float(
        "metaspot_multistructure_first_locus_max_projected_burden", 0.20
    )
    if max(
        first_locus_min_presence,
        first_locus_min_localization,
        first_locus_max_projected_burden,
    ) > 1.0:
        raise ValueError(
            "MetaSpot first-locus score and burden thresholds must be <= 1"
        )
    first_locus_thresholds_raw = annotation_cfg.get(
        "metaspot_multistructure_first_locus_min_evidence",
        {},
    )
    if not isinstance(first_locus_thresholds_raw, Mapping):
        raise ValueError(
            "annotation.metaspot_multistructure_first_locus_min_evidence "
            "must be a mapping"
        )
    unknown_first_locus_labels = set(first_locus_thresholds_raw).difference(
        fine_structure_labels
    )
    if unknown_first_locus_labels:
        raise ValueError(
            "Unknown first-locus structures: "
            f"{sorted(unknown_first_locus_labels)}"
        )
    first_locus_min_evidence = np.asarray(
        [
            float(first_locus_thresholds_raw.get(label, 0.75))
            for label in fine_structure_labels
        ],
        dtype=float,
    )
    if not np.isfinite(first_locus_min_evidence).all() or (
        (first_locus_min_evidence < 0.0)
        | (first_locus_min_evidence > 1.0)
    ).any():
        raise ValueError(
            "MetaSpot first-locus evidence thresholds must be in [0, 1]"
        )

    diminishing_return_enabled = bool(
        annotation_cfg.get("metaspot_diminishing_return_selection", False)
    )
    diminishing_labels_raw = annotation_cfg.get(
        "metaspot_diminishing_return_labels",
        ["Nucleolus", "Lamina", "Nuclear Pore", "ER", "OMM"],
    )
    if isinstance(diminishing_labels_raw, str):
        diminishing_labels_raw = [
            value.strip()
            for value in diminishing_labels_raw.split(",")
            if value.strip()
        ]
    if not isinstance(diminishing_labels_raw, (list, tuple)):
        raise ValueError(
            "annotation.metaspot_diminishing_return_labels must be a list "
            "or comma-separated string"
        )
    diminishing_return_labels: list[str] = []
    for raw_label in diminishing_labels_raw:
        normalized = str(raw_label).strip().casefold().replace("_", " ")
        resolved_label = coverage_label_lookup.get(normalized)
        if resolved_label in {None, "Nucleus", "Cytosol"}:
            raise ValueError(
                "MetaSpot diminishing-return selection accepts only fine "
                f"structures; got {raw_label!r}"
            )
        if resolved_label not in diminishing_return_labels:
            diminishing_return_labels.append(resolved_label)
    diminishing_return_log_penalty = _finite_float(
        "metaspot_diminishing_return_log_penalty", 0.10
    )
    diminishing_return_burden_penalty = _finite_float(
        "metaspot_diminishing_return_burden_penalty", 0.50
    )
    diminishing_return_min_dgi_support = _finite_float(
        "metaspot_diminishing_return_min_dgi_support", 0.50
    )
    if diminishing_return_min_dgi_support > 1.0:
        raise ValueError(
            "annotation.metaspot_diminishing_return_min_dgi_support "
            "must be <= 1"
        )

    hierarchical_cell_rescue_enabled = bool(
        annotation_cfg.get("metaspot_hierarchical_cell_rescue", False)
    )
    try:
        hierarchical_min_cell_transcripts = int(
            annotation_cfg.get(
                "metaspot_hierarchical_min_cell_transcripts", 200
            )
        )
        hierarchical_min_metaspots = int(
            annotation_cfg.get("metaspot_hierarchical_min_metaspots", 20)
        )
        hierarchical_min_local_transcripts = int(
            annotation_cfg.get(
                "metaspot_hierarchical_min_local_marker_transcripts", 1
            )
        )
        hierarchical_min_local_genes = int(
            annotation_cfg.get(
                "metaspot_hierarchical_min_local_marker_genes", 1
            )
        )
        hierarchical_min_cell_marker_transcripts = int(
            annotation_cfg.get(
                "metaspot_hierarchical_min_cell_marker_transcripts", 1
            )
        )
        hierarchical_min_cell_marker_genes = int(
            annotation_cfg.get(
                "metaspot_hierarchical_min_cell_marker_genes", 1
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "MetaSpot hierarchical callability and marker minima must be "
            "integers"
        ) from exc
    if min(
        hierarchical_min_cell_transcripts,
        hierarchical_min_metaspots,
        hierarchical_min_local_transcripts,
        hierarchical_min_local_genes,
        hierarchical_min_cell_marker_transcripts,
        hierarchical_min_cell_marker_genes,
    ) < 1:
        raise ValueError(
            "MetaSpot hierarchical callability and marker minima must be "
            ">= 1"
        )
    hierarchical_min_presence_score = _finite_float(
        "metaspot_hierarchical_min_presence_score", 0.55
    )
    hierarchical_min_localization_score = _finite_float(
        "metaspot_hierarchical_min_localization_score", 0.55
    )
    hierarchical_min_joint_score = _finite_float(
        "metaspot_hierarchical_min_joint_score", 0.60
    )
    hierarchical_min_dgi_support = _finite_float(
        "metaspot_hierarchical_min_dgi_support", 0.35
    )
    hierarchical_min_dgi_margin = _finite_float(
        "metaspot_hierarchical_min_dgi_margin", -0.25, lower=-1.0
    )
    if max(
        hierarchical_min_presence_score,
        hierarchical_min_localization_score,
        hierarchical_min_joint_score,
        hierarchical_min_dgi_support,
        hierarchical_min_dgi_margin,
    ) > 1.0:
        raise ValueError(
            "MetaSpot hierarchical evidence scores, DGI support, and DGI "
            "margin must be <= 1"
        )

    omm_cell_rescue_enabled = bool(
        annotation_cfg.get("metaspot_omm_cell_coherent_rescue", False)
    )
    try:
        omm_cell_min_transcripts = int(
            annotation_cfg.get("metaspot_omm_cell_min_marker_transcripts", 8)
        )
        omm_cell_min_genes = int(
            annotation_cfg.get("metaspot_omm_cell_min_marker_genes", 4)
        )
        omm_local_min_transcripts = int(
            annotation_cfg.get("metaspot_omm_local_min_marker_transcripts", 2)
        )
        omm_local_min_genes = int(
            annotation_cfg.get("metaspot_omm_local_min_marker_genes", 2)
        )
        omm_cell_max_per_cell = int(
            annotation_cfg.get("metaspot_omm_cell_max_per_cell", 3)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "OMM cell-coherent marker minima and max_per_cell must be integers"
        ) from exc
    if min(
        omm_cell_min_transcripts,
        omm_cell_min_genes,
        omm_local_min_transcripts,
        omm_local_min_genes,
        omm_cell_max_per_cell,
    ) < 1:
        raise ValueError(
            "OMM cell-coherent marker minima and max_per_cell must be >= 1"
        )
    omm_cell_min_enrichment = _finite_float(
        "metaspot_omm_cell_min_log2_enrichment", 0.50
    )
    omm_cell_max_p = _finite_float(
        "metaspot_omm_cell_max_binomial_p", 0.05
    )
    omm_cell_exploratory_max_p = _finite_float(
        "metaspot_omm_cell_exploratory_max_binomial_p", 0.20
    )
    omm_local_min_enrichment = _finite_float(
        "metaspot_omm_local_min_log2_enrichment", 0.50
    )
    omm_local_min_gap = _finite_float(
        "metaspot_omm_local_min_score_gap", 0.15
    )
    omm_local_min_dgi_support = _finite_float(
        "metaspot_omm_local_min_dgi_support", 0.60
    )
    omm_local_min_dgi_margin = _finite_float(
        "metaspot_omm_local_min_dgi_margin", 0.0
    )
    if (
        omm_cell_max_p > 1.0
        or omm_cell_exploratory_max_p > 1.0
        or omm_local_min_dgi_support > 1.0
        or (
        omm_local_min_dgi_margin > 1.0
        )
    ):
        raise ValueError(
            "OMM cell-coherent p-value, DGI support, and DGI margin must be <= 1"
        )
    if omm_cell_exploratory_max_p < omm_cell_max_p:
        raise ValueError(
            "annotation.metaspot_omm_cell_exploratory_max_binomial_p "
            "must be >= metaspot_omm_cell_max_binomial_p"
        )
    allowed = {
        "Nuclear": ["Nucleus", "Nucleolus", "Lamina", "Nuclear Pore"],
        "Cytoplasmic": ["Cytosol", "ER", "OMM"],
    }
    threshold_arrays, thresholds_by_label = (
        _resolve_metaspot_label_thresholds(
            annotation_cfg,
            labels,
            {
                "min_marker_transcripts": min_transcripts,
                "min_marker_genes": min_genes,
                "min_log2_enrichment": min_enrichment,
                "min_score_gap": min_gap,
                "subtype_fdr": strict_fdr,
                "exploratory_max_q": exploratory_q,
                "min_dgi_support": _finite_float(
                    "metaspot_min_dgi_support",
                    0.0,
                ),
                "min_dgi_margin": _finite_float(
                    "metaspot_min_dgi_margin",
                    0.0,
                ),
            },
        )
    )
    marker_union = sorted(
        set().union(*(set(marker_sets[label]) for label in labels))
    )
    if not marker_union:
        raise RuntimeError(
            "No measured genes overlap the exclusive MetaSpot annotation "
            "marker sets"
        )

    var_position: Dict[str, int] = {}
    for position, gene in enumerate(adata1.var_names.astype(str)):
        var_position.setdefault(gene.upper(), position)
    matrix = adata1.layers.get("counts", adata1.X)
    if not _is_count_like_matrix(matrix):
        raise ValueError(
            "MetaSpot evidence testing requires raw integer transcript counts "
            "in adata1.X or adata1.layers['counts']"
        )
    marker_positions = [var_position[gene] for gene in marker_union]
    marker_matrix = matrix[:, marker_positions]
    if not sp.issparse(marker_matrix):
        marker_matrix = sp.csr_matrix(marker_matrix)
    marker_matrix = marker_matrix.tocsr().astype(np.float64)
    union_position = {gene: idx for idx, gene in enumerate(marker_union)}

    n_obs = adata1.n_obs
    n_labels = len(labels)
    original_counts = np.zeros((n_obs, n_labels), dtype=np.float64)
    original_distinct_genes = np.zeros(
        (n_obs, n_labels),
        dtype=np.int32,
    )
    pooled_matrix: Optional[sp.csr_matrix] = None
    pooled_counts: Optional[np.ndarray] = None
    distinct_genes: Optional[np.ndarray] = None
    if build_score_table:
        pooled_matrix = (pooling @ marker_matrix).tocsr()
        pooled_counts = np.zeros((n_obs, n_labels), dtype=np.float64)
        distinct_genes = np.zeros((n_obs, n_labels), dtype=np.int32)
    for label_idx, label in enumerate(labels):
        columns = np.asarray(
            [union_position[gene] for gene in marker_sets[label]],
            dtype=np.int64,
        )
        if columns.size == 0:
            continue
        original_label = marker_matrix[:, columns]
        original_counts[:, label_idx] = np.asarray(
            original_label.sum(axis=1)
        ).ravel()
        original_distinct_genes[:, label_idx] = original_label.getnnz(
            axis=1
        )
        if pooled_matrix is not None:
            assert pooled_counts is not None and distinct_genes is not None
            pooled_label = pooled_matrix[:, columns]
            pooled_counts[:, label_idx] = np.asarray(
                pooled_label.sum(axis=1)
            ).ravel()
            distinct_genes[:, label_idx] = pooled_label.getnnz(axis=1)

    broad_values = broad.astype(str).to_numpy()

    # Hard annotation evidence is computed from the focal MetaSpot only.
    # DGI-neighbor pooling is retained strictly for consensus/support and
    # diagnostics, so neighboring transcripts cannot inflate marker counts,
    # enrichment, or binomial significance for the focal MetaSpot.
    scores = np.full((n_obs, n_labels), -np.inf, dtype=np.float64)
    raw_scores = scores
    pooled_scores: Optional[np.ndarray] = None
    if build_score_table:
        pooled_scores = np.full(
            (n_obs, n_labels),
            -np.inf,
            dtype=np.float64,
        )
    p_values = np.full((n_obs, n_labels), np.nan, dtype=np.float64)
    q_values = np.full((n_obs, n_labels), np.nan, dtype=np.float64)
    local_totals = np.zeros(n_obs, dtype=np.float64)
    pooled_totals: Optional[np.ndarray] = None
    if build_score_table:
        pooled_totals = np.zeros(n_obs, dtype=np.float64)
    background_probability = np.full(n_labels, np.nan, dtype=np.float64)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    for compartment, compartment_labels in allowed.items():
        rows = broad_values == compartment
        label_indices = np.asarray(
            [label_to_idx[label] for label in compartment_labels],
            dtype=np.int64,
        )
        measured_indices = np.asarray(
            [
                label_idx
                for label_idx in label_indices
                if marker_sets[labels[label_idx]]
            ],
            dtype=np.int64,
        )
        if not rows.any() or measured_indices.size == 0:
            continue

        row_indices = np.flatnonzero(rows)

        # Self-local marker totals are the only totals used by the statistical
        # evidence gate.
        self_total = original_counts[
            np.ix_(row_indices, measured_indices)
        ].sum(axis=1)
        local_totals[row_indices] = self_total

        # Neighbor-pooled totals are diagnostic only.
        if pooled_counts is not None:
            pooled_total = pooled_counts[
                np.ix_(row_indices, measured_indices)
            ].sum(axis=1)
            assert pooled_totals is not None
            pooled_totals[row_indices] = pooled_total

        # Estimate the compartment-specific marker background from focal
        # MetaSpots, never from DGI-pooled pseudo-replicates.
        background_label_counts = original_counts[
            np.ix_(row_indices, measured_indices)
        ].sum(axis=0)
        background_total = float(background_label_counts.sum())
        alpha = 0.5
        background = (background_label_counts + alpha) / (
            background_total + alpha * measured_indices.size
        )

        for local_idx, label_idx in enumerate(measured_indices):
            bg = float(np.clip(background[local_idx], 1e-8, 1.0 - 1e-8))

            # ------------------------------------------------------------
            # Self-local evidence used by candidate selection and hard gate
            # ------------------------------------------------------------
            self_count = original_counts[row_indices, label_idx]
            self_fraction = (self_count + prior_strength * bg) / (
                self_total + prior_strength
            )
            self_enrichment = np.log2(
                np.maximum(self_fraction, 1e-12) / bg
            )
            scores[row_indices, label_idx] = self_enrichment
            background_probability[label_idx] = bg

            tested = self_total > 0
            tested_rows = row_indices[tested]
            if tested_rows.size:
                p_values[tested_rows, label_idx] = binom.sf(
                    np.rint(self_count[tested]).astype(np.int64) - 1,
                    np.rint(self_total[tested]).astype(np.int64),
                    bg,
                )

            # ------------------------------------------------------------
            # DGI-neighbor pooled enrichment retained for diagnostics only
            # ------------------------------------------------------------
            if pooled_counts is not None and pooled_scores is not None:
                pooled_count = pooled_counts[row_indices, label_idx]
                pooled_fraction = (pooled_count + prior_strength * bg) / (
                    pooled_total + prior_strength
                )
                pooled_scores[row_indices, label_idx] = np.log2(
                    np.maximum(pooled_fraction, 1e-12) / bg
                )

    for label_idx in range(n_labels):
        tested = np.isfinite(p_values[:, label_idx])
        if tested.any():
            q_values[tested, label_idx] = multipletests(
                p_values[tested, label_idx],
                method="fdr_bh",
            )[1]

    # Fine-structure RNA can be distributed over several small MetaSpots in a
    # cell. Aggregate exact counts and distinct genes separately within the
    # nuclear and cytoplasmic compartments. These arrays establish cell-level
    # presence only; the final location must still be anchored by focal marker
    # evidence and independent leave-one-out DGI agreement.
    cell_marker_counts = np.zeros((n_obs, n_labels), dtype=np.float64)
    cell_marker_genes = np.zeros((n_obs, n_labels), dtype=np.int32)
    cell_marker_enrichment = np.full(
        (n_obs, n_labels), np.nan, dtype=np.float64
    )
    cell_marker_p = np.full((n_obs, n_labels), np.nan, dtype=np.float64)
    cell_marker_q = np.full((n_obs, n_labels), np.nan, dtype=np.float64)
    cell_codes: Optional[np.ndarray] = None
    unique_cells: Optional[np.ndarray] = None
    if "cell_id" in adata1.obs.columns:
        cell_codes, unique_cells = pd.factorize(
            adata1.obs["cell_id"],
            sort=False,
        )
        n_cells = len(unique_cells)
        if n_cells <= np.iinfo(np.int32).max:
            cell_codes = cell_codes.astype(np.int32, copy=False)

        for compartment, compartment_labels in allowed.items():
            compartment_rows = np.flatnonzero(
                broad_values == compartment
            )
            if compartment_rows.size == 0:
                continue
            membership = sp.csr_matrix(
                (
                    np.ones(compartment_rows.size, dtype=np.float64),
                    (cell_codes[compartment_rows], compartment_rows),
                ),
                shape=(n_cells, n_obs),
            )
            cell_compartment_matrix = (membership @ marker_matrix).tocsr()
            compartment_columns = np.asarray(
                sorted(
                    {
                        union_position[gene]
                        for label in compartment_labels
                        for gene in marker_sets[label]
                    }
                ),
                dtype=np.int64,
            )
            cell_total_values = np.asarray(
                cell_compartment_matrix[:, compartment_columns].sum(axis=1)
            ).ravel()

            for label in compartment_labels:
                label_idx = label_to_idx[label]
                label_columns = np.asarray(
                    [
                        union_position[gene]
                        for gene in marker_sets[label]
                    ],
                    dtype=np.int64,
                )
                if label_columns.size == 0:
                    continue
                cell_count_values = np.asarray(
                    cell_compartment_matrix[:, label_columns].sum(axis=1)
                ).ravel()
                cell_gene_values = np.asarray(
                    cell_compartment_matrix[:, label_columns].getnnz(axis=1),
                    dtype=np.int32,
                )
                bg = float(background_probability[label_idx])
                if not np.isfinite(bg):
                    continue
                bg = float(np.clip(bg, 1e-8, 1.0 - 1e-8))
                cell_fraction = (
                    cell_count_values + prior_strength * bg
                ) / (cell_total_values + prior_strength)
                cell_enrichment_values = np.log2(
                    np.maximum(cell_fraction, 1e-12) / bg
                )
                cell_p_values = np.full(n_cells, np.nan, dtype=float)
                tested_cells = cell_total_values > 0
                cell_p_values[tested_cells] = binom.sf(
                    np.rint(cell_count_values[tested_cells]).astype(np.int64)
                    - 1,
                    np.rint(cell_total_values[tested_cells]).astype(np.int64),
                    bg,
                )
                cell_q_values = np.full(n_cells, np.nan, dtype=float)
                finite_test = np.isfinite(cell_p_values)
                if finite_test.any():
                    cell_q_values[finite_test] = multipletests(
                        cell_p_values[finite_test], method="fdr_bh"
                    )[1]

                cell_marker_counts[:, label_idx] = cell_count_values[
                    cell_codes
                ]
                cell_marker_genes[:, label_idx] = cell_gene_values[
                    cell_codes
                ]
                cell_marker_enrichment[:, label_idx] = (
                    cell_enrichment_values[cell_codes]
                )
                cell_marker_p[:, label_idx] = cell_p_values[cell_codes]
                cell_marker_q[:, label_idx] = cell_q_values[cell_codes]

    rescue_cell_ids: Sequence[object] = (
        cell_codes if cell_codes is not None else adata1.obs["cell_id"]
    )
    omm_idx = label_to_idx["OMM"]
    omm_cell_counts = cell_marker_counts[:, omm_idx]
    omm_cell_genes = cell_marker_genes[:, omm_idx]
    omm_cell_enrichment = cell_marker_enrichment[:, omm_idx]
    omm_cell_p = cell_marker_p[:, omm_idx]
    omm_cell_q = cell_marker_q[:, omm_idx]

    (
        marker_candidate_indices,
        marker_candidate_gap,
        dgi_support,
        dgi_informative_neighbors,
    ) = _compute_metaspot_dgi_consensus(
        pooling,
        raw_scores,
        original_counts,
        original_distinct_genes,
        broad_values,
        labels,
        allowed,
        seed_min_transcripts=dgi_seed_min_transcripts,
        seed_min_genes=dgi_seed_min_genes,
        seed_min_enrichment=dgi_seed_min_enrichment,
        min_informative_neighbors=dgi_min_informative_neighbors,
    )
    dgi_available = (
        dgi_informative_neighbors >= dgi_min_informative_neighbors
    )

    fine_structure_indices = np.asarray(
        [label_to_idx[label] for label in fine_structure_labels],
        dtype=np.int64,
    )
    metaspot_total_transcripts = np.asarray(
        matrix.sum(axis=1), dtype=float
    ).ravel()
    structure_compartments = {
        label: str(_label_compartment(label))
        for label in fine_structure_labels
    }
    multistructure_spatial_coherence = (
        _compute_cell_structure_spatial_coherence(
            cell_codes,
            broad_values,
            original_counts[:, fine_structure_indices],
            metaspot_total_transcripts,
            fine_structure_labels,
            structure_compartments,
        )
    )
    adata1.obsm["X_subcellular_structure_spatial_coherence"] = (
        multistructure_spatial_coherence
    )
    metadomain_affinity = np.zeros(
        (n_obs, len(fine_structure_labels)), dtype=float
    )
    rapx_calibrated_score = np.full_like(metadomain_affinity, np.nan)
    structure_geometry = np.zeros_like(metadomain_affinity, dtype=bool)
    competitor_margin = np.full_like(metadomain_affinity, np.nan)
    for fine_idx, label in enumerate(fine_structure_labels):
        label_idx = label_to_idx[label]
        compartment = str(_label_compartment(label))
        compatible_indices = [
            label_to_idx[other]
            for other in allowed[compartment]
            if other != label
        ]
        alternative = np.max(scores[:, compatible_indices], axis=1)
        finite_margin = np.isfinite(scores[:, label_idx]) & np.isfinite(
            alternative
        )
        competitor_margin[finite_margin, fine_idx] = (
            scores[finite_margin, label_idx] - alternative[finite_margin]
        )

        affinity_column = f"subcell_relative_evidence__{label}"
        if affinity_column in adata1.obs.columns:
            metadomain_affinity[:, fine_idx] = pd.to_numeric(
                adata1.obs[affinity_column], errors="coerce"
            ).fillna(0.0)
        rapx_column = f"subcell_subtype_calibrated_score__{label}"
        if rapx_column in adata1.obs.columns:
            rapx_calibrated_score[:, fine_idx] = pd.to_numeric(
                adata1.obs[rapx_column], errors="coerce"
            )
        geometry_column = f"subcell_geometry_feasible__{label}"
        structure_geometry[:, fine_idx] = broad_values == compartment
        if geometry_column in adata1.obs.columns:
            structure_geometry[:, fine_idx] &= (
                adata1.obs[geometry_column].fillna(False).astype(bool)
            )

    (
        multistructure_presence,
        multistructure_localization,
        multistructure_evidence,
    ) = _compute_multistructure_evidence(
        original_counts[:, fine_structure_indices],
        original_distinct_genes[:, fine_structure_indices],
        scores[:, fine_structure_indices],
        metadomain_affinity,
        dgi_support[:, fine_structure_indices],
        structure_geometry,
        competitor_margin,
        rapx_calibrated_score,
        cell_marker_counts[:, fine_structure_indices],
        cell_marker_genes[:, fine_structure_indices],
        cell_marker_enrichment[:, fine_structure_indices],
        cell_marker_p[:, fine_structure_indices],
    )
    adata1.obsm["X_subcellular_structure_presence"] = (
        multistructure_presence
    )
    adata1.obsm["X_subcellular_structure_localization"] = (
        multistructure_localization
    )
    adata1.obsm["X_subcellular_structure_evidence"] = (
        multistructure_evidence
    )

    # Fused scores are retained as an audit quantity only.  They do not choose
    # the candidate and do not enter the count/enrichment/q-value hard gate.
    fused_scores: Optional[np.ndarray] = None
    if build_score_table:
        fused_scores = scores.copy()
    if fused_scores is not None and dgi_consensus_weight > 0.0:
        for label_idx, label in enumerate(labels):
            if label in {"Nucleus", "Cytosol"}:
                continue
            eligible = dgi_available & np.isfinite(
                fused_scores[:, label_idx]
            )
            fused_scores[eligible, label_idx] += (
                dgi_consensus_weight * dgi_support[eligible, label_idx]
            )

    # Candidate selection is self-local. DGI can support/rescue a candidate
    # later, but cannot create a candidate by importing neighbor transcripts.
    candidate = np.asarray(fallback.astype(str), dtype=object)
    score_gap = np.full(n_obs, np.nan, dtype=np.float64)
    for compartment, compartment_labels in allowed.items():
        rows = np.flatnonzero(broad_values == compartment)
        label_indices = np.asarray(
            [label_to_idx[label] for label in compartment_labels],
            dtype=np.int64,
        )
        if rows.size == 0:
            continue

        local_scores = scores[np.ix_(rows, label_indices)]
        best_local = np.argmax(local_scores, axis=1)
        best_scores = local_scores[np.arange(rows.size), best_local]
        finite_best = np.isfinite(best_scores)
        chosen_indices = label_indices[best_local]

        if finite_best.any():
            candidate[rows[finite_best]] = np.asarray(
                labels,
                dtype=object,
            )[chosen_indices[finite_best]]

        if label_indices.size > 1:
            sorted_scores = np.sort(local_scores, axis=1)
            gaps = sorted_scores[:, -1] - sorted_scores[:, -2]
            gaps[~np.isfinite(gaps)] = np.nan
            score_gap[rows] = gaps

    candidate_indices = pd.Categorical(
        candidate,
        categories=labels,
    ).codes.astype(np.int64, copy=False)
    if np.any(candidate_indices < 0):
        raise RuntimeError("MetaSpot candidate labels escaped the seven-label space")
    row_ids = np.arange(n_obs)

    # All hard-gate quantities below come from the focal MetaSpot itself.
    candidate_counts = original_counts[row_ids, candidate_indices]
    candidate_genes = original_distinct_genes[row_ids, candidate_indices]
    candidate_scores = scores[row_ids, candidate_indices]
    candidate_fused_scores = (
        fused_scores[row_ids, candidate_indices]
        if fused_scores is not None
        else None
    )
    candidate_p = p_values[row_ids, candidate_indices]
    candidate_q = q_values[row_ids, candidate_indices]
    candidate_cell_counts = cell_marker_counts[row_ids, candidate_indices]
    candidate_cell_genes = cell_marker_genes[row_ids, candidate_indices]
    candidate_cell_enrichment = cell_marker_enrichment[
        row_ids, candidate_indices
    ]
    candidate_cell_p = cell_marker_p[row_ids, candidate_indices]
    candidate_cell_q = cell_marker_q[row_ids, candidate_indices]

    # Neighbor-pooled values are kept only for diagnostics.
    candidate_pooled_counts = (
        pooled_counts[row_ids, candidate_indices]
        if pooled_counts is not None
        else None
    )
    candidate_pooled_genes = (
        distinct_genes[row_ids, candidate_indices]
        if distinct_genes is not None
        else None
    )
    candidate_pooled_scores = (
        pooled_scores[row_ids, candidate_indices]
        if pooled_scores is not None
        else None
    )
    marker_only_candidate = np.asarray(
        fallback.astype(str),
        dtype=object,
    ).copy()
    marker_candidate_valid = marker_candidate_indices >= 0
    marker_only_candidate[marker_candidate_valid] = np.asarray(
        labels,
        dtype=object,
    )[marker_candidate_indices[marker_candidate_valid]]
    candidate_dgi_support = dgi_support[row_ids, candidate_indices]
    candidate_dgi_margin = np.full(n_obs, np.nan, dtype=np.float64)
    for compartment, compartment_labels in allowed.items():
        rows = np.flatnonzero(broad_values == compartment)
        label_indices = np.asarray(
            [label_to_idx[label] for label in compartment_labels],
            dtype=np.int64,
        )
        if rows.size == 0:
            continue
        local_support = dgi_support[np.ix_(rows, label_indices)].copy()
        local_lookup = np.full(n_labels, -1, dtype=np.int8)
        local_lookup[label_indices] = np.arange(
            label_indices.size,
            dtype=np.int8,
        )
        selected_local = local_lookup[candidate_indices[rows]]
        valid_selected = selected_local >= 0
        local_support[
            np.arange(rows.size)[valid_selected],
            selected_local[valid_selected],
        ] = -np.inf
        alternative_support = np.max(local_support, axis=1)
        alternative_support[~np.isfinite(alternative_support)] = 0.0
        candidate_dgi_margin[rows] = (
            candidate_dgi_support[rows] - alternative_support
        )
    candidate_is_subtype = ~np.isin(
        candidate,
        ["Nucleus", "Cytosol", "Unknown"],
    )
    candidate_spatial_coherence = np.ones(n_obs, dtype=float)
    candidate_spatial_coherence_threshold = np.zeros(n_obs, dtype=float)
    for fine_idx, label in enumerate(fine_structure_labels):
        rows = candidate == label
        candidate_spatial_coherence[rows] = (
            multistructure_spatial_coherence[rows, fine_idx]
        )
        candidate_spatial_coherence_threshold[rows] = (
            min_spatial_coherence[fine_idx]
        )
    candidate_spatial_coherence_pass = (
        candidate_spatial_coherence >= candidate_spatial_coherence_threshold
    )

    candidate_min_transcripts = threshold_arrays[
        "min_marker_transcripts"
    ][candidate_indices]
    candidate_min_genes = threshold_arrays["min_marker_genes"][
        candidate_indices
    ]
    candidate_min_enrichment = threshold_arrays[
        "min_log2_enrichment"
    ][candidate_indices]
    candidate_min_gap = threshold_arrays["min_score_gap"][
        candidate_indices
    ]
    candidate_strict_fdr = threshold_arrays["subtype_fdr"][
        candidate_indices
    ]
    candidate_exploratory_q = threshold_arrays[
        "exploratory_max_q"
    ][candidate_indices]
    candidate_min_dgi_support = threshold_arrays[
        "min_dgi_support"
    ][candidate_indices]
    candidate_min_dgi_margin = threshold_arrays["min_dgi_margin"][
        candidate_indices
    ]
    enough_counts = candidate_counts >= candidate_min_transcripts
    enough_genes = candidate_genes >= candidate_min_genes
    enough_enrichment = candidate_scores >= candidate_min_enrichment
    enough_gap = score_gap >= candidate_min_gap
    needs_dgi_support = (
        require_dgi_support
        | (candidate_min_dgi_support > 0.0)
        | (candidate_min_dgi_margin > 0.0)
    )
    enough_dgi_support = (
        ~needs_dgi_support
        | (
            dgi_available
            & (candidate_dgi_support >= candidate_min_dgi_support)
            & (candidate_dgi_margin >= candidate_min_dgi_margin)
        )
    )
    marker_base_evidence = (
        candidate_is_subtype
        & enough_counts
        & enough_genes
        & enough_enrichment
        & enough_gap
        & np.isfinite(candidate_q)
        & candidate_spatial_coherence_pass
    )
    local_base_evidence = marker_base_evidence & enough_dgi_support

    # This is deliberately a separate exploratory track rather than an FDR
    # call. It may rescue a marker-qualified candidate only when independent,
    # leave-one-out DGI neighbors agree strongly. This rescue never upgrades a
    # call to Strict; that tier retains the marker test and label-wise BH gate.
    dgi_exploratory_support_threshold = np.maximum(
        dgi_exploratory_min_support,
        candidate_min_dgi_support,
    )
    dgi_exploratory_margin_threshold = np.maximum(
        dgi_exploratory_min_margin,
        candidate_min_dgi_margin,
    )
    dgi_exploratory_supported = (
        dgi_exploratory_enabled
        & marker_base_evidence
        & dgi_available
        & (
            candidate_dgi_support
            >= dgi_exploratory_support_threshold
        )
        & (
            candidate_dgi_margin
            >= dgi_exploratory_margin_threshold
        )
    )

    require_omm = bool(
        annotation_cfg.get("metaspot_require_omm_confirmation", True)
    )
    omm_confirmation_mode, omm_confirmed = (
        _resolve_metaspot_omm_confirmation(
            adata1,
            require_confirmation=require_omm,
            posterior_high=float(
                annotation_cfg.get("omm_metaspot_posterior_high", 0.80)
            ),
        )
    )
    is_omm = candidate == "OMM"
    local_strict_supported = local_base_evidence & (
        candidate_q <= candidate_strict_fdr
    )
    statistical_exploratory_supported = local_base_evidence & (
        candidate_q <= candidate_exploratory_q
    )
    dgi_exploratory_supported &= ~statistical_exploratory_supported
    local_exploratory_supported = (
        statistical_exploratory_supported
        | dgi_exploratory_supported
    )
    strict_supported = local_strict_supported.copy()
    exploratory_supported = local_exploratory_supported.copy()
    if require_omm and omm_confirmation_mode == "identifiable":
        strict_supported[is_omm] &= omm_confirmed[is_omm]
        exploratory_supported[is_omm] &= omm_confirmed[is_omm]
    elif require_omm and omm_confirmation_mode in {
        "unidentifiable",
        "unavailable",
    }:
        if omm_unidentifiable_policy == "strict_local":
            exploratory_supported[is_omm] = local_strict_supported[is_omm]
        else:
            # A missing independent confirmation is uncertainty, not positive
            # OMM evidence. Keep strict-local candidates in diagnostics but do
            # not turn them into hard production labels.
            strict_supported[is_omm] = False
            exploratory_supported[is_omm] = False

    supported_before_cell_coverage = exploratory_supported.copy()
    omm_cell_coherent_rescued = np.zeros(n_obs, dtype=bool)
    omm_cell_exploratory_candidate = np.zeros(n_obs, dtype=bool)
    if omm_cell_rescue_enabled:
        omm_idx = label_to_idx["OMM"]
        omm_cell_coherent_rescued = (
            _select_metaspot_omm_cell_coherent_rescues(
                rescue_cell_ids,
                candidate,
                exploratory_supported,
                broad_values,
                original_counts[:, omm_idx],
                original_distinct_genes[:, omm_idx],
                scores[:, omm_idx],
                score_gap,
                dgi_available,
                dgi_support[:, omm_idx],
                (
                    dgi_support[:, omm_idx]
                    - np.maximum(
                        dgi_support[:, label_to_idx["ER"]],
                        dgi_support[:, label_to_idx["Cytosol"]],
                    )
                ),
                omm_cell_counts,
                omm_cell_genes,
                omm_cell_enrichment,
                omm_cell_p,
                min_cell_transcripts=omm_cell_min_transcripts,
                min_cell_genes=omm_cell_min_genes,
                min_cell_enrichment=omm_cell_min_enrichment,
                max_cell_p=omm_cell_max_p,
                min_local_transcripts=omm_local_min_transcripts,
                min_local_genes=omm_local_min_genes,
                min_local_enrichment=omm_local_min_enrichment,
                min_local_score_gap=omm_local_min_gap,
                min_dgi_support=omm_local_min_dgi_support,
                min_dgi_margin=omm_local_min_dgi_margin,
                max_per_cell=omm_cell_max_per_cell,
            )
        )
        omm_cell_coherent_rescued &= (
            multistructure_spatial_coherence[:, fine_structure_labels.index("OMM")]
            >= min_spatial_coherence[fine_structure_labels.index("OMM")]
        )
        # An identifiable independent model still has veto authority. The
        # cell-coherent path is primarily a principled fallback for the
        # currently unidentifiable OMM confirmation model.
        if require_omm and omm_confirmation_mode == "identifiable":
            omm_cell_coherent_rescued &= omm_confirmed
        exploratory_supported |= omm_cell_coherent_rescued
        if omm_cell_exploratory_max_p > omm_cell_max_p:
            # A second, explicitly audit-only sensitivity layer.  It uses the
            # identical local marker, OMM-vs-ER/Cytosol, and leave-one-out DGI
            # gates; only the cell-level presence p-value is relaxed.  These
            # loci never enter ``exploratory_supported`` or public labels.
            omm_cell_exploratory_candidate = (
                _select_metaspot_omm_cell_coherent_rescues(
                    rescue_cell_ids,
                    candidate,
                    exploratory_supported,
                    broad_values,
                    original_counts[:, omm_idx],
                    original_distinct_genes[:, omm_idx],
                    scores[:, omm_idx],
                    score_gap,
                    dgi_available,
                    dgi_support[:, omm_idx],
                    (
                        dgi_support[:, omm_idx]
                        - np.maximum(
                            dgi_support[:, label_to_idx["ER"]],
                            dgi_support[:, label_to_idx["Cytosol"]],
                        )
                    ),
                    omm_cell_counts,
                    omm_cell_genes,
                    omm_cell_enrichment,
                    omm_cell_p,
                    min_cell_transcripts=omm_cell_min_transcripts,
                    min_cell_genes=omm_cell_min_genes,
                    min_cell_enrichment=omm_cell_min_enrichment,
                    max_cell_p=omm_cell_exploratory_max_p,
                    min_local_transcripts=omm_local_min_transcripts,
                    min_local_genes=omm_local_min_genes,
                    min_local_enrichment=omm_local_min_enrichment,
                    min_local_score_gap=omm_local_min_gap,
                    min_dgi_support=omm_local_min_dgi_support,
                    min_dgi_margin=omm_local_min_dgi_margin,
                    max_per_cell=omm_cell_max_per_cell,
                )
            )

    cell_coverage_rescued = np.zeros(n_obs, dtype=bool)
    cell_coverage_rescue_labels = np.full(n_obs, "", dtype=object)
    labelwise_cell_coverage_rescued = np.zeros(n_obs, dtype=bool)
    if cell_coverage_rescue_enabled:
        if "cell_id" not in adata1.obs.columns:
            raise KeyError(
                "MetaSpot cell-coverage rescue requires adata1.obs['cell_id']"
            )
        cell_coverage_rescued = _select_metaspot_cell_coverage_rescues(
            rescue_cell_ids,
            candidate,
            exploratory_supported,
            marker_base_evidence,
            dgi_available,
            candidate_dgi_support,
            candidate_dgi_margin,
            candidate_scores,
            score_gap,
            candidate_q,
            candidate_counts,
            candidate_cell_counts,
            candidate_cell_genes,
            candidate_cell_enrichment,
            candidate_cell_p,
            labels=cell_coverage_labels,
            min_cell_transcripts=cell_coverage_min_transcripts,
            min_cell_genes=cell_coverage_min_genes,
            min_cell_enrichment=cell_coverage_min_enrichment,
            max_cell_p=cell_coverage_max_p,
            min_dgi_support=cell_coverage_min_dgi_support,
            min_dgi_margin=cell_coverage_min_dgi_margin,
            max_per_cell_label=cell_coverage_max_per_label,
        )
        exploratory_supported |= cell_coverage_rescued
        cell_coverage_rescue_labels[cell_coverage_rescued] = candidate[
            cell_coverage_rescued
        ]

        # The primary head chooses one maximum-enrichment label before its
        # gates are evaluated. Independently test every still-missing label
        # against its own marker, exact cell-level, and DGI evidence. This
        # recovers real secondary structure signals without turning the
        # feature into a per-cell quota.
        labelwise_rescue_labels = (
            _select_metaspot_labelwise_cell_coverage_rescues(
                rescue_cell_ids,
                broad_values,
                candidate,
                exploratory_supported,
                original_counts,
                original_distinct_genes,
                scores,
                q_values,
                dgi_available,
                dgi_support,
                cell_marker_counts,
                cell_marker_genes,
                cell_marker_enrichment,
                cell_marker_p,
                labels=cell_coverage_labels,
                all_labels=labels,
                allowed=allowed,
                min_local_transcripts=np.full(
                    len(labels), cell_coverage_min_local_transcripts
                ),
                min_local_genes=np.full(
                    len(labels), cell_coverage_min_local_genes
                ),
                min_local_enrichment=np.full(
                    len(labels), cell_coverage_min_local_enrichment
                ),
                min_cell_transcripts=cell_coverage_min_transcripts,
                min_cell_genes=cell_coverage_min_genes,
                min_cell_enrichment=cell_coverage_min_enrichment,
                max_cell_p=cell_coverage_max_p,
                min_dgi_support=cell_coverage_min_dgi_support,
                min_dgi_margin=cell_coverage_min_dgi_margin,
                max_per_cell_label=cell_coverage_max_per_label,
            )
        )
        labelwise_cell_coverage_rescued = labelwise_rescue_labels != ""
        for fine_idx, label in enumerate(fine_structure_labels):
            rows = labelwise_rescue_labels == label
            labelwise_rescue_labels[
                rows
                & (
                    multistructure_spatial_coherence[:, fine_idx]
                    < min_spatial_coherence[fine_idx]
                )
            ] = ""
        labelwise_cell_coverage_rescued = labelwise_rescue_labels != ""
        cell_coverage_rescue_labels[
            labelwise_cell_coverage_rescued
        ] = labelwise_rescue_labels[labelwise_cell_coverage_rescued]
        cell_coverage_rescued |= labelwise_cell_coverage_rescued
        exploratory_supported |= labelwise_cell_coverage_rescued

    supported_before_hierarchical_rescue = exploratory_supported.copy()
    hierarchical_rescue_labels = np.full(n_obs, "", dtype=object)
    hierarchical_diagnostics: Dict[str, np.ndarray] = {
        "callable": np.zeros(n_obs, dtype=bool),
        "eligible_cell": np.zeros(n_obs, dtype=bool),
        "status": np.full(n_obs, "disabled", dtype=object),
        "cell_total_transcripts": np.full(n_obs, np.nan, dtype=float),
        "cell_metaspots": np.full(n_obs, np.nan, dtype=float),
        "presence_score": np.full(n_obs, np.nan, dtype=float),
        "localization_score": np.full(n_obs, np.nan, dtype=float),
        "joint_score": np.full(n_obs, np.nan, dtype=float),
        "best_label": np.full(n_obs, "", dtype=object),
        "best_presence_score": np.full(n_obs, np.nan, dtype=float),
        "best_localization_score": np.full(n_obs, np.nan, dtype=float),
        "best_joint_score": np.full(n_obs, np.nan, dtype=float),
        "best_dgi_support": np.full(n_obs, np.nan, dtype=float),
        "best_dgi_margin": np.full(n_obs, np.nan, dtype=float),
    }
    if hierarchical_cell_rescue_enabled:
        if "cell_id" not in adata1.obs.columns:
            raise KeyError(
                "MetaSpot hierarchical cell rescue requires "
                "adata1.obs['cell_id']"
            )
        metaspot_total_transcripts = np.asarray(
            matrix.sum(axis=1), dtype=float
        ).ravel()
        (
            hierarchical_rescue_labels,
            hierarchical_diagnostics,
        ) = _select_metaspot_hierarchical_cell_rescues(
            rescue_cell_ids,
            broad_values,
            exploratory_supported,
            original_counts,
            original_distinct_genes,
            scores,
            dgi_available,
            dgi_support,
            cell_marker_counts,
            cell_marker_genes,
            cell_marker_enrichment,
            cell_marker_p,
            metaspot_total_transcripts,
            labels=cell_coverage_labels,
            all_labels=labels,
            allowed=allowed,
            min_cell_transcripts=hierarchical_min_cell_transcripts,
            min_metaspots=hierarchical_min_metaspots,
            min_presence_score=hierarchical_min_presence_score,
            min_localization_score=(
                hierarchical_min_localization_score
            ),
            min_joint_score=hierarchical_min_joint_score,
            min_local_marker_transcripts=(
                hierarchical_min_local_transcripts
            ),
            min_local_marker_genes=hierarchical_min_local_genes,
            min_cell_marker_transcripts=(
                hierarchical_min_cell_marker_transcripts
            ),
            min_cell_marker_genes=(
                hierarchical_min_cell_marker_genes
            ),
            min_dgi_support=hierarchical_min_dgi_support,
            min_dgi_margin=hierarchical_min_dgi_margin,
        )
    hierarchical_cell_rescued = hierarchical_rescue_labels != ""
    for fine_idx, label in enumerate(fine_structure_labels):
        rows = hierarchical_rescue_labels == label
        hierarchical_rescue_labels[
            rows
            & (
                multistructure_spatial_coherence[:, fine_idx]
                < min_spatial_coherence[fine_idx]
            )
        ] = ""
    hierarchical_cell_rescued = hierarchical_rescue_labels != ""
    cell_coverage_rescue_labels[hierarchical_cell_rescued] = (
        hierarchical_rescue_labels[hierarchical_cell_rescued]
    )
    exploratory_supported |= hierarchical_cell_rescued
    all_cell_coverage_rescued = (
        cell_coverage_rescued | hierarchical_cell_rescued
    )

    cell_coverage_local_counts = np.full(n_obs, np.nan, dtype=float)
    cell_coverage_local_genes = np.full(n_obs, np.nan, dtype=float)
    cell_coverage_local_enrichment = np.full(n_obs, np.nan, dtype=float)
    cell_coverage_dgi_support = np.full(n_obs, np.nan, dtype=float)
    cell_coverage_dgi_margin = np.full(n_obs, np.nan, dtype=float)
    cell_coverage_cell_counts = np.full(n_obs, np.nan, dtype=float)
    cell_coverage_cell_genes = np.full(n_obs, np.nan, dtype=float)
    cell_coverage_cell_enrichment = np.full(n_obs, np.nan, dtype=float)
    cell_coverage_cell_p = np.full(n_obs, np.nan, dtype=float)
    for rescue_label in cell_coverage_labels:
        rescue_rows = np.flatnonzero(
            all_cell_coverage_rescued
            & (cell_coverage_rescue_labels == rescue_label)
        )
        if rescue_rows.size == 0:
            continue
        rescue_idx = label_to_idx[rescue_label]
        compartment = _label_compartment(rescue_label)
        alternative_indices = np.asarray(
            [
                label_to_idx[label]
                for label in allowed[str(compartment)]
                if label != rescue_label
            ],
            dtype=np.int64,
        )
        alternative_support = np.max(
            dgi_support[np.ix_(rescue_rows, alternative_indices)],
            axis=1,
        ) if alternative_indices.size else np.zeros(rescue_rows.size)
        cell_coverage_local_counts[rescue_rows] = original_counts[
            rescue_rows, rescue_idx
        ]
        cell_coverage_local_genes[rescue_rows] = original_distinct_genes[
            rescue_rows, rescue_idx
        ]
        cell_coverage_local_enrichment[rescue_rows] = scores[
            rescue_rows, rescue_idx
        ]
        cell_coverage_dgi_support[rescue_rows] = dgi_support[
            rescue_rows, rescue_idx
        ]
        cell_coverage_dgi_margin[rescue_rows] = (
            dgi_support[rescue_rows, rescue_idx] - alternative_support
        )
        cell_coverage_cell_counts[rescue_rows] = cell_marker_counts[
            rescue_rows, rescue_idx
        ]
        cell_coverage_cell_genes[rescue_rows] = cell_marker_genes[
            rescue_rows, rescue_idx
        ]
        cell_coverage_cell_enrichment[rescue_rows] = cell_marker_enrichment[
            rescue_rows, rescue_idx
        ]
        cell_coverage_cell_p[rescue_rows] = cell_marker_p[
            rescue_rows, rescue_idx
        ]

    strict_labels = np.asarray(fallback.astype(str), dtype=object)
    exploratory_labels = np.asarray(fallback.astype(str), dtype=object)
    strict_labels[strict_supported] = candidate[strict_supported]
    exploratory_labels[exploratory_supported] = candidate[
        exploratory_supported
    ]
    exploratory_labels[labelwise_cell_coverage_rescued] = (
        cell_coverage_rescue_labels[labelwise_cell_coverage_rescued]
    )
    exploratory_labels[hierarchical_cell_rescued] = (
        hierarchical_rescue_labels[hierarchical_cell_rescued]
    )
    supported_after_hierarchical_rescue = exploratory_supported.copy()
    first_locus_labels, first_locus_diagnostics = (
        _select_metaspot_multistructure_first_loci(
            rescue_cell_ids,
            exploratory_labels,
            exploratory_supported,
            multistructure_presence,
            multistructure_localization,
            multistructure_evidence,
            multistructure_spatial_coherence,
            structure_geometry,
            original_counts[:, fine_structure_indices],
            original_distinct_genes[:, fine_structure_indices],
            cell_marker_counts[:, fine_structure_indices],
            cell_marker_genes[:, fine_structure_indices],
            cell_marker_enrichment[:, fine_structure_indices],
            cell_marker_p[:, fine_structure_indices],
            dgi_available,
            metaspot_total_transcripts,
            structures=(fine_structure_labels if first_locus_enabled else []),
            min_presence_score=first_locus_min_presence,
            min_localization_score=first_locus_min_localization,
            min_spatial_coherence=min_spatial_coherence,
            min_evidence_score=first_locus_min_evidence,
            min_local_marker_transcripts=np.asarray(
                [1] * len(fine_structure_labels)
            ),
            min_local_marker_genes=np.asarray(
                [1] * len(fine_structure_labels)
            ),
            min_cell_marker_transcripts=np.asarray(
                [cell_coverage_min_transcripts] * 4
                + [omm_cell_min_transcripts]
            ),
            min_cell_marker_genes=np.asarray(
                [cell_coverage_min_genes] * 4 + [omm_cell_min_genes]
            ),
            min_cell_marker_enrichment=np.asarray(
                [cell_coverage_min_enrichment] * 4 + [omm_cell_min_enrichment]
            ),
            max_cell_marker_p=np.asarray(
                [cell_coverage_max_p] * 4 + [omm_cell_max_p]
            ),
            max_projected_burden=first_locus_max_projected_burden,
        )
    )
    first_locus_selected = first_locus_labels != ""
    exploratory_labels[first_locus_selected] = first_locus_labels[
        first_locus_selected
    ]
    exploratory_supported[first_locus_selected] = True
    final_labels = exploratory_labels.copy()
    diminishing_return_labels_before = final_labels.copy()
    _, diminishing_return_diagnostics = (
        _apply_metaspot_diminishing_return_selection(
            rescue_cell_ids,
            final_labels,
            strict_supported,
            broad_values,
            scores,
            dgi_support,
            metaspot_total_transcripts,
            enabled_labels=(
                diminishing_return_labels
                if diminishing_return_enabled
                else []
            ),
            all_labels=labels,
            allowed=allowed,
            min_local_enrichment=threshold_arrays[
                "min_log2_enrichment"
            ],
            dgi_weight=dgi_consensus_weight,
            min_dgi_support=diminishing_return_min_dgi_support,
            log_penalty=diminishing_return_log_penalty,
            burden_penalty=diminishing_return_burden_penalty,
        )
    )
    diminishing_return_pruned = diminishing_return_diagnostics["pruned"]
    exploratory_supported[diminishing_return_pruned] = False
    exploratory_labels[diminishing_return_pruned] = np.asarray(
        fallback.astype(str), dtype=object
    )[diminishing_return_pruned]
    final_labels[diminishing_return_pruned] = exploratory_labels[
        diminishing_return_pruned
    ]
    omm_evidence_tier = np.full(n_obs, "Not selected", dtype=object)
    omm_evidence_tier[omm_cell_exploratory_candidate] = "Exploratory"
    omm_evidence_tier[final_labels == "OMM"] = "Supported"
    omm_evidence_tier[
        diminishing_return_pruned
        & (diminishing_return_labels_before == "OMM")
    ] = "Diminishing-return pruned"
    tier = np.full(n_obs, "Broad-prior fallback", dtype=object)
    tier[exploratory_supported] = "Exploratory local evidence"
    tier[
        dgi_exploratory_supported & exploratory_supported
    ] = "DGI-supported exploratory evidence"
    tier[strict_supported] = "Strict local evidence"
    decision_provenance = np.full(
        n_obs,
        "broad_prior_fallback",
        dtype=object,
    )
    decision_provenance[exploratory_supported] = "local_exploratory"
    decision_provenance[
        dgi_exploratory_supported & exploratory_supported
    ] = "local_marker_plus_dgi_exploratory"
    decision_provenance[strict_supported] = "local_strict"
    tier[omm_cell_coherent_rescued] = "Cell-coherent OMM evidence"
    decision_provenance[omm_cell_coherent_rescued] = (
        "cell_multigene_plus_local_dgi_omm"
    )
    tier[cell_coverage_rescued] = "Cell-coherent fine-structure evidence"
    decision_provenance[cell_coverage_rescued] = (
        "cell_multigene_plus_local_marker_plus_dgi"
    )
    decision_provenance[labelwise_cell_coverage_rescued] = (
        "labelwise_cell_multigene_plus_local_marker_plus_dgi"
    )
    tier[hierarchical_cell_rescued] = (
        "Tier-B callable-cell multiscale evidence"
    )
    decision_provenance[hierarchical_cell_rescued] = (
        "callable_cell_presence_plus_local_marker_plus_dgi"
    )
    tier[first_locus_selected] = "Multi-structure supported first locus"
    decision_provenance[first_locus_selected] = (
        "cell_presence_plus_multistructure_localization_first_locus"
    )

    omm_confirmation_status = np.full(
        n_obs,
        "not_applicable",
        dtype=object,
    )
    omm_strict_local_candidate = np.zeros(n_obs, dtype=bool)
    omm_strict_local_fallback = np.zeros(n_obs, dtype=bool)
    if omm_confirmation_mode == "disabled":
        omm_confirmation_status[is_omm] = "confirmation_disabled"
    elif omm_confirmation_mode == "identifiable":
        omm_confirmation_status[is_omm] = "not_confirmed"
        omm_confirmation_status[is_omm & omm_confirmed] = "confirmed"
        confirmed_supported = is_omm & exploratory_supported
        decision_provenance[confirmed_supported] = (
            "local_plus_omm_confirmation_exploratory"
        )
        decision_provenance[
            confirmed_supported & dgi_exploratory_supported
        ] = (
            "local_marker_plus_dgi_plus_omm_confirmation_exploratory"
        )
        decision_provenance[is_omm & strict_supported] = (
            "local_plus_omm_confirmation_strict"
        )
    else:
        omm_strict_local_candidate = is_omm & local_strict_supported
        omm_confirmation_status[is_omm] = (
            f"{omm_confirmation_mode}_strict_local_required"
        )
        if omm_unidentifiable_policy == "strict_local":
            omm_strict_local_fallback = omm_strict_local_candidate.copy()
            omm_confirmation_status[omm_strict_local_fallback] = (
                f"{omm_confirmation_mode}_strict_local_fallback"
            )
            tier[omm_strict_local_fallback] = (
                "Strict local evidence "
                f"(OMM confirmation {omm_confirmation_mode})"
            )
            decision_provenance[omm_strict_local_fallback] = (
                "local_only_strict_omm_confirmation_"
                f"{omm_confirmation_mode}"
            )
        else:
            omm_confirmation_status[omm_strict_local_candidate] = (
                f"{omm_confirmation_mode}_strict_local_unresolved"
            )
    omm_confirmation_status[omm_cell_coherent_rescued] = (
        f"{omm_confirmation_mode}_cell_coherent_fallback"
    )
    omm_confirmation_status[omm_cell_exploratory_candidate] = (
        f"{omm_confirmation_mode}_cell_coherent_exploratory"
    )
    omm_confirmation_status[
        first_locus_selected & (first_locus_labels == "OMM")
    ] = f"{omm_confirmation_mode}_multistructure_first_locus"
    tier[diminishing_return_pruned] = "Diminishing-return pruned"
    decision_provenance[diminishing_return_pruned] = (
        "diminishing_return_pruned"
    )

    fallback_reason = np.full(n_obs, "broad_label_wins", dtype=object)
    subtype_rows = candidate_is_subtype
    fallback_reason[subtype_rows] = "q_above_exploratory_threshold"
    fallback_reason[subtype_rows & ~candidate_spatial_coherence_pass] = (
        "insufficient_spatial_coherence"
    )
    fallback_reason[subtype_rows & ~enough_dgi_support] = (
        "insufficient_dgi_support"
    )
    fallback_reason[subtype_rows & ~enough_gap] = "ambiguous_local_scores"
    fallback_reason[subtype_rows & ~enough_enrichment] = "weak_enrichment"
    fallback_reason[subtype_rows & ~enough_genes] = (
        "insufficient_marker_genes"
    )
    fallback_reason[subtype_rows & ~enough_counts] = (
        "insufficient_marker_transcripts"
    )
    fallback_reason[subtype_rows & (local_totals <= 0)] = (
        "no_measured_reference_markers"
    )
    if require_omm and omm_confirmation_mode == "identifiable":
        confirmation_veto = (
            is_omm & local_exploratory_supported & ~omm_confirmed
        )
        fallback_reason[confirmation_veto] = "omm_not_confirmed"
    elif require_omm and omm_confirmation_mode in {
        "unidentifiable",
        "unavailable",
    }:
        if omm_unidentifiable_policy == "strict_local":
            strict_required = (
                is_omm
                & local_exploratory_supported
                & ~local_strict_supported
            )
            fallback_reason[strict_required] = (
                f"omm_confirmation_{omm_confirmation_mode}_"
                "requires_strict_local"
            )
        else:
            unresolved_omm = is_omm & local_exploratory_supported
            fallback_reason[unresolved_omm] = (
                f"omm_confirmation_{omm_confirmation_mode}_unresolved"
            )
    fallback_reason[omm_cell_exploratory_candidate] = (
        "omm_cell_evidence_exploratory_only"
    )
    fallback_reason[exploratory_supported] = "none"
    fallback_reason[diminishing_return_pruned] = (
        "diminishing_return_penalty"
    )

    # Production output must use evidence-gated labels.  Unsupported
    # fine-structure candidates fall back to the broad Nucleus/Cytosol prior.
    fallback_array = np.asarray(fallback.astype(str), dtype=object)
    fine_labels = np.asarray(final_labels, dtype=object).copy()

    invalid_fine_label = (
        pd.isna(fine_labels)
        | np.isin(
            fine_labels.astype(str),
            ["", "Unknown", "nan", "None"],
        )
    )
    fine_labels[invalid_fine_label] = fallback_array[invalid_fine_label]

    # This state records why a public label was emitted.  It does not alter
    # the label itself.
    fine_state = np.full(
        n_obs,
        "Broad-prior fallback",
        dtype=object,
    )
    fine_state[
        exploratory_supported
    ] = "Exploratory-supported fine structure"
    fine_state[
        dgi_exploratory_supported & exploratory_supported
    ] = "DGI-supported exploratory fine structure"
    fine_state[
        strict_supported
    ] = "Strict-supported fine structure"
    fine_state[cell_coverage_rescued] = (
        "Cell-coherent multi-gene fine structure"
    )
    fine_state[hierarchical_cell_rescued] = (
        "Tier-B callable-cell fine structure"
    )
    fine_state[first_locus_selected] = (
        "Multi-structure evidence-supported first locus"
    )
    fine_state[omm_cell_coherent_rescued] = (
        "Cell-coherent multi-gene OMM evidence"
    )
    fine_state[diminishing_return_pruned] = "Diminishing-return pruned"

    dgi_branch_used = (
        candidate_is_subtype
        & dgi_available
        & (dgi_consensus_weight > 0.0)
    )

    adata1.obs = adata1.obs.drop(
        columns=[
            "subcellular_annotation_compatibility",
            "subcellular_annotation_metaspot_compatibility",
        ],
        errors="ignore",
    )
    # Public final label uses the same evidence-gated production decision as
    # the MetaSpot production column. Build every public/audit field first and
    # attach them in one operation so obs does not acquire one block per field.
    _replace_obs_columns(
        adata1,
        {
            "subcellular_annotation_metaspot_parent": pd.Categorical(fallback),
            "subcellular_annotation_metaspot_candidate": pd.Categorical(candidate),
            "subcellular_annotation_metaspot_strict": pd.Categorical(strict_labels),
            "subcellular_annotation_metaspot_exploratory": pd.Categorical(exploratory_labels),
            "subcellular_annotation_metaspot": pd.Categorical(final_labels),
            "subcellular_annotation": pd.Categorical(fine_labels),
            "subcellular_candidate": candidate,
            "subcellular_broad_compartment": pd.Categorical(broad_values),
            "subcellular_fine_annotation": pd.Categorical(fine_labels),
            "subcellular_fine_state": pd.Categorical(fine_state),
            "subcellular_fine_supported": exploratory_supported,
            "subcellular_metaspot_marker_only_candidate": pd.Categorical(marker_only_candidate),
            "subcellular_annotation_strict": pd.Categorical(strict_labels),
            "subcellular_annotation_exploratory": pd.Categorical(exploratory_labels),
            "subcellular_annotation_tier": pd.Categorical(tier),
            "subcellular_metaspot_decision_provenance": pd.Categorical(decision_provenance),
            "subcellular_metaspot_omm_confirmation_status": pd.Categorical(omm_confirmation_status),
            "subcellular_metaspot_pool_size": pool_size,
            "subcellular_metaspot_candidate_marker_transcripts": candidate_counts.astype(np.float32),
            "subcellular_metaspot_candidate_marker_genes": candidate_genes,
            "subcellular_metaspot_log2_enrichment": candidate_scores.astype(np.float32),
            "subcellular_metaspot_spatial_coherence": candidate_spatial_coherence.astype(np.float32),
            "subcellular_metaspot_spatial_coherence_pass": candidate_spatial_coherence_pass,
            **(
                {
                    "subcellular_metaspot_candidate_neighbor_pooled_marker_transcripts": candidate_pooled_counts.astype(np.float32),
                    "subcellular_metaspot_candidate_neighbor_pooled_marker_genes": candidate_pooled_genes,
                    "subcellular_metaspot_neighbor_pooled_log2_enrichment": candidate_pooled_scores.astype(np.float32),
                    "subcellular_metaspot_fused_score": candidate_fused_scores.astype(np.float32),
                }
                if build_score_table
                else {}
            ),
            "subcellular_metaspot_marker_only_score_gap": marker_candidate_gap.astype(np.float32),
            "subcellular_metaspot_score_gap": score_gap.astype(np.float32),
            "subcellular_metaspot_dgi_support": candidate_dgi_support.astype(np.float32),
            "subcellular_metaspot_dgi_margin": candidate_dgi_margin.astype(np.float32),
            "subcellular_metaspot_dgi_informative_neighbors": dgi_informative_neighbors,
            "subcellular_metaspot_dgi_branch_used": dgi_branch_used,
            "subcellular_metaspot_dgi_exploratory_supported": dgi_exploratory_supported & exploratory_supported,
            "subcellular_metaspot_cell_coverage_rescued": cell_coverage_rescued,
            "subcellular_metaspot_labelwise_cell_coverage_rescued": labelwise_cell_coverage_rescued,
            "subcellular_metaspot_hierarchical_cell_rescued": hierarchical_cell_rescued,
            "subcellular_metaspot_cell_coverage_rescue_label": pd.Categorical(cell_coverage_rescue_labels),
            "subcellular_metaspot_hierarchical_cell_callable": hierarchical_diagnostics["callable"],
            "subcellular_metaspot_hierarchical_cell_eligible": hierarchical_diagnostics["eligible_cell"],
            "subcellular_metaspot_hierarchical_cell_status": pd.Categorical(hierarchical_diagnostics["status"]),
            "subcellular_metaspot_hierarchical_best_label": pd.Categorical(hierarchical_diagnostics["best_label"]),
            "subcellular_metaspot_hierarchical_cell_total_transcripts": hierarchical_diagnostics["cell_total_transcripts"].astype(np.float32),
            "subcellular_metaspot_hierarchical_cell_metaspots": hierarchical_diagnostics["cell_metaspots"].astype(np.float32),
            "subcellular_metaspot_hierarchical_presence_score": hierarchical_diagnostics["presence_score"].astype(np.float32),
            "subcellular_metaspot_hierarchical_localization_score": hierarchical_diagnostics["localization_score"].astype(np.float32),
            "subcellular_metaspot_hierarchical_joint_score": hierarchical_diagnostics["joint_score"].astype(np.float32),
            "subcellular_metaspot_hierarchical_best_presence_score": hierarchical_diagnostics["best_presence_score"].astype(np.float32),
            "subcellular_metaspot_hierarchical_best_localization_score": hierarchical_diagnostics["best_localization_score"].astype(np.float32),
            "subcellular_metaspot_hierarchical_best_joint_score": hierarchical_diagnostics["best_joint_score"].astype(np.float32),
            "subcellular_metaspot_hierarchical_best_dgi_support": hierarchical_diagnostics["best_dgi_support"].astype(np.float32),
            "subcellular_metaspot_hierarchical_best_dgi_margin": hierarchical_diagnostics["best_dgi_margin"].astype(np.float32),
            "subcellular_metaspot_multistructure_first_locus_selected": first_locus_selected,
            "subcellular_metaspot_multistructure_first_locus_label": pd.Categorical(first_locus_labels),
            "subcellular_metaspot_multistructure_first_locus_eligible": first_locus_diagnostics["eligible"],
            "subcellular_metaspot_multistructure_first_locus_burden_rejected": first_locus_diagnostics["burden_rejected"],
            "subcellular_metaspot_multistructure_first_locus_presence_score": first_locus_diagnostics["presence_score"].astype(np.float32),
            "subcellular_metaspot_multistructure_first_locus_localization_score": first_locus_diagnostics["localization_score"].astype(np.float32),
            "subcellular_metaspot_multistructure_first_locus_evidence_score": first_locus_diagnostics["evidence_score"].astype(np.float32),
            "subcellular_metaspot_multistructure_first_locus_spatial_coherence": first_locus_diagnostics["spatial_coherence"].astype(np.float32),
            "subcellular_metaspot_multistructure_first_locus_projected_burden": first_locus_diagnostics["projected_burden"].astype(np.float32),
            "subcellular_metaspot_diminishing_return_eligible": diminishing_return_diagnostics["eligible"],
            "subcellular_metaspot_diminishing_return_pruned": diminishing_return_pruned,
            "subcellular_metaspot_diminishing_return_original_label": pd.Categorical(diminishing_return_labels_before),
            "subcellular_metaspot_diminishing_return_first_locus": diminishing_return_diagnostics["first_locus"],
            "subcellular_metaspot_diminishing_return_rank": diminishing_return_diagnostics["rank"],
            "subcellular_metaspot_diminishing_return_evidence_score": diminishing_return_diagnostics["evidence_score"].astype(np.float32),
            "subcellular_metaspot_diminishing_return_required_score": diminishing_return_diagnostics["required_score"].astype(np.float32),
            "subcellular_metaspot_diminishing_return_burden_before": diminishing_return_diagnostics["burden_before"].astype(np.float32),
            "subcellular_metaspot_cell_coverage_local_marker_transcripts": cell_coverage_local_counts.astype(np.float32),
            "subcellular_metaspot_cell_coverage_local_marker_genes": cell_coverage_local_genes,
            "subcellular_metaspot_cell_coverage_local_log2_enrichment": cell_coverage_local_enrichment.astype(np.float32),
            "subcellular_metaspot_cell_coverage_dgi_support": cell_coverage_dgi_support.astype(np.float32),
            "subcellular_metaspot_cell_coverage_dgi_margin": cell_coverage_dgi_margin.astype(np.float32),
            "subcellular_metaspot_cell_coverage_cell_marker_transcripts": cell_coverage_cell_counts.astype(np.float32),
            "subcellular_metaspot_cell_coverage_cell_marker_genes": cell_coverage_cell_genes,
            "subcellular_metaspot_cell_coverage_cell_log2_enrichment": cell_coverage_cell_enrichment.astype(np.float32),
            "subcellular_metaspot_cell_coverage_cell_binomial_p": cell_coverage_cell_p,
            "subcellular_metaspot_cell_marker_transcripts": candidate_cell_counts.astype(np.float32),
            "subcellular_metaspot_cell_marker_genes": candidate_cell_genes,
            "subcellular_metaspot_cell_log2_enrichment": candidate_cell_enrichment.astype(np.float32),
            "subcellular_metaspot_cell_binomial_p": candidate_cell_p,
            "subcellular_metaspot_cell_bh_q": candidate_cell_q,
            "subcellular_metaspot_omm_cell_coherent_rescued": omm_cell_coherent_rescued,
            "subcellular_metaspot_omm_exploratory_candidate": omm_cell_exploratory_candidate,
            "subcellular_metaspot_omm_evidence_tier": pd.Categorical(omm_evidence_tier),
            "subcellular_metaspot_omm_cell_marker_transcripts": omm_cell_counts.astype(np.float32),
            "subcellular_metaspot_omm_cell_marker_genes": omm_cell_genes,
            "subcellular_metaspot_omm_cell_log2_enrichment": omm_cell_enrichment.astype(np.float32),
            "subcellular_metaspot_omm_cell_binomial_p": omm_cell_p,
            "subcellular_metaspot_omm_cell_bh_q": omm_cell_q,
            "subcellular_metaspot_binomial_p": candidate_p,
            "subcellular_metaspot_bh_q": candidate_q,
            "subcellular_metaspot_supported": exploratory_supported,
            "subcellular_metaspot_candidate_min_marker_transcripts": candidate_min_transcripts,
            "subcellular_metaspot_candidate_min_marker_genes": candidate_min_genes,
            "subcellular_metaspot_candidate_min_log2_enrichment": candidate_min_enrichment.astype(np.float32),
            "subcellular_metaspot_candidate_min_score_gap": candidate_min_gap.astype(np.float32),
            "subcellular_metaspot_candidate_strict_fdr": candidate_strict_fdr.astype(np.float32),
            "subcellular_metaspot_candidate_exploratory_max_q": candidate_exploratory_q.astype(np.float32),
            "subcellular_metaspot_fallback_reason": pd.Categorical(fallback_reason),
        },
    )
    obs = adata1.obs

    score_table = (
        pd.DataFrame(
            {
            "metaspot": obs.get(
                "metaspot",
                pd.Series(obs.index, index=obs.index),
            ).astype(str).to_numpy(),
            "cell_id": obs["cell_id"].astype(str).to_numpy(),
            "broad_compartment": broad_values,
            "broad_parent": fallback.astype(str).to_numpy(),
            "fine_annotation": fine_labels,
            "fine_state": fine_state,
            "fine_supported": exploratory_supported,
            "marker_only_candidate": marker_only_candidate,
            "candidate": candidate,
            "strict_annotation": strict_labels,
            "exploratory_annotation": exploratory_labels,
            "final_annotation": fine_labels,
            "evidence_tier": tier,
            "decision_provenance": decision_provenance,
            "omm_confirmation_status": omm_confirmation_status,
            "fallback_reason": fallback_reason,
            "pool_size": pool_size,
            "candidate_marker_transcripts": candidate_counts,
            "candidate_marker_genes": candidate_genes,
            "candidate_log2_enrichment": candidate_scores,
            "candidate_spatial_coherence": candidate_spatial_coherence,
            "candidate_spatial_coherence_pass": candidate_spatial_coherence_pass,
            "candidate_neighbor_pooled_marker_transcripts": (
                candidate_pooled_counts
            ),
            "candidate_neighbor_pooled_marker_genes": (
                candidate_pooled_genes
            ),
            "candidate_neighbor_pooled_log2_enrichment": (
                candidate_pooled_scores
            ),
            "candidate_fused_score": candidate_fused_scores,
            "marker_only_score_gap": marker_candidate_gap,
            "candidate_score_gap": score_gap,
            "candidate_dgi_support": candidate_dgi_support,
            "candidate_dgi_margin": candidate_dgi_margin,
            "dgi_informative_neighbors": dgi_informative_neighbors,
            "dgi_branch_used": dgi_branch_used,
            "dgi_exploratory_supported": (
                dgi_exploratory_supported & exploratory_supported
            ),
            "cell_coverage_rescued": cell_coverage_rescued,
            "labelwise_cell_coverage_rescued": (
                labelwise_cell_coverage_rescued
            ),
            "hierarchical_cell_rescued": hierarchical_cell_rescued,
            "cell_coverage_rescue_label": cell_coverage_rescue_labels,
            "hierarchical_cell_callable": hierarchical_diagnostics[
                "callable"
            ],
            "hierarchical_cell_eligible": hierarchical_diagnostics[
                "eligible_cell"
            ],
            "hierarchical_cell_status": hierarchical_diagnostics[
                "status"
            ],
            "hierarchical_best_label": hierarchical_diagnostics[
                "best_label"
            ],
            "hierarchical_cell_total_transcripts": hierarchical_diagnostics[
                "cell_total_transcripts"
            ],
            "hierarchical_cell_metaspots": hierarchical_diagnostics[
                "cell_metaspots"
            ],
            "hierarchical_presence_score": hierarchical_diagnostics[
                "presence_score"
            ],
            "hierarchical_localization_score": hierarchical_diagnostics[
                "localization_score"
            ],
            "hierarchical_joint_score": hierarchical_diagnostics[
                "joint_score"
            ],
            "hierarchical_best_presence_score": hierarchical_diagnostics[
                "best_presence_score"
            ],
            "hierarchical_best_localization_score": hierarchical_diagnostics[
                "best_localization_score"
            ],
            "hierarchical_best_joint_score": hierarchical_diagnostics[
                "best_joint_score"
            ],
            "hierarchical_best_dgi_support": hierarchical_diagnostics[
                "best_dgi_support"
            ],
            "hierarchical_best_dgi_margin": hierarchical_diagnostics[
                "best_dgi_margin"
            ],
            "multistructure_first_locus_selected": first_locus_selected,
            "multistructure_first_locus_label": first_locus_labels,
            "multistructure_first_locus_eligible": first_locus_diagnostics[
                "eligible"
            ],
            "multistructure_first_locus_burden_rejected": (
                first_locus_diagnostics["burden_rejected"]
            ),
            "multistructure_first_locus_presence_score": (
                first_locus_diagnostics["presence_score"]
            ),
            "multistructure_first_locus_localization_score": (
                first_locus_diagnostics["localization_score"]
            ),
            "multistructure_first_locus_evidence_score": (
                first_locus_diagnostics["evidence_score"]
            ),
            "multistructure_first_locus_spatial_coherence": (
                first_locus_diagnostics["spatial_coherence"]
            ),
            "multistructure_first_locus_projected_burden": (
                first_locus_diagnostics["projected_burden"]
            ),
            "diminishing_return_eligible": diminishing_return_diagnostics[
                "eligible"
            ],
            "diminishing_return_pruned": diminishing_return_pruned,
            "diminishing_return_original_label": (
                diminishing_return_labels_before
            ),
            "diminishing_return_first_locus": diminishing_return_diagnostics[
                "first_locus"
            ],
            "diminishing_return_rank": diminishing_return_diagnostics[
                "rank"
            ],
            "diminishing_return_evidence_score": diminishing_return_diagnostics[
                "evidence_score"
            ],
            "diminishing_return_required_score": diminishing_return_diagnostics[
                "required_score"
            ],
            "diminishing_return_burden_before": diminishing_return_diagnostics[
                "burden_before"
            ],
            "cell_coverage_local_marker_transcripts": (
                cell_coverage_local_counts
            ),
            "cell_coverage_local_marker_genes": cell_coverage_local_genes,
            "cell_coverage_local_log2_enrichment": (
                cell_coverage_local_enrichment
            ),
            "cell_coverage_dgi_support": cell_coverage_dgi_support,
            "cell_coverage_dgi_margin": cell_coverage_dgi_margin,
            "cell_coverage_cell_marker_transcripts": (
                cell_coverage_cell_counts
            ),
            "cell_coverage_cell_marker_genes": cell_coverage_cell_genes,
            "cell_coverage_cell_log2_enrichment": (
                cell_coverage_cell_enrichment
            ),
            "cell_coverage_cell_binomial_p": cell_coverage_cell_p,
            "candidate_cell_marker_transcripts": candidate_cell_counts,
            "candidate_cell_marker_genes": candidate_cell_genes,
            "candidate_cell_log2_enrichment": candidate_cell_enrichment,
            "candidate_cell_binomial_p": candidate_cell_p,
            "candidate_cell_bh_q": candidate_cell_q,
            "omm_cell_coherent_rescued": omm_cell_coherent_rescued,
            "omm_cell_exploratory_candidate": omm_cell_exploratory_candidate,
            "omm_evidence_tier": omm_evidence_tier,
            "omm_cell_marker_transcripts": omm_cell_counts,
            "omm_cell_marker_genes": omm_cell_genes,
            "omm_cell_log2_enrichment": omm_cell_enrichment,
            "omm_cell_binomial_p": omm_cell_p,
            "omm_cell_bh_q": omm_cell_q,
            "candidate_binomial_p": candidate_p,
            "candidate_bh_q": candidate_q,
            "candidate_min_marker_transcripts": (
                candidate_min_transcripts
            ),
            "candidate_min_marker_genes": candidate_min_genes,
            "candidate_min_log2_enrichment": candidate_min_enrichment,
            "candidate_min_score_gap": candidate_min_gap,
            "candidate_strict_fdr": candidate_strict_fdr,
            "candidate_exploratory_max_q": candidate_exploratory_q,
            "candidate_min_dgi_support": candidate_min_dgi_support,
            "candidate_min_dgi_margin": candidate_min_dgi_margin,
            "subtype_supported": exploratory_supported,
            }
        )
        if build_score_table
        else None
    )
    for label_idx, label in enumerate(labels):
        if score_table is None:
            break
        suffix = label.lower().replace(" ", "_")
        score_table[f"marker_transcripts__{suffix}"] = original_counts[
            :, label_idx
        ]
        score_table[f"marker_genes__{suffix}"] = original_distinct_genes[
            :, label_idx
        ]
        score_table[f"log2_enrichment__{suffix}"] = scores[:, label_idx]
        score_table[f"bh_q__{suffix}"] = q_values[:, label_idx]
        score_table[
            f"neighbor_pooled_marker_transcripts__{suffix}"
        ] = pooled_counts[:, label_idx]
        score_table[
            f"neighbor_pooled_log2_enrichment__{suffix}"
        ] = pooled_scores[:, label_idx]
        score_table[f"dgi_support__{suffix}"] = dgi_support[:, label_idx]
        score_table[f"fused_score__{suffix}"] = fused_scores[:, label_idx]
        if label in fine_structure_labels:
            fine_idx = fine_structure_labels.index(label)
            score_table[f"presence_score__{suffix}"] = (
                multistructure_presence[:, fine_idx]
            )
            score_table[f"localization_score__{suffix}"] = (
                multistructure_localization[:, fine_idx]
            )
            score_table[f"structure_evidence__{suffix}"] = (
                multistructure_evidence[:, fine_idx]
            )
            score_table[f"spatial_coherence__{suffix}"] = (
                multistructure_spatial_coherence[:, fine_idx]
            )

    if cell_codes is None or unique_cells is None:
        metadata_cell_codes, metadata_unique_cells = pd.factorize(
            adata1.obs["cell_id"],
            sort=False,
        )
    else:
        metadata_cell_codes = cell_codes
        metadata_unique_cells = unique_cells
    if len(metadata_unique_cells) <= np.iinfo(np.int32).max:
        metadata_cell_codes = metadata_cell_codes.astype(
            np.int32,
            copy=False,
        )
    metadata_valid_cells = metadata_cell_codes >= 0
    metadata_n_cells = len(metadata_unique_cells)

    def _cell_presence(mask: Sequence[bool]) -> np.ndarray:
        mask_values = np.asarray(mask, dtype=bool)
        selected = metadata_cell_codes[metadata_valid_cells & mask_values]
        if selected.size == 0:
            return np.zeros(metadata_n_cells, dtype=bool)
        return np.bincount(
            selected,
            minlength=metadata_n_cells,
        ).astype(bool, copy=False)

    callable_cell_presence = _cell_presence(
        hierarchical_diagnostics["callable"]
    )
    fine_cells_before_hierarchical = _cell_presence(
        supported_before_hierarchical_rescue
    )
    fine_cells_after_hierarchical = _cell_presence(
        supported_after_hierarchical_rescue
    )
    callable_cells_with_fine_before = int(
        np.count_nonzero(
            callable_cell_presence & fine_cells_before_hierarchical
        )
    )
    callable_cells_with_fine_after = int(
        np.count_nonzero(
            callable_cell_presence & fine_cells_after_hierarchical
        )
    )
    callable_cell_count = int(np.count_nonzero(callable_cell_presence))
    cell_coverage_rescued_cell_count = int(
        np.count_nonzero(_cell_presence(cell_coverage_rescued))
    )
    hierarchical_rescued_cell_count = int(
        np.count_nonzero(_cell_presence(hierarchical_cell_rescued))
    )
    first_locus_selected_cell_count = int(
        np.count_nonzero(_cell_presence(first_locus_selected))
    )
    omm_rescued_cell_count = int(
        np.count_nonzero(_cell_presence(omm_cell_coherent_rescued))
    )
    omm_exploratory_cell_count = int(
        np.count_nonzero(_cell_presence(omm_cell_exploratory_candidate))
    )
    cell_coverage_cells_before_by_label = {
        label: int(
            np.count_nonzero(
                _cell_presence(
                    supported_before_cell_coverage & (candidate == label)
                )
            )
        )
        for label in cell_coverage_labels
    }
    cell_coverage_cells_after_by_label = {
        label: int(
            np.count_nonzero(
                _cell_presence(
                    exploratory_supported & (exploratory_labels == label)
                )
            )
        )
        for label in cell_coverage_labels
    }
    valid_positions = np.flatnonzero(metadata_valid_cells)
    _, first_valid_offsets = np.unique(
        metadata_cell_codes[metadata_valid_cells],
        return_index=True,
    )
    first_cell_rows = valid_positions[first_valid_offsets]
    hierarchical_status_counts = {
        str(key): int(value)
        for key, value in pd.Series(
            np.asarray(hierarchical_diagnostics["status"])[first_cell_rows]
        ).value_counts().items()
    }

    metadata: Dict[str, Any] = {
        "schema_version": 11,
        "mode": (
            "metadomain_broad_prior_plus_self_metaspot_marker_evidence_"
            "plus_leave_one_out_dgi_consensus"
        ),
        "hard_evidence_source": "self_metaspot_only",
        "neighbor_pooling_used_for_hard_evidence": False,
        "neighbor_pooling_role": "dgi_consensus_and_diagnostics_only",
        "candidate_selection_source": (
            "primary_self_metaspot_log2_enrichment_plus_labelwise_"
            "cell_coherent_rescue_plus_callable_cell_multiscale_rescue_"
            "plus_multistructure_first_locus"
        ),
        "binomial_test_source": "self_metaspot_marker_counts",
        "prior_unit": "metadomain",
        "prior_label_column": prior_column,
        "decision_unit": "metaspot",
        "multi_structure_evidence": {
            "schema_version": 3,
            "structures": list(fine_structure_labels),
            "interpretation": "bounded_evidence_score_not_probability",
            "used_for_final_selection": first_locus_enabled,
            "metaspot_obsm_keys": {
                "presence": "X_subcellular_structure_presence",
                "localization": "X_subcellular_structure_localization",
                "joint": "X_subcellular_structure_evidence",
                "spatial_coherence": "X_subcellular_structure_spatial_coherence",
            },
            "cell_uns_key": "cell_structure_evidence",
            "presence_weights": {
                "cell_marker_transcripts": 0.30,
                "cell_marker_genes": 0.25,
                "cell_log2_enrichment": 0.25,
                "one_minus_cell_binomial_p": 0.20,
            },
            "localization_weights": {
                "focal_marker": 0.25,
                "metadomain_affinity": 0.20,
                "leave_one_out_dgi_support": 0.20,
                "geometry_compatibility": 0.10,
                "competitor_separation": 0.15,
                "rapx_calibrated_evidence": 0.10,
            },
            "joint": "geometric_mean",
            "selection_role": "missing_cell_structure_first_locus_only",
        },
        "spatial_coherence": {
            "threshold_by_structure": {
                label: float(min_spatial_coherence[idx])
                for idx, label in enumerate(fine_structure_labels)
            },
            "definition": (
                "total_variation_between_structure_marker_and_all_transcript_"
                "metaspot_distributions_within_cell_parent_compartment"
            ),
            "role": "shared_hard_gate_for_all_fine_structure_paths",
        },
        "pool_rep": rep_key,
        "pool_k": int(annotation_cfg.get("metaspot_pool_k", 8)),
        "dgi_consensus_weight": dgi_consensus_weight,
        "dgi_seed_min_marker_transcripts": dgi_seed_min_transcripts,
        "dgi_seed_min_marker_genes": dgi_seed_min_genes,
        "dgi_seed_min_log2_enrichment": dgi_seed_min_enrichment,
        "dgi_min_informative_neighbors": (
            dgi_min_informative_neighbors
        ),
        "dgi_exploratory_enabled": dgi_exploratory_enabled,
        "dgi_exploratory_min_support": dgi_exploratory_min_support,
        "dgi_exploratory_min_margin": dgi_exploratory_min_margin,
        "require_dgi_support": require_dgi_support,
        "dgi_branch_used_metaspots": int(dgi_branch_used.sum()),
        "dgi_exploratory_supported_metaspots": int(
            (dgi_exploratory_supported & exploratory_supported).sum()
        ),
        "multistructure_first_locus_enabled": first_locus_enabled,
        "multistructure_first_locus_thresholds": {
            "min_presence_score": first_locus_min_presence,
            "min_localization_score": first_locus_min_localization,
            "min_spatial_coherence_by_structure": {
                label: float(min_spatial_coherence[idx])
                for idx, label in enumerate(fine_structure_labels)
            },
            "min_evidence_score_by_structure": {
                label: float(first_locus_min_evidence[idx])
                for idx, label in enumerate(fine_structure_labels)
            },
            "max_projected_burden": first_locus_max_projected_burden,
        },
        "multistructure_first_locus_selected_metaspots": int(
            first_locus_selected.sum()
        ),
        "multistructure_first_locus_selected_cells": (
            first_locus_selected_cell_count
        ),
        "multistructure_first_locus_burden_rejected_metaspots": int(
            first_locus_diagnostics["burden_rejected"].sum()
        ),
        "multistructure_first_locus_selected_by_label": {
            label: int(
                (first_locus_selected & (first_locus_labels == label)).sum()
            )
            for label in fine_structure_labels
        },
        "diminishing_return_selection_enabled": (
            diminishing_return_enabled
        ),
        "diminishing_return_labels": list(diminishing_return_labels),
        "diminishing_return_log_penalty": (
            diminishing_return_log_penalty
        ),
        "diminishing_return_burden_penalty": (
            diminishing_return_burden_penalty
        ),
        "diminishing_return_min_dgi_support": (
            diminishing_return_min_dgi_support
        ),
        "diminishing_return_evidence_formula": (
            "local_log2_enrichment + dgi_consensus_weight * "
            "leave_one_out_dgi_support"
        ),
        "diminishing_return_required_formula": (
            "label_min_log2_enrichment + dgi_consensus_weight * "
            "diminishing_return_min_dgi_support + log_penalty * log1p(k) + "
            "burden_penalty * prior_compartment_fraction"
        ),
        "diminishing_return_first_locus_unconditionally_kept": True,
        "diminishing_return_strict_loci_protected": True,
        "diminishing_return_pruned_metaspots": int(
            diminishing_return_pruned.sum()
        ),
        "diminishing_return_pruned_cells": int(
            np.unique(
                metadata_cell_codes[
                    metadata_valid_cells & diminishing_return_pruned
                ]
            ).size
        ),
        "diminishing_return_pruned_metaspots_by_label": {
            label: int(
                (
                    diminishing_return_pruned
                    & (diminishing_return_labels_before == label)
                ).sum()
            )
            for label in diminishing_return_labels
        },
        "cell_coverage_rescue_enabled": cell_coverage_rescue_enabled,
        "cell_coverage_labels": list(cell_coverage_labels),
        "cell_coverage_min_dgi_support": (
            cell_coverage_min_dgi_support
        ),
        "cell_coverage_min_dgi_margin": cell_coverage_min_dgi_margin,
        "cell_coverage_max_per_label": cell_coverage_max_per_label,
        "cell_coverage_cell_evidence_thresholds": {
            "min_marker_transcripts": cell_coverage_min_transcripts,
            "min_marker_genes": cell_coverage_min_genes,
            "min_log2_enrichment": cell_coverage_min_enrichment,
            "max_binomial_p": cell_coverage_max_p,
        },
        "cell_coverage_rescued_metaspots": int(
            cell_coverage_rescued.sum()
        ),
        "labelwise_cell_coverage_rescued_metaspots": int(
            labelwise_cell_coverage_rescued.sum()
        ),
        "cell_coverage_rescued_cells": cell_coverage_rescued_cell_count,
        "cell_coverage_rescued_metaspots_by_label": {
            label: int(
                (
                    cell_coverage_rescued
                    & (cell_coverage_rescue_labels == label)
                ).sum()
            )
            for label in cell_coverage_labels
        },
        "hierarchical_cell_rescue_enabled": (
            hierarchical_cell_rescue_enabled
        ),
        "hierarchical_cell_rescued_metaspots": int(
            hierarchical_cell_rescued.sum()
        ),
        "hierarchical_cell_rescued_cells": (
            hierarchical_rescued_cell_count
        ),
        "hierarchical_cell_rescued_metaspots_by_label": {
            label: int(
                (
                    hierarchical_cell_rescued
                    & (hierarchical_rescue_labels == label)
                ).sum()
            )
            for label in cell_coverage_labels
        },
        "hierarchical_cell_status_counts": hierarchical_status_counts,
        "hierarchical_cell_thresholds": {
            "min_cell_transcripts": hierarchical_min_cell_transcripts,
            "min_metaspots": hierarchical_min_metaspots,
            "min_presence_score": hierarchical_min_presence_score,
            "min_localization_score": (
                hierarchical_min_localization_score
            ),
            "min_joint_score": hierarchical_min_joint_score,
            "min_local_marker_transcripts": (
                hierarchical_min_local_transcripts
            ),
            "min_local_marker_genes": hierarchical_min_local_genes,
            "min_cell_marker_transcripts": (
                hierarchical_min_cell_marker_transcripts
            ),
            "min_cell_marker_genes": (
                hierarchical_min_cell_marker_genes
            ),
            "min_dgi_support": hierarchical_min_dgi_support,
            "min_dgi_margin": hierarchical_min_dgi_margin,
        },
        "hierarchical_score_definition": {
            "interpretation": "bounded_evidence_score_not_probability",
            "presence_weights": {
                "cell_marker_transcripts": 0.30,
                "cell_marker_genes": 0.25,
                "cell_log2_enrichment": 0.25,
                "one_minus_cell_binomial_p": 0.20,
            },
            "localization_weights": {
                "local_marker_transcripts": 0.15,
                "local_marker_genes": 0.10,
                "local_log2_enrichment": 0.20,
                "leave_one_out_dgi_support": 0.45,
                "leave_one_out_dgi_margin": 0.10,
            },
            "joint": "geometric_mean",
        },
        "callable_cells": callable_cell_count,
        "callable_cells_with_fine_before_hierarchical": (
            callable_cells_with_fine_before
        ),
        "callable_cells_with_fine_after_hierarchical": (
            callable_cells_with_fine_after
        ),
        "callable_cell_fine_coverage_before_hierarchical": (
            callable_cells_with_fine_before / callable_cell_count
            if callable_cell_count
            else None
        ),
        "callable_cell_fine_coverage_after_hierarchical": (
            callable_cells_with_fine_after / callable_cell_count
            if callable_cell_count
            else None
        ),
        "callable_cells_without_fine_after_hierarchical": (
            callable_cell_count - callable_cells_with_fine_after
        ),
        "omm_cell_coherent_rescue_enabled": omm_cell_rescue_enabled,
        "omm_cell_coherent_rescued_metaspots": int(
            omm_cell_coherent_rescued.sum()
        ),
        "omm_cell_coherent_rescued_cells": omm_rescued_cell_count,
        "omm_cell_exploratory_candidate_metaspots": int(
            omm_cell_exploratory_candidate.sum()
        ),
        "omm_cell_exploratory_candidate_cells": (
            omm_exploratory_cell_count
        ),
        "omm_cell_coherent_thresholds": {
            "min_cell_marker_transcripts": omm_cell_min_transcripts,
            "min_cell_marker_genes": omm_cell_min_genes,
            "min_cell_log2_enrichment": omm_cell_min_enrichment,
            "max_cell_binomial_p": omm_cell_max_p,
            "exploratory_max_cell_binomial_p": (
                omm_cell_exploratory_max_p
            ),
            "min_local_marker_transcripts": omm_local_min_transcripts,
            "min_local_marker_genes": omm_local_min_genes,
            "min_local_log2_enrichment": omm_local_min_enrichment,
            "min_local_score_gap": omm_local_min_gap,
            "min_local_dgi_support": omm_local_min_dgi_support,
            "min_local_dgi_margin": omm_local_min_dgi_margin,
            "max_per_cell": omm_cell_max_per_cell,
        },
        "cell_coverage_cells_before_by_label": (
            cell_coverage_cells_before_by_label
        ),
        "cell_coverage_cells_after_by_label": (
            cell_coverage_cells_after_by_label
        ),
        "same_cell_only": True,
        "same_broad_compartment_only": True,
        "dgi_consensus_excludes_self": True,
        "marker_genes_per_label": {
            label: int(len(marker_sets[label])) for label in labels
        },
        "strict_fdr": strict_fdr,
        "exploratory_max_q": exploratory_q,
        "min_marker_transcripts": min_transcripts,
        "min_marker_genes": min_genes,
        "min_log2_enrichment": min_enrichment,
        "min_score_gap": min_gap,
        "background_prior_strength": prior_strength,
        "label_thresholds": thresholds_by_label,
        "require_omm_confirmation": require_omm,
        "omm_confirmation_mode": omm_confirmation_mode,
        "omm_unidentifiable_policy": omm_unidentifiable_policy,
        "omm_confirmed_metaspots": int((is_omm & omm_confirmed).sum()),
        "omm_strict_local_candidate_metaspots": int(
            omm_strict_local_candidate.sum()
        ),
        "omm_strict_local_fallback_metaspots": int(
            omm_strict_local_fallback.sum()
        ),
        "decision_provenance_counts": {
            str(key): int(value)
            for key, value in pd.Series(decision_provenance)
            .value_counts()
            .items()
        },
        "strict_supported_metaspots": int(strict_supported.sum()),
        "exploratory_supported_metaspots": int(
            exploratory_supported.sum()
        ),
        "final_label_counts": {
            str(key): int(value)
            for key, value in pd.Series(fine_labels).value_counts().items()
        },
        "fine_label_counts": {
            str(key): int(value)
            for key, value in pd.Series(fine_labels).value_counts().items()
        },
        "fine_state_counts": {
            str(key): int(value)
            for key, value in pd.Series(fine_state).value_counts().items()
        },
        "fallback_reason_counts": {
            str(key): int(value)
            for key, value in pd.Series(fallback_reason).value_counts().items()
        },
    }
    if cell_codes is None or unique_cells is None:
        raise RuntimeError(
            "Multi-structure evidence requires cell-level marker aggregation"
        )
    valid_cell_rows = np.flatnonzero(cell_codes >= 0)
    observed_codes, first_offsets = np.unique(
        cell_codes[valid_cell_rows], return_index=True
    )
    if not np.array_equal(observed_codes, np.arange(len(unique_cells))):
        raise RuntimeError("Cell structure evidence cell ordering is incomplete")
    first_rows = valid_cell_rows[first_offsets]
    adata1.uns["cell_structure_evidence"] = {
        "schema_version": 2,
        "interpretation": "assay_detectability_evidence_not_organelle_presence",
        "cell_ids": np.asarray(unique_cells, dtype=str),
        "structures": np.asarray(fine_structure_labels, dtype=str),
        "presence_score": multistructure_presence[first_rows],
        "spatial_coherence": multistructure_spatial_coherence[first_rows],
    }
    adata1.uns["metaspot_local_annotation"] = metadata
    return adata1, score_table, metadata


def _configure_annotation_unit(
    adata2: ad.AnnData,
    unit: str = "leiden",
) -> ad.AnnData:
    """Choose the grain at which RAP-X makes independent decisions.

    ``metadomain`` preserves the embedding Leiden labels for visualization but
    gives each adata2 row its own annotation group.  ``metaspot`` deliberately
    uses the same MetaDomain RAP-X groups here: those calls are only broad
    priors, and a later same-cell DGI-neighborhood step makes the final
    MetaSpot-level subtype decisions.
    """
    mode = str(unit).strip().lower().replace("_", "-")
    if mode in {"leiden", "cluster", "annotation-cluster"}:
        adata2.obs["annotation_group"] = adata2.obs["leiden"].astype(str)
        adata2.uns["annotation_unit"] = "leiden"
        return adata2
    metaspot_mode = mode in {"metaspot", "meta-spot", "spot"}
    if not metaspot_mode and mode not in {
        "metadomain",
        "meta-domain",
        "domain",
    }:
        raise ValueError(
            "annotation.unit must be 'metaspot', 'metadomain', or 'leiden', "
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
    adata2.uns["annotation_unit"] = (
        "metaspot" if metaspot_mode else "metadomain"
    )
    if metaspot_mode:
        adata2.uns["annotation_prior_unit"] = "metadomain"
    return adata2


def  _maybe_cluster_and_annotate(adata1: ad.AnnData, adata2: ad.AnnData, final_df: pd.DataFrame, show_internal_progress: bool = False, out_dir: Optional[str] = None,
                                annotation_method: str = "rapx",
                                rapx_two_stage: bool = False, rapx_spatial_channel: bool = False,
                                gold_data: Optional[Union[str, pd.DataFrame]] = None,
                                cluster_rep_override: Optional[str] = None,
                                rep_priority_override: Optional[Sequence[str]] = None,
                                annotation_unit_override: Optional[str] = None,
                                require_cluster_rep_override: Optional[bool] = None):
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
    require_cluster_rep = False
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
    omm_policy = "balanced"
    omm_metaspot_ambiguous_policy = "competitor"
    omm_metaspot_min_informative_transcripts = 5
    omm_metaspot_shrinkage_count = 20.0
    omm_metaspot_posterior_high = 0.80
    omm_metaspot_posterior_low = 0.20
    metaspot_compartment_correction = True
    try:
        _cfg = load_params_yaml()
        rep_priority = _cfg.get('module7', {}).get('rep_priority', None)
        if isinstance(rep_priority, str):
            rep_priority = [s.strip() for s in rep_priority.split(',') if s.strip()]
        ann = resolve_annotation_configuration(_cfg)
        # Clustering neighbors for annotation stage (from annotation config)
        n_neighbors = int(ann["neighbors_k"])
        # Use annotation's Leiden resolution (no legacy fallbacks)
        leiden_resolution = float(ann["leiden_resolution"])
        # annotation-level params
        hvg_n_top_genes = ann.get('hvg_n_top_genes', None)
        pca_n_comps = ann.get('pca_n_comps', None)
        umap_min_dist = ann.get('umap_min_dist', None)
        umap_spread = ann.get('umap_spread', None)
        umap_neighbors_key = ann.get('umap_neighbors_key', None)
        cluster_rep = ann.get('cluster_rep', None)
        require_cluster_rep = bool(ann.get("require_cluster_rep", False))
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
    if cluster_rep_override is not None:
        cluster_rep = str(cluster_rep_override)
    if rep_priority_override is not None:
        rep_priority = [str(value) for value in rep_priority_override]
    if annotation_unit_override is not None:
        annotation_unit = str(annotation_unit_override)
    if require_cluster_rep_override is not None:
        require_cluster_rep = bool(require_cluster_rep_override)
    # Reference thresholds affect scientific identity and resume validity.
    # Parse them outside the broad optional-clustering fallback above so a
    # malformed value cannot be silently replaced by a default.
    strict_annotation_cfg = resolve_annotation_configuration(
        load_params_yaml() or {}
    )
    omm_policy = str(strict_annotation_cfg["omm_policy"])
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
    metaspot_compartment_correction = bool(
        strict_annotation_cfg.get(
            "metaspot_compartment_correction",
            metaspot_compartment_correction,
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
    annotation_mode_normalized = (
        str(annotation_unit).strip().lower().replace("_", "-")
    )
    hybrid_metaspot_mode = annotation_mode_normalized in {
        "metaspot",
        "meta-spot",
        "spot",
    }
    _logging.getLogger("cellscope").info(
        "Annotation semantics | prior_unit=%s | production_unit=%s | "
        "two_stage=%s | "
        "spatial_channel=%s | metaspot_compartment_correction=%s | "
        "omm_policy=%s | omm_metaspot_refinement=%s | "
        "omm_apply_to_primary=%s",
        "metadomain" if hybrid_metaspot_mode else annotation_unit,
        "metaspot" if hybrid_metaspot_mode else annotation_unit,
        bool(rapx_two_stage),
        bool(rapx_spatial_channel),
        metaspot_compartment_correction,
        omm_policy,
        omm_metaspot_refinement,
        omm_metaspot_apply_to_primary,
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
            require_cluster_rep=require_cluster_rep,
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
                diagnostic=True,
            )

    # Optional manual markers. Disabled by default because marker provenance and
    # compartment specificity must be curated explicitly.
    try:
        use_manual_markers = bool(
            strict_annotation_cfg.get("use_manual_markers", False)
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
    # The caller resolves these execution switches once from the effective
    # configuration. Do not silently replace explicit programmatic overrides by
    # re-reading mutable process-global YAML here.
    rapx_use_two_stage = bool(rapx_two_stage)
    rapx_use_spatial_channel = bool(rapx_spatial_channel)
    rapx_plot_umap = bool(
        strict_annotation_cfg.get("save_umap_plot", False)
        or strict_annotation_cfg.get("save_plots", False)
    )
    rapx_plot_diag = bool(strict_annotation_cfg.get("save_plots", False))
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
    if rapx_plot_umap:
        try:
            _plot_metadomain_subcellular_annotation(
                adata2,
                out_dir or os.getcwd(),
                label_col="subcellular_annotation",
            )
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "MetaDomain Subcellular Annotation UMAP generation failed: %s",
                exc,
                exc_info=True,
            )
    try:
        annotation_meta = adata2.uns.setdefault(
            "subcellular_annotation_meta",
            {},
        )
        annotation_meta["annotation_unit"] = str(
            adata2.uns.get("annotation_unit", annotation_unit)
        )
        if hybrid_metaspot_mode:
            annotation_meta["annotation_prior_unit"] = "metadomain"
            annotation_meta["annotation_decision_unit"] = "metaspot"
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
        _save_dataframe(
            annot_df.reset_index(),
            out_dir or os.getcwd(),
            "rapx_annotation_scores",
            diagnostic=True,
        )
        # Keep object labels separate from the Leiden-level majority label.
        try:
            if 'subcellular_annotation' in adata2.obs and 'leiden' in adata2.obs:
                _refresh_subcellular_group_columns(
                    adata2,
                    label_col='subcellular_annotation',
                )
        except Exception:
            pass
        # Preserve the confirmatory RAP-X track as exact audit tables only.
        # A strict-label UMAP is intentionally not generated because
        # strict ``Unknown`` means "not FDR-confirmed", not the final label.
        _save_rapx_metadomain_evidence_audit_tables(
            adata2,
            out_dir or os.getcwd(),
        )
    except Exception:
        pass

    omm_refinement_meta: Optional[Dict[str, Any]] = None
    omm_refinement_columns: Tuple[str, ...] = ()
    if omm_metaspot_refinement:
        from .omm_refinement import (
            OMM_REFINEMENT_COLUMNS,
            build_omm_contrast_marker_weights,
            refine_omm_metaspots,
        )
        omm_refinement_columns = tuple(OMM_REFINEMENT_COLUMNS)

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
            broadcast_to_final=not hybrid_metaspot_mode,
        )
        omm_refinement_meta["marker_builder"] = omm_marker_meta
        omm_refinement_meta["applied_to_primary_annotation"] = bool(
            omm_metaspot_apply_to_primary
            and omm_refinement_meta.get("identifiable", False)
        )
        omm_refinement_meta["ambiguous_primary_label_policy"] = (
            omm_metaspot_ambiguous_policy
        )
        omm_refinement_meta["primary_decision_policy"] = omm_policy
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
        if out_dir is not None and _full_metaspot_diagnostics_enabled(adata1):
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
                diagnostic=True,
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
                diagnostic=True,
            )
        elif out_dir is not None:
            _logging.getLogger("cellscope").info(
                "Skipping full OMM MetaSpot diagnostic table for scalable run"
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

    broadcast_wide_transcript_audits = (
        _broadcast_metaspot_diagnostics_enabled(adata1)
    )

    # Step 4: map annotation back to final point table via meta-domain
    if 'meta-domain' in final_df.columns:
        transcript_domain_indexer = _key_row_indexer(
            pd.Index(adata2.obs_names.astype(str)),
            final_df["meta-domain"],
            error_context="MetaDomain annotation",
        )
        domain_labels = _take_obs_column_by_row_indexer(
            adata2.obs['subcellular_annotation'].astype(str),
            transcript_domain_indexer,
            final_df.index,
        )
        transcript_domain_updates: Dict[str, Any] = {
            "subcellular_annotation_domain": pd.Categorical(domain_labels)
        }
        if "r_signed" in final_df.columns:
            (
                guarded_labels,
                geometry_conflict,
            ) = _apply_local_compartment_guard(
                domain_labels,
                final_df["r_signed"],
            )
            transcript_domain_updates[
                "subcellular_geometry_conflict"
            ] = geometry_conflict
            transcript_domain_updates[
                "subcellular_annotation_geometry_guarded"
            ] = (
                pd.Categorical(guarded_labels)
            )
        else:
            transcript_domain_updates[
                "subcellular_annotation_geometry_guarded"
            ] = (
                pd.Categorical(domain_labels)
            )
            transcript_domain_updates["subcellular_geometry_conflict"] = False
            guarded_labels = domain_labels
        selected_transcript_labels = (
            guarded_labels
            if metaspot_compartment_correction
            else domain_labels
        )
        transcript_domain_updates["subcellular_annotation"] = pd.Categorical(
            selected_transcript_labels
        )
        domain_evidence_columns = {
            "subcellular_annotation_strict": (
                "subcellular_annotation_domain_strict"
            ),
            "subcellular_annotation_exploratory": (
                "subcellular_annotation_domain_exploratory"
            ),
            "subcellular_annotation_tier": (
                "subcellular_annotation_domain_tier"
            ),
        }
        for legacy_column, domain_column in domain_evidence_columns.items():
            if legacy_column in adata2.obs.columns:
                mapped_values = _take_obs_column_by_row_indexer(
                    adata2.obs[legacy_column].astype(str),
                    transcript_domain_indexer,
                    final_df.index,
                )
                transcript_domain_updates[domain_column] = pd.Categorical(
                    mapped_values
                )
                # Compatibility alias. New consumers should use the explicit
                # ``*_domain_*`` name.
                transcript_domain_updates[legacy_column] = pd.Categorical(
                    mapped_values
                )
        if "subcellular_candidate" in adata2.obs.columns:
            candidate_map = adata2.obs["subcellular_candidate"].astype(str)
            mapped_candidate = _take_obs_column_by_row_indexer(
                candidate_map,
                transcript_domain_indexer,
                final_df.index,
            )
            transcript_domain_updates[
                "subcellular_candidate_domain"
            ] = mapped_candidate
            transcript_domain_updates["subcellular_candidate"] = (
                mapped_candidate
            )
        expression_only_columns = {
            "subcellular_annotation_expression_only": (
                "subcellular_annotation_domain_expression_only"
            ),
            "subcellular_candidate_expression_only": (
                "subcellular_candidate_domain_expression_only"
            ),
        }
        for source, target in expression_only_columns.items():
            if source in adata2.obs.columns:
                mapped_values = _take_obs_column_by_row_indexer(
                    adata2.obs[source].astype(str),
                    transcript_domain_indexer,
                    final_df.index,
                )
                transcript_domain_updates[target] = pd.Categorical(
                    mapped_values
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
        if broadcast_wide_transcript_audits:
            for source, target in evidence_output_map.items():
                if source in adata2.obs.columns:
                    transcript_domain_updates[
                        target
                    ] = _take_obs_column_by_row_indexer(
                        adata2.obs[source],
                        transcript_domain_indexer,
                        final_df.index,
                    )
        else:
            _logging.getLogger("cellscope").info(
                "Keeping wide MetaDomain evidence in adata1/adata2 only; "
                "transcript broadcast disabled for scalable run"
            )
        for column in (
            "subcellular_annotation_conservative",
            "subcellular_confidence",
        ):
            if column in adata2.obs.columns:
                transcript_domain_updates[
                    column
                ] = _take_obs_column_by_row_indexer(
                    adata2.obs[column],
                    transcript_domain_indexer,
                    final_df.index,
                )
        if "subcell_low_conf" in adata2.obs.columns:
            confidence_map = _coerce_bool_series(
                adata2.obs["subcell_low_conf"],
                default=True,
            )
            transcript_domain_updates[
                "subcell_low_conf"
            ] = _take_obs_column_by_row_indexer(
                confidence_map,
                transcript_domain_indexer,
                final_df.index,
            )
        final_df = _replace_frame_columns(
            final_df,
            transcript_domain_updates,
        )
        del transcript_domain_indexer

    if "meta-domain" in adata1.obs.columns:
        metaspot_domain_indexer = _key_row_indexer(
            pd.Index(adata2.obs_names.astype(str)),
            adata1.obs["meta-domain"],
            error_context="MetaDomain evidence",
        )
        domain_labels = _take_obs_column_by_row_indexer(
            adata2.obs["subcellular_annotation"].astype(str),
            metaspot_domain_indexer,
            adata1.obs.index,
        )
        domain_obs_updates: Dict[str, Any] = {
            "subcellular_annotation_domain": pd.Categorical(domain_labels)
        }
        metaspot_guard_values = _metaspot_compartment_guard_values(
            adata1.obs
        )
        if metaspot_guard_values is not None:
            (
                guarded_labels,
                geometry_conflict,
            ) = _apply_local_compartment_guard(
                domain_labels,
                metaspot_guard_values,
            )
            domain_obs_updates[
                "subcellular_annotation_geometry_guarded"
            ] = pd.Categorical(
                guarded_labels
            )
            domain_obs_updates["subcellular_geometry_conflict"] = (
                geometry_conflict.values
            )
        else:
            domain_obs_updates[
                "subcellular_annotation_geometry_guarded"
            ] = pd.Categorical(
                domain_labels
            )
            domain_obs_updates["subcellular_geometry_conflict"] = False
            guarded_labels = domain_labels
        domain_obs_updates[
            "subcellular_annotation_metaspot_corrected"
        ] = pd.Categorical(guarded_labels)
        domain_obs_updates[
            "subcellular_annotation_metaspot_uncorrected"
        ] = pd.Categorical(domain_labels)
        selected_metaspot_labels = (
            guarded_labels
            if metaspot_compartment_correction
            else domain_labels
        )
        domain_obs_updates["subcellular_annotation"] = pd.Categorical(
            selected_metaspot_labels
        )
        domain_obs_updates[
            "subcellular_annotation_metaspot"
        ] = pd.Categorical(
            selected_metaspot_labels
        )
        domain_evidence_columns = {
            "subcellular_annotation_strict": (
                "subcellular_annotation_domain_strict"
            ),
            "subcellular_annotation_exploratory": (
                "subcellular_annotation_domain_exploratory"
            ),
            "subcellular_annotation_tier": (
                "subcellular_annotation_domain_tier"
            ),
        }
        for legacy_column, domain_column in domain_evidence_columns.items():
            if legacy_column in adata2.obs.columns:
                mapped_values = _take_obs_column_by_row_indexer(
                    adata2.obs[legacy_column].astype(str),
                    metaspot_domain_indexer,
                    adata1.obs.index,
                )
                domain_obs_updates[domain_column] = pd.Categorical(
                    mapped_values
                )
                domain_obs_updates[legacy_column] = pd.Categorical(
                    mapped_values
                )
        if "subcellular_candidate" in adata2.obs.columns:
            mapped_candidate = _take_obs_column_by_row_indexer(
                adata2.obs["subcellular_candidate"].astype(str),
                metaspot_domain_indexer,
                adata1.obs.index,
            )
            domain_obs_updates["subcellular_candidate_domain"] = (
                mapped_candidate.values
            )
            domain_obs_updates["subcellular_candidate"] = (
                mapped_candidate.values
            )
        expression_only_columns = {
            "subcellular_annotation_expression_only": (
                "subcellular_annotation_domain_expression_only"
            ),
            "subcellular_candidate_expression_only": (
                "subcellular_candidate_domain_expression_only"
            ),
        }
        for source, target in expression_only_columns.items():
            if source in adata2.obs.columns:
                mapped_values = _take_obs_column_by_row_indexer(
                    adata2.obs[source].astype(str),
                    metaspot_domain_indexer,
                    adata1.obs.index,
                )
                domain_obs_updates[target] = pd.Categorical(mapped_values)
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
                domain_obs_updates[target] = _take_obs_column_by_row_indexer(
                    adata2.obs[source],
                    metaspot_domain_indexer,
                    adata1.obs.index,
                )
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
                domain_obs_updates[source] = _take_obs_column_by_row_indexer(
                    adata2.obs[source],
                    metaspot_domain_indexer,
                    adata1.obs.index,
                )
        for column in (
            "subcellular_annotation_conservative",
            "subcellular_confidence",
        ):
            if column in adata2.obs.columns:
                domain_obs_updates[column] = _take_obs_column_by_row_indexer(
                    adata2.obs[column],
                    metaspot_domain_indexer,
                    adata1.obs.index,
                )
        if "subcell_low_conf" in adata2.obs.columns:
            confidence_map = _coerce_bool_series(
                adata2.obs["subcell_low_conf"],
                default=True,
            )
            domain_obs_updates[
                "subcell_low_conf"
            ] = _take_obs_column_by_row_indexer(
                confidence_map,
                metaspot_domain_indexer,
                adata1.obs.index,
            )
        _replace_obs_columns(adata1, domain_obs_updates)
        del metaspot_domain_indexer
        if {
            "cell_id",
            "subcellular_annotation",
            "subcellular_confidence",
        }.issubset(adata1.obs.columns):
            total_cells, coverage = _high_confidence_cell_coverage(
                adata1.obs["cell_id"],
                adata1.obs["subcellular_annotation"],
                adata1.obs["subcellular_confidence"],
            )
            _coverage_logger = _logging.getLogger("cellscope")
            _coverage_logger.info(
                "High-confidence original-cell coverage "
                "(labels are not mutually exclusive) | total_cells=%d",
                total_cells,
            )
            for label, count in coverage.items():
                pct = count / total_cells * 100 if total_cells else 0.0
                _coverage_logger.info(
                    "High-confidence cell coverage | label=%s | cells=%d | "
                    "fraction=%.4f",
                    label,
                    int(count),
                    pct / 100.0,
                )

    if (
        omm_refinement_meta is not None
        and "subcellular_annotation_domain" in adata1.obs.columns
        and "omm_refined_state" in adata1.obs.columns
        and "omm_domain_gate" in adata1.obs.columns
    ):
        metaspot_uncorrected_labels = _compose_omm_metaspot_labels(
            adata1.obs["subcellular_annotation_domain"],
            adata1.obs["omm_refined_state"],
            adata1.obs["omm_domain_gate"],
            adata1.obs.get("omm_raw_competitor_label"),
            adata1.obs.get("omm_reference_posterior"),
            identifiable=bool(
                omm_refinement_meta.get("identifiable", False)
            ),
            ambiguous_policy=omm_metaspot_ambiguous_policy,
            decision_policy=(
                "balanced" if omm_policy == "balanced" else "strict"
            ),
            posterior_high=omm_metaspot_posterior_high,
        )
        metaspot_guard_values = _metaspot_compartment_guard_values(
            adata1.obs
        )
        if metaspot_guard_values is not None:
            metaspot_corrected_labels, _ = _apply_local_compartment_guard(
                metaspot_uncorrected_labels,
                metaspot_guard_values,
            )
        else:
            metaspot_corrected_labels = metaspot_uncorrected_labels
        metaspot_labels = (
            metaspot_corrected_labels
            if metaspot_compartment_correction
            else metaspot_uncorrected_labels
        )
        adata1.obs[
            "subcellular_annotation_metaspot_uncorrected"
        ] = pd.Categorical(metaspot_uncorrected_labels)
        adata1.obs[
            "subcellular_annotation_metaspot_corrected"
        ] = pd.Categorical(metaspot_corrected_labels)
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

    metaspot_local_annotation_meta: Optional[Dict[str, Any]] = None
    if hybrid_metaspot_mode:
        build_full_score_table = _full_metaspot_diagnostics_enabled(adata1)
        (
            adata1,
            metaspot_local_scores,
            metaspot_local_annotation_meta,
        ) = _refine_metaspot_annotations_with_local_dgi(
            adata1,
            gold_aug,
            strict_annotation_cfg,
            build_score_table=build_full_score_table,
        )
        adata2.uns["metaspot_local_annotation"] = (
            metaspot_local_annotation_meta
        )
        if out_dir is not None and metaspot_local_scores is not None:
            _save_dataframe(
                metaspot_local_scores,
                out_dir,
                "metaspot_local_annotation_scores",
                diagnostic=True,
            )
            supported_omm = (
                metaspot_local_scores["final_annotation"].astype(str)
                == "OMM"
            )
            exploratory_omm = metaspot_local_scores[
                "omm_cell_exploratory_candidate"
            ].astype(bool)
            tier_masks = (
                supported_omm,
                exploratory_omm,
                supported_omm | exploratory_omm,
            )
            total_cells, tier_cell_counts = _unique_cell_counts_for_masks(
                metaspot_local_scores["cell_id"],
                tier_masks,
            )
            omm_tier_rows = []
            for tier_name, tier_mask, tier_cells in zip(
                (
                    "Supported",
                    "Exploratory only",
                    "Supported or exploratory",
                ),
                tier_masks,
                tier_cell_counts,
            ):
                omm_tier_rows.append(
                    {
                        "omm_evidence_tier": tier_name,
                        "n_metaspots": int(tier_mask.sum()),
                        "n_cells": tier_cells,
                        "n_cells_total": total_cells,
                        "cell_detection_fraction": (
                            tier_cells / total_cells if total_cells else np.nan
                        ),
                        "writes_public_omm_label": tier_name == "Supported",
                    }
                )
            _save_dataframe(
                pd.DataFrame(omm_tier_rows),
                out_dir,
                "omm_cell_detection_tiers",
                diagnostic=True,
            )
        elif out_dir is not None:
            supported_omm = (
                adata1.obs["subcellular_annotation"].astype(str) == "OMM"
            )
            exploratory_omm = adata1.obs[
                "subcellular_metaspot_omm_exploratory_candidate"
            ].astype(bool)
            tier_masks = (
                supported_omm,
                exploratory_omm,
                supported_omm | exploratory_omm,
            )
            total_cells, tier_cell_counts = _unique_cell_counts_for_masks(
                adata1.obs["cell_id"],
                tier_masks,
            )
            omm_tier_rows = []
            for tier_name, tier_mask, tier_cells in zip(
                (
                    "Supported",
                    "Exploratory only",
                    "Supported or exploratory",
                ),
                tier_masks,
                tier_cell_counts,
            ):
                omm_tier_rows.append(
                    {
                        "omm_evidence_tier": tier_name,
                        "n_metaspots": int(tier_mask.sum()),
                        "n_cells": tier_cells,
                        "n_cells_total": total_cells,
                        "cell_detection_fraction": (
                            tier_cells / total_cells if total_cells else np.nan
                        ),
                        "writes_public_omm_label": tier_name == "Supported",
                    }
                )
            _save_dataframe(
                pd.DataFrame(omm_tier_rows),
                out_dir,
                "omm_cell_detection_tiers",
                diagnostic=True,
            )
            _logging.getLogger("cellscope").info(
                "Skipping 138-column MetaSpot score table | metaspots=%d",
                int(adata1.n_obs),
            )
        _logging.getLogger("cellscope").info(
            "Hybrid MetaSpot annotation | strict_supported=%d | "
            "exploratory_supported=%d | dgi_exploratory=%d | "
            "omm_confirmation=%s | "
            "omm_strict_local_fallback=%d | final_counts=%s | "
            "fallbacks=%s",
            metaspot_local_annotation_meta[
                "strict_supported_metaspots"
            ],
            metaspot_local_annotation_meta[
                "exploratory_supported_metaspots"
            ],
            metaspot_local_annotation_meta[
                "dgi_exploratory_supported_metaspots"
            ],
            metaspot_local_annotation_meta["omm_confirmation_mode"],
            metaspot_local_annotation_meta[
                "omm_strict_local_fallback_metaspots"
            ],
            metaspot_local_annotation_meta["final_label_counts"],
            metaspot_local_annotation_meta["fallback_reason_counts"],
        )

    # Provenance is constant within each MetaSpot. Compute it once at the
    # production decision grain and broadcast compact codes with the labels,
    # instead of repeating string comparisons over every transcript.
    adata1.obs = _attach_annotation_provenance_columns(adata1.obs)

    # ``subcellular_annotation`` remains the established public seven-class
    # result. Broadcast it and its evidence-state audits from MetaSpots so
    # transcripts in one MetaSpot cannot disagree.
    if (
        "subcellular_annotation" in adata1.obs.columns
        and "metaspot" in final_df.columns
    ):
        final_df = final_df.drop(
            columns=[
                "subcellular_annotation_compatibility",
                "subcellular_annotation_metaspot_compatibility",
            ],
            errors="ignore",
        )
        metaspot_row_indexer = _metaspot_transcript_row_indexer(
            adata1,
            final_df,
        )
        final_primary_labels = _take_obs_column_by_row_indexer(
            adata1.obs["subcellular_annotation"],
            metaspot_row_indexer,
            final_df.index,
        )
        transcript_annotation_updates: Dict[str, Any] = {
            "subcellular_annotation": pd.Categorical(final_primary_labels)
        }
        for label_column in (
            "subcellular_annotation_metaspot",
            "subcellular_annotation_metaspot_uncorrected",
            "subcellular_annotation_metaspot_corrected",
            "subcellular_annotation_geometry_guarded",
            "subcellular_annotation_metaspot_parent",
            "subcellular_annotation_metaspot_candidate",
            "subcellular_annotation_metaspot_strict",
            "subcellular_annotation_metaspot_exploratory",
            "subcellular_annotation_strict",
            "subcellular_annotation_exploratory",
            "subcellular_annotation_tier",
            "subcellular_candidate",
            "subcellular_broad_compartment",
            "subcellular_fine_annotation",
            "subcellular_fine_state",
            "subcellular_metaspot_marker_only_candidate",
            "subcellular_metaspot_decision_provenance",
            "subcellular_metaspot_omm_confirmation_status",
            "subcellular_metaspot_cell_coverage_rescue_label",
            "subcellular_metaspot_hierarchical_cell_status",
            "subcellular_metaspot_hierarchical_best_label",
            "subcellular_metaspot_diminishing_return_original_label",
            "subcellular_annotation_provenance",
        ):
            if label_column in adata1.obs.columns:
                transcript_annotation_updates[label_column] = pd.Categorical(
                    _take_obs_column_by_row_indexer(
                        adata1.obs[label_column],
                        metaspot_row_indexer,
                        final_df.index,
                    )
                )
        for provenance_column in (
            "final_label_matches_domain_annotation",
            "final_label_supported_by_domain_evidence",
        ):
            if provenance_column in adata1.obs.columns:
                transcript_annotation_updates[provenance_column] = (
                    _take_obs_column_by_row_indexer(
                        adata1.obs[provenance_column],
                        metaspot_row_indexer,
                        final_df.index,
                    )
                )
        broadcast_metaspot_diagnostics = (
            hybrid_metaspot_mode and broadcast_wide_transcript_audits
        )
        if broadcast_metaspot_diagnostics:
            for column in (
                "subcellular_metaspot_pool_size",
                "subcellular_metaspot_candidate_marker_transcripts",
                "subcellular_metaspot_candidate_marker_genes",
                "subcellular_metaspot_log2_enrichment",
                "subcellular_metaspot_fused_score",
                "subcellular_metaspot_marker_only_score_gap",
                "subcellular_metaspot_score_gap",
                "subcellular_metaspot_dgi_support",
                "subcellular_metaspot_dgi_margin",
                "subcellular_metaspot_dgi_informative_neighbors",
                "subcellular_metaspot_dgi_branch_used",
                "subcellular_metaspot_dgi_exploratory_supported",
                "subcellular_metaspot_cell_coverage_rescued",
                "subcellular_metaspot_labelwise_cell_coverage_rescued",
                "subcellular_metaspot_hierarchical_cell_rescued",
                "subcellular_metaspot_hierarchical_cell_callable",
                "subcellular_metaspot_hierarchical_cell_eligible",
                "subcellular_metaspot_hierarchical_cell_total_transcripts",
                "subcellular_metaspot_hierarchical_cell_metaspots",
                "subcellular_metaspot_hierarchical_presence_score",
                "subcellular_metaspot_hierarchical_localization_score",
                "subcellular_metaspot_hierarchical_joint_score",
                "subcellular_metaspot_hierarchical_best_presence_score",
                "subcellular_metaspot_hierarchical_best_localization_score",
                "subcellular_metaspot_hierarchical_best_joint_score",
                "subcellular_metaspot_hierarchical_best_dgi_support",
                "subcellular_metaspot_hierarchical_best_dgi_margin",
                "subcellular_metaspot_diminishing_return_eligible",
                "subcellular_metaspot_diminishing_return_pruned",
                "subcellular_metaspot_diminishing_return_first_locus",
                "subcellular_metaspot_diminishing_return_rank",
                "subcellular_metaspot_diminishing_return_evidence_score",
                "subcellular_metaspot_diminishing_return_required_score",
                "subcellular_metaspot_diminishing_return_burden_before",
                "subcellular_metaspot_cell_coverage_local_marker_transcripts",
                "subcellular_metaspot_cell_coverage_local_marker_genes",
                "subcellular_metaspot_cell_coverage_local_log2_enrichment",
                "subcellular_metaspot_cell_coverage_dgi_support",
                "subcellular_metaspot_cell_coverage_dgi_margin",
                "subcellular_metaspot_cell_coverage_cell_marker_transcripts",
                "subcellular_metaspot_cell_coverage_cell_marker_genes",
                "subcellular_metaspot_cell_coverage_cell_log2_enrichment",
                "subcellular_metaspot_cell_coverage_cell_binomial_p",
                "subcellular_metaspot_omm_cell_coherent_rescued",
                "subcellular_metaspot_omm_exploratory_candidate",
                "subcellular_metaspot_omm_evidence_tier",
                "subcellular_metaspot_omm_cell_marker_transcripts",
                "subcellular_metaspot_omm_cell_marker_genes",
                "subcellular_metaspot_omm_cell_log2_enrichment",
                "subcellular_metaspot_omm_cell_binomial_p",
                "subcellular_metaspot_omm_cell_bh_q",
                "subcellular_metaspot_binomial_p",
                "subcellular_metaspot_bh_q",
                "subcellular_metaspot_supported",
                "subcellular_fine_supported",
                "subcellular_metaspot_candidate_min_marker_transcripts",
                "subcellular_metaspot_candidate_min_marker_genes",
                "subcellular_metaspot_candidate_min_log2_enrichment",
                "subcellular_metaspot_candidate_min_score_gap",
                "subcellular_metaspot_candidate_strict_fdr",
                "subcellular_metaspot_candidate_exploratory_max_q",
                "subcellular_metaspot_fallback_reason",
                *omm_refinement_columns,
            ):
                if column not in adata1.obs.columns:
                    continue
                mapped_values = _take_obs_column_by_row_indexer(
                    adata1.obs[column],
                    metaspot_row_indexer,
                    final_df.index,
                )
                if mapped_values.isna().any() and not pd.api.types.is_float_dtype(
                    adata1.obs[column]
                ):
                    raise ValueError(
                        f"Failed to map MetaSpot diagnostic {column!r} to "
                        "every transcript"
                    )
                transcript_annotation_updates[column] = mapped_values
        elif hybrid_metaspot_mode:
            _logging.getLogger("cellscope").info(
                "Keeping wide MetaSpot diagnostics in adata1 only; "
                "transcript broadcast disabled for scalable run"
            )

        final_df = _replace_frame_columns(
            final_df,
            transcript_annotation_updates,
        )

    annotation_provenance = {
        "schema_version": 3,
        "domain_evidence_unit": (
            "metadomain" if hybrid_metaspot_mode else str(annotation_unit)
        ),
        "production_unit": "metaspot",
        "fine_label_column": "subcellular_annotation",
        "fine_label_alias": "subcellular_fine_annotation",
        "hybrid_metaspot_annotation": bool(hybrid_metaspot_mode),
        "metaspot_compartment_correction": bool(
            metaspot_compartment_correction
        ),
        "omm_policy": omm_policy,
        "omm_metaspot_refinement": bool(omm_metaspot_refinement),
        "omm_metaspot_apply_to_primary": bool(
            omm_metaspot_apply_to_primary
        ),
        "transcript_diagnostic_broadcast": bool(
            broadcast_wide_transcript_audits
        ),
        "legacy_evidence_aliases": {
            "subcellular_annotation_strict": (
                "subcellular_annotation_domain_strict"
            ),
            "subcellular_annotation_exploratory": (
                "subcellular_annotation_domain_exploratory"
            ),
            "subcellular_annotation_tier": (
                "subcellular_annotation_domain_tier"
            ),
        },
    }
    if metaspot_local_annotation_meta is not None:
        annotation_provenance["metaspot_local_annotation"] = (
            metaspot_local_annotation_meta
        )
    adata1.uns["subcellular_annotation_provenance"] = (
        annotation_provenance
    )
    adata2.uns["subcellular_annotation_provenance"] = (
        annotation_provenance
    )
    provenance_counts = (
        adata1.obs["subcellular_annotation_provenance"]
        .value_counts()
        .to_dict()
    )
    _logging.getLogger("cellscope").info(
        "Production annotation provenance | correction_enabled=%s | "
        "metaspot_counts=%s",
        metaspot_compartment_correction,
        provenance_counts,
    )

    # Keep the compact MetaDomain UMAP for orientation.  Add both a
    # quantitative aggregation/label-transfer audit and the final
    # user-facing MetaSpot result view after all local refinements.
    if (
        ann_save_plots
        and out_dir is not None
        and _io_should_write_figures()
    ):
        _plot_metadomain_annotation_quality(adata1, out_dir)
        _plot_metaspot_annotation_result(adata1, out_dir)
        _save_transcript_annotation_proportion_diagnostics(
            final_df,
            out_dir,
        )

    return adata1, adata2, final_df
