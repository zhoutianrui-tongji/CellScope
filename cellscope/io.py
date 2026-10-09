import os
import pathlib
import pandas as pd
from typing import Iterable, Optional
from shapely.geometry import Polygon
from shapely import wkb


def _is_parquet(path: str) -> bool:
    return pathlib.Path(path).suffix.lower() in {".parquet", ".pq"}


def _table_columns(path: str) -> list[str]:
    if _is_parquet(path):
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    return pd.read_csv(path, nrows=0).columns.astype(str).tolist()


def read_table_preview(
    path: str,
    nrows: int = 5,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Read a bounded CSV/Parquet preview without loading a full Parquet column."""
    selected = list(columns) if columns is not None else None
    if not _is_parquet(path):
        return pd.read_csv(path, nrows=nrows, usecols=selected)

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    batches = parquet.iter_batches(
        batch_size=max(1, int(nrows)),
        columns=selected,
    )
    try:
        batch = next(batches)
    except StopIteration:
        return pd.DataFrame(columns=selected or parquet.schema_arrow.names)
    return batch.to_pandas().iloc[:nrows].copy()


def _normalize_cell_id(series: pd.Series) -> pd.Series:
    """Preserve cell identifiers as validated, trimmed string keys."""
    s = series.astype("string").str.strip()
    invalid = s.isna() | s.eq("")
    if invalid.any():
        raise ValueError(
            f"cell_id contains {int(invalid.sum())} missing or empty values"
        )
    return s


def read_spatial_transcripts(path: str) -> pd.DataFrame:
    available = set(_table_columns(path))
    required = {"cell_id", "x_location", "y_location", "feature_name"}
    missing = required - available
    if missing:
        raise ValueError(f"Missing columns in spatial file {path}: {sorted(missing)}")
    usecols = [
        column
        for column in (
            "cell_id",
            "overlaps_nucleus",
            "x_location",
            "y_location",
            "feature_name",
        )
        if column in available
    ]
    dtype_map = {
        "cell_id": "string",
        "x_location": "float32",
        "y_location": "float32",
        "feature_name": "category",
    }

    # Read only the columns required by the pipeline and keep dtypes compact.
    if _is_parquet(path):
        df = pd.read_parquet(path, columns=usecols)
    else:
        df = pd.read_csv(
            path,
            usecols=usecols,
            dtype={k: v for k, v in dtype_map.items() if k in usecols},
        )
    df["cell_id"] = _normalize_cell_id(df["cell_id"])
    for column in ("x_location", "y_location"):
        df[column] = pd.to_numeric(df[column], errors="raise").astype("float32")
    df["feature_name"] = df["feature_name"].astype("category")
    if "overlaps_nucleus" in df.columns:
        normalized = (
            df["overlaps_nucleus"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({"true": "1", "false": "0"})
        )
        numeric = pd.to_numeric(normalized, errors="raise")
        if not numeric.dropna().isin([0, 1]).all():
            raise ValueError("overlaps_nucleus must contain only boolean or 0/1 values")
        df["overlaps_nucleus"] = numeric.astype("Int8")
    return df


def read_boundaries(path: str) -> pd.DataFrame:
    required = {"cell_id", "vertex_x", "vertex_y"}
    available = set(_table_columns(path))
    missing = required - available
    if missing:
        raise ValueError(f"Missing columns in boundaries file {path}: {sorted(missing)}")
    dtype_map = {
        "cell_id": "string",
        "vertex_x": "float32",
        "vertex_y": "float32",
    }

    usecols = ["cell_id", "vertex_x", "vertex_y"]
    if _is_parquet(path):
        df = pd.read_parquet(path, columns=usecols)
    else:
        df = pd.read_csv(path, usecols=usecols, dtype=dtype_map)
    if "cell_id" in df.columns:
        df["cell_id"] = _normalize_cell_id(df["cell_id"])
    for column in ("vertex_x", "vertex_y"):
        df[column] = pd.to_numeric(df[column], errors="raise").astype("float32")
    return df


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(df: pd.DataFrame, path: str) -> None:
    # Convert shapely geometries to WKB hex for parquet compatibility.
    # Avoid df.copy() to prevent duplicating large tables in memory.
    for col in df.columns:
        if col.endswith("_polygon"):
            df[col] = df[col].apply(lambda g: wkb.dumps(g).hex() if g is not None else None)
    df.to_parquet(path, index=False)


def load_checkpoint(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Convert WKB hex back to shapely geometries where applicable
    for col in df.columns:
        if col.endswith("_polygon"):
            def to_poly(val: Optional[str]) -> Optional[Polygon]:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return None
                try:
                    return wkb.loads(bytes.fromhex(val))
                except Exception:
                    return None
            df[col] = df[col].apply(to_poly)
    return df
