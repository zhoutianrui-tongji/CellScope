import os
import pandas as pd
from typing import Optional
from shapely.geometry import Polygon
from shapely import wkb


_CELL_ID_SAMPLE_ROWS = 10000


def _cell_id_kind(path: str) -> str:
    """Return the cheapest safe representation for cell_id in a CSV."""
    try:
        sample = pd.read_csv(path, usecols=["cell_id"], nrows=_CELL_ID_SAMPLE_ROWS, dtype={"cell_id": "string"})
    except Exception:
        return "string"
    if sample.empty or "cell_id" not in sample.columns:
        return "string"

    s = sample["cell_id"].dropna().astype("string").str.strip()
    if s.empty:
        return "string"
    if s.str.fullmatch(r"[0-9]+").all():
        return "uint32"
    if s.str.fullmatch(r"[0-9]+\.0+").all():
        return "float_to_uint32"
    return "string"


def _compact_numeric_cell_id(series: pd.Series) -> pd.Series:
    """Convert numeric-looking cell_id values to compact unsigned integers."""
    if pd.api.types.is_unsigned_integer_dtype(series.dtype):
        return series
    if pd.api.types.is_integer_dtype(series.dtype):
        return pd.to_numeric(series, downcast="unsigned")

    numeric = pd.to_numeric(series, errors="raise")
    if pd.api.types.is_float_dtype(numeric.dtype):
        if numeric.isna().any():
            raise ValueError("cell_id contains missing values")
        rounded = numeric.round()
        if not (numeric == rounded).all():
            raise ValueError("cell_id contains non-integer numeric values")
        numeric = rounded.astype("uint64", copy=False)
    return pd.to_numeric(numeric, downcast="unsigned")


def _normalize_cell_id(series: pd.Series) -> pd.Series:
    """Normalize cell_id values to comparable string keys.

    This keeps alphanumeric ids intact while mapping numeric-looking ids like
    1, 1.0, '1', and '1.0' to the same canonical value '1'.
    """
    s = series.astype("string")
    s = s.str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    return s


def read_spatial_transcripts(path: str) -> pd.DataFrame:
    usecols = ["cell_id", "overlaps_nucleus", "x_location", "y_location", "feature_name"]
    cell_kind = _cell_id_kind(path)
    dtype_map = {
        "overlaps_nucleus": "int8",
        "x_location": "float32",
        "y_location": "float32",
        "feature_name": "category",
    }
    if cell_kind == "uint32":
        dtype_map["cell_id"] = "uint32"
    elif cell_kind == "float_to_uint32":
        dtype_map["cell_id"] = "float64"
    else:
        dtype_map["cell_id"] = "string"

    # Read only the columns required by the pipeline and keep dtypes compact.
    df = pd.read_csv(path, usecols=usecols, dtype=dtype_map)
    if cell_kind in {"uint32", "float_to_uint32"}:
        df["cell_id"] = _compact_numeric_cell_id(df["cell_id"])
    else:
        df["cell_id"] = _normalize_cell_id(df["cell_id"])

    required = {"cell_id", "x_location", "y_location", "feature_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in spatial file {path}: {sorted(missing)}")
    return df


def read_boundaries(path: str) -> pd.DataFrame:
    cell_kind = _cell_id_kind(path)
    dtype_map = {
        "vertex_x": "float32",
        "vertex_y": "float32",
    }
    if cell_kind == "uint32":
        dtype_map["cell_id"] = "uint32"
    elif cell_kind == "float_to_uint32":
        dtype_map["cell_id"] = "float64"
    else:
        dtype_map["cell_id"] = "string"

    df = pd.read_csv(path, dtype=dtype_map)
    if "cell_id" in df.columns:
        if cell_kind in {"uint32", "float_to_uint32"}:
            df["cell_id"] = _compact_numeric_cell_id(df["cell_id"])
        else:
            df["cell_id"] = _normalize_cell_id(df["cell_id"])
    required = {"cell_id", "vertex_x", "vertex_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in boundaries file {path}: {sorted(missing)}")
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
