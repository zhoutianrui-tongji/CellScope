import os
import pandas as pd
from typing import Optional
from shapely.geometry import Polygon
from shapely import wkb


def read_spatial_transcripts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cell_id", "x_location", "y_location", "feature_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in spatial file {path}: {sorted(missing)}")
    return df


def read_boundaries(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cell_id", "vertex_x", "vertex_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in boundaries file {path}: {sorted(missing)}")
    return df


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(df: pd.DataFrame, path: str) -> None:
    # Convert shapely geometries to WKB hex for parquet compatibility
    df2 = df.copy()
    for col in df2.columns:
        if col.endswith("_polygon"):
            df2[col] = df2[col].apply(lambda g: wkb.dumps(g).hex() if g is not None else None)
    df2.to_parquet(path, index=False)


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
