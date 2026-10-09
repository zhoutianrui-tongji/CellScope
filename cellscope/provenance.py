"""Run provenance and atomic metadata writers for CellScoper."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Union

import pandas as pd
import yaml

from .resource_paths import bundled_resource_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_mapping(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_provenance(
    path: Optional[str],
    *,
    hash_content: bool = True,
) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    absolute = os.path.realpath(os.path.abspath(path))
    stat_result = os.stat(absolute)
    record: Dict[str, Any] = {
        "path": absolute,
        "size_bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }
    if hash_content:
        record["sha256"] = sha256_file(absolute)
    return record


def dataframe_provenance(frame: pd.DataFrame) -> Dict[str, Any]:
    normalized = frame.copy()
    normalized.columns = normalized.columns.astype(str)
    row_hashes = pd.util.hash_pandas_object(
        normalized,
        index=True,
        categorize=True,
    ).to_numpy(dtype="uint64", copy=False)
    digest = hashlib.sha256()
    digest.update("\0".join(normalized.columns).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return {
        "kind": "in_memory_dataframe",
        "n_rows": int(len(normalized)),
        "n_columns": int(normalized.shape[1]),
        "columns": normalized.columns.tolist(),
        "sha256": digest.hexdigest(),
    }


def source_tree_provenance(package_dir: str) -> Dict[str, Any]:
    package_dir = os.path.realpath(os.path.abspath(package_dir))
    files = []
    for root, directories, names in os.walk(package_dir):
        directories[:] = sorted(
            name
            for name in directories
            if name not in {"__pycache__", ".pytest_cache"}
        )
        for name in sorted(names):
            if name.endswith(".py"):
                files.append(os.path.join(root, name))
    digest = hashlib.sha256()
    relative_files = []
    for path in files:
        relative = os.path.relpath(path, package_dir)
        relative_files.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "package_dir": package_dir,
        "n_python_files": len(relative_files),
        "python_source_sha256": digest.hexdigest(),
    }


def git_provenance(start_dir: str) -> Dict[str, Any]:
    def _git(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=start_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    root = _git("rev-parse", "--show-toplevel")
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "repository_root": root,
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def build_input_provenance(
    *,
    spatial_path: str,
    cell_boundaries_path: str,
    nucleus_boundaries_path: str,
    gold_data: Optional[Union[str, pd.DataFrame]] = None,
    hash_inputs: bool = True,
) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {
        "spatial": file_provenance(
            spatial_path,
            hash_content=hash_inputs,
        ),
        "cell_boundaries": file_provenance(
            cell_boundaries_path,
            hash_content=hash_inputs,
        ),
        "nucleus_boundaries": file_provenance(
            nucleus_boundaries_path,
            hash_content=hash_inputs,
        ),
    }
    if isinstance(gold_data, str):
        inputs["gold_data"] = file_provenance(
            gold_data,
            hash_content=hash_inputs,
        )
    elif isinstance(gold_data, pd.DataFrame):
        inputs["gold_data"] = dataframe_provenance(gold_data)
    elif gold_data is not None:
        inputs["gold_data"] = dataframe_provenance(pd.DataFrame(gold_data))
    else:
        inputs["gold_data"] = None
    return inputs


def bundled_reference_provenance(
    package_dir: str,
    *,
    hash_content: bool = True,
) -> Dict[str, Any]:
    del package_dir  # kept for API compatibility
    records: Dict[str, Any] = {}
    for name in (
        "2019_CELL_APEXSeq.tsv",
        "RNALocateV3.txt",
        "manual_marker.csv",
    ):
        path = bundled_resource_path(name)
        records[name] = (
            file_provenance(path, hash_content=hash_content)
            if os.path.isfile(path)
            else None
        )
    return records


def _atomic_write(path: str, writer) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    def _writer(temporary: str) -> None:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            handle.write("\n")

    _atomic_write(path, _writer)


def atomic_write_yaml(path: str, payload: Mapping[str, Any]) -> None:
    def _writer(temporary: str) -> None:
        with open(temporary, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                dict(payload),
                handle,
                allow_unicode=True,
                sort_keys=False,
            )

    _atomic_write(path, _writer)
