import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest


if os.environ.get("CELLSCOPER_RUN_EXTERNAL_BENCHMARK_TESTS") != "1":
    pytest.skip(
        "requires the outer-workspace SeqFISH benchmark runner; set "
        "CELLSCOPER_RUN_EXTERNAL_BENCHMARK_TESTS=1 to enable",
        allow_module_level=True,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "run_seqfish_representation_benchmark.py"
sys.path.insert(0, str(PROJECT_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "seqfish_representation_benchmark",
    SCRIPT_PATH,
)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_runner_sets_numba_before_sparse_scanpy_work(tmp_path):
    config = tmp_path / "params.yaml"
    config.write_text(
        "runtime:\n"
        "  n_jobs: 12\n"
        "  joblib_backend: threading\n"
        "  blas_threads: 1\n",
        encoding="utf-8",
    )
    code = (
        "import importlib.util, os, sys; "
        f"sys.argv=['runner', '--config', r'{config}']; "
        f"spec=importlib.util.spec_from_file_location('bench_boot', r'{SCRIPT_PATH}'); "
        "module=importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name]=module; spec.loader.exec_module(module); "
        "import numba, scanpy as sc, scipy.sparse as sp; "
        "import anndata as ad, numpy as np; "
        "adata=ad.AnnData(sp.csr_matrix(np.array([[1,0,2],[0,1,3],[2,1,0]], dtype=np.float32))); "
        "sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2); "
        "print(os.environ['NUMBA_NUM_THREADS'], numba.get_num_threads(), sc.settings.n_jobs)"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "NUMBA_NUM_THREADS": "192",
            "SCANPY_N_JOBS": "192",
            "PYTHONPATH": str(PROJECT_ROOT),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    assert result.stdout.strip().splitlines()[-1] == "1 1 1"


def test_cellscope_audit_lock_file_is_not_treated_as_active(tmp_path):
    lock_path = tmp_path / ".cellscoper.run.lock"
    lock_path.write_text('{"status": "completed"}', encoding="utf-8")

    assert benchmark.cellscope_lock_is_active(lock_path) is False


def test_resume_rejects_cached_rows_that_did_not_honor_fixed_k(tmp_path):
    path = tmp_path / "clustering_sweep.csv"
    pd.DataFrame(
        {
            "requested_k": [16, 24],
            "actual_k": [16, 61],
        }
    ).to_csv(path, index=False)

    with pytest.raises(RuntimeError, match="actual_k != requested_k"):
        benchmark.validate_resumable_fixed_k_tables(tmp_path)

    pd.DataFrame(
        {
            "requested_k": [16, 24],
            "actual_k": [16, 24],
        }
    ).to_csv(path, index=False)
    benchmark.validate_resumable_fixed_k_tables(tmp_path)


def test_cellscope_source_requires_native_metaspot_provenance(tmp_path):
    final_data = pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c2", "c2"],
            "metaspot": ["m1", "m1", "m2", "m2"],
            "overlaps_nucleus": [1, 1, 0, 0],
        }
    )
    population = benchmark.validate_benchmark_population(
        final_data,
        pd.Index(["m1", "m2"]),
    )
    source = ad.AnnData(
        X=np.ones((2, 3), dtype=np.float32),
        obs=pd.DataFrame(index=["m1", "m2"]),
        var=pd.DataFrame(index=["A", "B", "C"]),
    )
    source.obsm["X_dgi_gcn_metaspot"] = np.ones(
        (2, 4),
        dtype=np.float32,
    )
    source.uns["benchmark_output"] = {
        "protocol": "benchmark_native_metaspot_dgi",
        "observation_unit": "metaspot",
        "metaspot_membership_sha256": population[
            "metaspot_membership_sha256"
        ],
        "dgi_provenance": {"node_unit": "metaspot"},
        "graph_provenance": {
            "node_unit": "metaspot",
            "uses_boundary_geometry_features": False,
        },
    }
    source_path = (
        tmp_path
        / "cellscoper_benchmark_out"
        / "adata_metaspot.h5ad"
    )
    source_path.parent.mkdir()
    source.write_h5ad(source_path)
    spec = benchmark.METHOD_SPECS[0]

    _, metadata, obs_names = benchmark.validate_method_source(
        tmp_path,
        spec,
        final_data,
        None,
    )
    assert obs_names.tolist() == ["m1", "m2"]
    assert (
        metadata["benchmark_output"]["dgi_provenance"]["node_unit"]
        == "metaspot"
    )

    source.uns["benchmark_output"]["dgi_provenance"]["node_unit"] = (
        "metadomain"
    )
    source.write_h5ad(source_path)
    with pytest.raises(ValueError, match="DGI node_unit is not metaspot"):
        benchmark.validate_method_source(
            tmp_path,
            spec,
            final_data,
            None,
        )


def test_annotation_runner_keeps_shared_refinement_tracks_separate(
    tmp_path,
    monkeypatch,
):
    obs_names = pd.Index(["m1", "m2", "m3", "m4"])
    source = ad.AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=["A", "B", "C"]),
    )
    source.obsm["X_toy"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    source_path = tmp_path / "toy.h5ad"
    source.write_h5ad(source_path)

    final_data = pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c1", "c1", "c2", "c2", "c2", "c2"],
            "metaspot": ["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4"],
            "overlaps_nucleus": [1, 1, 0, 0, 1, 1, 0, 0],
        }
    )
    raw = pd.DataFrame(
        {
            "production": ["Nucleus", "Cytosol", "Nucleus", "Cytosol"],
            "candidate": ["Nucleus", "Cytosol", "Nucleus", "Cytosol"],
            "exploratory": ["Nucleus", "Cytosol", "Nucleus", "Cytosol"],
            "strict": ["Nucleus", "Cytosol", "Nucleus", "Cytosol"],
            "tier": ["Strict"] * 4,
        },
        index=obs_names,
    )
    guarded = raw.copy()
    guarded["production"] = ["Nucleus", "Cytosol", "Cytosol", "Cytosol"]

    def fake_shared(*args, **kwargs):
        return {
            "raw_metadomain_tracks": raw,
            "geometry_guarded_tracks": guarded,
            "shared_metaspot_refinement_tracks": raw,
            "shared_metaspot_refinement_geometry_guarded_tracks": guarded,
            "metadata": {"n_metadomains": 2},
        }

    monkeypatch.setattr(
        benchmark,
        "run_shared_metadomain_rapx_track",
        fake_shared,
    )
    method = benchmark.MethodSpec(
        name="Toy",
        relative_h5ad="toy.h5ad",
        embedding_key="X_toy",
        metadomain_embedding_key="X_toy_domain",
        role="test_backbone",
    )
    output_dir = tmp_path / "results"
    benchmark.run_annotation_benchmark(
        sources={"Toy": source_path},
        specs=[method],
        final_data=final_data,
        output_dir=output_dir,
        annotation_k=2,
        seeds=[7],
        resume=False,
        shared_metaspot_refinement=True,
    )

    global_metrics = pd.read_csv(
        output_dir / "annotation_global_metrics.csv"
    )
    assert len(global_metrics) == 12
    assert set(global_metrics["track_source"]) == {
        "raw_metadomain",
        "shared_metaspot_refinement",
        "shared_metaspot_refinement_geometry_guarded",
    }
    assert set(global_metrics["track"]) == {
        "production",
        "candidate",
        "exploratory",
        "strict",
    }

    sensitivity = pd.read_csv(output_dir / "boundary_sensitivity.csv")
    assert sensitivity.loc[0, "n_binary_labels_changed"] == 1
    assert sensitivity.loc[0, "binary_change_fraction"] == 0.25

    predictions = pd.read_csv(
        output_dir / "annotation_metaspot_predictions.csv.gz"
    )
    assert len(predictions) == 12
    assert set(predictions["track_source"]) == {
        "raw_metadomain",
        "shared_metaspot_refinement",
        "shared_metaspot_refinement_geometry_guarded",
    }
