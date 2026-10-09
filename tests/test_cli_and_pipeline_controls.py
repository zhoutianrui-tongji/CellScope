import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from scipy import sparse

from cellscope import cli
from cellscope import pipeline
from cellscope.config import (
    ANNOTATION_SCHEMA_VERSION,
    fingerprint_module,
    fingerprint_stage,
    get_default_params_path,
    load_params_yaml,
    resolve_annotation_configuration,
    stage_config_subset,
)
from cellscope.io import read_boundaries, read_spatial_transcripts
from cellscope.plotting_style import (
    SUBCELLULAR_LABEL_GROUPS,
    SUBCELLULAR_LABEL_ORDER,
    SUBCELLULAR_LABEL_PALETTE,
)


def _write_inputs(tmp_path):
    paths = {}
    for name in ("spatial", "cell", "nucleus"):
        path = tmp_path / f"{name}.csv"
        path.write_text("x,y\n0,0\n", encoding="utf-8")
        paths[name] = path
    return paths


def test_load_inputs_refreshes_stale_raw_checkpoints_when_reuse_is_disabled(
    tmp_path,
):
    spatial_path = tmp_path / "spatial.csv"
    cell_path = tmp_path / "cell.csv"
    nucleus_path = tmp_path / "nucleus.csv"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    pd.DataFrame(
        {
            "cell_id": ["fresh"],
            "overlaps_nucleus": [1],
            "x_location": [1.0],
            "y_location": [2.0],
            "feature_name": ["G1"],
        }
    ).to_csv(spatial_path, index=False)
    boundary = pd.DataFrame(
        {
            "cell_id": ["fresh"] * 3,
            "vertex_x": [0.0, 1.0, 0.0],
            "vertex_y": [0.0, 0.0, 1.0],
        }
    )
    boundary.to_csv(cell_path, index=False)
    boundary.to_csv(nucleus_path, index=False)

    stale_spatial = pd.DataFrame(
        {
            "cell_id": ["stale"],
            "overlaps_nucleus": [0],
            "x_location": [9.0],
            "y_location": [9.0],
            "feature_name": ["OLD"],
        }
    )
    stale_boundary = boundary.assign(cell_id="stale")
    pipeline.save_checkpoint(
        stale_spatial,
        pipeline._checkpoint_path(str(out_dir), "spatial"),
    )
    pipeline.save_checkpoint(
        stale_boundary,
        pipeline._checkpoint_path(str(out_dir), "cell_boundaries"),
    )
    pipeline.save_checkpoint(
        stale_boundary,
        pipeline._checkpoint_path(str(out_dir), "nucleus_boundaries"),
    )

    spatial, cell, nucleus = pipeline.load_inputs(
        str(spatial_path),
        str(cell_path),
        str(nucleus_path),
        str(out_dir),
        resume=True,
        save_intermediate=False,
        reuse_cached_inputs=False,
    )

    assert spatial["cell_id"].astype(str).tolist() == ["fresh"]
    assert set(cell["cell_id"].astype(str)) == {"fresh"}
    assert set(nucleus["cell_id"].astype(str)) == {"fresh"}
    refreshed = pipeline.load_checkpoint(
        pipeline._checkpoint_path(str(out_dir), "spatial")
    )
    assert refreshed["feature_name"].astype(str).tolist() == ["G1"]


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_input_readers_accept_csv_and_parquet_with_optional_nucleus_flags(
    tmp_path,
    suffix,
):
    spatial_path = tmp_path / f"spatial{suffix}"
    boundary_path = tmp_path / f"boundary{suffix}"
    spatial = pd.DataFrame(
        {
            "cell_id": ["c1", "c1"],
            "x_location": [1.0, 2.0],
            "y_location": [3.0, 4.0],
            "feature_name": ["G1", "G2"],
        }
    )
    boundary = pd.DataFrame(
        {
            "cell_id": ["c1"] * 3,
            "vertex_x": [0.0, 1.0, 0.0],
            "vertex_y": [0.0, 0.0, 1.0],
        }
    )
    if suffix == ".csv":
        spatial.to_csv(spatial_path, index=False)
        boundary.to_csv(boundary_path, index=False)
    else:
        spatial.to_parquet(spatial_path, index=False)
        boundary.to_parquet(boundary_path, index=False)

    loaded_spatial = read_spatial_transcripts(str(spatial_path))
    loaded_boundary = read_boundaries(str(boundary_path))

    assert "overlaps_nucleus" not in loaded_spatial.columns
    assert loaded_spatial["x_location"].dtype == np.dtype("float32")
    assert loaded_boundary["vertex_x"].dtype == np.dtype("float32")
    assert loaded_spatial["cell_id"].astype(str).tolist() == ["c1", "c1"]


def test_csv_input_readers_preserve_cell_ids_without_overflow_or_suffix_loss(
    tmp_path,
):
    spatial_path = tmp_path / "spatial.csv"
    boundary_path = tmp_path / "boundary.csv"
    cell_ids = ["4294967296", "A.0", "sample.000", "001"]
    pd.DataFrame(
        {
            "cell_id": cell_ids,
            "x_location": np.arange(4, dtype=float),
            "y_location": np.arange(4, dtype=float),
            "feature_name": ["G1"] * 4,
        }
    ).to_csv(spatial_path, index=False)
    pd.DataFrame(
        {
            "cell_id": cell_ids,
            "vertex_x": np.arange(4, dtype=float),
            "vertex_y": np.arange(4, dtype=float),
        }
    ).to_csv(boundary_path, index=False)

    spatial = read_spatial_transcripts(str(spatial_path))
    boundaries = read_boundaries(str(boundary_path))

    assert spatial["cell_id"].tolist() == cell_ids
    assert boundaries["cell_id"].tolist() == cell_ids


def test_preprocess_ignores_nucleus_flags_when_polygon_geometry_is_selected():
    spatial = pd.DataFrame(
        {
            "cell_id": ["mixed", "mixed", "single"],
            "overlaps_nucleus": [0, 1, 0],
        }
    )
    boundary = pd.DataFrame(
        {
            "cell_id": ["mixed", "single"],
            "vertex_x": [0.0, 1.0],
            "vertex_y": [0.0, 1.0],
        }
    )

    polygon_mode, _, _ = pipeline.preprocess(
        spatial,
        boundary,
        boundary,
        use_input_nucleus_flags=False,
    )
    flag_mode, _, _ = pipeline.preprocess(
        spatial,
        boundary,
        boundary,
        use_input_nucleus_flags=True,
    )

    assert set(polygon_mode["cell_id"]) == {"mixed", "single"}
    assert set(flag_mode["cell_id"]) == {"mixed"}


def test_cli_executes_merged_effective_config_and_preserves_dry_run_provenance(
    tmp_path,
    monkeypatch,
):
    paths = _write_inputs(tmp_path)
    custom = tmp_path / "partial.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "io": {"provenance_hash_inputs": False},
                "annotation": {"save_plots": False},
                "module8": {"dgi_unit": "metaspot"},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured["kwargs"] = kwargs
        captured["loaded_config"] = load_params_yaml()
        captured["config_path"] = os.environ["CELLSCOPE_CONFIG"]
        return {
            "dry_run": True,
            "changed_stages": ["load_inputs"],
            "start_from": "load_inputs",
        }

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
    monkeypatch.delenv("CELLSCOPE_CONFIG", raising=False)

    result = CliRunner().invoke(
        cli.main,
        [
            "--spatial",
            str(paths["spatial"]),
            "--cell-boundaries",
            str(paths["cell"]),
            "--nucleus-boundaries",
            str(paths["nucleus"]),
            "--out-dir",
            str(out_dir),
            "--config",
            str(custom),
            "--no-show-sample",
            "--no-save-intermediate",
            "--resume-policy",
            "force",
            "--resume-from",
            "annotate",
            "--dry-run-diff",
        ],
    )

    assert result.exit_code == 0, result.output
    effective = yaml.safe_load(
        (out_dir / "effective_params.yaml").read_text(encoding="utf-8")
    )
    packaged = load_params_yaml(path=get_default_params_path())
    assert effective["module3"]["target_sites_per_metaspot"] == (
        packaged["module3"]["target_sites_per_metaspot"]
    )
    assert effective["annotation"]["save_plots"] is False
    assert effective["module8"]["dgi_unit"] == "metaspot"
    assert effective["module8"]["dgi_framework"] == "cell_aware_dmgi"
    assert effective["module5"]["clustering_method"] == "consensus_leiden"
    assert effective["module5"]["hierarchical_block_nodes"] == 100000
    assert effective["module8"]["sampled_batch_size"] == 4096
    assert effective["module8"]["dgi_encoder"] == "residual"
    assert effective["module8"]["dgi_layers"] == 2
    assert effective["module8"]["dgi_hidden"] == 64
    assert effective["module8"]["dgi_out"] == 32
    assert effective["module8"]["dgi_dropout"] == 0.1
    assert effective["module8"]["dgi_residual_alpha"] == 0.1
    assert effective["module8"]["dgi_layer_norm"] is True
    assert effective["module7"]["graph_unit"] == "metaspot"
    assert effective["module5"]["cluster_rep"] == "X_dgi_gcn_metaspot"
    assert effective["module6"]["pool_reps"] == ["X_dgi_gcn_metaspot"]
    assert effective["io"]["write_intermediate"] is False
    assert effective["io"]["resume_policy"] == "force"
    assert effective["io"]["resume_from"] == "annotate"
    assert effective["show_sample"] is False
    assert captured["loaded_config"] == effective
    assert captured["kwargs"]["save_intermediate"] is False
    assert captured["kwargs"]["resume_policy"] == "force"
    assert captured["kwargs"]["resume_from"] == "annotate"
    assert captured["config_path"] == str(
        out_dir / "effective_params.yaml"
    )
    assert not (out_dir / "params_used.yaml").exists()
    manifest = json.loads(
        (out_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "dry_run_completed"
    assert manifest["invocation"]["resume_from"] == "annotate"
    assert manifest["configuration"]["effective_sha256"]


def test_cli_marks_stale_running_manifest_as_interrupted(tmp_path, monkeypatch):
    paths = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "run_manifest.json").write_text(
        json.dumps({"status": "running", "started_at": "2026-08-18T00:00:00Z"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pipeline,
        "run_pipeline",
        lambda **_: {"dry_run": True, "changed_stages": [], "start_from": "annotate"},
    )

    result = CliRunner().invoke(
        cli.main,
        [
            "--spatial",
            str(paths["spatial"]),
            "--cell-boundaries",
            str(paths["cell"]),
            "--nucleus-boundaries",
            str(paths["nucleus"]),
            "--out-dir",
            str(out_dir),
            "--no-show-sample",
            "--dry-run-diff",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run_completed"
    assert manifest["previous_invocation"]["status"] == "interrupted"
    assert manifest["previous_invocation"]["started_at"] == "2026-08-18T00:00:00Z"
    assert manifest["previous_invocation"]["interrupted_at"]


def test_pipeline_import_uses_single_blas_thread_with_threading_backend(tmp_path):
    config = tmp_path / "params.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "blas_threads": 12,
                    "numba_threads": 12,
                    "scanpy_n_jobs": 12,
                    "n_jobs": 12,
                    "joblib_backend": "threading",
                }
            }
        ),
        encoding="utf-8",
    )
    package_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "CELLSCOPE_CONFIG": str(config),
            "OPENBLAS_NUM_THREADS": "128",
            "OMP_NUM_THREADS": "128",
            "MKL_NUM_THREADS": "128",
            "NUMEXPR_MAX_THREADS": "128",
            "NUMEXPR_NUM_THREADS": "128",
            "NUMBA_NUM_THREADS": "128",
            "SCANPY_N_JOBS": "128",
            "PYTHONPATH": str(package_root),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; import cellscope.cli; "
                "assert 'cellscope.pipeline' not in sys.modules; "
                "import cellscope.pipeline; import numba; "
                "import scanpy as sc; "
                "numba.set_num_threads(sc.settings.n_jobs); "
                "print('|'.join(os.environ[name] for name in "
                "('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', "
                "'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', "
                "'NUMBA_NUM_THREADS')) + '|' + str(numba.get_num_threads()) "
                "+ '|' + str(sc.settings.n_jobs))"
            ),
        ],
        cwd=str(package_root),
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )

    assert result.stdout.strip().splitlines()[-1] == "1|1|1|1|5|5|5"


def test_run_pipeline_explicit_arguments_override_yaml(tmp_path, monkeypatch):
    paths = _write_inputs(tmp_path)
    config = tmp_path / "params.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "io": {
                    "write_intermediate": True,
                    "resume_policy": "minimal",
                    "provenance_hash_inputs": False,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CELLSCOPE_CONFIG", str(config))

    previous_switch = pipeline._WRITE_INTERMEDIATE
    result = pipeline.run_pipeline(
        spatial_path=str(paths["spatial"]),
        cell_boundaries_path=str(paths["cell"]),
        nucleus_boundaries_path=str(paths["nucleus"]),
        out_dir=str(tmp_path / "out"),
        save_intermediate=False,
        resume_policy="force",
        dry_run_diff=True,
    )

    assert result["start_from"] == "load_inputs"
    assert pipeline._WRITE_INTERMEDIATE is previous_switch


def test_run_pipeline_keeps_blas_single_thread_with_threading_backend(tmp_path, monkeypatch):
    paths = _write_inputs(tmp_path)
    config = tmp_path / "params.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "blas_threads": 12,
                    "n_jobs": 12,
                    "joblib_backend": "threading",
                },
                "io": {"provenance_hash_inputs": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CELLSCOPE_CONFIG", str(config))
    names = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        result = pipeline.run_pipeline(
            spatial_path=str(paths["spatial"]),
            cell_boundaries_path=str(paths["cell"]),
            nucleus_boundaries_path=str(paths["nucleus"]),
            out_dir=str(tmp_path / "out"),
            dry_run_diff=True,
        )
        assert result["dry_run"] is True
        assert {os.environ[name] for name in names} == {"1"}
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_cli_rejects_invalid_dgi_unit(tmp_path):
    paths = _write_inputs(tmp_path)
    config = tmp_path / "params.yaml"
    config.write_text(
        yaml.safe_dump({"module8": {"dgi_unit": "transcript"}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli.main,
        [
            "--spatial",
            str(paths["spatial"]),
            "--cell-boundaries",
            str(paths["cell"]),
            "--nucleus-boundaries",
            str(paths["nucleus"]),
            "--out-dir",
            str(tmp_path / "out"),
            "--config",
            str(config),
            "--dry-run-diff",
        ],
    )

    assert result.exit_code != 0
    assert "module8.dgi_unit must be" in result.output


def test_cli_rejects_malformed_yaml_instead_of_sanitizing_it(tmp_path):
    paths = _write_inputs(tmp_path)
    config = tmp_path / "broken.yaml"
    config.write_text("module8:\n\tdgi_unit: metaspot\n  broken", encoding="utf-8")

    result = CliRunner().invoke(
        cli.main,
        [
            "--spatial",
            str(paths["spatial"]),
            "--cell-boundaries",
            str(paths["cell"]),
            "--nucleus-boundaries",
            str(paths["nucleus"]),
            "--out-dir",
            str(tmp_path / "out"),
            "--config",
            str(config),
            "--dry-run-diff",
        ],
    )

    assert result.exit_code != 0
    assert "could not parse configuration" in result.output


def test_default_annotation_config_resolves_production_defaults():
    config = load_params_yaml(path=get_default_params_path())
    public = config["annotation"]

    assert config["module5"]["clustering_method"] == "consensus_leiden"
    assert config["module3"]["parallel_n_jobs"] == 12
    assert len(config["module5"]["consensus_seeds"]) >= 2
    assert config["module7"]["metaspot_expr_cross_cell_only"] is True
    assert config["module8"]["dgi_framework"] == "cell_aware_dmgi"
    assert config["module8"]["dgi_epochs"] == 200
    assert config["module8"]["dgi_device"] == "auto"
    assert config["module8"]["dgi_cpu_threads"] == 0
    assert config["module8"]["dgi_sparse_layout"] == "auto"
    assert config["module8"]["dmgi_fused_corruption_pass"] is True
    assert config["module8"]["dmgi_gate_floor"] == 0.1
    assert config["module8"]["dgi_local_loss_weight"] == 0.0

    assert public["omm_policy"] == "audit"
    assert public["metaspot_dgi_consensus_weight"] == 0.6
    assert public["metaspot_dgi_exploratory"] is True
    assert public["metaspot_cell_coverage_min_local_marker_transcripts"] == 2
    assert public["metaspot_cell_coverage_min_local_marker_genes"] == 2
    assert public["metaspot_cell_coverage_min_dgi_support"] == 0.0
    assert public["metaspot_cell_coverage_min_dgi_margin"] == -1.0
    assert set(public["metaspot_label_thresholds"]) == {
        "Nucleolus",
        "Lamina",
        "Nuclear Pore",
        "ER",
        "OMM",
    }
    assert public["metaspot_omm_unidentifiable_policy"] == "strict_local"
    assert public["metaspot_omm_cell_min_marker_transcripts"] == 6
    assert public["metaspot_omm_cell_min_marker_genes"] == 3
    assert public["metaspot_omm_cell_min_log2_enrichment"] == 0.25
    assert public["metaspot_omm_cell_max_binomial_p"] == 1.0
    assert public["metaspot_omm_cell_exploratory_max_binomial_p"] == 1.0
    assert public["metaspot_omm_local_min_marker_transcripts"] == 1
    assert public["metaspot_omm_local_min_marker_genes"] == 1
    assert public["metaspot_omm_local_min_dgi_support"] == 0.50
    assert public["metaspot_omm_cell_max_per_cell"] == 5
    assert public["metaspot_hierarchical_cell_rescue"] is True
    assert public["metaspot_hierarchical_min_cell_transcripts"] == 200
    assert public["metaspot_hierarchical_min_metaspots"] == 20
    assert public["metaspot_hierarchical_min_presence_score"] == 0.55
    assert public["metaspot_hierarchical_min_localization_score"] == 0.55
    assert public["metaspot_hierarchical_min_joint_score"] == 0.60
    assert public["metaspot_hierarchical_min_dgi_support"] == 0.35
    assert public["metaspot_hierarchical_min_dgi_margin"] == -0.25
    assert "omm_metaspot_posterior_high" not in public
    assert "single_stage_compartment_max_q" not in public

    resolved = resolve_annotation_configuration(config)
    assert resolved["independent_assignment_mode"] == "hierarchical"
    assert resolved["omm_metaspot_posterior_high"] == 0.80
    assert resolved["metaspot_dgi_consensus_weight"] == 0.6
    assert resolved["metaspot_omm_unidentifiable_policy"] == "strict_local"
    assert resolved["metaspot_omm_cell_min_marker_transcripts"] == 6
    assert resolved["metaspot_omm_cell_min_marker_genes"] == 3
    assert resolved["metaspot_omm_cell_min_log2_enrichment"] == 0.25
    assert resolved["metaspot_omm_cell_max_binomial_p"] == 1.0
    assert resolved["metaspot_omm_cell_exploratory_max_binomial_p"] == 1.0
    assert resolved["metaspot_omm_local_min_marker_transcripts"] == 1
    assert resolved["metaspot_omm_local_min_marker_genes"] == 1
    assert resolved["metaspot_omm_local_min_dgi_support"] == 0.50
    assert resolved["metaspot_omm_cell_max_per_cell"] == 5
    assert resolved["metaspot_hierarchical_cell_rescue"] is True
    assert resolved["metaspot_hierarchical_min_cell_transcripts"] == 200
    assert resolved["metaspot_hierarchical_min_metaspots"] == 20
    assert resolved["metaspot_hierarchical_min_joint_score"] == 0.60
    assert resolved["metaspot_hierarchical_min_dgi_margin"] == -0.25
    assert resolved["omm_metaspot_refinement"] is True
    assert resolved["omm_metaspot_apply_to_primary"] is False


@pytest.mark.parametrize(
    ("legacy_annotation", "expected_policy"),
    [
        ({"omm_metaspot_refinement": False}, "off"),
        ({"omm_metaspot_apply_to_primary": False}, "audit"),
        ({"omm_metaspot_refinement": True}, "strict"),
    ],
)
def test_legacy_omm_controls_resolve_without_semantic_drift(
    legacy_annotation,
    expected_policy,
):
    resolved = resolve_annotation_configuration(
        {"annotation": legacy_annotation}
    )
    assert resolved["omm_policy"] == expected_policy


def test_cli_merge_does_not_impose_balanced_on_legacy_user_config():
    defaults = {"annotation": {"omm_policy": "balanced"}}
    legacy_user = {
        "annotation": {"omm_metaspot_refinement": False}
    }
    merged = cli._preserve_legacy_omm_policy(
        cli._merge_with_defaults(defaults, legacy_user),
        legacy_user,
    )

    assert "omm_policy" not in merged["annotation"]
    assert (
        resolve_annotation_configuration(merged)["omm_policy"] == "off"
    )

    explicit_user = {"annotation": {"omm_policy": "strict"}}
    explicit = cli._preserve_legacy_omm_policy(
        cli._merge_with_defaults(defaults, explicit_user),
        explicit_user,
    )
    assert explicit["annotation"]["omm_policy"] == "strict"


def test_packaged_fast_overlay_matches_documented_iteration_profile():
    default_path = Path(get_default_params_path())
    fast_path = default_path.with_name("params.fast.yaml")
    overlay = load_params_yaml(
        path=str(fast_path),
        resolve_config=False,
    )
    effective = cli._merge_with_defaults(
        load_params_yaml(path=str(default_path), resolve_config=False),
        overlay,
    )
    effective = cli.resolve_dgi_configuration(effective)

    assert effective["module7"]["knn_backend"] == "pynndescent"
    assert effective["module8"]["dgi_epochs"] == 120
    assert effective["module5"]["consensus_seeds"] == [17, 42]
    assert effective["annotation"]["compute_umap"] is False
    assert effective["annotation"]["save_plots"] is False
    assert effective["io"]["write_intermediate_figures"] is False


def test_omm_policy_changes_only_annotation_fingerprint():
    strict = {"annotation": {"omm_policy": "strict"}}
    balanced = {"annotation": {"omm_policy": "balanced"}}

    assert fingerprint_stage(strict, "annotate") != fingerprint_stage(
        balanced,
        "annotate",
    )
    assert fingerprint_stage(strict, "build_anndata") == fingerprint_stage(
        balanced,
        "build_anndata",
    )


def test_omm_sensitivity_defaults_change_only_annotation_fingerprint():
    current = load_params_yaml(path=get_default_params_path())
    previous = copy.deepcopy(current)
    previous_annotation = previous["annotation"]
    previous_annotation["metaspot_omm_cell_min_marker_transcripts"] = 8
    previous_annotation["metaspot_omm_cell_min_marker_genes"] = 4
    previous_annotation["metaspot_omm_cell_min_log2_enrichment"] = 0.50
    previous_annotation["metaspot_omm_cell_max_binomial_p"] = 0.05
    previous_annotation["metaspot_omm_local_min_marker_transcripts"] = 2
    previous_annotation["metaspot_omm_local_min_marker_genes"] = 2
    previous_annotation["metaspot_omm_local_min_dgi_support"] = 0.55
    previous_annotation["metaspot_omm_cell_max_per_cell"] = 5

    assert fingerprint_stage(current, "annotate") != fingerprint_stage(
        previous,
        "annotate",
    )
    assert fingerprint_stage(current, "build_anndata") == fingerprint_stage(
        previous,
        "build_anndata",
    )


def test_annotation_failure_is_logged_and_stops_output(tmp_path, monkeypatch):
    paths = _write_inputs(tmp_path)
    config = tmp_path / "params.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "io": {
                    "write_intermediate": False,
                    "provenance_hash_inputs": False,
                },
                "annotation": {
                    "enable": True,
                    "method": "rapx",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CELLSCOPE_CONFIG", str(config))

    raw = pd.DataFrame(
        {
            "cell_id": ["c1"],
            "x_location": [0.0],
            "y_location": [0.0],
        }
    )
    adata1 = ad.AnnData(
        X=sparse.csr_matrix(np.ones((1, 1), dtype=np.float32)),
        obs=pd.DataFrame(
            {"cell_id": ["c1"], "meta-domain": ["d1"]},
            index=["m1"],
        ),
        var=pd.DataFrame(index=["G1"]),
    )
    adata2 = ad.AnnData(
        X=sparse.csr_matrix(np.ones((1, 1), dtype=np.float32)),
        obs=pd.DataFrame(index=["d1"]),
        var=pd.DataFrame(index=["G1"]),
    )
    final_data = pd.DataFrame(
        {
            "cell_id": ["c1"],
            "metaspot": ["m1"],
            "meta-domain": ["d1"],
        }
    )

    monkeypatch.setattr(
        pipeline,
        "load_inputs",
        lambda *args, **kwargs: (raw.copy(), raw.copy(), raw.copy()),
    )
    monkeypatch.setattr(
        pipeline,
        "preprocess",
        lambda spatial, cell, nucleus, **kwargs: (
            spatial,
            cell,
            nucleus,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_anndata",
        lambda *args, **kwargs: (
            adata1.copy(),
            adata2.copy(),
            final_data.copy(),
        ),
    )

    def fail_annotation(*args, **kwargs):
        raise ValueError("synthetic annotation failure")

    monkeypatch.setattr(
        pipeline,
        "_maybe_cluster_and_annotate",
        fail_annotation,
    )
    out_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="Annotation stage failed"):
        pipeline.run_pipeline(
            spatial_path=str(paths["spatial"]),
            cell_boundaries_path=str(paths["cell"]),
            nucleus_boundaries_path=str(paths["nucleus"]),
            out_dir=str(out_dir),
            resume=False,
            save_intermediate=False,
        )

    assert not (out_dir / "final_data.csv").exists()
    log_text = (out_dir / "cellscoper.log").read_text(encoding="utf-8")
    assert "Effective runtime" in log_text
    assert "refusing to write a successful production output" in log_text


def test_stage_fingerprint_changes_with_input_provenance():
    config = {"module3": {"target_sites_per_metaspot": 15}}
    first = fingerprint_stage(
        config,
        "build_anndata",
        context={"inputs": {"spatial": {"sha256": "a"}}},
    )
    second = fingerprint_stage(
        config,
        "build_anndata",
        context={"inputs": {"spatial": {"sha256": "b"}}},
    )
    assert first != second


def test_output_settings_do_not_invalidate_input_loading():
    config = {
        "io": {
            "final_format": "csv",
            "h5ad_compression": "lzf",
        }
    }
    changed = copy.deepcopy(config)
    changed["io"]["final_format"] = "parquet"
    changed["io"]["h5ad_compression"] = "gzip"

    assert fingerprint_stage(config, "load_inputs") == fingerprint_stage(
        changed,
        "load_inputs",
    )
    assert fingerprint_stage(config, "write_outputs") != fingerprint_stage(
        changed,
        "write_outputs",
    )


def test_final_column_pruning_invalidates_build_stage():
    config = {"io": {"drop_embeddings": False}}
    changed = {"io": {"drop_embeddings": True}}

    assert fingerprint_stage(config, "build_anndata") != fingerprint_stage(
        changed,
        "build_anndata",
    )
    assert fingerprint_stage(config, "write_outputs") != fingerprint_stage(
        changed,
        "write_outputs",
    )


def test_compact_final_h5ad_keeps_canonical_dgi_and_restores_source():
    adata = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    adata.obsm[pipeline.METASPOT_DGI_KEY] = np.arange(
        6,
        dtype=np.float32,
    ).reshape(3, 2)
    adata.obsm["X_expr_metaspot"] = np.ones((3, 2), dtype=np.float32)
    adata.obsp["A_metaspot_norm"] = sparse.eye(
        3,
        format="csr",
        dtype=np.float32,
    )

    with pipeline._compact_final_adata1_for_write(adata, enabled=True):
        assert list(adata.obsm) == [pipeline.METASPOT_DGI_KEY]
        assert list(adata.obsp) == []
        metadata = adata.uns["cellscope_final_output_compaction"]
        assert metadata["kept_obsm"] == [pipeline.METASPOT_DGI_KEY]
        assert metadata["omitted_obsp"] == ["A_metaspot_norm"]

    assert set(adata.obsm) == {
        pipeline.METASPOT_DGI_KEY,
        "X_expr_metaspot",
    }
    assert list(adata.obsp) == ["A_metaspot_norm"]
    assert "cellscope_final_output_compaction" not in adata.uns


def test_annotation_schema_version_is_internal_to_fingerprint():
    subset = stage_config_subset({"annotation": {}}, "annotate")

    assert (
        subset["__annotation_schema_version__"]
        == ANNOTATION_SCHEMA_VERSION
    )


def test_metadomain_random_seed_invalidates_build_and_module5():
    current = load_params_yaml(path=get_default_params_path())
    changed = copy.deepcopy(current)
    changed["module5"]["random_seed"] += 1

    assert fingerprint_stage(current, "build_anndata") != fingerprint_stage(
        changed,
        "build_anndata",
    )
    assert fingerprint_module(current, "m5") != fingerprint_module(
        changed,
        "m5",
    )


def test_multistructure_first_locus_settings_invalidate_annotation():
    current = load_params_yaml(path=get_default_params_path())
    variants = []
    for key in (
        "metaspot_multistructure_first_locus",
        "metaspot_multistructure_first_locus_min_presence",
        "metaspot_multistructure_first_locus_min_localization",
        "metaspot_multistructure_first_locus_max_projected_burden",
    ):
        changed = copy.deepcopy(current)
        value = changed["annotation"][key]
        changed["annotation"][key] = (
            not value if isinstance(value, bool) else float(value) / 2.0
        )
        variants.append(changed)
    for key in (
        "metaspot_multistructure_first_locus_min_evidence",
        "metaspot_fine_min_spatial_coherence",
    ):
        changed = copy.deepcopy(current)
        changed["annotation"][key]["ER"] /= 2.0
        variants.append(changed)

    for changed in variants:
        assert fingerprint_stage(
            current,
            "annotate",
        ) != fingerprint_stage(changed, "annotate")


def test_annotation_execution_switches_honor_explicit_arguments(
    tmp_path,
    monkeypatch,
):
    config = {
        "annotation": {
            "unit": "metadomain",
            "fast": True,
            "save_plots": False,
            "save_umap_plot": False,
            "two_stage": False,
            "spatial_channel": False,
            "omm_metaspot_refinement": False,
            "metaspot_compartment_correction": False,
        },
        "module7": {"rep_priority": []},
    }
    monkeypatch.setattr(pipeline, "load_params_yaml", lambda: config)
    monkeypatch.setattr(pipeline, "_save_dataframe", lambda *args, **kwargs: None)
    captured = {}

    def fake_run_rapx(adata, **kwargs):
        captured.update(kwargs)
        adata.obs["subcellular_annotation"] = pd.Categorical(["Nucleus"])
        adata.obs["subcellular_candidate"] = pd.Categorical(["Nucleus"])
        return pd.DataFrame(
            {"label": ["Nucleus"]},
            index=pd.Index(["d1"], name="leiden"),
        )

    monkeypatch.setattr(pipeline, "run_rapx", fake_run_rapx)
    adata1 = ad.AnnData(
        X=sparse.csr_matrix(np.ones((1, 1), dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "cell_id": ["c1"],
                "metaspot": ["m1"],
                "meta-domain": ["d1"],
                "r_mean": [-0.5],
            },
            index=["m1"],
        ),
        var=pd.DataFrame(index=["G1"]),
    )
    adata2 = ad.AnnData(
        X=sparse.csr_matrix(np.ones((1, 1), dtype=np.float32)),
        obs=pd.DataFrame({"leiden": ["0"]}, index=["d1"]),
        var=pd.DataFrame(index=["G1"]),
    )
    final_data = pd.DataFrame(
        {
            "cell_id": ["c1"],
            "metaspot": ["m1"],
            "meta-domain": ["d1"],
        }
    )
    gold = pd.DataFrame({"Common_Gene": ["G1"], "Nucleus": [1.0]})

    pipeline._maybe_cluster_and_annotate(
        adata1,
        adata2,
        final_data,
        out_dir=str(tmp_path),
        rapx_two_stage=True,
        rapx_spatial_channel=True,
        gold_data=gold,
    )

    assert captured["use_two_stage"] is True
    assert captured["use_spatial_channel"] is True


def test_parquet_failure_returns_actual_csv_fallback_path(
    tmp_path,
    monkeypatch,
):
    paths = _write_inputs(tmp_path)
    config = tmp_path / "params.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "io": {
                    "write_intermediate": False,
                    "provenance_hash_inputs": False,
                },
                "annotation": {"enable": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CELLSCOPE_CONFIG", str(config))

    raw = pd.DataFrame(
        {
            "cell_id": ["c1"],
            "x_location": [0.0],
            "y_location": [0.0],
        }
    )
    adata1 = ad.AnnData(
        X=sparse.csr_matrix(np.ones((1, 1), dtype=np.float32)),
        obs=pd.DataFrame({"cell_id": ["c1"]}, index=["m1"]),
        var=pd.DataFrame(index=["G1"]),
    )
    final_data = pd.DataFrame(
        {"cell_id": ["c1"], "metaspot": ["m1"]}
    )
    monkeypatch.setattr(
        pipeline,
        "load_inputs",
        lambda *args, **kwargs: (raw.copy(), raw.copy(), raw.copy()),
    )
    monkeypatch.setattr(
        pipeline,
        "preprocess",
        lambda spatial, cell, nucleus, **kwargs: (
            spatial,
            cell,
            nucleus,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_anndata",
        lambda *args, **kwargs: (
            adata1.copy(),
            None,
            final_data.copy(),
        ),
    )

    def fail_parquet(*args, **kwargs):
        raise RuntimeError("synthetic parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_parquet)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale_parquet = out_dir / "final_data.parquet"
    stale_parquet.write_bytes(b"stale parquet from an older run")
    outputs = pipeline.run_pipeline(
        spatial_path=str(paths["spatial"]),
        cell_boundaries_path=str(paths["cell"]),
        nucleus_boundaries_path=str(paths["nucleus"]),
        out_dir=str(out_dir),
        resume=False,
        save_intermediate=False,
        enable_annotation=False,
        final_format="parquet",
    )

    assert outputs["final_parquet"] is None
    assert outputs["final_csv"] == str(out_dir / "final_data.csv")
    assert (out_dir / "final_data.csv").is_file()
    assert not stale_parquet.exists()
    quarantined = Path(outputs["stale_final_parquet"])
    assert quarantined.is_file()
    assert quarantined.read_bytes() == b"stale parquet from an older run"
    guide = (out_dir / "OUTPUT_GUIDE.md").read_text(encoding="utf-8")
    assert "`final_data.csv`: transcript-level production result" in guide
    assert outputs["counts"]["n_transcripts"] == 1
    assert "falling back to CSV" in (
        out_dir / "cellscoper.log"
    ).read_text(encoding="utf-8")


def test_compact_diagnostics_write_when_large_intermediates_are_disabled(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "params.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "io": {
                    "intermediate_dirname": "intermediate",
                    "write_intermediate": False,
                    "write_intermediate_csv": False,
                    "write_intermediate_parquet": False,
                    "write_diagnostic_tables": True,
                    "write_diagnostic_csv": True,
                    "write_diagnostic_parquet": False,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CELLSCOPE_CONFIG", str(config))
    frame = pd.DataFrame({"metric": ["coverage"], "value": [1.0]})

    pipeline._save_dataframe(
        frame,
        str(tmp_path / "out"),
        "compact_audit",
        diagnostic=True,
    )
    pipeline._save_dataframe(
        frame,
        str(tmp_path / "out"),
        "large_intermediate",
    )

    intermediate = tmp_path / "out" / "intermediate"
    assert (intermediate / "compact_audit.csv").is_file()
    assert not (intermediate / "large_intermediate.csv").exists()


def test_resume_checkpoint_can_force_parquet_in_lite_mode(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "params.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "io": {
                    "intermediate_dirname": "intermediate",
                    "write_intermediate": True,
                    "write_intermediate_csv": False,
                    "write_intermediate_parquet": False,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CELLSCOPE_CONFIG", str(config))
    frame = pd.DataFrame({"metaspot": ["c1::0"], "meta-domain": ["0"]})

    pipeline._save_dataframe(
        frame,
        str(tmp_path / "out"),
        "module5_final_data",
        force_parquet=True,
    )

    intermediate = tmp_path / "out" / "intermediate"
    assert (intermediate / "module5_final_data.parquet").is_file()
    assert not (intermediate / "module5_final_data.csv").exists()


def test_lite_resume_recovers_strictly_aligned_late_point_table(
    tmp_path,
):
    out_dir = tmp_path / "out"
    intermediate = out_dir / "intermediate_results"
    intermediate.mkdir(parents=True)
    point_df = pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c2"],
            "feature_name": ["G1", "G2", "G1"],
            "x_location": [0.0, 1.0, 2.0],
            "y_location": [0.0, 1.0, 2.0],
            "r_signed": [-0.5, 0.5, 0.25],
            "metaspot_in_cell": [0, 0, 0],
            "metaspot": ["c1:0", "c1:0", "c2:0"],
            "meta-domain": ["0", "0", "1"],
        }
    )
    point_df.to_parquet(
        intermediate / "module5_final_data.parquet",
        index=False,
    )
    module4 = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[2.0], [1.0]], dtype=np.float32)),
        obs=pd.DataFrame(
            {"r_mean": [0.0, 0.25]},
            index=pd.Index(["c1:0", "c2:0"]),
        ),
    )
    module4.write_h5ad(intermediate / "module4_adata1.h5ad")

    recovered = pipeline._load_late_point_resume_checkpoint(
        str(out_dir),
        point_df.drop(columns=["metaspot", "meta-domain"]),
    )

    assert recovered is not None
    pd.testing.assert_frame_equal(recovered, point_df)
    assert pipeline._load_late_point_resume_checkpoint(
        str(out_dir),
        point_df.iloc[:-1].drop(columns=["metaspot", "meta-domain"]),
    ) is None


def test_metadomain_umap_uses_final_subcellular_annotation_colors(
    tmp_path,
    monkeypatch,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend

    adata2 = ad.AnnData(
        X=np.zeros((4, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "domain_type": ["9", "2", "9", "1"],
                "leiden": ["9", "2", "9", "1"],
                "subcellular_annotation": [
                    "Cytosol",
                    "Nucleus",
                    "Unknown",
                    "Nucleolus",
                ],
            },
            index=["d0", "d1", "d2", "d3"],
        ),
    )
    adata2.obsm["X_umap"] = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    observed = {}

    def _fake_save_fig(fig, out_dir, name, **kwargs):
        observed["saved"] = (fig, out_dir, name, kwargs)

    monkeypatch.setattr(pipeline, "_save_fig", _fake_save_fig)

    assert pipeline._plot_metadomain_subcellular_annotation(
        adata2,
        str(tmp_path),
    )

    figure, out_dir, name, save_kwargs = observed["saved"]
    assert tuple(figure.get_size_inches()) == pytest.approx((7.4, 5.0))
    ax = figure.axes[0]
    assert ax.get_title() == "MetaDomain Subcellular Annotation"
    assert ax.title.get_fontweight() == "bold"
    legends = [
        child for child in ax.get_children() if isinstance(child, Legend)
    ]
    assert {legend.get_title().get_text() for legend in legends} == {
        "Nuclear side",
        "Cytoplasmic side",
        "Other",
    }
    plotted_colors = {
        collection.get_label(): mpl.colors.to_hex(
            collection.get_facecolor()[0],
            keep_alpha=False,
        ).upper()
        for collection in ax.collections
    }
    assert plotted_colors == {
        label: pipeline._SUBCELLULAR_LABEL_PALETTE[label]
        for label in (
            "Nucleus",
            "Nucleolus",
            "Cytosol",
            "Unknown",
        )
    }
    assert all(collection.get_alpha() == 0.8 for collection in ax.collections)
    assert all(collection.get_rasterized() for collection in ax.collections)
    assert (out_dir, name, save_kwargs) == (
        str(tmp_path),
        "metadomain_leiden_clusters",
        {"dpi": 800},
    )
    assert adata2.obs["domain_type"].tolist() == ["9", "2", "9", "1"]


def test_subcellular_publication_palette_has_one_canonical_side_assignment():
    nuclear = set(SUBCELLULAR_LABEL_GROUPS["Nuclear side"])
    cytoplasmic = set(SUBCELLULAR_LABEL_GROUPS["Cytoplasmic side"])

    assert nuclear.isdisjoint(cytoplasmic)
    assert nuclear | cytoplasmic == set(SUBCELLULAR_LABEL_ORDER) - {"Unknown"}
    assert set(SUBCELLULAR_LABEL_PALETTE) == set(SUBCELLULAR_LABEL_ORDER)
    assert SUBCELLULAR_LABEL_PALETTE == {
        "Nucleus": "#2F5D8C",
        "Nucleolus": "#6B5CA5",
        "Lamina": "#4F83B6",
        "Nuclear Pore": "#4FA7B8",
        "Cytosol": "#D6A23C",
        "ER": "#E2743E",
        "OMM": "#B84A62",
        "Unknown": "#9AA0A6",
    }
