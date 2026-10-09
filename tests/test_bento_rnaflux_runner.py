"""Regression tests for the standalone SeqFISH Bento RNAflux runner."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef


if os.environ.get("CELLSCOPER_RUN_EXTERNAL_BENCHMARK_TESTS") != "1":
    pytest.skip(
        "requires the outer-workspace Bento benchmark runner; set "
        "CELLSCOPER_RUN_EXTERNAL_BENCHMARK_TESTS=1 to enable",
        allow_module_level=True,
    )


RUNNER_PATH = Path(__file__).resolve().parents[2] / "run_bento_rnaflux_single.py"
SPEC = importlib.util.spec_from_file_location("cellscope_bento_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_binary_count_metrics_match_sklearn() -> None:
    truth = np.asarray([0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    prediction = np.asarray([0, 1, -1, 1, 0, 1, 1], dtype=np.int8)
    available = prediction >= 0

    observed = RUNNER.evaluate_binary_codes(
        prediction,
        truth,
        method="test",
    )

    assert observed["coverage"] == available.mean()
    assert observed["balanced_accuracy"] == balanced_accuracy_score(
        truth[available], prediction[available]
    )
    assert observed["macro_f1"] == f1_score(
        truth[available], prediction[available], average="macro"
    )
    assert observed["MCC"] == matthews_corrcoef(
        truth[available], prediction[available]
    )


def test_cell_grouped_oof_calibration_has_no_cell_leakage() -> None:
    records = []
    truth = []
    for fov_index in range(5):
        cell_id = f"FOV_{fov_index}_{fov_index}"
        for label in (0, 1) * 4:
            records.append(
                {
                    "benchmark_row_id": len(records),
                    "cell_id": cell_id,
                    "metaspot": f"metaspot_{fov_index}",
                    "bento_nuclear_contrast": -1.0 if label == 0 else 1.0,
                    "prediction_available": True,
                }
            )
            truth.append(label)

    predictions = pd.DataFrame.from_records(records)
    output, metrics, provenance = RUNNER.build_oof_boundary_calibration(
        predictions,
        pd.Series(truth, dtype=np.int8),
    )

    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["nuclear_f1"] == 1.0
    assert output["benchmark_row_id"].equals(predictions["benchmark_row_id"])
    assert output.groupby("cell_id")[
        "bento_boundary_calibration_fold"
    ].nunique().eq(1).all()
    for fold in provenance["folds"]:
        assert fold["held_out_fov"] not in set(fold["train_fovs"])


def test_seqfish_fov_batch_parser_rejects_unscoped_cell_ids() -> None:
    observed = RUNNER.derive_fov_batches(
        pd.Series(["FOV_1_1", "FOV_12_cell_9"])
    )
    assert observed.tolist() == ["FOV_1", "FOV_12"]

    with np.testing.assert_raises(ValueError):
        RUNNER.derive_fov_batches(pd.Series(["cell_1"]))


def test_protocol_identity_keeps_runner_hash_and_schema_scientific() -> None:
    old = {
        "schema_version": 3,
        "runner_path": "/old/runner.py",
        "runner_sha256": "old",
        "protocol_sha256": "old-protocol",
        "source_sha256": "source",
        "flux_res": 0.1,
        "seed": 42,
    }
    path_only_update = {
        **old,
        "runner_path": "/new/runner.py",
        "protocol_sha256": "new-protocol",
    }
    changed_runner = {**path_only_update, "runner_sha256": "new"}
    changed_schema = {**path_only_update, "schema_version": 4}

    assert RUNNER.protocol_scientific_identity(
        old
    ) == RUNNER.protocol_scientific_identity(path_only_update)
    assert RUNNER.protocol_scientific_identity(
        old
    ) != RUNNER.protocol_scientific_identity(changed_runner)
    assert RUNNER.protocol_scientific_identity(
        old
    ) != RUNNER.protocol_scientific_identity(changed_schema)


def test_paper_feature_universe_uses_per_cell_molecule_threshold() -> None:
    inputs = pd.DataFrame(
        {
            "cell_id": ["FOV_0_1"] * 10 + ["FOV_0_2"] * 9 + ["FOV_0_2"],
            "feature_name": ["A"] * 10 + ["B"] * 9 + ["A"],
        }
    )
    assert RUNNER.paper_feature_universe(
        inputs,
        min_molecules_in_any_cell=10,
    ) == ["A"]


def test_advisory_lock_rejects_second_writer(tmp_path: Path) -> None:
    lock_path = tmp_path / "coordinator.lock"
    descriptor = RUNNER.acquire_advisory_lock(
        lock_path,
        blocking=False,
        role="test",
    )
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        with np.testing.assert_raises(RuntimeError):
            RUNNER.acquire_advisory_lock(
                lock_path,
                blocking=False,
                role="second test",
            )
    finally:
        os.close(descriptor)


def test_parallel_coordinator_runs_each_batch_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_command(extra: list[str]) -> list[str]:
        return list(extra)

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        if "--prepare-only" in command:
            RUNNER.write_json(
                {"n_execution_batches": 3},
                tmp_path / "batch_protocol.json",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(RUNNER, "_parallel_child_command", fake_command)
    monkeypatch.setattr(RUNNER.subprocess, "run", fake_run)

    observed = RUNNER.run_parallel_coordinator(
        SimpleNamespace(out_dir=tmp_path, parallel_workers=2)
    )

    assert observed == 0
    worker_indices = sorted(
        int(command[command.index("--batch-index") + 1])
        for command in calls
        if "--worker-only" in command
    )
    assert worker_indices == [0, 1, 2]
    assert sum("--prepare-only" in command for command in calls) == 1
    assert sum("--finalize-only" in command for command in calls) == 1
